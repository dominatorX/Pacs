import sys
sys.path.insert(0, '../')
import torch
from now.hko.dataloader import HKOIterator
from now.config import cfg
import numpy as np
from now.hko.evaluation import HKOEvaluation
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os.path as osp
import os
import shutil
import cv2
import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, encoder_forecaster, optimizer, criterion, lr_scheduler, batch_size,
                 max_iterations, test_iteration_interval, test_and_save_checkpoint_iterations,
                 folder_name, probToPixel=None, rid=0):
        self.IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
        self.OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN
        self.height = cfg.HKO.ITERATOR.HEIGHT
        self.width = cfg.HKO.ITERATOR.WIDTH
        self.channel = cfg.HKO.ITERATOR.CHANNEL
        self.base_freq = '60min'
        self.folder_name = folder_name
        self.encoder_forecaster = encoder_forecaster
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.test_iteration_interval = test_iteration_interval
        self.test_and_save_checkpoint_iterations = test_and_save_checkpoint_iterations
        self.probToPixel = probToPixel
        self.device = cfg.GLOBAL.DEVICE
        self.rid = rid

        self.evaluater = HKOEvaluation(seq_len=self.OUT_LEN, use_central=False)
        self.train_hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TRAIN,
                                          sample_mode="random",
                                          batch_size=batch_size,
                                          seq_len=self.IN_LEN + self.OUT_LEN,
                                          max_consecutive_missing=0, base_freq=self.base_freq)

        self.test_hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TEST,
                                         sample_mode="sequent",
                                         batch_size=batch_size,
                                         seq_len=self.IN_LEN + self.OUT_LEN,
                                         stride=cfg.HKO.BENCHMARK.STRIDE,
                                         max_consecutive_missing=0, base_freq=self.base_freq)

        self.train_loss, self.test_loss = 0.0, 0.0

    def get_data(self, data_batch, mask):
        label = torch.from_numpy(data_batch[self.IN_LEN:self.IN_LEN + self.OUT_LEN, :, -1:, ...]).to(self.device)
        data = torch.from_numpy(data_batch[:self.IN_LEN, ...]).to(self.device)

        mask = torch.from_numpy(mask[self.IN_LEN:self.IN_LEN + self.OUT_LEN, ...].astype(int)).to(self.device)
        return data, label, mask

    def update_evaluator(self, output, label, mask):
        label_numpy = label.cpu().numpy()
        if self.probToPixel is None:
            output_numpy = np.clip(output.detach().cpu().numpy(), 0.0, 1.0)
        else:
            output_numpy = self.probToPixel(output.detach().cpu().numpy(), label, mask,
                                            self.lr_scheduler.get_lr()[0])
        self.evaluater.update(label_numpy, output_numpy, mask.cpu().numpy())

    def train_a_batch(self):
        train_batch, train_mask, sample_datetimes, _ = \
            self.train_hko_iter.sample(batch_size=self.batch_size)
        train_data, train_label, mask = self.get_data(train_batch, train_mask)
        self.encoder_forecaster.train()
        self.optimizer.zero_grad()

        output = self.encoder_forecaster(train_data)
        # train_label = torch.cat([train_data[1:, :, :, :, :], train_label])
        loss = self.criterion(output, train_label, mask)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.encoder_forecaster.parameters(), clip_value=50.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.train_loss += loss.item()

        self.update_evaluator(output, train_label, mask)

    def test_a_batch(self):
        test_batch, test_mask, sample_datetimes, _ = \
            self.test_hko_iter.sample(batch_size=self.batch_size)

        if len(sample_datetimes) < self.batch_size:
            return True
        self.test_time += 1

        test_data, test_label, mask = self.get_data(test_batch, test_mask)
        output = self.encoder_forecaster(test_data)

        loss = self.criterion(output, test_label, mask)
        self.test_loss += loss.item()
        self.update_evaluator(output, test_label, mask)
        return False

    def set_up_save_dir(self):
        save_dir = osp.join(cfg.GLOBAL.MODEL_SAVE_DIR, self.folder_name)
        self.model_save_dir = osp.join(save_dir, 'models'+str(self.rid))
        log_dir = osp.join(save_dir, 'logs'+str(self.rid))
        self.all_scalars_file_name = osp.join(save_dir, "all_scalars"+str(self.rid)+".json")
        if osp.exists(self.all_scalars_file_name):
            os.remove(self.all_scalars_file_name)
        if osp.exists(log_dir):
            shutil.rmtree(log_dir)
        if osp.exists(self.model_save_dir):
            shutil.rmtree(self.model_save_dir)
        os.mkdir(self.model_save_dir)

        self.writer = SummaryWriter(log_dir)

    def test_small_batch(self):
        for itera in tqdm(range(1, 5120//self.batch_size+1)):
            self.train_a_batch()
        _, _, train_csi, train_hss, _, train_mse, train_mae, train_balanced_mse, train_balanced_mae, _ = \
            self.evaluater.calculate_stat()
        print("round ", itera, "csi", train_csi, " bmse ", train_balanced_mse, " bmae ", train_balanced_mae)

        self.evaluater.clear_all()

        with torch.no_grad():
            self.test_loss = 0.0
            self.test_time = 0
            self.encoder_forecaster.eval()
            while not self.test_a_batch(): pass

            _, _, test_csi, test_hss, _, test_mse, test_mae, test_balanced_mse, test_balanced_mae, _ = \
                self.evaluater.calculate_stat()
            print(self.test_loss / self.test_time)
            print("test: csi", test_csi, " bmse ", test_balanced_mse, " bmae ", test_balanced_mae)

    def train(self):
        self.set_up_save_dir()
        for itera in tqdm(range(load_id, self.max_iterations+1)):
            # lr_scheduler.step()
            self.train_a_batch()
            if itera % self.test_iteration_interval == 0:

                _, _, train_csi, train_hss, _, train_mse, train_mae, train_balanced_mse, train_balanced_mae, _ = \
                    self.evaluater.calculate_stat()
                self.train_loss = self.train_loss/self.test_iteration_interval
                self.evaluater.clear_all()

                self.writer.add_scalars("loss", {
                        "train": self.train_loss,
                }, itera)

                plot_result(self.writer, itera, (train_csi, train_hss, train_mse, train_mae,
                                                     train_balanced_mse, train_balanced_mae))
                self.writer.export_scalars_to_json(self.all_scalars_file_name)

                self.train_loss = 0.0

            if itera % self.test_and_save_checkpoint_iterations == 0:
                torch.save(self.encoder_forecaster.state_dict(),
                           osp.join(self.model_save_dir, 'encoder_forecaster_{}.pth'.format(itera)))

        self.evaluater.clear_all()
        with torch.no_grad():
            self.encoder_forecaster.eval()
            torch.save(self.encoder_forecaster.state_dict(),
                           osp.join(self.model_save_dir, 'encoder_forecaster_best.pth'))
            self.test_hko_iter.reset()
            self.test_loss = 0.0
            self.test_time = 0

            while not self.test_hko_iter.use_up:
                if self.test_a_batch():
                    break

            _, _, test_csi, test_hss, _, test_mse, test_mae, test_balanced_mse, test_balanced_mae, _ = \
                self.evaluater.calculate_stat()
            self.evaluater.clear_all()
            self.test_loss = self.test_loss / self.test_time
            logger.info("test: csi " + str(test_csi) + " hss " + str(test_hss) +
                        " mse " + str(test_mse) + " mae " + str(test_mae) +
                        " bmse " + str(test_balanced_mse) + " bmae " + str(test_balanced_mae))
            logger.info('loss: ' + str(self.test_loss))
            print("test:\ncsi:", test_csi, "\nhss:", test_hss)
            print("bmse:", test_balanced_mse, "\nbmae:", test_balanced_mae, "\nloss:", self.test_loss)

        self.writer.close()

    def test(self, iter='best'):
        save_dir = osp.join(cfg.GLOBAL.MODEL_SAVE_DIR, self.folder_name)
        self.model_save_dir = osp.join(save_dir, 'models'+str(self.rid))
        stats = torch.load(self.model_save_dir+'/encoder_forecaster_{}.pth'.format(iter))

        with torch.no_grad():
            self.encoder_forecaster.load_state_dict(stats)
            self.encoder_forecaster.eval()
            self.test_hko_iter.reset()
            self.test_loss = 0.0
            self.test_time = 0

            while not self.test_hko_iter.use_up:
                if self.test_a_batch():
                    break

            _, _, test_csi, test_hss, _, test_mse, test_mae, test_balanced_mse, test_balanced_mae, _ = \
                self.evaluater.calculate_stat()
            self.evaluater.clear_all()
            self.test_loss = self.test_loss / self.test_time
            print("test:\ncsi:", test_csi, "\nhss:", test_hss)
            print("bmse:", test_balanced_mse, "\nbmae:", test_balanced_mae, "\nloss:", self.test_loss)
        return float(self.test_loss)

def plot_result(writer, itera, train_result):
    train_csi, train_hss, train_mse, train_mae, train_balanced_mse, train_balanced_mae = \
        train_result
    print("train:")
    print("csi:", train_csi)
    print("hss:", train_hss)
    print("mse:", train_mse)
    print("mae:", train_mae)
    print("bmse:", train_balanced_mse)
    print("bmae:", train_balanced_mae)
    train_csi, train_hss, train_mse, train_mae, train_balanced_mse, train_balanced_mae = \
        np.nan_to_num(train_csi), \
        np.nan_to_num(train_hss), \
        np.nan_to_num(train_mse), \
        np.nan_to_num(train_mae), \
        np.nan_to_num(train_balanced_mse), \
        np.nan_to_num(train_balanced_mae)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):

        writer.add_scalars("csi/{}".format(thresh), {
            "train": train_csi[:, i].mean(),
            "train_last_frame": train_csi[-1, i],
        }, itera)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):

        writer.add_scalars("hss/{}".format(thresh), {
            "train": train_hss[:, i].mean(),
            "train_last_frame": train_hss[-1, i],
        }, itera)

    writer.add_scalars("mse", {
        "train": train_mse.mean(),
        "train_last_frame": train_mse[-1],
    }, itera)

    writer.add_scalars("mae", {
        "train": train_mae.mean(),
        "train_last_frame": train_mae[-1],
    }, itera)

    writer.add_scalars("balanced_mse", {
        "train": train_balanced_mse.mean(),
        "train_last_frame": train_balanced_mse[-1],
    }, itera)

    writer.add_scalars("balanced_mae", {
        "train": train_balanced_mae.mean(),
        "train_last_frame": train_balanced_mae[-1],
    }, itera)

