import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import predrnn, mim
from core.models.evaluation import HKOEvaluation
import numpy as np


def rainfall_to_pixel(rainfall_intensity, a=58.53, b=1.56):
    dBR = np.log10(rainfall_intensity) * 10.0
    # dBZ = 10b log(R) +10log(a)
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals


class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=None):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA

    def mse_mae(self, input, target, mask):
        balancing_weights = (1, 1, 2, 5, 10, 30)
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in np.array([0.5, 2, 5, 10, 30])]
        for i, threshold in enumerate(thresholds):
            weights += (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        weights *= mask.float()
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input - target) ** 2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input - target))), (2, 3, 4))
        return mse, mae

    def forward(self, input, target, mask):
        balancing_weights = (1, 1, 2, 5, 10, 30)
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in np.array([0.5, 2, 5, 10, 30])]
        for i, threshold in enumerate(thresholds):
            weights += (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        weights *= mask.float()
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)

        train_loss = self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))
        return train_loss


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn': predrnn.PredRNN,
            'mim': mim.mim,
            'plqPR': predrnn.PredRNNpl, # polynomial-quadratic function + mim
            'plcPR': predrnn.PredRNNpl, # polynomial-cubic function + mim
            'oaPR': predrnn.PredRNNoa, # other activator + mim
            'plqMIM': mim.mimpl, # polynomial-quadratic function + predrnn
            'plcMIM': mim.mimpl, # polynomial-cubic function + predrnn
            'oaMIM': mim.mimoa, # other activator + predrnn
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.criterion = Weighted_mse_mae()
        self.evaluator = HKOEvaluation(self.configs.total_length-self.configs.input_length)

    def save(self, itr, rid):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, str(rid))
        checkpoint_path = os.path.join(checkpoint_path, 'model.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask_in, mask_out):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_out = torch.FloatTensor(mask_out).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask_in).to(self.configs.device)
        self.optimizer.zero_grad()
        next_frames = self.network(frames_tensor, mask_tensor)
        loss = self.criterion(next_frames, frames_tensor[1:, :, -1:], mask_out[1:])

        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask_in, mask_out):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask_in).to(self.configs.device)
        mask_out = torch.FloatTensor(mask_out).to(self.configs.device)
        with torch.no_grad():
            self.network.eval()
            next_frames = self.network(frames_tensor, mask_tensor)
            f_t_eva = frames_tensor[self.configs.input_length:, :, -1:]
            f_t_eva_numpy = f_t_eva.cpu().numpy()
            f_t_eva_numpy[np.isnan(f_t_eva_numpy)] = 0.0
            loss = self.criterion(next_frames[self.configs.input_length-1:],
                                  f_t_eva, mask_out[self.configs.input_length:])
            mse, mae = self.criterion.mse_mae(next_frames[self.configs.input_length-1:],
                                              f_t_eva, mask_out[self.configs.input_length:])
            self.evaluator.update(np.clip(next_frames[self.configs.input_length-1:].cpu().numpy(), 0.0, 1.0),
                                  f_t_eva_numpy,
                                  mask_out[self.configs.input_length:].cpu().numpy())
        return next_frames.detach().cpu().numpy(), loss, mse.detach().cpu().numpy().mean(1),\
               mae.detach().cpu().numpy().mean(1)
