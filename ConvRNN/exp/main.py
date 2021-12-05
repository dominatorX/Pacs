import sys
sys.path.insert(0, '../')
from now.config import cfg
import os
from now.models.forecaster import Forecaster
from now.models.encoder import Encoder
from now.models.model import *
from torch.optim import lr_scheduler
from now.models.loss import Weighted_mse_mae
from now.train_and_test import Trainer
from exp.net_params import *
import numpy as np
import random
import shutil


seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

batch_size = cfg.GLOBAL.BATCH_SIZE
early_factor = 4
max_iterations = 320000//batch_size//early_factor
test_iteration_interval = 40000//batch_size//early_factor
test_and_save_checkpoint_iterations = 40000//batch_size//early_factor
LR_step_size = 80000//batch_size//early_factor
gamma = 0.7

LR = 1e-4

folder_name = cfg.GLOBAL.MODEL_NAME
save_dir = os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, folder_name)
rounds = 3


def get_trainer(rid):
    criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)

    print(cfg.HKO.ITERATOR.FRAME + cfg.HKO.ITERATOR.MODEL + str(cfg.HKO.BENCHMARK.IN_LEN))
    if cfg.HKO.ITERATOR.FRAME == 'ende':
        params = (globals()["ende" + cfg.HKO.ITERATOR.MODEL + "_encoder_params"],
                  globals()["ende" + cfg.HKO.ITERATOR.MODEL + "_forecaster_params"])

        encoder = Encoder(params[0][0], params[0][1]).to(cfg.GLOBAL.DEVICE)
        forecaster = Forecaster(params[1][0], params[1][1]).to(cfg.GLOBAL.DEVICE)
        main_model = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

    optimizer = torch.optim.Adam(main_model.parameters(), lr=LR)  # Adadelta
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)

    trainer = Trainer(main_model, optimizer, criterion, exp_lr_scheduler, batch_size, max_iterations,
                      test_iteration_interval, test_and_save_checkpoint_iterations, folder_name, rid=rid)
    return trainer


def train():
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    for rid in range(rounds):
        trainer = get_trainer(rid)
        trainer.train()


def test(id='best'):
    loss = []
    for rid in range(rounds):
        trainer = get_trainer(rid)
        loss.append(trainer.test(id))
    print("load model of ", id)
    print(loss)
    print(np.mean(loss))


if __name__ == '__main__':
    test()
