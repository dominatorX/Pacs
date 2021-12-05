import sys
sys.path.insert(0, '../')
from now.hko.dataloader import HKOIterator
from now.hko.evaluation import HKOEvaluation
from now.config import cfg
import numpy as np
import torch


IN_LEN = 6
modes = 'weighted_average_of_near'  # from 'constant', 'average_of_near', 'weighted_average_of_near'

OUT_LEN = 6
max_iterations = 1
test_iteration_interval = 1
height = 162
width = 213
batch_size = 32

evaluater = HKOEvaluation(seq_len=OUT_LEN, use_central=False)
valid_hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TEST,
                             sample_mode="sequent",
                             batch_size=batch_size,
                             seq_len=IN_LEN + OUT_LEN,
                             stride=cfg.HKO.BENCHMARK.STRIDE,
                             max_consecutive_missing=0,
                             base_freq='60min')
valid_loss = 0.0
valid_time = 0

# for weighted average
thr = [0.4624, 0.5511, 0.6182, 0.7245]
weight = (-1, -3, -5, 10, 30)

while not valid_hko_iter.use_up:
    valid_batch, valid_mask, sample_datetimes, _ = \
        valid_hko_iter.sample(batch_size=batch_size)
    if valid_batch is None:
        print(valid_time)
        break

    valid_time += 1

    valid_data = valid_batch[:IN_LEN, ...]
    valid_label = valid_batch[IN_LEN:IN_LEN + OUT_LEN, :, -1:, ...]

    mask = valid_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)

    if mode == 'constant':
        # 1. constant
        # output = np.ones((1, batch_size, 1, 162, 213))*0.5
        output = np.ones((1, batch_size, 1, 162, 213))*0.03

    if mode == 'average_of_near':
        # 2. non-weighted average of near
        output = valid_data.detach().cpu().numpy().mean(0)[np.newaxis, ...]

    if mode == 'weighted_average_of_near':
        # 3. weighted average of near
        rains = np.zeros((OUT_LEN, batch_size, 1, 162, 213), dtype=np.float64)
        rained = np.zeros((OUT_LEN+IN_LEN-1, batch_size, 1, 162, 213), dtype=np.float64)
        nums = np.zeros((OUT_LEN+IN_LEN-1, batch_size, 1, 162, 213), dtype=np.int64)
        idx = 0
        for frame in valid_data:
            wei = np.zeros((1, batch_size, 1, 162, 213), dtype=np.int64)
            for i in range(len(thr)):
                wei += weight[i] * np.ones((1, batch_size, 1, 162, 213), dtype=np.int64) * (frame < thr[i])
            wei += weight[len(thr)] * np.ones((1, batch_size, 1, 162, 213), dtype=np.int64) * (frame >= thr[-1])
            rained[idx] = frame * wei
            nums[idx] = wei
            idx += 1
        for idx in range(IN_LEN, IN_LEN+OUT_LEN-1):
            rains[idx-IN_LEN] = np.sum(rained[idx-IN_LEN:idx], axis=0)/np.sum(nums[idx-IN_LEN:idx], axis=0)
            wei = np.zeros((1, batch_size, 1, 162, 213), dtype=np.int64)
            for i in range(len(thr)):
                wei += weight[i] * np.ones((1, batch_size, 1, 162, 213), dtype=np.int64) * (rains[idx-IN_LEN] < thr[i])
            wei += weight[len(thr)] * np.ones((1, batch_size, 1, 162, 213), dtype=np.int64) * (rains[idx-IN_LEN] >= thr[-1])
            rained[idx] = rains[idx-IN_LEN] * wei
            nums[idx] = wei
            idx += 1
        rains[OUT_LEN-1] = np.sum(rained[OUT_LEN-IN_LEN:OUT_LEN], axis=0) / np.sum(nums[OUT_LEN-IN_LEN:OUT_LEN], axis=0)

        output = rains

    evaluater.update(valid_label, output, mask)
_, _, valid_csi, valid_hss, _, valid_mse, valid_mae, valid_balanced_mse, valid_balanced_mae, _ = evaluater.calculate_stat()

print('input length: ', IN_LEN)
print('balance mse: ', valid_balanced_mse)
print('balance mae: ', valid_balanced_mae)
print('equivalent loss', np.mean((valid_balanced_mse+valid_balanced_mae))*0.00005)
