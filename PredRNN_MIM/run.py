import os
import shutil
import argparse
import numpy as np
from core.data_provider.dataloader import HKOIterator
from core.models.model_factory import Model
import core.trainer as trainer
import random
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

# data
in_l = 3
out_l = 6
model_name = 'mim'
'''
predrnn: predrnn
mim: mim
plqMIM: quadratic polynomial+mim
plcMIM: cubic polynomial+mim
oaMIM: other activator+mim
plqPR: quadratic polynomial+predrnn
plcPR: cubic polynomial+predrnn
oaPR: other activator+predrnn
'''
torch.autograd.set_detect_anomaly(True)
parser.add_argument('--dataset_name', type=str, default='HKO')
parser.add_argument('--RAINY_ALL', type=str, default='../data/pd/15hko7_rainy_all.pkl')
parser.add_argument('--train_data_paths', type=str, default='../data/pd/15hko7_rainy_train.pkl')
parser.add_argument('--test_data_paths', type=str, default='../data/pd/15hko7_rainy_test.pkl')
parser.add_argument('--save_dir', type=str, default='/export/data/wjc/learning/torch/hko/new/rain/save/b4/mim/m3')
parser.add_argument('--gen_frm_dir', type=str, default='/export/data/wjc/learning/torch/hko/new/rain/save/b4/mim/m3')
parser.add_argument('--input_length', type=int, default=in_l)
parser.add_argument('--total_length', type=int, default=in_l+out_l)
parser.add_argument('--img_height', type=int, default=162)
parser.add_argument('--img_width', type=int, default=213)
parser.add_argument('--img_channel', type=int, default=3)

# model
parser.add_argument('--model_name', type=str, default=model_name)
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64, 64, 64, 64')
parser.add_argument('--filter_size', type=int, default=3)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--norm', type=int, default=1)
parser.add_argument('--activator', type=str, default='relu')

# scheduled sampling
batch_size = 4
early_factor = 4
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=400000//batch_size//early_factor)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=early_factor*batch_size/400000)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=batch_size)
parser.add_argument('--max_iterations', type=int, default=640000//batch_size//early_factor)
parser.add_argument('--display_interval', type=int, default=8000//batch_size//early_factor)
parser.add_argument('--test_interval', type=int, default=40000//batch_size//early_factor)
parser.add_argument('--snapshot_interval', type=int, default=40000//batch_size//early_factor)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

args = parser.parse_args()
print(args)


def schedule_sampling(eta, itr):
    if not args.scheduled_sampling:
        return 0.0, np.zeros((args.batch_size,
                              args.total_length - args.input_length - 1,
                              (args.img_height-1) // args.patch_size + 1,
                              (args.img_width-1) // args.patch_size + 1,
                              args.patch_size ** 2 * args.img_channel))

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
            (args.batch_size, (args.total_length - args.input_length - 1)))
    true_token = (random_flip < eta)

    real_input_flag = np.zeros((args.batch_size, (args.total_length - args.input_length - 1),
                                (args.img_height-1) // args.patch_size + 1,
                                (args.img_width-1) // args.patch_size + 1,
                                args.patch_size ** 2 * args.img_channel))
    real_input_flag[true_token] = 1

    return eta, real_input_flag


def train_wrapper(model, rid, start_=1):
    # if args.pretrained_model:
    #    model.load(args.pretrained_model)
    # load data
    train_hko_iter = HKOIterator(args.train_data_paths, 'random', args, args.total_length, 0, base_freq='60min')

    eta = args.sampling_start_value - args.sampling_changing_rate*(start_-1)

    for itr in range(start_, args.max_iterations + 1):
        train_batch, train_mask, _, _ = \
            train_hko_iter.sample(batch_size=args.batch_size)

        eta, real_input_flag = schedule_sampling(eta, itr)

        trainer.train(model, train_batch, real_input_flag, train_mask, args, itr)

        if itr % args.snapshot_interval == 0:
            model.save(itr, rid)
    model.save('best', rid)


def test_wrapper(model, rid):
    # model.load(args.pretrained_model)
    test_hko_iter = HKOIterator(args.test_data_paths, 'sequent', args,
                                args.total_length, 0, stride=5, base_freq='60min')
    return trainer.test(model, test_hko_iter, args, 'test_result', rid)


rounds = 3


def train_():
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    if os.path.exists(args.gen_frm_dir):
        shutil.rmtree(args.gen_frm_dir)
    os.makedirs(args.gen_frm_dir)

    for rid in range(rounds):

        os.makedirs(args.save_dir+"/"+str(rid))
        # os.makedirs(args.gen_frm_dir+"/"+str(rid))
        print('Initializing models')

        model = Model(args)
        train_wrapper(model, rid)
        test_wrapper(model, rid)


def test_(id='best'):
    loss = []
    for rid in range(rounds):
        model = Model(args)
        model.load(args.save_dir+"/"+str(rid)+"/model.ckpt-"+str(id))
        loss.append(test_wrapper(model, rid))
    print("load model of", id)
    print(loss)
    print(np.mean(loss))


if __name__ == '__main__':
    test_()
