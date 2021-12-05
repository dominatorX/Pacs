from now.helpers.ordered_easydict import OrderedEasyDict as edict
import numpy as np
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


__C = edict()
cfg = __C
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


__C.GLOBAL.BATCH_SIZE = 4
__C.GLOBAL.MODEL_NAME = 'ende/conv/m3'


__C.GLOBAL.MODEL_SAVE_DIR = '/export/data/wjc/learning/torch/hko/new/rain/save/b4'
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
__C.HKO_DATA_BASE_PATH = os.path.join(__C.ROOT_DIR, 'data')

for dirs in ['/export/data/civil_proj/HKO_radar_images/radarPNG']:
    if os.path.exists(dirs):
        __C.HKO_PNG_PATH = dirs
for dirs in ['/export/data/civil_proj/HKO_radar_images/radarPNG_mask']:
    if os.path.exists(dirs):
        __C.HKO_MASK_PATH = dirs

__C.HKO = edict()


__C.HKO.EVALUATION = edict()
__C.HKO.EVALUATION.THRESHOLDS = np.array([0.5, 2, 5, 10, 30])  # [0.3283, 0.4624, 0.5511, 0.6182, 0.7245]
__C.HKO.EVALUATION.CENTRAL_REGION = (0, 0, 162, 213)
__C.HKO.EVALUATION.BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)

__C.HKO.EVALUATION.VALID_DATA_USE_UP = False
# __C.HKO.EVALUATION.VALID_TIME = 100
__C.HKO.EVALUATION.VALID_TIME = 128/__C.GLOBAL.BATCH_SIZE


__C.HKO.BENCHMARK = edict()
__C.HKO.BENCHMARK.VISUALIZE_SEQ_NUM = 10  # Number of sequences that will be plotted and saved to the benchmark directory
__C.HKO.BENCHMARK.IN_LEN = 3  # The maximum input length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.OUT_LEN = 6  # The maximum output length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.STRIDE = 5   # The stride

# pandas data
__C.HKO_PD_BASE_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'pd')
if not os.path.exists(__C.HKO_PD_BASE_PATH):
    os.makedirs(__C.HKO_PD_BASE_PATH)

__C.HKO_PD = edict()
__C.HKO_PD.RAINY_TRAIN = os.path.join(__C.HKO_PD_BASE_PATH, '15hko7_rainy_train.pkl')
__C.HKO_PD.RAINY_VALID = os.path.join(__C.HKO_PD_BASE_PATH, '15hko7_rainy_valid.pkl')
__C.HKO_PD.RAINY_TEST = os.path.join(__C.HKO_PD_BASE_PATH, '15hko7_rainy_test.pkl')
__C.HKO_PD.RAINY_ALL = os.path.join(__C.HKO_PD_BASE_PATH, '15hko7_rainy_all.pkl')


__C.HKO.ITERATOR = edict()
__C.HKO.ITERATOR.HEIGHT = 162
__C.HKO.ITERATOR.WIDTH = 213
__C.HKO.ITERATOR.CHANNEL = 3


from now.models.model import activation, RangeNorm

__C.HKO.ITERATOR.NORM = None  # RangeNorm or torch.nn.LayerNorm or None
__C.HKO.ITERATOR.NORM_PEND = 'prepend'  # prepend or append
__C.HKO.ITERATOR.MODEL = 'gru'
__C.HKO.ITERATOR.FRAME = 'ende'

__C.MODEL = edict()
__C.MODEL.CNN_ACT_TYPE = None
#'pac2', 'pac3', activation('relu'), activation("sigmoid"), or None
__C.MODEL.RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)
