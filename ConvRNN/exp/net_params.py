import sys
sys.path.insert(0, '..')
from now.hko.evaluation import *
from now.models.ConvGRU import ConvGRU


batch_size = cfg.GLOBAL.BATCH_SIZE

# convgru
f1, f2, f3 = 8, 64, 192
endegru_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [cfg.HKO.ITERATOR.CHANNEL, f1, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [f2, f2, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [f3, f3, 3, 2, 1]}),
    ],

    [
        ConvGRU(input_channel=f1, num_filter=f2, b_h_w=(batch_size, 32, 42),
                kernel_size=3, stride=1, padding=1, st_kernel=5),
        ConvGRU(input_channel=f2, num_filter=f3, b_h_w=(batch_size, 10, 14),
                kernel_size=3, stride=1, padding=1, st_kernel=5),
        ConvGRU(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 5, 7),
                kernel_size=3, stride=1, padding=1, st_kernel=3),
    ]
]
endegru_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [f3, f3, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [f3, f3, 5, 3, (0, 1)]}),
        OrderedDict({
            'deconv3_leaky_1': [f2, f1, (7, 8), 5, 0],
            # 'conv3_leaky_2': [f1, f1, 3, 1, 1],
            'conv3_3': [f1, 1, 1, 1, 0]
        }),
    ],

    [
        ConvGRU(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 5, 7),
                kernel_size=3, stride=1, padding=1, st_kernel=3),
        ConvGRU(input_channel=f3, num_filter=f3, b_h_w=(batch_size, 10, 14),
                kernel_size=3, stride=1, padding=1, st_kernel=5),
        ConvGRU(input_channel=f3, num_filter=f2, b_h_w=(batch_size, 32, 42),
                kernel_size=3, stride=1, padding=1, st_kernel=5),
    ]
]