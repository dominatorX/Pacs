import torch
from torch import nn
from now.config import cfg
# input: B, C, H, W
# flow: [B, 2, H, W]


def activate(act_type, layer):
    if act_type is not None:
        if act_type == 'pac2':
            return 2*layer**2-1
        elif act_type == 'pac3':
            return 4*layer**3-3*layer
        else:
            return act_type(layer)
    return layer


norm_pend=cfg.HKO.ITERATOR.NORM_PEND


class ConvGRU(nn.Module):
    # b_h_w: input feature map size
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1,
                 st_kernel=3, rnn_act_type=cfg.MODEL.RNN_ACT_TYPE, cnn_act_type=cfg.MODEL.CNN_ACT_TYPE,
                 norm=cfg.HKO.ITERATOR.NORM):
        super(ConvGRU, self).__init__()
        self._batch_size, self._state_height, self._state_width = b_h_w
        self._input_channel = input_channel
        self._norm = None if norm is None else norm([num_filter, self._state_height, self._state_width])
        self._cnn_act_type = cnn_act_type
        self._rnn_act_type = rnn_act_type
        self._num_filter = num_filter
        self.i2h = nn.Conv2d(in_channels=input_channel,
                             out_channels=self._num_filter * 3,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        self.h2h = nn.Conv2d(in_channels=num_filter,
                             out_channels=self._num_filter * 3,
                             kernel_size=st_kernel,
                             stride=stride,
                             padding=st_kernel // 2)

    def forward(self, inputs=None, states=None, seq_len=cfg.HKO.BENCHMARK.IN_LEN):
        if states is None:
            states = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)

        outputs = []
        for index in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((states.size(0), self._input_channel, self._state_height,
                                 self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
            else:
                x = inputs[index, ...]
            conv_x = self.i2h(x)
            conv_h = self.h2h(states)

            if self._norm is not None:
                if norm_pend == 'prepend':
                    xz, xr, xh = torch.chunk(conv_x, 3, dim=1)
                    hz, hr, hh = torch.chunk(conv_h, 3, dim=1)
                    xz, xr, xh, hz, hr, hh = [self._norm(xhs) for xhs in [xz, xr, xh, hz, hr, hh]]
                    if self._cnn_act_type:
                        xz, xr, xh, hz, hr, hh = [activate(self._cnn_act_type, xhs) for xhs
                                                  in [xz, xr, xh, hz, hr, hh]]
                elif norm_pend == 'append':
                    if self._cnn_act_type:
                        conv_x = activate(self._cnn_act_type, conv_x)
                        conv_h = activate(self._cnn_act_type, conv_h)
                    xz, xr, xh = torch.chunk(conv_x, 3, dim=1)
                    hz, hr, hh = torch.chunk(conv_h, 3, dim=1)
                    xz, xr, xh, hz, hr, hh = self._norm(xz), self._norm(xr), self._norm(xh), \
                                                 self._norm(hz), self._norm(hr), self._norm(hh)
            else:
                xz, xr, xh = torch.chunk(conv_x, 3, dim=1)
                hz, hr, hh = torch.chunk(conv_h, 3, dim=1)
                if self._cnn_act_type:
                    xz, xr, xh, hz, hr, hh = [activate(self._cnn_act_type, xhs) for xhs
                                              in [xz, xr, xh, hz, hr, hh]]

            zt = torch.sigmoid(xz+hz)
            rt = torch.sigmoid(xr+hr)
            h_ = activate(self._rnn_act_type, xh+rt*hh)
            h = (1-zt)*h_+zt*states
            outputs.append(h)
            states = h

        return torch.stack(outputs), states



