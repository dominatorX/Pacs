import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCellv2 as stlstm
from core.layers.MIMBlock import MIMBlock as mimblock
from core.layers.MIMN import MIMN as mimn
from core.layers.MIMBlock import MIMBlockPl as mimblock_pl
from core.layers.MIMN import MIMNPl as mimn_pl
from core.layers.MIMBlock import MIMBlockOA as mimblock_oa
from core.layers.MIMN import MIMNOA as mimn_oa


class mimBase(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(mimBase, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.patch_size = configs.patch_size
        height = (configs.img_height-1) // configs.patch_size + 1
        width = (configs.img_width-1) // configs.patch_size + 1
        self.hw_new = [height, width]
        self.padding = torch.nn.ReflectionPad2d([0, configs.patch_size - configs.img_width % configs.patch_size,
                                                 0, configs.patch_size - configs.img_height % configs.patch_size])

        self.stlstm_layer = []
        self.stlstm_layer_diff = []

    def forward(self, frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        length, batch, n_channel, height, width = frames.shape
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        frames = self.padding(torch.reshape(frames, [-1, n_channel, height, width]))
        a = torch.reshape(frames, [length, batch, n_channel,
                                   self.hw_new[0], self.patch_size,
                                   self.hw_new[1], self.patch_size])
        b = a.permute(1, 0, 2, 4, 6, 3, 5)
        frames = torch.reshape(b, [batch, length, -1, *self.hw_new]).contiguous()

        lstm_c, cell_state, hidden_state, cell_state_diff, hidden_state_diff = [], [], [], [], []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], *self.hw_new]).to(self.configs.device)
            cell_state.append(zeros)
            hidden_state.append(zeros)
        for i in range(self.num_layers-1):
            zeros = torch.zeros([batch, self.num_hidden[i], *self.hw_new]).to(self.configs.device)
            lstm_c.append(zeros)
            cell_state_diff.append(zeros)
            hidden_state_diff.append(zeros)

        st_memory = torch.zeros([batch, self.num_hidden[0], *self.hw_new]).to(self.configs.device)
        next_frames = []

        for t in range(self.configs.total_length-1):
            if t < self.configs.input_length:
                x_gen = frames[:, t]
            else:
                x_gen = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            preh = hidden_state[0]
            hidden_state[0], cell_state[0], st_memory = self.stlstm_layer[0](
                x_gen, hidden_state[0], cell_state[0], st_memory)
            for i in range(1, self.num_layers):
                if t > 0:
                    if i == 1:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            hidden_state[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                    else:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
                else:
                    self.stlstm_layer_diff[i - 1](torch.zeros_like(hidden_state[i - 1]).to(self.configs.device),
                                                  torch.zeros_like(hidden_state[i - 1]).to(self.configs.device),
                                                  torch.zeros_like(hidden_state[i - 1]).to(self.configs.device))
                preh = hidden_state[i]
                hidden_state[i], cell_state[i], st_memory, lstm_c[i-1] = self.stlstm_layer[i](
                    hidden_state[i - 1], hidden_state_diff[i - 1], hidden_state[i],
                    cell_state[i], st_memory, lstm_c[i-1])
            x_gen = self.conv_last(hidden_state[self.num_layers-1])

            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).contiguous()
        a = torch.reshape(next_frames, [self.configs.total_length - 1, batch, -1,
                                        self.patch_size, self.patch_size, *self.hw_new])[:, :, -1, ...]
        b = a.permute(0, 1, 4, 2, 5, 3)
        next_frames = torch.reshape(b, [self.configs.total_length - 1, batch, 1, self.hw_new[0] * self.patch_size,
                                        self.hw_new[1] * self.patch_size])[:, :, :, :height, :width].contiguous()
        return next_frames


class mim(mimBase):
    def __init__(self, num_layers, num_hidden, configs):
        super(mim, self).__init__(num_layers, num_hidden, configs)

        for i in range(num_layers):
            if i == 0:
                num_hidden_in = self.frame_channel
            else:
                num_hidden_in = num_hidden[i - 1]
            if i < 1:
                new_stlstm_layer = stlstm(num_hidden_in,
                                          num_hidden[i],
                                          self.hw_new,
                                          configs.filter_size,
                                          configs.stride,
                                          tln=configs.norm).to(self.configs.device)
            else:
                new_stlstm_layer = mimblock(num_hidden_in,
                                            num_hidden[i],
                                            self.hw_new,
                                            configs.filter_size,
                                            configs.stride,
                                            configs.device,
                                            tln=configs.norm).to(self.configs.device)
            self.stlstm_layer.append(new_stlstm_layer)

        for i in range(num_layers - 1):
            new_stlstm_layer = mimn(num_hidden[i + 1],
                                    self.hw_new,
                                    configs.filter_size,
                                    tln=configs.norm).to(self.configs.device)
            self.stlstm_layer_diff.append(new_stlstm_layer)
        self.stlstm_layer = nn.ModuleList(self.stlstm_layer)
        self.stlstm_layer_diff = nn.ModuleList(self.stlstm_layer_diff)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False).to(self.configs.device)


class mimpl(mimBase):
    def __init__(self, num_layers, num_hidden, configs):
        super(mimpl, self).__init__(num_layers, num_hidden, configs)

        order = 2 if 'plq' in configs.model_name else 3

        for i in range(num_layers):
            if i == 0:
                num_hidden_in = self.frame_channel
            else:
                num_hidden_in = num_hidden[i - 1]
            if i < 1:
                new_stlstm_layer = stlstm(num_hidden_in, num_hidden[i], self.hw_new, configs.filter_size,
                                          configs.stride, tln=1).to(self.configs.device)
            elif i % 2 == 1:
                new_stlstm_layer = mimblock_pl(num_hidden_in, num_hidden[i], self.hw_new, configs.filter_size,
                                            configs.stride, configs.device, tln=configs.norm, order=order).to(
                    self.configs.device)
            else:
                new_stlstm_layer = mimblock(num_hidden_in, num_hidden[i], self.hw_new, configs.filter_size,
                                            configs.stride, configs.device, tln=1).to(self.configs.device)
            self.stlstm_layer.append(new_stlstm_layer)

        for i in range(num_layers - 1):
            if i % 2 == 1:
                new_stlstm_layer = mimn_pl(num_hidden[i + 1], self.hw_new, configs.filter_size,
                                           tln=configs.norm, order=order).to(self.configs.device)
            else:
                new_stlstm_layer = mimn(num_hidden[i + 1], self.hw_new, configs.filter_size,
                                        tln=1).to(self.configs.device)
            self.stlstm_layer_diff.append(new_stlstm_layer)
        self.stlstm_layer = nn.ModuleList(self.stlstm_layer)
        self.stlstm_layer_diff = nn.ModuleList(self.stlstm_layer_diff)

        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False).to(self.configs.device)


class mimoa(mimBase):
    def __init__(self, num_layers, num_hidden, configs):
        super(mimoa, self).__init__(num_layers, num_hidden, configs)
        if configs.activator == 'relu':
            activator = nn.ReLU()
        elif configs.activator == 'sigmoid':
            activator = nn.Sigmoid()
        else:
            print('please define the activation function you use here')
            exit(1)

        for i in range(num_layers):
            if i == 0:
                num_hidden_in = self.frame_channel
            else:
                num_hidden_in = num_hidden[i - 1]
            if i < 1:
                new_stlstm_layer = stlstm(num_hidden_in, num_hidden[i], self.hw_new, configs.filter_size,
                                          configs.stride, tln=1).to(self.configs.device)
            elif i % 2 == 1:
                new_stlstm_layer = mimblock_oa(num_hidden_in, num_hidden[i], self.hw_new, configs.filter_size,
                                            configs.stride, configs.device, tln=configs.norm, act=activator).to(
                    self.configs.device)
            else:
                new_stlstm_layer = mimblock(num_hidden_in, num_hidden[i], self.hw_new, configs.filter_size,
                                            configs.stride, configs.device, tln=1).to(self.configs.device)
            self.stlstm_layer.append(new_stlstm_layer)

        for i in range(num_layers - 1):
            if i % 2 == 1:
                new_stlstm_layer = mimn_oa(num_hidden[i + 1], self.hw_new, configs.filter_size,
                                           tln=configs.norm, act=activator).to(self.configs.device)
            else:
                new_stlstm_layer = mimn(num_hidden[i + 1], self.hw_new, configs.filter_size,
                                        tln=1).to(self.configs.device)
            self.stlstm_layer_diff.append(new_stlstm_layer)
        self.stlstm_layer = nn.ModuleList(self.stlstm_layer)
        self.stlstm_layer_diff = nn.ModuleList(self.stlstm_layer_diff)

        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False).to(self.configs.device)
