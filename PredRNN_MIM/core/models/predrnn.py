import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell, SpatioTemporalLSTMCellOtherAct,\
    SpatioTemporalLSTMCellPoly


class PredRNNBase(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(PredRNNBase, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.patch_size = configs.patch_size
        cell_list = []
        height = (configs.img_height-1) // configs.patch_size + 1
        width = (configs.img_width-1) // configs.patch_size + 1
        self.hw_new = [height, width]

        self.padding = torch.nn.ReflectionPad2d([0, configs.patch_size - configs.img_width % configs.patch_size,
                                                 0, configs.patch_size - configs.img_height % configs.patch_size])

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

        next_frames, h_t, c_t = [], [], []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], *self.hw_new]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], *self.hw_new]).to(self.configs.device)

        for t in range(self.configs.total_length-1):
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).contiguous()
        a = torch.reshape(next_frames, [self.configs.total_length - 1, batch, -1,
                                        self.patch_size, self.patch_size, *self.hw_new])[:, :, -1, ...]
        b = a.permute(0, 1, 4, 2, 5, 3)
        next_frames = torch.reshape(b, [self.configs.total_length - 1, batch, 1, self.hw_new[0] * self.patch_size,
                                        self.hw_new[1] * self.patch_size])[:, :, :, :height, :width].contiguous()
        return next_frames


class PredRNN(PredRNNBase):
    def __init__(self, num_layers, num_hidden, configs):
        super(PredRNN, self).__init__(num_layers, num_hidden, configs)

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], (height, width), configs.filter_size,
                                       configs.stride, configs.norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)



class PredRNNpl(PredRNNBase):
    def __init__(self, num_layers, num_hidden, configs):
        super(PredRNNpl, self).__init__(num_layers, num_hidden, configs)

        order = 2 if 'plq' in configs.model_name else 3
        print('order', order)
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            new_layer = SpatioTemporalLSTMCellPoly(in_channel, num_hidden[i], (height, width), configs.filter_size,
                                                   configs.stride, configs.norm, order)
            cell_list.append(new_layer)
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, 1, stride=1, padding=0, bias=False)



class PredRNNoa(PredRNNBase):
    def __init__(self, num_layers, num_hidden, configs):
        super(PredRNNoa, self).__init__(num_layers, num_hidden, configs)

        if configs.activator == 'relu':
            activator = nn.ReLU()
        elif configs.activator == 'sigmoid':
            activator = nn.Sigmoid()
        else:
            print('please define the activation function you use here')
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            new_layer = SpatioTemporalLSTMCellOtherAct(in_channel, num_hidden[i], (height, width), configs.filter_size,
                                                   configs.stride, configs.norm, activator)
            cell_list.append(new_layer)
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, 1, stride=1, padding=0, bias=False)
