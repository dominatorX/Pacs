from torch import nn
import torch.nn.functional as F
import torch


class RangeNorm(torch.nn.Module):
    def __init__(self, normalized_shape):
        super(RangeNorm, self).__init__()
        #self.shape = 1
        #for i in normalized_shape:
        #    self.shape *= i

    def forward(self, x):
        if torch.eq(x.std(), 0.0):
            return x
        shape_ = x.shape
        x_ = x.view(shape_[0], -1)
        min_ = x_.min(dim=-1, keepdim=True)[0]
        max_ = x_.max(dim=-1, keepdim=True)[0]
        return ((2 * x_ - (min_ + max_)) / (max_ - min_)).view(shape_)


class activation():
    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


class EF(nn.Module):
    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster

    def forward(self, input):
        state = self.encoder(input)
        output = self.forecaster(state)
        return output


