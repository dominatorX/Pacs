from torch import nn
import torch
from now.utils import make_layers
from now.config import cfg


def forward_by_stage(input, state, subnet, rnn):
    input, state_stage = rnn(input, state, seq_len=cfg.HKO.BENCHMARK.OUT_LEN)
    seq_number, batch_size, input_channel, height, width = input.size()
    input = torch.reshape(input, (-1, input_channel, height, width))
    input = subnet(input)
    input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))

    return input


class BaseForecaster(nn.Module):
    def __init__(self, subnets, cores):
        super().__init__()
        assert len(subnets) == len(cores)

        self.blocks = len(subnets)

        for index, (params, core) in enumerate(zip(subnets, cores)):
            setattr(self, 'core' + str(self.blocks-index), core)
            setattr(self, 'stage' + str(self.blocks-index), make_layers(params))

    def forward(self, hidden_states):
        input = forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'),
                                 getattr(self, 'core3'))
        for i in list(range(1, self.blocks))[::-1]:
            input = forward_by_stage(input, hidden_states[i-1], getattr(self, 'stage' + str(i)),
                                     getattr(self, 'core' + str(i)))
        return input


class Forecaster(BaseForecaster):
    def __init__(self, subnets, cores):
        super(Forecaster, self).__init__(subnets=subnets, cores=cores)


