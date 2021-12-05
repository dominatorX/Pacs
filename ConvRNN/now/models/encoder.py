from torch import nn
import torch
from now.utils import make_layers
import logging


class BaseEncoder(nn.Module):
    def __init__(self, subnets, cores, output_state):
        super().__init__()
        assert len(subnets) == len(cores)

        self.blocks = len(subnets)
        self.output_state = output_state

        for index, (params, core) in enumerate(zip(subnets, cores), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'core'+str(index), core)

    def forward_(self, stage_forward, input):
        hidden_states = []
        logging.debug(input.size())
        for i in range(1, self.blocks + 1):
            input, state_stage = stage_forward(input, getattr(self, 'stage' + str(i)),
                                               getattr(self, 'core' + str(i)))
            hidden_states.append(state_stage if self.output_state else input)
        return tuple(hidden_states)


def forward_by_stage(input, subnet, core):
    seq_number, batch_size, input_channel, height, width = input.size()

    if subnet is not None:
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
    outputs_stage, state_stage = core(input, None, seq_len=seq_number)

    return outputs_stage, state_stage


class Encoder(BaseEncoder):
    def __init__(self, subnets, cores, output_state=True):
        super(Encoder, self).__init__(subnets, cores, output_state)

    def forward(self, input):
        return self.forward_(forward_by_stage, input)


