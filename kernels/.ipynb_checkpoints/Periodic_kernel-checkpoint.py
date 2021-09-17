from models_utility.param_gp import Param
from kernels.kernel import  StationaryKernel
import torch
import numpy as np


torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float32)


class Periodic(StationaryKernel):

    def __init__(self, variance, length_scale, periodic, device):
        super(Periodic, self).__init__(device)
        self.kernel_name = 'PERIODIC'
        self._assign_Periodic_param(variance, length_scale, periodic)

    def _assign_Periodic_param(self, variance, length_scale, periodic):
        self.variance = Param(torch.tensor(variance).to(self.device),
                              requires_grad=True, requires_transform=True, param_name='periodic_variance')

        self.length_scales = Param(torch.tensor(length_scale).to(self.device),
                                   requires_grad=True, requires_transform=True, param_name='periodic_length')

        self.periodic = Param(torch.tensor(periodic).to(self.device),
                              requires_grad=True, requires_transform=True, param_name='periodic_period')

        return

    def K(self, x1, x2=None):
        pi = np.pi
        x1, x2 = self.check_tensortype(x1, x2)
        x1 = pi * x1.div(self.periodic.transform())
        x2 = pi * x2.div(self.periodic.transform())
        outs = ((torch.sin(x1 - x2.t())) / self.length_scales.transform()).pow(2)

        return self.variance.transform() * torch.exp(-2 * outs)

