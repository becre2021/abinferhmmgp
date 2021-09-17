#from __future__ import absolute_import

from models_utility.param_gp import Param
from models_utility.function_gp import cholesky, lt_log_determinant
from torch import triangular_solve

from models_utility.likelihoods import Gaussian

import numpy as np
import torch
import torch.nn as nn


#torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float32)

zitter=1e-6
class gpmodel(nn.Module):
    def __init__(self, kernel, likelihood , device ):
        super(gpmodel,self).__init__()

        self.kernel = kernel
        self.device = torch.device("cuda") if device else torch.device('cpu')
        self.likelihood = Gaussian(variance= 1.00, device = device) if likelihood == None else likelihood
        self.zitter = torch.tensor(zitter).to(self.device)
        self.name = 'gp_sm'
        #self._check_observations(x,y)


    def compute_loss(self):
        raise NotImplementedError

    def _compute_Kxx(self,x):
        num_input = x.shape[0]
        return self.kernel.K(x)


    def _compute_Kxx_diag(self,x):
        return self._compute_Kxx(x).diag()

    def _compute_Kxs(self,x,xstar):
        return self.kernel.K(x,xstar)
    


class gpr(gpmodel):
    def __init__(self, kernel, likelihood, device, lr_hyp , noise_err ):
        super(gpr,self).__init__(kernel, likelihood, device )
        self.lr_emission_hyp = lr_hyp
        self.likelihood = Gaussian(variance= noise_err, device = device) if likelihood == None else likelihood

#         self.optimizer = torch.optim.Adam(self.kernel.parameters(),
#                                           lr=self.lr_emission_hyp,
#                                           betas=(0.9, 0.99),
#                                           eps=1e-08,
#                                           weight_decay=0.0)
                                          
        self.optimizer = torch.optim.Adam(list(self.kernel.parameters()) + list(self.likelihood.parameters()),
                                          lr=self.lr_emission_hyp,
                                          betas=(0.9, 0.99),
                                          eps=1e-08,
                                          weight_decay=0.0)

        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size= 1, gamma= 0.8)
        #self.lr_scheduler = torch.optim.MultiStepLR(self.optimizer, stepsize=10, gamma=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size= 10, gamma=0.9)


        
        
    def compute_loss(self, batch_x, batch_y ):
        num_input,dim_output = batch_y.shape
        # made by me
        K_xx = self._compute_Kxx(batch_x) + (self.likelihood.variance.transform() + self.zitter).expand(num_input,num_input).diag().diag()
        L = cholesky(K_xx)
        alpha = triangular_solve(batch_y,L,upper=False )[0]
        loss = 0.5 * alpha.pow(2).sum() + lt_log_determinant(L) + 0.5 * num_input * np.log(2.0 * np.pi)
        kl_term = 0.0
        return loss,kl_term


    def _get_log_prob(self, batch_x, batch_y ,num_batch ,test_option) :
        emission,kl_term = self.compute_loss(batch_x, batch_y)
        return -emission,kl_term





from kernels.SM_kernel import SM
from models_utility.likelihoods import Gaussian
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1d inputs

    device = False
    Fs = 1000.
    x = np.arange(0, 1, 1 / Fs).reshape(-1, 1)

    print(x)

    y = np.sin(2 * np.pi * x) + np.random.randn(x.shape[0], 1)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    # x = torch.from_numpy(x).to(torch.device("cuda"))
    # y = torch.from_numpy(y).to(torch.device("cuda"))
    # print(x)

    weight = np.array([10., 1., 30., 1., 5.]).reshape(-1, 1)
    mu = np.array([100., 200., 300., 400., 500.]).reshape(-1, 1)
    std = np.random.rand(5, 1)
    noise_variance = .1

    kernel = SM(weight, mu, std, device)
    likelihood = Gaussian(noise_variance, device)
    # gp = gpr(kernel=kernel, likelihood=likelihood, device=device, lr_hyp = 0.001)
    # print(gp._compute_Kxx(x))

