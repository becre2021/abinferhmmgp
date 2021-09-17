
import torch
import torch.nn as nn
import numpy as np

torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float32)



class Kernel(nn.Module):
    def __init__(self):
        super(Kernel, self).__init__()

    def check_tensortype(self, x1, x2=None):
        if torch.is_tensor(x1) == False:
            x1 = TensorType(x1)
        if x2 is None:
            return x1, x1
        else:
            if torch.is_tensor(x2) == False:
                x2 = TensorType(x2)
            return x1, x2


    def K(self, x1, x2 = None):
        return NotImplementedError

    def K_diag(self, x1, x2=None):
        return torch.diag(self.K(x1, x2))




class StationaryKernel(Kernel):
    def __init__(self, device ):
        super(StationaryKernel, self).__init__()
        self.kernel_name = None
        self.ARD = False
        self.input_dim = None
        self.device = torch.device("cuda") if device else torch.device('cpu')


    def _sq_dist(self,x1,x2 = None):
        x1,x2 = self.check_tensortype(x1,x2)

        x1_ = x1.pow(2).sum(-1,keepdim = True)
        x2_ = x2.pow(2).sum(-1,keepdim = True)

        #sq_dist = x1.matmul(x2.transpose(-2,-1)).mul_(-2).add_(x2_.transpose(-2,-1)).add_(x1_)
        sq_dist = -2*x1.matmul(x2.transpose(-2, -1)) + (x1_ + x2_.transpose(-2, -1))
        sq_dist.clamp_min(1e-16)

        return sq_dist



    def K(self, x1, x2 = None):
        return NotImplementedError


    def K_diag(self, x1, x2=None):
        return torch.diag(self.K(x1, x2))



class Sum_kernel(StationaryKernel):

    def __init__(self, kernel_list, device):
        super(Sum_kernel, self).__init__(device)
        self.kernel_name = 'SUM'
        self.kernel_list = kernel_list
        self._set_paramlist()

    def _set_paramlist(self):
        param_list = []
        for ith_kernel in self.kernel_list:
            for ith_param in ith_kernel.parameters():
                if ith_param.requires_grad:
                    param_list.append(ith_param)
        self.param_list = param_list
        return


    def K(self, x1, x2=None):
        out = 0
        for ith_kernel in self.kernel_list:
            out += ith_kernel.K(x1, x2)
        return out



class Product_kernel(StationaryKernel):
    def __init__(self, kernel_list, device):
        super(Product_kernel, self).__init__(device)
        self.kernel_list = kernel_list
        self._set_paramlist()

    def _set_paramlist(self):
        param_list = []
        for ith_kernel in self.kernel_list:
            for ith_param in ith_kernel.parameters():
                if ith_param.requires_grad:
                    param_list.append(ith_param)
        self.param_list = param_list
        return

    def K(self, x1, x2=None):
        out = 1
        for ith_kernel in self.kernel_list:
            out *= ith_kernel.K(x1, x2)
        return out
