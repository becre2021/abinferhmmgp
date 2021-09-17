from models_utility.param_gp import Param

import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.FloatTensor)
class Gaussian(nn.Module):
    def __init__(self, variance = None, device = None):
        super(Gaussian, self).__init__()
        #return super(Param, cls).__new__(cls, data.double(), requires_grad=requires_grad)
        self.device = torch.device("cuda") if device else torch.device('cpu')
        self.variance = Param(torch.tensor(variance).to(self.device), requires_grad = True, requires_transform=True , param_name= 'noise_variance')
        self.variance_bound = [.25*self.variance.transform().data.clone().detach(), 2.5*self.variance.transform().data.clone().detach()] 

        
    def bound_variance(self):
        self.variance.data = torch.clamp(self.variance.transform().data,min = self.variance_bound[0],max = self.variance_bound[1] ).log()
        return
    
        
    def log_p(slef,F,Y):
        #return densities.gaussian(F,Y,self.variance)
        return

    def predict_mean_variacne(self, mean_f, var_f):
        return mean_f, var_f + self.noise_variance.transform().expand_as(var_f)

    def predict_mean_covariance(self, mean_f, var_f):
        return mean_f, var_f + self.noise_variance.transform().expand_as(var_f).diag().diag()



if __name__ == "__main__":
    gaussian = Gaussian(variance=1.0 , device = True)
    #print([*gaussian.parameters()])
    #print(gaussian.noise_variance)