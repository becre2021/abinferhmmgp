from models.emission_gp import gpmodel
from models_utility.param_gp import Param
from models_utility.function_gp import cholesky, lt_log_determinant
from torch import triangular_solve


import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions import kl_divergence
from models_utility.likelihoods import Gaussian


torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float32)

import math

pi = torch.tensor(math.pi)

class ssgpr_rep_sm(gpmodel):
    def __init__(self, num_batch, num_sample_pt, param_dict, kernel=None, likelihood=None, device = None ):
        super(ssgpr_rep_sm, self).__init__(kernel = kernel, likelihood = likelihood, device = device)
        self.device = torch.device("cuda") if device else torch.device('cpu')
        self.num_batch = num_batch
        self.num_sample_pt = num_sample_pt
        self._set_up_param(param_dict)
        self.total_num_sample = self.num_sample_pt*self.weight.numel()
        self.lambda_w = 0.0


    def _set_up_param(self, param_dict):
        self.input_dim = param_dict['mean'].shape[1]
        self.num_Q = param_dict['mean'].shape[0]

#         self.sf2 = Param(torch.tensor(param_dict['noise_variance']).to(self.device), requires_grad=False, requires_transform=True , param_name='sf2')
#         self.mu = Param(torch.tensor(param_dict['mean']).view(-1,self.input_dim).to(self.device) , requires_grad=True, requires_transform=True , param_name='mu')
#         self.std = Param(0.5*torch.tensor(param_dict['std']).view(-1,self.input_dim).to(self.device), requires_grad=True, requires_transform=True , param_name='std')        
#         self.weight = Param(torch.tensor(param_dict['weight']).view(-1,self.input_dim).to(self.device) , requires_grad=True, requires_transform=True , param_name='weight')

        self.sf2 = Param(torch.tensor(param_dict['noise_variance']).to(self.device), requires_grad=False, requires_transform=True , param_name='sf2')
        self.mu = Param(torch.tensor(param_dict['mean']).view(-1,self.input_dim).to(self.device) , requires_grad=True, requires_transform=True , param_name='mu')
        self.std = Param(0.5*torch.tensor(param_dict['std']).view(-1,self.input_dim).to(self.device), requires_grad=True, requires_transform=True , param_name='std')        
        self.weight = Param(torch.tensor(param_dict['weight']).view(-1,self.input_dim).to(self.device) , requires_grad=True, requires_transform=True , param_name='weight')


        return


    def _get_param(self):
        for ith in self.parameters():
            print('%s : %s'%(ith.param_name,ith.transform()))


    def _sampling_gaussian(self, mu, std, num_sample):
        eps = Variable(torch.randn(num_sample, self.input_dim).to(self.device))
        return mu + std.mul(eps)


    def _compute_gaussian_basis(self, x, xstar=None):
        sampled_spectral_pt = self._sampling_gaussian(self.mu.transform(),
                                                  self.std.transform(),
                                                  self.num_sample_pt)  # self.num_sample x dim
        xdotspectral = x.matmul(sampled_spectral_pt.t())
        Phi = torch.cat([xdotspectral.cos(), xdotspectral.sin()], 1).to(self.device)
        if xstar is None:
            return Phi
        else:
            xstardotspectral = xstar.matmul(sampled_spectral_pt.t())
            Phi_star = torch.cat([xstardotspectral.cos(), xstardotspectral.sin()], 1).to(self.device)
            return Phi, Phi_star


    def _compute_sm_basis(self, x, xstar=None):
        # assert (self.weight.shape[0] > 1)
        if self.weight.shape[0] > 1:
            current_pi = self.weight.transform().reshape([1, -1]).squeeze()
        else:
            current_pi = self.weight.transform()

        multiple_Phi = []
        current_sampled_spectral_list = []


        for i_th in range(self.weight.numel()):
            sampled_spectal_pt = self._sampling_gaussian(self.mu.transform()[i_th],
                                                         self.std.transform()[i_th],
                                                         self.num_sample_pt)  # self.num_sample x dim

            if xstar is not None:
                current_sampled_spectral_list.append(sampled_spectal_pt)
            xdotspectral = (2 * pi) * x.matmul(sampled_spectal_pt.t())


            Phi_i_th = (current_pi[i_th] / self.num_sample_pt).sqrt() * torch.cat([xdotspectral.cos(), xdotspectral.sin()], 1).to(self.device)
            multiple_Phi.append(Phi_i_th)

        if xstar is None:
            return torch.cat(multiple_Phi, 1)
        else:
            multiple_Phi_star = []
            for i_th, current_sampled in enumerate(current_sampled_spectral_list):
                xstardotspectral = (2 * pi) * xstar.matmul(current_sampled.t())
                Phistar_i_th = (current_pi[i_th] / self.num_sample_pt).sqrt() * torch.cat([xstardotspectral.cos(), xstardotspectral.sin()],1).to(self.device)
                multiple_Phi_star.append(Phistar_i_th)
            return torch.cat(multiple_Phi, 1), torch.cat(multiple_Phi_star, 1)


        
        
    def _compute_gram_approximate(self, Phi):
        return  Phi.t().matmul(Phi) + (self.likelihood.variance.transform()**2 + self.zitter).expand(Phi.shape[1], Phi.shape[1]).diag().diag()
    
    

    def _compute_kernel_sm_approximate(self, x=None):
        if x == None:
            Phi_list = self._compute_sm_basis(self.x)
        else:
            Phi_list = self._compute_sm_basis(x)
        return (self.sf2.transform() / self.total_num_sample) * Phi_list.matmul(Phi_list.t())



    

    def compute_loss(self,batch_x,batch_y,num_batch = 1):
        num_input = batch_x.shape[0]
        loss = 0
        for j_th in range(self.num_batch):
            Phi = self._compute_sm_basis(batch_x)
            Approximate_gram = self._compute_gram_approximate(Phi)
            L = cholesky(Approximate_gram)
            Linv_PhiT = triangular_solve(Phi.t(), L ,upper=False)[0]           
            Linv_PhiT_y = Linv_PhiT.matmul(batch_y) 
            
            loss += (0.5 / self.likelihood.variance.transform()**2) * (batch_y.pow(2).sum() - Linv_PhiT_y.pow(2).sum())
            loss += lt_log_determinant(L)
            loss += (-self.total_num_sample)* (2* self.likelihood.variance)
            loss += 0.5 * num_input * (np.log(2*pi) + 2*self.likelihood.variance )


        kl_term = 0.0
        return (1 / self.num_batch) * loss ,kl_term    

    
    
    
    def _get_log_prob(self, batch_x, batch_y ,num_batch , test_option) :
        emission,kl_term = self.compute_loss(batch_x, batch_y,num_batch=1)
        return -emission,kl_term


    
    
    
    



class ssgpr_rep_sm_reg(ssgpr_rep_sm):
    def __init__(self, num_batch, num_sample_pt,  param_dict, kernel=None, likelihood=None, device=None):

        super(ssgpr_rep_sm_reg, self).__init__(num_batch = num_batch,
                                                           num_sample_pt = num_sample_pt,
                                                           param_dict = param_dict,
                                                           kernel=kernel,
                                                           likelihood=likelihood,
                                                           device = device)

        self.likelihood = Gaussian(variance = param_dict['noise_err'], device = device) if likelihood == None else likelihood
        self.name = 'gp_asm'



        self.lr_emission_hyp = param_dict['lr_hyp']
        self._set_up_param(param_dict)
#         self.optimizer = torch.optim.Adam(self.parameters(),
#                                           lr=self.lr_emission_hyp,
#                                           betas=(0.9, 0.99),
#                                           eps=1e-08,
#                                           weight_decay=0.0)

        self.optimizer = torch.optim.Adam([self.weight,self.mu,self.std]  + list(self.likelihood.parameters()),
                                           lr=self.lr_emission_hyp,
                                           betas=(0.9, 0.99),
                                           eps=1e-08,
                                           weight_decay=0.0)

    
#         self.optimizer = torch.optim.Adam([self.weight,self.mu,self.std],
#                                            lr=self.lr_emission_hyp,
#                                            betas=(0.9, 0.99),
#                                            eps=1e-08,
#                                            weight_decay=0.0)
        

        
        
#         self.optimizer = torch.optim.Adam([{'params': self.likelihood.parameters(), 'weight_decay': 0},
#                                            {'params': {'weight':self.weight,'mu':self.mu,'std':self.std}, 'weight_decay': 0}],
#                                            lr=self.lr_emission_hyp,
#                                            betas=(0.9, 0.99),
#                                            eps=1e-08,
#                                            weight_decay=0.0)
        

        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size= 1, gamma= 0.8)        
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size= 1, gamma= 0.8)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size= 10, gamma=0.9)
        
        self.lambda_w = 0.0
        


    def _set_up_param(self, param_dict):
        self.num_Q, self.input_dim = param_dict['mean'].shape
        self.sf2 = Param(torch.tensor(1.0).to(self.device), requires_grad=False, requires_transform=True , param_name='sf2')
        self.weight = Param(torch.tensor(param_dict['weight']).view(-1,1).to(self.device) , requires_grad=True, requires_transform=True , param_name='weight')
        self.mu = Param(torch.tensor(param_dict['mean']).view(-1,self.input_dim).to(self.device), requires_grad=True, requires_transform=True, param_name='mu')
        self.mu_prior = Param(torch.tensor(param_dict['mean']+ .1 * np.random.randn(self.num_Q, self.input_dim)).view(-1,self.input_dim).to(self.device) , requires_grad=False, requires_transform=True , param_name='mu_prior')

        self.std = Param(.2*torch.tensor(param_dict['std']).view(-1, self.input_dim).to(self.device), requires_grad=True,requires_transform=True, param_name='std')
        self.std_prior = Param(torch.tensor(param_dict['std']).view(-1,self.input_dim).to(self.device), requires_grad=False, requires_transform=True , param_name='std_prior')


        self.kl_option = True

        return



    def _assign_num_spectralpt(self,sampling_option= 'equal'):
        if sampling_option == 'naive_weight':
            assign_rate = F.softmax(self.weight,dim = 0).squeeze()
            assigned_spt = [max(int(ith),1) for ith in self.total_num_sample * assign_rate]
            return assigned_spt

        else:
            return [self.num_sample_pt for ith in range(self.num_Q)]



    def _compute_sm_basis(self, x , xstar=None):

        multiple_Phi = []
        current_sampled_spectral_list = []
        if self.weight.shape[0] > 1:
            current_pi = self.weight.transform().reshape([1, -1]).squeeze()
        else:
            current_pi = self.weight.transform()


        num_samplept_list = self._assign_num_spectralpt()
        for i_th in range(self.weight.numel()):
            ith_allocated_sample = num_samplept_list[i_th]
            sampled_spectal_pt = self._sampling_gaussian(mu = self.mu.transform()[i_th],
                                                         std = self.std.transform()[i_th],
                                                         num_sample = ith_allocated_sample)  # self.num_sample x dim

            if xstar is not None:
                current_sampled_spectral_list.append(sampled_spectal_pt)

                
            xdotspectral = (2 * pi) * x.matmul(sampled_spectal_pt.t())

            Phi_i_th = (current_pi[i_th] / ith_allocated_sample).sqrt() * torch.cat([xdotspectral.cos(), xdotspectral.sin()], 1).to(self.device)
            multiple_Phi.append(Phi_i_th)

        if xstar is None:
            return torch.cat(multiple_Phi, 1)
        else:
            multiple_Phi_star = []
            for i_th, current_sampled in enumerate(current_sampled_spectral_list):
                xstardotspectral = (2 * pi) * xstar.matmul(current_sampled.t())


                Phistar_i_th = (current_pi[i_th] / len(current_sampled)).sqrt() * torch.cat([xstardotspectral.cos(), xstardotspectral.sin()],1).to(self.device)
                multiple_Phi_star.append(Phistar_i_th)
            return torch.cat(multiple_Phi, 1), torch.cat(multiple_Phi_star, 1)


    def compute_gram_matrix(self,batch_x):
        phi  = self._compute_sm_basis(batch_x)
        #print(phi.shape)
        return phi.matmul(phi.t())



    def compute_loss(self,batch_x,batch_y,num_batch=1):
        num_input = batch_x.size(0)
        loss = 0  # negative logmarginal likelihood

        if num_batch == 1:
            num_batch = self.num_batch
            #print('train_num_batch %d'%(num_batch))

        for j_th in range(num_batch):

            Phi = self._compute_sm_basis(batch_x)
            Approximate_gram = self._compute_gram_approximate(Phi)
            L = cholesky(Approximate_gram)
            Linv_PhiT = triangular_solve(Phi.t(), L ,upper=False)[0]           
            Linv_PhiT_y = Linv_PhiT.matmul(batch_y) 
            
            loss += (0.5 / self.likelihood.variance.transform()**2) * (batch_y.pow(2).sum() - Linv_PhiT_y.pow(2).sum())
            loss += lt_log_determinant(L)
            loss += (-self.total_num_sample)* (2* self.likelihood.variance)
            loss += 0.5 * num_input * (np.log(2*pi) + 2*self.likelihood.variance )
            

        weight_reg = self.lambda_w*self.weight.transform().pow(2).sum().sqrt()
        kl_term = self._kl_div_qp()

        return (1 / num_batch) * loss , kl_term + weight_reg




    def _kl_div_qp(self):
        q_dist = MVN(loc = self.mu.transform().view(1, -1).squeeze() ,
                     covariance_matrix = self.std.transform().view(1,-1).squeeze().pow(2).diag() )

        p_dist = MVN(loc = self.mu_prior.transform().view(1, -1).squeeze(),
                     covariance_matrix = self.std_prior.transform().view(1,-1).squeeze().pow(2).diag() )

        return  self.num_sample_pt*kl_divergence(q_dist, p_dist)


    
class ssgpr_rep_sm_reg_v2(ssgpr_rep_sm_reg):
    def __init__(self, num_batch, num_sample_pt, param_dict, kernel=None, likelihood=None, device=None):

        super(ssgpr_rep_sm_reg_v2, self).__init__(num_batch=num_batch,
                                                  num_sample_pt=num_sample_pt,
                                                  param_dict=param_dict,
                                                  kernel=kernel,
                                                  likelihood=likelihood,
                                                  device=device)

        print('initialization_beta_rrff')

        
    def kernel_SM(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        weight_, mu_, std_ = self.weight.transform().detach(), self.mu.transform().detach(), self.std.transform().detach()

        out = 0
        for ith in range(self.num_Q):
            x1_, x2_ = (2 * pi) * x1.mul(std_[ith]), (2 * pi) * x2.mul(std_[ith])
            sq_dist = -0.5 * (-2 * x1_.matmul(x2_.t()) + (x1_.pow(2).sum(-1, keepdim=True) + x2_.pow(2).sum(-1, keepdim=True).t()))
            x11_, x22_ = (2 * pi) * x1.matmul(mu_[ith].reshape(-1, 1)), (2 * pi) * x2.matmul(mu_[ith].reshape(-1, 1))

            exp_term = sq_dist.exp()
            cos_term = (x11_ - x22_.t()).cos()
            out += weight_[ith] * exp_term.mul(cos_term)

        return out.detach()



    def compute_loss(self, batch_x, batch_y, num_batch=1, approximation=True):
        mean_y = batch_y.mean(dim=0)
        batch_y = batch_y.clone() - mean_y

        num_input = batch_x.size(0)
        loss = 0  # negative logmarginal likelihood
        if approximation:
            if num_batch == 1:
                num_batch = self.num_batch

            for j_th in range(num_batch):
                Phi = self._compute_sm_basis(batch_x)
                #print('Phi.shape {}'.format(Phi.shape))                  
                Approximate_gram = self._compute_gram_approximate(Phi)                             
                L = cholesky(Approximate_gram)
                Linv_PhiT = triangular_solve(Phi.t(), L ,upper=False)[0]           
                Linv_PhiT_y = Linv_PhiT.matmul(batch_y) 

                loss += (0.5 / self.likelihood.variance.transform()**2) * (batch_y.pow(2).sum() - Linv_PhiT_y.pow(2).sum())
                loss += lt_log_determinant(L)
                loss += (-self.total_num_sample)* (2* self.likelihood.variance)
                loss += 0.5 * num_input * (np.log(2*pi) + 2*self.likelihood.variance )
            

            weight_reg = self.lambda_w * self.weight.transform().pow(2).sum().sqrt()
            kl_term = self._kl_div_qp()
            return (1 / num_batch) * loss, kl_term + weight_reg
            #return (1 / (num_batch*num_input)) * loss, kl_term + weight_reg

        else:
            #print(batch_x)
            K_xx = self.kernel_SM(batch_x) + (self.likelihood.variance.transform() + self.zitter)*torch.eye(batch_x.shape[0]).to(self.device)
            L=cholesky(K_xx)
            alpha = triangular_solve( batch_y, L ,upper=False)[0]                         
            loss = 0.5 * alpha.pow(2).sum(dim = 0) + lt_log_determinant(L) + 0.5 * num_input * np.log(2.* pi)

            return loss, 0.0


    def _get_log_prob(self, batch_x, batch_y ,num_batch , test_option) :
        num_data = batch_x.shape[0]
        if test_option is False:
            emission_prob,kl_term = self.compute_loss(batch_x, batch_y, num_batch, approximation = True)
            return -emission_prob.detach().clone(),kl_term
            #return -emission_prob.detach().clone()/num_data,kl_term
        
        else:
            emission_prob,kl_term = self.compute_loss(batch_x, batch_y, num_batch=1, approximation= False)
            return -emission_prob.detach().clone(),kl_term
            #return -emission_prob.detach().clone()/num_data,kl_term













if __name__ == "__main__":
    #1d inputs

    device = True
    Fs = 200.
    x = np.arange(0,5,1/Fs).reshape(-1,1)
    print(x)


    y = 10*np.sin(2*np.pi*x) + 0.1*np.random.randn(x.shape[0],1)
    y -= y.mean()
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    weight = np.array([5.,1.]).reshape(-1,1)
    mu = 1*np.array([0.5, 1.0]).reshape(-1,1)
    std = 0.001*np.random.rand(2, 1).reshape(-1,1)




    noise_variance = 1.0
    param_dict = {}
    param_dict['noise_variance'] = 1.0
    param_dict['noise_err'] = 0.1
    param_dict['weight'] = weight
    param_dict['mean'] = mu
    param_dict['std'] = std
    param_dict['lr_hyp'] = 0.1
    num_sample_pt = 5
    num_batch = 1


    model = ssgpr_rep_sm_reg(num_batch = num_batch,
                              num_sample_pt = num_sample_pt,
                              param_dict=param_dict)

    model2 = ssgpr_rep_sm_reg_v2(num_batch = num_batch,
                                 num_sample_pt = num_sample_pt,
                                 param_dict=param_dict)


    #print(model.std_)
    #print(model.compute_gram_matrix(x)-model2.compute_gram_matrix(x))
    print('')
    print()

    print(model.compute_loss(x,y,num_batch=1))
    print(model2.compute_loss(x,y,num_batch=1,approximation=True))
    print('')
    print(model2.compute_loss(x,y,num_batch=1,approximation=False))
