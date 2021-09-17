import numpy as np
import torch
from torch.nn import Module, Parameter



from utility.eval_metric import _measure_metric, accuracy
import time
import random
from scipy.special import digamma, gammaln, logsumexp


#torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_tensor_type(torch.DoubleTensor)


def _transform_batchset(x, y, device):
    x_list = []
    y_list = []
    for ith,(ith_x_train,ith_y_train) in enumerate(zip(x,y)):
        x_list.append(torch.from_numpy(ith_x_train).view(-1, 1).to(device))
        y_list.append(torch.from_numpy(ith_y_train).view(-1, 1).to(device))

    return x_list, y_list



class HMM_EmissionGP(Module):
    def __init__(self, emission_model_list=None, param_dict=None):
        super(HMM_EmissionGP, self).__init__()

        self.emission_model_list = emission_model_list
        self.num_hidden_state = len(emission_model_list)
        self.lr_A = param_dict['lr_A']
        self.lr_pi = param_dict['lr_pi']
        self.lr_emission_hyp = param_dict['lr_hyp']
        self.num_batch = param_dict['Num_Batch']
        self.batch_length = param_dict['Len_Batch']
        self.full_length = param_dict['Len_Full']
        self.iter_train = param_dict['Iter_train']
        self.iter_hyp = param_dict['Iter_hyp']
        self.num_k_emission = param_dict['Num_K_Emission']

        self.device = torch.device("cuda") if param_dict['device'] else torch.device('cpu')
        self.corrupted_obs = False
        self.eps = 1e-8
        self.gamma_thres = 0.1
        self._init_param()

        
        self.emission_name = param_dict['emission'] 
        


    def _init_param(self):
        self.prior_A = np.ones([1,self.num_hidden_state])
        self.prior_pi = np.ones([self.num_hidden_state])
        self.var_param_A = self.prior_A.repeat(self.num_hidden_state,axis = 0)
        self.var_param_pi = self.prior_pi

        self.sample_factor_A = 1
        self.sample_factor_pi = 1
        self.sample_factor_hyp = 1

        return



    def _get_param_list(self):
        weight_list = []
        mu_list = []
        std_list = []
        for ith_emission in self.emission_model_list:

            if hasattr(ith_emission,'kernel') and ith_emission.kernel != None:
                #print('GP SM emission')
                if hasattr(ith_emission.kernel,'weight'):
                    weight_list.append(ith_emission.kernel.weight.transform().cpu().data.numpy())
                    mu_list.append(ith_emission.kernel.mu.transform().cpu().data.numpy())
                    std_list.append(ith_emission.kernel.std.transform().cpu().data.numpy())

            else:
                #print('GP RRFF emission')
                weight_list.append(ith_emission.weight.transform().cpu().data.numpy())
                mu_list.append(ith_emission.mu.transform().cpu().data.numpy())
                std_list.append(ith_emission.std.transform().cpu().data.numpy())


        return self.var_param_A,self.var_param_pi,weight_list,mu_list,std_list



    def _get_elbo(self,log_batch_obs,log_alpha):
        elbo = 0.

        # Initial distribution (only if more than one series, so ignore for now)
        p_pi = self.prior_pi
        p_pisum = np.sum(p_pi)
        q_pi = self.var_param_pi
        q_pidg = digamma(q_pi + self.eps)
        q_pisum = np.sum(q_pi)
        dg_q_pisum = digamma(q_pisum + self.eps)

        # Energy
        pi_energy = (gammaln(p_pisum + self.eps) - np.sum(gammaln(p_pi + self.eps))
                     + np.sum((p_pi-1.)*(q_pidg - dg_q_pisum)))
        # Entropy
        pi_entropy = -(gammaln(q_pisum + self.eps) - np.sum(gammaln(q_pi + self.eps))
                       + np.sum((q_pi-1.)*(q_pidg - dg_q_pisum)))

        # Transition matrix (each row is Dirichlet so can do like above)
        p_A = self.prior_A
        p_Asum = np.sum(p_A, axis=1)
        q_A = self.var_param_A
        q_Adg = digamma(q_A + self.eps)
        q_Asum = np.sum(q_A, axis=1,keepdims= True)
        dg_q_Asum = digamma(q_Asum +self.eps)

        A_energy = (gammaln(p_Asum + self.eps) - np.sum(gammaln(p_A + self.eps), axis=1)
                    + np.sum((p_A-1)*(q_Adg - dg_q_Asum), axis=1))
        A_entropy = -(gammaln(q_Asum + self.eps) - np.sum(gammaln(q_A + self.eps), axis=1)
                     + np.sum((q_A-1)*(q_Adg - dg_q_Asum), axis=1))
        A_energy = np.sum(A_energy)
        A_entropy = np.sum(A_entropy)


        # Emission distributions -- does both energy and entropy
        emit_vlb = log_batch_obs.sum().cpu().data.numpy()

        # We don't need the minus anymore b/c this is 1/ctable
        lZ = np.sum(np.logaddexp.reduce(log_alpha, axis=1))

        elbo = (pi_energy + pi_entropy + A_energy + A_entropy
                + emit_vlb + lZ)


        return elbo




    def _run_smoothing(self, batch_x, batch_y, num_test_batch):
        batch_x, batch_y = _transform_batchset(batch_x, batch_y, self.device)
        with torch.no_grad():
            log_batch_obs_prob,kl_term = self._calc_obs_prob(batch_x, batch_y, num_test_batch)
            log_A_star = digamma(self.var_param_A + self.eps) - digamma(self.var_param_A.sum(axis=1, keepdims=True) + self.eps)
            log_pi_star = digamma(self.var_param_pi + self.eps) - digamma(self.var_param_pi.sum() + self.eps)
            gamma,log_alpha,log_beta = self._forward_backward(log_batch_obs_prob, log_pi_star, log_A_star)
            elbo = self._get_elbo(log_batch_obs_prob,log_alpha)
            
        loglik = logsumexp(log_alpha,axis=1)
        #return gamma.argmax(axis = 1), elbo , np.log(logsumexp(np.exp(log_alpha[-1]))+1e-8)
        return gamma.argmax(axis = 1), elbo , loglik.sum()



    def _calc_obs_prob(self, batch_x_list, batch_y_list, num_batch=1 , test_option = False):
        batch_log_emission_prob = torch.from_numpy(np.zeros([len(batch_x_list),self.num_hidden_state])).to(self.device)
        kl_term = torch.from_numpy(np.zeros([len(batch_x_list),self.num_hidden_state])).to(self.device)

        for i_th, (i_th_batch_x, i_th_batch_y) in enumerate(zip(batch_x_list, batch_y_list)):
            for j_th, j_th_emission_model in enumerate(self.emission_model_list):

                batch_log_emission_prob[i_th,j_th], kl_term[i_th,j_th] = j_th_emission_model._get_log_prob(i_th_batch_x,
                                                                                                           i_th_batch_y,
                                                                                                           num_batch,
                                                                                                           test_option)
        return batch_log_emission_prob,kl_term




    def _log_forward(self, log_batch_obs_prob, log_pi_star, log_A_star):
        log_batch_obs_prob = log_batch_obs_prob.cpu().data.numpy()
        log_alpha = np.zeros([log_batch_obs_prob.shape[0] , self.num_hidden_state])
        log_alpha[0] = log_batch_obs_prob[0] + log_pi_star
        for i_th in range(1, log_batch_obs_prob.shape[0]):
            log_alpha[i_th] = np.logaddexp.reduce(log_alpha[i_th-1] + log_A_star.transpose(), axis=1) + log_batch_obs_prob[i_th]

        return log_alpha


    def _log_backward(self, log_batch_obs_prob, log_A_star):
        log_batch_obs_prob = log_batch_obs_prob.cpu().data.numpy()
        log_beta = np.zeros([log_batch_obs_prob.shape[0], self.num_hidden_state])
        log_beta[-1] = np.zeros([self.num_hidden_state])
        for i_th in reversed(range(log_batch_obs_prob.shape[0] - 1)):
            log_beta[i_th] = np.logaddexp.reduce(log_A_star + log_beta[i_th + 1] + log_batch_obs_prob[i_th + 1], axis=1 )

        return log_beta


    def _forward_backward(self, log_batch_obs_prob, log_pi_star, log_A_star):
        #_, batch_seq_length = log_batch_obs_prob.shape
        log_alpha = self._log_forward(log_batch_obs_prob, log_pi_star, log_A_star)
        log_beta = self._log_backward(log_batch_obs_prob,log_A_star)
        log_gamma = log_alpha + log_beta
        log_gamma -= log_gamma.max(axis = 1,keepdims = True)
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis = 1,keepdims = True)
        return gamma,log_alpha,log_beta


    def _run_Estep(self, batch_x, batch_y):
        log_batch_obs_prob,kl_term = self._calc_obs_prob(batch_x, batch_y)

        log_A_star = digamma(self.var_param_A + self.eps) - digamma(self.var_param_A.sum(axis=1,keepdims= True) + self.eps)
        log_pi_star = digamma(self.var_param_pi + self.eps) - digamma(self.var_param_pi.sum() + self.eps)
        gamma,log_alpha,log_beta = self._forward_backward(log_batch_obs_prob, log_pi_star, log_A_star)

        # print('_run_Estep_gamma')
        # print(gamma)
        return gamma,  log_batch_obs_prob, kl_term


    def _run_Mstep(self, gamma, log_batch_obs, kl_term, decay_option):
        self._update_var_param_A(gamma , option = 'NotNat')
        self._update_var_param_pi(gamma , option = 'NotNat' )
        #print('before_hypparam')
        self._update_hyp_param_emission_model(log_batch_obs, gamma , kl_term , decay_option)
        #print('after_hypparam')

        return


    def _update_var_param_pi(self, gamma , option ):
        if option  == 'Nat':
            nats_old = self.var_param_pi - 1
            nats_t = self.prior_A + gamma[0] - 1
            nat_new = (1 - self.lr_pi)*nats_old + self.sample_factor_pi*self.lr_pi*nats_t
            self.var_param_pi = nat_new +1
            return
        else:
            self.var_param_pi =  self.prior_pi + self.sample_factor_pi*self.var_param_A[0]
            return



    def _update_var_param_A(self, gamma , option ):
        nats_old = self.var_param_A - 1
        tran_mf = self.prior_A.repeat(self.num_hidden_state,axis = 0)
        for t in range(1, self.batch_length):
            tran_mf += np.outer(gamma[t-1], gamma[t])

        if option  == 'Nat':
            nats_t = (tran_mf - 1.)
            # Perform update according to stochastic gradient (Hoffman, pg. 17)
            nats_new = (1 - self.lr_A)*nats_old + self.sample_factor_A*self.lr_A*nats_t
            self.var_param_A = nats_new + 1.
            return
        else:
            self.var_param_A = tran_mf
            return


    def _update_hyp_param_emission_model(self, log_batch_obs, gamma , kl_term ,decay_option):
        gamma_tensor = torch.from_numpy(gamma).to(self.device)
        gamma_tensor = gamma_tensor.masked_fill(gamma_tensor < self.gamma_thres, 0)  # strong assign
        # negative marginal likelihood minimize
        loss = -(log_batch_obs.mul(gamma_tensor)).sum(dim=0)  + kl_term[-1] * gamma_tensor.shape[0]

        for ith_loss,ith_emission in zip(loss,self.emission_model_list):
            for ij in range(self.iter_hyp):
                ith_emission.optimizer.zero_grad()
                ith_loss.backward(retain_graph=True)
                ith_emission.optimizer.step()
            if decay_option :
               ith_emission.lr_scheduler.step()
            # restrict the noise vairance (clamping)
            ith_emission.likelihood.bound_variance()
                

        torch.cuda.empty_cache()
        return



    def train(self, x_train, y_train, z_train,x_test,y_test,z_test ):

        log_lik_list = []
        train_accuracy_list = []
        test_accuracy_list = []
        time_list = []
        num_cluster_list = []
        ith_var_param_A_list = []
        ith_var_param_pi_list = []
        ith_mu_list = []
        ith_std_list = []
        ith_weight_list = []


        batch_x,batch_y = _transform_batchset(x_train, y_train, self.device)

        #ith_var_param_A,ith_var_param_pi,ith_weight,ith_mu,ith_std = self._get_param_list()
        for i in range(self.iter_train):
            t0 = time.time()

            self.lr_A = 1/np.power(5+i,.5)
            self.lr_pi = 1/np.power(5+i,.5)
            gamma, log_batch_obs , kl_term = self._run_Estep(batch_x, batch_y)
            self._run_Mstep(gamma, log_batch_obs, kl_term , decay_option = True)
            t1 = time.time()

            torch.cuda.empty_cache()
            log_batch_obs.detach()

            z_train_pred, train_elbo, train_lik = self._run_smoothing(x_train,y_train,num_test_batch=1)
            z_test_pred, test_elbo, test_lik = self._run_smoothing(x_test, y_test,num_test_batch=1 )

            train_acc = accuracy(z_train, z_train_pred)
            test_acc = accuracy(z_test, z_test_pred)
            
            log_lik_list.append((train_elbo,train_lik,test_elbo,test_lik))
            train_accuracy_list.append(train_acc)
            test_accuracy_list.append(test_acc)
            time_list.append(t1 - t0)
            num_cluster_list.append( len(np.unique(z_train_pred)))

            

            
            ###################################################################################
            
            print('-'*100)
            #print('iter %d, train acc : %.3f \t test acc : %.3f \t iteration time : %.3f '%(i + 1, train_accuracy ,test_acc, (t1 - t0)) )
            print('iter {:d}, iteration time : {:.3f}| train acc : {:.3f}, train lik : {:.3f}, \t test acc : {:.3f}, test lik : {:.3f} '.format(i + 1, t1 - t0, train_acc,train_lik ,test_acc,test_lik) )

            print('-'*100)
            

            ith_var_param_A, ith_var_param_pi, ith_weight, ith_mu, ith_std = self._get_param_list()
            ith_var_param_A_list.append(ith_var_param_A)
            ith_var_param_pi_list.append(ith_var_param_pi)
            ith_weight_list.append(ith_weight)
            ith_mu_list.append(ith_mu)
            ith_std_list.append(ith_std)

        param_history_dict = {}
        param_history_dict['weight'] = ith_weight_list
        param_history_dict['mu'] = ith_mu_list
        param_history_dict['std'] = ith_std_list
        param_history_dict['var_A'] = ith_var_param_A_list
        param_history_dict['var_pi'] = ith_var_param_pi_list

        
        return np.asarray(log_lik_list),np.asarray(train_accuracy_list),np.asarray(test_accuracy_list),np.asarray(time_list), param_history_dict ,np.array(zpred_list)     
    
    
        #return np.asarray(log_lik_list),np.asarray(train_accuracy_list),np.asarray(test_accuracy_list),np.asarray(time_list),np.asarray(num_cluster_list) , param_history_dict
    
    
#         return np.asarray(log_lik_list),np.asarray(accuracy_list),np.asarray(test_accuracy_list), np.asarray(test_exact_accuracy_list),\
#                np.asarray(time_list), np.asarray(num_cluster_list),\
#                np.asarray(num_test_cluster_list),np.asarray(num_test_exact_cluster_list), param_history_dict


        
        
        
        
        
        


class SVI_HMM_EmissionGP(HMM_EmissionGP):
    def __init__(self, emission_model_list=None, param_dict=None):
        super(SVI_HMM_EmissionGP, self).__init__(emission_model_list=emission_model_list,param_dict=param_dict)
        self._init_param()
        self.emission_name = param_dict['emission'] 




    def _init_param(self):
        self.prior_A = np.ones([1,self.num_hidden_state])
        self.prior_pi = np.ones([self.num_hidden_state])
        self.var_param_A = self.prior_A.repeat(self.num_hidden_state,axis = 0)
        self.var_param_pi = self.prior_pi
        self.sample_factor_pi = 1/self.num_batch
        self.sample_factor_A = (self.full_length - self.batch_length + 1) / (self.batch_length * self.num_batch)
        self.sample_factor_hyp = (self.full_length - self.batch_length + 1) / (self.batch_length * self.num_batch)

        return



    def _calc_obs_prob(self, batch_x_list, batch_y_list, num_batch=1,test_option=False):
        #print('hihi in _calc_obs_prob_svi')

        batch_log_emission_prob = torch.from_numpy(np.zeros([len(batch_x_list),self.num_hidden_state])).to(self.device)
        kl_term = torch.from_numpy(np.zeros([len(batch_x_list),self.num_hidden_state])).to(self.device)

        for i_th, (i_th_batch_x, i_th_batch_y) in enumerate(zip(batch_x_list, batch_y_list)):
            for j_th, j_th_emission_model in enumerate(self.emission_model_list):
                batch_log_emission_prob[i_th,j_th], kl_term[i_th,j_th] = j_th_emission_model._get_log_prob(i_th_batch_x,
                                                                                                           i_th_batch_y,
                                                                                                           num_batch,
                                                                                                           test_option)
        return batch_log_emission_prob,kl_term



    def _run_Estep(self, batch_x, batch_y):
        #print('run_Estep')
        log_batch_obs_prob,kl_term = self._calc_obs_prob(batch_x, batch_y, num_batch=1 ,test_option=False)
        log_A_star = digamma(self.var_param_A + self.eps) - digamma(self.var_param_A.sum(axis=1,keepdims= True) + self.eps)
        log_pi_star = digamma(self.var_param_pi + self.eps) - digamma(self.var_param_pi.sum() + self.eps)
        gamma,log_alpha,log_beta = self._forward_backward(log_batch_obs_prob, log_pi_star, log_A_star)
        return gamma,  log_batch_obs_prob, kl_term



    def _run_smoothing(self, batch_x, batch_y, num_test_batch,test_option = False):
        batch_x, batch_y = _transform_batchset(batch_x, batch_y, self.device)
        with torch.no_grad():
            log_batch_obs_prob,kl_term = self._calc_obs_prob(batch_x, batch_y, num_test_batch, test_option)
            log_A_star = digamma(self.var_param_A + self.eps) - digamma(self.var_param_A.sum(axis=1, keepdims=True) + self.eps)
            log_pi_star = digamma(self.var_param_pi + self.eps) - digamma(self.var_param_pi.sum() + self.eps)
            gamma,log_alpha,log_beta = self._forward_backward(log_batch_obs_prob, log_pi_star, log_A_star)
            elbo = self._get_elbo(log_batch_obs_prob,log_alpha)
            
            
        #loglik = np.log(np.exp(log_alpha).sum(axis=1,keepdim=True)+1e-8).sum()
        loglik = logsumexp(log_alpha,axis=1)
        #print('loglik {}, logliksum {}'.format(loglik, loglik.sum()))
            
        return gamma.argmax(axis = 1), elbo , loglik.sum() 





    def _update_var_param_pi_batch(self,gamma_list):
        for ith_gamma in gamma_list:
            self._update_var_param_pi(ith_gamma,option = 'Nat')


    def _update_var_param_A_batch(self,gamma_list):
        for ith_gamma in gamma_list:
            self._update_var_param_A(ith_gamma,option = 'Nat')





    def _update_hyp_param_emission_model_batch(self, gamma_list , log_batch_obs_list, kl_term_list ,decay_option):
        #loss = torch.from_numpy(np.zeros(self.num_hidden_state)).to(self.device)
        loss = 0.0
        for ith_gamma, ith_log_batch_obs, ith_kl_term in zip( gamma_list , log_batch_obs_list, kl_term_list):
            gamma_tensor = torch.from_numpy(ith_gamma).to(self.device)
            gamma_tensor = gamma_tensor.masked_fill(gamma_tensor < self.gamma_thres, 0)  # strong assign

            loss += -self.sample_factor_hyp*(ith_log_batch_obs.mul(gamma_tensor)).sum(dim=0)
            loss += gamma_tensor.shape[0]*ith_kl_term[-1]


            # loss += -self.sample_factor_hyp*(ith_log_batch_obs.mul(gamma_tensor)).sum(dim=0).clone()
            # loss += gamma_tensor.shape[0]*ith_kl_term[-1].clone()



        for ith_loss,ith_emission in zip(loss,self.emission_model_list):
            for ij in range(self.iter_hyp):
                ith_emission.optimizer.zero_grad()
                ith_loss.backward(retain_graph=True)
                ith_emission.optimizer.step()
            if decay_option:    
                ith_emission.lr_scheduler.step()
            # restrict the noise vairance (clamping)
            ith_emission.likelihood.bound_variance()
                                

        torch.cuda.empty_cache()
        return


    def _run_Mstep(self, gamma_list, log_batch_obs_list,kl_term_list , decay_option ):
        self._update_var_param_pi_batch(gamma_list)
        self._update_var_param_A_batch(gamma_list)
        self._update_hyp_param_emission_model_batch(gamma_list , log_batch_obs_list ,kl_term_list ,decay_option)
        return



    def train(self,x_train,y_train,z_train,x_test = None,y_test = None,z_test = None):

        log_lik_list = []
        train_accuracy_list = []
        test_accuracy_list = []
        test_exact_accuracy_list = []
        time_list = []
        num_cluster_list = []
        num_test_cluster_list = []
        num_test_exact_cluster_list = []
        ith_var_param_A_list = []
        ith_var_param_pi_list = []
        ith_mu_list = []
        ith_std_list = []
        ith_weight_list = []

        
        zpred_list = []

        for i in range(1,self.iter_train+1):
            ####################################################################################
            # train SVI
            ####################################################################################                        
            t0 = time.time()            
            self.lr_A = 1/np.power(5+i,.5)
            self.lr_pi = 1/np.power(5+i,.5)            
                        
            gamma_list = []
            log_batch_obs_list = []
            kl_term_list = []                        

            batch_x, batch_y = _transform_batchset(x_train, y_train, device = self.device)
            for j in range(self.num_batch):
                rand_idx = random.choice(np.arange(self.full_length - self.batch_length))
                gamma, log_batch_obs, kl_term = self._run_Estep(batch_x[rand_idx: rand_idx + self.batch_length],
                                                                batch_y[rand_idx: rand_idx + self.batch_length])

                gamma_list.append(gamma)
                log_batch_obs_list.append(log_batch_obs)
                kl_term_list.append(kl_term)

            self._run_Mstep(gamma_list, log_batch_obs_list, kl_term_list , decay_option = True)
            t1 = time.time()

            
            
            
            ####################################################################################
            # evaluation 
            ####################################################################################
            #if (i % int(self.iter_train/(self.iter_train/4)) == 0) or i == 1:                
            #if (i % int(self.iter_train) == 0) or i == 1:                
            if (i % 2 == 0) or i == 1:                
                
                z_train_pred, train_elbo, train_lik  = self._run_smoothing(x_train,y_train,num_test_batch=self.num_k_emission,test_option=False)
                z_test_pred, test_elbo, test_lik = self._run_smoothing(x_test, y_test, num_test_batch=self.num_k_emission, test_option=False)            
                train_acc = accuracy(z_train, z_train_pred)
                test_acc = accuracy(z_test, z_test_pred)
                
                #log_lik_list.append((train_elbo,train_lik,test_elbo,test_lik))            
                #train_accuracy_list.append(train_acc)
                #test_accuracy_list.append(test_acc)
                time_list.append(t1 - t0)
                num_cluster_list.append( len(np.unique(z_train_pred)))
                num_test_cluster_list.append( len(np.unique(z_test_pred)))            
                
                if self.emission_name == 'gpsm':
                    z_train_pred_e, train_elbo_e, train_lik_e = z_train_pred, train_elbo, train_lik
                    z_test_pred_e, test_elbo_e, test_lik_e = z_test_pred, test_elbo, test_lik
                else:
                    z_train_pred_e, train_elbo_e, train_lik_e  = self._run_smoothing(x_train,y_train,num_test_batch=self.num_k_emission,test_option=True)
                    z_test_pred_e, test_elbo_e, test_lik_e = self._run_smoothing(x_test, y_test, num_test_batch=self.num_k_emission, test_option=True) 
                    
                    #cwru dataset
                    #z_train_pred_e, train_elbo_e, train_lik_e = z_train_pred, train_elbo, train_lik
                    #z_test_pred_e, test_elbo_e, test_lik_e = z_test_pred, test_elbo, test_lik

                train_acc_e = accuracy(z_train, z_train_pred_e)
                test_acc_e = accuracy(z_test, z_test_pred_e)


                print('-'*200)
                print('iter {:d}, iteration time : {:.3f}| train acc : {:.3f}, train lik : {:.3f}, \t test acc : {:.3f}, test lik : {:.3f} |e train acc : {:.3f}, e train lik : {:.3f} e test acc : {:.3f}, e test lik : {:.3f} '.format(i , t1 - t0, train_acc,train_lik ,test_acc,test_lik, train_acc_e, train_lik_e,test_acc_e, test_lik_e) )
                print('-'*200)
                
                train_accuracy_list.append((train_acc,train_acc_e))
                test_accuracy_list.append((test_acc,test_acc_e))                
                log_lik_list.append((train_elbo,train_lik,test_elbo,test_lik,train_lik_e,test_lik_e))
                zpred_list.append( (z_train_pred,z_test_pred) )
                

            ith_var_param_A, ith_var_param_pi, ith_weight, ith_mu, ith_std = self._get_param_list()
            ith_var_param_A_list.append(ith_var_param_A)
            ith_var_param_pi_list.append(ith_var_param_pi)
            ith_weight_list.append(ith_weight)
            ith_mu_list.append(ith_mu)
            ith_std_list.append(ith_std)

        param_history_dict = {}
        param_history_dict['weight'] = ith_weight_list
        param_history_dict['mu'] = ith_mu_list
        param_history_dict['std'] = ith_std_list
        param_history_dict['var_A'] = ith_var_param_A_list
        param_history_dict['var_pi'] = ith_var_param_pi_list



        return np.asarray(log_lik_list),np.asarray(train_accuracy_list),np.asarray(test_accuracy_list),\
               np.asarray(time_list), param_history_dict ,np.array(zpred_list)
        
        
#         return np.asarray(log_lik_list),np.asarray(train_accuracy_list),np.asarray(test_accuracy_list),\
#                np.asarray(time_list),np.asarray(num_cluster_list), param_history_dict ,np.array(zpred_list)


        #np.asarray(log_lik_list) : (train_elbo,train_li,test_elbo_test_lik)
#         return np.asarray(log_lik_list),np.asarray(train_accuracy_list),np.asarray(test_accuracy_list), np.asarray(test_exact_accuracy_list),\
#                np.asarray(time_list), np.asarray(num_cluster_list),\
#                np.asarray(num_test_cluster_list),np.asarray(num_test_exact_cluster_list), param_history_dict






if __name__ == '__main__':

    a1 = np.random.randn(2)
    a2 = np.random.randn(2)


    print(a1,a2)
    print(np.outer(a1,a2))





