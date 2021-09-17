from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import signal
import numpy as np
import torch

from kernels.RBF_kernel import RBF
from kernels.Periodic_kernel import Periodic
from kernels.SM_kernel import SM
from kernels.kernel import Sum_kernel

from models.emission_gp import gpr
from models.emission_gp_rrff import ssgpr_rep_sm_reg, ssgpr_rep_sm_reg_v2
from models.hmm_models_v4  import HMM_EmissionGP,SVI_HMM_EmissionGP



torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float32)


def _inverse_sampling_given_pdf(energies,empirical_pdf,sample_num):
    cum_prob = np.cumsum(empirical_pdf)
    R = np.random.uniform(0, 1, sample_num)
    gen_energies = [float(energies[np.argwhere(cum_prob == min(cum_prob[(cum_prob - r) > 0]))]) for r in R]
    return np.asarray(gen_energies).reshape(1,-1)


def _initialize_SMkernel_hyp(x_train, y_train, setting_dict, random_seed):
    np.random.seed(random_seed + 1000)
    thres_hold = 0.0
    psd_list = []
    for i_th, (i_th_x, i_th_y) in enumerate(zip(x_train, y_train)):
        #Fs = len(i_th_y)
        Fs = int(1/(i_th_x[1] - i_th_x[0]))
        
        freqs, psd = signal.welch(i_th_y.reshape(1, -1).squeeze(), fs=Fs, nperseg=len(i_th_y))
        psd_list.append(psd / psd.sum())

    Num_Q = setting_dict['Num_Q']
    Num_Hidden_state = setting_dict['Num_HiddenState']
    kmeans = KMeans(n_clusters=Num_Hidden_state, random_state=0).fit(psd_list)



    SMKernel_hyp_list = []
    psd_sample_list = []

    sample_num = 10000 #synthetic
    for i_th,i_th_empirical_pdf in enumerate(kmeans.cluster_centers_):
        SMKernel_hyp = {}
        psd_sample = _inverse_sampling_given_pdf(freqs, i_th_empirical_pdf, sample_num)
        gmm = GaussianMixture(n_components=Num_Q, covariance_type='diag').fit(np.asarray(psd_sample).reshape(-1, 1))
        idx_thres = np.where(gmm.weights_ >= thres_hold)[0]

        SMKernel_hyp['weight'] = gmm.weights_[idx_thres].reshape(-1, 1)
        SMKernel_hyp['mean'] = gmm.means_[idx_thres].reshape(-1, setting_dict['input_dim'])
        SMKernel_hyp['std'] = np.sqrt(gmm.covariances_[idx_thres].reshape(-1, setting_dict['input_dim']))
        SMKernel_hyp['variance'] = 1.0
        SMKernel_hyp['length_scales'] = np.random.rand()
        SMKernel_hyp['noise_variance'] = 1.0  #real v2


        chosen_idx = np.where(kmeans.labels_ == i_th)[0][0]
        SMKernel_hyp['noise_err'] = .1 * np.std(y_train[chosen_idx])
        SMKernel_hyp_list.append(SMKernel_hyp)
        psd_sample_list.append(np.asarray(psd_sample).reshape(-1, 1))



    return SMKernel_hyp_list ,  kmeans , psd_sample_list






def _make_gp_emission(model_name,param_dict,model_setting_dict,device):
    input_dim = 1
    #cuda_option = True

    if model_name == 'gprbf':
        Kern = RBF(variance=param_dict['variance'],
                   length_scale=param_dict['length_scales'],
                   device=device)

        lr_hyp = model_setting_dict['lr_hyp']
        model_gpr = gpr(Kern,
                        likelihood=None,
                        device = device,
                        lr_hyp=lr_hyp,
                        noise_err=param_dict['noise_err'])
        return model_gpr


    if model_name == 'gpsm':
        Kern = SM(param_dict['weight'],
                  param_dict['mean'],
                  param_dict['std'],
                  device=device)
        lr_hyp = model_setting_dict['lr_hyp']
        model_gpr = gpr(Kern,
                        likelihood=None,
                        device = device,
                        lr_hyp=lr_hyp,
                        noise_err=param_dict['noise_err'])
        return model_gpr


    if model_name == 'gprrff':

        num_Q = len(param_dict['weight'])
        num_sample_pt = int(model_setting_dict['Num_RRFFSpectralPt_total']/num_Q)
        num_batch = model_setting_dict['Num_RRFFBatch']
        param_dict['lr_hyp'] = model_setting_dict['lr_hyp']

        model_ssgpr_reg_sm = ssgpr_rep_sm_reg(num_batch=num_batch,
                                               num_sample_pt=num_sample_pt,
                                               param_dict=param_dict,
                                               kernel=None,
                                               likelihood=None,
                                               device=device)

        return model_ssgpr_reg_sm


    if model_name == 'gprrff_beta':

        num_Q = len(param_dict['weight'])
        num_sample_pt = int(model_setting_dict['Num_RRFFSpectralPt_total']/num_Q)
        num_batch = model_setting_dict['Num_RRFFBatch']
        param_dict['lr_hyp'] = model_setting_dict['lr_hyp']

        model_ssgpr_reg_sm = ssgpr_rep_sm_reg_v2(num_batch=num_batch,
                                                num_sample_pt=num_sample_pt,
                                                param_dict=param_dict,
                                                kernel=None,
                                                likelihood=None,
                                                device=device)

        return model_ssgpr_reg_sm


    if model_name == 'benchmark_v1':
        variance = 1.0
        a0 = 5.
        b0 = 50.
        Q = 3
        kernel_list = []
        length_scales = 1.1*np.ones(Q)
        periodic = a0 + b0 * np.random.rand(Q)

        kernel_list.append(RBF(variance, length_scales[0], device))
        for ith in range(1,Q) :
            kernel_list.append(Periodic(variance, length_scales[ith], periodic[ith], device))

        Sumkernel = Sum_kernel(kernel_list, device)
        lr_hyp = model_setting_dict['lr_hyp']
        model_gpr = gpr(Sumkernel,
                        likelihood=None,
                        device = device,
                        lr_hyp=lr_hyp,
                        noise_err=param_dict['noise_err'])


        # length_scales = [1.0]
        # Kern1 = RBF(variance, length_scales, device)
        #
        # #length_scales = [1.0]
        # # periodic = [10.1]
        # # Kern1 = Periodic(variance, length_scales, periodic, device)
        #
        # length_scales = [1.0]
        # periodic = [50.1]
        # Kern2 = Periodic(variance, length_scales, periodic, device)
        #
        # length_scales = [1.0]
        # periodic = [100.1]
        # Kern3 = Periodic(variance, length_scales, periodic, device)

        # length_scales = [1.0]
        # periodic = [30.1]
        # Kern4 = Periodic(variance, length_scales, periodic, device)
        #
        return model_gpr


    if model_name == 'benchmark_v2':
        # variance = 1.0
        # length_scales = [1.0]
        # Kern1 = RBF(variance, length_scales, device)
        #
        # length_scales = [1.0]
        # # periodic = [10.1]
        # # Kern1 = Periodic(variance, length_scales, periodic, device)
        #
        # length_scales = [1.0]
        # periodic = [50.1]
        # Kern2 = Periodic(variance, length_scales, periodic, device)
        #
        # length_scales = [1.0]
        # periodic = [100.1]
        # Kern3 = Periodic(variance, length_scales, periodic, device)
        #
        # length_scales = [1.0]
        # periodic = [200.1]
        # Kern4 = Periodic(variance, length_scales, periodic, device)
        #
        # kernel_list = [Kern1,Kern2,Kern3,Kern4]
        # Sumkernel = Sum_kernel(kernel_list, device)
        # lr_hyp = model_setting_dict['lr_hyp']
        # model_gpr = gpr(Sumkernel,
        #                 likelihood=None,
        #                 device = device,
        #                 lr_hyp=lr_hyp,
        #                 noise_err=param_dict['noise_err'])
        # return model_gpr

        variance = 1.0
        a0 = 5.
        b0 = 50.
        Q = 4
        kernel_list = []
        length_scales = 1.1*np.ones(Q)
        periodic = a0 + b0 * np.random.rand(Q)

        #kernel_list.append(RBF(variance, length_scales[0], device))
        kernel_list.append(Periodic(variance, length_scales[0], periodic[0], device))
        for ith in range(1, Q):
            kernel_list.append(Periodic(variance, length_scales[ith], periodic[ith], device))

        Sumkernel = Sum_kernel(kernel_list, device)
        lr_hyp = model_setting_dict['lr_hyp']
        model_gpr = gpr(Sumkernel,
                        likelihood=None,
                        device=device,
                        lr_hyp=lr_hyp,
                        noise_err=param_dict['noise_err'])
        return model_gpr


    if model_name == 'benchmark_v3':
        variance = 1.0
        a0 = 5.
        b0 = 500.
        Q = 1
        kernel_list = []
        length_scales = 1.1*np.ones(Q)
        periodic = a0 + b0 * np.random.rand(Q)

        kernel_list.append(Periodic(variance, length_scales[0], periodic[0], device))
        # for ith in range(1, Q):
        #     kernel_list.append(Periodic(variance, length_scales[ith], periodic[ith], device))

        Sumkernel = Sum_kernel(kernel_list, device)
        lr_hyp = model_setting_dict['lr_hyp']
        model_gpr = gpr(Sumkernel,
                        likelihood=None,
                        device=device,
                        lr_hyp=lr_hyp,
                        noise_err=param_dict['noise_err'])
        return model_gpr

    return





def _construct_emission_model(model_setting_dict, param_dict_list):
    model_name = model_setting_dict['emission']
    cuda_option = model_setting_dict['device']

    emission_model_list = []
    for i_th_param_dict in param_dict_list:
        emission_model_list.append(_make_gp_emission(model_name, i_th_param_dict ,model_setting_dict ,cuda_option))
    return emission_model_list






def _combine_models(x_train, y_train , exp_setting , model_setting , random_seed , num_init_iter):
    SMKernel_hyp_list, kmeans, _ = _initialize_SMkernel_hyp(x_train, y_train, exp_setting , random_seed)



    param_dict_list = SMKernel_hyp_list
    emission_model_list = _construct_emission_model(model_setting,param_dict_list)

    if model_setting['train'] == 'VBEM':
        model = HMM_EmissionGP(emission_model_list=emission_model_list,
                               param_dict = model_setting)

    elif model_setting['train'] == 'SVI':
        model = SVI_HMM_EmissionGP(emission_model_list=emission_model_list,
                                   param_dict = model_setting)

    else :
        raise ValueError

    for ith in range(kmeans.n_clusters):
        print(np.where(kmeans.labels_ == ith)[0][0])
        batch_x  = torch.from_numpy(np.asarray(x_train[ith]).reshape(-1,1)).to(model.device)
        batch_y =  torch.from_numpy(np.asarray(y_train[ith]).reshape(-1,1)).to(model.device)

        for jth in range(num_init_iter):
            ith_loss = model.emission_model_list[ith].compute_loss(batch_x, batch_y)[0]

            model.emission_model_list[ith].optimizer.zero_grad()
            ith_loss.backward(retain_graph=True)
            model.emission_model_list[ith].optimizer.step()

            if jth % 50 ==0:
                print('%d emission, %d iter, loss %.4f '%(ith,jth,ith_loss.cpu().data.numpy() ) )
        print('')

    return  model




# if __name__ ==  "__main__":
#     print('hi')