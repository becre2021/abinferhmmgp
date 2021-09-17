from utility.dataset import _process_Synthetic
from models_utility.construct_models import _combine_models
from utility.eval_metric import _measure_metric, accuracy, compute_bic

import os
import torch
import pickle
import numpy as np
import argparse



def make_expinfo(args):
    args_dict = args.__dict__
    model_info = ''
    for ith_keys in args_dict:
        model_info += ith_keys + str(args_dict[ith_keys]) + '_'
    return model_info


parser = argparse.ArgumentParser(description='data_file_load')
parser.add_argument('--numexp', type=int, default=5)
parser.add_argument('--init_iteration', type=int, default=501)
parser.add_argument('--iteration', type=int, default=10)
parser.add_argument('--lrhyp', type=float, default=0.0005)
parser.add_argument('--numhidden', type=int, default=8)
parser.add_argument('--numQ', type=int, default=3)
parser.add_argument('--savepicklepath',type=str,default='./')
parser.add_argument('--emission',type=str,default='gprrff_beta')
#parser.add_argument('--',type=str,default='gprrff_beta')

args = parser.parse_args()
expinfo = make_expinfo(args)

##############################################################################
# load dataset
##############################################################################

random_seed = 1000
# random_seed = 1001


file_path = './dataset/synthetic/'
file_name = 'Q6_Fs1000'

data_file_path = file_path + file_name + '.mat'

###############################################################################################
# experiment_setting
###############################################################################################

num_exp = args.numexp
exp_setting = {}
exp_setting['num_rep_exp'] = num_exp
exp_setting['data'] = file_name

# exp_setting['full_length'] = 50
# exp_setting['full_length'] = 75
exp_setting['full_length'] = 100

exp_setting['input_dim'] = 1
exp_setting['Num_Q'] = 1
exp_setting['Num_HiddenState'] = args.numhidden
exp_setting['device'] = False


x_train, y_train, z_train, x_test, y_test, z_test, true_freq, true_weight, true_noise_level, true_num_state = _process_Synthetic(data_file_path, exp_setting)
exp_setting['input_length'] = x_train.shape[1]
###############################################
###############################################

HMMGP_setting = {}
# HMMGP_setting['emission'] = 'gpsm' #
# HMMGP_setting['emission'] = 'gprrff' #

#HMMGP_setting['emission'] = 'gprrff_beta'  #
HMMGP_setting['emission'] = args.emission  #

# HMMGP_setting['emission'] = 'benchmark_v2'  #Q=4
# HMMGP_setting['emission'] = 'benchmark_v3'  #Q=4


# HMMGP_setting['train'] = 'VBEM' #
HMMGP_setting['train'] = 'SVI'  #
HMMGP_setting['Num_Q'] = exp_setting['Num_Q']

HMMGP_setting['lr_A'] = 0.05
HMMGP_setting['lr_pi'] = 0.05
HMMGP_setting['lr_hyp'] = 0.005  # for beta

HMMGP_setting['Len_Full'] = exp_setting['full_length']
if HMMGP_setting['train'] == 'VBEM':
    HMMGP_setting['Len_Batch'] = HMMGP_setting['Len_Full']
    HMMGP_setting['Num_Batch'] = 1
else:
    HMMGP_setting['Len_Batch'] = 10
    HMMGP_setting['Num_Batch'] = 1

HMMGP_setting['Init_iteration'] = args.init_iteration   
HMMGP_setting['Iter_train'] = args.iteration
HMMGP_setting['Iter_hyp'] = 10 

HMMGP_setting['Rate_RRFFSpectralPt'] = 0.1
HMMGP_setting['Num_RRFFSpectralPt_total'] = int((exp_setting['input_length'] * HMMGP_setting['Rate_RRFFSpectralPt']))
print('input_length : %d' % (exp_setting['input_length']))
print('Num_RRFFSpectralPt : %d' % (int((exp_setting['input_length'] * HMMGP_setting['Rate_RRFFSpectralPt']))))

HMMGP_setting['Num_RRFFBatch'] = 1
HMMGP_setting['Num_K_Emission'] = 3

HMMGP_setting['device'] = exp_setting['device']

###############################################
###############################################


Data_setting = {}
Data_setting['x_train'] = x_train
Data_setting['y_train'] = y_train
Data_setting['z_train'] = z_train
Data_setting['x_test'] = x_test
Data_setting['y_test'] = y_test
Data_setting['z_test'] = z_test
Data_setting['true_freq'] = true_freq
Data_setting['true_weight'] = true_weight
Data_setting['true_noise_level'] = true_noise_level

##############################################################################################
##############################################################################################

from itertools import product
#exp_var_v1 = [3]  # num_Q
exp_var_v1 = [args.numQ]  # num_Q
exp_var_v2 = [10]  # Len_Batch
if args.emission == 'gprrff_beta':
    #exp_var_v3 = [0.05,0.1,0.2,0.5]  # Ratio
    exp_var_v3 = [0.1,0.2,0.5]  # Ratio    
else:
    exp_var_v3 = [1.0]  # Ratio
    HMMGP_setting['Init_iteration'] =  int(args.init_iteration/2)
    
exp_var_v4 = [1]  # Ratio

for ith_Q, ith_len, ith_ratio, ith_batch in product(exp_var_v1, exp_var_v2, exp_var_v3, exp_var_v4):

    HMMGP_setting['Num_RRFFBatch'] = ith_batch

    exp_setting['Num_Q'] = ith_Q
    HMMGP_setting['Num_Q'] = exp_setting['Num_Q']
    HMMGP_setting['Len_Batch'] = ith_len
    HMMGP_setting['Rate_RRFFSpectralPt'] = ith_ratio
    HMMGP_setting['Num_RRFFSpectralPt_total'] = int((exp_setting['input_length'] * HMMGP_setting['Rate_RRFFSpectralPt']))

    EXP_Result_Dict = {}
    EXP_Result_Dict['Data_setting'] = Data_setting
    EXP_Result_Dict['Exp_setting'] = exp_setting
    EXP_Result_Dict['HMMGP_setting'] = HMMGP_setting
        
    #compute num parameters
    num_emission_param = exp_setting['Num_HiddenState']*(exp_setting['Num_Q']*(3*exp_setting['input_dim']) + 1 )
    num_trasition_param = exp_setting['Num_HiddenState']**2 + exp_setting['Num_HiddenState']
    num_modelparam = num_emission_param + num_trasition_param 
    
        

    print('#' * 100)
    print('xtrain.shape {}, xtest.shape {}'.format(x_train.shape,x_test.shape))
    numtrainset = x_train.shape[0]*x_train.shape[1]
    numtestset = x_test.shape[0]*x_test.shape[1]
    
    print(HMMGP_setting['train'] + HMMGP_setting[
        'emission'] + ' ' + 'ith_Q : %d, ith_len : %d, ith_ratio : %.3f, ith_lr : %.4f, ith_batch : %.d' % (
          ith_Q, ith_len, ith_ratio, HMMGP_setting['lr_hyp'], ith_batch))
    print('input_length : %d' % (exp_setting['input_length']))
    print('Num_RRFFSpectralPt_total : %d' % (int((exp_setting['input_length'] * HMMGP_setting['Rate_RRFFSpectralPt']))))

    # test_elbo0 = -inf
    for i_th in range(exp_setting['num_rep_exp']):
        print('#' * 100)
        print('# %d th iteration' % (i_th))

        
        # for beta setup
        model = _combine_models(x_train, y_train, exp_setting, HMMGP_setting, random_seed=random_seed,num_init_iter=HMMGP_setting['Init_iteration'])
        
#         loglik_list, train_accuracy_list, test_accuracy_list, test_exact_accuracy_list, \
#         time_list, num_cluster_list, num_test_cluster_list, num_test_exact_cluster_list, param_history_dict = model.train(
#             x_train, y_train, z_train, x_test, y_test, z_test)
        loglik_list, train_accuracy_list, test_accuracy_list, time_list, \
        num_cluster_list, param_history_dict = model.train(x_train, y_train, z_train, x_test, y_test, z_test)

        print('#' * 100)
        
        tr_elbo, tr_lik, te_elbo, te_lik, etr_lik, ete_lik = loglik_list[:,0], loglik_list[:,1], loglik_list[:,2], loglik_list[:,3], loglik_list[:,4], loglik_list[:,5]
        tr_bic = compute_bic(x_train,y_train,tr_lik[-1],num_modelparam)
        te_bic = compute_bic(x_test,y_test,te_lik[-1],num_modelparam)
        
        etr_bic = compute_bic(x_train,y_train,etr_lik[-1],num_modelparam)
        ete_bic = compute_bic(x_test,y_test,ete_lik[-1],num_modelparam)
        
             
        print('train_accuracy_list[:,0]')
        print(train_accuracy_list[:,0])
            
            
        if i_th == 0:
            EXP_Result_Dict['tr_elbo_list'] = [loglik_list[:,0]]
            EXP_Result_Dict['tr_lik_list'] = [loglik_list[:,1]]
            EXP_Result_Dict['te_elbo_list'] = [loglik_list[:,2]]
            EXP_Result_Dict['te_lik_list'] = [loglik_list[:,3]]
            
            EXP_Result_Dict['etr_lik_list'] = [loglik_list[:,4]]
            EXP_Result_Dict['ete_lik_list'] = [loglik_list[:,5]]
            
            EXP_Result_Dict['train_accuracy_list'] = [train_accuracy_list[:,0]]
            EXP_Result_Dict['test_accuracy_list'] = [test_accuracy_list[:,0]]
            EXP_Result_Dict['etrain_accuracy_list'] = [train_accuracy_list[:,1]]
            EXP_Result_Dict['etest_accuracy_list'] = [test_accuracy_list[:,1]]
            

            EXP_Result_Dict['Trained_Model'] = [model]
            EXP_Result_Dict['train_time'] = [time_list]
            EXP_Result_Dict['num_cluster_list'] = [num_cluster_list]
            EXP_Result_Dict['paramhistory_dict'] = [param_history_dict]
            EXP_Result_Dict['bic_list'] = [(tr_bic,te_bic,etr_bic,ete_bic)]            


        else:
            EXP_Result_Dict['tr_elbo_list'].append(loglik_list[:,0])
            EXP_Result_Dict['tr_lik_list'].append(loglik_list[:,1])
            EXP_Result_Dict['te_elbo_list'].append(loglik_list[:,2])
            EXP_Result_Dict['te_lik_list'].append(loglik_list[:,3])
            
            EXP_Result_Dict['etr_lik_list'].append(loglik_list[:,4])
            EXP_Result_Dict['ete_lik_list'].append(loglik_list[:,5])            

            EXP_Result_Dict['train_accuracy_list'].append(train_accuracy_list[:,0])
            EXP_Result_Dict['test_accuracy_list'].append(test_accuracy_list[:,0])
            EXP_Result_Dict['etrain_accuracy_list'].append(train_accuracy_list[:,1])
            EXP_Result_Dict['etest_accuracy_list'].append(test_accuracy_list[:,1])

            EXP_Result_Dict['Trained_Model'].append(model)
            EXP_Result_Dict['train_time'].append(time_list)
            EXP_Result_Dict['num_cluster_list'].append(num_cluster_list)
            EXP_Result_Dict['paramhistory_dict'].append(param_history_dict)
            EXP_Result_Dict['bic_list'].append((tr_bic,te_bic,etr_bic,ete_bic))            
            

            
            
    save_format = '.pickle'
    #pickle_savepath = './jupyters/result_pickle/'
    pickle_savepath = args.savepicklepath
    save_filename = 'data' + file_name \
                    + '_fulllen' + str(exp_setting['full_length']) \
                    + '_emission' + HMMGP_setting['emission'] \
                    + '_trainmethod' + HMMGP_setting['train'] \
                    + '_lenbatchsvi' + str(HMMGP_setting['Len_Batch']) \
                    + '_numbatchsvi' + str(HMMGP_setting['Num_Batch']) \
                    + '_numitertrain' + str(HMMGP_setting['Iter_train']) \
                    + '_numiterhyp' + str(HMMGP_setting['Iter_hyp']) \
                    + '_iterhyplr' + str(HMMGP_setting['lr_hyp']) \
                    + '_numQ' + str(HMMGP_setting['Num_Q']) \
                    + '_numspectralpttotal' + str(HMMGP_setting['Num_RRFFSpectralPt_total']) \
                    + '_numbatchrrff' + str(HMMGP_setting['Num_RRFFBatch']) \
                    + '_repetitiveexp' + str(args.numexp) \
                    + '_numinititeration' + str(args.init_iteration) \
                    + '_numhidden' + str(args.numhidden)


    if not os.path.exists(pickle_savepath):
        os.makedirs(pickle_savepath)
    
    with open(pickle_savepath + save_filename + save_format, 'wb') as outfile:
        pickle.dump(EXP_Result_Dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    print('-' * 50)
    print('saved file report')
    print('-' * 50)
    print('directory : {0}'.format(pickle_savepath))
    print('filename : {0}'.format(save_filename))

    with open(pickle_savepath + save_filename + save_format, 'wb') as outfile:
        pickle.dump(EXP_Result_Dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    print('-' * 50)
    print('saved file report')
    print('-' * 50)
    print('directory : {0}'.format(pickle_savepath))
    print('filename : {0}'.format(save_filename))

    
    num_digit = 3
    tr_acc_mean, tr_acc_std = np.asarray(EXP_Result_Dict['train_accuracy_list']).mean(axis=0).round(num_digit), np.asarray(EXP_Result_Dict['train_accuracy_list']).std(axis=0).round(num_digit)
    te_acc_mean, te_acc_std = np.asarray(EXP_Result_Dict['test_accuracy_list']).mean(axis=0).round(num_digit), np.asarray(EXP_Result_Dict['test_accuracy_list']).std(axis=0).round(num_digit)
    etr_acc_mean, etr_acc_std = np.asarray(EXP_Result_Dict['etrain_accuracy_list']).mean(axis=0).round(num_digit), np.asarray(EXP_Result_Dict['etrain_accuracy_list']).std(axis=0).round(num_digit)
    ete_acc_mean, ete_acc_std = np.asarray(EXP_Result_Dict['etest_accuracy_list']).mean(axis=0).round(num_digit), np.asarray(EXP_Result_Dict['etest_accuracy_list']).std(axis=0).round(num_digit)
    
    tr_elbo_mean, tr_elbo_std = np.asarray(EXP_Result_Dict['tr_elbo_list']).mean(axis=0).round(num_digit), np.asarray(EXP_Result_Dict['tr_elbo_list']).std(axis=0).round(num_digit)
    te_elbo_mean, te_elbo_std = np.asarray(EXP_Result_Dict['te_elbo_list']).mean(axis=0).round(num_digit), np.asarray(EXP_Result_Dict['te_elbo_list']).std(axis=0).round(num_digit)

    tr_lik_mean, tr_lik_std = np.asarray(EXP_Result_Dict['tr_lik_list']).mean(axis=0).round(num_digit), np.asarray(EXP_Result_Dict['tr_lik_list']).std(axis=0).round(num_digit)    
    te_lik_mean, te_lik_std = np.asarray(EXP_Result_Dict['te_lik_list']).mean(axis=0).round(num_digit), np.asarray(EXP_Result_Dict['te_lik_list']).std(axis=0).round(num_digit)
    
    tr_lik_mean /= numtestset
    tr_lik_std /= numtestset
    te_lik_mean /= numtestset
    te_lik_std /= numtestset
   
    etr_lik_mean, etr_lik_std = np.asarray(EXP_Result_Dict['etr_lik_list']).mean(axis=0).round(num_digit), np.asarray(EXP_Result_Dict['etr_lik_list']).std(axis=0).round(num_digit)
    ete_lik_mean, ete_lik_std = np.asarray(EXP_Result_Dict['ete_lik_list']).mean(axis=0).round(num_digit), np.asarray(EXP_Result_Dict['ete_lik_list']).std(axis=0).round(num_digit)

    etr_lik_mean /= numtestset
    etr_lik_std /= numtestset
    ete_lik_mean /= numtestset
    ete_lik_std /= numtestset
    
    
    
    ncluster_mean, ncluster_std = np.asarray(EXP_Result_Dict['num_cluster_list']).mean(axis=0).round(num_digit), np.asarray(EXP_Result_Dict['num_cluster_list']).std(axis=0).round(num_digit)    
    
    train_time_mean, train_time_std = np.asarray(EXP_Result_Dict['train_time']).mean().round(num_digit), np.asarray(EXP_Result_Dict['train_time']).std().round(num_digit)
    

    tr_bic_mean, tr_bic_std = np.asarray(EXP_Result_Dict['bic_list']).mean(axis=0).round(num_digit)[0], np.asarray(EXP_Result_Dict['bic_list']).std(axis=0).round(num_digit)[0]
    te_bic_mean, te_bic_std = np.asarray(EXP_Result_Dict['bic_list']).mean(axis=0).round(num_digit)[1], np.asarray(EXP_Result_Dict['bic_list']).std(axis=0).round(num_digit)[1]
    etr_bic_mean, etr_bic_std = np.asarray(EXP_Result_Dict['bic_list']).mean(axis=0).round(num_digit)[2], np.asarray(EXP_Result_Dict['bic_list']).std(axis=0).round(num_digit)[2]
    ete_bic_mean, ete_bic_std = np.asarray(EXP_Result_Dict['bic_list']).mean(axis=0).round(num_digit)[3], np.asarray(EXP_Result_Dict['bic_list']).std(axis=0).round(num_digit)[3]

    
    
    with open('./results/exp1-2_synthetic' + '.txt', 'a') as f:
        f.write('#' + '-' * 100 + '\n')
        f.write('filename %s \n' % (save_filename))        
        f.write('savepicklepath: %s \n' % (args.savepicklepath))
        f.write('\n')

        f.write('mean|,')
        f.write('tr_acc,{:.4f},tr_lik,{:.4f},tr_bic,{:.4f},te acc,{:.4f},te lik,{:.4f},te bic,{:.4f},ncluster,{:.4f},train_time,{:.4f} \n'.format(tr_acc_mean[-1],
                                                                                                                                                  tr_lik_mean[-1],
                                                                                                                                                  tr_bic_mean,                                                                                                         
                                                                                                                                                  te_acc_mean[-1],
                                                                                                                                                  te_lik_mean[-1],
                                                                                                                                                  te_bic_mean,  
                                                                                                                                                  ncluster_mean[-1],
                                                                                                                                                  train_time_mean))        

        scale_factors = 1/np.sqrt(exp_setting['num_rep_exp'])            
        f.write('std|,')
        f.write('tr_acc,{:.4f},tr_lik,{:.4f},tr_bic,{:.4f},te acc,{:.4f},te lik,{:.4f},te bic,{:.4f},ncluster,{:.4f},train_time,{:.4f} \n'.format(tr_acc_std[-1]*scale_factors,
                                                                                                                                                  tr_lik_std[-1]*scale_factors,
                                                                                                                                                  tr_bic_std*scale_factors,
                                                                                                                                                  te_acc_std[-1]*scale_factors,
                                                                                                                                                  te_lik_std[-1]*scale_factors,
                                                                                                                                                  te_bic_std*scale_factors,
                                                                                                                                                  ncluster_std[-1]*scale_factors,
                                                                                                                                                  train_time_std*scale_factors))

        f.write('\n')        
        f.write('mean exact|,')
        f.write('etr_acc,{:.4f},etr_lik,{:.4f},etr_bic,{:.4f},ete acc,{:.4f},ete lik,{:.4f},ete bic,{:.4f}  \n'.format(etr_acc_mean[-1],
                                                                                               etr_lik_mean[-1],
                                                                                               etr_bic_mean,                                                                                                      
                                                                                               ete_acc_mean[-1],
                                                                                               ete_lik_mean[-1],
                                                                                               ete_bic_mean))
        
        f.write('std exact|,')
        f.write('etr_acc,{:.4f},etr_lik,{:.4f},etr_bic,{:.4f},ete acc,{:.4f},ete lik,{:.4f},ete bic,{:.4f}  \n'.format(etr_acc_std[-1]*scale_factors,
                                                                                               etr_lik_std[-1]*scale_factors,
                                                                                               etr_bic_std*scale_factors,                                                                                                      
                                                                                               ete_acc_std[-1]*scale_factors,
                                                                                               ete_lik_std[-1]*scale_factors,
                                                                                               ete_bic_std*scale_factors))
                
                
        f.write('\n\n')