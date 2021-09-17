from utility.dataset import _process_Real3
from models_utility.construct_models import _combine_models
from utility.eval_metric import _measure_metric, accuracy

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
parser.add_argument('--iteration', type=int, default=20)
parser.add_argument('--lrhyp', type=float, default=0.0005)

args = parser.parse_args()
expinfo = make_expinfo(args)



##############################################################################
# load dataset
##############################################################################

random_seed = 1000

file_path =  './dataset/real/'
file_name = 'digit_v2_T4096_jackson_Hz0-511_13579'



#file_name = 'digit_nicolas_Hz1-500'
#

data_file_path = file_path + file_name
down_sample = 1
x_train, y_train, z_train, x_test, y_test, z_test, num_state = _process_Real3(data_file_path, down_sample)




###############################################################################################
# experiment_setting
###############################################################################################

#num_exp = 5
num_exp = args.numexp
##############################################################################################
##############################################################################################

exp_setting = {}
exp_setting['num_rep_exp'] = num_exp
exp_setting['data'] = file_name


#exp_setting['full_length'] = 50
#exp_setting['full_length'] = 75
exp_setting['full_length'] = len(x_train)



exp_setting['input_dim'] = 1
exp_setting['Num_Q'] = 4
exp_setting['Num_HiddenState'] = num_state

exp_setting['device'] = False


#print(x_train)
exp_setting['input_length'] = len(x_train[0])

#exp_setting['input_length'] = 1
##############################################################################################
##############################################################################################

HMMGP_setting = {}
#HMMGP_setting['emission'] = 'gprbf' #
#HMMGP_setting['emission'] = 'gpsm' #
#HMMGP_setting['emission'] = 'gprrff' #
HMMGP_setting['emission'] = 'gprrff_beta'


#HMMGP_setting['train'] = 'VBEM' #
HMMGP_setting['train'] = 'SVI' #
HMMGP_setting['Num_Q'] = exp_setting['Num_Q']

# HMMGP_setting['lr_A'] = 0.1
# HMMGP_setting['lr_pi'] = 0.1

HMMGP_setting['lr_A'] = 0.05
HMMGP_setting['lr_pi'] = 0.05
#HMMGP_setting['lr_hyp'] = 0.001
#HMMGP_setting['lr_hyp'] = 0.005


#HMMGP_setting['lr_hyp'] = 0.0005
#HMMGP_setting['lr_hyp'] = 0.0001
#HMMGP_setting['lr_hyp'] = 0.001
HMMGP_setting['lr_hyp'] = args.lrhyp



HMMGP_setting['Len_Full'] = exp_setting['full_length']
if HMMGP_setting['train'] == 'VBEM':
    HMMGP_setting['Len_Batch'] = HMMGP_setting['Len_Full']
    HMMGP_setting['Num_Batch'] = 1
else:
    HMMGP_setting['Len_Batch'] = 10
    HMMGP_setting['Num_Batch'] = 1


HMMGP_setting['device'] = exp_setting['device']


#HMMGP_setting['Iter_train'] = 4
#HMMGP_setting['Iter_train'] = 20
HMMGP_setting['Iter_train'] = args.iteration


#HMMGP_setting['Iter_hyp'] = 5
#HMMGP_setting['Iter_hyp'] = 10
HMMGP_setting['Iter_hyp'] = 10



HMMGP_setting['Num_RRFFBatch'] = 1
HMMGP_setting['Num_K_Emission'] = 2





Data_setting = {}
Data_setting['x_train'] = x_train
Data_setting['y_train'] = y_train
Data_setting['z_train'] = z_train
Data_setting['x_test'] = x_test
Data_setting['y_test'] = y_test
Data_setting['z_test'] = z_test




##############################################################################################
##############################################################################################

from itertools import product
exp_var_v1 = [4,5]      #num_Q
exp_var_v2 = [10] #Len_Batch
#exp_var_v3 = [.1,.2] #Ratio
exp_var_v3 = [.1] #Ratio


for ith_Q,ith_len,ith_ratio in product(exp_var_v1,exp_var_v2,exp_var_v3):

    exp_setting['Num_Q'] = ith_Q
    HMMGP_setting['Num_Q'] = exp_setting['Num_Q']
    HMMGP_setting['Len_Batch'] = ith_len
    HMMGP_setting['Rate_RRFFSpectralPt'] = ith_ratio
    HMMGP_setting['Num_RRFFSpectralPt_total'] = int((exp_setting['input_length'] * HMMGP_setting['Rate_RRFFSpectralPt']))

    EXP_Result_Dict = {}
    EXP_Result_Dict['Data_setting'] = Data_setting
    EXP_Result_Dict['Exp_setting'] = exp_setting
    EXP_Result_Dict['HMMGP_setting'] = HMMGP_setting

    print('#' * 100)
    print(HMMGP_setting['train']  + HMMGP_setting['emission'] + ' ' + 'ith_Q : %d, ith_len : %d, ith_ratio : %.3f'%(ith_Q, ith_len, ith_ratio))
    print('input_length : %d' % (exp_setting['input_length']))
    print('Num_RRFFSpectralPt_total : %d' % (int((exp_setting['input_length'] * HMMGP_setting['Rate_RRFFSpectralPt']))))

    for i_th in range(exp_setting['num_rep_exp']):
        print('#'*50)
        print('# %d th iteration'%(i_th))


        model = _combine_models(x_train, y_train, exp_setting, HMMGP_setting,random_seed=random_seed,num_init_iter = 501)
        loglik_list, train_accuracy_list, test_accuracy_list, test_exact_accuracy_list, \
        time_list, num_cluster_list, num_test_cluster_list, num_test_exact_cluster_list, param_history_dict = model.train(x_train, y_train, z_train, x_test, y_test, z_test)
        

        if i_th ==0 :
            EXP_Result_Dict['loglik_list'] = [loglik_list]
            EXP_Result_Dict['train_accuracy_list'] = [train_accuracy_list]
            EXP_Result_Dict['test_accuracy_list'] = [test_accuracy_list]
            
            EXP_Result_Dict['Trained_Model'] = [model]
            EXP_Result_Dict['train_time'] = [time_list]
            EXP_Result_Dict['num_cluster_list'] = [num_cluster_list]
            EXP_Result_Dict['paramhistory_dict'] = [param_history_dict]

            #EXP_Result_Dict['test_elbo'] = [test_elbo]
            #EXP_Result_Dict['test_accuracy'] = [test_accuracy]
            #EXP_Result_Dict['test_pred'] = [z_test_pred]
            #EXP_Result_Dict['test_elbo_before'] = [test_elbo_before]
            #EXP_Result_Dict['test_accuracy_before'] = [test_accuracy_before]

        else:
            EXP_Result_Dict['loglik_list'].append(loglik_list)
            EXP_Result_Dict['train_accuracy_list'].append(train_accuracy_list)
            EXP_Result_Dict['test_accuracy_list'].append(test_accuracy_list)
            
            EXP_Result_Dict['Trained_Model'].append(model)
            EXP_Result_Dict['train_time'].append(time_list)
            EXP_Result_Dict['num_cluster_list'].append(num_cluster_list)
            EXP_Result_Dict['paramhistory_dict'].append(param_history_dict)

            #EXP_Result_Dict['test_elbo'].append(test_elbo)
            #EXP_Result_Dict['test_accuracy'].append(test_accuracy)
            #EXP_Result_Dict['test_pred'].append(z_test_pred)
            #EXP_Result_Dict['test_elbo_before'].append(test_elbo_before)
            #EXP_Result_Dict['test_accuracy_before'].append(test_accuracy_before)




    save_format = '.pickle'
    pickle_savepath = './jupyters/result_pickle/'
    save_filename =    'data' + file_name \
                     + '_fulllen' + str(exp_setting['full_length']) \
                     + '_emission' + HMMGP_setting['emission'] \
                     + '_trainmethod' + HMMGP_setting['train'] \
                     + '_lenbatchsvi' +  str(HMMGP_setting['Len_Batch']) \
                     + '_numbatchsvi' + str(HMMGP_setting['Num_Batch']) \
                     + '_numhypiter' + str(HMMGP_setting['Iter_hyp']) \
                     + '_numQ' + str(HMMGP_setting['Num_Q']) \
                     + '_numspttotal' +  str(HMMGP_setting['Num_RRFFSpectralPt_total']) \
                     + '_numbatchrrff' + str(HMMGP_setting['Num_RRFFBatch']) \
                     + '_repetitiveexp' + str(num_exp) \




    with open(pickle_savepath + save_filename + save_format, 'wb') as outfile:
        pickle.dump(EXP_Result_Dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    print('-' * 50)
    print('saved file report')
    print('-' * 50)
    print('directory : {0}'.format(pickle_savepath))
    print('filename : {0}'.format(save_filename))










######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################


# from utility.dataset import _process_Real3
# from models_utility.construct_models import _combine_models
# from utility.eval_metric import _measure_metric, accuracy

# import torch
# import pickle
# import numpy as np
# import argparse


# def make_expinfo(args):
#     args_dict = args.__dict__
#     model_info = ''
#     for ith_keys in args_dict:
#         model_info += ith_keys + str(args_dict[ith_keys]) + '_'
#     return model_info


# parser = argparse.ArgumentParser(description='data_file_load')
# parser.add_argument('--numexp', type=int, default=5)
# parser.add_argument('--iteration', type=int, default=20)
# args = parser.parse_args()
# expinfo = make_expinfo(args)



# ##############################################################################
# # load dataset
# ##############################################################################

# random_seed = 1000

# file_path =  './dataset/real/'
# file_name = 'digit_v2_T4096_jackson_Hz0-511_13579'



# #file_name = 'digit_nicolas_Hz1-500'
# #

# data_file_path = file_path + file_name
# down_sample = 1
# x_train, y_train, z_train, x_test, y_test, z_test, num_state = _process_Real3(data_file_path, down_sample)




# ###############################################################################################
# # experiment_setting
# ###############################################################################################

# #num_exp = 5
# num_exp = args.numexp
# ##############################################################################################
# ##############################################################################################

# exp_setting = {}
# exp_setting['num_rep_exp'] = num_exp
# exp_setting['data'] = file_name


# #exp_setting['full_length'] = 50
# #exp_setting['full_length'] = 75
# exp_setting['full_length'] = len(x_train)



# exp_setting['input_dim'] = 1
# exp_setting['Num_Q'] = 4
# exp_setting['Num_HiddenState'] = num_state

# exp_setting['device'] = False


# #print(x_train)
# exp_setting['input_length'] = len(x_train[0])

# #exp_setting['input_length'] = 1
# ##############################################################################################
# ##############################################################################################

# HMMGP_setting = {}
# #HMMGP_setting['emission'] = 'gprbf' #
# #HMMGP_setting['emission'] = 'gpsm' #
# #HMMGP_setting['emission'] = 'gprrff' #
# HMMGP_setting['emission'] = 'gprrff_beta'


# #HMMGP_setting['train'] = 'VBEM' #
# HMMGP_setting['train'] = 'SVI' #
# HMMGP_setting['Num_Q'] = exp_setting['Num_Q']

# # HMMGP_setting['lr_A'] = 0.1
# # HMMGP_setting['lr_pi'] = 0.1

# HMMGP_setting['lr_A'] = 0.05
# HMMGP_setting['lr_pi'] = 0.05
# #HMMGP_setting['lr_hyp'] = 0.001
# #HMMGP_setting['lr_hyp'] = 0.005


# #HMMGP_setting['lr_hyp'] = 0.0005
# #HMMGP_setting['lr_hyp'] = 0.0001
# HMMGP_setting['lr_hyp'] = 0.001




# HMMGP_setting['Len_Full'] = exp_setting['full_length']
# if HMMGP_setting['train'] == 'VBEM':
#     HMMGP_setting['Len_Batch'] = HMMGP_setting['Len_Full']
#     HMMGP_setting['Num_Batch'] = 1
# else:
#     HMMGP_setting['Len_Batch'] = 10
#     HMMGP_setting['Num_Batch'] = 1


# HMMGP_setting['device'] = exp_setting['device']


# #HMMGP_setting['Iter_train'] = 4
# #HMMGP_setting['Iter_train'] = 20
# HMMGP_setting['Iter_train'] = args.iteration


# #HMMGP_setting['Iter_hyp'] = 5
# #HMMGP_setting['Iter_hyp'] = 10
# HMMGP_setting['Iter_hyp'] = 10



# HMMGP_setting['Num_RRFFBatch'] = 1
# HMMGP_setting['Num_K_Emission'] = 2





# Data_setting = {}
# Data_setting['x_train'] = x_train
# Data_setting['y_train'] = y_train
# Data_setting['z_train'] = z_train
# Data_setting['x_test'] = x_test
# Data_setting['y_test'] = y_test
# Data_setting['z_test'] = z_test




# ##############################################################################################
# ##############################################################################################

# from itertools import product
# exp_var_v1 = [4,5]      #num_Q
# exp_var_v2 = [10] #Len_Batch
# #exp_var_v3 = [.1,.2] #Ratio
# exp_var_v3 = [.1] #Ratio


# for ith_Q,ith_len,ith_ratio in product(exp_var_v1,exp_var_v2,exp_var_v3):

#     exp_setting['Num_Q'] = ith_Q
#     HMMGP_setting['Num_Q'] = exp_setting['Num_Q']
#     HMMGP_setting['Len_Batch'] = ith_len
#     HMMGP_setting['Rate_RRFFSpectralPt'] = ith_ratio
#     HMMGP_setting['Num_RRFFSpectralPt_total'] = int((exp_setting['input_length'] * HMMGP_setting['Rate_RRFFSpectralPt']))

#     EXP_Result_Dict = {}
#     EXP_Result_Dict['Data_setting'] = Data_setting
#     EXP_Result_Dict['Exp_setting'] = exp_setting
#     EXP_Result_Dict['HMMGP_setting'] = HMMGP_setting

#     print('#' * 100)
#     print(HMMGP_setting['train']  + HMMGP_setting['emission'] + ' ' + 'ith_Q : %d, ith_len : %d, ith_ratio : %.3f'%(ith_Q, ith_len, ith_ratio))
#     print('input_length : %d' % (exp_setting['input_length']))
#     print('Num_RRFFSpectralPt_total : %d' % (int((exp_setting['input_length'] * HMMGP_setting['Rate_RRFFSpectralPt']))))

#     for i_th in range(exp_setting['num_rep_exp']):
#         print('#'*50)
#         print('# %d th iteration'%(i_th))


#         model = _combine_models(x_train, y_train, exp_setting, HMMGP_setting,random_seed=random_seed,num_init_iter = 200)
#         #model = _combine_models_v2(x_train, y_train, exp_setting, HMMGP_setting ,random_seed=random_seed,num_init_iter = 100)

#         z_test_pred_before, test_elbo_before = model._run_smoothing(x_test, y_test, num_test_batch=2, test_option=True)
#         test_accuracy_before = accuracy(z_test, z_test_pred_before)
#         print('train before accuracy %.4f'%(test_accuracy_before))

#         train_loglik_list, train_accuracy_list, test_accuracy_list, test_exact_accuracy_list, \
#         time_list, num_cluster_list, num_test_cluster_list, num_test_exact_cluster_list, param_history_dict = model.train(x_train, y_train, z_train, x_test, y_test, z_test)
        
#         z_train_pred, train_elbo, train_lik = model._run_smoothing(x_train, y_train, num_test_batch=2, test_option=True)
#         z_test_pred, test_elbo, test_lik = model._run_smoothing(x_test, y_test, num_test_batch=2, test_option=True)

#         test_accuracy = accuracy(z_test, z_test_pred)
#         print('train before %.4f, after %.4f'%(test_accuracy_before,test_accuracy))


#         if i_th ==0 :
#             EXP_Result_Dict['train_elbo'] = [train_loglik_list]
#             EXP_Result_Dict['train_accuracy_list'] = [train_accuracy_list]
#             EXP_Result_Dict['test_accuracy_list'] = [test_accuracy_list]


#             EXP_Result_Dict['test_elbo'] = [test_elbo]
#             EXP_Result_Dict['test_accuracy'] = [test_accuracy]
#             EXP_Result_Dict['test_pred'] = [z_test_pred]
#             EXP_Result_Dict['Trained_Model'] = [model]
#             EXP_Result_Dict['train_time'] = [time_list]
#             EXP_Result_Dict['num_cluster_list'] = [num_cluster_list]

#             EXP_Result_Dict['test_elbo_before'] = [test_elbo_before]
#             EXP_Result_Dict['test_accuracy_before'] = [test_accuracy_before]
#             EXP_Result_Dict['paramhistory_dict'] = [param_history_dict]

#         else:
#             EXP_Result_Dict['train_elbo'].append(train_loglik_list)
#             EXP_Result_Dict['train_accuracy_list'].append(train_accuracy_list)
#             EXP_Result_Dict['test_accuracy_list'].append(test_accuracy_list)

#             EXP_Result_Dict['test_elbo'].append(test_elbo)
#             EXP_Result_Dict['test_accuracy'].append(test_accuracy)
#             EXP_Result_Dict['test_pred'].append(z_test_pred)
#             EXP_Result_Dict['Trained_Model'].append(model)
#             EXP_Result_Dict['train_time'].append(time_list)
#             EXP_Result_Dict['num_cluster_list'].append(num_cluster_list)

#             EXP_Result_Dict['test_elbo_before'].append(test_elbo_before)
#             EXP_Result_Dict['test_accuracy_before'].append(test_accuracy_before)
#             EXP_Result_Dict['paramhistory_dict'].append(param_history_dict)




#     save_format = '.pickle'
#     pickle_savepath = './jupyters/result_pickle/'
#     save_filename =    'data' + file_name \
#                      + '_fulllen' + str(exp_setting['full_length']) \
#                      + '_emission' + HMMGP_setting['emission'] \
#                      + '_trainmethod' + HMMGP_setting['train'] \
#                      + '_lenbatchsvi' +  str(HMMGP_setting['Len_Batch']) \
#                      + '_numbatchsvi' + str(HMMGP_setting['Num_Batch']) \
#                      + '_numhypiter' + str(HMMGP_setting['Iter_hyp']) \
#                      + '_numQ' + str(HMMGP_setting['Num_Q']) \
#                      + '_numspectralpttotal' +  str(HMMGP_setting['Num_RRFFSpectralPt_total']) \
#                      + '_numbatchrrff' + str(HMMGP_setting['Num_RRFFBatch']) \
#                      + '_repetitiveexp' + str(num_exp) \




#     with open(pickle_savepath + save_filename + save_format, 'wb') as outfile:
#         pickle.dump(EXP_Result_Dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
#     print('-' * 50)
#     print('saved file report')
#     print('-' * 50)
#     print('directory : {0}'.format(pickle_savepath))
#     print('filename : {0}'.format(save_filename))


