from utility.dataset import _process_Real2
from models_utility.construct_models import _combine_models
from utility.eval_metric import _measure_metric, accuracy

import torch
import pickle
import numpy as np
#torch.set_default_tensor_type(torch.FloatTensor)


##############################################################################
# load dataset
##############################################################################

random_seed = 1000

file_path = './dataset/real/'
file_name = 'PigCVP10_Set_Dynamic_Downsample_rate1'


data_file_path = file_path + file_name
x_train, y_train, z_train, x_test, y_test, z_test, num_state = _process_Real2(data_file_path,down_sample=10)

print(np.shape(x_train))
print(z_train)

###############################################################################################
# experiment_setting
###############################################################################################

num_exp = 3

##############################################################################################
##############################################################################################

exp_setting = {}
exp_setting['num_rep_exp'] = num_exp
exp_setting['data'] = file_name


#exp_setting['full_length'] = 50
#exp_setting['full_length'] = 75
exp_setting['full_length'] = len(x_train)



exp_setting['input_dim'] = 1
exp_setting['Num_Q'] = 5
exp_setting['Num_HiddenState'] = num_state
exp_setting['input_length'] = x_train.shape[1]


exp_setting['device'] = False


##############################################################################################
##############################################################################################

HMMGP_setting = {}
HMMGP_setting['emission'] = 'gpsm' #
#HMMGP_setting['emission'] = 'gprrff' #

#HMMGP_setting['train'] = 'VBEM' #
HMMGP_setting['train'] = 'SVI' #

HMMGP_setting['Num_Q'] = exp_setting['Num_Q']


HMMGP_setting['lr_A'] = 0.05
HMMGP_setting['lr_pi'] = 0.05
HMMGP_setting['lr_hyp'] = 0.001


HMMGP_setting['Len_Full'] = exp_setting['full_length']
if HMMGP_setting['train'] == 'VBEM':
    HMMGP_setting['Len_Batch'] = HMMGP_setting['Len_Full']
    HMMGP_setting['Num_Batch'] = 1
else:
    HMMGP_setting['Len_Batch'] = 10
    HMMGP_setting['Num_Batch'] = 1


HMMGP_setting['device'] = exp_setting['device']


HMMGP_setting['Iter_train'] = 40
HMMGP_setting['Iter_hyp'] = 10
HMMGP_setting['Rate_RRFFSpectralPt'] = 0.1
HMMGP_setting['Num_RRFFSpectralPt_total'] = int ( (exp_setting['input_length']*HMMGP_setting['Rate_RRFFSpectralPt']) )
print('input_length : %d'%(exp_setting['input_length']))
print('Num_RRFFSpectralPt_total : %d'%(int ( (exp_setting['input_length']*HMMGP_setting['Rate_RRFFSpectralPt']) )))

HMMGP_setting['Num_RRFFBatch'] = 1





Data_setting = {}
Data_setting['x_train'] = x_train
Data_setting['y_train'] = y_train
Data_setting['z_train'] = z_train
Data_setting['x_test'] = x_test
Data_setting['y_test'] = y_test
Data_setting['z_test'] = z_test


EXP_Result_Dict = {}
EXP_Result_Dict['Data_setting'] = Data_setting
EXP_Result_Dict['Exp_setting'] = exp_setting
EXP_Result_Dict['HMMGP_setting'] = HMMGP_setting


##############################################################################################
##############################################################################################

for i_th in range(exp_setting['num_rep_exp']):
    print('#'*100)
    print('# %d th iteration'%(i_th))


    model = _combine_models(x_train, y_train, exp_setting, HMMGP_setting,random_seed=random_seed,num_init_iter=10)

    z_test_pred_before, test_elbo_before = model._run_smoothing(x_test, y_test)
    test_accuracy_before = accuracy(z_test, z_test_pred_before)
    print('#'*10)
    print('before test accuracy before')
    print(test_accuracy_before)

    train_loglik_list, train_accuracy_list, time_list ,num_cluster_list , paramhistory_dict = model.train(x_train, y_train, z_train )
    z_test_pred, test_elbo = model._run_smoothing(x_test, y_test)
    test_accuracy = accuracy(z_test, z_test_pred)

    print('#'*10)
    print('before, after test accuracy')
    print(test_accuracy_before, test_accuracy)


    if i_th ==0 :
        EXP_Result_Dict['train_elbo'] = [train_loglik_list]
        EXP_Result_Dict['train_accuracy'] = [train_accuracy_list]
        EXP_Result_Dict['test_elbo'] = [test_elbo]
        EXP_Result_Dict['test_accuracy'] = [test_accuracy]
        EXP_Result_Dict['test_pred'] = [z_test_pred]
        EXP_Result_Dict['Trained_Model'] = [model]
        EXP_Result_Dict['train_time'] = [time_list]
        EXP_Result_Dict['num_cluster_list'] = [num_cluster_list]

        EXP_Result_Dict['test_elbo_before'] = [test_elbo_before]
        EXP_Result_Dict['test_accuracy_before'] = [test_accuracy_before]
        EXP_Result_Dict['paramhistory_dict'] = [paramhistory_dict]

    else:
        EXP_Result_Dict['train_elbo'].append(train_loglik_list)
        EXP_Result_Dict['train_accuracy'].append(train_accuracy_list)
        EXP_Result_Dict['test_elbo'].append(test_elbo)
        EXP_Result_Dict['test_accuracy'].append(test_accuracy)
        EXP_Result_Dict['test_pred'].append(z_test_pred)
        EXP_Result_Dict['Trained_Model'].append(model)
        EXP_Result_Dict['train_time'].append(time_list)
        EXP_Result_Dict['num_cluster_list'].append(num_cluster_list)

        EXP_Result_Dict['test_elbo_before'].append(test_elbo_before)
        EXP_Result_Dict['test_accuracy_before'].append(test_accuracy_before)
        EXP_Result_Dict['paramhistory_dict'].append(paramhistory_dict)



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
                 + '_numspectralpttotal' +  str(HMMGP_setting['Num_RRFFSpectralPt_total']) \
                 + '_numbatchrrff' + str(HMMGP_setting['Num_RRFFBatch']) \
                 + '_repetitiveexp' + str(num_exp)  \




with open(pickle_savepath + save_filename + save_format, 'wb') as outfile:
    pickle.dump(EXP_Result_Dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
print('-' * 50)
print('saved file report')
print('-' * 50)
print('directory : {0}'.format(pickle_savepath))
print('filename : {0}'.format(save_filename))


