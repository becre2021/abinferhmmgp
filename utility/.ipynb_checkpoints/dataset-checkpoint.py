import torch
import numpy as np
import scipy.io as sio
import pickle



def _process_Synthetic(data_file_path,exp_setting):
    #data_formatted = '.mat'
    T = exp_setting['full_length']
    Synthetic = sio.loadmat(data_file_path)

    # save(save_dir2, 'xfull', 'yfull', 'zfull', 'Fs', ...
    # 'freq1', 'weight1', 'noise_level', 'num_state'

    x_full,y_full,z_full = Synthetic['xfull'],Synthetic['yfull'],Synthetic['zfull']
    x_train,x_test = x_full[:T,:],x_full[T:,:]
    y_train, y_test = y_full[:T,:], y_full[T:,:]
    z_train, z_test = z_full[0,:T], z_full[0,T:]

    # to make z \in [0,1,..,K-1] for K states
    z_test -= 1
    z_train -= 1
    

    freq, weight = Synthetic['freq1'], Synthetic['weight1']
    noise_level,num_state = Synthetic['noise_level'], Synthetic['num_state']
    return  x_train,y_train,z_train , \
            x_test,y_test,z_test,\
            freq,weight,noise_level,num_state




def _process_Real2(data_file_path, down_sample):

    format_name = '.mat'
    Real = sio.loadmat(data_file_path + format_name)

    x_train = Real['x_train']
    y_train = Real['y_train']
    z_train = Real['z_train'][0]

    x_test = Real['x_test']
    y_test = Real['y_test']
    z_test = Real['z_test'][0]

    num_state = Real['num_state'][0,0]
    #print(num_state)

    return x_train[:,::down_sample], y_train[:,::down_sample],z_train, \
           x_test[:,::down_sample], y_test[:,::down_sample], z_test, num_state



def _process_Real3(data_file_path, down_sample):

    format_name = '.pickle'
    with open(data_file_path + format_name, 'rb') as f:
        data_pickle = pickle.load(f)  # 단 한줄씩 읽어옴

    #print(data_pickle.keys())
    x_train = np.asarray(data_pickle['x_train'])
    y_train = np.asarray(data_pickle['y_train'])
    y_train = (y_train - y_train.mean(axis = 1,keepdims= True))/y_train.std(axis=1,keepdims= True)
    z_train = np.asarray(data_pickle['z_train'])


    x_test = np.asarray(data_pickle['x_test'])
    y_test = np.asarray(data_pickle['y_test'])
    y_test = (y_test - y_test.mean(axis = 1,keepdims= True))/y_test.std(axis=1,keepdims= True)
    z_test = np.asarray(data_pickle['z_test'])

    num_state = data_pickle['num_state']
    #print(num_state)


    if down_sample == 1:
        return x_train, y_train, z_train, \
               x_test, y_test, z_test, num_state

    else:
        x_train_sampled = []
        y_train_sampled = []
        for ith_x,ith_y in zip(x_train,y_train):
            x_train_sampled.append(ith_x[::down_sample])
            y_train_sampled.append(ith_y[::down_sample])

        x_test_sampled = []
        y_test_sampled = []
        for ith_x,ith_y in zip(x_test,y_test):
            x_test_sampled.append(ith_x[::down_sample])
            y_test_sampled.append(ith_y[::down_sample])


        return x_train_sampled, y_train_sampled, z_train, \
               x_test_sampled, y_test_sampled, z_test, num_state




