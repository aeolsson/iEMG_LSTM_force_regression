import os
import gc
import numpy as np
from tensorflow.keras.backend import clear_session
from scipy.io import savemat

from preprocess import preprocess
from regression_models import One_to_one, All_to_one, All_to_all
from errors import root_mean_squared, variance_accounted_for

import params

d = 10 #downsampling factor

emg_win_size = 512 # iEMG window size
emg_win_step_size = 16# iEMG window step size

train_part = 7/10 # Sinusoid periods for training
val_part = 1/10 # Sinusoid periods for validation
test_part = 2/10 # Sinusoid periods for testing

LSTM_size = 64 # Number of units in LSTM layer

dropout = 0.20 # Network dropout probability

learning_rate = 1e-4 # AdamW learning rate
reg_const = 1.0e-9 # L2 regularization

batch_size = 1000#32 # AdamW batch size
epochs = 1#250 # AdamW epochs 

val_patience = 25 # Consecutive nonincreasing epochs before early stop

filenames = [n for n in os.listdir(params.dataset_path) if '.csv' in n]

VAFs_a2a = np.zeros(16)
RMSEs_a2a = np.zeros(16)

VAFs_a2o = np.zeros(16)
RMSEs_a2o = np.zeros(16)

VAFs_o2o = np.zeros(16)
RMSEs_o2o = np.zeros(16)

for name in filenames:
    subject_name = name[0:-4]
    subject = int(subject_name[1::])
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, iemg_force_correspondance = preprocess(name, d, emg_win_size, emg_win_step_size, train_part, val_part, test_part)
    num_iemg = X_train.shape[-1]
    num_forces = y_train.shape[-1]
    
    y_true = np.zeros((X_test.shape[0], num_iemg))
    for iemg_channel in range(num_iemg):
        y_true[:, iemg_channel] = y_test[:, iemg_force_correspondance[iemg_channel]]
    
    # All-to-All
    model_a2a = All_to_all(num_iemg, num_forces, iemg_force_correspondance, LSTM_size, dropout, learning_rate, reg_const)
    model_a2a.train(X_train, y_train, X_val, y_val, epochs, batch_size, val_patience)
    y_pred_a2a = model_a2a.predict(X_test)
    
    VAF_a2a = variance_accounted_for(y_true, y_pred_a2a)
    RMSE_a2a = root_mean_squared(y_true, y_pred_a2a)
    
    VAFs_a2a[subject-1] = VAF_a2a
    RMSEs_a2a[subject-1] = RMSE_a2a
    
    del model_a2a
    clear_session()
    gc.collect()
    
    # All-to-One
    model_a2o = All_to_one(num_iemg, num_forces, iemg_force_correspondance, LSTM_size, dropout, learning_rate, reg_const)
    model_a2o.train(X_train, y_train, X_val, y_val, epochs, batch_size, val_patience)
    y_pred_a2o = model_a2o.predict(X_test)
    
    VAF_a2o = variance_accounted_for(y_true, y_pred_a2o)
    RMSE_a2o = root_mean_squared(y_true, y_pred_a2o)
    
    VAFs_a2o[subject-1] = VAF_a2o
    RMSEs_a2o[subject-1] = RMSE_a2o
    
    del model_a2o
    clear_session()
    gc.collect()
        
    # One-to-one
    model_o2o = One_to_one(num_iemg, num_forces, iemg_force_correspondance, LSTM_size, dropout, learning_rate, reg_const)
    model_o2o.train(X_train, y_train, X_val, y_val, epochs, batch_size, val_patience)
    y_pred_o2o = model_o2o.predict(X_test)
        
    clear_session()
    del model_o2o
    gc.collect()
    
    VAF_o2o = variance_accounted_for(y_true, y_pred_o2o)
    RMSE_o2o = root_mean_squared(y_true, y_pred_o2o)
    
    VAFs_o2o[subject-1] = VAF_o2o
    RMSEs_o2o[subject-1] = RMSE_o2o
        




    break
    #Save
    metrics_dict = {'VAFs_a2a': VAFs_a2a,
                    'RMSEs_a2a': RMSEs_a2a,
                    'VAFs_a2o': VAFs_a2o,
                    'RMSEs_a2o': RMSEs_a2o,
                    'VAFs_o2o': VAFs_o2o,
                    'RMSEs_o2o': RMSEs_o2o}
    
    signals_dict = {'y_true': y_true,
                    'y_pred_a2a': y_pred_a2a,
                    'y_pred_a2o': y_pred_a2o,
                    'y_pred_o2o': y_pred_o2o,}
    
    savemat('{}/{}/metrics.mat'.format(params.results_path, str(subject)), metrics_dict)
    savemat('{}/{}/signals.mat'.format(params.results_path, str(subject)), signals_dict)
    
    del X_train, y_train, X_test, y_test
    gc.collect()