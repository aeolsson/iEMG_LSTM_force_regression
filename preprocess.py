import numpy as np
import pandas as pd
import gc

from scipy.signal import butter, filtfilt

import params
import utils

def load(name):
    subject_name = name[0:-4]
    subject = int(subject_name[1::])
    proto = params.subject_protocols[subject_name]
    
    muscle2channel = params.muscle2channel[subject-1]
    channel2force = params.channel2force[subject-1]
    force2name = params.force2name
    channel2sign = params.channel2sign[subject-1]
    
    df = pd.read_csv("{}\{}".format(params.dataset_path, name))
    mat = np.asarray(df)
    
    stage = np.round(np.asarray(df['Stage']), decimals=1)
    cue = np.asarray(df['Tracking_Cue'])
    
    
    if subject == 12:
        muscle_names = params.muscles[12]
            
    else:
        muscle_names = params.muscles[proto]

    
    iemg = np.zeros((mat.shape[0], len(muscle_names)))
    
    force_names = []
    force_signs = []
    iemg_force_correspondance = {}
    
    for i, muscle_name in enumerate(muscle_names):
        iemg[:, i] = np.asarray(df[muscle_name])
        iemg_channel_index = muscle2channel[muscle_name]
        force_index = channel2force[iemg_channel_index]
        force_name = force2name[force_index]
        force_sign = channel2sign[iemg_channel_index]
        
        new_force = True
        for j, (name, sign) in enumerate(zip(force_names, force_signs)):
            if force_name == name and force_sign == sign:
                new_force = False
        
        if new_force:
            force_names.append(force_name)
            force_signs.append(force_sign)
            iemg_force_correspondance[i] = len(force_names) - 1
        else:
            iemg_force_correspondance[i] = j
    
    force = np.zeros((mat.shape[0], len(force_names)))
    for i, force_name in enumerate(force_names):
        force[:, i] = np.asarray(df[force_name])
        
    force = 40*force - 100
    
    del mat, df
    gc.collect()
    
    # Throw away superflous samples
    last_sample = np.where(stage >= 9.0)[0][0]
    iemg = iemg[:last_sample:, :]
    force = force[:last_sample:, :]
    cue = cue[:last_sample:]
    stage = stage[:last_sample:]
    
    return iemg, force, cue, stage, iemg_force_correspondance, force_signs
    
    
def filter(iemg, force):
    t = np.percentile(iemg, 99, axis=0)
    b = np.percentile(iemg, 1, axis=0)
    iemg = np.clip(iemg,
                   a_min=b,
                   a_max=t)
    
    fs = 10240
    
    b, a = butter(N=2,
                  Wn=500,
                  btype='lowpass',
                  fs=fs)
    iemg = filtfilt(b,
                    a,
                    iemg,
                    axis=0)
    
    b, a = butter(N=2,
                  Wn=10.0,
                  btype='lowpass',
                  fs=fs)
    force = filtfilt(b,
                     a,
                     force,
                     axis=0)
    
    return iemg, force
    
    
def downsample(iemg, force, stage, cue, d):
    iemg = iemg[::d]
    force = force[::d]
    stage = stage[::d]
    cue = cue[::d]
    
    return iemg, force, stage, cue
    
def segment(iemg, force, stage, cue, iemg_force_correspondance, force_signs, subject):
    channel2movement = params.channel2movement[subject-1]
    num_iemg = np.shape(iemg)[-1]
    
    tasks = []
    for code in params.movement_codes:
        move = int(code)
        include_task = False
        relevant_emg = []
        relevant_force = []
        for iemg_channel in range(num_iemg):
            if channel2movement[iemg_channel+1]==move:
                include_task = True
                relevant_emg.append(iemg_channel)
                relevant_force.append(iemg_force_correspondance[iemg_channel])
                
        if not include_task:
            continue
                
        
        relevant_inds = np.where(stage == code)[0]
        
        task_emg = iemg[relevant_inds, ...]
        task_force = force[relevant_inds, ...]
        task_cue = cue[relevant_inds, ...]
        
        task_cue = utils.remove_jumps(task_cue)
        task_cue = (task_cue - np.mean(task_cue)) / np.std(task_cue)
        
        task_force = task_force - np.mean(task_force, axis=0)
        
        for i in range(force.shape[1]):
            phase = (force_signs[i]*task_cue > 0.0).astype(np.float32)
            task_force[:, i] = phase * task_force[:, i]
            task_force[:, i] = force_signs[i] * np.maximum(force_signs[i] * task_force[:, i], 0.0)
        
        task = {'code': code,
                'emg': task_emg,
                'force': task_force,
                'cue': task_cue,
                'relevant_emg': relevant_emg,
                'relevant_force': relevant_force}
            
        tasks.append(task)
    
    return tasks

def partition(tasks, emg_win_size, emg_win_step_size, train_part, val_part, test_part):
    X_train = []
    y_train = []
    
    X_val = []
    y_val = []
    
    X_test = []
    y_test = []
    
    for i, task in enumerate(tasks):
        task_len = np.size(task['cue'])
        break_point_1 = int(train_part * task_len)
        break_point_2 = break_point_1 + int(val_part * task_len)
        
        train_inds = list(range(0, break_point_1))#list(range(0, task_len))
        emg_train_t = task['emg'][train_inds,:]
        force_train_t = task['force'][train_inds, :]
        j = emg_win_size
        while j < np.size(train_inds):
            emg_win = emg_train_t[j-emg_win_size:j, :]
            
            X_train.append(emg_win)
            y_train.append(force_train_t[j])
            j += emg_win_step_size
        
        val_inds = list(range(break_point_1, break_point_2))
        emg_val_t = task['emg'][val_inds,:]
        force_val_t = task['force'][val_inds, :]
        j = emg_win_size
        while j < np.size(val_inds):
            emg_win = emg_val_t[j-emg_win_size:j, :]
            
            X_val.append(emg_win)
            y_val.append(force_val_t[j])
            j += emg_win_step_size
        
        test_inds = list(range(break_point_2, task_len))
        emg_test_t = task['emg'][test_inds,:]
        force_test_t = task['force'][test_inds, :]
        j = emg_win_size
        while j < np.size(test_inds):
            emg_win = emg_test_t[j-emg_win_size:j, :]
            
            X_test.append(emg_win)
            y_test.append(force_test_t[j])
            j += 1
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def standardize(X_train, y_train, X_val, y_val, X_test, y_test):
    for channel in range(X_train.shape[-1]):
        mu = np.mean(X_train[:, 0, channel])
        sigma = np.std(X_train[:, 0, channel])
        X_train[:, :, channel] = (X_train[:, :, channel] - mu) / sigma
        X_val[:, :, channel] = (X_val[:, :, channel] - mu) / sigma
        X_test[:, :, channel] = (X_test[:, :, channel] - mu) / sigma
    
    for channel in range(y_train.shape[-1]):
        scale = np.std(y_train[:, channel])
        y_train[:, channel] = y_train[:, channel] / scale
        y_val[:, channel] = y_val[:, channel] / scale
        y_test[:, channel] = y_test[:, channel] / scale
    
    return X_train, y_train, X_val, y_val, X_test, y_test








def preprocess(name, d, emg_win_size, emg_win_step_size, train_part, val_part, test_part):
    iemg, force, cue, stage, iemg_force_correspondance, force_signs = load(name)
    
    iemg, force = filter(iemg, force)
    
    iemg, force, stage, cue = downsample(iemg, force, stage, cue, d)
    
    subject_name = name[0:-4]
    subject = int(subject_name[1::])
    tasks = segment(iemg, force, stage, cue, iemg_force_correspondance, force_signs, subject)
    
    X_train, y_train, X_val, y_val, X_test, y_test = partition(tasks, emg_win_size, emg_win_step_size, train_part, val_part, test_part)
    
    X_train, y_train, X_val, y_val, X_test, y_test = standardize(X_train, y_train, X_val, y_val, X_test, y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, iemg_force_correspondance