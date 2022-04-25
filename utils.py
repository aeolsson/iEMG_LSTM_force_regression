import numpy as np

def remove_jumps(win, thres=0.001):
    tmp = np.concatenate((np.zeros((1,)), win), axis=0)
    diffs = np.diff(tmp, n=1, axis=0)
    jump_inds = np.where(np.abs(diffs) > thres)[0]
    
    new_win = np.copy(win)
    
    for jump_ind in jump_inds:
        new_win[jump_ind::] = new_win[jump_ind::] - diffs[jump_ind]
        
    return new_win