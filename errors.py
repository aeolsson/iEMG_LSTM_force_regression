import numpy as np
from sklearn.metrics import explained_variance_score


def root_mean_squared(y_true, y_pred):
    diffs = y_true - y_pred
    sq_errors = diffs ** 2
    RMSs = np.sqrt(np.mean(sq_errors, axis=1))
    return np.mean(RMSs)

def variance_accounted_for(y_true, y_pred):
    return explained_variance_score(y_true, y_pred)