
from typing import Dict
import torch

def clone_state_dict(sd: Dict[str, torch.Tensor]):
    return {k: v.detach().clone() for k, v in sd.items()}

def state_dict_interp_inplace(theta, local, alpha):
    for k in theta.keys():
        theta[k] = (1 - alpha)*theta[k] + alpha*local[k]

def fedavg_aggregate(local_thetas, weights, theta_out):
    total = sum(weights)
    for k in theta_out.keys():
        theta_out[k].zero_()
        for th, w in zip(local_thetas, weights):
            theta_out[k] += (w/total) * th[k]

def fedasync_update(theta, theta_local, time_since_pull_sec, alpha_base,
                    eps, To_max_min, a):
    To_max_sec = To_max_min * 60.0
    if time_since_pull_sec <= (1.0 + eps)*To_max_sec:
        s = 1.0
    else:
        s = 1.0 / (1.0 + a*(time_since_pull_sec - (1.0+eps)*To_max_sec)/To_max_sec)
    alpha = alpha_base * s
    state_dict_interp_inplace(theta, theta_local, alpha)
    return alpha, s

def fedsat_update(theta, theta_prevpull, theta_local, nk, n_total):
    alpha_k = nk / float(n_total)
    for k in theta.keys():
        theta[k] = theta[k] - alpha_k*(theta_prevpull[k] - theta_local[k])
    return alpha_k
