import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
def gaussian_kl_divergence(mu_q, ln_var_q, mu_p, ln_var_p):
    ln_det_q = torch.sum(ln_var_q, dim=(1, 2, 3))
    ln_det_p = torch.sum(ln_var_p, dim=(1, 2, 3))
    var_p = torch.exp(ln_var_p)
    var_q = torch.exp(ln_var_q)
    tr_qp = torch.sum(var_q / var_p, dim=(1, 2, 3))
    k = mu_q.shape[1] * mu_q.shape[2] * mu_q.shape[3]
    diff = mu_p - mu_q
    term2 = torch.sum(diff * diff / var_p, dim=(1, 2, 3))
    return 0.5 * (tr_qp + term2 - k + ln_det_p - ln_det_q)


def gaussian_negative_log_likelihood(x, mu, var, ln_var):
    k = mu.shape[1] * mu.shape[2] * mu.shape[3]
    diff = x - mu
    return 0.5 * (k * math.log(2 * math.pi) + torch.sum(ln_var + diff * diff / var, dim=(1, 2, 3)))