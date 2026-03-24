import math
import logging
import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
import torch
import torch.nn as nn
from torch import spmm

def get_norm_inter(inter):
    user_degree = np.array(inter.sum(axis=1)).flatten() # Du
    item_degree = np.array(inter.sum(axis=0)).flatten() # Di
    user_d_inv_sqrt = np.power(user_degree.clip(min=1), -0.5)
    item_d_inv_sqrt = np.power(item_degree.clip(min=1), -0.5)
    user_d_inv_sqrt[user_degree == 0] = 0
    item_d_inv_sqrt[item_degree == 0] = 0
    user_d_inv_sqrt = sp.diags(user_d_inv_sqrt)  # Du^(-0.5)
    item_d_inv_sqrt = sp.diags(item_d_inv_sqrt)  # Di^(-0.5)
    norm_inter = (user_d_inv_sqrt @ inter @ item_d_inv_sqrt).tocoo() # Du^(-0.5) * R * Di^(-0.5)
    return norm_inter # R_tilde

def sparse_coo_tensor(mat):
    # scipy.sparse.coo_matrix -> torch.sparse.coo_tensor
    return torch.sparse_coo_tensor(
        indices=torch.tensor(np.vstack([mat.row, mat.col])),
        values=torch.tensor(mat.data, dtype=torch.float32),
        size=mat.shape)

class Laplacian(nn.Module):
    def __init__(self, inter):
        super().__init__()
        norm_inter = get_norm_inter(inter)
        norm_inter = sparse_coo_tensor(norm_inter)
        self.register_buffer('norm_inter', norm_inter) # shape (num_users, num_items)
        
    def __mul__(self, x):
        # L_tilde = 2L/lambda_max - I
        # = 2 (I - R_tilde^T * R_tilde)/1 - I
        # = R_tilde^T * R_tilde * -2 + I
        y = spmm(self.norm_inter, x)
        y = spmm(self.norm_inter.t(), y) * (-2)
        y += x
        return y
    
class ChebyFilter(nn.Module):
    def __init__(self, order, flatness):
        super().__init__()
        self.order = order
        self.flatness = flatness
        
    def plateau(self):
        x = torch.arange(self.order + 1)
        x = torch.cos((self.order - x) / self.order * math.pi).round(decimals=3)
        output = torch.zeros_like(x)
        output[x<0]  = (-x[x<0]).pow(self.flatness) *  0.5  + 0.5
        output[x>=0] = (x[x>=0]).pow(self.flatness) * (-0.5) + 0.5
        return output.round(decimals=3)

    def cheby(self, x, init):
        if self.order==0: return [init]
        output = [init, x * init]
        for _ in range(2, self.order+1):
            output.append(x * output[-1] * 2 - output[-2])
        return torch.stack(output)
    
    def fit(self, inter):
        # Laplacian_tilde
        self.laplacian = Laplacian(inter) # shape (num_items, num_items)
        
        # Chebyshev Nodes and Target Transfer Function Values
        cheby_nodes = torch.arange(1, (self.order+1)+1)
        cheby_nodes = torch.cos(((self.order+1) + 0.5 - cheby_nodes) / (self.order+1) * math.pi)
        target = self.plateau()
        # Chebyshev Interpolation Coefficients
        coeffs = self.cheby(x=cheby_nodes, init=target).sum(dim=1) * (2/(self.order+1))
        coeffs[0] /= 2
        self.register_buffer('coeffs', coeffs)
    
    def forward(self, signal):
        signal = signal.T  # (num_items, batch_size)

        if self.order == 0:
            return (self.coeffs[0] * signal).T

        # Incremental weighted Chebyshev sum — avoids storing K+1 full tensors simultaneously.
        # Memory: O(3 * N * B) instead of O((K+1) * N * B), ~(K/3)x reduction.
        t0 = signal                       # T_0(L) * signal
        t1 = self.laplacian * signal      # T_1(L) * signal
        output = self.coeffs[0] * t0 + self.coeffs[1] * t1

        for k in range(2, self.order + 1):
            t2 = self.laplacian * t1 * 2 - t0
            output = output + self.coeffs[k] * t2
            t0 = t1
            t1 = t2

        return output.T

class IdealFilter(nn.Module):
    def __init__(self, threshold, weight):
        super().__init__()
        self.threshold = threshold
        self.weight = weight
    
    def fit(self, inter):
        norm_inter = get_norm_inter(inter)
        logging.info(f'[IdealFilter] Computing top-{self.threshold} SVD (randomized) ...')
        _, _, vt = randomized_svd(norm_inter, n_components=self.threshold, random_state=0)
        logging.info(f'[IdealFilter] SVD done.')
        ideal_pass = torch.tensor(vt.T.copy(), dtype=torch.float32)
        self.register_buffer('ideal_pass', ideal_pass) # shape (num_items, threshold)
        
    def forward(self, signal):
        ideal_preds = signal @ self.ideal_pass @ self.ideal_pass.T
        return ideal_preds * self.weight

class DegreeNorm(nn.Module):
    def __init__(self, power):
        super().__init__()
        self.power = power
    
    def fit(self, inter):
        item_degree = torch.tensor(np.array(inter.sum(axis=0)).flatten())
        zero_mask = (item_degree == 0)
        pre_norm = item_degree.clamp(min=1).pow(-self.power)
        pst_norm = item_degree.clamp(min=1).pow(+self.power)
        pre_norm[zero_mask], pst_norm[zero_mask] = 0, 0
        self.register_buffer('pre_normalize', pre_norm)  # (num_items,)
        self.register_buffer('post_normalize', pst_norm) # (num_items,)
        
    def forward_pre(self, signal):
        return signal * self.pre_normalize
    
    def forward_post(self, signal):
        return signal * self.post_normalize

class LinearFilter(nn.Module):
    def __init__(self):
        super().__init__()
        
    def fit(self, inter):
        norm_inter = get_norm_inter(inter)
        norm_inter = sparse_coo_tensor(norm_inter)
        self.register_buffer('norm_inter', norm_inter) # shape (num_users, num_items)
        
    def forward(self, signal):
        # I - L = R_tilde^T * R_tilde
        signal = signal.T
        output = spmm(self.norm_inter.t(), spmm(self.norm_inter, signal))
        return output.T