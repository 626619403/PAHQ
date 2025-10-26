import torch
from torch.nn.parameter import Parameter

def quantize_weight_on_cpu(weight, weight_cache):
    # weight must be 3d tensor
    weight_high = weight_cache[:,:,target_h].to(weight_high.device)
    return weight_low1, weight_high, weight_low2

def quantize_weight_on_cuda(weight_low1, weight_high, weight_low2, weight_cache, h=32, target_h=0):
    # weight must be 3d tensor
    weight_high = weight_cache[:,:,target_h].to(weight_high.device)
    return weight_low1, weight_high, weight_low2

def weight_arrd(weight, h=32, target_h=0):
    d1, d2 = weight.shape
    weight = weight.reshape(d1, d2 // h, h)
    weight_low1, weight_high, weight_low2 = weight[:,:,:target_h], weight[:,:,target_h], weight[:,:,target_h+1:]
    return weight_low1.reshape(d1, d2//h*target_h), weight_high.reshape(d1, d2//h), weight_low2.reshape(d1, d2-d2//h*(target_h+1))

def mix_percision_matmul(x, weight_low1, weight_high, weight_low2):
    def _mix_percision_matmul(x, weight):
        x = x.to(weight.dtype)
        x = torch.matmul(x, weight)
    x_low1, x_high, x_low2 = x[:,:,:weight_low1.shape[-1]], x[:,:,weight_low1.shape[-1]:-weight_low2.shape[-1]], x[:,:,-weight_low2.shape[-1]:]
    x_low1 = _mix_percision_matmul(x_low1, weight_low1)
    x_high = torch.matmul(x_high, weight_high)
    x_low2 = _mix_percision_matmul(x_low2, weight_low2)
    return x_low1, x_high, x_low2


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, h=32, d=512):
        super().__init__()
        self.weight_qkv = Parameter(torch.empty((d, d*3), device=torch.device("cpu")))
        self.out = torch.nn.Linear(d, d)
        self.h = h
        self.d = d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape[:2]
        weight_qkv_device = self.weight_qkv.to(x.device)
        weight_qkv_device_low1, weight_qkv_device_high, weight_qkv_device_low2 = \
            quantize_weight_on_cuda(weight_qkv_device.t())
        qkv_low1, qkv_high, qkv_low2 = mix_percision_matmul(
            x, weight_qkv_device_low1, weight_qkv_device_high, weight_qkv_device_low2)
        # q, k, v = q.reshape(B, N, self.h, -1), k.reshape(B, N, self.h, -1), v.reshape(B, N, self.h, -1)

        q, k, v = q / (self.d ** 0.5), k, v
        qk = torch.einsum('bqd,bkd->bqk', q, k)
        s = torch.nn.functional.softmax(qk, dim=-1)
        y = torch.einsum('bqk,bkd->bqd', s, v)
        return self.out(y)