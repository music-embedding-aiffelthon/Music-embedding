import jax
import jax.numpy as jnp
import torch.nn.functional as F
import torch
import numpy as np
import random

@jax.jit
def post_norm(X):
    _mean = jnp.mean(X, axis=[0,1,2], keepdims=True)
    _std = jax.lax.clamp(jnp.finfo('float32').eps, jnp.std(X, axis=[0,1,2], keepdims=True), jnp.finfo('float32').max)
    # print(_mean)
    return ((X - _mean) / _std)

def log_mixup_exp(xa, xb, alpha):
    xa = jnp.exp(xa)
    xb = jnp.exp(xb)
    x = alpha * xa + (1. - alpha) * xb
    return np.log(x + jnp.finfo(x.dtype).eps)

class MixupBYOLA:

    def __init__(self, ratio=0.2, n_memory=2048, log_mixup_exp=True):
        self.ratio = ratio
        self.n = n_memory
        self.log_mixup_exp = log_mixup_exp
        self.memory_bank = []

    def __call__(self, x):
        # mix random
        alpha = self.ratio * np.random.random()
        if self.memory_bank:
            # get z as a mixing background sound
            z = self.memory_bank[np.random.randint(len(self.memory_bank))]
            # mix them
            mixed = log_mixup_exp(x, z, 1. - alpha) if self.log_mixup_exp \
                    else alpha * z + (1. - alpha) * x
        else:
            mixed = x
        # update memory bank
        self.memory_bank = (self.memory_bank + [x])[-self.n:]

        return mixed

class RandomResizeCrop:
    """Random Resize Crop block.
    Args:
        virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
        freq_scale: Random frequency range `(min, max)`.
        time_scale: Random time frame range `(min, max)`.
    """

    def __init__(self, virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)):
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = 'bicubic'
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w
    
    
    def __call__(self, lms):
        lms = torch.tensor(lms.reshape(lms.shape[-1], lms.shape[0], lms.shape[1]))
        virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
        virtual_crop_area = (torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]))).to(torch.float)
                             
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y:y+h, x:x+w] = lms
        # get random area
        i, j, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], self.time_scale, self.freq_scale)
        crop = virtual_crop_area[:, i:i+h, j:j+w]
        # print(f'shapes {virtual_crop_area.shape} {crop.shape} -> {lms.shape}')
        lms = F.interpolate(crop.unsqueeze(0), size=lms.shape[-2:],
            mode=self.interpolation, align_corners=True)
                
        return np.expand_dims(np.squeeze(lms.numpy(), axis=(0,1)), axis=-1)

