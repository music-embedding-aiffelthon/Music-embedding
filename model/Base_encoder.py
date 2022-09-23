# reference : https://github.com/andrebola/contrastive-mir-learning

import jax
import jax.numpy as jnp
from jax import random
import optax

import flaxmodels
from flax import linen as nn
from typing import Callable, Any, Optional, Text


class fc_audio(nn.Module):
    size_w_rep: int = 512
    train: bool = True    
    @nn.compact
    def __call__(self, x):
        x = jax.nn.relu(nn.Dense(512)(x))
        x = nn.Dropout(rate=0.5)(x, deterministic=not self.train)
        x = nn.Dense(self.size_w_rep)(x)
        x = nn.LayerNorm()(x)
        
        return x   
    
class SimCLR(nn.Module):
    
    size_w_rep: int = 512    
    train: bool = True    
    temperature : float = 0.07
    project_head: int = 1
    def setup(self):    
        self.convnet = flaxmodels.ResNet18(pretrained=False,
                               normalize=False,
                               num_classes=self.size_w_rep)
        
        self.project = nn.Sequential([nn.Dense(self.size_w_rep) for i in range(self.project_head)])
        self.fc_audio = fc_audio(self.size_w_rep, self.train)

    def __call__(self, x):
        # x = self.audio_encoder(x)
        x = self.convnet(x)
        x = self.project(x)
        feats = self.fc_audio(x)
        
        cos_sim = optax.cosine_similarity(feats[:,None,:], feats[None,:,:])
        return_cos_sim = cos_sim
        cos_sim /= self.temperature
        diag_range = jnp.arange(feats.shape[0], dtype=jnp.int32)
        cos_sim = cos_sim.at[diag_range, diag_range].set(-9e15)
        
        shifted_diag = jnp.roll(diag_range, x.shape[0]//2)
        pos_logits = cos_sim[diag_range, shifted_diag]
        
        # InfoNCE loss
        nll = - pos_logits + nn.logsumexp(cos_sim, axis=-1)
        nll = nll.mean()

        # Logging
        metrics = {'loss': nll}
        # Determine ranking position of positive example
        comb_sim = jnp.concatenate([pos_logits[:,None],
                                    cos_sim.at[shifted_diag, diag_range].set(-9e15)],
                                   axis=-1)
        sim_argsort = (-comb_sim).argsort(axis=-1).argmin(axis=-1)
        
        # Logging of ranking position
        metrics['acc_top1'] = (sim_argsort == 0).mean()
        metrics['acc_top5'] = (sim_argsort < 5).mean()
        metrics['acc_mean_pos'] = (sim_argsort + 1).mean()

        return nll, metrics, return_cos_sim
    
    def encode(self, x):
        x = self.convnet(x)
        return x 

    
# https://github.com/deepmind/deepmind-research/blob/6fcb84268e74af981ae1496bfc2cb9ba9d701ef2/byol/byol_experiment.py    
    
def l2_normalize(x: jnp.ndarray,
                axis: Optional[int] = None,
                epsilon: float = 1e-12,
                ) -> jnp.ndarray:
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
    return x * x_inv_norm    

# --- BYOL ---


def regression_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Byol's regression loss. This is a simple cosine similarity."""
    normed_x, normed_y = l2_normalize(x, axis=-1), l2_normalize(y, axis=-1)
    return jnp.sum((normed_x - normed_y)**2, axis=-1)


class MLP_Block(nn.Module):
    hidden_size: int = 4096
    size_w_rep: int = 512
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.normalization.BatchNorm(use_running_average=True)(x)
        x = jax.nn.relu(x)
        x = nn.Dense(self.size_w_rep)(x)     
        return x 

    
class BYOL(nn.Module):
    hidden_size: int = 4096    
    size_w_rep: int = 512    
    train: bool = True        
    project_head:int = 1
    predict_head:int = 1
    def setup(self):    
        self.online_rep = flaxmodels.ResNet18(pretrained=False,
                               normalize=False,
                               num_classes=self.size_w_rep)
        
        self.target_rep = flaxmodels.ResNet18(pretrained=False,
                               normalize=False,
                               num_classes=self.size_w_rep)
        

        self.online_pro = nn.Sequential([
            MLP_Block(size_w_rep=self.size_w_rep,
                     hidden_size=self.hidden_size) for i in range(self.project_head)
                            ])
        
        self.target_pro = nn.Sequential([
            MLP_Block(size_w_rep=self.size_w_rep,
                     hidden_size=self.hidden_size) for i in range(self.project_head)
                            ])

        self.predict_layer = nn.Sequential([
            MLP_Block(size_w_rep=self.size_w_rep,
                     hidden_size=self.hidden_size) for i in range(self.predict_head)
                            ])

    def __call__(self, x):
        batch_1 = x[:x.shape[0]//2,:,:,:]
        batch_2 = x[x.shape[0]//2:,:,:,:]

        online_1 = self.online_pro(self.online_rep(batch_1))
        online_1 = self.predict_layer(online_1)         
        target_2 = self.target_pro(self.target_rep(batch_2))
        
        loss = regression_loss(online_1, jax.lax.stop_gradient(target_2))  
        
        online_2 = self.online_pro(self.online_rep(batch_2))
        online_2 = self.predict_layer(online_2)         
        target_1 = self.target_pro(self.target_rep(batch_1))
        
        loss += regression_loss(online_2, jax.lax.stop_gradient(target_1))
        metrics = {'loss':jnp.mean(loss)}
        return jnp.mean(loss), metrics
    
    def encode(self, x):
        x = self.online_rep(x)
        return x 
    
class Encoder(nn.Module):
    
    dilation:bool=False
    latent_size:int=512
    hidden_layer:int=512
    
    @nn.compact
    def __call__(self, x):
        
        # 0 
        if self.dilation:
            x = nn.Conv(512, kernel_size=(3,3),  strides=[2,2], kernel_dilation=1, padding='same')(x)
        else:
            x = nn.Conv(512, kernel_size=(3,3),  strides=[2,2], padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)

        # 1
        if self.dilation:
            x = nn.Conv(512,kernel_size=(3,3), kernel_dilation=1, padding='same')(x)
        else:
            x = nn.Conv(512,kernel_size=(3,3),  padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        # 2 
        if self.dilation:
            x = nn.Conv(256,kernel_size=(3,3), kernel_dilation=2, padding='same')(x)
        else:            
            x = nn.Conv(256,kernel_size=(3,3),  padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
 
        # 3
        if self.dilation:
            x = nn.Conv(128,kernel_size=(3,3), kernel_dilation=2, padding='same')(x)
        else:
            x = nn.Conv(128,kernel_size=(3,3), padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        # 4
        if self.dilation:
            x = nn.Conv(64, kernel_size=(3,3), kernel_dilation=4, padding='same')(x)
        else:
            x = nn.Conv(64,kernel_size=(3,3), padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        # 5
        if self.dilation:
            x = nn.Conv(32, kernel_size=(3,3), kernel_dilation=4, padding='same')(x)
        else:
            x = nn.Conv(32, kernel_size=(3,3),  padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        # 6
        if self.dilation:
            x = nn.Conv(16, kernel_size=(3,3), kernel_dilation=4, padding='same')(x)
        else:
            x = nn.Conv(16, kernel_size=(3,3), padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        # 7
        if self.dilation:
            x = nn.Conv(1,kernel_size=(3,3), strides=[1,1], kernel_dilation=4, padding='same')(x)
        else:
            x = nn.Conv(1,kernel_size=(3,3), strides=[1,1],  padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)

        
        x = x.reshape(x.shape[0], -1)         
        z = nn.Dense(features=self.latent_size, name='latent_vector')(x)                
        return z 
    
    
class Decoder(nn.Module):
    
    dilation:bool=False
    latent_size:int=512
    
    @nn.compact
    def __call__(self, x):
        
        x = nn.Dense(12 * 469 * 1)(x)
        x = x.reshape(x.shape[0], 12, 469, 1)
        
    
        # 0
        if self.dilation:
            x = nn.ConvTranspose(32, kernel_size=(3,3), strides=[1,1], kernel_dilation=(4,4))(x)
        else:
            x = nn.ConvTranspose(32, kernel_size=(3,3), strides=[1,1])(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        # 1
        if self.dilation:
            x = nn.ConvTranspose(64, kernel_size=(3,3))(x)
        else:
            x = nn.ConvTranspose(64, kernel_size=(3,3), strides=[1,1],kernel_dilation=(2,2))(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)        
        
        # 2
        if self.dilation:
            x = nn.ConvTranspose(128, kernel_size=(3,3), strides=[2,2], kernel_dilation=(2,2))(x)
        else:             
            x = nn.ConvTranspose(128, kernel_size=(3,3), strides=[2,2])(x)                   
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        
        # 3
        if self.dilation:
            x = nn.ConvTranspose(256, kernel_size=(3,3), strides=[2,2], kernel_dilation=(2,2))(x)
        else:
            x = nn.ConvTranspose(256, kernel_size=(3,3), strides=[2,2])(x)
            
        x = jax.nn.leaky_relu(x)
        
        
        x = nn.ConvTranspose(1, kernel_size=(3,3), strides=[1,1])(x)
        x = jax.nn.tanh(x)
        x = jnp.squeeze(x, axis=-1)
        return x
        

    
    
    
class Conv2d_AE(nn.Module):
    dilation:bool=False
    latent_size:int=512
    
    def setup(self):
        self.encoder = Encoder(dilation=self.dilation, 
                               latent_size=self.latent_size)
        self.decoder = Decoder(dilation=self.dilation, latent_size=self.latent_size)

    def __call__(self, x):
     
        z = self.encoder(x)
        recon_x = self.decoder(z)
        recon_x = jnp.expand_dims(recon_x, axis=-1)
        return recon_x
    
    def encode(self, x):
        x = self.encoder(x)
        return x