# 2022-09-23 14:10 Seoul
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial17/SimCLR.html


import os
import wandb
from tqdm import tqdm


import jax
import optax
import numpy as np
import jax.numpy as jnp


from functools import partial
from typing import Sequence, Any
from collections import defaultdict
from utils.config_hook import yaml_config_hook


import flax 
import flax.linen as nn
from flax import jax_utils
from flax.core import freeze, unfreeze
from flax.training import train_state, common_utils, checkpoints

from model.Base_encoder import *




# --- Define config ---
config = yaml_config_hook(os.path.join('config', 'config.yaml'))



# --- top_k accuarcy ---

@partial(jax.jit, static_argnames=['k'])
def top_k(logits, y,k):
    top_k = jax.lax.top_k(logits, k)[1]
    ts = jnp.argmax(y, axis=1)
    correct = 0
    for i in range(ts.shape[0]):
        b = (jnp.where(top_k[i,:] == ts[i], jnp.ones((top_k[i,:].shape)), 0)).sum()
        correct += b
    correct /= ts.shape[0]
    return correct 



# --- Base trainer module ---

class TrainState(train_state.TrainState):
    batch_stats : Any
    rng : Any

class TrainerModule:

    def __init__(self,
                 model_name : str,
                 model_class : Any,
                 eval_key : str,
                 exmp : Any,
                 lr : float = 5e-4,
                 weight_decay : float = 0.01,
                 check_val_every_n_epoch : int = 1,
                 seed : int = 33,
                **model_hparams):

        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.eval_key = eval_key
        self.model_name = model_name
        self.model = model_class(**model_hparams)
        self.log_dir = os.path.join(config['checkpoint_dir'], self.model_name)
        self.create_functions()
        self.init_model(exmp)

    def create_functions(self):
        raise NotImplementedError

    def init_model(self, exmp):
        # Initialize model
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)
        variables = self.model.init({'params':init_rng,'dropout':init_rng}, exmp)
        
        self.state = TrainState(step=0,
                                apply_fn=self.model.apply,
                                params=unfreeze(variables['params']),
                                batch_stats=variables.get('batch_stats'),
                                rng=rng,
                                tx=None, opt_state=None)
        

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr,
            warmup_steps=0.0,
            decay_steps=int(num_epochs * num_steps_per_epoch),
            end_value=2e-2*self.lr
        )
        optimizer = optax.adamw(lr_schedule, weight_decay=self.weight_decay)
        
        self.create_train_state(optimizer)

    def create_train_state(self, optimizer):
        self.state = TrainState.create(apply_fn=self.state.apply_fn,
                                       params=self.state.params,
                                       batch_stats=self.state.batch_stats,
                                       rng=self.state.rng,
                                       tx=optimizer)

    def train_model(self, train_loader, test_loader, num_epochs):
        
        self.init_optimizer(num_epochs, len(train_loader))
        best_eval = 100
        
        for epoch_idx in range(1, num_epochs+1):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(test_loader, epoch=epoch_idx)
                if eval_metrics[self.eval_key] <= best_eval:
                    best_eval = eval_metrics[self.eval_key]
                    self.save_model(step=epoch_idx)
                    
    def train_epoch(self, train_dataloader, epoch):
        metrics = defaultdict(float)
        
        train_iter = iter(train_dataloader)
        for n in tqdm(range(len(train_dataloader)), desc=f'Epoch {epoch}'):            
            self.state, train_metrics = self.train_step(self.state, next(train_iter))  
            for key in train_metrics:
                train_key = 'train_' + key
                metrics[train_key] += train_metrics[key]
        num_train_steps = len(train_dataloader)
        metrics = {k:(v/num_train_steps) for k,v in metrics.items()}
        wandb.log(metrics, step=epoch)

    def eval_model(self, data_loader, epoch):
        metrics = defaultdict(float)
        count = 0
        for batch_idx, batch in tqdm(enumerate(data_loader), desc=f'Eval step'):
            batch_metrics = self.eval_step(self.state, jax.random.PRNGKey(batch_idx), batch)
            count += 1
            for key in batch_metrics:
                eval_key = 'eval_' + key
                metrics[eval_key] += batch_metrics[key]
                
        metrics = {k:(v/count) for k,v in metrics.items()}
        wandb.log(metrics, step=epoch)

        return metrics
    
    
                
    def save_model(self, step=0):
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.state.params,
                                            'batch_stats': self.state.batch_stats},
                                    step=step,
                                    overwrite=True)

    def load_model(self, pretrained=False):
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        num_params = sum([np.prod(p.shape) for p in jax.tree_util.tree_leaves(state_dict)])
        self.state = TrainState.create(apply_fn=self.state.apply_fn,
                                       params=unfreeze(state_dict['params']),
                                       batch_stats=unfreeze(state_dict['batch_stats']),
                                       rng=self.state.rng,
                                       tx=self.state.tx if self.state.tx else optax.sgd(self.lr)  
                                      )

    def checkpoint_exists(self):
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}'))



# --- SimCLR Trainer ---

class SimCLRTrainer(TrainerModule):

    def __init__(self, **kwargs):
        super().__init__(model_class=SimCLR,
                         model_name='SimCLR',
                         eval_key='eval_acc_top1',
                         **kwargs)

    def create_functions(self):
        
        # Training function
        def train_step(state, batch):
            batch, _ = batch
            rng, forward_rng = jax.random.split(state.rng)
            
            def loss_fn(params):
                outs = self.model.apply({'params': params, 'batch_stats': state.batch_stats},
                                    batch, 
                                    rngs={"dropout": rng},
                                    mutable=['batch_stats'])
                (loss, metrics, cos_sim), new_model_state = outs
                return loss, (metrics, new_model_state, cos_sim)
            
            
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (_, (metrics, new_model_state, cos_sim)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads,
                                          batch_stats=new_model_state['batch_stats'],
                                          rng=rng)
            return state, metrics                     
            
        # Eval function
        def eval_step(state, rng, batch): 
            batch, _ = batch
            rng, forward_rng = jax.random.split(state.rng)         
            
            outs = self.model.apply({'params': state.params, 'batch_stats': state.batch_stats},
                                    batch, 
                                    rngs={"dropout": rng},
                                    mutable=['batch_stats'])
            (loss, metrics, cos_sim), new_model_state = outs
            return metrics
        
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

        

# --- BYOL Trainer ---        
        
class BYOLTrainer(TrainerModule):

    def __init__(self, **kwargs):
        super().__init__(model_class=BYOL,
                         model_name='BYOL',                         
                         eval_key='eval_loss',
                         check_val_every_n_epoch=1,                        
                         **kwargs)

    def create_functions(self):
        
        # Training function
        def train_step(state, batch):
            batch, _ = batch            
            rng, forward_rng = jax.random.split(state.rng)
            
            def loss_fn(params):
                outs = self.model.apply({'params': params, 'batch_stats': state.batch_stats},
                                    batch, 
                                    rngs={"dropout": rng},
                                    mutable=['batch_stats'])
                (loss, metrics), new_model_state = outs
                return loss, (metrics, new_model_state)
            
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (_, (metrics, new_model_state)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state.params['target_rep'] = jax.tree_map(lambda x, y: x + (1 - 0.99) * (y - x),
                                      state.params['target_rep'], state.params['online_rep'])
            state.params['target_pro'] = jax.tree_map(lambda x, y: x + (1 - 0.99) * (y - x),
                                      state.params['target_pro'], state.params['online_pro'])
            return state, metrics
        
        # Eval function
        def eval_step(state, rng, batch):
            batch, _ = batch                        
            outs = self.model.apply({'params': state.params, 'batch_stats': state.batch_stats},
                                    batch, 
                                    rngs={"dropout": rng},
                                    mutable=['batch_stats'])
            (loss, metrics), new_model_state = outs
            return metrics
        
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)
        


# --- Conv2d_AE trainer ---        
        
class Conv2d_AETrainer(TrainerModule):

    def __init__(self, **kwargs):
        super().__init__(model_class=Conv2d_AE,
                         model_name='BYOL_A',                         
                         eval_key='eval_loss',
                         check_val_every_n_epoch=1,                        
                         **kwargs)

    def create_functions(self):
        
        # Training function        
        def train_step(state, batch):
            batch, _ = batch
            
            def loss_fn(params):
                recon_x = self.model.apply({'params':params, 'batch_stats':state.batch_stats}, batch)
                loss = ((recon_x - batch) ** 2).mean()
                return loss
            
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)  
            metrics = {'loss' : loss}            
            return state, metrics
        
        # Eval function        
        def eval_step(state, rng, batch):
            batch, _ = batch
            recon_x = self.model.apply({'params':state.params, 'batch_stats':state.batch_stats}, batch)
            loss = ((recon_x - batch) ** 2).mean()
            metrics = {'loss' : loss}
            return metrics                   
        
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)          
        

# --- linear evaluation Trainer ---

class LogisticRegression(nn.Module):
    num_classes : int
    def setup(self):
        self.dense = nn.Dense(self.num_classes)
    
    def __call__(self,x):
        return self.dense(x)
    
    def encode(self,x):
        return self.dense(x)

    
class LGTrainer(TrainerModule):

    def __init__(self, **kwargs):
        super().__init__(model_name='LogisticRegression',
                         model_class=LogisticRegression,
                         eval_key='eval_linear_acc',
                         check_val_every_n_epoch=1,
                         **kwargs)

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        optimizer = optax.adam(self.lr)
        self.create_train_state(optimizer)

    def create_functions(self):
        def calculate_loss(params, batch):
            imgs, labels = batch
            logits = self.model.apply({'params': params}, imgs)
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels.argmax(axis=-1)).mean()
            top_5_accuracy = top_k(logits, labels, 5)
            metrics = {'linear_loss': loss, 'linear_acc': acc, 'linear_top_5_accuracy': top_5_accuracy}
            return loss, metrics
        
        # Training function
        def train_step(state, batch):
            (_, metrics), grads  = jax.value_and_grad(calculate_loss,has_aux=True)(state.params, batch)
            state = state.apply_gradients(grads=grads)
            return state, metrics
        
        # Eval function
        def eval_step(state, rng, batch):
            _, metrics = calculate_loss(state.params, batch)
            return metrics
        
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

        
