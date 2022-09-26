import jax
import jax.numpy as jnp

import os
import random
import numpy as np
import wandb
import util
from util import *

import flax.linen as nn

from torch.utils.data import DataLoader, random_split

from utils.config_hook import yaml_config_hook
from utils.dataloader import mel_dataset # dataloader for kakao mel dataset. https://arena.kakao.com/c/8


# --- please read README.md carefully. ---
if not input('\n\nDid you check your config file and dataset path?\nType enter to continue.'):
    pass
else: 
    exit()

# --- Define config ---
config = yaml_config_hook(os.path.join('config', 'config.yaml'))

# --- initialize wandb --- 
wandb.init(project=config['pretrain_model'], config=config)


    
if __name__=='__main__':
    
    data = mel_dataset(config['dataset_dir'], 'total')
    
    # split data
    
    dataset_size = len(data)
    train_size = int(dataset_size * config['split_ratio'])    
    test_size = dataset_size - train_size
    train_dataset, test_dataset, = random_split(data, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['pretrain_batch_size'], shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=config['pretrain_batch_size'], shuffle=True, num_workers=0, collate_fn=collate_batch)        
    
    
####  You can set label fraction size with scripts below.
    
#     linear_train_size = int(train_size * 0.8)
#     linear_test_size = train_size - linear_train_size
#    
#     linear_train_dataset, linear_test_dataset = random_split(train_dataset,[linear_train_size,linear_test_size])
    
    
    
# --------------------------------------------------------------------
#    ______           _       _                __                
#   /_  __/________ _(_)___  (_)___  ____ _   / /___  ____  ____ 
#    / / / ___/ __ `/ / __ \/ / __ \/ __ `/  / / __ \/ __ \/ __ \
#   / / / /  / /_/ / / / / / / / / / /_/ /  / / /_/ / /_/ / /_/ /
#  /_/ /_/   \__,_/_/_/ /_/_/_/ /_/\__, /  /_/\____/\____/ .___/ 
#                                 /____/                /_/       
# --------------------------------------------------------------------
        
    
    pretrain_trainer = pretrain(num_epochs=config['pretrain_epoch'], config=config)
    
    # --- bind model ---
    
    pretrain_model = pretrain_trainer.model.bind({'params': pretrain_trainer.state.params,
                                          'batch_stats': pretrain_trainer.state.batch_stats},
                                        mutable=['batch_stats'])
    
    encode_fn = jax.jit(lambda img: pretrain_model.encode(img))                
    train_feats_pretrain = prepare_data_features(encode_fn, train_dataset)
    test_feats_pretrain = prepare_data_features(encode_fn, test_dataset)
    
    feats_array = [(x, y) for x,y in train_feats_pretrain]
    np.save(f'{config['pretrain_model']}_feats_array.npy', feats_array)    
    
    
    # --- linear evaluation ---
    
    trainer = linear_evaluation(train_feats_data=train_feats_pretrain,
                               test_feats_data=test_feats_pretrain)
    
    trainer_model = trainer.model.bind({'params':trainer.state.params})
    linear_encode_fn = jax.jit(lambda img: trainer_model.encode(img))
    
    linear_train_feats = prepare_data_label(linear_encode_fn, train_feats_pretrain)
    linear_array = [(x, y) for x,y in linear_train_feats]
    np.save(f'{config['pretrain_model']}_linear_array.npy', linear_array)
