import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from trainer import *

from utils import augment
from utils.augment import *
from utils.config_hook import yaml_config_hook
from utils.dataloader import mel_dataset 


mix = MixupBYOLA()
crop = RandomResizeCrop()


config = yaml_config_hook(os.path.join('config', 'config.yaml'))



def collate_batch(batch):
    x_train_1 = []
    x_train_2 = []
    
    for x, y in batch:
        x = (np.array(x)+127)/100
        x = np.expand_dims(x, axis=-1)
        x = crop(mix(x))        
        x_train_1.append(x)
        
    for x, y in batch:
        x = (np.array(x)+127)/100
        x = np.expand_dims(x, axis=-1)
        x = crop(mix(x))        
        x_train_2.append(x)
            
    y_train = [y for _, y in batch]           
    
    return augment.post_norm(np.stack(x_train_1 + x_train_2, axis=0)), np.array(y_train)


def eval_collate_batch(batch):
    x_train = [(np.array(x)+127)/100 for x, _ in batch]
    y_train = [y for _, y in batch]                  
        
    return np.expand_dims(np.array(x_train),axis=-1), np.array(y_train)


def pretrain(num_epochs, config):
    if config['pretrain_model'] == 'SimCLR':
        trainer = SimCLRTrainer(exmp=jnp.ones((config['pretrain_batch_size'],48,1876,1)), 
                                **config['SimCLR_option'])
        
    elif config['pretrain_model'] == 'BYOL':
        trainer = BYOLTrainer(exmp=jnp.ones((config['pretrain_batch_size'],48,1876,1)),
                                **config['BYOL_option'])
        
    elif config['pretrain_model'] == 'Conv2d_AE':
        trainer = Conv2d_AETrainer(exmp=jnp.ones((config['pretrain_batch_size'],48,1876,1)), 
                                **config['Conv2d_AE_option'])
    
    if not config['pretrain']:
        trainer.load_model()
    else:
        if not trainer.checkpoint_exists():                             
            trainer.train_model(train_dataloader, test_dataloader, num_epochs=num_epochs)
        else:
            trainer.load_model()
            trainer.train_model(train_dataloader, test_dataloader, num_epochs=num_epochs)
        
    return trainer


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
        
def prepare_data_features(encode_fn, data):
    # Encode all data
    dataset = DataLoader(data, batch_size=config['pretrain_batch_size'],
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=0,
                                  collate_fn=collate_batch)
    
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(dataset):
        batch_feats = encode_fn(batch_imgs)
        feats.append(jax.device_get(batch_feats))
        labels.append(batch_labels)

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)

    return NumpyDataset(feats, labels)

def prepare_data_label(encode_fn, data):
    # Encode all images to label
    dataset = DataLoader(data, batch_size=128,
                                  shuffle=False,
                                  drop_last=True,
                                  num_workers=0,
                                  collate_fn=numpy_collate)
    
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(dataset):
        batch_feats = encode_fn(batch_imgs)
        feats.append(jax.device_get(batch_feats))
        labels.append(batch_labels)

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)

    return NumpyDataset(feats, labels)
    
class NumpyDataset(mel_dataset):
    # data.TensorDataset for numpy arrays

    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self):
        return self.arrays[0].shape[0]

    def __getitem__(self, idx):
        return [arr[idx] for arr in self.arrays]

    
    
def linear_evaluation(train_feats_data, test_feats_data):
    # Data loaders
    train_loader = DataLoader(train_feats_data,
                                   batch_size=config['linear_batch_size'],
                                   shuffle=True,
                                   drop_last=True,
                                   generator=torch.Generator().manual_seed(42),
                                   collate_fn=numpy_collate)
    
    test_loader = DataLoader(test_feats_data,
                                  batch_size=config['linear_batch_size'],
                                  shuffle=False,
                                  drop_last=False,
                                  collate_fn=numpy_collate)

    trainer = LGTrainer(exmp=next(iter(train_loader))[0],
                        num_classes=config['num_classes'],
                        lr=1e-3)
    
    trainer.train_model(train_loader, test_loader, num_epochs=num_epochs)

    return trainer    
    
