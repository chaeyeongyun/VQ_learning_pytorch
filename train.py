import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from PIL import Image
import os
from tqdm import tqdm


from utils.load_config import get_config_from_yaml
from utils.logging import Logger
from utils.env_utils import device_setting
from utils.lr_schedulers import WarmUpPolyLR
import wandb
from models import Network
from dataset import FolderDataset

def train(cfg):
    if cfg.wandb_logging:
        logger_name = cfg.project_name+str(len(os.listdir(cfg.train.save_dir)))
        save_dir = os.path.join(cfg.train.save_dir, logger_name)
        os.makedirs(save_dir)
        ckpoints_dir = os.path.join(save_dir, 'ckpoints')
        os.mkdir(ckpoints_dir)
        log_txt = open(os.path.join(save_dir, 'log_txt'), 'w')
    logger = Logger(cfg, logger_name) if cfg.wandb_logging else None
    
    half=cfg.train.half
    if logger!=None:wandb.config.update(cfg.train)
    batch_size = cfg.train.batch_size
    num_epochs: cfg.train.num_epochs
    device = device_setting(cfg.train.device)
    model = Network(backbone=cfg.backbone, vq_cfg=cfg.vector_quantizer)
    
    dataset = FolderDataset(cfg.data_dir)
    dataloader= DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lr_scheduler = WarmUpPolyLR(cfg.train.learning_rate, lr_power=cfg.train.lr_scheduler.lr_power, 
                                total_iters=len(dataloader)*num_epochs,
                                warmup_steps=len(dataloader)*cfg.train.lr_scheduler.warmup_epoch)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.999))
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(range(len(dataloader)))
        for batch_idx, input in zip(pbar, dataloader):
            input = input.to(device)
            
            optimizer.zero_grad()
            output, commitment_loss = model(input)
            #TODO: 여기부터 traincode~