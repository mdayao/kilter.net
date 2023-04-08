import os
import argparse
import numpy as np
from pathlib import Path
from experiment import VAEXperiment
from dataset import KilterDataModule
from models.cvae import ConditionalVAE as CVAE

import torch.backends.cudnn as cudnn

import lightning.pytorch as pl
from lightning.pytorch import Trainer

from lightning.pytorch.loggers import WandbLogger
import wandb


parser = argparse.ArgumentParser(description='Running kilter.net CVAE')

parser.add_argument('--latent-dim','-L', type=int, default=32, help='Latent dimension')
parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch-size', '-bs', type=int, default=128, help='Batch size')
parser.add_argument('--kld-weight', '-kld', type=float, default=2.5e-4, help='KL divergence weight')

parser.add_argument('--random-seed', type=int, default=9473, help='random seed')
parser.add_argument('--max-epochs', type=int, default=50, help='max number of epochs to run')

parser.add_argument('--board-path', type=str, default='training_data/kilter_climb_features.npy', help='path to board data array')
parser.add_argument('--vgrade-path', '-vg', type=str, default='training_data/kilter_vgrades.npy', help='path to vgrade array')
parser.add_argument('--angle-path', type=str, default='training_data/kilter_angles.npy', help='path to board angle array')
parser.add_argument('--sends-path', type=str, default='training_data/kilter_ascents.npy', help='path to board sends array')

parser.add_argument('--num-workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--pin-memory', action='store_true', help='set this to pin memory for DataLoader')

args = parser.parse_args()
config = vars(args)

WANDB_PROJECT = "cvae"
WANDB_ENTITY = "cvae"
WANDB_NAME = "cvae"
#wandb_logger = WandbLogger(project=WANDB_PROJECT)

model = CVAE(in_channels=4,
             latent_dim=args.latent_dim, 
             )

experiment = VAEXperiment(model,
                          params = {
                              'kld_weight': args.kld_weight,
                              'LR': args.learning_rate
                              })

datamodule = KilterDataModule(
        board_path=args.board_path,
        v_path=args.vgrade_path,
        angle_path=args.angle_path,
        sends_path=args.sends_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        random_seed=args.random_seed
        )
datamodule.setup()

trainer = Trainer(
                 #logger=wandb_logger, 
                 max_epochs=args.max_epochs,
                 accelerator='cpu',
                 devices='auto'
                 )

#Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
#Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training CVAE =======")
trainer.fit(experiment, datamodule=datamodule)

