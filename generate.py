import os
import argparse
import numpy as np
from pathlib import Path
from experiment import VAEXperiment
from dataset import KilterDataModule
from models.cvae import ConditionalVAE as CVAE
from visualization_utils import plot_climb
from matplotlib import pyplot as plt

import torch.backends.cudnn as cudnn

import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer

from lightning.pytorch.loggers import WandbLogger


parser = argparse.ArgumentParser(description='Running kilter.net CVAE')

parser.add_argument('--path_to_checkpoint', help='path to saved model checkpoint')
parser.add_argument('--angle','-A', default=40, type=int, help='angle of the wall')
parser.add_argument('--grade','-V', default=1,type=int, help='V-grade of the problem')
parser.add_argument('--latent-dim','-L', type=int, default=32, help='Latent dimension')
parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch-size', '-bs', type=int, default=128, help='Batch size')
parser.add_argument('--kld-weight', '-kld', type=float, default=2.5e-4, help='KL divergence weight')


args = parser.parse_args()

base_model = CVAE(in_channels=4,
             latent_dim=args.latent_dim, 
             )
# experiment = VAEXperiment(base_model,
#                           params = {
#                               'kld_weight': args.kld_weight,
#                               'LR': args.learning_rate
#                               })
# trained_model = experiment.load_from_checkpoint(args.path_to_checkpoint)
checkpoint = torch.load(args.path_to_checkpoint,map_location=torch.device('cpu'))
print(checkpoint['state_dict'].keys())
base_model.load_state_dict(checkpoint['state_dict'])
labels = torch.tensor([args.grade, args.angle],dtype=float)
predicted_out = trained_model.sample(labels, num_samples=10)
for i in range(10):
    fig = plot_climb(predicted_out[i])
    plt.savefig(f'./figures/generated_climbs/{i}.png',dpi=300)


