import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from experiment import VAEXperiment
from dataset import KilterDataset
from models.cvae import ConditionalVAE as CVAE
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import wandb


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--latent_dim','-L', type=int, default=32, help='Latent dimension')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size')


args = parser.parse_args()
config = vars(args)

WANDB_PROJECT = "cvae"
WANDB_ENTITY = "cvae"
WANDB_NAME = "cvae"
wandb_logger = WandbLogger(project=WANDB_PROJECT)
model = CVAE(in_channels=5,num_classes=10, latent_dim=args.latent_dim, img_size=img_size)

experiment = VAEXperiment(model,
                          config['exp_params'])

data = KilterDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

data.setup()
runner = Trainer(logger=wandb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
