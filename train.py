import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from experiment import VAEXperiment
from models.cvae import CVAE
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin
import wandb


# parser = argparse.ArgumentParser(description='Generic runner for VAE models')

# args = parser.parse_args()


# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)


model = CVAE(in_channels=5,num_classes=10, latent_dim=32, img_size=img_size)

experiment = VAEXperiment(model,
                          config['exp_params'])

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

data.setup()
runner = Trainer(logger=tb_logger,
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
