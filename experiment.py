import os
import math
import torch
from torch import optim
from models.cvae import ConditionalVAE as CVAE
from utils import data_loader
import lightning.pytorch as pl
from torchvision import transforms
import torchvision.utils as vutils

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: CVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: torch.tensor, y: torch.tensor, **kwargs) -> torch.tensor:
        return self.model(input, y, **kwargs)

    #def on_train_epoch_start(self) -> None:
    #    lr = self.trainer.optimizers[0].param_groups[0]['lr']
    #    self.logger.experiment.log({'learning_rate': lr}, step=self.current_epoch)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        board_arrs, v_grads, num_sends = batch
        board_metadata = np.concatenate((v_grads, num_sends), axis=0)
        self.curr_device = board_arrs.device

        results = self.model(board_arrs, board_metadata)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        board_arrs, v_grads, num_sends = batch
        board_metadata = np.concatenate((v_grads, num_sends), axis=0)
        self.curr_device = real_img.device

        results = self.forward(board_arrs, board_metadata)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        return optimizer

