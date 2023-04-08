import os
import torch
from torch import optim
from models.cvae import ConditionalVAE as CVAE
import lightning.pytorch as pl

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: CVAE,
                 params: dict
                 ) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params

        self.save_hyperparameters(ignore=['vae_model'])

    def forward(self, input: torch.tensor, y: torch.tensor, **kwargs) -> torch.tensor:
        return self.model(input, y, **kwargs)

    def training_step(self, batch, batch_idx):
        board_arrs, v_grades, angles, num_sends = batch
        board_metadata = torch.stack((v_grades, angles), dim=0).T

        results = self.model(board_arrs, board_metadata)
        train_loss = self.model.loss_function(*results,
                                              kld_weight = self.params['kld_weight'],
                                              batch_idx = batch_idx)

        for key, val in train_loss.items():
            self.log(f"train_{key}", val.item())

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        board_arrs, v_grades, angles, num_sends = batch
        board_metadata = torch.stack((v_grades, angles), dim=0).T

        results = self.forward(board_arrs, board_metadata)
        val_loss = self.model.loss_function(*results,
                                            kld_weight = self.params['kld_weight'],
                                            batch_idx = batch_idx)

        for key, val in val_loss.items():
            self.log(f"val_{key}", val.item())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               )
        return optimizer

