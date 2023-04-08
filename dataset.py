import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import lightning.pytorch as pl

from sklearn.model_selection import train_test_split

class KilterDataset(Dataset):
    """
    Pytorch Kilter board dataset
    """
    def __init__(self,
                 board_path: str,
                 v_path: str,
                 sends_path: str,
                 split: str,
                 random_state: int = 4892
                 ):
        super().__init__()

        
        board_train, board_val, v_train, v_val, send_train, send_val = train_test_split(
                np.load(board_path), np.load(v_path), np.load(sends_path),
                test_size=0.1, random_state=random_state
                )
        if split == 'train':
            self.board_data = board_train
            self.v_grade = v_train
            self.send_counts = send_train
        elif split == 'val':
            self.board_data = board_val
            self.v_grade = v_val
            self.send_counts = send_val
        else:
            raise ValueError(f"split must be one of ['train', 'val'] but got {split}")

    def __len__(self):
        return self.board_data.shape[0]

    def __getitem__(self, idx):

        board_arr = self.board_data[0]
        v_grade = self.v_grade[0]
        num_sends = self.send_counts[0]
        
        return board_arr, v_grade, num_sends

class KilterDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning data module for Kilterboard data

    Args:
        
    """

    def __init__(self,
                 board_path: str,
                 v_path: str,
                 sends_path: str,
                 batch_size: int,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 **kwargs,
                 ):
        super().__init__()

        self.board_path = board_path
        self.v_path = v_path
        self.sends_path = sends_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage)

        self.train_dataset = KilterDataset(
                self.board_path,
                self.v_path,
                self.sends_path,
                split = 'train'
                )
        self.val_dataset = KilterDataset(
                self.board_path,
                self.v_path,
                self.sends_path,
                split = 'val'
                )

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory
                )

    def val_dataloader(self):
        return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory
                )

