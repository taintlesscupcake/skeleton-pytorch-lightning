import os
from typing import Any, List, Union
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
from dataloader import DefaultDataset
from model import DefaultModel
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Evaluator(object):
    def __init__(self, args):
        self.args = args


class YourLightningModule(LightningModule):
    def __init__(self, args):
        super(YourLightningModule, self).__init__()
        self.args = args

        self.model = DefaultModel(self.args)

        self.evaluator = Evaluator(self.args)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        # implement your own data
        x, y = batch

        # implement your own forward
        y_hat = self.model(x.float())

        # implement your own loss
        loss = F.cross_entropy(y_hat, y)

        # implement your own logging
        self.log("train/loss", loss.item(), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.args.batch_size)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        # implement your own data
        x, y = batch

        # implement your own forward
        y_hat = self.model(x.float())

        # implement your own loss
        loss = F.cross_entropy(y_hat, y)

        # implement your own logging
        self.log("val/loss", loss.item(), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.args.batch_size)

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        # implement your own data
        x, y = batch

        # implement your own forward
        y_hat = self.model(x.float())

        # implement your own loss
        loss = F.cross_entropy(y_hat, y)

        # implement your own logging
        self.log("test/loss", loss.item(), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.args.batch_size)

        return loss

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT]) -> None:
        # implement your own logging
        self.log("val/epoch_loss", torch.stack(outputs).mean(),
                 on_epoch=True, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT]) -> None:
        # implement your own logging
        self.log("test/epoch_loss", torch.stack(outputs).mean(),
                 on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # implement your own optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)

        # implement your own lr scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1)

        return [optimizer], [scheduler]


class YourLightningDataModule(LightningDataModule):
    def __init__(self, args):
        super(YourLightningDataModule, self).__init__()
        self.args = args

        def setup(self, stage):
            # implement your own data
            self.train_dataset = self.set_train_dataset()
            self.val_dataset = self.set_val_dataset()
            self.test_dataset = self.set_test_dataset()

        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.val_dataset, batch_size=1, num_workers=self.args.num_workers, shuffle=False)

        def test_dataloader(self):
            return DataLoader(self.test_dataset, batch_size=1, num_workers=self.args.num_workers, shuffle=False)

        def set_train_dataset(self):
            # implement your own data
            img_path = os.path.join(self.args.data_path, 'train')
            img_files = os.listdir(img_path)

            return DefaultDataset(
                mode='train', data_path=img_path, data_files=img_files)

        def set_val_dataset(self):
            # implement your own data
            img_path = os.path.join(self.args.data_path, 'val')
            img_files = os.listdir(img_path)

            return DefaultDataset(
                mode='val', data_path=img_path, data_files=img_files)

        def set_test_dataset(self):
            # implement your own data
            img_path = os.path.join(self.args.data_path, 'test')
            img_files = os.listdir(img_path)

            return DefaultDataset(
                mode='test', data_path=img_path, data_files=img_files)


# Replace YourLightningModule and YourLightningDataModule with your own classes
DefaultModule = YourLightningModule
DefaultDataModule = YourLightningDataModule
