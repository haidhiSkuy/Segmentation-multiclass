import torch 
import numpy as np 
import lightning as L
from callback import get_callbacks
from model.lightning_module import LitResUnet
from lightning.pytorch.loggers import WandbLogger
from dataset.lightning_datamodule import UwmgiDataModule


model = LitResUnet()
data = UwmgiDataModule("data.csv")
wandb_logger = WandbLogger(project="ResUnet", log_model=True)
callbacks = get_callbacks()

# Trainer
trainer = L.Trainer(max_epochs=10, logger=wandb_logger, callbacks=callbacks)
trainer.fit(model, data) 