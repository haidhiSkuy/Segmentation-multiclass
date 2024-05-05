import lightning as L 
from .resunet import get_model 
from .losses_metrics import * 
from resunet import get_model

class LitResUnet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        prediction = self.model(x)

        loss = criterion(prediction, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)

        loss = criterion(prediction, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer