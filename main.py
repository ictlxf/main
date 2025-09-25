import pytorch_lightning as pl
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LitModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss)  # 自动记录到进度条和日志
        return loss

trainer = pl.Trainer(max_epochs=10)
# trainer.fit(model, train_loader)