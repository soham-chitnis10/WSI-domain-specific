"""
Pytorch Lightning Writer callback at Prediction
"""
import os
from pytorch_lightning.callbacks import BasePredictionWriter
import pytorch_lightning as pl
import torch


class PredWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval: str = "batch"):
        """_summary_

        Args:
            output_dir (_type_): _description_
            write_interval (str, optional): _description_. Defaults to "batch".
        """
        super().__init__(write_interval)
        self.output_dir = output_dir
    def on_predict_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int):
        """_summary_

        Args:
            trainer (pl.Trainer): _description_
            pl_module (pl.LightningModule): _description_
            outputs (_type_): _description_
            batch (_type_): _description_
            batch_idx (int): _description_
            dataloader_idx (int): _description_
        """
        os.makedirs(os.path.join(self.output_dir,batch[2]),exist_ok=True)
        torch.save({'features': outputs,'coords': batch[1]},os.path.join(self.output_dir,batch[2],'Rank_'+str(trainer.global_rank)+'_batch_'+str(batch_idx)+'.pt'))

