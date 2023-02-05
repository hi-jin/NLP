import argparse

import numpy as np

import torch
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from transformers import AutoTokenizer, BertForSequenceClassification
from datasets import load_dataset


PRETRAINED_MODEL = "kykim/bert-kor-base"
DATASET = "klue"
SUBSET_NAME_FOR_TC = "ynat"


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 texts,
                 labels,
                 tokenizer,
                 max_seq_length,
                 
                 **kwargs):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    
    def __len__(self):
        return len(self.texts)
    
    
    def __getitem__(self, index):
        text, label = self.texts[index], self.labels[index]
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_seq_length, return_tensors="pt")

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        
        return input_ids.squeeze(0), token_type_ids.squeeze(0), attention_mask.squeeze(0), label


class Datamodule(pl.LightningDataModule):
    def __init__(self,
                 max_seq_length,
                 batch_size,
                 num_workers,
                 **kwargs) -> None:
        super().__init__()
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    
    def prepare_data(self):
        AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        load_dataset(DATASET, SUBSET_NAME_FOR_TC)
    
    
    def setup(self, stage):
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        all_dataset = load_dataset(DATASET, SUBSET_NAME_FOR_TC)
        
        train_test_data = all_dataset["train"].train_test_split(train_size=0.8, test_size=0.2)
        train_data = train_test_data["train"]
        test_data = train_test_data["test"]
        val_data = all_dataset["validation"]
        
        if stage == "fit":
            self.train_dataset = Dataset(train_data["title"], train_data["label"], self.tokenizer, self.max_seq_length)
            self.val_dataset = Dataset(val_data["title"], val_data["label"], self.tokenizer, self.max_seq_length)
        if stage == "test":
            self.test_dataset = Dataset(test_data["title"], test_data["label"], self.tokenizer, self.max_seq_length)
    
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True)


    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)


    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)


class TopicClassifier(pl.LightningModule):
    def __init__(self,
                 lr,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=7)
        self.lr = lr
    
    
    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids,
                   token_type_ids=token_type_ids,
                   attention_mask=attention_mask,
                   labels=labels)
        return outputs.loss, outputs.logits
    
    
    def training_step(self, batch, batch_idx):
        loss, logits = self(*batch)
        self.log_dict({
            "train_loss": loss.item(),
        })
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        loss, logits = self(*batch)
        labels = batch[3].detach().cpu()
        predicts = torch.argmax(logits, dim=1).detach().cpu()
        
        return {
            "val_predicts": predicts,
            "val_labels": labels,
            "val_loss": loss.item(),
        }
        
    
    def validation_epoch_end(self, outputs):
        predicts = []
        labels = []
        loss = []
        for output in outputs:
            predicts.append(output["val_predicts"])
            labels.append(output["val_labels"])
            loss.append(output["val_loss"])
        
        self.log_dict({
            "val_loss": np.mean(loss),
            "val_f1": torchmetrics.functional.f1_score(torch.cat(predicts), torch.cat(labels), num_classes=7, task="multiclass"),
        })
    
    
    def test_step(self, batch, batch_idx):
        loss, logits = self(*batch)
        labels = batch[3].detach().cpu()
        predicts = torch.argmax(logits, dim=1).detach().cpu()
        
        return {
            "test_predicts": predicts,
            "test_labels": labels,
            "test_loss": loss.item(),
        }
        
    
    def test_epoch_end(self, outputs):
        predicts = []
        labels = []
        loss = []
        for output in outputs:
            predicts.append(output["test_predicts"])
            labels.append(output["test_labels"])
            loss.append(output["test_loss"])
        
        self.log_dict({
            "test_loss": np.mean(loss),
            "test_f1": torchmetrics.functional.f1_score(torch.cat(predicts), torch.cat(labels), num_classes=7, task="multiclass"),
        })
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)
    
    
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("TopicClassifier")
        parser.add_argument("--lr", type=float, default=0.0001)
        return parent_parser


def main():
    ######
    # args
    ######
    parser = argparse.ArgumentParser()
    parser = TopicClassifier.add_model_specific_args(parser)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=64)
    args = vars(parser.parse_args())
    
    ######
    # trainer
    ######
    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=5),
            ModelCheckpoint(monitor="val_loss", mode="min", dirpath="./checkpoints", filename="{val_f1:.2f}-{epoch}", save_top_k=3),
        ],
        accelerator="gpu",
        devices=4,
    )
    
    ######
    # datamodule
    ######
    datamodule = Datamodule(**args)

    ######
    # model
    ######
    model = TopicClassifier(**args)
    
    ######
    # train
    ######
    trainer.test(model, datamodule)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
