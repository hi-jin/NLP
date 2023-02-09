import argparse

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel


class Sentence2SentenceDataset():
    def __init__(self,
                 input_sentences,
                 output_sentences,
                 tokenizer,
                 max_seq_length,
                 **kwargs) -> None:
        self.input_sentences = input_sentences
        self.output_sentences = output_sentences
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    
    def __len__(self):
        return len(self.input_sentences)
    
    
    def __getitem__(self, index):
        input_sentence = self.input_sentences[index]
        output_sentence = self.output_sentences[index]
        
        inputs = self.tokenizer(
            input_sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        outputs = self.tokenizer(
            output_sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        output_ids = outputs["input_ids"]

        return (
            ### inputs
            input_ids.squeeze(0),
            attention_mask.squeeze(0),
            
            ### labels
            output_ids.squeeze(0)
        )


class KLUEDatamodule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_name, 
                 dataset_subset_name,
                 pretrained_model_name,
                 max_seq_length,
                 batch_size,
                 num_workers,
                 tokenizer,
                 **kwargs) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_subset_name = dataset_subset_name
        self.pretrained_model_name = pretrained_model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
    
    
    def prepare_data(self) -> None:
        load_dataset(self.dataset_name, self.dataset_subset_name)

    
    def setup(self, stage: str) -> None:
        all_data = load_dataset(self.dataset_name, self.dataset_subset_name)
        train_test_data = all_data["train"].train_test_split(train_size=0.8, test_size=0.2, shuffle=True)

        self.train_data = train_test_data["train"]
        self.test_data = train_test_data["test"]
        self.val_data = all_data["validation"]

        if stage == "fit":
            self.train_dataset = Sentence2SentenceDataset(
                self.train_data["sentence1"],
                self.train_data["sentence2"],
                self.tokenizer,
                self.max_seq_length,
            )
            self.val_dataset = Sentence2SentenceDataset(
                self.val_data["sentence1"],
                self.val_data["sentence2"],
                self.tokenizer,
                self.max_seq_length,
            )
        if stage == "test":
            self.test_dataset = Sentence2SentenceDataset(
                self.test_data["sentence1"],
                self.test_data["sentence2"],
                self.tokenizer,
                self.max_seq_length,
            )
    
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class TextGenerationModel(pl.LightningModule):
    def __init__(self,
                 pretrained_model_name,
                 tokenizer,
                 lr,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        self.tokenizer = tokenizer
        self.lr = lr
    
    
    def forward(self, input_ids, attention_mask, labels=None):
        if labels is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss, outputs.logits
    
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss, logits = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss, logits = self(input_ids, attention_mask, labels)
        
        scores = []
        preds = self.tokenizer.batch_decode(logits.argmax(dim=-1))
        targets = self.tokenizer.batch_decode(labels.masked_fill(attention_mask == 0, self.tokenizer.pad_token_id))
        for pred, target in zip(preds, targets):
            score = torchmetrics.functional.bleu_score([pred], [[target]])
            scores.append(score)
        
        bleu_score = np.mean(scores)
        
        metrics = {
            "val_loss": loss,
            "val_bleu_score": bleu_score,
        }
        
        self.log_dict(metrics)
        return metrics
    
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss, logits = self(input_ids, attention_mask, labels)
        
        scores = []
        preds = self.tokenizer.batch_decode(logits.argmax(dim=-1))
        targets = self.tokenizer.batch_decode(labels.masked_fill(attention_mask == 0, self.tokenizer.pad_token_id))
        for pred, target in zip(preds, targets):
            score = torchmetrics.functional.bleu_score([pred], [[target]])
            scores.append(score)
        
        bleu_score = np.mean(scores)
        
        metrics = {
            "test_loss": loss,
            "test_bleu_score": bleu_score,
        }
        
        self.log_dict(metrics)
        return metrics
        
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="klue")
    parser.add_argument("--dataset_subset_name", type=str, default="sts")
    parser.add_argument("--pretrained_model_name", type=str, default="skt/kogpt2-base-v2")
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    return vars(parser.parse_args())


def main():
    pl.seed_everything(1234)
    
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args["pretrained_model_name"],
        bos_token='</s>',
        eos_token='</s>',
        unk_token='<unk>',
        pad_token='<pad>',
        mask_token='<mask>'
    )
    
    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=3),
            ModelCheckpoint("./checkpoints", "val_loss:{val_loss}-val_bleu:{val_bleu_score}", monitor="val_loss", mode="min", verbose=True)
        ],
        # accelerator="gpu",
        # devices=4,
        # strategy="dp"
    )
    
    data = KLUEDatamodule(tokenizer=tokenizer, **args)

    model = TextGenerationModel(tokenizer=tokenizer, **args)
    
    trainer.test(model, data)
    trainer.fit(model, data)
    trainer.test(model, data)


if __name__ == "__main__":
    main()
    