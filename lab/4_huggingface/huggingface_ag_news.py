import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NewsDataset(Dataset):
    def __init__(self,
                 text,
                 label,
                 tokenizer,
                 max_seq_length,
                 **kwargs):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    
    def __len__(self):
        return len(self.text)
    
    
    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer(text,
                       padding="max_length",
                       truncation=True,
                       max_length=self.max_seq_length,
                       return_tensors="pt")
        
        input_ids = inputs["input_ids"]  # [1, max_length]
        token_type_ids = inputs["token_type_ids"]  # [1, max_length]
        attention_mask = inputs["attention_mask"]  # [1, max_length]
        
        label = self.label[index]  # scalar
        
        return (
            input_ids.squeeze(0),  # [max_length]
            token_type_ids.squeeze(0),  # [max_length]
            attention_mask.squeeze(0),  # [max_length]
            
            label,  # scalar
        )


class NewsDataModule(LightningDataModule):
    def __init__(self, 
                 batch_size,
                 num_workers,
                 max_seq_length,
                 **kwargs) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.dataset_path = "ag_news"
        self.model_path = "bert-base-uncased"
    
    
    def prepare_data(self):
        """dataset, tokenizer를 다운로드
        """
        
        ### 여기서 self.에 등록해둬도 multiprocessing하면 적용 할 수 없음
        load_dataset(path=self.dataset_path)
        AutoTokenizer.from_pretrained(self.model_path)
    
    
    def setup(self, stage: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        all_dataset = load_dataset(path=self.dataset_path)
        
        if stage == "fit":
            train_val_dataset = all_dataset["train"].train_test_split(train_size=0.8, test_size=0.2, shuffle=True)
            train_data = train_val_dataset["train"]
            val_data = train_val_dataset["test"]
            self.train_dataset = NewsDataset(train_data["text"], train_data["label"], self.tokenizer, self.max_seq_length)
            self.val_dataset = NewsDataset(val_data["text"], val_data["label"], self.tokenizer, self.max_seq_length)
        if stage == "test":
            test_data = all_dataset["test"]
            self.test_dataset = NewsDataset(test_data["text"], test_data["label"], self.tokenizer, self.max_seq_length)
    
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class NewsClassifier(LightningModule):
    def __init__(self, 
                 lr,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_path = "bert-base-uncased"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=4)
        self.lr = lr
    
    
    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,  # [batch_size, max_length]
            token_type_ids=token_type_ids,  # [batch_size, max_length]
            attention_mask=attention_mask,  # [batch_size, max_length]
            labels=labels  # [batch_size, ]
            )
        
        return (
            outputs.loss,  # scalar
            outputs.logits,  # batch_size, num_labels
        )
    
    
    def training_step(self, batch, batch_idx):
        loss, logits = self(*batch)
        self.log_dict({
            "train_loss": loss.item(),
        })
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        loss, logits = self(*batch)
        
        predicts = torch.argmax(logits, dim=1).detach().cpu().numpy()  # [batch_size, ]
        labels = batch[3].detach().cpu().numpy()  # [batch_size, ]
        
        corrects = predicts == labels
        
        return {
            "val_corrects": corrects,
            "val_loss": loss.item(),
        }
        
    
    def validation_epoch_end(self, outputs):
        bunch_of_loss = []
        bunch_of_corrects = np.array([])
        for output in outputs:
            loss = output["val_loss"]
            corrects = output["val_corrects"]
            
            bunch_of_loss.append(loss)
            bunch_of_corrects = np.concatenate((bunch_of_corrects, corrects))
        
        bunch_of_loss  # [steps, ]
        bunch_of_corrects  # [batch_size, ]
        
        self.log_dict({
            "val_loss": np.mean(bunch_of_loss),
            "val_acc": np.mean(bunch_of_corrects),
        })
    
    
    def test_step(self, batch, batch_idx):
        loss, logits = self(*batch)
        
        predicts = torch.argmax(logits, dim=1).detach().cpu().numpy()  # [batch_size, ]
        labels = batch[3].detach().cpu().numpy()  # [batch_size, ]
        
        corrects = predicts == labels
        
        return {
            "test_corrects": corrects,
            "test_loss": loss.item(),
        }
    
    
    def test_epoch_end(self, outputs):
        bunch_of_loss = []
        bunch_of_corrects = np.array([])
        for output in outputs:
            loss = output["test_loss"]
            corrects = output["test_corrects"]
            
            bunch_of_loss.append(loss)
            bunch_of_corrects = np.concatenate((bunch_of_corrects, corrects))
        
        bunch_of_loss  # [steps, ]
        bunch_of_corrects  # [batch_size, ]
        
        self.log_dict({
            "test_loss": np.mean(bunch_of_loss),
            "test_acc": np.mean(bunch_of_corrects),
        })
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)
    
    
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("NewsClassifier")
        parser.add_argument("--lr", type=float, default=0.0001)
        return parent_parser


def main():
    seed_everything(1234)
    
    ######
    # argparser
    ######
    parser = argparse.ArgumentParser()
    parser = NewsClassifier.add_model_specific_args(parser)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--max_seq_length", type=int)
    
    args = vars(parser.parse_args())
    
    ######
    # trainer
    ######
    trainer = Trainer(
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=5),
            ModelCheckpoint(monitor="val_loss", dirpath="./checkpoints/", filename="{epoch}-{val_acc:.2f}", save_top_k=3, mode="min"),
        ],
        # accelerator="gpu",
        # devices=4,
    )

    ######
    # datamodule
    ######
    datamodule = NewsDataModule(**args)
    
    ######
    # model
    ######
    model = NewsClassifier(**args)
    
    ######
    # training
    ######
    trainer.test(model, datamodule)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
    

if __name__ == "__main__":
    main()
