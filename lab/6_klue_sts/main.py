import argparse

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class SentenceDataset(torch.utils.data.Dataset):
    def __init__(
                    self,
                    sentence1,
                    sentence2,
                    tokenizer,
                ):
        super().__init__()
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.tokenizer = tokenizer
    
    
    def __len__(self):
        return len(self.sentence1)

    
    def __getitem__(self, idx):
        inputs = self.tokenizer(self.sentence1[idx] + " " + self.sentence2[idx], return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        return (input_ids.squeeze(0), attention_mask.squeeze(0))


class KLUEDatamodule(pl.LightningDataModule):
    def __init__(
                    self,
                    dataset_path,
                    dataset_name,
                    tokenizer,
                    batch_size,
                    num_workers,
                ):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    
    def prepare_data(self):
        load_dataset(self.dataset_path, self.dataset_name)
    
    
    def setup(self, stage=None):
        all_data = load_dataset(self.dataset_path, self.dataset_name)
        train_test_data = all_data["train"].train_test_split(test_size=0.2)
        train_data = train_test_data["train"]
        test_data = train_test_data["test"]
        val_data = all_data["validation"]

        self.train_dataset = SentenceDataset(
                                                train_data["sentence1"],
                                                train_data["sentence2"],
                                                self.tokenizer,
                                            )
        self.test_dataset = SentenceDataset(
                                                test_data["sentence1"],
                                                test_data["sentence2"],
                                                self.tokenizer,
                                            )
        self.val_dataset = SentenceDataset(
                                                val_data["sentence1"],
                                                val_data["sentence2"],
                                                self.tokenizer,
                                           )
    
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                                                self.train_dataset,
                                                batch_size=self.batch_size,
                                                num_workers=self.num_workers,
                                                shuffle=True,
                                                collate_fn=self.collate_fn,
                                            )
    
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
                                                self.test_dataset,
                                                batch_size=self.batch_size,
                                                num_workers=self.num_workers,
                                                shuffle=False,
                                                collate_fn=self.collate_fn,
                                            )
    
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                                                self.val_dataset,
                                                batch_size=self.batch_size,
                                                num_workers=self.num_workers,
                                                shuffle=False,
                                                collate_fn=self.collate_fn,
                                            )
    
    
    def collate_fn(self, batch):
        input_ids, attention_mask = zip(*batch)
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return (input_ids, attention_mask)


class STSModel(pl.LightningModule):
    def __init__(
                    self,
                    model_name,
                    tokenizer,
                    lr,
                ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.lr = lr
    
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits
    
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        logits = self(input_ids=input_ids, attention_mask=attention_mask)  # [batch_size, seq_len, vocab_size]
        loss = torch.nn.functional.cross_entropy(torch.permute(logits, (0, 2, 1)), input_ids)
        self.log("train_loss", loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(torch.permute(logits, (0, 2, 1)), input_ids)
        preds = self.tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
        inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        preds = list(map(lambda sentence: " ".join(self.tokenizer.tokenize(sentence)), preds))
        inputs = list(map(lambda sentence: " ".join(self.tokenizer.tokenize(sentence)), inputs))
        
        bleu_score = torchmetrics.functional.bleu_score(preds, inputs)
        self.log_dict({
            "val_loss": loss,
            "val_bleu_score": bleu_score,
        })
    
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(torch.permute(logits, (0, 2, 1)), input_ids)
        preds = self.tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
        inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        preds = list(map(lambda sentence: " ".join(self.tokenizer.tokenize(sentence)), preds))
        inputs = list(map(lambda sentence: " ".join(self.tokenizer.tokenize(sentence)), inputs))
        
        bleu_score = torchmetrics.functional.bleu_score(preds, inputs)
        self.log_dict({
            "test_loss": loss,
            "test_bleu_score": bleu_score,
        })
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dataset_path", type=str, default="klue")
    parser.add_argument("--dataset_name", type=str, default="sts")
    parser.add_argument("--model_name", type=str, default="skt/kogpt2-base-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--strategy", type=str, default="dp")
    args = vars(parser.parse_args())
    return args


def main():
    args = load_args()
    
    tokenizer = AutoTokenizer.from_pretrained(
                                                args["model_name"],
                                                bos_token='</s>', 
                                                eos_token='</s>',
                                                unk_token='<unk>',
                                                pad_token='<pad>',
                                                mask_token='<mask>',
                                            )

    datamodule = KLUEDatamodule(
                                    args["dataset_path"],
                                    args["dataset_name"],
                                    tokenizer,
                                    args["batch_size"],
                                    args["num_workers"],
                                )
    datamodule.prepare_data()
    
    trainer = pl.Trainer(
                            callbacks=[
                                EarlyStopping(monitor="val_loss", patience=5, mode="min"),
                                ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
                            ],
                            accelerator=args["accelerator"],
                            devices=args["devices"],
                            strategy=args["strategy"],
                        )
    
    model = STSModel(args["model_name"], tokenizer, args["lr"])
    
    trainer.test(model, datamodule)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
