import os
import argparse

import numpy as np
import pandas as pd

import torch
import torchmetrics
import pytorch_lightning as pl


class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, queries, targets, input_vocab, output_vocab, max_sequence_length) -> None:
        super().__init__()
        self.sequences = sequences
        self.queries = queries
        self.targets = targets
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_sequence_length = max_sequence_length
    
    
    def __len__(self):
        return len(self.sequences)

    
    def __getitem__(self, index):
        sequence, query, target = self.sequences[index], self.queries[index], self.targets[index]
        
        ### token -> vocab_id
        sequence = [ self.input_vocab[token] for token in sequence ]
        query = self.input_vocab[query]
        target = self.output_vocab[target]
        
        ### add padding
        pad_id = self.input_vocab["[PAD]"]
        num_to_fill = self.max_sequence_length - len(sequence)
        weights = [1] * len(sequence) + [0] * num_to_fill  # mask
        sequence = sequence + [pad_id] * num_to_fill
        
        return (
            torch.tensor(sequence, dtype=torch.long),  # [max_sequence_length]
            torch.tensor([query], dtype=torch.long),  # [1]
            torch.tensor(weights, dtype=torch.long),  # [max_sequence_length]
            
            target  # scalar
        )


class DigitDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 batch_size,
                 num_workers,
                 **kwargs) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers


    def prepare_data(self):
        self.train_sequences, \
            self.train_queries, \
                self.train_targets = self.__load_txt(data_path=os.path.join(self.data_dir, "train.txt"))
        
        self.test_sequences, \
            self.test_queries, \
                self.test_targets = self.__load_txt(data_path=os.path.join(self.data_dir, "test.txt"))

        self.max_sequence_length = max(
            self.__get_max_sequence_length(self.train_sequences),
            self.__get_max_sequence_length(self.test_sequences)
        )

        print(f"max_sequence_length : {self.max_sequence_length}")

        self.input_vocab, self.output_vocab = self.__create_vocab()
        
        self.num_of_input_classes = len(self.input_vocab)
        self.num_of_output_classes = len(self.output_vocab)


    def setup(self, stage):
        if stage == "fit":
            all_train_dataset = DigitDataset(
                sequences=self.train_sequences,
                queries=self.train_queries,
                targets=self.train_targets,
                input_vocab=self.input_vocab,
                output_vocab=self.output_vocab,
                max_sequence_length=self.max_sequence_length
            )
            
            all_size = len(all_train_dataset)
            train_size = int(0.8 * all_size)
            val_size = all_size - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(all_train_dataset, [train_size, val_size])

        if stage == "test":
            self.test_dataset = DigitDataset(
                sequences=self.test_sequences,
                queries=self.test_queries,
                targets=self.test_targets,
                input_vocab=self.input_vocab,
                output_vocab=self.output_vocab,
                max_sequence_length=self.max_sequence_length
            )
    
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    
    def __create_vocab(self):
        input_tokens = []
        for sequence in self.train_sequences:
            input_tokens.extend(sequence)
        
        input_tokens = ["[PAD]"] + sorted(list(set(input_tokens)))
        output_tokens = ["[PAD]"] + sorted(list(set(self.train_targets)))
        
        input_vocab = { token: index for index, token in enumerate(input_tokens) }
        output_vocab = { token: index for index, token in enumerate(output_tokens) }
        return input_vocab, output_vocab


    def __get_max_sequence_length(self, sequences):
        max_sequence_length = 0

        for sequence in sequences:
            max_sequence_length = max(max_sequence_length, len(sequence))

        return max_sequence_length


    def __load_txt(self, data_path):
        sequences, queries, targets = [], [], []

        with open(data_path, mode="r") as stream:
            for line in stream.readlines():
                sequence, query, target = line.split()
                sequence = sequence.split(",")
                
                sequences.append(sequence)
                queries.append(query)
                targets.append(target)
        
        return sequences, queries, targets


class BahdanauAttention(torch.nn.Module):
    def __init__(self,
                 query_dim,
                 item_dim,
                 attention_dim) -> None:
        super().__init__()
        
        self.query_dim = query_dim
        self.item_dim = item_dim
        self.attention_dim = attention_dim
        
        # query_dim -> attention_dim
        self.W = torch.nn.Linear(self.query_dim, self.attention_dim, bias=False)
        
        # item_dim -> attention_dim
        self.U = torch.nn.Linear(self.item_dim, self.attention_dim, bias=False)
        
        # attention_dim -> scalar
        self.v = torch.nn.Linear(self.attention_dim, 1, bias=False)
    
    
    def forward(self, 
                queries,  # [batch_size, 1, query_dim]
                multiple_items,  # [batch_size, num_of_items, item_dim]
                weights  # [batch_size, num_of_items]
                ):
        projected_queries = self.W(queries)  # [batch_size, 1, attention_dim]
        projected_items = self.U(multiple_items)  # [batch_size, num_of_items, attention_dim]
        added = torch.tanh(projected_queries + projected_items)  # [batch_size, num_of_items, attention_dim]
        
        reactivity = self.v(added)  # [batch_size, num_of_items, 1]
        reactivity: torch.Tensor = reactivity.squeeze(-1)  # [batch_size, num_of_items]
        reactivity.data.masked_fill_(weights == 0, -float("inf"))  # mask된 값을 매우 작은 값으로
    
        attention_scores = torch.nn.functional.softmax(reactivity, dim=1)   # [batch_size, num_of_items]
        attention_scores = attention_scores.unsqueeze(dim=1)  # [batch_size, 1, num_of_items]
        
        blendded_vector = torch.matmul(attention_scores, multiple_items)  # [batch_size, 1, item_dim]
        
        return blendded_vector.squeeze(1), attention_scores.squeeze(1)


class DigitModel(pl.LightningModule):
    def __init__(self, model_dim, attention_dim, num_of_input_classes, num_of_output_classes, lr, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters("model_dim", "attention_dim", "lr")
        
        self.model_dim = model_dim
        self.attention_dim = attention_dim
        self.num_of_input_classes = num_of_input_classes
        self.num_of_output_classes = num_of_output_classes
        self.lr = lr
       
        ### layers 
        self.embed = torch.nn.Embedding(self.num_of_input_classes, self.model_dim)
        self.attention = BahdanauAttention(self.model_dim, self.model_dim, self.attention_dim)
        self.to_output = torch.nn.Linear(self.model_dim, self.num_of_output_classes, bias=False)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
    
    def forward(self, 
                sequences,  # [batch_size, max_sequence_length]
                queries,  # [batch_size, 1]
                weights  # [batch_size, max_sequence_length]
                ):
        multiple_items = self.embed(sequences)  # [batch_size, max_sequence_length, model_dim]
        queries = self.embed(queries)  # [batch_size, 1, model_dim]
        
        blendded_vector, attention_scores = self.attention(queries, multiple_items, weights)
        # blendded_vector : [batch_size, model_dim]
        # attention_scores : [batch_size, max_sequence_length(num_of_items)]
        
        logits = self.to_output(blendded_vector)  # [batch_size, num_of_output_classes]
        return logits
    
    
    def training_step(self, batch, batch_idx):
        sequences, queries, weights, targets = batch
        logits = self(sequences, queries, weights)
        
        # logits : [batch_size, num_of_output_classes]
        # targets : [batch_size, 1]
        loss = self.criterion(logits, targets)
        
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        sequences, queries, weights, targets = batch
        logits = self(sequences, queries, weights)
        
        loss = self.criterion(logits, targets)
        
        probability = torch.nn.functional.softmax(logits, dim=1)  # [batch_size, num_of_output_classes]
        acc = torchmetrics.functional.accuracy(probability, targets, task="multiclass", num_classes=self.num_of_output_classes)
        
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        
        return metrics
    
    
    def test_step(self, batch, batch_idx):
        sequences, queries, weights, targets = batch
        logits = self(sequences, queries, weights)
        
        loss = self.criterion(logits, targets)
        
        probability = torch.nn.functional.softmax(logits, dim=1)  # [batch_size, num_of_output_classes]
        acc = torchmetrics.functional.accuracy(probability, targets, task="multiclass", num_classes=self.num_of_output_classes)
        
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        
        return metrics
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    
    @staticmethod
    def add_model_specifig_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("DigitModel")
        parser.add_argument("--model_dim", type=int, default=10)
        parser.add_argument("--attention_dim", type=int, default=10)
        parser.add_argument("--lr", type=float, default=0.0001)
        return parent_parser


if __name__ == "__main__":
    pl.seed_everything(1234)
    
    os.chdir(os.path.dirname(__file__))
    from pytorch_lightning.callbacks import EarlyStopping
    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor="val_acc", mode="max", verbose=True, patience=8)],
        # accelerator="gpu",
        # devices=1,
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser = DigitModel.add_model_specifig_args(parser)
    parser = trainer.add_argparse_args(parser)
    
    args = vars(parser.parse_args())
    
    data = DigitDataModule(**args)
    data.prepare_data()  # to use self.num_of_output_classes
    model = DigitModel(num_of_input_classes=data.num_of_input_classes, num_of_output_classes=data.num_of_output_classes, **args)
    
    trainer.test(model, data)
    trainer.fit(model, data)
    trainer.test(model, data)
