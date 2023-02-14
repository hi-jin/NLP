import os
import sys
import argparse

import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, queries, targets, input_vocab, output_vocab, max_seq_length) -> None:
        super().__init__()
        self.sequences = sequences
        self.queries = queries
        self.targets = targets
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_seq_length = max_seq_length
    
    
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
        num_to_fill = self.max_seq_length - len(sequence)
        weights = [1] * len(sequence) + [0] * num_to_fill  # mask
        sequence = sequence + [pad_id] * num_to_fill
        
        return (
            torch.tensor(sequence, dtype=torch.long),  # [max_seq_length]
            torch.tensor(query, dtype=torch.long),  # scalar
            torch.tensor(weights, dtype=torch.long),  # [max_seq_length]
            
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

        self.max_seq_length = max(
            self.__get_max_seq_length(self.train_sequences),
            self.__get_max_seq_length(self.test_sequences)
        )

        print(f"max_seq_length : {self.max_seq_length}")

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
                max_seq_length=self.max_seq_length
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
                max_seq_length=self.max_seq_length
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


    def __get_max_seq_length(self, sequences):
        max_seq_length = 0

        for sequence in sequences:
            max_seq_length = max(max_seq_length, len(sequence))

        return max_seq_length


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


class Attention(torch.nn.Module):
    def __init__(self,
                 d_embed,
                 num_heads,
                 d_model,
                 **kwargs) -> None:
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_embed = d_embed
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model // self.num_heads
        
        self.wq = torch.nn.Linear(d_embed, d_model)
        self.wk = torch.nn.Linear(d_embed, d_model)
        self.wv = torch.nn.Linear(d_embed, d_model)
        self.wo = torch.nn.Linear(d_model, d_embed)
    
    
    def split_heads(self,
                    x: torch.Tensor):  # [batch_size, seq_length, d_model]
        batch_size, seq_length, d_model = x.shape
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        x = x.transpose(1, 2)  # [batch_size, num_heads, seq_length, d_k]
        return x
        
    
    
    def forward(self, 
                input_embeds,  # [batch_size, max_seq_length, d_embed]
                query_embeds,  # [batch_size, d_query, d_embed]
                mask: torch.Tensor,  # [batch_size, max_seq_length]
                label):  # [batch_size]
        Q = self.wq(query_embeds)  # [batch_size, d_query, d_model]
        K = self.wk(input_embeds)  # [batch_size, max_seq_length, d_model]
        V = self.wv(input_embeds)  # [batch_size, max_seq_length, d_model]

        splitted_Q = self.split_heads(Q)  # [batch_size, num_heads, d_query, d_k]
        splitted_K = self.split_heads(K)  # [batch_size, num_heads, max_seq_length, d_k]
        splitted_V = self.split_heads(V)  # [batch_size, num_heads, max_seq_length, d_k]
        
        # [batch_size, num_heads, d_query, d_model] @ [batch_size, num_heads, d_model, max_seq_length]
        # -> [batch_size, num_heads, d_query, max_seq_length]
        reactivity_scores = torch.matmul(splitted_Q, splitted_K.transpose(2, 3))

        mask = mask[:, None, None, :]  # [batch_size, 1, 1, max_seq_length]
        mask = 1 - mask
        mask = mask.masked_fill(mask.bool(), -sys.maxsize-1)

        # [batch_size, num_heads, d_query, max_seq_length] + [batch_size, 1, 1, max_seq_length]  (broadcast)
        reactivity_scores += mask
        
        attention_scores = torch.nn.functional.softmax(reactivity_scores, dim=-1)  # [batch_size, num_heads, d_query, max_seq_length]

        blended_vector = torch.matmul(attention_scores, splitted_V)  # [batch_size, num_heads, d_query, d_k]
        batch_size, num_heads, d_query, d_k = blended_vector.shape
        
        concatenated_vector = blended_vector.transpose(1, 2).view(batch_size, d_query, num_heads * d_k)
        concatenated_vector  # [batch_size, d_query, d_model]

        output_vector = self.wo(concatenated_vector)  # [batch_size, d_query, d_embed]
        return output_vector, attention_scores


class DigitModel(pl.LightningModule):
    def __init__(self, 
                 d_input,
                 d_output,
                 d_embed,
                 num_heads,
                 d_model,
                 lr,
                 **kwargs):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.d_embed = d_embed
        self.num_heads = num_heads
        self.d_model = d_model
        self.lr = lr
        
        self.save_hyperparameters()
        
        self.embed = torch.nn.Embedding(self.d_input, self.d_embed)
        self.attention = Attention(self.d_embed, self.num_heads, self.d_model)
        self.to_output = torch.nn.Linear(self.d_embed, self.d_output)
    
    
    def forward(self, 
                input_ids,  # [batch_size, max_seq_length]
                query,  # [batch_size]
                mask,  # [batch_size, max_seq_length]
                label):  # [batch_size]
        input_embeds = self.embed(input_ids)  # [batch_size, max_seq_length, d_embed]
        query_embeds = self.embed(query)  # [batch_size, d_embed]
        
        blended_vector, attention_scores = self.attention(input_embeds,  # [batch_size, max_seq_length, d_embed]
                                                          query_embeds.unsqueeze(1),  # [batch_size, d_query=1, d_embed]
                                                          mask,  # [batch_size, max_seq_length]
                                                          label)  # [batch_size]
        blended_vector  # [batch_size, d_query=1, d_embed]
        blended_vector = blended_vector.squeeze(1)  # [batch_size, d_embed]
        
        logits = self.to_output(blended_vector)  # [batch_size, d_output]
        
        return logits, attention_scores


    def training_step(self, batch, batch_idx):
        input_ids, query, mask, label = batch
        logits, attention_scores = self(input_ids, query, mask, label)

        loss = torch.nn.functional.cross_entropy(logits, label)
        self.log("train_loss", loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input_ids, query, mask, label = batch
        logits, attention_scores = self(input_ids, query, mask, label)
        
        loss = torch.nn.functional.cross_entropy(logits, label)
        acc = np.mean((torch.argmax(logits, dim=-1) == label).tolist())
        
        self.log_dict({
            "val_loss": loss,
            "val_acc": acc,
        })
    
    
    def test_step(self, batch, batch_idx):
        input_ids, query, mask, label = batch
        logits, attention_scores = self(input_ids, query, mask, label)
        
        loss = torch.nn.functional.cross_entropy(logits, label)
        acc = np.mean((torch.argmax(logits, dim=-1) == label).tolist())
        
        self.log_dict({
            "test_loss": loss,
            "test_acc": acc,
        })

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../dataset/attention_prac/data/numbers")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--d_embed", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = vars(parser.parse_args())

    data = DigitDataModule(**args)
    data.prepare_data()
    
    model = DigitModel(d_input=len(data.input_vocab),
                       d_output=len(data.output_vocab),
                       **args)
    
    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min"),
            ModelCheckpoint(filename="{val_acc}-{val_loss}", monitor="val_loss", verbose=True, mode="min")
        ]
    )
    
    trainer.test(model, data)
    trainer.fit(model, data)
    trainer.test(model, data)
