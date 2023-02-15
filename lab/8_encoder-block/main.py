import os
import sys
import argparse
import copy

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def load_data(fn):
    data = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()

            query_item_seq_str, y = line.split('\t')
            all_tokens = query_item_seq_str.split(',')
            q_tokens = all_tokens[0].split('|')
            i_tokens = all_tokens[1:]

            tokens = [q_tokens[0], '|'] + [q_tokens[1]] + i_tokens 
            data.append( (tokens, y) )
    return data

# you can define any type of dataset
# dataset : return an example for batch construction. 
class NumberDataset(Dataset):
    """Dataset."""

    def __init__(self, fn, input_vocab, output_vocab, max_seq_length):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_seq_length = max_seq_length 
        
        # load 
        self.data = load_data(fn)


    def __len__(self):
        return len(self.data) 


    def __getitem__(self, idx): 
        seq, y = self.data[idx]

        # [ input ]
        seq_ids = [ self.input_vocab[t] for t in seq ]

        # <pad> processing
        pad_id      = self.input_vocab['<pad>']
        num_to_fill = self.max_seq_length - len(seq)
        seq_ids     = seq_ids + [pad_id]*num_to_fill

        # mask processing (1 for valid, 0 for invalid)
        weights = [1]*len(seq) + [0]*num_to_fill

        # [ ouput ] 
        y_id = self.output_vocab[y]

        item = [
                    # input
                    np.array(seq_ids),
                    np.array(weights),

                    # output
                    y_id
               ]
        return item 


class NumberDataModule(pl.LightningDataModule):
    def __init__(self, 
                 max_seq_length: int=15,
                 batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length 

        input_vocab, output_vocab = self.make_vocab('../../dataset/attention_prac/data/numbers/train.seq.txt')
        self.input_vocab_size = len( input_vocab )
        self.output_vocab_size = len( output_vocab )
        self.padding_idx = input_vocab['<pad>']

        self.input_r_vocab  = { v:k for k,v in input_vocab.items() }
        self.output_r_vocab = { v:k for k,v in output_vocab.items() }

        self.all_train_dataset = NumberDataset('../../dataset/attention_prac/data/numbers/train.seq.txt', input_vocab, output_vocab, max_seq_length)
        self.test_dataset      = NumberDataset('../../dataset/attention_prac/data/numbers/test.seq.txt', input_vocab, output_vocab, max_seq_length)

        # random split train / valiid for early stopping
        N = len(self.all_train_dataset)
        tr = int(N*0.8) # 8 for the training
        va = N - tr     # 2 for the validation 
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.all_train_dataset, [tr, va])


    def make_vocab(self, fn):
        input_tokens = []
        output_tokens = []
        data = load_data(fn)

        for tokens, y in data:
            for token in tokens:
                input_tokens.append(token)
            output_tokens.append(y)
        
        input_tokens = list(set(input_tokens))
        output_tokens = list(set(output_tokens)) 

        input_tokens.sort()
        output_tokens.sort()

        # [input vocab]
        # add <pad> symbol to input tokens as a first item
        input_tokens = ['<pad>'] + input_tokens 
        input_vocab = { str(token):index for index, token in enumerate(input_tokens) }

        # [output voab]
        output_vocab = { str(token):index for index, token in enumerate(output_tokens) }

        return input_vocab, output_vocab


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # NOTE : Shuffle


    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)
    
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class Attention(torch.nn.Module):
    def __init__(self,
                 d_embed,
                 d_model,
                 num_heads,
                 **kwargs) -> None:
        super().__init__()

        assert d_model % num_heads == 0
        
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        
        self.wq = torch.nn.Linear(self.d_embed, self.d_model)
        self.wk = torch.nn.Linear(self.d_embed, self.d_model)
        self.wv = torch.nn.Linear(self.d_embed, self.d_model)
        self.wo = torch.nn.Linear(self.d_model, self.d_embed)


    def split_heads(
                        self,
                        x: torch.Tensor,  # [batch_size, seq_length, d_model]
                    ):
        batch_size, seq_length, d_model = x.shape
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        x = x.transpose(1, 2)
        return x  # [batch_size, num_heads, seq_length, d_k]


    def forward(
                    self,
                    query,  # [batch_size, d_query, d_embed]
                    input_embeds,  # [batch_size, max_seq_length, d_embed]
                    mask: torch.Tensor,  # [batch_size, max_seq_length]
                ):
        Q = self.split_heads(self.wq(query))  # [batch_size, num_heads, d_query, d_k]
        K = self.split_heads(self.wk(input_embeds))  # [batch_size, num_heads, max_seq_length, d_k]
        V = self.split_heads(self.wv(input_embeds))  # [batch_size, num_heads, max_seq_length, d_k]
        
        reactivity_scores = torch.matmul(Q, K.transpose(2, 3))  # [batch_size, num_heads, d_query, max_seq_length]
        
        mask = mask[:, None, None, :]  # [batch_size, 1, 1, max_seq_length]
        mask = 1 - mask
        mask = mask.masked_fill(mask.bool(), -sys.maxsize-1)
        reactivity_scores += mask
        
        attention_scores = torch.nn.functional.softmax(reactivity_scores, dim=-1)  # [batch_size, num_heads, d_query, max_seq_length]

        blended_vector = torch.matmul(attention_scores, V)  # [batch_size, num_heads, d_query, d_k]
        blended_vector = blended_vector.transpose(1, 2)  # [batch_size, d_query, num_heads, d_k]
        batch_size, d_query, num_heads, d_k = blended_vector.shape

        concatenated_vector = blended_vector.contiguous().view(batch_size, d_query, self.d_model)
        
        output_vector = self.wo(concatenated_vector)  # [batch_size, d_query, d_embed]

        return output_vector, attention_scores
        
        
class Encoder(torch.nn.Module):
    def __init__(
                    self,
                    d_embed,
                    d_model,
                    num_heads,
                 ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention = Attention(self.d_embed, self.d_model, self.num_heads)

        self.fc1 = torch.nn.Linear(self.d_embed, self.d_embed*4)
        self.fc2 = torch.nn.Linear(self.d_embed*4, self.d_embed)
        
        self.norm = torch.nn.LayerNorm(self.d_embed)


    def forward(self, input_embeds, mask):
        residual = input_embeds
        x, attention_scores = self.attention(query=input_embeds, input_embeds=input_embeds, mask=mask)
        x = residual + x
        x = self.norm(x)

        residual = x
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        x = self.norm(x)
        
        return x, attention_scores


class MultiLayerEncoder(torch.nn.Module):
    def __init__(
                    self,
                    d_embed,
                    d_model,
                    num_heads,
                    num_layers,
                 ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        a_layer = Encoder(self.d_embed, self.d_model, self.num_heads)

        self.layers = torch.nn.ModuleList([
            copy.deepcopy(a_layer) for _ in range(self.num_layers)
        ])
    
    
    def forward(self, input_ids, mask):
        x = input_ids
        for layer in self.layers:
            x, attention_scores = layer(x, mask)
        return x, attention_scores
    
    
class NumFinder(pl.LightningModule):
    def __init__(
                    self, 
                    d_input_vocab,
                    d_output_vocab,
                    d_embed, 
                    d_model, 
                    num_heads,
                    lr,
                    ):
        super().__init__()
        self.save_hyperparameters()
        self.d_input_vocab = d_input_vocab
        self.d_output_vocab = d_output_vocab
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.lr = lr
        
        self.embed = torch.nn.Embedding(self.d_input_vocab, self.d_embed)
        # self.encode = Encoder(self.d_embed, self.d_model, self.num_heads)
        self.encode = MultiLayerEncoder(self.d_embed, self.d_model, self.num_heads, 3)
        self.to_output = torch.nn.Linear(self.d_embed, self.d_output_vocab)
        
    
    def forward(self, input_ids, mask):
        input_embeds = self.embed(input_ids)
        
        # [batch_size, d_query, d_embed]
        encoded, attention_scores = self.encode(input_embeds, mask)

        blended_vector = encoded[:, 0]
        logits = self.to_output(blended_vector)

        return logits, attention_scores
    
    
    def training_step(self, batch, batch_idx):
        input_ids, mask, label = batch
        logits, attention_scores = self(input_ids, mask)

        loss = torch.nn.functional.cross_entropy(logits, label)

        self.log("train_loss", loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input_ids, mask, label = batch
        logits, attention_scores = self(input_ids, mask)

        loss = torch.nn.functional.cross_entropy(logits, label)
        acc = np.mean((torch.argmax(logits, dim=-1) == label).tolist())

        self.log_dict({
            "val_loss": loss,
            "val_acc": acc,
        })

    
    def test_step(self, batch, batch_idx):
        input_ids, mask, label = batch
        logits, attention_scores = self(input_ids, mask)

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
    data = NumberDataModule()
    data.prepare_data()
    
    model = NumFinder(data.input_vocab_size, data.output_vocab_size, 512, 512, 8, 0.0001)

    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=True),
            ModelCheckpoint(filename="{val_acc}-{val_loss}", monitor="val_loss", verbose=True, mode="min")
        ]
    )

    trainer.test(model, data)
    trainer.fit(model, data)
    trainer.test(model, data)
