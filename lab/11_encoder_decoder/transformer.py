import typing as t
import os
import sys
import argparse
import math

import torch
import torchmetrics
import pytorch_lightning as pl


PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'


class NumberDataset(torch.utils.data.Dataset):
    def __init__(
                    self,
                    input_sequences: t.List[t.List[str]],
                    output_sequences: t.List[t.List[str]],
                    input_vocab: t.Dict[str, int],
                    output_vocab: t.Dict[str, int],
                    encoder_max_length: int,
                    decoder_max_length: int,
                ):
        super().__init__()
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
    
    
    def __len__(self):
        return len(self.input_sequences)
    

    def __fill_pads(self, sequence_ids: t.List[int], vocab: t.Dict[str, int], max_length: int) -> t.List[int]:
        mask = [1] * len(sequence_ids) + [0] * (max_length - len(sequence_ids))
        sequence_ids = sequence_ids + [vocab[PAD_TOKEN]] * (max_length - len(sequence_ids))

        return sequence_ids, mask
    
    
    def __getitem__(self, idx):
        input_sequence, output_sequence = self.input_sequences[idx], self.output_sequences[idx]
        
        encoder_input_ids = [self.input_vocab[token] for token in input_sequence]
        encoder_input_ids, encoder_input_mask = self.__fill_pads(encoder_input_ids, self.input_vocab, self.encoder_max_length)
        
        decoder_input_ids = [self.output_vocab[START_TOKEN]] + [self.output_vocab[token] for token in output_sequence]
        decoder_input_ids, decoder_input_mask = self.__fill_pads(decoder_input_ids, self.output_vocab, self.decoder_max_length)
        
        decoder_output_ids = [self.output_vocab[token] for token in output_sequence] + [self.output_vocab[END_TOKEN]]
        decoder_output_ids, decoder_output_mask = self.__fill_pads(decoder_output_ids, self.output_vocab, self.decoder_max_length)
        
        return (
            torch.tensor(encoder_input_ids, dtype=torch.long),
            torch.tensor(encoder_input_mask, dtype=torch.long),
            
            torch.tensor(decoder_input_ids, dtype=torch.long),
            
            torch.tensor(decoder_output_ids, dtype=torch.long),
            torch.tensor(decoder_output_mask, dtype=torch.long),
        )


class NumberDatamodule(pl.LightningDataModule):
    def __init__(
                    self,
                    train_file_path: str,
                    test_file_path: str,
                    encoder_max_length: int,
                    decoder_max_length: int,
                    batch_size: int,
                    num_workers: int,
                ):
        super().__init__()
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    
    def setup(self, stage: t.Optional[str] = None):
        self.train_data = self.__read_txt(self.train_file_path)
        self.test_data = self.__read_txt(self.test_file_path)
        
        self.input_vocab, self.output_vocab = self.__create_vocab(self.train_data)
        
        self.train_dataset = NumberDataset(
                                            input_sequences=self.train_data[0],
                                            output_sequences=self.train_data[1],
                                            input_vocab=self.input_vocab,
                                            output_vocab=self.output_vocab,
                                            encoder_max_length=self.encoder_max_length,
                                            decoder_max_length=self.decoder_max_length,
                                        )
        train_size = int(len(self.train_dataset) * 0.8)
        test_size = len(self.train_dataset) - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [train_size, test_size])
        self.test_dataset = NumberDataset(
                                            input_sequences=self.test_data[0],
                                            output_sequences=self.test_data[1],
                                            input_vocab=self.input_vocab,
                                            output_vocab=self.output_vocab,
                                            encoder_max_length=self.encoder_max_length,
                                            decoder_max_length=self.decoder_max_length,
                                        )
        
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    
    def __create_vocab(
                            self, 
                            data: t.Tuple[t.List[t.List[str]], t.List[t.List[str]]]
                        ) -> t.Tuple[t.Dict[str, int], t.Dict[str, int]]:
        input_tokens = []
        output_tokens = []
        
        input_sequences, output_sequences = data
        for input_sequence, output_sequence in zip(input_sequences, output_sequences):
            input_tokens.extend(input_sequence)
            output_tokens.extend(output_sequence)
        
        input_tokens = [PAD_TOKEN] + sorted(list(set(input_tokens)))
        output_tokens = [PAD_TOKEN, START_TOKEN, END_TOKEN] + sorted(list(set(output_tokens)))

        input_vocab = {token: idx for idx, token in enumerate(input_tokens)}
        output_vocab = {token: idx for idx, token in enumerate(output_tokens)}

        return input_vocab, output_vocab
    
    
    def __read_txt(self, file_path: str) -> t.Tuple[t.List[t.List[str]], t.List[t.List[str]]]:
        input_sequences = []
        output_sequences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                input_sequence, output_sequence = line.split('\t')
                input_sequence = input_sequence.split(',')
                output_sequence = output_sequence.split(',')

                input_sequences.append(input_sequence)
                output_sequences.append(output_sequence)
        return (
                    input_sequences,  # (batch_size, sequence_length)
                    output_sequences  # (batch_size, sequence_length)
                )
                

class QKV_MultiHead_Attention(torch.nn.Module):
    def __init__(
                    self,
                    d_embed: int,
                    d_model: int,
                    num_heads: int,
                ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads

        self.wq = torch.nn.Linear(d_embed, d_model)
        self.wk = torch.nn.Linear(d_embed, d_model)
        self.wv = torch.nn.Linear(d_embed, d_model)
        self.wo = torch.nn.Linear(d_model, d_embed)
    

    def __split_heads(
                        self,
                        x: torch.Tensor,  # (batch_size, x_length, d_model)
                    ):
        batch_size, x_length, d_model = x.shape
        x = x.reshape(batch_size, x_length, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        return x  # (batch_size, num_heads, x_length, d_k)
    
    
    def forward(
                    self,
                    query,  # (batch_size, query_length, d_embed)
                    key,  # (batch_size, key_length, d_embed)
                    value,  # (batch_size, key_length, d_embed)
                    attention_mask,  # (batch_size, 1, 1, key_length) || (key_length, key_length)
                ):
        Q = self.__split_heads(self.wq(query))
        K = self.__split_heads(self.wk(key))
        V = self.__split_heads(self.wv(value))
        
        reactivity = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, query_length, key_length)
        
        attention_mask = 1 - attention_mask
        attention_mask = attention_mask.masked_fill(attention_mask == 1, -sys.maxsize-1)
        reactivity += attention_mask
        reactivity /= math.sqrt(reactivity.shape[-1])
        
        attention_scores = torch.nn.functional.softmax(reactivity, dim=-1)
        
        blended_vector = torch.matmul(attention_scores, V)  # (batch_size, num_heads, query_length, d_k)
        
        batch_size, num_heads, query_length, d_k = blended_vector.shape
        blended_vector = blended_vector.transpose(1, 2).reshape(batch_size, query_length, self.d_model)
        blended_vector = self.wo(blended_vector)  # (batch_size, query_length, d_embed)

        return blended_vector, attention_scores


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
                    self,
                    d_embed: int,
                    d_model: int,
                    num_heads: int,
                    d_intermediate: int,
                ):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_intermediate = d_intermediate
        
        self.attention = QKV_MultiHead_Attention(
                                                    d_embed=self.d_embed,
                                                    d_model=self.d_model,
                                                    num_heads=self.num_heads,
                                                )

        self.fc1 = torch.nn.Linear(self.d_embed, self.d_intermediate)
        self.fc2 = torch.nn.Linear(self.d_intermediate, self.d_embed)

        self.norm1 = torch.nn.LayerNorm(self.d_embed)
        self.norm2 = torch.nn.LayerNorm(self.d_embed)
        


    def forward(
                    self,
                    x,  # (batch_size, encoder_max_length, d_embed)
                    attention_mask,  # (batch_size, encoder_max_length)
                ):
        attention_mask = attention_mask[:, None, None, :]
        
        residual = x
        x, attention_scores = self.attention(
                                                query=x,
                                                key=x,
                                                value=x,
                                                attention_mask=attention_mask,
                                            )
        x += residual
        x = self.norm1(x)
        
        residual = x
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        x += residual
        x = self.norm2(x)
        
        return (
                    x,  # (batch_size, encoder_max_length, d_embed)
                    attention_scores  # (batch_size, num_heads, encoder_max_length, encoder_max_length)
                )


class TransformerEncoder(torch.nn.Module):
    def __init__(
                    self,
                    num_layers,
                    d_embed,
                    d_model,
                    num_heads,
                    d_intermediate,
                ):
        super().__init__()
        self.num_layers = num_layers
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_intermediate = d_intermediate
        
        self.layers = torch.nn.ModuleList([
                    TransformerEncoderLayer(
                                                d_embed=self.d_embed,
                                                d_model=self.d_model,
                                                num_heads=self.num_heads,
                                                d_intermediate=self.d_intermediate,
                                            )
                    for _ in range(self.num_layers)
                ])
    
    
    def forward(
                    self,
                    x,  # (batch_size, encoder_max_length, d_embed)
                    attention_mask,  # (batch_size, encoder_max_length)
                ):
        layers_attention_scores = []
        for layer in self.layers:
            x, attention_scores = layer(x=x, attention_mask=attention_mask)
            layers_attention_scores.append(attention_scores)
        return (
                    x,  # (batch_size, encoder_max_length, d_embed)
                    layers_attention_scores  # (num_layers, batch_size, num_heads, encoder_max_length, encoder_max_length)
                )


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(
                    self,
                    d_embed: int,
                    d_model: int,
                    num_heads: int,
                    d_intermediate: int,
                ):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_intermediate = d_intermediate
        
        self.self_attention = QKV_MultiHead_Attention(
                                                        d_embed=self.d_embed,
                                                        d_model=self.d_model,
                                                        num_heads=self.num_heads,
                                                    )
        self.cross_attention = QKV_MultiHead_Attention(
                                                        d_embed=self.d_embed,
                                                        d_model=self.d_model,
                                                        num_heads=self.num_heads,
                                                    )
        self.fc1 = torch.nn.Linear(self.d_embed, self.d_intermediate)
        self.fc2 = torch.nn.Linear(self.d_intermediate, self.d_embed)
        
        self.norm1 = torch.nn.LayerNorm(self.d_embed)
        self.norm2 = torch.nn.LayerNorm(self.d_embed)
        self.norm3 = torch.nn.LayerNorm(self.d_embed)
        
    
    
    def forward(
                    self,
                    x,  # (batch_size, decoder_max_length, d_embed)
                    look_ahead_mask,  # (decoder_max_length, decoder_max_length)
                    encoder_output,  # (batch_size, encoder_max_length, d_embed)
                    encoder_input_attention_mask,  # (batch_size, encoder_max_length)
                ):
        residual = x
        x, attention_scores = self.self_attention(
                                                    query=x,
                                                    key=x,
                                                    value=x,
                                                    attention_mask=look_ahead_mask,
                                                )
        x += residual
        x = self.norm1(x)
        
        residual = x
        x, cross_attention_scores = self.cross_attention(
                                                            query=x,
                                                            key=encoder_output,
                                                            value=encoder_output,
                                                            attention_mask=encoder_input_attention_mask[:, None, None, :],
                                                        )
        x += residual
        x = self.norm2(x)

        residual = x
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        x += residual
        x = self.norm3(x)
        
        return (
                    x,  # (batch_size, decoder_max_length, d_embed)
                    attention_scores,  # (batch_size, num_heads, decoder_max_length, decoder_max_length)
                    cross_attention_scores  # (batch_size, num_heads, decoder_max_length, encoder_max_length)
                )


class TransformerDecoder(torch.nn.Module):
    def __init__(
                    self,
                    num_layers: int,
                    d_embed: int,
                    d_model: int,
                    num_heads: int,
                    d_intermediate: int,
                ):
        super().__init__()
        self.num_layers = num_layers
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_intermediate = d_intermediate
        
        self.layers = torch.nn.ModuleList([
                    TransformerDecoderLayer(
                                                d_embed=self.d_embed,
                                                d_model=self.d_model,
                                                num_heads=self.num_heads,
                                                d_intermediate=self.d_intermediate,
                                            )
                    for _ in range(self.num_layers)
                ])
    
    
    def forward(
                    self,
                    x,  # (batch_size, decoder_max_length, d_embed)
                    look_ahead_mask,  # (decoder_max_length, decoder_max_length)
                    encoder_output,  # (batch_size, encoder_max_length, d_embed)
                    encoder_input_attention_mask,  # (batch_size, encoder_max_length)
                ):
        layers_attention_scores = []
        layers_cross_attention_scores = []
        
        for layer in self.layers:
            x, attention_scores, cross_attention_scores = layer(
                                                                    x=x,
                                                                    look_ahead_mask=look_ahead_mask,
                                                                    encoder_output=encoder_output,
                                                                    encoder_input_attention_mask=encoder_input_attention_mask,
                                                                )
            layers_attention_scores.append(attention_scores)
            layers_cross_attention_scores.append(cross_attention_scores)
        
        return (
                    x,  # (batch_size, decoder_max_length, d_embed)
                    layers_attention_scores,  # (num_layers, batch_size, num_heads, decoder_max_length, decoder_max_length)
                    layers_cross_attention_scores  # (num_layers, batch_size, num_heads, decoder_max_length, encoder_max_length)
                )


class Transformer(torch.nn.Module):
    def __init__(
                    self,
                    num_layers: int,
                    d_embed: int,
                    d_model: int,
                    num_heads: int,
                    d_intermediate: int,
                ):
        super().__init__()
        self.num_layers = num_layers
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_intermediate = d_intermediate
        
        self.encoder = TransformerEncoder(
                                            num_layers=self.num_layers,
                                            d_embed=self.d_embed,
                                            d_model=self.d_model,
                                            num_heads=self.num_heads,
                                            d_intermediate=self.d_intermediate,
                                        )
        self.decoder = TransformerDecoder(
                                            num_layers=self.num_layers,
                                            d_embed=self.d_embed,
                                            d_model=self.d_model,
                                            num_heads=self.num_heads,
                                            d_intermediate=self.d_intermediate,
                                        )
    
    
    def forward(
                    self,
                    encoder_input_embeds,  # (batch_size, encoder_max_length, d_embed)
                    encoder_input_attention_mask,  # (batch_size, encoder_max_length)
                    decoder_input_embeds,  # (batch_size, decoder_max_length, d_embed)
                ):
        encoder_output, encoder_layers_attention_scores = self.encoder(
                                                                        x=encoder_input_embeds,
                                                                        attention_mask=encoder_input_attention_mask,
                                                                    )

        decoder_max_length = decoder_input_embeds.shape[1]
        look_ahead_mask = 1 - torch.triu(torch.ones(decoder_max_length, decoder_max_length, dtype=torch.long), diagonal=1)
        look_ahead_mask = look_ahead_mask.to(decoder_input_embeds.device)

        decoder_output, \
        decoder_layers_attention_scores, \
        decoder_layers_cross_attention_scores = self.decoder(
                                                                x=decoder_input_embeds,
                                                                look_ahead_mask=look_ahead_mask,
                                                                encoder_output=encoder_output,
                                                                encoder_input_attention_mask=encoder_input_attention_mask,
                                                            )
        
        return (
            decoder_output,  # (batch_size, decoder_max_length, d_embed)
            (
                encoder_layers_attention_scores,
                decoder_layers_attention_scores,
                decoder_layers_cross_attention_scores
            )
        )


class NumberSorter(pl.LightningModule):
    def __init__(
                    self,
                    input_vocab_size: int,
                    output_vocab_size: int,
                    num_layers: int,
                    d_embed: int,
                    d_model: int,
                    num_heads: int,
                    d_intermediate: int,
                    output_pad_id: int,
                    lr: float,
                ):
        super().__init__()
        self.save_hyperparameters()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.num_layers = num_layers
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_intermediate = d_intermediate
        self.output_pad_id = output_pad_id
        self.lr = lr
        
        self.input_embedding = torch.nn.Embedding(self.input_vocab_size, self.d_embed)
        self.output_embedding = torch.nn.Embedding(self.output_vocab_size, self.d_embed)
        self.transformer = Transformer(
                                        num_layers=self.num_layers,
                                        d_embed=self.d_embed,
                                        d_model=self.d_model,
                                        num_heads=self.num_heads,
                                        d_intermediate=self.d_intermediate,
                                    )
        self.to_output = torch.nn.Linear(self.d_embed, self.output_vocab_size)
    
    
    def forward(
                    self,
                    encoder_input_ids,  # (batch_size, encoder_max_length)
                    encoder_input_attention_mask,  # (batch_size, encoder_max_length)
                    decoder_input_ids,  # (batch_size, decoder_max_length)
                ):
        encoder_input_embeds = self.input_embedding(encoder_input_ids)
        decoder_input_embeds = self.output_embedding(decoder_input_ids)
        decoder_output, _ = self.transformer(
                                                encoder_input_embeds=encoder_input_embeds,
                                                encoder_input_attention_mask=encoder_input_attention_mask,
                                                decoder_input_embeds=decoder_input_embeds,
                                            )
        
        logits = self.to_output(decoder_output)
        return logits  # (batch_size, decoder_max_length, output_vocab_size)
    
    
    def training_step(self, batch, batch_idx):
        encoder_input_ids, encoder_input_mask, decoder_input_ids, decoder_output_ids, decoder_output_mask = batch
        
        logits = self(
                        encoder_input_ids=encoder_input_ids,
                        encoder_input_attention_mask=encoder_input_mask,
                        decoder_input_ids=decoder_input_ids,
                    )
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), decoder_output_ids, ignore_index=self.output_pad_id)
        self.log('train_loss', loss)
        return loss

    
    def validation_step(self, batch, batch_idx):
        encoder_input_ids, encoder_input_mask, decoder_input_ids, decoder_output_ids, decoder_output_mask = batch
        
        logits = self(
                        encoder_input_ids=encoder_input_ids,
                        encoder_input_attention_mask=encoder_input_mask,
                        decoder_input_ids=decoder_input_ids,
                    )
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), decoder_output_ids, ignore_index=self.output_pad_id)
        
        self.log('val_loss', loss)
    
    
    def test_step(self, batch, batch_idx):
        encoder_input_ids, encoder_input_mask, decoder_input_ids, decoder_output_ids, decoder_output_mask = batch
        
        logits = self(
                        encoder_input_ids=encoder_input_ids,
                        encoder_input_attention_mask=encoder_input_mask,
                        decoder_input_ids=decoder_input_ids,
                    )
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), decoder_output_ids, ignore_index=self.output_pad_id)
        
        self.log('test_loss', loss)
        
        return {
                    'test_logits': logits,
                    'test_output_ids': decoder_output_ids,
                }
    
    
    def test_epoch_end(self, outputs):
        logits = []
        output_ids = []
        for output in outputs:
            logits.append(output['test_logits'])
            output_ids.append(output['test_output_ids'])
        logits = torch.cat(logits, dim=0)  # (batch_size, decoder_max_length, output_vocab_size)
        output_ids = torch.cat(output_ids, dim=0)  # (batch_size, decoder_max_length)

        with open('result.txt', 'w') as f:
            for pred, output_id in zip(logits.argmax(dim=-1), output_ids):
                f.write(f'pred: {",".join(map(str, pred.tolist()))}\n')
                f.write(f'label: {",".join(map(str, output_id.tolist()))}\n\n')
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', type=str, default=os.path.join(os.path.dirname(__file__), '../../dataset/sorted_numbers', 'train.txt'))
    parser.add_argument('--test_file_path', type=str, default=os.path.join(os.path.dirname(__file__), '../../dataset/sorted_numbers', 'test.txt'))
    parser.add_argument('--encoder_max_length', type=int, default=20)
    parser.add_argument('--decoder_max_length', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--d_embed', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    kwargs = vars(parser.parse_args())

    datamodule = NumberDatamodule(
                                    train_file_path=kwargs['train_file_path'],
                                    test_file_path=kwargs['test_file_path'],
                                    encoder_max_length=kwargs['encoder_max_length'],
                                    decoder_max_length=kwargs['decoder_max_length'],
                                    batch_size=kwargs['batch_size'],
                                    num_workers=kwargs['num_workers'],
                                )
    datamodule.setup()
    
    model = NumberSorter(
                            input_vocab_size=len(datamodule.input_vocab),
                            output_vocab_size=len(datamodule.output_vocab),
                            num_layers=kwargs['num_layers'],
                            d_embed=kwargs['d_embed'],
                            d_model=kwargs['d_model'],
                            num_heads=kwargs['num_heads'],
                            d_intermediate=kwargs['d_model']*4,
                            output_pad_id=datamodule.output_vocab[PAD_TOKEN],
                            lr=kwargs['lr'],
                        )
    
    trainer = pl.Trainer(
                            callbacks=[
                                pl.callbacks.EarlyStopping(monitor='val_loss', patience=3),
                                pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, save_last=True),
                            ],
                            accelerator='gpu',
                            devices=2,
                            strategy='dp',
                        )
    
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
