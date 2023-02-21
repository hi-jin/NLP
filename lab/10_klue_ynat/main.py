import argparse

import pandas as pd

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datasets import load_dataset
from transformers import BertConfig

import bert


class TopicLabeledDataset(torch.utils.data.Dataset):
    def __init__(
                    self,
                    titles,
                    labels,
                    tokenizer,
                    max_position_embeddings,
                ):
        super().__init__()
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_position_embeddings = max_position_embeddings
    
    
    def __len__(self):
        return len(self.titles)
    
    
    def __getitem__(self, idx):
        title = self.titles[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(title, max_length=self.max_position_embeddings, padding='max_length', truncation=True, return_tensors='pt')
        
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        token_type_ids = encoded.token_type_ids
        
        return input_ids.squeeze(0), attention_mask.squeeze(0), token_type_ids.squeeze(0), label


class YNATDatamodule(pl.LightningDataModule):
    def __init__(
                    self,
                    dataset_name,
                    dataset_subset_name,
                    tokenizer,
                    max_position_embeddings,
                    batch_size,
                    num_workers,
                ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_subset_name = dataset_subset_name
        self.tokenizer = tokenizer
        self.max_position_embeddings = max_position_embeddings
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    
    def prepare_data(self):
        load_dataset(self.dataset_name, self.dataset_subset_name)
    
    
    def setup(self, stage=None):
        all_data = load_dataset(self.dataset_name, self.dataset_subset_name)
        train_test_data = all_data['train'].train_test_split(test_size=0.2)
        train_data = train_test_data['train']
        test_data = train_test_data['test']
        val_data = all_data['validation']
        
        self.train_dataset = TopicLabeledDataset(
                                                    titles=train_data['title'],
                                                    labels=train_data['label'],
                                                    tokenizer=self.tokenizer,
                                                    max_position_embeddings=self.max_position_embeddings,
                                                )
        
        self.test_dataset = TopicLabeledDataset(
                                                    titles=test_data['title'],
                                                    labels=test_data['label'],
                                                    tokenizer=self.tokenizer,
                                                    max_position_embeddings=self.max_position_embeddings,
                                                )
        
        self.val_dataset = TopicLabeledDataset(
                                                    titles=val_data['title'],
                                                    labels=val_data['label'],
                                                    tokenizer=self.tokenizer,
                                                    max_position_embeddings=self.max_position_embeddings,
                                                )
    
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class TopicClassifier(pl.LightningModule):
    def __init__(
                    self,
                    num_classes,
                    dim_embed,
                    lr,
                    **kwargs
                ):
        super().__init__()
        if kwargs['eval'] is None or kwargs['eval'] is False:
            self.save_hyperparameters()
        self.num_classes = num_classes
        self.dim_embed = dim_embed
        self.lr = lr
        
        self.tokenizer, self.embeddings, self.encoder, self.pooler = bert.copy_from_huggingface('bert-base-multilingual-cased')

        self.fc1 = torch.nn.Linear(self.dim_embed, self.dim_embed // 2)
        self.fc2 = torch.nn.Linear(self.dim_embed // 2, self.num_classes)
        
    
    def forward(
                    self,
                    input_ids,
                    attention_mask,
                    token_type_ids,
                ):
        seq_embs = self.embeddings(input_ids)
        encoded = self.encoder(seq_embs, attention_mask)
        out = self.pooler(encoded[0])
        
        out = torch.nn.functional.gelu(self.fc1(out))
        logits = self.fc2(out)
        
        return logits
    

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        
        logits = self(input_ids, attention_mask, token_type_ids)
        
        labels  # (batch_size)
        logits  # (batch_size, num_classes)
        
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        self.log('train_loss', loss)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        
        logits = self(input_ids, attention_mask, token_type_ids)
        
        labels  # (batch_size)
        logits  # (batch_size, num_classes)
        
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        self.log('val_loss', loss)
        
        return {'val_logits': logits, 'val_labels': labels}
    
    
    def validation_epoch_end(self, outputs):
        val_logits = []
        val_labels = []
        for output in outputs:
            val_logits.append(output['val_logits'])
            val_labels.append(output['val_labels'])
        
        val_logits = torch.cat(val_logits, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        
        acc = torchmetrics.functional.accuracy(val_logits.argmax(dim=-1), val_labels, 'multiclass', num_classes=self.num_classes)
        self.log('val_acc', acc)
    
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        
        logits = self(input_ids, attention_mask, token_type_ids)
        
        labels  # (batch_size)
        logits  # (batch_size, num_classes)
        
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        self.log('test_loss', loss)
        
        return {'test_inputs': input_ids, 'test_logits': logits, 'test_labels': labels}
    
    
    def test_epoch_end(self, outputs):
        test_inputs = []
        test_logits = []
        test_labels = []
        for output in outputs:
            test_inputs.append(output['test_inputs'])
            test_logits.append(output['test_logits'])
            test_labels.append(output['test_labels'])
        
        test_inputs = torch.cat(test_inputs, dim=0)
        test_logits = torch.cat(test_logits, dim=0)
        test_labels = torch.cat(test_labels, dim=0)
        
        
        idx_to_label = ['과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']  
        text = self.tokenizer.batch_decode(test_inputs, skip_special_tokens=True)
        reference_label = list(map(lambda label: idx_to_label[label], test_labels))
        predicted_label = list(map(lambda label: idx_to_label[label], test_logits.argmax(dim=-1)))
        correct = (test_logits.argmax(dim=-1) == test_labels).tolist()
        
        pd.DataFrame({'text': text, 'reference_label': reference_label, 'predicted_label': predicted_label, 'correct': correct}).to_csv('test_result.csv', index=False, sep=',')


        acc = torchmetrics.functional.accuracy(test_logits.argmax(dim=-1), test_labels, 'multiclass', num_classes=self.num_classes)
        f1_score = torchmetrics.functional.f1_score(test_logits.argmax(dim=-1), test_labels, 'multiclass', num_classes=self.num_classes, average='macro')
        self.log('test_acc', acc)
        self.log('test_macro_f1_score', f1_score)

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def load_args():
    config = BertConfig.from_pretrained('bert-base-multilingual-cased')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dataset_name', type=str, default='klue')
    parser.add_argument('--dataset_subset_name', type=str, default='ynat')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--max_position_embeddings', type=int, default=config.max_position_embeddings)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--dim_embed', type=int, default=config.hidden_size)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--strategy', type=str, default='dp')
    args = vars(parser.parse_args())
    return args


def main():
    args = load_args()
    
    pl.seed_everything(args['seed'])
    
    tokenizer, embeddings, encoder, pooler = bert.copy_from_huggingface('bert-base-multilingual-cased')
    
    datamodule = YNATDatamodule(
                                    dataset_name=args['dataset_name'],
                                    dataset_subset_name=args['dataset_subset_name'],
                                    tokenizer=tokenizer,
                                    max_position_embeddings=args['max_position_embeddings'],
                                    batch_size=args['batch_size'],
                                    num_workers=args['num_workers'],
                                )
    datamodule.prepare_data()
    
    model = TopicClassifier(**args)
    
    trainer = pl.Trainer(
                            callbacks=[
                                EarlyStopping(monitor='val_loss', patience=3),
                                ModelCheckpoint(monitor='val_loss', save_top_k=1),
                            ],
                            accelerator=args['accelerator'],
                            devices=args['devices'],
                            strategy=args['strategy'],
                        )
    
    trainer.test(model, datamodule)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
