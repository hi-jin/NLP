import torch
import pytorch_lightning as pl
from datasets import load_dataset
import torchmetrics

import main as m

import gc


class GLUEDataset(torch.utils.data.Dataset):
    def __init__(
                    self,
                    sentence1,
                    sentence2,
                    max_seq_len,
                    tokenizer,
                ):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        
    
    def __len__(self):
        return len(self.sentence1)
    
    
    def __getitem__(self, idx):
        s1, s2 = self.sentence1[idx], self.sentence2[idx]
        ids = self.tokenizer(s1 + " " + s2, padding='max_length', truncation=True, max_length=self.max_seq_len-1).input_ids
        
        return (
                    torch.tensor([self.tokenizer.bos_token_id] + ids, dtype=torch.long),
                    torch.tensor(ids + [self.tokenizer.eos_token_id], dtype=torch.long)
                )


class GLUEDatamodule(pl.LightningDataModule):
    def __init__(
                    self,
                    max_seq_len,
                    tokenizer,
                ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    
    def prepare_data(self) -> None:
        load_dataset('glue', 'stsb')
    
    
    def setup(self, stage = None):
        all_data = load_dataset('glue', 'stsb')
        train_data = all_data['train']
        val_data = all_data['validation']
        test_data = all_data['test']

        self.train_dataset = GLUEDataset(train_data['sentence1'], train_data['sentence2'], self.max_seq_len, self.tokenizer)
        self.val_dataset = GLUEDataset(val_data['sentence1'], val_data['sentence2'], self.max_seq_len, self.tokenizer)
        self.test_dataset = GLUEDataset(test_data['sentence1'], test_data['sentence2'], self.max_seq_len, self.tokenizer)
    
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=16, num_workers=4, shuffle=True)
    
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=16, num_workers=4, shuffle=False)
    
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=16, num_workers=4, shuffle=False)


class MyModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    
    def forward(self, input_ids):
        logits = self.model(input_ids)
        return logits
    
    
    def training_step(self, batch, batch_idx):
        input_ids, output_ids = batch
        
        logits = self(input_ids)

        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), output_ids)
        self.log('train_loss', loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input_ids, output_ids = batch
        
        logits = self(input_ids)

        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), output_ids)
        self.log('val_loss', loss)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        input_ids, output_ids = batch
        
        logits = self(input_ids)

        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), output_ids)
        self.log('test_loss', loss)

        # TODO : gpu 메모리 관리 방법?
        # return {
        #     'test_logits': logits.argmax(dim=-1),
        #     'test_labels': output_ids,
        # }
    
    
    def test_epoch_end(self, outputs):
        logits, labels = [], []
        for output in outputs:
            logits.append(output['test_logits'])
            labels.append(output['test_labels'])
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
        
        bleu_1 = torchmetrics.functional.bleu_score(" ".join(map(str, logits.argmax(dim=-1))), " ".join(map(str, labels)), n_gram=1)
        bleu_2 = torchmetrics.functional.bleu_score(" ".join(map(str, logits.argmax(dim=-1))), " ".join(map(str, labels)), n_gram=2)
        bleu_3 = torchmetrics.functional.bleu_score(" ".join(map(str, logits.argmax(dim=-1))), " ".join(map(str, labels)), n_gram=3)
        bleu_4 = torchmetrics.functional.bleu_score(" ".join(map(str, logits.argmax(dim=-1))), " ".join(map(str, labels)), n_gram=4)
    
        print(f'bleu_1: {bleu_1}')
        print(f'bleu_2: {bleu_2}')
        print(f'bleu_3: {bleu_3}')
        print(f'bleu_4: {bleu_4}')

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)


def main():
    model, tokenizer, hg_model = m.copy_from_huggingface()
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # TODO : model의 입력 출력에 start / end를 언제는 붙이고 언제는 붙이지 않아도 되는지?
    # TODO : bos token, eos token 둘다 같은 거로 설정되어 있는데 이래도 되나?
    # tensor로 만드려면 padding은 필수인거같은데, 강의에서 말씀하신 패딩 필요없다는건 어떤 의미??
    
    datamodule = GLUEDatamodule(max_seq_len=model.max_seq_len, tokenizer=tokenizer)
    datamodule.prepare_data()
    
    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=3),
            pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, save_last=True),
        ],
        accelerator='gpu',
        devices=2,
        strategy='dp',
    )
    
    model = MyModel.load_from_checkpoint('./lightning_logs/version_2/checkpoints/epoch=1-step=360.ckpt', model=model)

    trainer.test(model, datamodule)
    # trainer.fit(model, datamodule)
    # trainer.test(model, datamodule)
    
    
if __name__ == '__main__':
    main()
