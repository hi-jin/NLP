import os
import argparse

import numpy as np
import pandas as pd

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import seaborn as sns
import matplotlib.pyplot as plt


class NaverReviewDataset(torch.utils.data.Dataset):
    def __init__(self,
                 docs: np.ndarray,
                 labels: np.ndarray,
                 input_vocab,
                 max_sequence_len,
                 **kwargs) -> None:
        super().__init__()
        self.docs = docs
        self.labels = labels
        self.input_vocab = input_vocab
        self.max_sequence_len = max_sequence_len
    
    
    def __len__(self):
        return len(self.docs)
    
    
    def __getitem__(self, index):
        doc, label = self.docs[index], self.labels[index]
        
        ### create sequence : list[str] -> list[int]
        unk_id = self.input_vocab["[UNK]"]
        sequence = [(self.input_vocab[token] if token in self.input_vocab else unk_id) for token in doc]
        
        ### add padding
        pad_id = self.input_vocab["[PAD]"]
        num_to_fill = self.max_sequence_len - len(sequence)
        weight = [1] * len(sequence) + [0] * num_to_fill  # mask
        sequence = sequence + [pad_id] * num_to_fill
        
        return (
            torch.tensor(sequence, dtype=torch.long),  # [max_sequence_len]
            torch.tensor(weight, dtype=torch.long),  # [max_sequence_len]
            
            torch.tensor(label, dtype=torch.float32)  # 1 (scalar)
        )
        


class NaverReviewDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 train_file_name,
                 test_file_name,
                 batch_size,
                 num_workers,
                 **kwargs) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    
    def prepare_data(self) -> None:
        train_df = self.__read_txt(os.path.join(self.data_dir, self.train_file_name))
        test_df = self.__read_txt(os.path.join(self.data_dir, self.test_file_name))
        
        ### drop nan
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        
        ### docs, labels(0 | 1)
        self.train_docs, self.train_labels = self.__get_docs_and_labels(train_df)
        self.test_docs, self.test_labels = self.__get_docs_and_labels(test_df)
        
        self.input_vocab = self.__create_vocab(self.train_docs)
        self.max_sequence_len = max(
            self.__get_max_sequence_len(self.train_docs),
            self.__get_max_sequence_len(self.test_docs)
        )


    def setup(self, stage):
        if stage == "fit":
            all_train_dataset = NaverReviewDataset(self.train_docs,
                                                    self.train_labels,
                                                    self.input_vocab,
                                                    self.max_sequence_len)
            all_size = len(all_train_dataset)
            train_size = int(0.8 * all_size)
            val_size = all_size - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(all_train_dataset, [train_size, val_size])
        if stage == "test":
            self.test_dataset = NaverReviewDataset(self.test_docs,
                                                   self.test_labels,
                                                   self.input_vocab,
                                                   self.max_sequence_len)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def __create_vocab(self, docs: np.ndarray):
        words = set()
        for doc in docs:  # TODO 제대로 된 전처리
            for word in doc:
                words.add(word)
        words = sorted(list(words))
        
        words = ["[PAD]", "[UNK]"] + words
        
        vocab = { token: index for index, token in enumerate(words) }
        return vocab

    
    def __get_max_sequence_len(self, docs: np.ndarray):
        max_sequence_len = 0
        for doc in docs:
            max_sequence_len = max(max_sequence_len, len(doc))
        return max_sequence_len


    def __get_docs_and_labels(self, df: pd.DataFrame):
        docs = df["document"].to_numpy()
        labels = df["label"].to_numpy()
        
        splitted_docs = []
        for doc in docs:
            splitted_docs.append(doc.split())
        
        return np.array(splitted_docs, dtype=object), labels
        
    
    def __read_txt(self, file_path) -> pd.DataFrame:
        df = pd.read_table(file_path)
        return df


class BahdanauAttention(torch.nn.Module):
    def __init__(self,
                 sequence_dim,
                 attention_dim) -> None:
        super().__init__()
        self.sequence_dim = sequence_dim
        self.attention_dim = attention_dim
        
        ### layers
        # self.W  # NOTE In this model, queries are not used.
        self.U = torch.nn.Linear(self.sequence_dim, self.attention_dim, bias=False)
        self.v = torch.nn.Linear(self.attention_dim, 1, bias=False)
    
    
    def forward(self,
                sequences,  # [batch_size, max_sequence_len, sequence_dim]
                weights  # [batch_size, max_sequence_len]
                ):
        projected_sequences = self.U(sequences)  # [batch_size, max_sequence_len, attention_dim]
        reactivity = torch.tanh(self.v(projected_sequences))  # [batch_size, max_sequence_len, 1]
        reactivity = reactivity.squeeze(-1)  # [batch_size, max_sequence_len]
        
        ### ignore padded values
        reactivity = reactivity.masked_fill(weights == 0, -float("inf"))
        
        attention_scores = torch.nn.functional.softmax(reactivity, dim=1).unsqueeze(1)  # [batch_size, 1, max_sequence_len]
        
        blendded_vector = torch.matmul(attention_scores, sequences)  # [batch_size, 1, sequence_dim]
        blendded_vector = blendded_vector.squeeze(1)  # [batch_size, sequence_dim]
        attention_scores = attention_scores.squeeze(1)  # [batch_size, max_sequence_len]
        
        return blendded_vector, attention_scores


class ReviewClassifier(pl.LightningModule):
    def __init__(self,
                 num_embeds,  # len(input_vocab)
                 embed_dim,  # embed_dim == sequence_dim
                 attention_dim,
                 lr,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.num_embeds = num_embeds
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.lr = lr
        
        self.embed = torch.nn.Embedding(self.num_embeds, self.embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(self.embed_dim, self.attention_dim)
        self.to_output = torch.nn.Linear(self.embed_dim, 1)
        
        self.criterion = torch.nn.BCELoss()
    
    
    def forward(self, 
                sequences,  # [batch_size, max_sequence_len]
                weights  # [batch_size, max_sequence_len]
                ):
        sequences = self.embed(sequences)  # [batch_size, max_sequence_len, embed_dim]
        
        blendded_vector, attention_scores = self.attention(sequences, weights)
        # blendded_vector : [batch_size, embed_dim]
        # attention_scores : [batch_size, max_sequence_len]
        
        logits = self.to_output(blendded_vector)  # [batch_size, 1]
        probs = torch.sigmoid(logits).squeeze(1)
        return probs, attention_scores  # [batch_size]
    
    
    def training_step(self, batch, batch_idx):
        sequences, weights, labels = batch
        probs, _ = self(sequences, weights)
        
        loss = self.criterion(probs, labels)
        
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        sequences, weights, labels = batch
        probs, _ = self(sequences, weights)
        
        loss = self.criterion(probs, labels)
        acc = torchmetrics.functional.accuracy(probs, labels, "binary")
        
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics
        
    
    def test_step(self, batch, batch_idx):
        sequences, weights, labels = batch
        probs, _ = self(sequences, weights)
        
        loss = self.criterion(probs, labels)
        acc = torchmetrics.functional.accuracy(probs, labels, "binary")
        
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer
    
    @staticmethod
    def add_model_specifig_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("ReviewClassifier")
        parser.add_argument("--embed_dim", type=int, default=10)
        parser.add_argument("--attention_dim", type=int, default=10)
        parser.add_argument("--lr", type=float, default=0.0001)
        return parent_parser


def train(args):
    # ------
    # trainer
    # ------
    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor="val_acc", mode="max", verbose=True, patience=3)],
        # accelerator="gpu",
        # devices=4,
        # strategy="ddp"
    )
    
    # ------
    # dataset
    # ------
    data = NaverReviewDataModule(**args)
    data.prepare_data()
    
    # ------
    # model
    # ------
    model = ReviewClassifier(num_embeds=len(data.input_vocab), **args)
    
    # ------
    # test, train
    # ------
    trainer.test(model, data)
    trainer.fit(model, data)
    
    # ------
    # result
    # ------
    trainer.test(model, data)
    

def result(args, checkpoint_file_path):
    data = NaverReviewDataModule(**args)
    data.prepare_data()
    data.setup("test")
    
    model = ReviewClassifier.load_from_checkpoint(checkpoint_file_path)
    
    with torch.no_grad():
        id_to_word = { data.input_vocab[token]: token for token in data.input_vocab.keys() }
        
        result_arr = []
        
        for sequences, weights, labels in data.test_dataloader():
            probs, attention_scores = model(sequences, weights)
            
            batch_size = len(sequences)
            for i in range(batch_size):
                sequence, weight, label = sequences[i], weights[i], labels[i]
                prob, attention_score = probs[i], attention_scores[i]
                
                sentence = []
                for j in range(len(sequence)):
                    # if weight[j] == 0:
                    #     break
                    sentence.append(id_to_word[int(sequence[j])])
                sentence = " ".join(sentence)
                
                important_idx = np.argmax(attention_score)
                
                loss = abs(label - prob)
                result_arr.append({
                    "sentence": sentence,
                    "label": int(label),
                    "loss": float(loss),
                    "prob": float(prob),
                    "attention_score": attention_score,
                    "important_word": id_to_word[int(sequence[important_idx])],
                })

        result_arr.sort(key=lambda obj: obj["loss"])
        
        def print_result_arr(result_arr):
            for result_obj in result_arr:
                print(f"원문 : { result_obj['sentence'] }")
                print(f"라벨 : { '긍정' if int(result_obj['label']) == 1 else '부정' }")
                print(f"오차: { result_obj['loss'] }")
                if result_obj["prob"] >= 0.5:
                    print(f"모델의 분류 결과 : 긍정 { int(result_obj['prob'] * 100) }%")
                else:
                    print(f"모델의 분류 결과 : 부정 { 100 - int(result_obj['prob'] * 100) }%")
                print(f"모델이 판별한 주요 단어 : { result_obj['important_word'] }")
                print()

        high_score = result_arr[160:170]
        low_score = result_arr[-300:-290]

        print("----------상위 10개----------")
        print_result_arr(high_score)
        print("---------------------------")
        print()
        print("----------하위 10개----------")
        print_result_arr(low_score)
        print("---------------------------")
        
        high_score_df = pd.DataFrame([list(map(float, obj["attention_score"])) for obj in high_score], columns=[i for i in range(data.max_sequence_len)])
        low_score_df = pd.DataFrame([list(map(float, obj["attention_score"])) for obj in low_score], columns=[i for i in range(data.max_sequence_len)])
        
        plt.rcParams.update({
            "font.family": "AppleGothic"
        })

        heatmap_text_data = np.array([obj["sentence"].split() for obj in high_score]).T
        fig, ax = plt.subplots(figsize=(30, 100))
        ax = sns.heatmap(high_score_df.T, annot=heatmap_text_data, fmt="", annot_kws={"size": 20}, linewidths=.5, cmap="YlGnBu")
        fig.savefig("3_naver_review_figure_high_score.png")

        heatmap_text_data = np.array([obj["sentence"].split() for obj in low_score]).T
        fig, ax = plt.subplots(figsize=(30, 100))
        ax = sns.heatmap(low_score_df.T, annot=heatmap_text_data, fmt="", annot_kws={"size": 20}, linewidths=.5, cmap="YlGnBu")
        fig.savefig("3_naver_review_figure_low_score.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../dataset/naver_review")
    parser.add_argument("--train_file_name", type=str, default="ratings_train.txt")
    parser.add_argument("--test_file_name", type=str, default="ratings_test.txt")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=1)
    parser = ReviewClassifier.add_model_specifig_args(parser)
    args = vars(parser.parse_args())
    
    return args


if __name__ == "__main__":
    # ------
    # settings
    # ------
    pl.seed_everything(1234)
    os.chdir(os.path.dirname(__file__))
    args = parse_args()
    
    # ------
    # train
    # ------
    # train(args)
    
    # ------
    # print results
    # ------
    result(args, "3_naver_review_checkpoint.ckpt")
