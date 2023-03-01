import torch
import torchmetrics
import pytorch_lightning as pl
from transformers import top_k_top_p_filtering

import main as m
import glue_stsb as g


def main():
    with torch.no_grad():
        model, tokenizer, _ = m.copy_from_huggingface()
        tokenizer.pad_token = tokenizer.eos_token
        datamodule = g.GLUEDatamodule(model.max_seq_len, tokenizer)
        datamodule.prepare_data()
        datamodule.setup()
        
        model = g.MyModel(model)

        # model = model.to('cuda')
        
        for idx, batch in enumerate(iter(datamodule.test_dataloader())):
            print(idx)
            print(len(datamodule.test_dataloader()))
            
            input_ids, output_ids = batch

            # input_ids = input_ids.to('cuda')
            # output_ids = output_ids.to('cuda')
            
            
            logits = model(input_ids)
            
            filtered = top_k_top_p_filtering(logits, top_k=50, top_p=0.95)
            
            print(filtered.shape)


if __name__ == "__main__":
    main()
