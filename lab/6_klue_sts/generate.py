import os
import main as m

from transformers import AutoTokenizer


def main():
    args = m.load_args()
    
    CKPT_PATH = os.path.join(os.path.dirname(__file__))
    CKPT_FILE_NAME = 'epoch=7-step=1168.ckpt'
    
    tokenizer = AutoTokenizer.from_pretrained(
                                                args["model_name"],
                                                bos_token='</s>', 
                                                eos_token='</s>',
                                                unk_token='<unk>',
                                                pad_token='<pad>',
                                                mask_token='<mask>',
                                            )
    
    model = m.STSModel.load_from_checkpoint(os.path.join(CKPT_PATH, CKPT_FILE_NAME), tokenizer=tokenizer)
    model.eval()
    
    datamodule = m.KLUEDatamodule(args['dataset_path'], args['dataset_name'], tokenizer, args['batch_size'], args['num_workers'])
    datamodule.prepare_data()
    datamodule.setup()
    
    batch = next(iter(datamodule.test_dataloader()))
    
    logits = model(batch[0], batch[1])
    preds = logits.argmax(dim=-1)

    pred_sentences = tokenizer.batch_decode(preds, skip_special_tokens=True)

    for input_sentence, pred_sentence in zip(tokenizer.batch_decode(batch[0], skip_special_tokens=True), pred_sentences):
        print(f"input : {input_sentence}")
        print(f"pred : {pred_sentence}")
        print()


if __name__ == '__main__':
    main()
