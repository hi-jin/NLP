import pytorch_lightning as pl

import main as m
import bert


def main():
    args = m.load_args()
    
    pl.seed_everything(args['seed'])
    
    tokenizer, _, _, _ = bert.copy_from_huggingface('bert-base-multilingual-cased')
    model = m.TopicClassifier.load_from_checkpoint('./lightning_logs/version_0/checkpoints/epoch=3-step=2284.ckpt', eval=True, **args)
    model.eval()
    
    datamodule = m.YNATDatamodule(
                                    dataset_name=args['dataset_name'],
                                    dataset_subset_name=args['dataset_subset_name'],
                                    tokenizer=tokenizer,
                                    max_position_embeddings=args['max_position_embeddings'],
                                    batch_size=args['batch_size'],
                                    num_workers=args['num_workers'],
                                )

    trainer = pl.Trainer(
                            accelerator=args['accelerator'],
                            devices=args['devices'],
                            strategy=args['strategy'],
                        )

    trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
