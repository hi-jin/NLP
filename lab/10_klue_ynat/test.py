import argparse
import pytest

import torch
from transformers import BertModel, BertConfig

import bert


@pytest.fixture
def args():
    return argparse.Namespace(
        num_layers=10,
        dim_embed=2000,
        dim_model=200,
        num_heads=4,
        batch_size=64,
        dim_query=20,
        dim_key=20,
        dim_value=20,
    )


def test_attention_assertion_failure(args):
    try:
        bert.QKV_MultiHead_Attention(
                                        dim_embed=args.dim_embed,
                                        dim_model=3,
                                        num_heads=2,
                                    )
    except:
        assert True
    else:
        assert False, 'failed to check dim_model % num_heads == 0'


def test_attention_return_shape(args):
    attention = bert.QKV_MultiHead_Attention(
                                                dim_embed=args.dim_embed,
                                                dim_model=args.dim_model,
                                                num_heads=args.num_heads,
                                            )
    
    query = torch.randn(args.batch_size, args.dim_query, args.dim_embed)
    key = torch.randn(args.batch_size, args.dim_key, args.dim_embed)
    value = torch.randn(args.batch_size, args.dim_value, args.dim_embed)
    out, _ = attention(query, key, value)
    
    assert out.shape == query.shape == (args.batch_size, args.dim_query, args.dim_embed)


def test_encoder_return_shape(args):
    encoder = bert.EncoderLayer(
                                    dim_embed=args.dim_embed,
                                    dim_model=args.dim_model,
                                    num_heads=args.num_heads,
                                    dim_ff=args.dim_embed*4,
                                )

    sequences = torch.randn(args.batch_size, args.dim_key, args.dim_embed)
    out, _ = encoder(sequences)
    
    assert out.shape == sequences.shape


def test_multilayer_encoder_return_shape(args):
    encoder = bert.MultiLayerEncoder(
                                        num_layers=args.num_layers,
                                        dim_embed=args.dim_embed,
                                        dim_model=args.dim_model,
                                        num_heads=args.num_heads,
                                        dim_ff=args.dim_embed*4,
                                    )
    
    sequences = torch.randn(args.batch_size, args.dim_key, args.dim_embed)
    out, _ = encoder(sequences)
    
    assert out.shape == sequences.shape


def test_bert_implementation(args):
    model_name = 'bert-base-multilingual-cased'
    
    tokenizer, embeddings, encoder, pooler = bert.copy_from_huggingface(model_name)
    
    hg_bert = BertModel.from_pretrained(model_name)
    hg_config = BertConfig.from_pretrained(model_name)
    
    input_texts =   [
                        "this is a test text", 
                        "is it working?"
                    ]
                
    tokenized_ouptut = tokenizer(input_texts, max_length=hg_config.max_position_embeddings, padding="max_length")

    input_ids        = torch.tensor(tokenized_ouptut.input_ids)
    attention_mask = torch.tensor(tokenized_ouptut.attention_mask)
    token_type_ids   = torch.tensor(tokenized_ouptut.token_type_ids)

    with torch.no_grad():
        ## disable dropout -- huggingface
        hg_bert.eval()

        ## disable dropout -- my code
        embeddings.eval()
        pooler.eval()
        encoder.eval()

        seq_embs   = embeddings(input_ids) 
        output     = encoder(seq_embs, attention_mask)
        pooled_out = pooler(output[0]) 

        hg_output = hg_bert( 
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
                          )

        assert torch.all( torch.eq(hg_output.pooler_output, pooled_out) ), "Not same result!"