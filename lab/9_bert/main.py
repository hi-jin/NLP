import sys
import copy
import math

import numpy as np

import torch
import pytorch_lightning as pl

from transformers import BertModel, BertTokenizer, BertConfig


# Embedding and Pooling
# this embedding is from huggingface
class BertEmbeddings(torch.nn.Module):
    """ this embedding moudles are from huggingface implementation
        but, it is simplified for just testing 
    """

    def __init__(self, vocab_size, hidden_size, pad_token_id, max_bert_length_size, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.word_embeddings        = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings    = torch.nn.Embedding(max_bert_length_size, hidden_size)
        self.token_type_embeddings  = torch.nn.Embedding(2, hidden_size) # why 2 ? 0 and 1 

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout   = torch.nn.Dropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_bert_length_size).expand((1, -1)))
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

        # always absolute
        self.position_embedding_type = "absolute"

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def cp_weight(source, target, include_eps=False):
    assert source.weight.size() == target.weight.size()
    target.load_state_dict(source.state_dict())
    
    if include_eps:
        with torch.no_grad():
            target.eps = source.eps


# this pooler is from huggingface
class BertPooler(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Attention(torch.nn.Module):
    def __init__(
                    self,
                    d_embed,
                    d_model,
                    num_heads,
                ) -> None:
        super().__init__()

        assert d_model % num_heads == 0
        
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        
        self.wq = torch.nn.Linear(self.d_embed, self.d_model)
        self.wk = torch.nn.Linear(self.d_embed, self.d_model)
        self.wv = torch.nn.Linear(self.d_embed, self.d_model)
        self.wo = torch.nn.Linear(self.d_model, d_embed)
    
    
    def split_heads(
                        self,
                        x: torch.Tensor,  # [batch_size, d_x, d_model]
                    ):
        batch_size, d_x, d_model = x.shape
        x = x.view(batch_size, d_x, self.num_heads, self.d_k)
        x = x.contiguous().transpose(1, 2)  # [batch_size, num_heads, d_x, d_k]
        return x
    
    
    def forward(
                    self,
                    query,  # [batch_size, d_query, d_embed]
                    key,  # [batch_size, d_key, d_embed]
                    value,  # [batch_size, d_value, d_embed]
                    mask: torch.Tensor,  # [batch_size, d_key]
                ):
        Q = self.split_heads(self.wq(query))  # [batch_size, num_heads, d_query, d_k]
        K = self.split_heads(self.wk(key))  # [batch_size, num_heads, d_key, d_k]
        V = self.split_heads(self.wv(value))  # [batch_size, num_heads, d_value, d_k]

        reactivity_scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.d_k)  # [batch_size, num_heads, d_query, d_key]
        mask = mask[:, None, None, :]  # [batch_size, 1, 1, d_key]
        mask = 1 - mask
        mask = mask.masked_fill(mask.bool(), -sys.maxsize-1)
        reactivity_scores += mask
        
        attention_scores = torch.nn.functional.softmax(reactivity_scores, dim=-1)
        
        # d_key == d_value
        blended_vector = torch.matmul(attention_scores, V)  # [batch_size, num_heads, d_query, d_k]
        batch_size, num_heads, d_query, d_k = blended_vector.shape

        concatenated_vector = blended_vector.transpose(1, 2).contiguous().view(batch_size, d_query, self.d_model)

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
        
        # TODO : 여기서 normalization을 두 개 만드셨는데, weight가 따로 없는 것 같은데 하나로 통일하면 안되나??
        self.norm1 = torch.nn.LayerNorm(self.d_embed)
        self.norm2 = torch.nn.LayerNorm(self.d_embed)
        
        self.fc1 = torch.nn.Linear(self.d_embed, self.d_embed * 4)
        self.fc2 = torch.nn.Linear(self.d_embed * 4, self.d_embed)
    
    
    def forward(
                    self,
                    input_embeds,  # [batch_size, d_input, d_embed]
                    mask,  # [batch_size, d_input]
                ):
        x = input_embeds
        
        residual = x
        x, attention_scores = self.attention(x, x, x, mask)
        x = residual + x
        x = self.norm1(x)
        
        residual = x
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        x = self.norm2(x)
        
        return x, attention_scores


class MultiLayerEncoder(torch.nn.Module):
    def __init__(
                    self,
                    d_embed: int,
                    d_model: int,
                    num_heads: int,
                    num_layers: int,
                 ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        encoder = Encoder(self.d_embed, self.d_model, self.num_heads)
        
        self.encoder_layers = torch.nn.ModuleList([
            copy.deepcopy(encoder) for _ in range(self.num_layers)
        ])

    
    def forward(
                    self,
                    input_embeds,
                    mask,
                ):
        x = input_embeds
        
        layers_attention_scores = []
        
        for layer in self.encoder_layers:
            x, attention_scores = layer(x, mask)
            layers_attention_scores.append(attention_scores)
        
        return x, layers_attention_scores


if __name__ == "__main__":
    pl.seed_everything(1234)
    model_name = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    hg_bert = BertModel.from_pretrained(model_name)
    hg_config = BertConfig.from_pretrained(model_name)

    embeddings = BertEmbeddings(
                                    tokenizer.vocab_size,
                                    hg_config.hidden_size,
                                    tokenizer.convert_tokens_to_ids("[PAD]"),
                                    hg_config.max_position_embeddings,
                                    hg_config.layer_norm_eps,
                                    hg_config.hidden_dropout_prob
                                )
    embeddings.load_state_dict(hg_bert.embeddings.state_dict())

    pooler = BertPooler(hg_config.hidden_size)
    pooler.load_state_dict(hg_bert.pooler.state_dict())

    encoder = MultiLayerEncoder(
                                    hg_config.hidden_size,
                                    hg_config.hidden_size,
                                    hg_config.num_attention_heads,
                                    hg_config.num_hidden_layers,
                                )
    
    for idx, layer in enumerate(hg_bert.encoder.layer):
        cp_weight(layer.attention.self.query, encoder.encoder_layers[idx].attention.wq)
        cp_weight(layer.attention.self.key, encoder.encoder_layers[idx].attention.wk)
        cp_weight(layer.attention.self.value, encoder.encoder_layers[idx].attention.wv)
        cp_weight(layer.attention.output.dense, encoder.encoder_layers[idx].attention.wo)

        cp_weight(layer.intermediate.dense, encoder.encoder_layers[idx].fc1)
        cp_weight(layer.output.dense, encoder.encoder_layers[idx].fc2)
        
        cp_weight(layer.attention.output.LayerNorm, encoder.encoder_layers[idx].norm1, True)
        cp_weight(layer.output.LayerNorm, encoder.encoder_layers[idx].norm2, True)
        # TODO : normalization 고치니 해결되었는데, functional에도 weight가 존재하는 건가????
    
    
    input_texts =   [
                        "this is a test text", 
                        "is it working?"
                    ]
                
    tokenized_ouptut = tokenizer(input_texts, max_length=hg_config.max_position_embeddings, padding="max_length")

    input_ids        = torch.tensor(tokenized_ouptut.input_ids)
    o_attention_mask = torch.tensor(tokenized_ouptut.attention_mask)
    token_type_ids   = torch.tensor(tokenized_ouptut.token_type_ids)

    with torch.no_grad():
        ## disable dropout -- huggingface
        hg_bert.eval() 

        ## disable dropout -- my code
        embeddings.eval() 
        pooler.eval() 
        encoder.eval() 

        ## now we need to feedforward both on huggingface BERT and our BERT
        attention_mask = o_attention_mask[:, None, None, :] # [B, 1, 1, seq_len] 

        seq_embs   = embeddings(input_ids) 
        output     = encoder(seq_embs, o_attention_mask)
        pooled_out = pooler(output[0]) 

        hg_output = hg_bert( 
                            input_ids=input_ids,
                            attention_mask=o_attention_mask,
                            token_type_ids=token_type_ids
                          )

        assert torch.all( torch.eq(hg_output.pooler_output, pooled_out) ), "Not same result!"

        print("\n\nSAME RESULT! -- Huggingface and My Code")
        
    
    print()
