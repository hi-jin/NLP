import sys
import math

import torch
from transformers import BertTokenizer, BertModel, BertConfig


class QKV_MultiHead_Attention(torch.nn.Module):
    def __init__(
                    self,
                    dim_embed: int,
                    dim_model: int,
                    num_heads: int,
                ):
        super().__init__()
        self.dim_embed = dim_embed
        self.dim_model = dim_model
        self.num_heads = num_heads
        
        assert self.dim_model % self.num_heads == 0
        self.dim_k = self.dim_model // self.num_heads

        self.wq = torch.nn.Linear(self.dim_embed, self.dim_model)
        self.wk = torch.nn.Linear(self.dim_embed, self.dim_model)
        self.wv = torch.nn.Linear(self.dim_embed, self.dim_model)
        self.wo = torch.nn.Linear(self.dim_model, self.dim_embed)
    
    
    def __split_heads(
                        self,
                        x: torch.Tensor,  # (batch_size, dim_x, dim_model)
                    ):
        batch_size, dim_x, dim_model = x.shape
        
        x = x.view(batch_size, dim_x, self.num_heads, self.dim_k).contiguous()
        x = torch.transpose(x, 1, 2)
        return x  # (batch_size, num_heads, dim_x, dim_k)


    def forward(
                    self,
                    query,  # (batch_size, dim_query, dim_embed)
                    key,  # (batch_size, dim_key, dim_embed)
                    value,  # (batch_size, dim_key, dim_embed)
                    attention_mask: torch.Tensor = None,  # (batch_size, dim_key)
                ):
        assert key.shape[1] == value.shape[1]
        
        Q = self.__split_heads(self.wq(query))  # (batch_size, num_heads, dim_query, dim_k)
        K = self.__split_heads(self.wk(key))  # (batch_size, num_heads, dim_key, dim_k)
        V = self.__split_heads(self.wv(value))  # (batch_size, num_heads, dim_key, dim_k)
        
        reactivity = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, dim_query, dim_key)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = 1 - attention_mask
            attention_mask = attention_mask * (-sys.maxsize - 1)
            reactivity += attention_mask
        attention_score = torch.softmax(reactivity / math.sqrt(self.dim_k), dim=-1)  # probability
        
        blended_vector = torch.matmul(attention_score, V)  # (batch_size, num_heads, dim_query, dim_k)
        blended_vector = torch.transpose(blended_vector, 1, 2).contiguous()  # (batch_size, dim_query, num_heads, dim_k)
        blended_vector = blended_vector.view(blended_vector.shape[0], -1, self.dim_model)  # (batch_size, dim_query, dim_model)
        blended_vector = self.wo(blended_vector)  # (batch_size, dim_query, dim_embed)
        return blended_vector, attention_score


class EncoderLayer(torch.nn.Module):
    def __init__(
                    self,
                    dim_embed: int,
                    dim_model: int,
                    num_heads: int,
                    dim_ff: int,
                ):
        super().__init__()
        self.dim_embed = dim_embed
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_ff = dim_ff
        
        self.attention = QKV_MultiHead_Attention(
            dim_embed=self.dim_embed,
            dim_model=self.dim_model,
            num_heads=self.num_heads,
        )
        
        self.norm1 = torch.nn.LayerNorm(self.dim_embed)
        self.norm2 = torch.nn.LayerNorm(self.dim_embed)
        
        self.fc1 = torch.nn.Linear(self.dim_embed, self.dim_ff)
        self.fc2 = torch.nn.Linear(self.dim_ff, self.dim_embed)
        
    
    def forward(
                    self,
                    sequences: torch.Tensor,  # (batch_size, len_seq, dim_embed)
                    attention_mask: torch.Tensor = None,  # (batch_size, len_seq)
                ):
        x = sequences
        
        residual = x
        x, attention_score = self.attention(
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
        
        return x, attention_score


class MultiLayerEncoder(torch.nn.Module):
    def __init__(
                    self,
                    num_layers: int,
                    dim_embed: int,
                    dim_model: int,
                    num_heads: int,
                    dim_ff: int,
                ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_embed = dim_embed
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_ff = dim_ff
        
        self.layers = torch.nn.ModuleList([
            EncoderLayer(
                dim_embed=self.dim_embed,
                dim_model=self.dim_model,
                num_heads=self.num_heads,
                dim_ff=self.dim_ff,
            )
            for _ in range(self.num_layers)
        ])
    
    
    def forward(
                    self,
                    sequences,
                    attention_mask=None,
                ):
        x = sequences
        
        layers_attention_scores = []
        
        for layer in self.layers:
            x, attention_score = layer(
                sequences=x,
                attention_mask=attention_mask,
            )
            layers_attention_scores.append(attention_score)
        
        return x, layers_attention_scores


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


def copy_from_huggingface(model_name):
    def cp_weight(source, target, include_eps=False):
        assert source.weight.size() == target.weight.size()
        target.load_state_dict(source.state_dict())
        
        if include_eps:
            with torch.no_grad():
                target.eps = source.eps
    
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
        num_layers=hg_config.num_hidden_layers,
        dim_embed=hg_config.hidden_size,
        dim_model=hg_config.hidden_size,
        num_heads=hg_config.num_attention_heads,
        dim_ff=hg_config.intermediate_size,
    )

    for idx, layer in enumerate(hg_bert.encoder.layer):
        cp_weight(layer.attention.self.query, encoder.layers[idx].attention.wq)
        cp_weight(layer.attention.self.key, encoder.layers[idx].attention.wk)
        cp_weight(layer.attention.self.value, encoder.layers[idx].attention.wv)
        cp_weight(layer.attention.output.dense, encoder.layers[idx].attention.wo)

        cp_weight(layer.intermediate.dense, encoder.layers[idx].fc1)
        cp_weight(layer.output.dense, encoder.layers[idx].fc2)
        
        cp_weight(layer.attention.output.LayerNorm, encoder.layers[idx].norm1, True)
        cp_weight(layer.output.LayerNorm, encoder.layers[idx].norm2, True)

    return tokenizer, embeddings, encoder, pooler


def main():
    pass


if __name__ == "__main__":
    main()
