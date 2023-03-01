import sys
import math
import typing as t

import torch
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer


def gelu_new(x):
    """
    this code is from https://github.com/huggingface/transformers/blob/3fefa292c1c419f0c4c3e2697cdd94cafaeb4b66/src/transformers/activations.py#L37
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Conv1D(torch.nn.Module):
    # this code is from https://amaarora.github.io/2020/02/18/annotatedGPT2.html -- and huggingface code
    # basically, it is quite ok to use Linear instead of Conv1D
    # but, to keep the consistency of original openai-gpt2 implmentation
    # we used conv1d as well.
    def __init__(self, nx, nf):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        torch.nn.init.normal_(w, std=0.02)
        self.weight = torch.nn.Parameter(w)
        self.bias   = torch.nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        # [B, S, dim_x] -> [B, S, dim_nf]
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(torch.nn.Module):
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
        
        self.wk = Conv1D(self.d_embed, self.d_embed * 3)
        self.wo = Conv1D(self.d_embed, self.d_embed)
    
    
    def __split_heads(
                        self,
                        x,  # (batch_size, max_seq_len, d_model)
                    ):
        batch_size, max_seq_len, d_model = x.shape
        x = x.reshape(batch_size, max_seq_len, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        return x  # (batch_size, num_heads, max_seq_len, d_k)

    
    def forward(
                    self,
                    x,
                    attention_mask,
                ):
        x = self.wk(x)
        query, key, value = x.split(self.d_embed, dim=2)  # TODO : 이 방식과 기존 방식 차이점?
        Q = self.__split_heads(query)  # (batch_size, num_heads, max_query_len, d_k)
        K = self.__split_heads(key)  # (batch_size, num_heads, max_key_len, d_k)
        V = self.__split_heads(value)  # (batch_size, num_heads, max_key_len, d_k)

        reactivity = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, max_query_len, max_key_len)
        reactivity /= math.sqrt(V.shape[-1])
        attention_mask = 1 - attention_mask
        attention_mask = attention_mask.masked_fill(attention_mask == 1, -sys.maxsize-1)
        reactivity += attention_mask
        
        attention_scores = torch.softmax(reactivity, dim=-1)  # (batch_size, num_heads, max_query_len, max_key_len)
        
        blended_vector = torch.matmul(attention_scores, V)  # (batch_size, num_heads, max_query_len, d_k)
        batch_size, num_heads, max_query_len, d_k = blended_vector.shape
        blended_vector = blended_vector.transpose(1, 2).reshape(batch_size, max_query_len, self.d_model)
        blended_vector = self.wo(blended_vector)  # (batch_size, max_query_len, d_embed)
        
        return blended_vector, attention_scores


class GPT2DecoderLayer(torch.nn.Module):
    def __init__(
                    self,
                    d_embed: int,
                    d_model: int,
                    num_heads: int,
                    intermediate_size: int,
                ):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size

        self.norm1 = torch.nn.LayerNorm(self.d_embed)
        self.norm2 = torch.nn.LayerNorm(self.d_embed)
        self.attention = Attention(
                                    d_embed=self.d_embed,
                                    d_model=self.d_model,
                                    num_heads=self.num_heads,
                                )
        self.fc1 = Conv1D(self.d_embed, self.intermediate_size)
        self.fc2 = Conv1D(self.intermediate_size, self.d_embed)
        
    
    
    def forward(
                    self,
                    x,  # (batch_size, max_seq_len, d_embed)
                ):
        look_ahead_mask = 1 - torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.long), diagonal=1)
        look_ahead_mask = look_ahead_mask.to(x.device)
        
        
        residual = x
        x = self.norm1(x)
        x, attention_scores = self.attention(
                                                x=x,
                                                attention_mask=look_ahead_mask,
                                            )
        x += residual
        
        residual = x
        x = self.norm2(x)
        x = gelu_new(self.fc1(x))
        x = self.fc2(x)
        x += residual
        
        return x, attention_scores


class GPT2Decoder(torch.nn.Module):
    def __init__(
                    self,
                    num_layers: int,
                    d_embed: int,
                    d_model: int,
                    num_heads: int,
                    intermediate_size: int,
                ):
        super().__init__()
        self.num_layers = num_layers
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        
        self.layers = torch.nn.ModuleList([
                                    GPT2DecoderLayer(
                                        d_embed=self.d_embed,
                                        d_model=self.d_model,
                                        num_heads=self.num_heads,
                                        intermediate_size=self.intermediate_size,
                                    )
                                    for _ in range(self.num_layers)
                                ])
    
    
    def forward(
                    self,
                    x,
                ):
        layers_attention_scores = []
        for layer in self.layers:
            x, attention_scores = layer(x)
            layers_attention_scores.append(attention_scores)
        return x, layers_attention_scores


class GPT2(torch.nn.Module):
    def __init__(
                    self,
                    vocab_size: int,
                    max_seq_len: int,
                    num_layers: int,
                    d_embed: int,
                    d_model: int,
                    num_heads: int,
                    intermediate_size: int,
                ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.d_embed = d_embed
        self.d_model = d_model
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        
        self.wte = torch.nn.Embedding(self.vocab_size, self.d_embed)
        self.wpe = torch.nn.Embedding(self.max_seq_len, self.d_embed)
        self.register_buffer("position_ids", torch.arange(self.max_seq_len).expand((1, -1)))
        
        self.blocks = GPT2Decoder(
                                    num_layers=self.num_layers,
                                    d_embed=self.d_embed,
                                    d_model=self.d_model,
                                    num_heads=self.num_heads,
                                    intermediate_size=self.intermediate_size,
                                )
        self.norm = torch.nn.LayerNorm(self.d_embed)
        self.head = torch.nn.Linear(self.d_embed, self.vocab_size, bias=False)
        
    
    def forward(
                    self,
                    input_ids,  # (batch_size, seq_len)
                ):
        token_embeds = self.wte(input_ids)  # (batch_size, seq_len, d_embed)
        position_ids = self.position_ids[:, :input_ids.shape[1]]
        position_embeds = self.wpe(position_ids)
        x = token_embeds + position_embeds
        
        x, layer_attention_scores = self.blocks(x)
        x = self.norm(x)
        
        logits = self.head(x)
        return logits


def cp_weight(src, tar, copy_bias=True, include_eps=False):
    assert tar.weight.size() == src.weight.size(), "Not compatible parameter size"
    tar.load_state_dict( src.state_dict() )
    
    if include_eps:
        # in case of LayerNorm. 
        with torch.no_grad():
            tar.eps = src.eps


def copy_from_huggingface():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    hg_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    model = GPT2(
                    vocab_size= hg_model.config.vocab_size,
                    max_seq_len= hg_model.config.n_ctx,
                    num_layers= hg_model.config.n_layer,
                    d_embed= hg_model.config.n_embd,
                    d_model= hg_model.config.n_embd,
                    intermediate_size= hg_model.config.n_embd * 4,
                    num_heads= hg_model.config.n_head,
                )
    
    model.wte.load_state_dict(hg_model.transformer.wte.state_dict())
    model.wpe.load_state_dict(hg_model.transformer.wpe.state_dict())

    model.head.load_state_dict(hg_model.lm_head.state_dict())
    
    cp_weight(hg_model.transformer.ln_f, model.norm, include_eps=True)
    
    for layer_num, block in enumerate(hg_model.transformer.h):
        cp_weight(block.attn.c_attn, model.blocks.layers[layer_num].attention.wk)
        cp_weight(block.attn.c_proj, model.blocks.layers[layer_num].attention.wo)

        cp_weight(block.mlp.c_fc, model.blocks.layers[layer_num].fc1)
        cp_weight(block.mlp.c_proj, model.blocks.layers[layer_num].fc2)
        
        cp_weight(block.ln_1, model.blocks.layers[layer_num].norm1, include_eps=True)
        cp_weight(block.ln_2, model.blocks.layers[layer_num].norm2, include_eps=True)
    
    return model, tokenizer, hg_model


def main():
    model, tokenizer, hg_model = copy_from_huggingface()
    
    inputs = tokenizer('Hello, my dog is cute', return_tensors='pt')

    model.eval()
    hg_model.eval()

    with torch.no_grad():
        hg_outputs = hg_model(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask
                        )
        my_output = model(
                            input_ids=inputs.input_ids,
                        )
        assert torch.all( torch.eq(hg_outputs.logits, my_output) ), "Not same result!"
        print("SAME RESULT! -- Huggingface-GPT2 and My Code")


if __name__ == '__main__':
    main()
