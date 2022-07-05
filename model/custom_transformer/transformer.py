from collections import defaultdict
# Import PyTorch
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

# Import custom modules
from .embedding import TransformerEmbedding
from ..latent_module.latent import Latent_module 

class Transformer(nn.Module):
    def __init__(self, pad_idx=0, d_model=512, d_embedding=256, n_head=8, dim_feedforward=2048, 
            d_latent=256, num_encoder_layer=10, src_max_len=100,
            dropout=0.1, embedding_dropout=0.1):

        super(Transformer, self).__init__()

        # Hyper-paramter setting
        self.pad_idx = pad_idx
        self.src_max_len = src_max_len

        # Dropout setting
        self.dropout = nn.Dropout(dropout)

        # Source embedding part
        self.src_embedding = TransformerEmbedding(27, d_model, d_embedding, 
            pad_idx=self.pad_idx, max_len=self.src_max_len, dropout=embedding_dropout)

        # Transformer Encoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

        # Target linear part
        self.src_output_linear = nn.Linear(d_model, d_embedding)
        self.src_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.src_output_linear2 = nn.Linear(d_embedding, 2)
            
    def forward(self, input_ids, attention_mask):

        # Source attention masking
        src_key_padding_mask = ~attention_mask.bool()

        # Embedding
        encoder_out = self.src_embedding(input_ids).transpose(0, 1) # (token, batch, d_model)

        # Encoder
        for encoder in self.encoders:
            encoder_out = encoder(encoder_out, src_key_padding_mask=src_key_padding_mask)

        encoder_out = self.src_output_norm(self.dropout(F.gelu(self.src_output_linear(encoder_out))))
        encoder_out = self.src_output_linear2(encoder_out)
        return encoder_out.mean(dim=0)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src