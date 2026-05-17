import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncodings(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # div term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
        # sin and cos functions
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        
        
        pe = pe.unsqueeze(0) 

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) 
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps:float = 10**-6):
            super().__init__()
            self.eps = eps
            self.bias = nn.Parameter(torch.zeros(features)) # paramter to define a learable paramter
            self.alpha = nn.Parameter(torch.ones(features))

    def forward(self, x):
            mean= x.mean(dim = -1, keepdim  = True)
            std= x.std(dim = -1, keepdim  = True, unbiased = False)
            return self.alpha * (x-mean)/(std+self.eps) + self.bias


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
          super().__init__()
          self.linear_1 = nn.Linear(d_model, d_ff) #w1 and b1
          self.dropout = nn.Dropout(dropout)
          self.linear_2 = nn.Linear(d_ff, d_model) #w2 and b2

    def forward(self, x):
         return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
     
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.heads = heads

        assert d_model % heads == 0, "Dimension of the model is not divisible by the number of heads"

        self.d_k = d_model // heads  # dimension of vector seen by every head
        self.w_q = nn.Linear(d_model, d_model, bias = False) # query
        self.w_k = nn.Linear(d_model, d_model, bias = False) # key
        self.w_v = nn.Linear(d_model, d_model, bias = False) # value
        self.w_o = nn.Linear(d_model, d_model, bias = False) # output
        self.dropout = nn.Dropout(dropout)
        
    '''
    b: Batch Size
    h: Number of Heads
    i,j : Sequence Lenght for query and key/value respectively. They are equal
    d: d_k 
    query : (B,H,S,D)
    key^T : (B,H,D,S)
    '''
    @staticmethod
    def attention(query, key, value, mask, dropout=None):
        d_k = query.shape[-1]
        attention_scores = (torch.einsum("bhid,bhjd->bhij", query, key)/ math.sqrt(d_k))

        if mask is not None:
            attention_scores.masked_fill_(mask == 0,-1e9)
            
        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        output = torch.einsum( "bhij,bhjd->bhid", attention_scores, value)

        return output, attention_scores
        

    def forward(self, q, k, v, mask):
        '''
        B: Batch size
        S: Sequence Length
        D: D_Model
        '''
        key = self.w_k(k) # BSD 
        value = self.w_v(v) # BSD 
        query = self.w_q(q) # BSD

        '''
        h: heads
        d_k: vectors
        '''
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1,2) # BSD -> BShd_k -> BhSd_k
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1,2) # BSD -> BShd_k -> BhSd_k
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1,2) # BSD -> BShd_k -> BhSd_k

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.heads * self.d_k) # BhSd_k -> BShd_k -> BSD

        return self.w_o(x)

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardNetwork, dropout: float):
         super().__init__()
         self.self_attention_block = self_attention_block
         self.feed_forward_block = feed_forward_block  
         self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
         x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
         x = self.residual_connections[1](x, self.feed_forward_network)
         return x
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, features:int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block[x,x,x,src_mask])
        x = self.residual_connections[0](x, lambda x: self.cross_attention_block[x,encoder_output, encoder_output,tgt_mask])
        x = self.residual_connections[1](x, self.feed_forward_network)

class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class LinearLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        '''
        B: Batch size
        S: Sequence Length
        D: D_Model
        V: vocab length
        '''
        x = self.linear(x) #BSD -> BSV




         
          
          