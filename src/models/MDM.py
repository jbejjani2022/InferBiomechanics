import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.AddBiomechanicsDataset import InputDataKeys, OutputDataKeys
from typing import Dict

class MDM(nn.Module):
    def __init__(self, dofs: int, window_size=50, history_len=5, stride=1, latent_dim=256, ff_size=1024,
                 num_layers=8, num_heads=4, dropout=0.1, activation='gelu', dtype=torch.float32, device='cpu'):
        super().__init__()

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout
        self.dtype = dtype
        self.device = device

        # Compute the size of the input vector to the model, which is the concatenation
        # of input keys
        self.timestep_vector_dim = 2 + (dofs * 3) 

        # Output vector is 2 contact labels and three metrics per dof
        self.output_vector_dim = 2 + (dofs * 3)
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.num_output_frames = (window_size // stride)


        self.input_process = InputProcess(self.timestep_vector_dim, self.latent_dim)
        self.positional_encoding  = PositionalEncoding(self.latent_dim)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                 num_layers=self.num_layers)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.positional_encoding)
        self.temporal_embedding = TemporalEmbedding(
            window_size, latent_dim, dtype=dtype)
        
    def parameters(self):
        return [p for name, p in self.named_parameters()]

    def forward(self, x, timesteps):
        x = self.input_process(x)
        emb = self.embed_timestep(timesteps)
        xseq = torch.cat((emb, x), axis=1)
        xseq = self.positional_encoding(xseq).to(self.dtype)
        output = self.seqTransEncoder(xseq).to(self.device)
        output_decoder = nn.Linear(self.latent_dim, self.output_vector_dim, dtype=self.dtype, device=self.device)
        output = output_decoder(output)[:, 1:, :] #[bs, wlen+1, feats] -> [bs, wlen, feats]


        # Split output into different components
        output_dict: Dict[str, torch.Tensor] = {}
        
        output_dict[OutputDataKeys.CONTACT] = output[:, :, :2]
        output_dict[OutputDataKeys.ACC] = output[:, :, 2:25]
        output_dict[OutputDataKeys.VEL] = output[:, :, 25:48]
        output_dict[OutputDataKeys.POS] = output[:, :, 48:71]

        return output_dict


class PositionalEncoding(nn.Module):

    def __init__(self, latent_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, latent_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2).float() * (-np.log(10000.0) / latent_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_encoder = pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(nn.Linear(self.latent_dim, time_embed_dim),
                                        nn.SiLU(),
                                        nn.Linear(time_embed_dim, time_embed_dim))
        
    def forward(self, timesteps):
        return self.time_embed(self.pos_encoder.pe[timesteps])
    
class TemporalEmbedding(nn.Module):
    def __init__(self, window_size, embedding_dim, dtype=torch.float32):
        super().__init__()
        self.embedding = nn.Embedding(window_size, embedding_dim, dtype=dtype)

    def forward(self, x):
        embedded = self.embedding(x)
        return embedded
    
class InputProcess(nn.Module):
    # Linear layer to project feature dimension to latent space
    def __init__(self, features, latent_dim, dtype=torch.float32):
        super().__init__()
        self.linear = nn.Linear(features, latent_dim, dtype=dtype)

    def forward(self, x):

        batch_size, seq_len, features = x.size()

        # Reshape x to [-1, features] to apply the linear layer to each timestep
        x_reshaped = x.view(-1, features)
        
        # Apply the linear transformation
        transformed = self.linear(x_reshaped)
        
        # Reshape back to [batch_size, seq_len, latent_dim]
        output = transformed.view(batch_size, seq_len, -1)
        
        return output