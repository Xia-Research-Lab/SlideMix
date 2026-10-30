import torch
from torch import nn
from torch.nn import functional as F

from models.layers import GatedAttention


class ABMIL(nn.Module):
    def __init__(self, args, embed_dim=500, latent_dim=128, dropout=0.25):
        super().__init__()
        self.args = args
        self.input_dim = embed_dim
        self.output_dim = embed_dim
        self.MTL_token_num = len(args.tasks)

        self.attention = GatedAttention(embed_dim, latent_dim, self.MTL_token_num, dropout)

    def forward(self, x, coords=None):
        A = self.attention(x)
        A = A.transpose(1, 2)
        A = F.softmax(A, dim=2)
        M = torch.matmul(A, x)
        return M
