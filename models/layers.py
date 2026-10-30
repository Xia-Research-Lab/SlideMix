from torch import nn


class GatedAttention(nn.Module):
    def __init__(self, L, D, K, dropout=0.25):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_w = nn.Linear(D, K)

    def forward(self, H):
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_w(A_V * A_U)
        return A
