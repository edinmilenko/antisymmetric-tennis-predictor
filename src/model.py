import torch
import torch.nn as nn
from config import HIDDEN, DROPOUT


class AntisymmetricNet(nn.Module):

    def __init__(self, n_feature: int, n_feature_context: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(in_features=2 * n_feature + n_feature_context, out_features=hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden, out_features=hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden, out_features=1),
        )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        full_vector_a = torch.cat((x_a, x_b, c), dim=1)
        full_vector_b = torch.cat((x_b, x_a, c), dim=1)
        final_a = self.sequential(full_vector_a)
        final_b = self.sequential(full_vector_b)
        diff = final_a - final_b
        return diff.squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    n, n_player, n_ctx = 5, 28, 12
    model = AntisymmetricNet(n_player, n_ctx)
    model.eval()

    x_a, x_b, c = torch.randn(n, n_player), torch.randn(n, n_player), torch.randn(n, n_ctx)
    with torch.no_grad():
        logit_ab = model(x_a, x_b, c)
        logit_ba = model(x_b, x_a, c)

    p_ab, p_ba = torch.sigmoid(logit_ab), torch.sigmoid(logit_ba)
    print("shape:", tuple(logit_ab.shape))
    print("p(A,B) + p(B,A):", (p_ab + p_ba).tolist())

    assert logit_ab.shape == (n,), "Wrong output shape"
    assert torch.allclose(p_ab + p_ba, torch.ones(n), atol=1e-6), "Antisymmetry violated"
    print("OK")