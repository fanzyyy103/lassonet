import torch
from torch import nn
from torch.nn import functional as F

from .prox import inplace_prox


class LassoNet(nn.Module):
    def __init__(self, *dims, dropout=None, skip_only=False):
        if len(dims) < 3:
            raise ValueError("LassoNet needs at least input, hidden, and output dims.")

        super().__init__()
        self.skip_only = bool(skip_only)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.skip = nn.Linear(dims[0], dims[-1], bias=False)
        if self.skip_only:
            self._zero_hidden_parameters()

    def _zero_hidden_parameters(self):
        for layer in self.layers:
            layer.weight.data.zero_()
            if layer.bias is not None:
                layer.bias.data.zero_()

    def forward(self, inputs):
        result = self.skip(inputs)
        if self.skip_only:
            return result

        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden)
            if layer is not self.layers[-1]:
                if self.dropout is not None:
                    hidden = self.dropout(hidden)
                hidden = F.relu(hidden)
        return result + hidden

    def prox(self, *, lambda_, lambda_bar=0.0, M=1.0):
        with torch.no_grad():
            inplace_prox(
                skip_layer=self.skip,
                first_hidden_layer=self.layers[0],
                lambda_=lambda_,
                lambda_bar=lambda_bar,
                M=M,
            )

    def skip_regularization(self, penalty_factor=None):
        column_norms = torch.norm(self.skip.weight, p=2, dim=0)
        if penalty_factor is None:
            return column_norms.sum()
        return (column_norms * penalty_factor).sum()

    def input_mask(self):
        with torch.no_grad():
            return torch.norm(self.skip.weight, p=2, dim=0) != 0

    def selected_count(self):
        return int(self.input_mask().sum().item())

    def cpu_state_dict(self):
        return {key: value.detach().clone().cpu() for key, value in self.state_dict().items()}
