"""Implementation of ISeeCube Transformer architecture used in.

https://github.com/ChenLi2049/ISeeCube/
"""
import torch
import torch.nn as nn

from graphnet.models.components.embedding import FourierEncoder
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import array_to_sequence

from torchscale.architecture.config import EncoderConfig
from torchscale.architecture.encoder import Encoder

from torch_geometric.data import Data
from torch import Tensor


class ISeeCube(GNN):
    """ISeeCube model."""

    def __init__(
        self,
        hidden_dim: int = 384,
        seq_length: int = 196,
        num_layers: int = 16,
        num_heads: int = 12,
        mlp_dim: int = 1536,
        rel_pos_buckets: int = 32,
        max_rel_pos: int = 256,
        num_register_tokens: int = 3,
        scaled_emb: bool = False,
    ):
        """Construct `ISeeCube`.

        Args:
            hidden_dim: The latent feature dimension.
            seq_length: The number of pulses in a neutrino event.
            num_layers: The depth of the transformer.
            num_heads: The number of the attention heads.
            mlp_dim: The mlp dimension of FourierEncoder and Transformer.
            rel_pos_buckets: Relative position buckets for relative position bias.
            max_rel_pos: Maximum relative position for relative position bias.
            num_register_tokens: The number of register tokens.
            scaled_emb: Whether to scale the sinusoidal positional embeddings.
        """
        super().__init__(seq_length, hidden_dim)
        self.fourier_ext = FourierEncoder(
            seq_length=seq_length,
            mlp_dim=mlp_dim,
            output_dim=hidden_dim,
            scaled=scaled_emb,
        )
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02),
            requires_grad=True,
        )

        self.class_token = nn.Parameter(
            torch.empty(1, 1, hidden_dim),
            requires_grad=True,
        )
        self.register_tokens = nn.Parameter(
            torch.empty(1, num_register_tokens, hidden_dim),
            requires_grad=True,
        )

        encoder_config = EncoderConfig(
            encoder_attention_heads=num_heads,
            encoder_embed_dim=hidden_dim,
            encoder_ffn_embed_dim=mlp_dim,
            encoder_layers=num_layers,
            rel_pos_buckets=rel_pos_buckets,
            max_rel_pos=max_rel_pos,
        )
        self.encoder = Encoder(encoder_config)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        x, _, _ = array_to_sequence(
            data.x, data.batch, padding_value=0
        )
        x = self.fourier_ext(x)
        batch_size = x.shape[0]

        x += self.pos_embedding

        batch_class_token = self.class_token.expand(batch_size, -1, -1)
        batch_register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        x = torch.cat([batch_class_token, batch_register_tokens, x], dim=1)

        x = self.encoder(src_tokens=None, token_embeddings=x)
        x = x["encoder_out"]

        x = self.layer_norm(x)

        return x[:, 0]
