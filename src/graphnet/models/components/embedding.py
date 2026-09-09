"""Classes for performing embedding of input data."""

import torch
import torch.nn as nn
from torch.functional import Tensor

from typing import Dict, Optional, Tuple, Union

from pytorch_lightning import LightningModule
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from torch_sparse import coalesce

from graphnet.models.utils import full_edge_index


class LinearEdgeEncoder(LightningModule):
    """Linear encoding for edge attributes."""

    def __init__(self, dim_emb: int):
        """Construct `LinearEdgeEncoder`.

        Args:
            dim_emb: Embedding dimension.
        """
        super().__init__()

        self.in_dim = 1  # TODO: generalize to more edge features -PW
        self.encoder = torch.nn.Linear(self.in_dim, dim_emb)

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        data.edge_attr = self.encoder(data.edge_attr.view(-1, self.in_dim))
        return data


class LinearNodeEncoder(LightningModule):
    """Linear encoding for nodes."""

    def __init__(self, dim_in: int, dim_emb: int):
        """Construct `LinearNodeEncoder`.

        Args:
            dim_in: Input dimension.
            dim_emb: Embedding dimension.
        """
        super().__init__()

        self.encoder = torch.nn.Linear(dim_in, dim_emb)

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        data.x = self.encoder(data.x)
        return data


class SinusoidalPosEmb(LightningModule):
    """Sinusoidal positional embeddings module.

    This module is from the kaggle competition 2nd place solution (see
    arXiv:2310.15674): It performs what is called Fourier encoding or
    it's used in the Attention is all you need arXiv:1706.03762. It can
    be seen as a soft digitization of the input data
    """

    def __init__(
        self,
        dim: int = 16,
        n_freq: float = 10000.0,
        scaled: bool = False,
    ):
        """Construct `SinusoidalPosEmb`.

        Args:
            dim: Embedding dimension. The ladder holds `dim / 2`
                frequencies, each contributing a sine and a cosine.
            n_freq: Span of the frequency ladder. The frequencies run from 1
                down to `n_freq ** -((dim/2 - 1)/(dim/2))`, so this sets how
                far the embedding reaches beyond its finest scale, not how
                many frequencies there are.
            scaled: Whether or not to scale the output.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim has to be even. Got: {dim}")
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5) if scaled else 1.0
        self.dim = dim
        self.n_freq = torch.Tensor([n_freq])
        # The ladder is fixed at construction, so build it once instead of per
        # forward pass. Non-persistent: it is derived from `dim` and `n_freq`,
        # and keeping it out of the state dict leaves checkpoints unchanged.
        half_dim = dim / 2
        self.register_buffer(
            "freqs",
            torch.exp(
                torch.arange(half_dim) * (-torch.log(self.n_freq) / half_dim)
            ),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        emb = x.unsqueeze(-1) * self.freqs
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb * self.scale


class FourierEncoderEPJC(LightningModule):
    """Fourier encoder of the IceMix Kaggle solution.

    The graphnet implementation of the encoder as presented in the EPJ-C
    publication (arXiv:2310.15674). It incorporates sinusoidal positional
    embeddings and auxiliary embeddings to process input sequences and
    produce meaningful representations.

    It carries assumptions from that competition that make it hard to use
    elsewhere: the input must be in the order (x, y, z, time, charge,
    auxiliary) with the first four mandatory, and the multipliers applied
    before the sinusoidal ladder are fixed to the Kaggle dataset's
    normalisation. See `FourierEncoder` for a version without them.
    """

    def __init__(
        self,
        seq_length: int = 128,
        mlp_dim: Optional[int] = None,
        output_dim: int = 384,
        scaled: bool = False,
        n_features: int = 6,
    ):
        """Construct `FourierEncoder`.

        Args:
            seq_length: Dimensionality of the base sinusoidal positional
                embeddings.
            mlp_dim (Optional): Size of hidden, latent space of MLP. If not
                given, `mlp_dim` is set automatically as multiples of
                `seq_length` (in consistent with the 2nd place solution),
                depending on `n_features`.
            output_dim: Dimension of the output (I.e. number of columns).
            scaled: Whether or not to scale the embeddings.
            n_features: The number of features in the input data.
        """
        super().__init__()

        self.sin_emb = SinusoidalPosEmb(dim=seq_length, scaled=scaled)
        self.sin_emb2 = SinusoidalPosEmb(dim=seq_length // 2, scaled=scaled)

        if n_features < 4:
            raise ValueError(
                f"At least x, y, z and time of the DOM are required. Got only "
                f"{n_features} features."
            )
        elif n_features >= 6:
            self.aux_emb = nn.Embedding(2, seq_length // 2)
            hidden_dim = 6 * seq_length
        else:
            hidden_dim = int((n_features + 0.5) * seq_length)

        if mlp_dim is None:
            mlp_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

        self.n_features = n_features

    def forward(
        self,
        x: Tensor,
        seq_length: Tensor,
    ) -> Tensor:
        """Forward pass."""
        length = torch.log10(seq_length.to(dtype=x.dtype))
        embeddings = [self.sin_emb(4096 * x[:, :, :3]).flatten(-2)]  # Position

        if self.n_features >= 5:
            embeddings.append(self.sin_emb(1024 * x[:, :, 4]))  # Charge

        embeddings.append(self.sin_emb(4096 * x[:, :, 3]))  # Time

        if self.n_features >= 6:
            embeddings.append(self.aux_emb(x[:, :, 5].long()))  # Auxiliary

        embeddings.append(
            self.sin_emb2(length).unsqueeze(1).expand(-1, max(seq_length), -1)
        )  # Length

        x = torch.cat(embeddings, -1)
        x = self.mlp(x)

        return x


class FourierEncoder(LightningModule):
    """Apply sinusoidal positional encodings to sequence representations.

    Embeds `[B, K, D]` sequences -- batch size, padded sequence length, and
    features per step -- into sinusoidal positions, optionally alongside the
    unpadded length of each sequence.

    A superficial refactor of `FourierEncoderEPJC` that drops the
    assumptions tying it to the Kaggle dataset. `schema` states which
    columns to embed and over which band of scales, so nothing is assumed
    about the order or meaning of the input columns, and the boolean
    embedding and MLP projection are left to the model, since both are
    problem-specific.
    """

    def __init__(
        self,
        schema: Dict[int, Union[float, Tuple[float, float]]],
        seq_length: int = 128,
        scaled: bool = False,
        add_sequence_length: bool = True,
        n_freq: float = 10000.0,
    ):
        """Construct `FourierEncoder`.

        Args:
            schema: Maps an input column index to how it is embedded, either
                as a multiplier or as a `(multiplier, n_freq)` pair, e.g.
                `{1: 1024, 3: (4096, 500)}`. Columns absent from the schema
                are not embedded.

                The two together set the band of separations a column can
                resolve: wavelengths from `2 * pi * unit / multiplier` up to
                that times `n_freq ** ((dim/2 - 1)/(dim/2))`, where `unit` is
                the physical size of one unit of the (normalised) column. So
                the multiplier places the band and `n_freq` sets its width,
                and both are properties of the data rather than of the model
                -- values carried over from a different normalisation resolve
                a different physical range.
            seq_length: Desired dimensionality of the base sinusoidal
                positional embeddings.
            scaled: Whether or not to scale the embeddings.
            add_sequence_length: If True, the unpadded length of each
                sequence is embedded as well, at half width.
            n_freq: Ladder span for columns whose schema entry gives only a
                multiplier, and for the sequence-length embedding.
        """
        super().__init__()
        if not schema:
            raise ValueError("`schema` must map at least one column.")

        self.schema = {
            col: (
                (float(v[0]), float(v[1]))
                if isinstance(v, (tuple, list))
                else (float(v), float(n_freq))
            )
            for col, v in schema.items()
        }
        self._add_sequence_length = add_sequence_length
        # One embedder per distinct span, not per column: columns sharing a
        # span share a ladder, so the common case of a single span builds a
        # single module and keeps `scaled`'s one learnable scale.
        spans = sorted({span for _, span in self.schema.values()})
        self._span_index = {span: i for i, span in enumerate(spans)}
        self.sin_feature = nn.ModuleList(
            [
                SinusoidalPosEmb(dim=seq_length, n_freq=span, scaled=scaled)
                for span in spans
            ]
        )
        if add_sequence_length:
            self.sin_length = SinusoidalPosEmb(
                dim=seq_length // 2, n_freq=n_freq, scaled=scaled
            )
        # Width of the concatenation, so the model can size what follows.
        self.output_dim = len(self.schema) * seq_length + (
            seq_length // 2 if add_sequence_length else 0
        )

    def forward(
        self,
        x: Tensor,
        seq_length: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply positional encoding of `x`.

        Args:
            x: `[B, K, D]`-dimensional sequence representation of an event.
            seq_length: `[B]`-dimensional unpadded length of each sequence in
                `x`, one entry per sequence rather than per step. Required
                when `add_sequence_length` is True, in which case each step
                of a sequence receives a copy of its sequence's embedded
                length.

        Returns:
            Embedded `[B, K, J]`-dimensional sequence.
        """
        embeddings = [
            self.sin_feature[self._span_index[span]](scale * x[:, :, col])
            for col, (scale, span) in self.schema.items()
        ]

        if self._add_sequence_length:
            if seq_length is None:
                raise ValueError(
                    "Must pass `seq_length` when `add_sequence_length` is "
                    "True."
                )
            length = torch.log10(seq_length.to(dtype=x.dtype))
            embeddings.append(
                self.sin_length(length).unsqueeze(1).expand(-1, x.shape[1], -1)
            )

        return torch.cat(embeddings, -1)


class SpacetimeEncoder(LightningModule):
    """Spacetime encoder module."""

    def __init__(
        self,
        seq_length: int = 32,
    ):
        """Construct `SpacetimeEncoder`.

        This module calculates space-time interval between each pair of events
        and generates sinusoidal positional embeddings to be added to input
        sequences.

        Args:
            seq_length: Dimensionality of the sinusoidal positional embeddings.
        """
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=seq_length)
        self.projection = nn.Linear(seq_length, seq_length)

    def forward(
        self,
        x: Tensor,
        # Lmax: Optional[int] = None,
    ) -> Tensor:
        """Forward pass."""
        pos = x[:, :, :3]
        time = x[:, :, 3]
        spacetime_interval = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(
            -1
        ) - ((time[:, :, None] - time[:, None, :]) * (3e4 / 500 * 3e-1)).pow(2)
        four_distance = torch.sign(spacetime_interval) * torch.sqrt(
            torch.abs(spacetime_interval)
        )
        sin_emb = self.sin_emb(1024 * four_distance.clip(-4, 4))
        rel_attn = self.projection(sin_emb)
        return rel_attn


class RRWPLinearNodeEncoder(LightningModule):
    """Relative random walk probability node encoder.

    Original code:
    https://github.com/LiamMa/GRIT/blob/main/grit/encoder/rrwp_encoder.py
    """

    def __init__(
        self,
        emb_dim: int,
        out_dim: int,
        use_bias: bool = False,
        apply_norm: bool = True,
        norm_layer: nn.Module = nn.BatchNorm1d,
        pe_name: str = "rrwp",
    ):
        """Construct `RRWPLinearNodeEncoder`.

        Args:
            emb_dim: Embedding dimension.
            out_dim: Output dimension.
            use_bias: Apply bias to linear layer.
            apply_norm: Apply normalization layer.
            norm_layer: Normalization layer.
            pe_name: Positional encoding name.
        """
        super().__init__()
        self.name = pe_name
        self.apply_norm = apply_norm

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        if self.apply_norm:
            self.norm = norm_layer(out_dim)

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        rrwp = data[f"{self.name}"]
        rrwp = self.fc(rrwp)

        if self.norm:
            rrwp = self.norm(rrwp)

        if "x" in data:
            data.x = data.x + rrwp
        else:
            data.x = rrwp

        return data


class RRWPLinearEdgeEncoder(LightningModule):
    """Relative random walk probability edge encoder.

    Original code:
    https://github.com/LiamMa/GRIT/blob/main/grit/encoder/rrwp_encoder.py
    """

    def __init__(
        self,
        emb_dim: int,
        out_dim: int,
        use_bias: bool = False,
        apply_norm: bool = True,
        norm_layer: nn.Module = nn.BatchNorm1d,
        pad_to_full_graph: bool = True,
        fill_value: float = 0.0,
        add_node_attr_as_self_loop: bool = False,
        overwrite_old_attr: bool = False,
    ):
        """Construct `RRWPLinearEdgeEncoder`.

        Args:
            emb_dim: Embedding dimension.
            out_dim: Output dimension.
            use_bias: Apply bias to linear layer.
            apply_norm: Apply normalization layer.
            norm_layer: Normalization layer.
            pad_to_full_graph: Pad edges to fully-connected graph.
            fill_value: Fill value for padding.
            add_node_attr_as_self_loop: Add self loop edges with node attr.
            overwrite_old_attr: Overwrite old edge attr.
        """
        super().__init__()

        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr = overwrite_old_attr
        self.apply_norm = apply_norm

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = 0.0

        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)

        if self.apply_norm:
            self.norm = norm_layer(out_dim)

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        rrwp_idx = data.rrwp_index
        rrwp_val = data.rrwp_val
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        rrwp_val = self.fc(rrwp_val)

        if edge_attr is None:
            edge_attr = edge_index.new_zeros(
                edge_index.size(1), rrwp_val.size(1)
            )
            # zero padding for non-existing edges

        if self.overwrite_old_attr:
            out_idx, out_val = rrwp_idx, rrwp_val
        else:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=data.num_nodes, fill_value=0.0
            )

            out_idx, out_val = coalesce(
                torch.cat([edge_index, rrwp_idx], dim=1),
                torch.cat([edge_attr, rrwp_val], dim=0),
                data.num_nodes,
                data.num_nodes,
                op="add",
            )

        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=data.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = coalesce(
                out_idx, out_val, data.num_nodes, data.num_nodes, op="add"
            )

        if self.apply_norm:
            out_val = self.norm(out_val)

        data.edge_index, data.edge_attr = out_idx, out_val
        return data


class RWSELinearNodeEncoder(LightningModule):
    """Random walk structural node encoding."""

    def __init__(
        self,
        emb_dim: int,
        out_dim: int,
        use_bias: bool = False,
    ):
        """Construct `RWSELinearEdgeEncoder`.

        Args:
            emb_dim: Embedding dimension.
            out_dim: Output dimension.
            use_bias: Apply bias to linear layer.
        """
        super().__init__()

        self.emb_dim = emb_dim
        self.out_dim = out_dim

        self.encoder = nn.Linear(emb_dim, out_dim, bias=use_bias)

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        rwse = data.rwse
        x = data.x

        rwse = self.encoder(rwse)

        data.x = torch.cat((x, rwse), dim=1)

        return data
