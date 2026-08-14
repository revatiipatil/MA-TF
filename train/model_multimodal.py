#!/usr/bin/env python3
"""
Multi-Modal Transformer model for chromatin accessibility prediction.

Architecture:
    - Sequence encoder: Transformer over one-hot DNA sequence
    - Histone encoder: 1D CNN over histone modification bins
    - TF encoder: MLP over transcription factor binding profile bins
    - Cell embedding: learned embedding for the target cell type
    - Cross-modal fusion: Transformer encoder over modality embeddings
    - Regression head: predicts continuous ATAC-seq accessibility

Important:
    ATAC-seq is the prediction target and is NOT used as an input modality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_track_index_map(
    feature_map: Optional[Dict]
) -> Dict[str, Tuple[int, int]]:
    """
    Build mapping from track filename to (start_idx, end_idx).

    Assumes each track contributes `n_bins` consecutive features.

    Args:
        feature_map: Dictionary describing the functional genomic tracks.

    Returns:
        Dictionary mapping each track name to its feature slice.
    """
    if feature_map is None:
        return {}

    n_bins = feature_map.get("n_bins", 16)

    track_map = {}

    for i, track in enumerate(feature_map.get("bigwigs", [])):
        start = i * n_bins
        end = start + n_bins
        track_map[track] = (start, end)

    return track_map


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 2000
    ):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float
            )
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added.
        """
        return x + self.pe[:, :x.size(1)]


class SequenceEncoder(nn.Module):
    """
    Transformer encoder for one-hot encoded DNA sequences.
    """

    def __init__(
        self,
        seq_len: int,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(4, embed_dim)

        self.positional_encoding = PositionalEncoding(
            embed_dim,
            max_len=seq_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence:
                One-hot encoded DNA sequence.
                Shape: (batch, seq_len, 4)

        Returns:
            Encoded sequence representation.
            Shape: (batch, embed_dim)
        """

        x = self.input_proj(sequence)

        x = self.positional_encoding(x)

        x = self.encoder(x)

        # (batch, seq_len, embed_dim)
        # -> (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)

        # Global average pooling
        x = F.adaptive_avg_pool1d(
            x,
            1
        ).squeeze(-1)

        return self.dropout(x)


class HistoneEncoder(nn.Module):
    """
    CNN encoder for histone modification profiles.

    Expected input:
        (batch, n_histone_tracks * n_bins)
    """

    def __init__(
        self,
        in_channels: int = 2,
        n_bins: int = 16,
        embed_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.out = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.in_channels = in_channels
        self.n_bins = n_bins

    def forward(
        self,
        histone: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            histone:
                Flattened histone features.
                Shape:
                    (batch, in_channels * n_bins)

        Returns:
            Histone representation.
            Shape: (batch, embed_dim)
        """

        batch = histone.shape[0]

        x = histone.view(
            batch,
            self.in_channels,
            self.n_bins
        )

        x = self.conv(x)

        x = self.pool(x).squeeze(-1)

        return self.out(x)


class TFEncoder(nn.Module):
    """
    MLP encoder for transcription factor binding profiles.
    """

    def __init__(
        self,
        n_bins: int = 16,
        embed_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                n_bins,
                64
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                64,
                embed_dim
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        tf: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tf:
                Flattened TF binding features.

        Returns:
            TF representation.
        """
        return self.net(tf)


class CrossModalFusion(nn.Module):
    """
    Transformer encoder for cross-modal fusion.

    Each modality is represented as one embedding/token.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

    def forward(
        self,
        modalities: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            modalities:
                Shape:
                (batch, num_modalities, embed_dim)

        Returns:
            Fused representation.
            Shape:
            (batch, embed_dim)
        """

        x = self.encoder(modalities)

        return x.mean(dim=1)


@dataclass
class ModalityConfig:
    """
    Configuration describing the functional genomic inputs.

    ATAC is intentionally absent because it is the prediction target.
    """

    histone_tracks: List[str]
    tf_tracks: List[str]
    n_bins: int


def infer_modality_config(
    feature_map: Optional[Dict]
) -> ModalityConfig:
    """
    Infer histone and TF tracks from the feature map.

    ATAC tracks are deliberately excluded because ATAC-seq
    is the prediction target rather than a model input.
    """

    if not feature_map:
        return ModalityConfig(
            histone_tracks=[],
            tf_tracks=[],
            n_bins=16,
        )

    tracks = feature_map.get(
        "bigwigs",
        []
    )

    n_bins = feature_map.get(
        "n_bins",
        16
    )

    histones: List[str] = []
    tf_tracks: List[str] = []

    for track in tracks:

        track_lower = track.lower()

        # Histone modification tracks
        if "h3k" in track_lower:
            histones.append(track)

        # Everything else is treated as a TF track.
        # ATAC tracks are explicitly ignored.
        elif "atac" not in track_lower:
            tf_tracks.append(track)

    return ModalityConfig(
        histone_tracks=histones,
        tf_tracks=tf_tracks,
        n_bins=n_bins,
    )


class ModalityGate(nn.Module):
    """
    Learnable gate controlling the contribution of a modality.
    """

    def __init__(
        self,
        embed_dim: int,
        init_value: float = 0.0,
    ):
        super().__init__()

        self.logit = nn.Parameter(
            torch.full(
                (1,),
                init_value
            )
        )

        self.proj = nn.Linear(
            embed_dim,
            embed_dim
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        gate = torch.sigmoid(
            self.logit
        )

        return gate * self.proj(x)


class MultiModalAccessibilityModel(nn.Module):
    """
    Multi-Modal Transformer for chromatin accessibility prediction.

    Inputs:
        1. DNA sequence
        2. Histone modification signals
        3. Transcription factor binding signals

    Target:
        ATAC-seq accessibility signal

    ATAC-seq is NOT provided to this model as an input.
    """

    def __init__(
        self,
        seq_len: int,
        feature_map: Optional[Dict] = None,
        seq_embed_dim: int = 256,
        fusion_embed_dim: int = 128,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.modality_cfg = infer_modality_config(
            feature_map
        )

        self.track_index_map = _build_track_index_map(
            feature_map
        )

        # --------------------------------------------------
        # DNA sequence encoder
        # --------------------------------------------------

        self.sequence_encoder = SequenceEncoder(
            seq_len=seq_len,
            embed_dim=seq_embed_dim,
            n_heads=8,
            n_layers=4,
            dim_feedforward=1024,
            dropout=dropout,
        )

        self.sequence_projection = nn.Sequential(
            nn.Linear(
                seq_embed_dim,
                fusion_embed_dim
            ),
            nn.LayerNorm(
                fusion_embed_dim
            ),
        )

        # --------------------------------------------------
        # Histone encoder
        # --------------------------------------------------

        num_histone_tracks = max(
            len(self.modality_cfg.histone_tracks),
            1
        )

        self.histone_encoder = HistoneEncoder(
            in_channels=num_histone_tracks,
            n_bins=self.modality_cfg.n_bins,
            embed_dim=fusion_embed_dim,
            dropout=dropout,
        )

        self.histone_gate = ModalityGate(
            fusion_embed_dim,
            init_value=0.5
        )

        # --------------------------------------------------
        # TF encoder
        # --------------------------------------------------

        num_tf_features = (
            self.modality_cfg.n_bins
            * max(
                len(self.modality_cfg.tf_tracks),
                1
            )
        )

        self.tf_encoder = TFEncoder(
            n_bins=num_tf_features,
            embed_dim=fusion_embed_dim,
            dropout=dropout,
        )

        self.tf_gate = ModalityGate(
            fusion_embed_dim,
            init_value=0.5
        )

        # --------------------------------------------------
        # Cell-type embedding
        # --------------------------------------------------

        self.cell_embedding = nn.Parameter(
            torch.randn(
                1,
                fusion_embed_dim
            )
        )

        # --------------------------------------------------
        # Cross-modal fusion
        # --------------------------------------------------

        self.fusion = CrossModalFusion(
            embed_dim=fusion_embed_dim,
            n_heads=fusion_heads,
            n_layers=fusion_layers,
            dropout=dropout,
        )

        # --------------------------------------------------
        # Regression head
        # --------------------------------------------------

        self.output_head = nn.Sequential(
            nn.Linear(
                fusion_embed_dim,
                fusion_embed_dim // 2
            ),
            nn.LayerNorm(
                fusion_embed_dim // 2
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                fusion_embed_dim // 2,
                1
            ),
        )

    def _slice_track(
        self,
        functional: torch.Tensor,
        track: str
    ) -> Optional[torch.Tensor]:
        """
        Extract a specific track from the flattened
        functional genomic feature tensor.
        """

        if track not in self.track_index_map:
            return None

        start, end = self.track_index_map[track]

        return functional[:, start:end]

    def forward(
        self,
        sequence: torch.Tensor,
        functional: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequence:
                One-hot encoded DNA sequence.
                Shape: (batch, seq_len, 4)

            functional:
                Flattened functional genomic signals containing
                histone and TF tracks.
                ATAC must NOT be included.

        Returns:
            Predicted continuous accessibility.
            Shape: (batch,)
        """

        batch_size = sequence.size(0)

        # ==================================================
        # Sequence modality
        # ==================================================

        seq_repr = self.sequence_encoder(
            sequence
        )

        seq_repr = self.sequence_projection(
            seq_repr
        )

        modality_embeddings = [
            seq_repr.unsqueeze(1)
        ]

        # ==================================================
        # Functional genomic modalities
        # ==================================================

        if functional is not None:

            # --------------------------------------------------
            # Histone modifications
            # --------------------------------------------------

            histone_tracks = (
                self.modality_cfg.histone_tracks
            )

            if histone_tracks:

                histone_slices = [
                    self._slice_track(
                        functional,
                        track
                    )
                    for track in histone_tracks
                ]

                # Make sure all requested tracks exist.
                if any(
                    x is None
                    for x in histone_slices
                ):
                    missing = [
                        track
                        for track, value
                        in zip(
                            histone_tracks,
                            histone_slices
                        )
                        if value is None
                    ]

                    raise ValueError(
                        "Missing histone tracks in "
                        f"functional feature map: {missing}"
                    )

                histone_tensor = torch.stack(
                    histone_slices,
                    dim=1
                )

                # (batch, n_histone, n_bins)
                # -> (batch, n_histone * n_bins)
                histone_tensor = histone_tensor.view(
                    batch_size,
                    -1
                )

                histone_repr = self.histone_gate(
                    self.histone_encoder(
                        histone_tensor
                    )
                )

                modality_embeddings.append(
                    histone_repr.unsqueeze(1)
                )

            # --------------------------------------------------
            # Transcription factor binding
            # --------------------------------------------------

            tf_tracks = (
                self.modality_cfg.tf_tracks
            )

            if tf_tracks:

                tf_slices = [
                    self._slice_track(
                        functional,
                        track
                    )
                    for track in tf_tracks
                ]

                # Make sure all requested tracks exist.
                if any(
                    x is None
                    for x in tf_slices
                ):
                    missing = [
                        track
                        for track, value
                        in zip(
                            tf_tracks,
                            tf_slices
                        )
                        if value is None
                    ]

                    raise ValueError(
                        "Missing TF tracks in "
                        f"functional feature map: {missing}"
                    )

                tf_tensor = torch.stack(
                    tf_slices,
                    dim=1
                )

                # (batch, n_tf, n_bins)
                # -> (batch, n_tf * n_bins)
                tf_tensor = tf_tensor.view(
                    batch_size,
                    -1
                )

                tf_repr = self.tf_gate(
                    self.tf_encoder(
                        tf_tensor
                    )
                )

                modality_embeddings.append(
                    tf_repr.unsqueeze(1)
                )

        # ==================================================
        # Cell embedding
        # ==================================================

        cell_embed = (
            self.cell_embedding
            .expand(
                batch_size,
                -1
            )
            .unsqueeze(1)
        )

        modality_embeddings.append(
            cell_embed
        )

        # ==================================================
        # Cross-modal fusion
        # ==================================================

        fused_input = torch.cat(
            modality_embeddings,
            dim=1
        )

        fused = self.fusion(
            fused_input
        )

        # ==================================================
        # ATAC prediction
        # ==================================================

        prediction = self.output_head(
            fused
        ).squeeze(-1)

        return prediction