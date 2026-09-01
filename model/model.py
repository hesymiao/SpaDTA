from __future__ import annotations

import math
import os
import random
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, kl_divergence as kld
from torch.utils.data import DataLoader, TensorDataset
import tqdm

from .loss import LossFunction
from .balance_min_norm_solvers import MinNormSolver


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except NameError:
        return False
    return shell == "ZMQInteractiveShell"


def get_tqdm():
    if is_notebook():
        from tqdm.autonotebook import tqdm as tqdm_notebook

        return tqdm_notebook
    return tqdm.tqdm


class LatentDecoderMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float,
    ):
        super().__init__()
        num_layers = max(int(num_layers), 1)
        hidden_dim = max(int(hidden_dim), int(out_dim))
        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
        else:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            for _ in range(num_layers - 2):
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout_rate),
                    ]
                )
            layers.extend(
                [
                    nn.Linear(hidden_dim, out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_private_encoder(
    dim: int,
    num_layers: int,
    activation: str,
) -> nn.Module:
    num_layers = max(int(num_layers), 1)
    activation = str(activation).strip().lower()
    if activation not in {"none", "gelu"}:
        raise ValueError("private_encoder_activation must be 'none' or 'gelu'")
    if num_layers == 1 and activation == "none":
        return nn.Linear(dim, dim, bias=False)

    layers: list[nn.Module] = []
    for layer_index in range(num_layers):
        layers.append(nn.Linear(dim, dim, bias=False))
        if activation == "gelu" and (num_layers == 1 or layer_index < num_layers - 1):
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def normalize_branch_mean_variance_torch(
    branch: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    centered = branch - branch.mean(dim=0, keepdim=True)
    mean_dimension_variance = centered.square().mean(dim=0).mean()
    return centered / mean_dimension_variance.clamp_min(float(eps)).sqrt()


def mask_private_features_for_decoder(
    z_private: torch.Tensor,
    mask_probability: float,
    training: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if z_private.ndim != 2:
        raise ValueError(
            "Expected private latent with shape [batch_size, latent_dim], "
            f"got {tuple(z_private.shape)}"
        )

    if not 0.0 <= float(mask_probability) < 1.0:
        raise ValueError(f"mask_probability must satisfy 0 <= p < 1, got {mask_probability}")

    if not torch.isfinite(z_private).all():
        raise ValueError("Private latent contains NaN or Inf")

    latent_dim = int(z_private.shape[1])
    configured = torch.as_tensor(
        float(mask_probability),
        dtype=z_private.dtype,
        device=z_private.device,
    )

    if (not training) or float(mask_probability) <= 0.0:
        diagnostics = {
            "configured_mask_probability": configured,
            "actual_mask_fraction": torch.zeros((), dtype=z_private.dtype, device=z_private.device),
            "masked_dimension_count": torch.zeros((), dtype=torch.long, device=z_private.device),
            "kept_dimension_count": torch.as_tensor(latent_dim, dtype=torch.long, device=z_private.device),
        }
        return z_private, diagnostics

    keep_mask = (
        torch.rand(1, latent_dim, device=z_private.device) >= float(mask_probability)
    ).to(dtype=z_private.dtype)
    if int(keep_mask.sum().item()) == 0:
        restore_index = torch.randint(low=0, high=latent_dim, size=(1,), device=z_private.device)
        keep_mask[0, restore_index] = 1.0

    masked_private = z_private * keep_mask
    masked_dimension_count = keep_mask.eq(0).sum()
    kept_dimension_count = keep_mask.eq(1).sum()
    actual_mask_fraction = masked_dimension_count.to(dtype=z_private.dtype) / float(latent_dim)

    diagnostics = {
        "configured_mask_probability": configured,
        "actual_mask_fraction": actual_mask_fraction.detach(),
        "masked_dimension_count": masked_dimension_count.detach(),
        "kept_dimension_count": kept_dimension_count.detach(),
    }
    return masked_private, diagnostics


def get_decoder_private_mask_probability(
    epoch: int,
    enabled: bool,
    target_probability: float,
    warmup_start: int,
    warmup_end: int,
) -> float:
    del epoch
    del warmup_start
    del warmup_end
    if not bool(enabled):
        return 0.0
    if float(target_probability) <= 0.0:
        return 0.0
    return float(target_probability)


def compute_recent_linear_slope(values: list[float], window_epochs: int) -> tuple[float, float]:
    if int(window_epochs) <= 1 or len(values) < int(window_epochs):
        return float("nan"), float("nan")
    recent = np.asarray(values[-int(window_epochs):], dtype=np.float64)
    x = np.arange(recent.shape[0], dtype=np.float64)
    slope = float(np.polyfit(x, recent, 1)[0])
    return slope, abs(slope)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, embed_dropout=0.0, attn_mask=False):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.attn_mask = attn_mask
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, attn_dropout, relu_dropout, res_dropout)
            for _ in range(layers)
        ])
        
    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """
        Args:
            x_in (FloatTensor): embedded inputs of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded inputs of shape `(src_len, batch, embed_dim)` for key
            x_in_v (FloatTensor): embedded inputs of shape `(src_len, batch, embed_dim)` for value
        """
        x = self.embed_scale * x_in
        x = self.embed_dropout(x)
        
        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
        else:
            x_k = x_v = x
        
        # Create attention mask if needed
        attn_mask = None
        if self.attn_mask:
            src_len = x.size(0)
            attn_mask = torch.triu(torch.ones(src_len, src_len), diagonal=1).bool()
            attn_mask = attn_mask.to(x.device)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, x_k, x_v, attn_mask)
        
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = MultiheadAttention(embed_dim, num_heads, attn_dropout=attn_dropout)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        
        self.relu_dropout = nn.Dropout(relu_dropout)
        self.res_dropout = nn.Dropout(res_dropout)
        
    def forward(self, x, x_k=None, x_v=None, attn_mask=None):
        """
        Args:
            x: input to the layer of shape `(seq_len, batch, embed_dim)`
            x_k: key input shape `(seq_len, batch, embed_dim)`
            x_v: value input shape `(seq_len, batch, embed_dim)`
            attn_mask: attention mask of shape `(seq_len, seq_len)`
        """
        residual = x
        x, _ = self.self_attn(query=x, key=x_k if x_k is not None else x, 
                             value=x_v if x_v is not None else x, attn_mask=attn_mask)
        x = self.res_dropout(x)
        x = residual + x
        x = self.attn_layer_norm(x)
        
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.relu_dropout(x)
        x = self.fc2(x)
        x = self.res_dropout(x)
        x = residual + x
        x = self.final_layer_norm(x)
        
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, bias=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
            
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
            
    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: `(L, N, E)` where L is the target sequence length, N is the batch size, E is embedding dim
            key: `(S, N, E)`, where S is the source sequence length
            value: `(S, N, E)`
            attn_mask: `(L, S)` where L is target length, S is source length
        """
        return self._multi_head_attention_forward(query, key, value, attn_mask)
        
    def _multi_head_attention_forward(self, query, key, value, attn_mask=None):
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        
        # Linear projections
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Scaled dot-product attention
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).expand(bsz * self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn = torch.bmm(attn_weights, v)
        
        # Reshape and project
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        
        return attn, attn_weights


class BaseDecAlignSpatialMetaLinear(nn.Module):
    def __init__(
        self,
        adata,
        proj_dim: int = 256,
        token_dim: int = 128,
        n_latent: int = 10,
        num_prototypes: int = 8,
        dropout_rate: float = 0.1,
        device: str = "cpu",
        reconstruction_method_st: Literal["mse", "zg", "zinb"] = "zinb",
        reconstruction_method_sm: Literal["mse", "zg", "g"] = "g",
        lambda_ot: float = 0.1,
        ot_num_iters: int = 50,
        num_heads: int = 8,
        nlevels: int = 2,
        attn_dropout: float = 0.1,
        relu_dropout: float = 0.1,
        res_dropout: float = 0.1,
        embed_dropout: float = 0.1,
        standardize_inputs: bool = True,
        use_standardized_reconstruction: bool = True,
        feature_input_mode: bool = False,
        decoder_private_feature_masking: bool = False,
        decoder_private_mask_probability: float = 0.0,
        decoder_private_mask_warmup_start: int = 0,
        decoder_private_mask_warmup_end: int = 0,
        private_encoder_num_layers: int = 1,
        private_encoder_activation: str = "none",
        shared_graph_mode: str = "praga_fused",
    ):
        super().__init__()
        self.adata = adata
        self.device = torch.device(device)
        self.proj_dim = proj_dim
        self.token_dim = token_dim
        self.n_latent = n_latent
        self.reconstruction_method_st = reconstruction_method_st
        self.reconstruction_method_sm = reconstruction_method_sm
        self.num_prototypes = num_prototypes
        self.ot_reg = lambda_ot
        self.ot_num_iters = ot_num_iters
        self.standardize_inputs = standardize_inputs
        self.use_standardized_reconstruction = use_standardized_reconstruction
        self.feature_input_mode = bool(feature_input_mode)
        self.decoder_private_feature_masking = bool(decoder_private_feature_masking)
        self.decoder_private_mask_probability = float(decoder_private_mask_probability)
        self.decoder_private_mask_warmup_start = int(decoder_private_mask_warmup_start)
        self.decoder_private_mask_warmup_end = int(decoder_private_mask_warmup_end)
        self.private_encoder_num_layers = max(int(private_encoder_num_layers), 1)
        self.private_encoder_activation = str(private_encoder_activation).strip().lower()
        self.shared_graph_mode = str(shared_graph_mode).strip().lower()
        if self.shared_graph_mode not in {"praga_fused", "spatial_only", "no_graph"}:
            raise ValueError(
                f"Unsupported shared_graph_mode={shared_graph_mode!r}; expected 'praga_fused', 'spatial_only', or 'no_graph'."
            )
        self._decoder_private_mask_probability_current = 0.0

        self._init_dataset()

        self.dropout = nn.Dropout(dropout_rate)
        self.proj_st = nn.Linear(self.in_dim_st, proj_dim, bias=False)
        self.proj_sm = nn.Linear(self.in_dim_sm, proj_dim, bias=False)
        self.norm_st = nn.LayerNorm(proj_dim)
        self.norm_sm = nn.LayerNorm(proj_dim)

        self.encoder_uni_st = build_private_encoder(
            proj_dim,
            self.private_encoder_num_layers,
            self.private_encoder_activation,
        )
        self.encoder_uni_sm = build_private_encoder(
            proj_dim,
            self.private_encoder_num_layers,
            self.private_encoder_activation,
        )
        self.encoder_com = nn.Linear(proj_dim, proj_dim, bias=False)
        self.encoder_com_activation = nn.GELU()
        self.graph_fuse_st = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.graph_fuse_sm = nn.Parameter(torch.zeros(2, dtype=torch.float32))

        self.private_st_token = nn.Linear(proj_dim, token_dim)
        self.private_sm_token = nn.Linear(proj_dim, token_dim)
        self.common_token = nn.Linear(proj_dim, token_dim)

        self.proto_st = nn.Parameter(torch.randn(num_prototypes, proj_dim))
        self.proto_sm = nn.Parameter(torch.randn(num_prototypes, proj_dim))
        self.logvar_st = nn.Parameter(torch.zeros(num_prototypes, proj_dim))
        self.logvar_sm = nn.Parameter(torch.zeros(num_prototypes, proj_dim))

        self.transformer_fusion = TransformerEncoder(
            embed_dim=token_dim,
            num_heads=num_heads,
            layers=nlevels,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            attn_mask=False,
        )
        self.trans_st_with_sm = TransformerEncoder(
            embed_dim=token_dim,
            num_heads=num_heads,
            layers=nlevels,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            attn_mask=False,
        )
        self.trans_sm_with_st = TransformerEncoder(
            embed_dim=token_dim,
            num_heads=num_heads,
            layers=nlevels,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
            embed_dropout=embed_dropout,
            attn_mask=False,
        )

        self.cma_proj = nn.Linear(2 * token_dim, 2 * token_dim)
        self.q_mu_fc = nn.Linear(4 * token_dim, n_latent)
        self.q_logvar_fc = nn.Linear(4 * token_dim, n_latent)

        self.decoder = nn.Sequential(
            nn.Linear(n_latent, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        if self.feature_input_mode:
            self.px_rna_scale_decoder = nn.Linear(proj_dim, self.in_dim_st)
        else:
            self.px_rna_scale_decoder = nn.Sequential(
                nn.Linear(proj_dim, self.in_dim_st),
                nn.Softmax(dim=-1),
            )
        self.px_rna_rate_decoder = nn.Linear(proj_dim, self.in_dim_st)
        self.px_rna_dropout_decoder = nn.Linear(proj_dim, self.in_dim_st)
        self.px_sm_scale_decoder = nn.Linear(proj_dim, self.in_dim_sm)
        self.px_sm_rate_decoder = nn.Linear(proj_dim, self.in_dim_sm)
        self.px_sm_dropout_decoder = nn.Linear(proj_dim, self.in_dim_sm)

        self.modality_gate = nn.Linear(2 * token_dim, 1)
        self.to(self.device)

    def _init_dataset(self) -> None:
        if "type" not in self.adata.var.columns:
            raise ValueError("joint adata.var 中缺少 type 列。")

        self.X = self.adata.X if scipy.sparse.issparse(self.adata.X) else np.asarray(self.adata.X)
        self.types = np.asarray(self.adata.var["type"].astype(str).values)
        self.st_mask = self.types == "ST"
        self.sm_mask = self.types == "SM"
        self.in_dim_st = int(self.st_mask.sum())
        self.in_dim_sm = int(self.sm_mask.sum())
        self.n_obs = int(self.adata.n_obs)
        self.indices = np.arange(self.n_obs)
        self.encoder_X = self.adata.layers["spadta_encoder_input"]
        st_pca = np.asarray(self.adata.obsm["spadta_expression_pca_st"], dtype=np.float32)
        sm_pca = np.asarray(self.adata.obsm["spadta_expression_pca_sm"], dtype=np.float32)
        self.register_buffer("expression_triplet_st", torch.tensor(st_pca, dtype=torch.float32))
        self.register_buffer("expression_triplet_sm", torch.tensor(sm_pca, dtype=torch.float32))
        self._init_praga_graphs(st_pca=st_pca, sm_pca=sm_pca)


    def _csr_to_undirected_edge_arrays(
        self,
        graph_csr: scipy.sparse.csr_matrix,
    ) -> tuple[np.ndarray, np.ndarray]:
        graph_coo = graph_csr.tocoo()
        upper_mask = graph_coo.row < graph_coo.col
        rows = graph_coo.row[upper_mask].astype(np.int64, copy=False)
        cols = graph_coo.col[upper_mask].astype(np.int64, copy=False)
        return rows, cols

    def _symmetric_edges_to_csr(self, row: np.ndarray, col: np.ndarray) -> scipy.sparse.csr_matrix:
        rows = np.concatenate([row, col]).astype(np.int64, copy=False)
        cols = np.concatenate([col, row]).astype(np.int64, copy=False)
        data = np.ones(rows.shape[0], dtype=np.float32)
        return scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_obs, self.n_obs),
            dtype=np.float32,
        )

    def _compute_graph_basic_stats(self, graph_csr: scipy.sparse.csr_matrix) -> dict[str, float]:
        degree = np.asarray(graph_csr.sum(axis=1)).reshape(-1)
        return {
            "node_count": int(graph_csr.shape[0]),
            "undirected_edge_count": int(graph_csr.nnz // 2),
            "average_degree": float(degree.mean()) if degree.size > 0 else 0.0,
            "isolated_node_ratio": float((degree == 0).mean()) if degree.size > 0 else 0.0,
        }

    def _inverse_softplus_scalar(self, value: float) -> float:
        return float(np.log(np.expm1(float(value))))

    def _register_graph_edge_buffers(
        self,
        prefix: str,
        row: np.ndarray,
        col: np.ndarray,
    ) -> None:
        self.register_buffer(f"{prefix}_row", torch.tensor(row, dtype=torch.long))
        self.register_buffer(f"{prefix}_col", torch.tensor(col, dtype=torch.long))

    def _init_praga_graphs(self, st_pca: np.ndarray, sm_pca: np.ndarray) -> None:
        expr_k = min(max(int(self.spatial_contrastive_pos_k), 1), max(self.n_obs - 1, 1))
        spatial_k = min(6, max(self.n_obs - 1, 1))
        self.spatial_graph_csr = self.adata.obsp["spadta_graph_spatial"].tocsr()
        self.expr_st_graph_csr = self.adata.obsp["spadta_graph_expression_st"].tocsr()
        self.expr_sm_graph_csr = self.adata.obsp["spadta_graph_expression_sm"].tocsr()

        self.expr_st_graph_indptr = self.expr_st_graph_csr.indptr.astype(np.int64, copy=False)
        self.expr_st_graph_indices = self.expr_st_graph_csr.indices.astype(np.int64, copy=False)
        self.expr_sm_graph_indptr = self.expr_sm_graph_csr.indptr.astype(np.int64, copy=False)
        self.expr_sm_graph_indices = self.expr_sm_graph_csr.indices.astype(np.int64, copy=False)
        self.spatial_graph_indptr = self.spatial_graph_csr.indptr.astype(np.int64, copy=False)
        self.spatial_graph_indices = self.spatial_graph_csr.indices.astype(np.int64, copy=False)

        self.expr_graph_k = int(expr_k)
        self.spatial_graph_k = int(spatial_k)
        self.expr_st_graph_nnz = int(self.expr_st_graph_csr.nnz)
        self.expr_sm_graph_nnz = int(self.expr_sm_graph_csr.nnz)
        self.spatial_graph_nnz = int(self.spatial_graph_csr.nnz)

        spatial_row, spatial_col = self._csr_to_undirected_edge_arrays(self.spatial_graph_csr)
        expr_st_row, expr_st_col = self._csr_to_undirected_edge_arrays(self.expr_st_graph_csr)
        expr_sm_row, expr_sm_col = self._csr_to_undirected_edge_arrays(self.expr_sm_graph_csr)
        self._register_graph_edge_buffers("spatial_edge", spatial_row, spatial_col)
        self._register_graph_edge_buffers("expr_st_edge", expr_st_row, expr_st_col)
        self._register_graph_edge_buffers("expr_sm_edge", expr_sm_row, expr_sm_col)
        init_edge_logit = self._inverse_softplus_scalar(1.0)
        self.expr_st_edge_logits = nn.Parameter(
            torch.full((expr_st_row.shape[0],), init_edge_logit, dtype=torch.float32)
        )
        self.expr_sm_edge_logits = nn.Parameter(
            torch.full((expr_sm_row.shape[0],), init_edge_logit, dtype=torch.float32)
        )

        self.expr_st_graph_stats = self._compute_graph_basic_stats(self.expr_st_graph_csr)
        self.expr_sm_graph_stats = self._compute_graph_basic_stats(self.expr_sm_graph_csr)
        self.spatial_graph_stats = self._compute_graph_basic_stats(self.spatial_graph_csr)

        self.st_graph_union_csr = self.spatial_graph_csr.maximum(self.expr_st_graph_csr).tocsr()
        self.st_graph_union_csr.data[:] = 1.0
        self.st_graph_union_csr.eliminate_zeros()
        self.sm_graph_union_csr = self.spatial_graph_csr.maximum(self.expr_sm_graph_csr).tocsr()
        self.sm_graph_union_csr.data[:] = 1.0
        self.sm_graph_union_csr.eliminate_zeros()
        self.st_graph_union_nnz = int(self.st_graph_union_csr.nnz)
        self.sm_graph_union_nnz = int(self.sm_graph_union_csr.nnz)

    def _collect_neighbors_from_graph(
        self,
        graph_indptr: np.ndarray,
        graph_indices: np.ndarray,
        anchor_indices: np.ndarray,
    ) -> np.ndarray:
        collected: list[np.ndarray] = []
        for anchor_idx in np.asarray(anchor_indices, dtype=np.int64):
            start = int(graph_indptr[anchor_idx])
            end = int(graph_indptr[anchor_idx + 1])
            if end > start:
                collected.append(graph_indices[start:end])
        if not collected:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(collected).astype(np.int64, copy=False)

    def _expand_shared_graph_indices(self, batch_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base_indices = np.asarray(batch_indices, dtype=np.int64)
        ordered = list(base_indices.tolist())
        seen = set(ordered)
        if self.shared_graph_mode == "no_graph":
            neighbor_sources = ()
        elif self.shared_graph_mode == "spatial_only":
            neighbor_sources = (
                self._collect_neighbors_from_graph(self.spatial_graph_indptr, self.spatial_graph_indices, base_indices),
            )
        else:
            neighbor_sources = (
                self._collect_neighbors_from_graph(self.spatial_graph_indptr, self.spatial_graph_indices, base_indices),
                self._collect_neighbors_from_graph(self.expr_st_graph_indptr, self.expr_st_graph_indices, base_indices),
                self._collect_neighbors_from_graph(self.expr_sm_graph_indptr, self.expr_sm_graph_indices, base_indices),
            )
        for neighbor_values in neighbor_sources:
            for idx in neighbor_values.tolist():
                if idx not in seen:
                    seen.add(idx)
                    ordered.append(int(idx))
        expanded = np.asarray(ordered, dtype=np.int64)
        batch_local_idx = np.arange(base_indices.shape[0], dtype=np.int64)
        return expanded, batch_local_idx

    def _csr_subgraph_to_torch_sparse(
        self,
        graph_csr: scipy.sparse.csr_matrix,
        node_indices: np.ndarray,
    ) -> torch.Tensor:
        subgraph = graph_csr[node_indices][:, node_indices].tocoo()
        if subgraph.nnz == 0:
            empty_idx = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            empty_vals = torch.zeros((0,), dtype=torch.float32, device=self.device)
            return torch.sparse_coo_tensor(empty_idx, empty_vals, (len(node_indices), len(node_indices)), device=self.device).coalesce()
        indices = torch.tensor(
            np.vstack([subgraph.row, subgraph.col]),
            dtype=torch.long,
            device=self.device,
        )
        values = torch.tensor(subgraph.data, dtype=torch.float32, device=self.device)
        return torch.sparse_coo_tensor(indices, values, (len(node_indices), len(node_indices)), device=self.device).coalesce()

    def _undirected_edges_to_bidirectional(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_row = torch.cat([row, col], dim=0)
        edge_col = torch.cat([col, row], dim=0)
        edge_weight = torch.cat([weights, weights], dim=0)
        return edge_row, edge_col, edge_weight

    def _build_full_normalized_fused_graph(
        self,
        *,
        expr_edge_row: torch.Tensor,
        expr_edge_col: torch.Tensor,
        expr_edge_logits: torch.Tensor,
        graph_fuse_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expr_edge_weight = F.softplus(expr_edge_logits)
        if self.shared_graph_mode == "no_graph":
            graph_weights = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.device)
            diag_idx = torch.arange(self.n_obs, device=self.device, dtype=torch.long)
            diag_indices = torch.stack([diag_idx, diag_idx], dim=0)
            diag_values = torch.ones(self.n_obs, dtype=torch.float32, device=self.device)
            normalized_graph = torch.sparse_coo_tensor(
                diag_indices,
                diag_values,
                (self.n_obs, self.n_obs),
                device=self.device,
            ).coalesce()
            return normalized_graph, graph_weights, expr_edge_weight
        if self.shared_graph_mode == "spatial_only":
            graph_weights = torch.tensor([1.0, 0.0], dtype=torch.float32, device=self.device)
            spatial_edge_weight = torch.ones_like(self.spatial_edge_row, dtype=torch.float32)
            fused_indices = torch.stack(
                [
                    torch.cat([self.spatial_edge_row, self.spatial_edge_col], dim=0),
                    torch.cat([self.spatial_edge_col, self.spatial_edge_row], dim=0),
                ],
                dim=0,
            )
            fused_values = torch.cat([spatial_edge_weight, spatial_edge_weight], dim=0)
        else:
            graph_weights = torch.softmax(graph_fuse_logits, dim=0)
            spatial_weight = graph_weights[0]
            expression_weight = graph_weights[1]
            spatial_edge_weight = spatial_weight * torch.ones_like(self.spatial_edge_row, dtype=torch.float32)
            expr_scaled_weight = expression_weight * expr_edge_weight

            spatial_row, spatial_col, spatial_values = self._undirected_edges_to_bidirectional(
                self.spatial_edge_row,
                self.spatial_edge_col,
                spatial_edge_weight,
            )
            expr_row, expr_col, expr_values = self._undirected_edges_to_bidirectional(
                expr_edge_row,
                expr_edge_col,
                expr_scaled_weight,
            )

            fused_indices = torch.stack(
                [torch.cat([spatial_row, expr_row], dim=0), torch.cat([spatial_col, expr_col], dim=0)],
                dim=0,
            )
            fused_values = torch.cat([spatial_values, expr_values], dim=0)
        fused_graph = torch.sparse_coo_tensor(
            fused_indices,
            fused_values,
            (self.n_obs, self.n_obs),
            device=self.device,
        ).coalesce()

        diag_idx = torch.arange(self.n_obs, device=self.device, dtype=torch.long)
        diag_indices = torch.stack([diag_idx, diag_idx], dim=0)
        diag_values = torch.ones(self.n_obs, dtype=fused_values.dtype, device=self.device)
        norm_graph = torch.sparse_coo_tensor(
            torch.cat([fused_graph.indices(), diag_indices], dim=1),
            torch.cat([fused_graph.values(), diag_values], dim=0),
            (self.n_obs, self.n_obs),
            device=self.device,
        ).coalesce()

        row_idx, col_idx = norm_graph.indices()
        values = norm_graph.values()
        degree = torch.zeros(self.n_obs, dtype=values.dtype, device=self.device)
        degree.scatter_add_(0, row_idx, values)
        inv_sqrt_degree = degree.clamp(min=1e-12).pow(-0.5)
        norm_values = inv_sqrt_degree[row_idx] * values * inv_sqrt_degree[col_idx]
        normalized_graph = torch.sparse_coo_tensor(
            norm_graph.indices(),
            norm_values,
            norm_graph.shape,
            device=self.device,
        ).coalesce()
        return normalized_graph, graph_weights, expr_edge_weight

    def _slice_full_normalized_graph(
        self,
        normalized_graph: torch.Tensor,
        expanded_indices: np.ndarray,
    ) -> torch.Tensor:
        expanded_indices_t = torch.as_tensor(expanded_indices, dtype=torch.long, device=self.device)
        local_index = torch.full((self.n_obs,), -1, dtype=torch.long, device=self.device)
        local_index[expanded_indices_t] = torch.arange(expanded_indices_t.shape[0], device=self.device, dtype=torch.long)

        full_row, full_col = normalized_graph.indices()
        local_row = local_index[full_row]
        local_col = local_index[full_col]
        keep_mask = (local_row >= 0) & (local_col >= 0)

        sub_indices = torch.stack([local_row[keep_mask], local_col[keep_mask]], dim=0)
        sub_values = normalized_graph.values()[keep_mask]
        return torch.sparse_coo_tensor(
            sub_indices,
            sub_values,
            (expanded_indices_t.shape[0], expanded_indices_t.shape[0]),
            device=self.device,
        ).coalesce()

    def _encode_shared_with_fused_graphs(
        self,
        batch_indices: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        expanded_indices, batch_local_idx = self._expand_shared_graph_indices(batch_indices)
        X_union = self._fetch_encoder_rows(expanded_indices)
        X_st_union = X_union[:, self.st_mask]
        X_sm_union = X_union[:, self.sm_mask]
        h_st_union, h_sm_union = self.project_inputs(X_st_union, X_sm_union)

        fused_st_full, weight_st, expr_st_edge_weight = self._build_full_normalized_fused_graph(
            expr_edge_row=self.expr_st_edge_row,
            expr_edge_col=self.expr_st_edge_col,
            expr_edge_logits=self.expr_st_edge_logits,
            graph_fuse_logits=self.graph_fuse_st,
        )
        fused_sm_full, weight_sm, expr_sm_edge_weight = self._build_full_normalized_fused_graph(
            expr_edge_row=self.expr_sm_edge_row,
            expr_edge_col=self.expr_sm_edge_col,
            expr_edge_logits=self.expr_sm_edge_logits,
            graph_fuse_logits=self.graph_fuse_sm,
        )
        fused_st = self._slice_full_normalized_graph(fused_st_full, expanded_indices)
        fused_sm = self._slice_full_normalized_graph(fused_sm_full, expanded_indices)

        aggregated_st = torch.sparse.mm(fused_st, h_st_union)
        aggregated_sm = torch.sparse.mm(fused_sm, h_sm_union)
        c_st_union = self.dropout(self.encoder_com_activation(self.encoder_com(aggregated_st)))
        c_sm_union = self.dropout(self.encoder_com_activation(self.encoder_com(aggregated_sm)))

        batch_local_idx_t = torch.as_tensor(batch_local_idx, dtype=torch.long, device=self.device)
        return {
            "c_st": c_st_union[batch_local_idx_t],
            "c_sm": c_sm_union[batch_local_idx_t],
            "weight_st": weight_st,
            "weight_sm": weight_sm,
            "expr_st_edge_weight": expr_st_edge_weight,
            "expr_sm_edge_weight": expr_sm_edge_weight,
        }

    def get_graph_fusion_weights(self) -> dict[str, float]:
        if self.shared_graph_mode == "no_graph":
            return {
                "graph_fuse_st_spatial_weight": 0.0,
                "graph_fuse_st_expression_weight": 0.0,
                "graph_fuse_sm_spatial_weight": 0.0,
                "graph_fuse_sm_expression_weight": 0.0,
            }
        if self.shared_graph_mode == "spatial_only":
            return {
                "graph_fuse_st_spatial_weight": 1.0,
                "graph_fuse_st_expression_weight": 0.0,
                "graph_fuse_sm_spatial_weight": 1.0,
                "graph_fuse_sm_expression_weight": 0.0,
            }
        with torch.no_grad():
            weight_st = torch.softmax(self.graph_fuse_st, dim=0)
            weight_sm = torch.softmax(self.graph_fuse_sm, dim=0)
        return {
            "graph_fuse_st_spatial_weight": float(weight_st[0].item()),
            "graph_fuse_st_expression_weight": float(weight_st[1].item()),
            "graph_fuse_sm_spatial_weight": float(weight_sm[0].item()),
            "graph_fuse_sm_expression_weight": float(weight_sm[1].item()),
        }

    def summarize_fused_graphs(self) -> dict[str, object]:
        return {
            "shared_graph_mode": self.shared_graph_mode,
            "graph_shape": [int(self.n_obs), int(self.n_obs)],
            "expr_graph_k": int(self.expr_graph_k),
            "spatial_graph_k": int(self.spatial_graph_k),
            "A_expr_st_nnz": int(self.expr_st_graph_nnz),
            "A_expr_sm_nnz": int(self.expr_sm_graph_nnz),
            "A_spatial_nnz": int(self.spatial_graph_nnz),
            "A_st_union_nnz": int(self.st_graph_union_nnz),
            "A_sm_union_nnz": int(self.sm_graph_union_nnz),
            "expr_st_graph_stats": dict(self.expr_st_graph_stats),
            "expr_sm_graph_stats": dict(self.expr_sm_graph_stats),
            "spatial_graph_stats": dict(self.spatial_graph_stats),
            "expr_st_edge_parameter_count": int(self.expr_st_edge_logits.numel()),
            "expr_sm_edge_parameter_count": int(self.expr_sm_edge_logits.numel()),
            "expression_graph_construction": "union-symmetrized KNN on PCA features",
            "normalization": "full-graph symmetric D^{-1/2}(A+I)D^{-1/2} after sparse weighted fusion, then subgraph extraction",
            "raw_graph_stage": "unnormalized sparse undirected graph without self-loops",
        }

    def get_expression_edge_weight_stats(self) -> dict[str, float]:
        with torch.no_grad():
            st_weight = F.softplus(self.expr_st_edge_logits)
            sm_weight = F.softplus(self.expr_sm_edge_logits)
        return {
            "expr_st_edge_weight_mean": float(st_weight.mean().item()),
            "expr_st_edge_weight_std": float(st_weight.std(unbiased=False).item()),
            "expr_st_edge_weight_min": float(st_weight.min().item()),
            "expr_st_edge_weight_max": float(st_weight.max().item()),
            "expr_sm_edge_weight_mean": float(sm_weight.mean().item()),
            "expr_sm_edge_weight_std": float(sm_weight.std(unbiased=False).item()),
            "expr_sm_edge_weight_min": float(sm_weight.min().item()),
            "expr_sm_edge_weight_max": float(sm_weight.max().item()),
        }

    def get_expression_edge_gradient_stats(self) -> dict[str, object]:
        def summarize_grad(grad: Optional[torch.Tensor]) -> dict[str, object]:
            if grad is None:
                return {
                    "has_gradient": False,
                    "nonzero_gradient_count": 0,
                    "gradient_abs_mean": 0.0,
                    "gradient_abs_max": 0.0,
                }
            abs_grad = grad.abs()
            return {
                "has_gradient": bool(torch.any(abs_grad > 0).item()),
                "nonzero_gradient_count": int((abs_grad > 0).sum().item()),
                "gradient_abs_mean": float(abs_grad.mean().item()),
                "gradient_abs_max": float(abs_grad.max().item()),
            }

        return {
            "expr_st_edge_gradient": summarize_grad(self.expr_st_edge_logits.grad),
            "expr_sm_edge_gradient": summarize_grad(self.expr_sm_edge_logits.grad),
        }

    def as_dataloader(self, batch_size: int = 128, shuffle: bool = True, generator: Optional[torch.Generator] = None) -> DataLoader:
        dataset = TensorDataset(torch.tensor(self.indices, dtype=torch.long))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

    def _fetch_rows(self, indices: np.ndarray) -> torch.Tensor:
        if scipy.sparse.issparse(self.X):
            rows = self.X[indices].toarray()
        else:
            rows = self.X[indices]
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _fetch_encoder_rows(self, indices: np.ndarray) -> torch.Tensor:
        if scipy.sparse.issparse(self.encoder_X):
            rows = self.encoder_X[indices].toarray()
        else:
            rows = self.encoder_X[indices]
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _transform_st_features(self, X_st: torch.Tensor) -> torch.Tensor:
        if self.feature_input_mode:
            return X_st
        return torch.log1p(X_st.clamp_min(0.0))

    def _transform_sm_features(self, X_sm: torch.Tensor) -> torch.Tensor:
        if self.feature_input_mode:
            return X_sm
        return torch.log1p(X_sm.clamp_min(0.0))

    def _transform_sm_prediction(self, X_sm: torch.Tensor) -> torch.Tensor:
        if self.feature_input_mode:
            return X_sm
        return torch.log1p(F.softplus(X_sm))

    def _get_decoder_private_mask_probability_current(self) -> float:
        return float(getattr(self, "_decoder_private_mask_probability_current", 0.0))

    def _prepare_private_latents_for_decoder(
        self,
        H: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        mask_probability = self._get_decoder_private_mask_probability_current()
        z_st_for_decoder, st_mask_stats = mask_private_features_for_decoder(
            H["z_st"],
            mask_probability=mask_probability,
            training=self.training and self.decoder_private_feature_masking,
        )
        z_sm_for_decoder, sm_mask_stats = mask_private_features_for_decoder(
            H["z_sm"],
            mask_probability=mask_probability,
            training=self.training and self.decoder_private_feature_masking,
        )
        return z_st_for_decoder, z_sm_for_decoder, st_mask_stats, sm_mask_stats

    def _standardize_st_features(self, X_st: torch.Tensor) -> torch.Tensor:
        X_st = self._transform_st_features(X_st)
        return (X_st - self.st_feature_mean.unsqueeze(0)) / self.st_feature_std.unsqueeze(0)

    def _standardize_sm_features(self, X_sm: torch.Tensor) -> torch.Tensor:
        X_sm = self._transform_sm_features(X_sm)
        return (X_sm - self.sm_feature_mean.unsqueeze(0)) / self.sm_feature_std.unsqueeze(0)

    def _standardize_sm_prediction(self, X_sm: torch.Tensor) -> torch.Tensor:
        X_sm = self._transform_sm_prediction(X_sm)
        return (X_sm - self.sm_feature_mean.unsqueeze(0)) / self.sm_feature_std.unsqueeze(0)

    def _reconstruction_mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        loss = (pred - target).pow(2)
        if reduction == "sum":
            return loss.sum(dim=1)
        if reduction == "mean":
            return loss.mean(dim=1)
        if reduction == "none":
            return loss
        raise ValueError(f"不支持的 reconstruction reduction: {reduction}")

    def project_inputs(self, X_st: torch.Tensor, X_sm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_st = self.norm_st(self.proj_st(X_st))
        h_sm = self.norm_sm(self.proj_sm(X_sm))
        return self.dropout(h_st), self.dropout(h_sm)

    def compute_decoupling_loss(self, specific: torch.Tensor, common: torch.Tensor) -> torch.Tensor:
        cos = F.cosine_similarity(specific, common, dim=1)
        return torch.mean(cos.pow(2))

    def compute_prototypes(self, features: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        diff = features.unsqueeze(1) - proto.unsqueeze(0)
        dist_sq = torch.sum(diff.pow(2), dim=2)
        return torch.softmax(-dist_sq, dim=1)

    def pairwise_cost(
        self,
        mu1: torch.Tensor,
        logvar1: torch.Tensor,
        mu2: torch.Tensor,
        logvar2: torch.Tensor,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)
        dist_sq = torch.sum(diff.pow(2), dim=2)
        sigma1 = torch.exp(logvar1)
        sigma2 = torch.exp(logvar2)
        cov_term = torch.sum(
            sigma1.unsqueeze(1) + sigma2.unsqueeze(0) - 2 * torch.sqrt(sigma1.unsqueeze(1) * sigma2.unsqueeze(0) + eps),
            dim=2,
        )
        return dist_sq + cov_term

    def sinkhorn_transport(
        self,
        cost_matrix: torch.Tensor,
        nu_st: torch.Tensor,
        nu_sm: torch.Tensor,
        reg: float,
        num_iters: int = 50,
        eps: float = 1e-9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel = torch.exp(-cost_matrix / reg)
        u = torch.ones_like(nu_st)
        v = torch.ones_like(nu_sm)
        for _ in range(num_iters):
            u = nu_st / (torch.matmul(kernel, v) + eps)
            v = nu_sm / (torch.matmul(kernel.t(), u) + eps)
        transport = u.unsqueeze(1) * kernel * v.unsqueeze(0)
        ot_loss = torch.sum(transport * cost_matrix)
        entropy = -torch.sum(transport * torch.log(transport + eps))
        return transport, ot_loss + 0.001 * reg * entropy

    def compute_hetero_loss(self, s_st: torch.Tensor, s_sm: torch.Tensor) -> torch.Tensor:
        w_st = self.compute_prototypes(s_st, self.proto_st)
        w_sm = self.compute_prototypes(s_sm, self.proto_sm)

        eps = 1e-9
        nu_st = w_st.mean(dim=0)
        nu_sm = w_sm.mean(dim=0)
        nu_st = nu_st / (nu_st.sum() + eps)
        nu_sm = nu_sm / (nu_sm.sum() + eps)

        cost = self.pairwise_cost(self.proto_st, self.logvar_st, self.proto_sm, self.logvar_sm, eps=eps)
        _, ot_loss = self.sinkhorn_transport(cost, nu_st, nu_sm, reg=self.ot_reg, num_iters=self.ot_num_iters)

        loss_st_to_sm = torch.mean(
            w_st * torch.sum((s_st.unsqueeze(1) - self.proto_sm.unsqueeze(0)).pow(2), dim=2)
        )
        loss_sm_to_st = torch.mean(
            w_sm * torch.sum((s_sm.unsqueeze(1) - self.proto_st.unsqueeze(0)).pow(2), dim=2)
        )
        return ot_loss + loss_st_to_sm + loss_sm_to_st

    def compute_homo_loss(self, c_st: torch.Tensor, c_sm: torch.Tensor) -> torch.Tensor:
        def stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu = x.mean(dim=0)
            sigma = x.var(dim=0)
            centered = x - mu.unsqueeze(0)
            skew = (centered.pow(3)).mean(dim=0) / (sigma + 1e-6).pow(1.5)
            return mu, sigma, skew

        mu_st, sigma_st, skew_st = stats(c_st)
        mu_sm, sigma_sm, skew_sm = stats(c_sm)
        semantic = (
            (mu_st - mu_sm).pow(2).sum()
            + (sigma_st - sigma_sm).pow(2).sum()
            + (skew_st - skew_sm).pow(2).sum()
        )
        return semantic + LossFunction.mmd_loss_trvae(c_st, c_sm)

    def encode(self, X: torch.Tensor):
        X_st = X[:, self.st_mask]
        X_sm = X[:, self.sm_mask]

        h_st, h_sm = self.project_inputs(X_st, X_sm)
        s_st = self.encoder_uni_st(h_st)
        s_sm = self.encoder_uni_sm(h_sm)
        c_st = self.encoder_com(h_st)
        c_sm = self.encoder_com(h_sm)

        dec_loss = self.compute_decoupling_loss(s_st, c_st) + self.compute_decoupling_loss(s_sm, c_sm)
        hete_loss = self.compute_hetero_loss(s_st, s_sm)

        st_token = self.private_st_token(s_st)
        sm_token = self.private_sm_token(s_sm)
        common_st_token = self.common_token(c_st)
        common_sm_token = self.common_token(c_sm)
        homo_loss = self.compute_homo_loss(common_st_token, common_sm_token)

        hetero_tokens = torch.stack([st_token, sm_token], dim=0)
        trans_out = self.transformer_fusion(hetero_tokens)
        if isinstance(trans_out, tuple):
            trans_out = trans_out[0]
        fusion_rep_trans = torch.cat([trans_out[0], trans_out[1]], dim=1)
        st_cross = self.trans_st_with_sm(st_token.unsqueeze(0), sm_token.unsqueeze(0), sm_token.unsqueeze(0))
        sm_cross = self.trans_sm_with_st(sm_token.unsqueeze(0), st_token.unsqueeze(0), st_token.unsqueeze(0))
        if isinstance(st_cross, tuple):
            st_cross = st_cross[0]
        if isinstance(sm_cross, tuple):
            sm_cross = sm_cross[0]
        fusion_rep_cma = self.cma_proj(torch.cat([st_cross[0], sm_cross[0]], dim=1))

        fusion_rep_hete = fusion_rep_trans + fusion_rep_cma

        fusion_rep_homo = torch.cat([common_st_token, common_sm_token], dim=1)
        final_rep = torch.cat([fusion_rep_hete, fusion_rep_homo], dim=1)

        q_mu = self.q_mu_fc(final_rep)
        q_logvar = self.q_logvar_fc(final_rep).clamp(min=-8.0, max=8.0)
        q_var = torch.exp(q_logvar) + 1e-4
        z = Normal(q_mu, q_var.sqrt()).rsample()

        contribution_st = torch.sigmoid(
            self.modality_gate(torch.cat([common_st_token, common_sm_token], dim=1))
        ).squeeze(1)

        return {
            "X_st": X_st,
            "X_sm": X_sm,
            "q_mu": q_mu,
            "q_var": q_var,
            "z": z,
            "dec_loss": dec_loss,
            "hete_loss": hete_loss,
            "homo_loss": homo_loss,
            "contribution_st": contribution_st,
        }

    def decode(self, H: dict[str, torch.Tensor], lib_size: torch.Tensor):
        hidden = self.decoder(H["z"])
        if self.feature_input_mode:
            raw_mu_logits = self.px_rna_scale_decoder(hidden)
            px_rna_scale_normalized = raw_mu_logits
            px_rna_scale = raw_mu_logits
            px_rna_log_mu = torch.log(px_rna_scale)
        else:
            raw_mu_logits = self.px_rna_scale_decoder[0](hidden)
            log_library_size = torch.log(lib_size).unsqueeze(1)
            px_rna_log_mu = F.log_softmax(raw_mu_logits, dim=-1) + log_library_size
            px_rna_scale = torch.exp(px_rna_log_mu)
            px_rna_scale_normalized = torch.exp(px_rna_log_mu - log_library_size)
        px_rna_rate = self.px_rna_rate_decoder(hidden)
        px_rna_dropout = self.px_rna_dropout_decoder(hidden)
        px_sm_scale = self.px_sm_scale_decoder(hidden)
        px_sm_rate = self.px_sm_rate_decoder(hidden)
        px_sm_dropout = self.px_sm_dropout_decoder(hidden)
        return {
            "raw_mu_logits": raw_mu_logits,
            "px_rna_scale_normalized": px_rna_scale_normalized,
            "px_rna_log_mu": px_rna_log_mu,
            "px_rna_scale": px_rna_scale,
            "px_rna_rate": px_rna_rate,
            "px_rna_dropout": px_rna_dropout,
            "px_sm_scale": px_sm_scale,
            "px_sm_rate": px_sm_rate,
            "px_sm_dropout": px_sm_dropout,
        }

    def forward(self, X: torch.Tensor, reduction: str = "mean"):
        H = self.encode(X)
        kldiv_loss = kld(
            Normal(H["q_mu"], H["q_var"].sqrt()),
            Normal(torch.zeros_like(H["q_mu"]), torch.ones_like(H["q_var"])),
        ).sum(dim=1)
        kldiv_loss_shared = kld(
            Normal(H["q_mu_shared"], H["q_var"][:, : self.n_latent_shared].sqrt()),
            Normal(
                torch.zeros_like(H["q_mu_shared"]),
                torch.ones_like(H["q_mu_shared"]),
            ),
        ).sum(dim=1)
        kldiv_loss_st = kld(
            Normal(
                H["q_mu_st"],
                H["q_var"][:, self.n_latent_shared : self.n_latent_shared + self.n_latent_st].sqrt(),
            ),
            Normal(
                torch.zeros_like(H["q_mu_st"]),
                torch.ones_like(H["q_mu_st"]),
            ),
        ).sum(dim=1)
        kldiv_loss_sm = kld(
            Normal(H["q_mu_sm"], H["q_var"][:, self.n_latent_shared + self.n_latent_st :].sqrt()),
            Normal(
                torch.zeros_like(H["q_mu_sm"]),
                torch.ones_like(H["q_mu_sm"]),
            ),
        ).sum(dim=1)

        if self.feature_input_mode:
            lib_size = torch.ones(H["X_st"].shape[0], device=H["X_st"].device, dtype=H["X_st"].dtype)
        else:
            lib_size = H["X_st"].sum(1).clamp(min=1.0)
        R = self.decode(H, lib_size)

        if self.use_standardized_reconstruction:
            rec_st = self._reconstruction_mse_loss(
                self._standardize_st_features(R["px_rna_scale"]),
                self._standardize_st_features(H["X_st"]),
                reduction=reduction,
            )
            rec_sm = self._reconstruction_mse_loss(
                self._standardize_sm_prediction(R["px_sm_scale"]),
                self._standardize_sm_features(H["X_sm"]),
                reduction=reduction,
            )
        else:
            if self.reconstruction_method_st == "zinb":
                rec_st = LossFunction.zinb_reconstruction_loss(
                    H["X_st"],
                    log_mu=R["px_rna_log_mu"],
                    theta=R["px_rna_rate"].exp(),
                    gate_logits=R["px_rna_dropout"],
                    reduction=reduction,
                )
            elif self.reconstruction_method_st == "zg":
                rec_st = LossFunction.zi_gaussian_reconstruction_loss(
                    H["X_st"],
                    mean=R["px_rna_scale"],
                    variance=R["px_rna_rate"].exp(),
                    gate_logits=R["px_rna_dropout"],
                    reduction=reduction,
                )
            else:
                rec_st = F.mse_loss(R["px_rna_scale"], H["X_st"], reduction=reduction)

            if self.reconstruction_method_sm == "zg":
                rec_sm = LossFunction.zi_gaussian_reconstruction_loss(
                    H["X_sm"],
                    mean=R["px_sm_scale"],
                    variance=R["px_sm_rate"].exp(),
                    gate_logits=R["px_sm_dropout"],
                    reduction=reduction,
                )
            elif self.reconstruction_method_sm == "mse":
                rec_sm = F.mse_loss(
                    self._transform_sm_prediction(R["px_sm_scale"]),
                    H["X_sm_encoder"],
                    reduction=reduction,
                )
            else:
                rec_sm = LossFunction.gaussian_reconstruction_loss(
                    H["X_sm"],
                    mean=R["px_sm_scale"],
                    variance=R["px_sm_rate"].exp(),
                    reduction=reduction,
                )

        return H, R, {
            "reconstruction_loss_st": rec_st,
            "reconstruction_loss_sm": rec_sm,
            "kldiv_loss": kldiv_loss,
            "kldiv_loss_shared": kldiv_loss_shared,
            "kldiv_loss_st": kldiv_loss_st,
            "kldiv_loss_sm": kldiv_loss_sm,
            "dec_loss": H["dec_loss"],
            "hete_loss": H["hete_loss"],
            "homo_loss": H["homo_loss"],
        }

    def fit(
        self,
        max_epoch: int = 256,
        n_per_batch: int = 128,
        reconstruction_reduction: str = "mean",
        reconstruction_st_weight: float = 0.5,
        reconstruction_sm_weight: float = 0.5,
        dec_weight: float = 1.0,
        hete_weight: float = 0.05,
        homo_weight: float = 0.05,
        hete_warmup_epochs: int = 0,
        homo_warmup_epochs: int = 0,
        kl_weight: float = 0.0,
        n_epochs_kl_warmup: Optional[int] = 0,
        shared_kl_weight_scale: float = 1.0,
        private_kl_weight_scale: float = 1.0,
        late_kl_start_epoch: int = 0,
        late_kl_ramp_epochs: int = 0,
        late_shared_kl_weight_scale: Optional[float] = None,
        late_private_kl_weight_scale: Optional[float] = None,
        late_reconstruction_start_epoch: int = 0,
        late_reconstruction_ramp_epochs: int = 0,
        late_reconstruction_st_weight_scale: float = 1.0,
        late_reconstruction_sm_weight_scale: float = 1.0,
        weight_decay: float = 1e-6,
        lr: float = 5e-4,
        random_seed: int = 42,
        kl_loss_reduction: str = "mean",
    ):
        self.train()
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

        data_loader_generator = torch.Generator()
        data_loader_generator.manual_seed(random_seed)

        if n_epochs_kl_warmup:
            n_epochs_kl_warmup = min(max_epoch, n_epochs_kl_warmup)
            kl_warmup_gradient = kl_weight / max(n_epochs_kl_warmup, 1)
            kl_weight_max = kl_weight
            kl_weight = 0.0

        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        target_hete_weight = hete_weight
        target_homo_weight = homo_weight

        def ramp_weight(target: float, warmup_epochs: int, epoch_idx: int) -> float:
            if warmup_epochs <= 0:
                return target
            scale = min(max(epoch_idx, 0) / float(warmup_epochs), 1.0)
            return target * scale
        pbar = get_tqdm()(range(max_epoch), desc="Epoch", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        history = {
            "epoch_reconstruction_loss_st_list": [],
            "epoch_reconstruction_loss_sm_list": [],
            "epoch_kldiv_loss_list": [],
            "epoch_dec_loss_list": [],
            "epoch_hete_loss_list": [],
            "epoch_homo_loss_list": [],
            "epoch_total_loss_list": [],
        }
        ran_epochs = 0

        for epoch_idx in range(max_epoch):
            stats = {key: 0.0 for key in [
                "reconstruction_loss_st",
                "reconstruction_loss_sm",
                "kldiv_loss",
                "dec_loss",
                "hete_loss",
                "homo_loss",
                "total_loss",
            ]}
            n_batches = 0

            current_hete_weight = ramp_weight(target_hete_weight, hete_warmup_epochs, epoch_idx)
            current_homo_weight = ramp_weight(target_homo_weight, homo_warmup_epochs, epoch_idx)

            for batch_idx in self.as_dataloader(batch_size=n_per_batch, shuffle=True, generator=data_loader_generator):
                indices = batch_idx[0].cpu().numpy()
                X_batch = self._fetch_rows(indices)
                _, _, L = self.forward(X_batch, reduction=reconstruction_reduction)

                rec_st = L["reconstruction_loss_st"].mean()
                rec_sm = L["reconstruction_loss_sm"].mean()
                kl = L["kldiv_loss"].sum() / max(len(indices), 1) if kl_loss_reduction == "sum" else L["kldiv_loss"].mean()
                dec = L["dec_loss"]
                hete = L["hete_loss"]
                homo = L["homo_loss"]

                loss = (
                    reconstruction_st_weight * rec_st
                    + reconstruction_sm_weight * rec_sm
                    + kl_weight * kl
                    + dec_weight * dec
                    + current_hete_weight * hete
                    + current_homo_weight * homo
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                stats["reconstruction_loss_st"] += rec_st.item()
                stats["reconstruction_loss_sm"] += rec_sm.item()
                stats["kldiv_loss"] += kl.item()
                stats["dec_loss"] += dec.item()
                stats["hete_loss"] += hete.item()
                stats["homo_loss"] += homo.item()
                stats["total_loss"] += loss.item()
                n_batches += 1

            for key in stats:
                stats[key] /= max(n_batches, 1)

            pbar.set_postfix(
                {
                    "rec_st": f"{stats['reconstruction_loss_st']:.2e}",
                    "rec_sm": f"{stats['reconstruction_loss_sm']:.2e}",
                    "kl": f"{stats['kldiv_loss']:.2e}",
                    "dec": f"{stats['dec_loss']:.2e}",
                    "hete": f"{stats['hete_loss']:.2e}",
                    "homo": f"{stats['homo_loss']:.2e}",
                    "w_hete": f"{current_hete_weight:.2e}",
                    "w_homo": f"{current_homo_weight:.2e}",
                }
            )
            pbar.update(1)

            history["epoch_reconstruction_loss_st_list"].append(stats["reconstruction_loss_st"])
            history["epoch_reconstruction_loss_sm_list"].append(stats["reconstruction_loss_sm"])
            history["epoch_kldiv_loss_list"].append(stats["kldiv_loss"])
            history["epoch_dec_loss_list"].append(stats["dec_loss"])
            history["epoch_hete_loss_list"].append(stats["hete_loss"])
            history["epoch_homo_loss_list"].append(stats["homo_loss"])
            history["epoch_total_loss_list"].append(stats["total_loss"])

            if n_epochs_kl_warmup:
                kl_weight = min(kl_weight + kl_warmup_gradient, kl_weight_max)

            ran_epochs = epoch_idx + 1

        pbar.close()
        total_loss_history = history["epoch_total_loss_list"]
        if total_loss_history:
            min_total_loss = float(min(total_loss_history))
            min_total_loss_epoch = int(np.argmin(total_loss_history)) + 1
            final_total_loss = float(total_loss_history[-1])
        else:
            min_total_loss = float("nan")
            min_total_loss_epoch = 0
            final_total_loss = float("nan")

        self.fit_metadata = {
            "ran_epochs": int(ran_epochs),
            "min_total_loss_epoch": int(min_total_loss_epoch),
            "min_total_loss": float(min_total_loss),
            "final_total_loss": float(final_total_loss),
        }
        return history

    @torch.no_grad()
    def _iterate_full(self, n_per_batch: int = 128, latent_key: str = "q_mu"):
        self.eval()
        latents = []
        recon_st = []
        recon_sm = []
        contribution = []
        optional_keys = (
            "contribution_sm",
            "similarity_st_joint",
            "similarity_sm_joint",
            "homo_st_embedding",
            "homo_sm_embedding",
            "homo_joint_embedding",
        )
        extras: dict[str, list[np.ndarray]] = {key: [] for key in optional_keys}
        for batch_idx in self.as_dataloader(batch_size=n_per_batch, shuffle=False):
            indices = batch_idx[0].cpu().numpy()
            X_batch = self._fetch_rows(indices)
            H, R, _ = self.forward(X_batch, reduction="sum")
            latents.append(H[latent_key].detach().cpu().numpy())
            recon_st.append(R["px_rna_scale"].detach().cpu().numpy())
            recon_sm.append(R["px_sm_scale"].detach().cpu().numpy())
            contribution.append(H["contribution_st"].detach().cpu().numpy())
            for key in optional_keys:
                if key in H:
                    extras[key].append(H[key].detach().cpu().numpy())

        extra_outputs: dict[str, np.ndarray] = {}
        for key, values in extras.items():
            if not values:
                continue
            first = values[0]
            extra_outputs[key] = np.concatenate(values) if first.ndim == 1 else np.vstack(values)

        return (
            np.vstack(latents),
            np.vstack(recon_st),
            np.vstack(recon_sm),
            np.concatenate(contribution),
            extra_outputs,
        )

    @torch.no_grad()
    def get_latent_embedding(self, latent_key: Literal["q_mu", "z"] = "q_mu", n_per_batch: int = 128):
        latents, _, _, _, _ = self._iterate_full(n_per_batch=n_per_batch, latent_key=latent_key)
        return latents

    @torch.no_grad()
    def get_normalized_expression(self, n_per_batch: int = 128):
        _, recon_st, recon_sm, _, _ = self._iterate_full(n_per_batch=n_per_batch, latent_key="q_mu")
        output = np.zeros((self.n_obs, self.in_dim_st + self.in_dim_sm), dtype=np.float32)
        output[:, self.st_mask] = recon_st
        output[:, self.sm_mask] = recon_sm
        return output

    @torch.no_grad()
    def get_modality_contribution(self, n_per_batch: int = 128):
        _, _, _, contribution, _ = self._iterate_full(n_per_batch=n_per_batch, latent_key="q_mu")
        return contribution

    @torch.no_grad()
    def get_modality_contribution_details(self, n_per_batch: int = 128):
        _, _, _, _, extras = self._iterate_full(n_per_batch=n_per_batch, latent_key="q_mu")
        return extras


class DecAlignSpatialMetaLinear(BaseDecAlignSpatialMetaLinear):
    TASK_KEYS = ("shared", "reconstruction_st", "reconstruction_sm")
    SPATIAL_FOURIER_SCALES = (1.0, 2.0, 4.0)

    def _equal_task_weights(self) -> dict[str, float]:
        weight = 1.0 / len(self.TASK_KEYS)
        return {key: weight for key in self.TASK_KEYS}

    def _normalize_task_weights(
        self,
        weights: dict[str, float],
        task_weight_floor: float,
    ) -> dict[str, float]:
        raw = torch.tensor(
            [max(float(weights.get(key, 0.0)), 0.0) for key in self.TASK_KEYS],
            device=self.device,
            dtype=torch.float32,
        )
        if (not torch.isfinite(raw).all()) or raw.sum() <= 0:
            return self._equal_task_weights()

        raw = raw / raw.sum()
        floor = max(float(task_weight_floor), 0.0)
        floor = min(floor, (1.0 / len(self.TASK_KEYS)) - 1e-6)
        if floor > 0:
            raw = (1.0 - len(self.TASK_KEYS) * floor) * raw + floor

        return {key: float(raw[idx].item()) for idx, key in enumerate(self.TASK_KEYS)}

    def _flatten_task_gradient(self, loss: torch.Tensor) -> torch.Tensor:
        params = [param for param in self.parameters() if param.requires_grad]
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            if not params:
                return torch.zeros(1, device=self.device, dtype=torch.float32)
            return torch.cat([torch.zeros_like(param).reshape(-1) for param in params])
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        flat_grads = []
        for param, grad in zip(params, grads):
            if grad is None:
                flat_grads.append(torch.zeros_like(param).reshape(-1))
            else:
                flat_grads.append(grad.detach().reshape(-1))
        if not flat_grads:
            return torch.zeros(1, device=self.device, dtype=torch.float32)
        return torch.cat(flat_grads)

    def _compute_balanced_task_weights(
        self,
        task_losses: dict[str, torch.Tensor],
        task_weight_floor: float,
    ) -> dict[str, float]:
        grad_vectors = []
        non_zero_grad_found = False

        for key in self.TASK_KEYS:
            grad = self._flatten_task_gradient(task_losses[key])
            if not torch.isfinite(grad).all():
                return self._equal_task_weights()
            if torch.norm(grad, p=2).item() > 1e-12:
                non_zero_grad_found = True
            grad_vectors.append([grad])

        if not non_zero_grad_found:
            return self._equal_task_weights()

        try:
            solution, _ = MinNormSolver.find_min_norm_element(grad_vectors)
        except Exception:
            return self._equal_task_weights()

        raw_weights = {key: float(solution[idx].item()) for idx, key in enumerate(self.TASK_KEYS)}
        return self._normalize_task_weights(raw_weights, task_weight_floor=task_weight_floor)

    def __init__(
        self,
        *args,
        spatial_hidden_dim: int = 128,
        spatial_context_hidden_dim: int = 128,
        spatial_context_k: int = 12,
        spatial_encoder_mode: str = "local_context",
        spatial_fourier_scales: Optional[tuple[float, ...] | list[float]] = None,
        spatial_token_scale: float = 1.0,
        spatial_token_dropout: float = 0.0,
        spatial_contrastive_pos_k: int = 4,
        spatial_contrastive_neg_k: int = 16,
        spatial_contrastive_temperature: float = 0.2,
        spatial_contrastive_neg_strategy: str = "farthest",
        spatial_contrastive_mode: str = "positive_negative",
        spatial_negative_margin: float = 0.2,
        spatial_positive_weighting: str = "uniform",
        spatial_positive_aggregation: str = "shared_numerator",
        spatial_positive_weight_temperature: float = 1.0,
        decoder_hidden_dim: int = 256,
        decoder_num_layers: int = 1,
        **kwargs,
    ):
        self.spatial_hidden_dim = int(spatial_hidden_dim)
        self.spatial_context_hidden_dim = int(spatial_context_hidden_dim)
        self.spatial_context_k = max(int(spatial_context_k), 0)
        self.spatial_encoder_mode = str(spatial_encoder_mode).strip().lower()
        if spatial_fourier_scales is None:
            spatial_fourier_scales = self.SPATIAL_FOURIER_SCALES
        parsed_scales = tuple(float(scale) for scale in spatial_fourier_scales if float(scale) > 0)
        if not parsed_scales:
            raise ValueError("spatial_fourier_scales must contain at least one positive value.")
        self.spatial_fourier_scales = parsed_scales
        self.initial_spatial_token_scale = min(max(float(spatial_token_scale), 1e-4), 1.0 - 1e-4)
        self.spatial_token_dropout = min(max(float(spatial_token_dropout), 0.0), 0.95)
        # Keep contrastive positives aligned with SMART-style mutual KNN mining.
        self.spatial_contrastive_pos_k = min(max(int(spatial_contrastive_pos_k), 0), 3)
        self.spatial_contrastive_neg_k = max(int(spatial_contrastive_neg_k), 0)
        self.spatial_contrastive_temperature = max(float(spatial_contrastive_temperature), 1e-4)
        self.spatial_contrastive_neg_strategy = str(spatial_contrastive_neg_strategy).strip().lower()
        self.spatial_contrastive_mode = str(spatial_contrastive_mode).strip().lower()
        self.spatial_negative_margin = float(spatial_negative_margin)
        self.spatial_positive_weighting = str(spatial_positive_weighting).strip().lower()
        self.spatial_positive_aggregation = str(spatial_positive_aggregation).strip().lower()
        self.spatial_positive_weight_temperature = max(float(spatial_positive_weight_temperature), 1e-6)
        self.decoder_hidden_dim = max(int(decoder_hidden_dim), 1)
        self.decoder_num_layers = max(int(decoder_num_layers), 1)
        self.disable_spatial_contrastive = False
        if self.spatial_contrastive_neg_strategy not in {"farthest", "mid"}:
            raise ValueError(
                "spatial_contrastive_neg_strategy must be 'farthest' or 'mid'."
            )
        if self.spatial_contrastive_mode not in {"positive_negative", "negative_only"}:
            raise ValueError(
                "spatial_contrastive_mode must be 'positive_negative' or 'negative_only'."
            )
        if self.spatial_positive_weighting not in {"uniform", "feature_distance"}:
            raise ValueError(
                "spatial_positive_weighting must be 'uniform' or 'feature_distance'."
            )
        if self.spatial_positive_aggregation not in {"shared_numerator", "individual_loss"}:
            raise ValueError(
                "spatial_positive_aggregation must be 'shared_numerator' or 'individual_loss'."
            )
        if self.spatial_encoder_mode not in {"baseline", "local_context"}:
            raise ValueError(
                f"Unsupported spatial_encoder_mode={spatial_encoder_mode!r}; expected 'baseline' or 'local_context'."
            )

        super().__init__(*args, **kwargs)
        dropout_rate = float(self.dropout.p)

        del self.transformer_fusion
        del self.trans_st_with_sm
        del self.trans_sm_with_st
        del self.cma_proj
        del self.q_mu_fc
        del self.q_logvar_fc
        del self.decoder

        base_n_latent = int(self.n_latent)
        self.n_latent_shared = base_n_latent
        self.n_latent_st = max(1, base_n_latent // 2)
        self.n_latent_sm = max(1, base_n_latent // 2)
        self.n_latent = self.n_latent_shared + self.n_latent_st + self.n_latent_sm

        self.shared_resample = nn.Sequential(
            nn.LayerNorm(2 * self.token_dim),
            nn.Linear(2 * self.token_dim, 2 * self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.q_mu_shared_fc = nn.Linear(2 * self.token_dim, self.n_latent_shared)
        self.q_logvar_shared_fc = nn.Linear(2 * self.token_dim, self.n_latent_shared)
        self.q_mu_st_fc = nn.Linear(self.token_dim, self.n_latent_st)
        self.q_logvar_st_fc = nn.Linear(self.token_dim, self.n_latent_st)
        self.q_mu_sm_fc = nn.Linear(self.token_dim, self.n_latent_sm)
        self.q_logvar_sm_fc = nn.Linear(self.token_dim, self.n_latent_sm)

        self.decoder_st = LatentDecoderMLP(
            in_dim=self.n_latent_shared + self.n_latent_st,
            out_dim=self.proj_dim,
            hidden_dim=self.decoder_hidden_dim,
            num_layers=self.decoder_num_layers,
            dropout_rate=dropout_rate,
        )
        self.decoder_sm = LatentDecoderMLP(
            in_dim=self.n_latent_shared + self.n_latent_sm,
            out_dim=self.proj_dim,
            hidden_dim=self.decoder_hidden_dim,
            num_layers=self.decoder_num_layers,
            dropout_rate=dropout_rate,
        )
        self.to(self.device)

        dropout_rate = float(self.dropout.p)
        self.shared_resample = nn.Sequential(
            nn.LayerNorm(2 * self.token_dim),
            nn.Linear(2 * self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.q_mu_shared_fc = nn.Linear(self.token_dim, self.n_latent_shared)
        self.q_logvar_shared_fc = nn.Linear(self.token_dim, self.n_latent_shared)
        self.to(self.device)

        self.spatial_token_scale_logit = nn.Parameter(
            torch.tensor(
                np.log(self.initial_spatial_token_scale / (1.0 - self.initial_spatial_token_scale)),
                dtype=torch.float32,
            )
        )

        if "spatial" not in self.adata.obsm:
            raise ValueError("adata.obsm['spatial'] is required for the spatial-coordinate branch.")
        spatial = np.asarray(self.adata.obsm["spatial"], dtype=np.float32)
        if spatial.ndim != 2 or spatial.shape[1] < 2:
            raise ValueError(f"adata.obsm['spatial'] must be 2D with >=2 columns, got {spatial.shape}")

        coords = spatial[:, :2]
        self.register_buffer("spatial_coords", torch.tensor(coords, dtype=torch.float32))
        standardized_coords = np.asarray(self.adata.obsm["spadta_spatial_standardized"], dtype=np.float32)
        self.register_buffer("spatial_coords_standardized", torch.tensor(standardized_coords, dtype=torch.float32))

        self._init_spatial_context(coords=coords)

        dropout_rate = float(self.dropout.p)
        abs_feature_dim = 2 + 2 * 2 * len(self.spatial_fourier_scales)
        rel_feature_dim = 3 + 2 * 2 * len(self.spatial_fourier_scales)
        hidden_dim = max(self.spatial_hidden_dim, self.token_dim)
        context_hidden_dim = max(self.spatial_context_hidden_dim, self.token_dim)

        self.spatial_abs_token = nn.Sequential(
            nn.LayerNorm(abs_feature_dim),
            nn.Linear(abs_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self.spatial_local_encoder = nn.Sequential(
            nn.LayerNorm(rel_feature_dim),
            nn.Linear(rel_feature_dim, context_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(context_hidden_dim, self.token_dim),
        )
        self.spatial_local_pool = nn.Sequential(
            nn.LayerNorm(2 * self.token_dim),
            nn.Linear(2 * self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.spatial_token_fuse = nn.Sequential(
            nn.LayerNorm(2 * self.token_dim),
            nn.Linear(2 * self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.spatial_st_gate = nn.Sequential(
            nn.LayerNorm(2 * self.token_dim),
            nn.Linear(2 * self.token_dim, self.token_dim),
            nn.Sigmoid(),
        )
        self.spatial_sm_gate = nn.Sequential(
            nn.LayerNorm(2 * self.token_dim),
            nn.Linear(2 * self.token_dim, self.token_dim),
            nn.Sigmoid(),
        )
        self.spatial_joint_gate = nn.Sequential(
            nn.LayerNorm(3 * self.token_dim),
            nn.Linear(3 * self.token_dim, self.token_dim),
            nn.Sigmoid(),
        )
        self.shared_resample = nn.Sequential(
            nn.LayerNorm(2 * self.token_dim),
            nn.Linear(2 * self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.q_mu_shared_fc = nn.Linear(self.token_dim, self.n_latent_shared)
        self.q_logvar_shared_fc = nn.Linear(self.token_dim, self.n_latent_shared)
        self.to(self.device)

    def encode(self, X: torch.Tensor):
        if X.shape[0] != self.n_obs:
            raise ValueError("spatial-coordinate branch requires explicit batch indices for encode.")
        return self.encode_with_indices(X, indices=self.indices)

    def decode(self, H: dict[str, torch.Tensor], lib_size: torch.Tensor):
        z_st_for_decoder, z_sm_for_decoder, st_mask_stats, sm_mask_stats = self._prepare_private_latents_for_decoder(H)
        hidden_st = self.decoder_st(torch.cat([H["z_shared"], z_st_for_decoder], dim=1))
        hidden_sm = self.decoder_sm(torch.cat([H["z_shared"], z_sm_for_decoder], dim=1))
        if self.feature_input_mode:
            raw_mu_logits = self.px_rna_scale_decoder(hidden_st)
            px_rna_scale_normalized = raw_mu_logits
            px_rna_scale = raw_mu_logits
            px_rna_log_mu = torch.log(px_rna_scale)
        else:
            raw_mu_logits = self.px_rna_scale_decoder[0](hidden_st)
            log_library_size = torch.log(lib_size).unsqueeze(1)
            px_rna_log_mu = F.log_softmax(raw_mu_logits, dim=-1) + log_library_size
            px_rna_scale = torch.exp(px_rna_log_mu)
            px_rna_scale_normalized = torch.exp(px_rna_log_mu - log_library_size)
        px_rna_rate = self.px_rna_rate_decoder(hidden_st)
        px_rna_dropout = self.px_rna_dropout_decoder(hidden_st)
        px_sm_scale = self.px_sm_scale_decoder(hidden_sm)
        px_sm_rate = self.px_sm_rate_decoder(hidden_sm)
        px_sm_dropout = self.px_sm_dropout_decoder(hidden_sm)
        return {
            "raw_mu_logits": raw_mu_logits,
            "px_rna_scale_normalized": px_rna_scale_normalized,
            "px_rna_log_mu": px_rna_log_mu,
            "px_rna_scale": px_rna_scale,
            "px_rna_rate": px_rna_rate,
            "px_rna_dropout": px_rna_dropout,
            "px_sm_scale": px_sm_scale,
            "px_sm_rate": px_sm_rate,
            "px_sm_dropout": px_sm_dropout,
            "decoder_private_mask_probability_current": st_mask_stats["configured_mask_probability"],
            "decoder_st_private_actual_mask_fraction": st_mask_stats["actual_mask_fraction"],
            "decoder_sm_private_actual_mask_fraction": sm_mask_stats["actual_mask_fraction"],
            "decoder_st_private_masked_dimension_count": st_mask_stats["masked_dimension_count"],
            "decoder_sm_private_masked_dimension_count": sm_mask_stats["masked_dimension_count"],
            "decoder_st_private_kept_dimension_count": st_mask_stats["kept_dimension_count"],
            "decoder_sm_private_kept_dimension_count": sm_mask_stats["kept_dimension_count"],
        }

    def _compute_shared_cosine_contributions(
        self,
        common_st_token: torch.Tensor,
        common_sm_token: torch.Tensor,
        shared_rep: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if common_st_token.shape[1] != common_sm_token.shape[1]:
            raise ValueError(
                "common_st_token and common_sm_token must share the same feature width: "
                f"{common_st_token.shape[1]} vs {common_sm_token.shape[1]}"
            )
        if shared_rep.shape[1] != common_st_token.shape[1]:
            raise ValueError(
                "shared_rep must be compressed to the same width as each common token: "
                f"{shared_rep.shape[1]} vs {common_st_token.shape[1]}"
            )

        st_norm = F.normalize(common_st_token, p=2, dim=1, eps=1e-8)
        sm_norm = F.normalize(common_sm_token, p=2, dim=1, eps=1e-8)
        joint_norm = F.normalize(shared_rep, p=2, dim=1, eps=1e-8)

        similarity_st_joint = F.cosine_similarity(st_norm, joint_norm, dim=1)
        similarity_sm_joint = F.cosine_similarity(sm_norm, joint_norm, dim=1)
        clipped_similarity_st_joint = similarity_st_joint.clamp(min=-1.0, max=1.0)
        clipped_similarity_sm_joint = similarity_sm_joint.clamp(min=-1.0, max=1.0)

        angular_similarity_st_joint = 1.0 - (torch.arccos(clipped_similarity_st_joint) / torch.pi)
        angular_similarity_sm_joint = 1.0 - (torch.arccos(clipped_similarity_sm_joint) / torch.pi)
        contribution_st = angular_similarity_st_joint - angular_similarity_sm_joint + 0.5
        contribution_sm = 1.0 - contribution_st

        return {
            "homo_st_embedding": common_st_token,
            "homo_sm_embedding": common_sm_token,
            "homo_joint_embedding": shared_rep,
            "similarity_st_joint": similarity_st_joint,
            "similarity_sm_joint": similarity_sm_joint,
            "contribution_st": contribution_st,
            "contribution_sm": contribution_sm,
        }

    def _init_spatial_context(self, coords: np.ndarray) -> None:
        del coords
        neighbor_idx = np.asarray(self.adata.obsm["spadta_spatial_neighbor_idx"], dtype=np.int64)
        neighbor_rel = np.asarray(self.adata.obsm["spadta_spatial_neighbor_rel"], dtype=np.float32).reshape(
            self.n_obs, neighbor_idx.shape[1], 2
        )
        neighbor_dist = np.asarray(self.adata.obsm["spadta_spatial_neighbor_dist"], dtype=np.float32)[..., None]

        self.actual_spatial_context_k = int(neighbor_idx.shape[1])
        self.register_buffer("spatial_neighbor_idx", torch.tensor(neighbor_idx, dtype=torch.long))
        self.register_buffer("spatial_neighbor_rel", torch.tensor(neighbor_rel, dtype=torch.float32))
        self.register_buffer("spatial_neighbor_dist", torch.tensor(neighbor_dist, dtype=torch.float32))

    def _absolute_spatial_features_from_indices(self, indices: np.ndarray) -> torch.Tensor:
        indices_t = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        coords = self.spatial_coords_standardized[indices_t]
        features = [coords]
        for scale in self.spatial_fourier_scales:
            scaled = coords * float(scale)
            features.append(torch.sin(scaled))
            features.append(torch.cos(scaled))
        return torch.cat(features, dim=1)

    def _relative_spatial_features_from_indices(self, indices: np.ndarray) -> torch.Tensor:
        if self.actual_spatial_context_k == 0:
            return torch.zeros((len(indices), 0, 1), dtype=torch.float32, device=self.device)

        indices_t = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        rel = self.spatial_neighbor_rel[indices_t]
        dist = self.spatial_neighbor_dist[indices_t]
        features = [rel, dist]
        for scale in self.spatial_fourier_scales:
            scaled = rel * float(scale)
            features.append(torch.sin(scaled))
            features.append(torch.cos(scaled))
        return torch.cat(features, dim=2)

    def _build_spatial_branch(
        self,
        indices: np.ndarray,
        common_st_token: torch.Tensor,
        common_sm_token: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del indices
        zero_token = common_st_token.new_zeros(common_st_token.shape[0], self.token_dim)
        zero_scalar = common_st_token.new_zeros(common_st_token.shape[0])

        spatial_token = zero_token
        common_st_token_ctx = common_st_token
        common_sm_token_ctx = common_sm_token
        shared_rep = self.shared_resample(
            torch.cat([common_st_token_ctx, common_sm_token_ctx], dim=1)
        )

        return {
            "spatial_token": spatial_token,
            "shared_rep": shared_rep,
            "common_st_token_ctx": common_st_token_ctx,
            "common_sm_token_ctx": common_sm_token_ctx,
            "spatial_joint_gate_mean": zero_scalar,
            "spatial_st_gate_mean": zero_scalar,
            "spatial_sm_gate_mean": zero_scalar,
        }

    def encode_with_indices(self, X: torch.Tensor, indices: np.ndarray):
        X_st = X[:, self.st_mask]
        X_sm = X[:, self.sm_mask]
        encoder_X = self._fetch_encoder_rows(indices)
        h_st, h_sm = self.project_inputs(encoder_X[:, self.st_mask], encoder_X[:, self.sm_mask])
        s_st = self.encoder_uni_st(h_st)
        s_sm = self.encoder_uni_sm(h_sm)
        shared_outputs = self._encode_shared_with_fused_graphs(indices)
        c_st = shared_outputs["c_st"]
        c_sm = shared_outputs["c_sm"]

        dec_loss = self.compute_decoupling_loss(s_st, c_st) + self.compute_decoupling_loss(s_sm, c_sm)
        hete_loss = s_st.new_zeros(())

        st_token = self.private_st_token(s_st)
        sm_token = self.private_sm_token(s_sm)
        common_st_token = self.common_token(c_st)
        common_sm_token = self.common_token(c_sm)
        homo_loss = self.compute_homo_loss(common_st_token, common_sm_token)

        spatial_outputs = self._build_spatial_branch(
            indices=indices,
            common_st_token=common_st_token,
            common_sm_token=common_sm_token,
        )
        contribution_outputs = self._compute_shared_cosine_contributions(
            common_st_token=spatial_outputs["common_st_token_ctx"],
            common_sm_token=spatial_outputs["common_sm_token_ctx"],
            shared_rep=spatial_outputs["shared_rep"],
        )

        q_mu_shared = self.q_mu_shared_fc(spatial_outputs["shared_rep"])
        q_logvar_shared = self.q_logvar_shared_fc(spatial_outputs["shared_rep"]).clamp(min=-8.0, max=8.0)
        q_mu_st = self.q_mu_st_fc(st_token)
        q_logvar_st = self.q_logvar_st_fc(st_token).clamp(min=-8.0, max=8.0)
        q_mu_sm = self.q_mu_sm_fc(sm_token)
        q_logvar_sm = self.q_logvar_sm_fc(sm_token).clamp(min=-8.0, max=8.0)

        q_var_shared = torch.exp(q_logvar_shared) + 1e-4
        q_var_st = torch.exp(q_logvar_st) + 1e-4
        q_var_sm = torch.exp(q_logvar_sm) + 1e-4
        z_shared = Normal(q_mu_shared, q_var_shared.sqrt()).rsample()
        z_st = Normal(q_mu_st, q_var_st.sqrt()).rsample()
        z_sm = Normal(q_mu_sm, q_var_sm.sqrt()).rsample()

        q_mu = torch.cat([q_mu_shared, q_mu_st, q_mu_sm], dim=1)
        q_var = torch.cat([q_var_shared, q_var_st, q_var_sm], dim=1)
        z = torch.cat([z_shared, z_st, z_sm], dim=1)

        return {
            "X_st": X_st,
            "X_sm": X_sm,
            "X_sm_encoder": encoder_X[:, self.sm_mask],
            "q_mu": q_mu,
            "q_var": q_var,
            "q_mu_shared": q_mu_shared,
            "q_mu_st": q_mu_st,
            "q_mu_sm": q_mu_sm,
            "z": z,
            "z_shared": z_shared,
            "z_st": z_st,
            "z_sm": z_sm,
            "dec_loss": dec_loss,
            "hete_loss": hete_loss,
            "homo_loss": homo_loss,
            "contribution_st": contribution_outputs["contribution_st"],
            "contribution_sm": contribution_outputs["contribution_sm"],
            "similarity_st_joint": contribution_outputs["similarity_st_joint"],
            "similarity_sm_joint": contribution_outputs["similarity_sm_joint"],
            "homo_st_embedding": contribution_outputs["homo_st_embedding"],
            "homo_sm_embedding": contribution_outputs["homo_sm_embedding"],
            "homo_joint_embedding": contribution_outputs["homo_joint_embedding"],
            "spatial_token": spatial_outputs["spatial_token"],
            "spatial_gate_mean": spatial_outputs["spatial_joint_gate_mean"],
            "spatial_st_gate_mean": spatial_outputs["spatial_st_gate_mean"],
            "spatial_sm_gate_mean": spatial_outputs["spatial_sm_gate_mean"],
            "spatial_token_scale": torch.sigmoid(self.spatial_token_scale_logit).expand(X.shape[0]),
            "graph_fuse_st_weights": shared_outputs["weight_st"].unsqueeze(0).expand(X.shape[0], -1),
            "graph_fuse_sm_weights": shared_outputs["weight_sm"].unsqueeze(0).expand(X.shape[0], -1),
        }

    def forward_with_indices(
        self,
        X: torch.Tensor,
        indices: np.ndarray,
        reduction: str = "mean",
    ):
        H = self.encode_with_indices(X, indices=indices)
        kldiv_loss = kld(
            Normal(H["q_mu"], H["q_var"].sqrt()),
            Normal(torch.zeros_like(H["q_mu"]), torch.ones_like(H["q_var"])),
        ).sum(dim=1)
        kldiv_loss_shared = kld(
            Normal(H["q_mu_shared"], H["q_var"][:, : self.n_latent_shared].sqrt()),
            Normal(
                torch.zeros_like(H["q_mu_shared"]),
                torch.ones_like(H["q_mu_shared"]),
            ),
        ).sum(dim=1)
        kldiv_loss_st = kld(
            Normal(
                H["q_mu_st"],
                H["q_var"][:, self.n_latent_shared : self.n_latent_shared + self.n_latent_st].sqrt(),
            ),
            Normal(
                torch.zeros_like(H["q_mu_st"]),
                torch.ones_like(H["q_mu_st"]),
            ),
        ).sum(dim=1)
        kldiv_loss_sm = kld(
            Normal(H["q_mu_sm"], H["q_var"][:, self.n_latent_shared + self.n_latent_st :].sqrt()),
            Normal(
                torch.zeros_like(H["q_mu_sm"]),
                torch.ones_like(H["q_mu_sm"]),
            ),
        ).sum(dim=1)

        if self.feature_input_mode:
            lib_size = torch.ones(H["X_st"].shape[0], device=H["X_st"].device, dtype=H["X_st"].dtype)
        else:
            lib_size = H["X_st"].sum(1).clamp(min=1.0)
        R = self.decode(H, lib_size)

        if self.use_standardized_reconstruction:
            rec_st = self._reconstruction_mse_loss(
                self._standardize_st_features(R["px_rna_scale"]),
                self._standardize_st_features(H["X_st"]),
                reduction=reduction,
            )
            rec_sm = self._reconstruction_mse_loss(
                self._standardize_sm_prediction(R["px_sm_scale"]),
                self._standardize_sm_features(H["X_sm"]),
                reduction=reduction,
            )
        else:
            if self.reconstruction_method_st == "zinb":
                rec_st = LossFunction.zinb_reconstruction_loss(
                    H["X_st"],
                    log_mu=R["px_rna_log_mu"],
                    theta=R["px_rna_rate"].exp(),
                    gate_logits=R["px_rna_dropout"],
                    reduction=reduction,
                )
            elif self.reconstruction_method_st == "zg":
                rec_st = LossFunction.zi_gaussian_reconstruction_loss(
                    H["X_st"],
                    mean=R["px_rna_scale"],
                    variance=R["px_rna_rate"].exp(),
                    gate_logits=R["px_rna_dropout"],
                    reduction=reduction,
                )
            else:
                rec_st = F.mse_loss(R["px_rna_scale"], H["X_st"], reduction=reduction)

            if self.reconstruction_method_sm == "zg":
                rec_sm = LossFunction.zi_gaussian_reconstruction_loss(
                    H["X_sm"],
                    mean=R["px_sm_scale"],
                    variance=R["px_sm_rate"].exp(),
                    gate_logits=R["px_sm_dropout"],
                    reduction=reduction,
                )
            elif self.reconstruction_method_sm == "mse":
                rec_sm = F.mse_loss(
                    self._transform_sm_prediction(R["px_sm_scale"]),
                    H["X_sm_encoder"],
                    reduction=reduction,
                )
            else:
                rec_sm = LossFunction.gaussian_reconstruction_loss(
                    H["X_sm"],
                    mean=R["px_sm_scale"],
                    variance=R["px_sm_rate"].exp(),
                    reduction=reduction,
                )

        return H, R, {
            "reconstruction_loss_st": rec_st,
            "reconstruction_loss_sm": rec_sm,
            "kldiv_loss": kldiv_loss,
            "kldiv_loss_shared": kldiv_loss_shared,
            "kldiv_loss_st": kldiv_loss_st,
            "kldiv_loss_sm": kldiv_loss_sm,
            "dec_loss": H["dec_loss"],
            "hete_loss": H["hete_loss"],
            "homo_loss": H["homo_loss"],
        }

    def forward(self, X: torch.Tensor, reduction: str = "mean", indices: Optional[np.ndarray] = None):
        if indices is None:
            if X.shape[0] != self.n_obs:
                raise ValueError("spatial-coordinate branch requires explicit batch indices for mini-batch forward.")
            indices = self.indices
        return self.forward_with_indices(X, indices=np.asarray(indices), reduction=reduction)

    def compute_shared_latent_geometry_losses(
        self,
        latent_batch: torch.Tensor,
        std_target: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if latent_batch.ndim != 2:
            raise ValueError(f"Expected latent_batch to be 2D, got shape={tuple(latent_batch.shape)}")
        if latent_batch.shape[0] <= 1 or latent_batch.shape[1] <= 1:
            zero = latent_batch.sum() * 0.0
            return {
                "std_loss": zero,
                "cov_loss": zero,
                "std_mean": zero,
                "std_min": zero,
                "std_max": zero,
                "anisotropy_ratio": zero,
                "cov_offdiag_abs_mean": zero,
            }

        centered = latent_batch - latent_batch.mean(dim=0, keepdim=True)
        denom = max(int(latent_batch.shape[0]) - 1, 1)
        variance = centered.pow(2).sum(dim=0) / float(denom)
        std = torch.sqrt(variance + 1.0e-4)
        target = std.new_full(std.shape, float(std_target))
        std_loss = (std - target).pow(2).mean()

        normalized = centered / std.unsqueeze(0)
        covariance = normalized.T @ normalized / float(denom)
        offdiag = covariance - torch.diag_embed(torch.diagonal(covariance))
        offdiag_count = max(int(latent_batch.shape[1]) * (int(latent_batch.shape[1]) - 1), 1)
        cov_loss = offdiag.pow(2).sum() / float(offdiag_count)

        std_mean = std.mean()
        anisotropy_ratio = std.max() / std_mean.clamp_min(1.0e-6)
        cov_offdiag_abs_mean = offdiag.abs().sum() / float(offdiag_count)
        return {
            "std_loss": std_loss,
            "cov_loss": cov_loss,
            "std_mean": std_mean,
            "std_min": std.min(),
            "std_max": std.max(),
            "anisotropy_ratio": anisotropy_ratio,
            "cov_offdiag_abs_mean": cov_offdiag_abs_mean,
        }

    def compute_private_latent_ceiling_losses(
        self,
        shared_latent_batch: torch.Tensor,
        st_latent_batch: torch.Tensor,
        sm_latent_batch: torch.Tensor,
        ceiling_ratio: float = 0.9,
    ) -> dict[str, torch.Tensor]:
        if shared_latent_batch.ndim != 2:
            raise ValueError(
                f"Expected shared_latent_batch to be 2D, got shape={tuple(shared_latent_batch.shape)}"
            )
        zero = shared_latent_batch.sum() * 0.0
        if shared_latent_batch.shape[0] <= 1 or shared_latent_batch.shape[1] <= 0:
            return {
                "loss": zero,
                "shared_std_reference": zero,
                "private_st_std_mean": zero,
                "private_sm_std_mean": zero,
                "private_st_excess_fraction": zero,
                "private_sm_excess_fraction": zero,
            }

        def compute_std(latent_batch: torch.Tensor) -> torch.Tensor:
            centered = latent_batch - latent_batch.mean(dim=0, keepdim=True)
            denom = max(int(latent_batch.shape[0]) - 1, 1)
            variance = centered.pow(2).sum(dim=0) / float(denom)
            return torch.sqrt(variance + 1.0e-4)

        shared_std = compute_std(shared_latent_batch)
        shared_std_reference = shared_std.mean().detach()
        std_ceiling = shared_std_reference * max(float(ceiling_ratio), 0.0)

        private_losses: list[torch.Tensor] = []
        private_std_means: list[torch.Tensor] = []
        private_excess_fractions: list[torch.Tensor] = []

        for private_latent_batch in (st_latent_batch, sm_latent_batch):
            if private_latent_batch.ndim != 2 or private_latent_batch.shape[0] <= 1 or private_latent_batch.shape[1] <= 0:
                private_losses.append(zero)
                private_std_means.append(zero)
                private_excess_fractions.append(zero)
                continue
            private_std = compute_std(private_latent_batch)
            excess = F.relu(private_std - std_ceiling)
            private_losses.append(excess.pow(2).mean())
            private_std_means.append(private_std.mean())
            private_excess_fractions.append((private_std > std_ceiling).float().mean())

        stacked_losses = torch.stack(private_losses)
        loss = stacked_losses.mean() if stacked_losses.numel() > 0 else zero
        return {
            "loss": loss,
            "shared_std_reference": shared_std_reference,
            "private_st_std_mean": private_std_means[0],
            "private_sm_std_mean": private_std_means[1],
            "private_st_excess_fraction": private_excess_fractions[0],
            "private_sm_excess_fraction": private_excess_fractions[1],
        }

    def fit(
        self,
        max_epoch: int = 256,
        n_per_batch: int = 128,
        reconstruction_reduction: str = "mean",
        reconstruction_st_weight: float = 0.5,
        reconstruction_sm_weight: float = 0.5,
        dec_weight: float = 1.0,
        hete_weight: float = 0.05,
        homo_weight: float = 0.05,
        hete_warmup_epochs: int = 0,
        homo_warmup_epochs: int = 0,
        kl_weight: float = 0.0,
        n_epochs_kl_warmup: Optional[int] = 0,
        shared_kl_weight_scale: float = 1.0,
        private_kl_weight_scale: float = 1.0,
        late_kl_start_epoch: int = 0,
        late_kl_ramp_epochs: int = 0,
        late_shared_kl_weight_scale: Optional[float] = None,
        late_private_kl_weight_scale: Optional[float] = None,
        late_reconstruction_start_epoch: int = 0,
        late_reconstruction_ramp_epochs: int = 0,
        late_reconstruction_st_weight_scale: float = 1.0,
        late_reconstruction_sm_weight_scale: float = 1.0,
        weight_decay: float = 1e-6,
        lr: float = 5e-4,
        random_seed: int = 42,
        kl_loss_reduction: str = "mean",
        balance_start_epoch: int = 16,
        balance_ema: float = 0.8,
        task_weight_floor: float = 0.05,
        spatial_consistency_weight: float = 0.0,
        spatial_consistency_warmup_epochs: int = 16,
        spatial_consistency_use_all_latent: bool = False,
        spatial_contrastive_weight: float = 0.02,
        spatial_contrastive_warmup_epochs: int = 16,
        spatial_contrastive_stop_epoch: int = 0,
        spatial_contrastive_use_all_latent: bool = False,
        spatial_contrastive_latent_mode: str = "auto",
        spatial_negative_margin_weight: float = 0.0,
        spatial_negative_margin_warmup_epochs: int = 16,
        spatial_negative_margin_stop_epoch: int = 0,
        spatial_negative_margin_decay_epochs: int = 0,
        shared_latent_std_weight: float = 0.0,
        shared_latent_cov_weight: float = 0.0,
        shared_latent_geometry_warmup_epochs: int = 16,
        shared_latent_std_target: float = 1.0,
        private_latent_ceiling_weight: float = 0.0,
        private_latent_ceiling_ratio: float = 0.9,
        private_latent_ceiling_start_epoch: int = 0,
        private_latent_ceiling_ramp_epochs: int = 0,
        decoder_private_feature_masking: Optional[bool] = None,
        decoder_private_mask_probability: Optional[float] = None,
        decoder_private_mask_warmup_start: Optional[int] = None,
        decoder_private_mask_warmup_end: Optional[int] = None,
        spatial_contrastive_early_stop_enabled: bool = True,
        spatial_contrastive_early_stop_window_epochs: int = 70,
        spatial_contrastive_early_stop_slope_threshold: float = 1.0e-4,
        spatial_contrastive_early_stop_min_epoch: int = 400,
        spatial_contrastive_early_stop_patience: int = 20,
        epoch_end_callback: Optional[Callable[[int, dict[str, float], dict[str, list[float]]], None]] = None,
    ):
        self.train()
        contrastive_latent_mode = str(spatial_contrastive_latent_mode).strip().lower()
        if contrastive_latent_mode == "auto":
            contrastive_latent_mode = "raw_full" if spatial_contrastive_use_all_latent else "shared"
        if contrastive_latent_mode not in {"shared", "raw_full", "branch_scaled_full"}:
            raise ValueError(
                "spatial_contrastive_latent_mode must be 'auto', 'shared', "
                "'raw_full', or 'branch_scaled_full'"
            )
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

        data_loader_generator = torch.Generator()
        data_loader_generator.manual_seed(random_seed)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        target_hete_weight = hete_weight
        target_homo_weight = homo_weight
        target_spatial_consistency_weight = float(spatial_consistency_weight)
        target_spatial_contrastive_weight = 0.0 if self.disable_spatial_contrastive else float(spatial_contrastive_weight)
        spatial_contrastive_stop_epoch = max(int(spatial_contrastive_stop_epoch), 0)
        target_spatial_negative_margin_weight = (
            0.0 if self.disable_spatial_contrastive else max(float(spatial_negative_margin_weight), 0.0)
        )
        spatial_negative_margin_warmup_epochs = max(int(spatial_negative_margin_warmup_epochs), 0)
        spatial_negative_margin_stop_epoch = max(int(spatial_negative_margin_stop_epoch), 0)
        spatial_negative_margin_decay_epochs = max(int(spatial_negative_margin_decay_epochs), 0)
        target_kl_weight = float(kl_weight)
        target_shared_kl_weight_scale = float(shared_kl_weight_scale)
        target_private_kl_weight_scale = float(private_kl_weight_scale)
        late_kl_start_epoch = max(int(late_kl_start_epoch), 0)
        late_kl_ramp_epochs = max(int(late_kl_ramp_epochs), 0)
        target_late_shared_kl_weight_scale = (
            target_shared_kl_weight_scale
            if late_shared_kl_weight_scale is None
            else float(late_shared_kl_weight_scale)
        )
        target_late_private_kl_weight_scale = (
            target_private_kl_weight_scale
            if late_private_kl_weight_scale is None
            else float(late_private_kl_weight_scale)
        )
        late_reconstruction_start_epoch = max(int(late_reconstruction_start_epoch), 0)
        late_reconstruction_ramp_epochs = max(int(late_reconstruction_ramp_epochs), 0)
        target_late_reconstruction_st_weight_scale = max(float(late_reconstruction_st_weight_scale), 0.0)
        target_late_reconstruction_sm_weight_scale = max(float(late_reconstruction_sm_weight_scale), 0.0)
        target_shared_latent_std_weight = max(float(shared_latent_std_weight), 0.0)
        target_shared_latent_cov_weight = max(float(shared_latent_cov_weight), 0.0)
        shared_latent_geometry_warmup_epochs = max(int(shared_latent_geometry_warmup_epochs), 0)
        target_shared_latent_std_target = max(float(shared_latent_std_target), 1.0e-4)
        target_private_latent_ceiling_weight = max(float(private_latent_ceiling_weight), 0.0)
        target_private_latent_ceiling_ratio = max(float(private_latent_ceiling_ratio), 0.0)
        private_latent_ceiling_start_epoch = max(int(private_latent_ceiling_start_epoch), 0)
        private_latent_ceiling_ramp_epochs = max(int(private_latent_ceiling_ramp_epochs), 0)
        kl_warmup_epochs = max(int(n_epochs_kl_warmup or 0), 0)
        target_decoder_private_feature_masking = (
            self.decoder_private_feature_masking
            if decoder_private_feature_masking is None
            else bool(decoder_private_feature_masking)
        )
        target_decoder_private_mask_probability = (
            self.decoder_private_mask_probability
            if decoder_private_mask_probability is None
            else float(decoder_private_mask_probability)
        )
        target_decoder_private_mask_warmup_start = (
            self.decoder_private_mask_warmup_start
            if decoder_private_mask_warmup_start is None
            else int(decoder_private_mask_warmup_start)
        )
        target_decoder_private_mask_warmup_end = (
            self.decoder_private_mask_warmup_end
            if decoder_private_mask_warmup_end is None
            else int(decoder_private_mask_warmup_end)
        )
        early_stop_enabled = bool(spatial_contrastive_early_stop_enabled)
        early_stop_window_epochs = max(int(spatial_contrastive_early_stop_window_epochs), 2)
        early_stop_slope_threshold = max(float(spatial_contrastive_early_stop_slope_threshold), 0.0)
        early_stop_min_epoch = max(int(spatial_contrastive_early_stop_min_epoch), 1)
        early_stop_patience = max(int(spatial_contrastive_early_stop_patience), 1)
        early_stop_triggered = False
        early_stop_epoch = 0
        early_stop_consecutive_hits = 0
        prev_task_weights = self._equal_task_weights()
        task_scale = float(len(self.TASK_KEYS))

        def ramp_weight(target: float, warmup_epochs: int, epoch_idx: int) -> float:
            if warmup_epochs <= 0:
                return target
            scale = min(max(epoch_idx, 0) / float(warmup_epochs), 1.0)
            return target * scale

        def ramp_after_epoch(
            base: float,
            target: float,
            start_epoch: int,
            ramp_epochs: int,
            epoch_idx: int,
        ) -> float:
            if start_epoch <= 0 or target == base:
                return base
            current_epoch = int(epoch_idx) + 1
            if current_epoch < start_epoch:
                return base
            if ramp_epochs <= 0:
                return target
            progress = min(max((current_epoch - start_epoch + 1) / float(ramp_epochs), 0.0), 1.0)
            return base + (target - base) * progress

        def ramp_from_zero(
            target: float,
            start_epoch: int,
            ramp_epochs: int,
            epoch_idx: int,
        ) -> float:
            if target <= 0.0:
                return 0.0
            current_epoch = int(epoch_idx) + 1
            effective_start_epoch = max(int(start_epoch), 1)
            if current_epoch < effective_start_epoch:
                return 0.0
            if ramp_epochs <= 0:
                return target
            progress = min(max((current_epoch - effective_start_epoch + 1) / float(ramp_epochs), 0.0), 1.0)
            return target * progress

        pbar = get_tqdm()(range(max_epoch), desc="Epoch", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        history = {
            "epoch_reconstruction_loss_st_list": [],
            "epoch_reconstruction_loss_sm_list": [],
            "epoch_kldiv_loss_list": [],
            "epoch_dec_loss_list": [],
            "epoch_hete_loss_list": [],
            "epoch_homo_loss_list": [],
            "epoch_task_loss_shared_list": [],
            "epoch_task_loss_reconstruction_st_list": [],
            "epoch_task_loss_reconstruction_sm_list": [],
            "epoch_task_weight_shared_list": [],
            "epoch_task_weight_reconstruction_st_list": [],
            "epoch_task_weight_reconstruction_sm_list": [],
            "epoch_spatial_consistency_loss_list": [],
            "epoch_spatial_contrastive_loss_list": [],
            "epoch_spatial_negative_margin_loss_list": [],
            "epoch_private_latent_ceiling_loss_list": [],
            "epoch_positive_count_mean_list": [],
            "epoch_positive_weight_sum_mean_list": [],
            "epoch_positive_weight_mean_list": [],
            "epoch_positive_weight_min_list": [],
            "epoch_positive_weight_max_list": [],
            "epoch_rank1_weight_mean_list": [],
            "epoch_rank2_weight_mean_list": [],
            "epoch_rank3_weight_mean_list": [],
            "epoch_weighted_positive_distance_list": [],
            "epoch_unweighted_positive_distance_list": [],
            "epoch_total_loss_list": [],
        }
        ran_epochs = 0

        for epoch_idx in range(max_epoch):
            stats = {
                key: 0.0
                for key in [
                    "reconstruction_loss_st",
                    "reconstruction_loss_sm",
                    "kldiv_loss",
                    "dec_loss",
                    "hete_loss",
                    "homo_loss",
                    "task_loss_shared",
                    "task_loss_reconstruction_st",
                    "task_loss_reconstruction_sm",
                    "task_weight_shared",
                    "task_weight_reconstruction_st",
                    "task_weight_reconstruction_sm",
                    "spatial_consistency_loss",
                    "spatial_contrastive_loss",
                    "current_spatial_contrastive_weight",
                    "spatial_negative_margin_loss",
                    "weighted_spatial_negative_margin_term",
                    "current_spatial_negative_margin_weight",
                    "negative_mean_cosine",
                    "negative_max_cosine",
                    "negative_violation_rate",
                    "effective_negative_pairs",
                    "positive_count_mean",
                    "positive_weight_sum_mean",
                    "positive_weight_mean",
                    "positive_weight_min",
                    "positive_weight_max",
                    "rank1_weight_mean",
                    "rank2_weight_mean",
                    "rank3_weight_mean",
                    "weighted_positive_distance",
                    "unweighted_positive_distance",
                    "decoder_private_mask_probability_current",
                    "decoder_st_private_actual_mask_fraction",
                    "decoder_sm_private_actual_mask_fraction",
                    "decoder_st_private_masked_dimensions_mean",
                    "decoder_sm_private_masked_dimensions_mean",
                    "decoder_st_private_kept_dimensions_mean",
                    "decoder_sm_private_kept_dimensions_mean",
                    "spatial_contrastive_early_stop_recent_slope",
                    "spatial_contrastive_early_stop_recent_abs_slope",
                    "spatial_contrastive_early_stop_recent_mean",
                    "spatial_contrastive_early_stop_window_epochs",
                    "spatial_contrastive_early_stop_slope_threshold",
                    "spatial_contrastive_early_stop_min_epoch",
                    "spatial_contrastive_early_stop_patience",
                    "spatial_contrastive_early_stop_consecutive_hits",
                    "spatial_contrastive_early_stop_triggered",
                    "current_kl_weight",
                    "current_shared_kl_weight_scale",
                    "current_private_kl_weight_scale",
                    "current_reconstruction_st_weight_scale",
                    "current_reconstruction_sm_weight_scale",
                    "effective_reconstruction_st_weight",
                    "effective_reconstruction_sm_weight",
                    "kldiv_loss_shared",
                    "kldiv_loss_st",
                    "kldiv_loss_sm",
                    "weighted_kl_shared_term",
                    "weighted_kl_private_term",
                    "shared_latent_std_loss",
                    "shared_latent_cov_loss",
                    "weighted_shared_latent_geometry_term",
                    "shared_latent_std_mean",
                    "shared_latent_std_min",
                    "shared_latent_std_max",
                    "shared_latent_anisotropy_ratio",
                    "shared_latent_cov_offdiag_abs_mean",
                    "private_latent_ceiling_loss",
                    "weighted_private_latent_ceiling_term",
                    "current_private_latent_ceiling_weight",
                    "private_latent_shared_std_reference",
                    "private_st_latent_std_mean",
                    "private_sm_latent_std_mean",
                    "private_st_latent_excess_fraction",
                    "private_sm_latent_excess_fraction",
                    "total_loss",
                ]
            }
            n_batches = 0

            current_hete_weight = ramp_weight(target_hete_weight, hete_warmup_epochs, epoch_idx)
            current_homo_weight = ramp_weight(target_homo_weight, homo_warmup_epochs, epoch_idx)
            current_spatial_consistency_weight = ramp_weight(
                target_spatial_consistency_weight,
                spatial_consistency_warmup_epochs,
                epoch_idx,
            )
            current_spatial_contrastive_weight = ramp_weight(
                target_spatial_contrastive_weight,
                spatial_contrastive_warmup_epochs,
                epoch_idx,
            )
            if spatial_contrastive_stop_epoch > 0 and (epoch_idx + 1) > spatial_contrastive_stop_epoch:
                current_spatial_contrastive_weight = 0.0
            current_spatial_negative_margin_weight = ramp_weight(
                target_spatial_negative_margin_weight,
                spatial_negative_margin_warmup_epochs,
                epoch_idx,
            )
            if spatial_negative_margin_stop_epoch > 0 and (epoch_idx + 1) > spatial_negative_margin_stop_epoch:
                if spatial_negative_margin_decay_epochs > 0:
                    decay_progress = min(
                        max(
                            (epoch_idx + 1 - spatial_negative_margin_stop_epoch)
                            / float(spatial_negative_margin_decay_epochs),
                            0.0,
                        ),
                        1.0,
                    )
                    current_spatial_negative_margin_weight *= max(1.0 - decay_progress, 0.0)
                else:
                    current_spatial_negative_margin_weight = 0.0
            current_shared_latent_std_weight = ramp_weight(
                target_shared_latent_std_weight,
                shared_latent_geometry_warmup_epochs,
                epoch_idx,
            )
            current_shared_latent_cov_weight = ramp_weight(
                target_shared_latent_cov_weight,
                shared_latent_geometry_warmup_epochs,
                epoch_idx,
            )
            current_private_latent_ceiling_weight = ramp_from_zero(
                target_private_latent_ceiling_weight,
                private_latent_ceiling_start_epoch,
                private_latent_ceiling_ramp_epochs,
                epoch_idx,
            )
            current_kl_weight = ramp_weight(target_kl_weight, kl_warmup_epochs, epoch_idx)
            current_shared_kl_weight_scale = ramp_after_epoch(
                target_shared_kl_weight_scale,
                target_late_shared_kl_weight_scale,
                late_kl_start_epoch,
                late_kl_ramp_epochs,
                epoch_idx,
            )
            current_private_kl_weight_scale = ramp_after_epoch(
                target_private_kl_weight_scale,
                target_late_private_kl_weight_scale,
                late_kl_start_epoch,
                late_kl_ramp_epochs,
                epoch_idx,
            )
            current_reconstruction_st_weight_scale = ramp_after_epoch(
                1.0,
                target_late_reconstruction_st_weight_scale,
                late_reconstruction_start_epoch,
                late_reconstruction_ramp_epochs,
                epoch_idx,
            )
            current_reconstruction_sm_weight_scale = ramp_after_epoch(
                1.0,
                target_late_reconstruction_sm_weight_scale,
                late_reconstruction_start_epoch,
                late_reconstruction_ramp_epochs,
                epoch_idx,
            )
            effective_reconstruction_st_weight = reconstruction_st_weight * current_reconstruction_st_weight_scale
            effective_reconstruction_sm_weight = reconstruction_sm_weight * current_reconstruction_sm_weight_scale
            current_decoder_private_mask_probability = get_decoder_private_mask_probability(
                epoch=epoch_idx,
                enabled=target_decoder_private_feature_masking,
                target_probability=target_decoder_private_mask_probability,
                warmup_start=target_decoder_private_mask_warmup_start,
                warmup_end=target_decoder_private_mask_warmup_end,
            )
            self._decoder_private_mask_probability_current = float(current_decoder_private_mask_probability)

            for batch_idx in self.as_dataloader(batch_size=n_per_batch, shuffle=True, generator=data_loader_generator):
                indices = batch_idx[0].cpu().numpy()
                X_batch = self._fetch_rows(indices)
                H, R, losses = self.forward_with_indices(X_batch, indices=indices, reduction=reconstruction_reduction)

                rec_st = losses["reconstruction_loss_st"].mean()
                rec_sm = losses["reconstruction_loss_sm"].mean()
                kl = (
                    losses["kldiv_loss"].sum() / max(len(indices), 1)
                    if kl_loss_reduction == "sum"
                    else losses["kldiv_loss"].mean()
                )
                kl_shared = (
                    losses["kldiv_loss_shared"].sum() / max(len(indices), 1)
                    if kl_loss_reduction == "sum"
                    else losses["kldiv_loss_shared"].mean()
                )
                kl_st = (
                    losses["kldiv_loss_st"].sum() / max(len(indices), 1)
                    if kl_loss_reduction == "sum"
                    else losses["kldiv_loss_st"].mean()
                )
                kl_sm = (
                    losses["kldiv_loss_sm"].sum() / max(len(indices), 1)
                    if kl_loss_reduction == "sum"
                    else losses["kldiv_loss_sm"].mean()
                )
                weighted_kl = (
                    current_shared_kl_weight_scale * kl_shared
                    + current_private_kl_weight_scale * (kl_st + kl_sm)
                )
                dec = losses["dec_loss"]
                hete = losses["hete_loss"]
                homo = losses["homo_loss"]

                task_losses = {
                    "shared": dec_weight * dec + current_hete_weight * hete + current_homo_weight * homo,
                    "reconstruction_st": effective_reconstruction_st_weight * rec_st,
                    "reconstruction_sm": effective_reconstruction_sm_weight * rec_sm,
                }

                if epoch_idx < balance_start_epoch:
                    task_weights = dict(prev_task_weights)
                else:
                    raw_task_weights = self._compute_balanced_task_weights(
                        task_losses,
                        task_weight_floor=task_weight_floor,
                    )
                    if balance_ema > 0:
                        blended = {
                            key: balance_ema * prev_task_weights[key] + (1.0 - balance_ema) * raw_task_weights[key]
                            for key in self.TASK_KEYS
                        }
                        task_weights = self._normalize_task_weights(blended, task_weight_floor=task_weight_floor)
                    else:
                        task_weights = raw_task_weights
                    prev_task_weights = dict(task_weights)

                latent_for_consistency = H["q_mu"] if spatial_consistency_use_all_latent else H["q_mu_shared"]
                if contrastive_latent_mode == "shared":
                    latent_for_contrastive = H["q_mu_shared"]
                elif contrastive_latent_mode == "raw_full":
                    latent_for_contrastive = H["q_mu"]
                else:
                    latent_for_contrastive = torch.cat(
                        [
                            normalize_branch_mean_variance_torch(H["q_mu_shared"]),
                            normalize_branch_mean_variance_torch(H["q_mu_st"]),
                            normalize_branch_mean_variance_torch(H["q_mu_sm"]),
                        ],
                        dim=1,
                    )
                if current_spatial_consistency_weight > 0.0:
                    spatial_consistency_loss = self.compute_spatial_consistency_loss(
                        latent_batch=latent_for_consistency,
                        batch_indices=indices,
                    )
                else:
                    spatial_consistency_loss = latent_for_consistency.sum() * 0.0
                if current_spatial_contrastive_weight > 0.0 or current_spatial_negative_margin_weight > 0.0:
                    spatial_contrastive_loss = self.compute_spatial_contrastive_loss(
                        latent_batch=latent_for_contrastive,
                        batch_indices=indices,
                        return_details=True,
                    )
                    spatial_contrastive_loss, spatial_contrastive_details = spatial_contrastive_loss
                else:
                    spatial_contrastive_loss = latent_for_contrastive.sum() * 0.0
                    spatial_contrastive_details = self._zero_spatial_contrastive_details(latent_for_contrastive)
                if current_shared_latent_std_weight > 0.0 or current_shared_latent_cov_weight > 0.0:
                    shared_latent_geometry = self.compute_shared_latent_geometry_losses(
                        H["q_mu_shared"],
                        std_target=target_shared_latent_std_target,
                    )
                else:
                    zero = H["q_mu_shared"].sum() * 0.0
                    shared_latent_geometry = {
                        "std_loss": zero,
                        "cov_loss": zero,
                        "std_mean": zero,
                        "std_min": zero,
                        "std_max": zero,
                        "anisotropy_ratio": zero,
                        "cov_offdiag_abs_mean": zero,
                    }
                if current_private_latent_ceiling_weight > 0.0:
                    private_latent_ceiling = self.compute_private_latent_ceiling_losses(
                        H["q_mu_shared"],
                        H["q_mu_st"],
                        H["q_mu_sm"],
                        ceiling_ratio=target_private_latent_ceiling_ratio,
                    )
                else:
                    zero = H["q_mu_shared"].sum() * 0.0
                    private_latent_ceiling = {
                        "loss": zero,
                        "shared_std_reference": zero,
                        "private_st_std_mean": zero,
                        "private_sm_std_mean": zero,
                        "private_st_excess_fraction": zero,
                        "private_sm_excess_fraction": zero,
                    }
                weighted_shared_latent_geometry = (
                    current_shared_latent_std_weight * shared_latent_geometry["std_loss"]
                    + current_shared_latent_cov_weight * shared_latent_geometry["cov_loss"]
                )
                weighted_private_latent_ceiling = (
                    current_private_latent_ceiling_weight * private_latent_ceiling["loss"]
                )
                weighted_spatial_negative_margin = (
                    current_spatial_negative_margin_weight * spatial_contrastive_details["negative_margin_loss"]
                )
                loss = task_scale * sum(task_weights[key] * task_losses[key] for key in self.TASK_KEYS)
                loss = loss + current_spatial_consistency_weight * spatial_consistency_loss
                loss = loss + current_spatial_contrastive_weight * spatial_contrastive_loss
                loss = loss + weighted_spatial_negative_margin
                loss = loss + current_kl_weight * weighted_kl
                loss = loss + weighted_shared_latent_geometry
                loss = loss + weighted_private_latent_ceiling

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                stats["reconstruction_loss_st"] += rec_st.item()
                stats["reconstruction_loss_sm"] += rec_sm.item()
                stats["kldiv_loss"] += kl.item()
                stats["kldiv_loss_shared"] += kl_shared.item()
                stats["kldiv_loss_st"] += kl_st.item()
                stats["kldiv_loss_sm"] += kl_sm.item()
                stats["weighted_kl_shared_term"] += (current_kl_weight * current_shared_kl_weight_scale * kl_shared).item()
                stats["weighted_kl_private_term"] += (
                    current_kl_weight * current_private_kl_weight_scale * (kl_st + kl_sm)
                ).item()
                stats["current_shared_kl_weight_scale"] += current_shared_kl_weight_scale
                stats["current_private_kl_weight_scale"] += current_private_kl_weight_scale
                stats["current_reconstruction_st_weight_scale"] += current_reconstruction_st_weight_scale
                stats["current_reconstruction_sm_weight_scale"] += current_reconstruction_sm_weight_scale
                stats["effective_reconstruction_st_weight"] += effective_reconstruction_st_weight
                stats["effective_reconstruction_sm_weight"] += effective_reconstruction_sm_weight
                stats["shared_latent_std_loss"] += shared_latent_geometry["std_loss"].item()
                stats["shared_latent_cov_loss"] += shared_latent_geometry["cov_loss"].item()
                stats["weighted_shared_latent_geometry_term"] += weighted_shared_latent_geometry.item()
                stats["shared_latent_std_mean"] += shared_latent_geometry["std_mean"].item()
                stats["shared_latent_std_min"] += shared_latent_geometry["std_min"].item()
                stats["shared_latent_std_max"] += shared_latent_geometry["std_max"].item()
                stats["shared_latent_anisotropy_ratio"] += shared_latent_geometry["anisotropy_ratio"].item()
                stats["shared_latent_cov_offdiag_abs_mean"] += shared_latent_geometry["cov_offdiag_abs_mean"].item()
                stats["private_latent_ceiling_loss"] += private_latent_ceiling["loss"].item()
                stats["weighted_private_latent_ceiling_term"] += weighted_private_latent_ceiling.item()
                stats["current_private_latent_ceiling_weight"] += current_private_latent_ceiling_weight
                stats["private_latent_shared_std_reference"] += private_latent_ceiling["shared_std_reference"].item()
                stats["private_st_latent_std_mean"] += private_latent_ceiling["private_st_std_mean"].item()
                stats["private_sm_latent_std_mean"] += private_latent_ceiling["private_sm_std_mean"].item()
                stats["private_st_latent_excess_fraction"] += private_latent_ceiling["private_st_excess_fraction"].item()
                stats["private_sm_latent_excess_fraction"] += private_latent_ceiling["private_sm_excess_fraction"].item()
                stats["dec_loss"] += dec.item()
                stats["hete_loss"] += hete.item()
                stats["homo_loss"] += homo.item()
                stats["task_loss_shared"] += task_losses["shared"].item()
                stats["task_loss_reconstruction_st"] += task_losses["reconstruction_st"].item()
                stats["task_loss_reconstruction_sm"] += task_losses["reconstruction_sm"].item()
                stats["task_weight_shared"] += task_weights["shared"]
                stats["task_weight_reconstruction_st"] += task_weights["reconstruction_st"]
                stats["task_weight_reconstruction_sm"] += task_weights["reconstruction_sm"]
                stats["spatial_consistency_loss"] += spatial_consistency_loss.item()
                stats["spatial_contrastive_loss"] += spatial_contrastive_loss.item()
                stats["current_spatial_contrastive_weight"] += current_spatial_contrastive_weight
                stats["spatial_negative_margin_loss"] += spatial_contrastive_details["negative_margin_loss"].item()
                stats["weighted_spatial_negative_margin_term"] += weighted_spatial_negative_margin.item()
                stats["current_spatial_negative_margin_weight"] += current_spatial_negative_margin_weight
                stats["negative_mean_cosine"] += spatial_contrastive_details["negative_mean_cosine"].item()
                stats["negative_max_cosine"] += spatial_contrastive_details["negative_max_cosine"].item()
                stats["negative_violation_rate"] += spatial_contrastive_details["negative_violation_rate"].item()
                stats["effective_negative_pairs"] += spatial_contrastive_details["effective_negative_pairs"].item()
                stats["positive_count_mean"] += spatial_contrastive_details["positive_count_mean"].item()
                stats["positive_weight_sum_mean"] += spatial_contrastive_details["positive_weight_sum_mean"].item()
                stats["positive_weight_mean"] += spatial_contrastive_details["positive_weight_mean"].item()
                stats["positive_weight_min"] += spatial_contrastive_details["positive_weight_min"].item()
                stats["positive_weight_max"] += spatial_contrastive_details["positive_weight_max"].item()
                stats["rank1_weight_mean"] += spatial_contrastive_details["rank1_weight_mean"].item()
                stats["rank2_weight_mean"] += spatial_contrastive_details["rank2_weight_mean"].item()
                stats["rank3_weight_mean"] += spatial_contrastive_details["rank3_weight_mean"].item()
                stats["weighted_positive_distance"] += spatial_contrastive_details["weighted_positive_distance"].item()
                stats["unweighted_positive_distance"] += spatial_contrastive_details["unweighted_positive_distance"].item()
                stats["decoder_private_mask_probability_current"] += float(
                    R["decoder_private_mask_probability_current"].item()
                )
                stats["decoder_st_private_actual_mask_fraction"] += float(
                    R["decoder_st_private_actual_mask_fraction"].item()
                )
                stats["decoder_sm_private_actual_mask_fraction"] += float(
                    R["decoder_sm_private_actual_mask_fraction"].item()
                )
                stats["decoder_st_private_masked_dimensions_mean"] += float(
                    R["decoder_st_private_masked_dimension_count"].item()
                )
                stats["decoder_sm_private_masked_dimensions_mean"] += float(
                    R["decoder_sm_private_masked_dimension_count"].item()
                )
                stats["decoder_st_private_kept_dimensions_mean"] += float(
                    R["decoder_st_private_kept_dimension_count"].item()
                )
                stats["decoder_sm_private_kept_dimensions_mean"] += float(
                    R["decoder_sm_private_kept_dimension_count"].item()
                )
                stats["current_kl_weight"] += current_kl_weight
                stats["total_loss"] += loss.item()
                n_batches += 1

            for key in stats:
                stats[key] /= max(n_batches, 1)

            pbar.set_postfix(
                {
                    "rec_st": f"{stats['reconstruction_loss_st']:.2e}",
                    "rec_sm": f"{stats['reconstruction_loss_sm']:.2e}",
                    "shared": f"{stats['task_loss_shared']:.2e}",
                    "w_sh": f"{stats['task_weight_shared']:.2f}",
                    "w_st": f"{stats['task_weight_reconstruction_st']:.2f}",
                    "w_sm": f"{stats['task_weight_reconstruction_sm']:.2f}",
                }
            )
            pbar.update(1)

            history["epoch_reconstruction_loss_st_list"].append(stats["reconstruction_loss_st"])
            history["epoch_reconstruction_loss_sm_list"].append(stats["reconstruction_loss_sm"])
            history["epoch_kldiv_loss_list"].append(stats["kldiv_loss"])
            history["epoch_dec_loss_list"].append(stats["dec_loss"])
            history["epoch_hete_loss_list"].append(stats["hete_loss"])
            history["epoch_homo_loss_list"].append(stats["homo_loss"])
            history["epoch_task_loss_shared_list"].append(stats["task_loss_shared"])
            history["epoch_task_loss_reconstruction_st_list"].append(stats["task_loss_reconstruction_st"])
            history["epoch_task_loss_reconstruction_sm_list"].append(stats["task_loss_reconstruction_sm"])
            history["epoch_task_weight_shared_list"].append(stats["task_weight_shared"])
            history["epoch_task_weight_reconstruction_st_list"].append(stats["task_weight_reconstruction_st"])
            history["epoch_task_weight_reconstruction_sm_list"].append(stats["task_weight_reconstruction_sm"])
            history["epoch_spatial_consistency_loss_list"].append(stats["spatial_consistency_loss"])
            history["epoch_spatial_contrastive_loss_list"].append(stats["spatial_contrastive_loss"])
            history["epoch_spatial_negative_margin_loss_list"].append(stats["spatial_negative_margin_loss"])
            history["epoch_private_latent_ceiling_loss_list"].append(stats["private_latent_ceiling_loss"])
            history["epoch_positive_count_mean_list"].append(stats["positive_count_mean"])
            history["epoch_positive_weight_sum_mean_list"].append(stats["positive_weight_sum_mean"])
            history["epoch_positive_weight_mean_list"].append(stats["positive_weight_mean"])
            history["epoch_positive_weight_min_list"].append(stats["positive_weight_min"])
            history["epoch_positive_weight_max_list"].append(stats["positive_weight_max"])
            history["epoch_rank1_weight_mean_list"].append(stats["rank1_weight_mean"])
            history["epoch_rank2_weight_mean_list"].append(stats["rank2_weight_mean"])
            history["epoch_rank3_weight_mean_list"].append(stats["rank3_weight_mean"])
            history["epoch_weighted_positive_distance_list"].append(stats["weighted_positive_distance"])
            history["epoch_unweighted_positive_distance_list"].append(stats["unweighted_positive_distance"])
            history["epoch_total_loss_list"].append(stats["total_loss"])
            ran_epochs = epoch_idx + 1

            recent_slope, recent_abs_slope = compute_recent_linear_slope(
                history["epoch_spatial_contrastive_loss_list"],
                early_stop_window_epochs,
            )
            if np.isnan(recent_slope):
                recent_mean = float("nan")
            else:
                recent_values = history["epoch_spatial_contrastive_loss_list"][-early_stop_window_epochs:]
                recent_mean = float(np.mean(np.asarray(recent_values, dtype=np.float64)))

            hit_condition = (
                early_stop_enabled
                and ran_epochs >= early_stop_min_epoch
                and not np.isnan(recent_abs_slope)
                and recent_abs_slope <= early_stop_slope_threshold
            )
            if hit_condition:
                early_stop_consecutive_hits += 1
            else:
                early_stop_consecutive_hits = 0
            if hit_condition and early_stop_consecutive_hits >= early_stop_patience:
                early_stop_triggered = True
                early_stop_epoch = int(ran_epochs)

            stats["spatial_contrastive_early_stop_recent_slope"] = float(recent_slope)
            stats["spatial_contrastive_early_stop_recent_abs_slope"] = float(recent_abs_slope)
            stats["spatial_contrastive_early_stop_recent_mean"] = float(recent_mean)
            stats["spatial_contrastive_early_stop_window_epochs"] = float(early_stop_window_epochs)
            stats["spatial_contrastive_early_stop_slope_threshold"] = float(early_stop_slope_threshold)
            stats["spatial_contrastive_early_stop_min_epoch"] = float(early_stop_min_epoch)
            stats["spatial_contrastive_early_stop_patience"] = float(early_stop_patience)
            stats["spatial_contrastive_early_stop_consecutive_hits"] = float(early_stop_consecutive_hits)
            stats["spatial_contrastive_early_stop_triggered"] = float(early_stop_triggered)

            if epoch_end_callback is not None:
                epoch_end_callback(ran_epochs, dict(stats), history)
            if early_stop_triggered:
                break

        pbar.close()
        total_loss_history = history["epoch_total_loss_list"]
        if total_loss_history:
            min_total_loss = float(min(total_loss_history))
            min_total_loss_epoch = int(np.argmin(total_loss_history)) + 1
            final_total_loss = float(total_loss_history[-1])
        else:
            min_total_loss = float("nan")
            min_total_loss_epoch = 0
            final_total_loss = float("nan")

        self.fit_metadata = {
            "ran_epochs": int(ran_epochs),
            "min_total_loss_epoch": int(min_total_loss_epoch),
            "min_total_loss": float(min_total_loss),
            "final_total_loss": float(final_total_loss),
            "balance_start_epoch": int(balance_start_epoch),
            "balance_ema": float(balance_ema),
            "task_weight_floor": float(task_weight_floor),
            "kl_used": bool(kl_weight > 0),
            "shared_kl_weight_scale": float(target_shared_kl_weight_scale),
            "private_kl_weight_scale": float(target_private_kl_weight_scale),
            "late_kl_start_epoch": int(late_kl_start_epoch),
            "late_kl_ramp_epochs": int(late_kl_ramp_epochs),
            "late_shared_kl_weight_scale": float(target_late_shared_kl_weight_scale),
            "late_private_kl_weight_scale": float(target_late_private_kl_weight_scale),
            "late_reconstruction_start_epoch": int(late_reconstruction_start_epoch),
            "late_reconstruction_ramp_epochs": int(late_reconstruction_ramp_epochs),
            "late_reconstruction_st_weight_scale": float(target_late_reconstruction_st_weight_scale),
            "late_reconstruction_sm_weight_scale": float(target_late_reconstruction_sm_weight_scale),
            "spatial_negative_margin_decay_epochs": int(spatial_negative_margin_decay_epochs),
            "shared_latent_std_weight": float(target_shared_latent_std_weight),
            "shared_latent_cov_weight": float(target_shared_latent_cov_weight),
            "shared_latent_geometry_warmup_epochs": int(shared_latent_geometry_warmup_epochs),
            "shared_latent_std_target": float(target_shared_latent_std_target),
            "private_latent_ceiling_weight": float(target_private_latent_ceiling_weight),
            "private_latent_ceiling_ratio": float(target_private_latent_ceiling_ratio),
            "private_latent_ceiling_start_epoch": int(private_latent_ceiling_start_epoch),
            "private_latent_ceiling_ramp_epochs": int(private_latent_ceiling_ramp_epochs),
            "decoder_hidden_dim": int(self.decoder_hidden_dim),
            "decoder_num_layers": int(self.decoder_num_layers),
            "spatial_branch": "disabled_no_spatial",
            "spatial_context_k": int(self.actual_spatial_context_k),
            "spatial_coord_hidden_dim": int(self.spatial_hidden_dim),
            "spatial_context_hidden_dim": int(self.spatial_context_hidden_dim),
            "spatial_fourier_scales": [float(scale) for scale in self.spatial_fourier_scales],
            "spatial_token_scale": float(torch.sigmoid(self.spatial_token_scale_logit).detach().cpu().item()),
            "spatial_token_dropout": float(self.spatial_token_dropout),
            "spatial_consistency_weight": float(target_spatial_consistency_weight),
            "spatial_consistency_use_all_latent": bool(spatial_consistency_use_all_latent),
            "spatial_contrastive_weight": float(target_spatial_contrastive_weight),
            "spatial_contrastive_stop_epoch": int(spatial_contrastive_stop_epoch),
            "spatial_contrastive_use_all_latent": bool(spatial_contrastive_use_all_latent),
            "spatial_contrastive_latent_mode": contrastive_latent_mode,
            "spatial_negative_margin_weight": float(target_spatial_negative_margin_weight),
            "spatial_negative_margin_warmup_epochs": int(spatial_negative_margin_warmup_epochs),
            "spatial_negative_margin_stop_epoch": int(spatial_negative_margin_stop_epoch),
            "spatial_contrastive_mode": self.spatial_contrastive_mode,
            "spatial_negative_margin": float(self.spatial_negative_margin),
            "spatial_positive_weighting": self.spatial_positive_weighting,
            "spatial_positive_aggregation": self.spatial_positive_aggregation,
            "spatial_positive_weight_temperature": float(self.spatial_positive_weight_temperature),
            "spatial_contrastive_source": "expression_pca_per_modality",
            "spatial_contrastive_pos_k": int(self.spatial_contrastive_pos_k),
            "spatial_contrastive_neg_k": int(self.spatial_contrastive_neg_k),
            "spatial_contrastive_temperature": float(self.spatial_contrastive_temperature),
            "spatial_contrastive_neg_strategy": self.spatial_contrastive_neg_strategy,
            "decoder_private_feature_masking": bool(target_decoder_private_feature_masking),
            "decoder_private_mask_probability": float(target_decoder_private_mask_probability),
            "decoder_private_mask_warmup_start": int(target_decoder_private_mask_warmup_start),
            "decoder_private_mask_warmup_end": int(target_decoder_private_mask_warmup_end),
            "spatial_contrastive_early_stop_enabled": bool(early_stop_enabled),
            "spatial_contrastive_early_stop_window_epochs": int(early_stop_window_epochs),
            "spatial_contrastive_early_stop_slope_threshold": float(early_stop_slope_threshold),
            "spatial_contrastive_early_stop_min_epoch": int(early_stop_min_epoch),
            "spatial_contrastive_early_stop_patience": int(early_stop_patience),
            "spatial_contrastive_early_stop_triggered": bool(early_stop_triggered),
            "spatial_contrastive_early_stop_epoch": int(early_stop_epoch),
        }
        return history

    def compute_spatial_consistency_loss(
        self,
        latent_batch: torch.Tensor,
        batch_indices: np.ndarray,
    ) -> torch.Tensor:
        if self.actual_spatial_context_k == 0 or latent_batch.shape[0] <= 1:
            return latent_batch.sum() * 0.0

        batch_indices_t = torch.as_tensor(batch_indices, dtype=torch.long, device=self.device)
        membership = torch.full((self.n_obs,), -1, dtype=torch.long, device=self.device)
        membership[batch_indices_t] = torch.arange(len(batch_indices), device=self.device)

        neighbor_idx_global = self.spatial_neighbor_idx[batch_indices_t]
        neighbor_idx_local = membership[neighbor_idx_global]
        valid_mask = neighbor_idx_local >= 0
        if not torch.any(valid_mask):
            return latent_batch.sum() * 0.0

        center_latent = latent_batch.unsqueeze(1).expand(-1, neighbor_idx_local.shape[1], -1)
        safe_neighbor_idx_local = neighbor_idx_local.clamp(min=0)
        neighbor_latent = latent_batch[safe_neighbor_idx_local]
        pair_dist = torch.norm(center_latent - neighbor_latent, dim=2)
        pair_dist = pair_dist[valid_mask]
        if pair_dist.numel() == 0:
            return latent_batch.sum() * 0.0
        return pair_dist.mean()

    def _zero_spatial_contrastive_details(self, latent_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = latent_batch.sum() * 0.0
        return {
            "negative_margin_loss": zero,
            "negative_mean_cosine": zero,
            "negative_max_cosine": zero,
            "negative_violation_rate": zero,
            "effective_negative_pairs": zero,
            "positive_count_mean": zero,
            "positive_weight_sum_mean": zero,
            "positive_weight_mean": zero,
            "positive_weight_min": zero,
            "positive_weight_max": zero,
            "rank1_weight_mean": zero,
            "rank2_weight_mean": zero,
            "rank3_weight_mean": zero,
            "weighted_positive_distance": zero,
            "unweighted_positive_distance": zero,
        }

    def _compute_positive_weights_and_details(
        self,
        *,
        positive_mask: torch.Tensor,
        positive_distance: torch.Tensor,
        zero_template: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        safe_positive_distance = positive_distance.masked_fill(~positive_mask, 0.0)
        positive_count = positive_mask.sum(dim=1, keepdim=True)
        safe_positive_count = positive_count.clamp(min=1)

        if self.spatial_positive_weighting == "feature_distance":
            logits = (-positive_distance / self.spatial_positive_weight_temperature).masked_fill(
                ~positive_mask,
                float("-inf"),
            )
            positive_weights = torch.softmax(logits, dim=1)
            positive_weights = positive_weights.masked_fill(~positive_mask, 0.0)
        else:
            positive_weights = positive_mask.float()

        if self.spatial_positive_aggregation == "shared_numerator":
            if self.spatial_positive_weighting == "feature_distance":
                positive_weights = positive_weights * safe_positive_count.float()
        else:
            positive_weights = positive_weights / safe_positive_count.float()

        active_weights = positive_weights[positive_mask]
        active_distances = positive_distance[positive_mask]
        if active_weights.numel() == 0:
            zero_details = self._zero_spatial_contrastive_details(zero_template)
            return positive_weights, {
                key: zero_details[key]
                for key in [
                    "positive_count_mean",
                    "positive_weight_sum_mean",
                    "positive_weight_mean",
                    "positive_weight_min",
                    "positive_weight_max",
                    "rank1_weight_mean",
                    "rank2_weight_mean",
                    "rank3_weight_mean",
                    "weighted_positive_distance",
                    "unweighted_positive_distance",
                ]
            }

        rank_means: dict[str, torch.Tensor] = {}
        sorted_idx = positive_distance.masked_fill(~positive_mask, float("inf")).argsort(dim=1)
        for rank_idx in range(3):
            rank_key = f"rank{rank_idx + 1}_weight_mean"
            valid_rank_mask = positive_mask.sum(dim=1) > rank_idx
            if not torch.any(valid_rank_mask):
                rank_means[rank_key] = zero_template.sum() * 0.0
                continue
            selected_idx = sorted_idx[valid_rank_mask, rank_idx]
            selected_weights = positive_weights[valid_rank_mask].gather(1, selected_idx.unsqueeze(1)).squeeze(1)
            rank_means[rank_key] = selected_weights.mean()

        positive_weight_sum = positive_weights.sum(dim=1)
        safe_weight_sum = positive_weight_sum.clamp(min=1e-12).unsqueeze(1)
        normalized_weighted_distance = (
            (positive_weights * safe_positive_distance).sum(dim=1) / safe_weight_sum.squeeze(1)
        ).mean()
        details = {
            "positive_count_mean": positive_count.float().squeeze(1).mean(),
            "positive_weight_sum_mean": positive_weight_sum.mean(),
            "positive_weight_mean": active_weights.mean(),
            "positive_weight_min": active_weights.min(),
            "positive_weight_max": active_weights.max(),
            "weighted_positive_distance": normalized_weighted_distance,
            "unweighted_positive_distance": active_distances.mean(),
        }
        details.update(rank_means)
        return positive_weights, details

    def _compute_shared_numerator_loss(
        self,
        *,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        positive_mask: torch.Tensor,
        positive_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.spatial_positive_weighting == "uniform":
            numerator = torch.logsumexp(pos_logits, dim=1)
            denominator = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=1), dim=1)
            return -(numerator - denominator).mean()

        safe_log_weights = torch.full_like(pos_logits, float("-inf"))
        safe_log_weights[positive_mask] = torch.log(positive_weights[positive_mask])
        weighted_pos_logits = pos_logits + safe_log_weights
        numerator = torch.logsumexp(weighted_pos_logits, dim=1)
        denominator = torch.logsumexp(torch.cat([weighted_pos_logits, neg_logits], dim=1), dim=1)
        return -(numerator - denominator).mean()

    def _compute_individual_positive_loss(
        self,
        *,
        neg_logits: torch.Tensor,
        positive_mask: torch.Tensor,
        pos_logits: torch.Tensor,
        positive_weights: torch.Tensor,
    ) -> torch.Tensor:
        neg_logsumexp = torch.logsumexp(neg_logits, dim=1)
        pos_row_idx, pos_col_idx = positive_mask.nonzero(as_tuple=True)
        pos_logits_vec = pos_logits[pos_row_idx, pos_col_idx]
        neg_logsumexp_vec = neg_logsumexp[pos_row_idx]
        loss_vec = -(pos_logits_vec - torch.logaddexp(pos_logits_vec, neg_logsumexp_vec))
        per_positive_loss = torch.zeros_like(positive_weights)
        per_positive_loss[pos_row_idx, pos_col_idx] = loss_vec
        loss_anchor = (positive_weights * per_positive_loss).sum(dim=1)
        return loss_anchor.mean()

    def _select_spatial_contrastive_pairs(
        self,
        feature_batch: torch.Tensor,
        require_positive: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        batch_size = feature_batch.shape[0]
        if batch_size <= 2 or self.spatial_contrastive_neg_k <= 0:
            return None

        pos_k = min(self.spatial_contrastive_pos_k, batch_size - 1)
        max_neg = max(batch_size - 1 - pos_k, 0)
        neg_k = min(self.spatial_contrastive_neg_k, max_neg)
        if neg_k <= 0:
            return None

        feature_dist = torch.cdist(feature_batch, feature_batch, p=2)
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)

        mutual_mask = torch.zeros_like(eye_mask)
        if pos_k > 0:
            pos_source = feature_dist.masked_fill(eye_mask, float("inf"))
            knn_idx = pos_source.topk(k=pos_k, largest=False).indices
            candidate_mask = torch.zeros_like(eye_mask)
            candidate_mask.scatter_(1, knn_idx, True)
            mutual_mask = candidate_mask & candidate_mask.T

        neg_source = feature_dist.masked_fill(eye_mask, float("inf"))
        neg_source = neg_source.masked_fill(mutual_mask, float("inf"))
        finite_neg = torch.isfinite(neg_source)
        valid_neg_count = finite_neg.sum(dim=1)
        if require_positive:
            valid_anchor_mask = (mutual_mask.sum(dim=1) > 0) & (valid_neg_count >= neg_k)
        else:
            valid_anchor_mask = valid_neg_count >= neg_k
        if not torch.any(valid_anchor_mask):
            return None

        if self.spatial_contrastive_neg_strategy == "mid":
            safe_neg_source = neg_source.masked_fill(~finite_neg, 0.0)
            row_mean = safe_neg_source.sum(dim=1, keepdim=True) / valid_neg_count.clamp(min=1).unsqueeze(1)
            neg_score = (neg_source - row_mean).abs().masked_fill(~finite_neg, float("inf"))
            neg_idx = neg_score.topk(k=neg_k, largest=False).indices
        else:
            neg_idx = neg_source.masked_fill(~finite_neg, float("-inf")).topk(
                k=neg_k,
                largest=True,
            ).indices
        return mutual_mask, neg_idx, valid_anchor_mask

    def compute_spatial_contrastive_loss(
        self,
        latent_batch: torch.Tensor,
        batch_indices: np.ndarray,
        return_details: bool = False,
        feature_source: str = "both",
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        def zero_result() -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
            zero = latent_batch.sum() * 0.0
            if return_details:
                return zero, self._zero_spatial_contrastive_details(latent_batch)
            return zero

        def modality_contrastive_loss(
            feature_batch: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            batch_size = latent_batch.shape[0]
            require_positive = self.spatial_contrastive_mode == "positive_negative"
            if batch_size <= 2 or self.spatial_contrastive_neg_k <= 0:
                return latent_batch.sum() * 0.0, self._zero_spatial_contrastive_details(latent_batch)
            if require_positive and self.spatial_contrastive_pos_k <= 0:
                return latent_batch.sum() * 0.0, self._zero_spatial_contrastive_details(latent_batch)

            pair_selection = self._select_spatial_contrastive_pairs(
                feature_batch=feature_batch,
                require_positive=require_positive,
            )
            if pair_selection is None:
                return latent_batch.sum() * 0.0, self._zero_spatial_contrastive_details(latent_batch)

            mutual_mask, neg_idx, valid_anchor_mask = pair_selection
            feature_dist = torch.cdist(feature_batch, feature_batch, p=2)
            latent_norm = F.normalize(latent_batch, p=2, dim=1, eps=1e-8)
            similarity = torch.matmul(latent_norm, latent_norm.T)
            neg_sim = similarity.gather(1, neg_idx)[valid_anchor_mask]
            if neg_sim.numel() == 0:
                return latent_batch.sum() * 0.0, self._zero_spatial_contrastive_details(latent_batch)

            negative_margin_loss = F.relu(neg_sim - self.spatial_negative_margin).mean()
            negative_mean_cosine = neg_sim.mean()
            negative_max_cosine = neg_sim.max()
            negative_violation_rate = (neg_sim > self.spatial_negative_margin).float().mean()
            effective_negative_pairs = torch.tensor(
                float(neg_sim.numel()),
                device=latent_batch.device,
                dtype=latent_batch.dtype,
            )
            details = {
                "negative_margin_loss": negative_margin_loss,
                "negative_mean_cosine": negative_mean_cosine,
                "negative_max_cosine": negative_max_cosine,
                "negative_violation_rate": negative_violation_rate,
                "effective_negative_pairs": effective_negative_pairs,
            }

            if self.spatial_contrastive_mode == "negative_only":
                details.update(
                    {
                        key: value
                        for key, value in self._zero_spatial_contrastive_details(latent_batch).items()
                        if key not in details
                    }
                )
                return negative_margin_loss, details

            positive_mask = mutual_mask[valid_anchor_mask]
            positive_distance = feature_dist[valid_anchor_mask]
            positive_weights, positive_weight_details = self._compute_positive_weights_and_details(
                positive_mask=positive_mask,
                positive_distance=positive_distance,
                zero_template=latent_batch,
            )
            details.update(positive_weight_details)

            pos_logits = similarity.masked_fill(~mutual_mask, float("-inf")) / self.spatial_contrastive_temperature
            pos_logits = pos_logits[valid_anchor_mask]
            neg_logits = neg_sim / self.spatial_contrastive_temperature
            if self.spatial_positive_aggregation == "shared_numerator":
                loss = self._compute_shared_numerator_loss(
                    pos_logits=pos_logits,
                    neg_logits=neg_logits,
                    positive_mask=positive_mask,
                    positive_weights=positive_weights,
                )
                return loss, details

            loss = self._compute_individual_positive_loss(
                neg_logits=neg_logits,
                positive_mask=positive_mask,
                pos_logits=pos_logits,
                positive_weights=positive_weights,
            )
            return loss, details

        batch_size = latent_batch.shape[0]
        if batch_size <= 2 or self.spatial_contrastive_neg_k <= 0:
            return zero_result()
        if self.spatial_contrastive_mode == "positive_negative" and self.spatial_contrastive_pos_k <= 0:
            return zero_result()

        feature_source = str(feature_source).strip().lower()
        if feature_source not in {"both", "st", "sm"}:
            raise ValueError("feature_source must be 'both', 'st', or 'sm'")
        batch_indices_t = torch.as_tensor(batch_indices, dtype=torch.long, device=self.device)
        feature_st = self.expression_triplet_st[batch_indices_t]
        feature_sm = self.expression_triplet_sm[batch_indices_t]
        if feature_source == "st":
            st_loss, st_details = modality_contrastive_loss(feature_st)
            if return_details:
                return st_loss, st_details
            return st_loss
        if feature_source == "sm":
            sm_loss, sm_details = modality_contrastive_loss(feature_sm)
            if return_details:
                return sm_loss, sm_details
            return sm_loss
        st_loss, st_details = modality_contrastive_loss(feature_st)
        sm_loss, sm_details = modality_contrastive_loss(feature_sm)
        combined_loss = 0.5 * (st_loss + sm_loss)
        combined_details = {
            key: 0.5 * (st_details[key] + sm_details[key])
            for key in st_details
        }
        if return_details:
            return combined_loss, combined_details
        return combined_loss

    @torch.no_grad()
    def _iterate_full(self, n_per_batch: int = 128, latent_key: str = "q_mu"):
        self.eval()
        latents = []
        recon_st = []
        recon_sm = []
        contribution = []
        optional_keys = (
            "contribution_sm",
            "q_mu_shared",
            "q_mu_st",
            "q_mu_sm",
            "similarity_st_joint",
            "similarity_sm_joint",
            "homo_st_embedding",
            "homo_sm_embedding",
            "homo_joint_embedding",
            "spatial_gate_mean",
            "spatial_st_gate_mean",
            "spatial_sm_gate_mean",
            "spatial_token_scale",
            "shared_rep",
        )
        extras: dict[str, list[np.ndarray]] = {key: [] for key in optional_keys}

        for batch_idx in self.as_dataloader(batch_size=n_per_batch, shuffle=False):
            indices = batch_idx[0].cpu().numpy()
            X_batch = self._fetch_rows(indices)
            H, R, _ = self.forward_with_indices(X_batch, indices=indices, reduction="sum")
            latents.append(H[latent_key].detach().cpu().numpy())
            recon_st.append(R["px_rna_scale"].detach().cpu().numpy())
            recon_sm.append(R["px_sm_scale"].detach().cpu().numpy())
            contribution.append(H["contribution_st"].detach().cpu().numpy())
            for key in optional_keys:
                if key in H:
                    extras[key].append(H[key].detach().cpu().numpy())

        extra_outputs: dict[str, np.ndarray] = {}
        for key, values in extras.items():
            if not values:
                continue
            first = values[0]
            extra_outputs[key] = np.concatenate(values) if first.ndim == 1 else np.vstack(values)

        return (
            np.vstack(latents),
            np.vstack(recon_st),
            np.vstack(recon_sm),
            np.concatenate(contribution),
            extra_outputs,
        )
