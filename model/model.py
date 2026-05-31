from __future__ import annotations

import math
import random
from typing import Literal, Optional, Tuple

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
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

        self._init_dataset()

        self.dropout = nn.Dropout(dropout_rate)
        self.proj_st = nn.Linear(self.in_dim_st, proj_dim, bias=False)
        self.proj_sm = nn.Linear(self.in_dim_sm, proj_dim, bias=False)
        self.norm_st = nn.LayerNorm(proj_dim)
        self.norm_sm = nn.LayerNorm(proj_dim)

        self.encoder_uni_st = nn.Linear(proj_dim, proj_dim, bias=False)
        self.encoder_uni_sm = nn.Linear(proj_dim, proj_dim, bias=False)
        self.encoder_com = nn.Linear(proj_dim, proj_dim, bias=False)

        self.private_st_token = nn.Linear(proj_dim, token_dim)
        self.private_sm_token = nn.Linear(proj_dim, token_dim)
        self.common_st_token = nn.Linear(proj_dim, token_dim)
        self.common_sm_token = nn.Linear(proj_dim, token_dim)

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

        if scipy.sparse.issparse(self.X):
            X_dense = self.X.toarray()
        else:
            X_dense = np.asarray(self.X)
        X_dense = X_dense.astype(np.float32, copy=False)

        st_stats_source = np.log1p(np.clip(X_dense[:, self.st_mask], a_min=0.0, a_max=None))
        sm_stats_source = np.log1p(np.clip(X_dense[:, self.sm_mask], a_min=0.0, a_max=None))
        st_mean = st_stats_source.mean(axis=0).astype(np.float32, copy=False)
        sm_mean = sm_stats_source.mean(axis=0).astype(np.float32, copy=False)
        st_std = np.clip(st_stats_source.std(axis=0), a_min=1e-4, a_max=None).astype(np.float32, copy=False)
        sm_std = np.clip(sm_stats_source.std(axis=0), a_min=1e-4, a_max=None).astype(np.float32, copy=False)

        self.register_buffer("st_feature_mean", torch.tensor(st_mean, dtype=torch.float32))
        self.register_buffer("st_feature_std", torch.tensor(st_std, dtype=torch.float32))
        self.register_buffer("sm_feature_mean", torch.tensor(sm_mean, dtype=torch.float32))
        self.register_buffer("sm_feature_std", torch.tensor(sm_std, dtype=torch.float32))

    def as_dataloader(self, batch_size: int = 128, shuffle: bool = True, generator: Optional[torch.Generator] = None) -> DataLoader:
        dataset = TensorDataset(torch.tensor(self.indices, dtype=torch.long))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

    def _fetch_rows(self, indices: np.ndarray) -> torch.Tensor:
        if scipy.sparse.issparse(self.X):
            rows = self.X[indices].toarray()
        else:
            rows = self.X[indices]
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _transform_st_features(self, X_st: torch.Tensor) -> torch.Tensor:
        return torch.log1p(X_st.clamp_min(0.0))

    def _transform_sm_features(self, X_sm: torch.Tensor) -> torch.Tensor:
        return torch.log1p(X_sm.clamp_min(0.0))

    def _transform_sm_prediction(self, X_sm: torch.Tensor) -> torch.Tensor:
        return torch.log1p(F.softplus(X_sm))

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
        if self.standardize_inputs:
            st_input = self._standardize_st_features(X_st)
            sm_input = self._standardize_sm_features(X_sm)
        else:
            st_input = self._transform_st_features(X_st)
            sm_input = X_sm
        h_st = self.norm_st(self.proj_st(st_input))
        h_sm = self.norm_sm(self.proj_sm(sm_input))
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
        homo_loss = self.compute_homo_loss(c_st, c_sm)

        st_token = self.private_st_token(s_st)
        sm_token = self.private_sm_token(s_sm)
        common_st_token = self.common_st_token(c_st)
        common_sm_token = self.common_sm_token(c_sm)

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
        px_rna_scale = self.px_rna_scale_decoder(hidden) * lib_size.unsqueeze(1)
        px_rna_rate = self.px_rna_rate_decoder(hidden)
        px_rna_dropout = self.px_rna_dropout_decoder(hidden)
        px_sm_scale = self.px_sm_scale_decoder(hidden)
        px_sm_rate = self.px_sm_rate_decoder(hidden)
        px_sm_dropout = self.px_sm_dropout_decoder(hidden)
        return {
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
                    mu=R["px_rna_scale"],
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
                    self._transform_sm_features(H["X_sm"]),
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
        self.spatial_contrastive_pos_k = max(int(spatial_contrastive_pos_k), 0)
        self.spatial_contrastive_neg_k = max(int(spatial_contrastive_neg_k), 0)
        self.spatial_contrastive_temperature = max(float(spatial_contrastive_temperature), 1e-4)
        self.spatial_contrastive_neg_strategy = str(spatial_contrastive_neg_strategy).strip().lower()
        if self.spatial_contrastive_neg_strategy not in {"farthest", "mid"}:
            raise ValueError(
                "spatial_contrastive_neg_strategy must be 'farthest' or 'mid'."
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

        self.decoder_st = nn.Sequential(
            nn.Linear(self.n_latent_shared + self.n_latent_st, self.proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.decoder_sm = nn.Sequential(
            nn.Linear(self.n_latent_shared + self.n_latent_sm, self.proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
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
        coord_mean = coords.mean(axis=0).astype(np.float32, copy=False)
        coord_std = np.clip(coords.std(axis=0), a_min=1e-4, a_max=None).astype(np.float32, copy=False)
        standardized_coords = (coords - coord_mean[None, :]) / coord_std[None, :]

        self.register_buffer("spatial_coords", torch.tensor(coords, dtype=torch.float32))
        self.register_buffer("spatial_coord_mean", torch.tensor(coord_mean, dtype=torch.float32))
        self.register_buffer("spatial_coord_std", torch.tensor(coord_std, dtype=torch.float32))
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
            nn.LayerNorm(3 * self.token_dim),
            nn.Linear(3 * self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.q_mu_shared_fc = nn.Linear(self.token_dim, self.n_latent_shared)
        self.q_logvar_shared_fc = nn.Linear(self.token_dim, self.n_latent_shared)
        self.to(self.device)

    def encode(self, X: torch.Tensor):
        X_st = X[:, self.st_mask]
        X_sm = X[:, self.sm_mask]

        h_st, h_sm = self.project_inputs(X_st, X_sm)
        s_st = self.encoder_uni_st(h_st)
        s_sm = self.encoder_uni_sm(h_sm)
        c_st = self.encoder_com(h_st)
        c_sm = self.encoder_com(h_sm)

        dec_loss = self.compute_decoupling_loss(s_st, c_st) + self.compute_decoupling_loss(s_sm, c_sm)
        hete_loss = s_st.new_zeros(())
        homo_loss = self.compute_homo_loss(c_st, c_sm)

        st_token = self.private_st_token(s_st)
        sm_token = self.private_sm_token(s_sm)
        common_st_token = self.common_st_token(c_st)
        common_sm_token = self.common_sm_token(c_sm)

        shared_rep = self.shared_resample(torch.cat([common_st_token, common_sm_token], dim=1))
        contribution_outputs = self._compute_shared_cosine_contributions(
            common_st_token=common_st_token,
            common_sm_token=common_sm_token,
            shared_rep=shared_rep,
        )
        q_mu_shared = self.q_mu_shared_fc(shared_rep)
        q_logvar_shared = self.q_logvar_shared_fc(shared_rep).clamp(min=-8.0, max=8.0)
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
        }

    def decode(self, H: dict[str, torch.Tensor], lib_size: torch.Tensor):
        hidden_st = self.decoder_st(torch.cat([H["z_shared"], H["z_st"]], dim=1))
        hidden_sm = self.decoder_sm(torch.cat([H["z_shared"], H["z_sm"]], dim=1))
        px_rna_scale = self.px_rna_scale_decoder(hidden_st) * lib_size.unsqueeze(1)
        px_rna_rate = self.px_rna_rate_decoder(hidden_st)
        px_rna_dropout = self.px_rna_dropout_decoder(hidden_st)
        px_sm_scale = self.px_sm_scale_decoder(hidden_sm)
        px_sm_rate = self.px_sm_rate_decoder(hidden_sm)
        px_sm_dropout = self.px_sm_dropout_decoder(hidden_sm)
        return {
            "px_rna_scale": px_rna_scale,
            "px_rna_rate": px_rna_rate,
            "px_rna_dropout": px_rna_dropout,
            "px_sm_scale": px_sm_scale,
            "px_sm_rate": px_sm_rate,
            "px_sm_dropout": px_sm_dropout,
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
        if self.n_obs <= 1 or self.spatial_context_k <= 0:
            neighbor_idx = np.zeros((self.n_obs, 0), dtype=np.int64)
            neighbor_rel = np.zeros((self.n_obs, 0, 2), dtype=np.float32)
            neighbor_dist = np.zeros((self.n_obs, 0, 1), dtype=np.float32)
        else:
            n_neighbors = min(self.spatial_context_k + 1, self.n_obs)
            knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
            knn.fit(coords)
            dists, indices = knn.kneighbors(coords)
            neighbor_idx = indices[:, 1:].astype(np.int64, copy=False)
            base_dists = dists[:, 1:].astype(np.float32, copy=False)
            local_scale = np.clip(np.median(base_dists, axis=1, keepdims=True), a_min=1e-4, a_max=None)
            neighbor_coords = coords[neighbor_idx]
            neighbor_rel = (neighbor_coords - coords[:, None, :]) / local_scale[:, :, None]
            neighbor_rel = neighbor_rel.astype(np.float32, copy=False)
            neighbor_dist = np.linalg.norm(neighbor_rel, axis=2, keepdims=True).astype(np.float32, copy=False)

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
        abs_token = self.spatial_abs_token(self._absolute_spatial_features_from_indices(indices))
        zero_gate = abs_token.new_zeros(abs_token.shape[0], self.token_dim)
        zero_scalar = abs_token.new_zeros(abs_token.shape[0])

        if self.spatial_encoder_mode == "baseline":
            spatial_token = abs_token
        else:
            rel_features = self._relative_spatial_features_from_indices(indices)
            if rel_features.shape[1] > 0:
                local_neighbor_tokens = self.spatial_local_encoder(rel_features)
                local_mean = local_neighbor_tokens.mean(dim=1)
                local_max = local_neighbor_tokens.max(dim=1).values
                local_token = self.spatial_local_pool(torch.cat([local_mean, local_max], dim=1))
            else:
                local_token = abs_token.new_zeros(abs_token.shape)

            spatial_token = self.spatial_token_fuse(torch.cat([abs_token, local_token], dim=1))

        spatial_scale = torch.sigmoid(self.spatial_token_scale_logit)
        if self.training and self.spatial_token_dropout > 0.0:
            keep_prob = 1.0 - self.spatial_token_dropout
            keep_mask = torch.empty(
                spatial_token.shape[0],
                1,
                dtype=spatial_token.dtype,
                device=spatial_token.device,
            ).bernoulli_(keep_prob)
            spatial_token = spatial_token * keep_mask / max(keep_prob, 1e-6)
        spatial_token = spatial_scale * spatial_token
        if self.spatial_encoder_mode == "baseline":
            common_st_token_ctx = common_st_token
            common_sm_token_ctx = common_sm_token
            st_gate = zero_gate
            sm_gate = zero_gate
        else:
            st_gate = self.spatial_st_gate(torch.cat([common_st_token, spatial_token], dim=1))
            sm_gate = self.spatial_sm_gate(torch.cat([common_sm_token, spatial_token], dim=1))
            common_st_token_ctx = common_st_token + st_gate * spatial_token
            common_sm_token_ctx = common_sm_token + sm_gate * spatial_token

        joint_gate = self.spatial_joint_gate(
            torch.cat([common_st_token_ctx, common_sm_token_ctx, spatial_token], dim=1)
        )
        gated_spatial_token = joint_gate * spatial_token
        shared_rep = self.shared_resample(
            torch.cat([common_st_token_ctx, common_sm_token_ctx, gated_spatial_token], dim=1)
        )

        return {
            "spatial_token": spatial_token,
            "shared_rep": shared_rep,
            "common_st_token_ctx": common_st_token_ctx,
            "common_sm_token_ctx": common_sm_token_ctx,
            "spatial_joint_gate_mean": joint_gate.mean(dim=1),
            "spatial_st_gate_mean": st_gate.mean(dim=1) if st_gate.numel() > 0 else zero_scalar,
            "spatial_sm_gate_mean": sm_gate.mean(dim=1) if sm_gate.numel() > 0 else zero_scalar,
        }

    def encode_with_indices(self, X: torch.Tensor, indices: np.ndarray):
        X_st = X[:, self.st_mask]
        X_sm = X[:, self.sm_mask]

        h_st, h_sm = self.project_inputs(X_st, X_sm)
        s_st = self.encoder_uni_st(h_st)
        s_sm = self.encoder_uni_sm(h_sm)
        c_st = self.encoder_com(h_st)
        c_sm = self.encoder_com(h_sm)

        dec_loss = self.compute_decoupling_loss(s_st, c_st) + self.compute_decoupling_loss(s_sm, c_sm)
        hete_loss = s_st.new_zeros(())
        homo_loss = self.compute_homo_loss(c_st, c_sm)

        st_token = self.private_st_token(s_st)
        sm_token = self.private_sm_token(s_sm)
        common_st_token = self.common_st_token(c_st)
        common_sm_token = self.common_sm_token(c_sm)

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
                    mu=R["px_rna_scale"],
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
                    self._transform_sm_features(H["X_sm"]),
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
        spatial_contrastive_use_all_latent: bool = False,
    ):
        self.train()
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
        target_spatial_contrastive_weight = float(spatial_contrastive_weight)
        prev_task_weights = self._equal_task_weights()
        task_scale = float(len(self.TASK_KEYS))

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
            "epoch_task_loss_shared_list": [],
            "epoch_task_loss_reconstruction_st_list": [],
            "epoch_task_loss_reconstruction_sm_list": [],
            "epoch_task_weight_shared_list": [],
            "epoch_task_weight_reconstruction_st_list": [],
            "epoch_task_weight_reconstruction_sm_list": [],
            "epoch_spatial_consistency_loss_list": [],
            "epoch_spatial_contrastive_loss_list": [],
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

            for batch_idx in self.as_dataloader(batch_size=n_per_batch, shuffle=True, generator=data_loader_generator):
                indices = batch_idx[0].cpu().numpy()
                X_batch = self._fetch_rows(indices)
                H, _, losses = self.forward_with_indices(X_batch, indices=indices, reduction=reconstruction_reduction)

                rec_st = losses["reconstruction_loss_st"].mean()
                rec_sm = losses["reconstruction_loss_sm"].mean()
                kl = (
                    losses["kldiv_loss"].sum() / max(len(indices), 1)
                    if kl_loss_reduction == "sum"
                    else losses["kldiv_loss"].mean()
                )
                dec = losses["dec_loss"]
                hete = losses["hete_loss"]
                homo = losses["homo_loss"]

                task_losses = {
                    "shared": dec_weight * dec + current_hete_weight * hete + current_homo_weight * homo,
                    "reconstruction_st": reconstruction_st_weight * rec_st,
                    "reconstruction_sm": reconstruction_sm_weight * rec_sm,
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
                spatial_consistency_loss = self.compute_spatial_consistency_loss(
                    latent_batch=latent_for_consistency,
                    batch_indices=indices,
                )
                latent_for_contrastive = H["q_mu"] if spatial_contrastive_use_all_latent else H["q_mu_shared"]
                spatial_contrastive_loss = self.compute_spatial_contrastive_loss(
                    latent_batch=latent_for_contrastive,
                    batch_indices=indices,
                )
                loss = task_scale * sum(task_weights[key] * task_losses[key] for key in self.TASK_KEYS)
                loss = loss + current_spatial_consistency_weight * spatial_consistency_loss
                loss = loss + current_spatial_contrastive_weight * spatial_contrastive_loss
                loss = loss + kl_weight * kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                stats["reconstruction_loss_st"] += rec_st.item()
                stats["reconstruction_loss_sm"] += rec_sm.item()
                stats["kldiv_loss"] += kl.item()
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
            history["epoch_total_loss_list"].append(stats["total_loss"])
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
            "balance_start_epoch": int(balance_start_epoch),
            "balance_ema": float(balance_ema),
            "task_weight_floor": float(task_weight_floor),
            "kl_used": bool(kl_weight > 0),
            "spatial_branch": f"explicit_spatial_coord_token::{self.spatial_encoder_mode}",
            "spatial_context_k": int(self.actual_spatial_context_k),
            "spatial_coord_hidden_dim": int(self.spatial_hidden_dim),
            "spatial_context_hidden_dim": int(self.spatial_context_hidden_dim),
            "spatial_fourier_scales": [float(scale) for scale in self.spatial_fourier_scales],
            "spatial_token_scale": float(torch.sigmoid(self.spatial_token_scale_logit).detach().cpu().item()),
            "spatial_token_dropout": float(self.spatial_token_dropout),
            "spatial_consistency_weight": float(target_spatial_consistency_weight),
            "spatial_consistency_use_all_latent": bool(spatial_consistency_use_all_latent),
            "spatial_contrastive_weight": float(target_spatial_contrastive_weight),
            "spatial_contrastive_use_all_latent": bool(spatial_contrastive_use_all_latent),
            "spatial_contrastive_pos_k": int(self.spatial_contrastive_pos_k),
            "spatial_contrastive_neg_k": int(self.spatial_contrastive_neg_k),
            "spatial_contrastive_temperature": float(self.spatial_contrastive_temperature),
            "spatial_contrastive_neg_strategy": self.spatial_contrastive_neg_strategy,
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

    def compute_spatial_contrastive_loss(
        self,
        latent_batch: torch.Tensor,
        batch_indices: np.ndarray,
    ) -> torch.Tensor:
        batch_size = latent_batch.shape[0]
        if batch_size <= 2 or self.spatial_contrastive_pos_k <= 0 or self.spatial_contrastive_neg_k <= 0:
            return latent_batch.sum() * 0.0

        pos_k = min(self.spatial_contrastive_pos_k, batch_size - 1)
        neg_k = min(self.spatial_contrastive_neg_k, max(batch_size - 1 - pos_k, 0))
        if pos_k <= 0 or neg_k <= 0:
            return latent_batch.sum() * 0.0

        batch_indices_t = torch.as_tensor(batch_indices, dtype=torch.long, device=self.device)
        coords = self.spatial_coords_standardized[batch_indices_t]
        spatial_dist = torch.cdist(coords, coords, p=2)
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)

        pos_source = spatial_dist.masked_fill(eye_mask, float("inf"))
        pos_idx = pos_source.topk(k=pos_k, largest=False).indices

        neg_source = spatial_dist.masked_fill(eye_mask, float("inf"))
        neg_source.scatter_(1, pos_idx, float("inf"))
        if self.spatial_contrastive_neg_strategy == "mid":
            finite_neg = torch.isfinite(neg_source)
            safe_neg_source = neg_source.masked_fill(~finite_neg, 0.0)
            valid_count = finite_neg.sum(dim=1, keepdim=True).clamp(min=1)
            row_mean = safe_neg_source.sum(dim=1, keepdim=True) / valid_count
            neg_score = (neg_source - row_mean).abs().masked_fill(~finite_neg, float("inf"))
            neg_idx = neg_score.topk(k=neg_k, largest=False).indices
        else:
            neg_idx = neg_source.masked_fill(~torch.isfinite(neg_source), float("-inf")).topk(
                k=neg_k,
                largest=True,
            ).indices

        latent_norm = F.normalize(latent_batch, p=2, dim=1, eps=1e-8)
        similarity = torch.matmul(latent_norm, latent_norm.T)
        pos_logits = similarity.gather(1, pos_idx) / self.spatial_contrastive_temperature
        neg_logits = similarity.gather(1, neg_idx) / self.spatial_contrastive_temperature

        numerator = torch.logsumexp(pos_logits, dim=1)
        denominator = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=1), dim=1)
        return -(numerator - denominator).mean()

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
            "spatial_gate_mean",
            "spatial_st_gate_mean",
            "spatial_sm_gate_mean",
            "spatial_token_scale",
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
