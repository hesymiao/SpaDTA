from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kld

from .loss import LossFunction
from .model import DecAlignSpatialMetaLinear, get_tqdm


class DecAlignSpatialMetaLinearHorizontal(DecAlignSpatialMetaLinear):
    def __init__(
        self,
        *args,
        batch_keys: Optional[list[str] | str] = None,
        batch_embedding: Literal["embedding", "onehot"] = "embedding",
        batch_hidden_dim: int = 8,
        **kwargs,
    ):
        self.requested_batch_keys = [batch_keys] if isinstance(batch_keys, str) else list(batch_keys or [])
        self.batch_embedding = str(batch_embedding)
        self.batch_hidden_dim = int(batch_hidden_dim)
        self.batch_keys: list[str] = []
        self.batch_categories: list[pd.Categorical] = []
        self.batch_codes: list[np.ndarray] = []
        self.n_batch_keys: list[int] = []
        self.batch_condition_dim = 0

        super().__init__(*args, **kwargs)

        if self.batch_embedding not in {"embedding", "onehot"}:
            raise ValueError(f"Unsupported batch_embedding: {self.batch_embedding}")

        if self.requested_batch_keys:
            if self.batch_embedding == "embedding":
                self.batch_embeddings = nn.ModuleList(
                    [nn.Embedding(n_categories, self.batch_hidden_dim) for n_categories in self.n_batch_keys]
                )
                self.batch_condition_dim = len(self.n_batch_keys) * self.batch_hidden_dim
            else:
                self.batch_embeddings = nn.ModuleList()
                self.batch_condition_dim = int(sum(self.n_batch_keys))
        else:
            self.batch_embeddings = nn.ModuleList()
            self.batch_condition_dim = 0

        dropout_rate = float(self.dropout.p)
        self.decoder_st = nn.Sequential(
            nn.Linear(self.n_latent_shared + self.n_latent_st + self.batch_condition_dim, self.proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.decoder_sm = nn.Sequential(
            nn.Linear(self.n_latent_shared + self.n_latent_sm + self.batch_condition_dim, self.proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.to(self.device)

    def _init_dataset(self) -> None:
        super()._init_dataset()

        if not self.requested_batch_keys:
            self.batch_keys = []
            self.batch_categories = []
            self.batch_codes = []
            self.n_batch_keys = []
            return

        self.batch_keys = list(self.requested_batch_keys)
        missing = [key for key in self.batch_keys if key not in self.adata.obs.columns]
        if missing:
            raise ValueError(f"adata.obs is missing batch key(s): {missing}")

        self.batch_categories = [pd.Categorical(self.adata.obs[key].astype(str)) for key in self.batch_keys]
        self.batch_codes = [np.asarray(cat.codes, dtype=np.int64) for cat in self.batch_categories]
        self.n_batch_keys = [len(cat.categories) for cat in self.batch_categories]

    def _init_spatial_context(self, coords: np.ndarray) -> None:
        if self.n_obs <= 1 or self.spatial_context_k <= 0:
            return super()._init_spatial_context(coords)

        if not self.requested_batch_keys:
            return super()._init_spatial_context(coords)

        primary_key = self.requested_batch_keys[0]
        if primary_key not in self.adata.obs.columns:
            return super()._init_spatial_context(coords)

        labels = np.asarray(self.adata.obs[primary_key].astype(str))
        neighbor_idx_chunks: list[tuple[np.ndarray, np.ndarray]] = []
        neighbor_rel_chunks: list[tuple[np.ndarray, np.ndarray]] = []
        neighbor_dist_chunks: list[tuple[np.ndarray, np.ndarray]] = []
        max_k = 0

        for label in pd.unique(labels):
            global_idx = np.flatnonzero(labels == label)
            sub_coords = coords[global_idx]
            if sub_coords.shape[0] <= 1:
                local_idx = np.zeros((sub_coords.shape[0], 0), dtype=np.int64)
                local_rel = np.zeros((sub_coords.shape[0], 0, 2), dtype=np.float32)
                local_dist = np.zeros((sub_coords.shape[0], 0, 1), dtype=np.float32)
            else:
                n_neighbors = min(self.spatial_context_k + 1, sub_coords.shape[0])
                from sklearn.neighbors import NearestNeighbors

                knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
                knn.fit(sub_coords)
                dists, indices = knn.kneighbors(sub_coords)
                local_idx = global_idx[indices[:, 1:]].astype(np.int64, copy=False)
                base_dists = dists[:, 1:].astype(np.float32, copy=False)
                local_scale = np.clip(np.median(base_dists, axis=1, keepdims=True), a_min=1e-4, a_max=None)
                neighbor_coords = coords[local_idx]
                local_rel = ((neighbor_coords - sub_coords[:, None, :]) / local_scale[:, :, None]).astype(
                    np.float32,
                    copy=False,
                )
                local_dist = np.linalg.norm(local_rel, axis=2, keepdims=True).astype(np.float32, copy=False)

            max_k = max(max_k, int(local_idx.shape[1]))
            neighbor_idx_chunks.append((global_idx, local_idx))
            neighbor_rel_chunks.append((global_idx, local_rel))
            neighbor_dist_chunks.append((global_idx, local_dist))

        neighbor_idx = np.zeros((self.n_obs, max_k), dtype=np.int64)
        neighbor_rel = np.zeros((self.n_obs, max_k, 2), dtype=np.float32)
        neighbor_dist = np.zeros((self.n_obs, max_k, 1), dtype=np.float32)

        for global_idx, local_idx in neighbor_idx_chunks:
            k = local_idx.shape[1]
            if k > 0:
                neighbor_idx[global_idx, :k] = local_idx
        for global_idx, local_rel in neighbor_rel_chunks:
            k = local_rel.shape[1]
            if k > 0:
                neighbor_rel[global_idx, :k, :] = local_rel
        for global_idx, local_dist in neighbor_dist_chunks:
            k = local_dist.shape[1]
            if k > 0:
                neighbor_dist[global_idx, :k, :] = local_dist

        self.actual_spatial_context_k = int(max_k)
        self.register_buffer("spatial_neighbor_idx", torch.tensor(neighbor_idx, dtype=torch.long))
        self.register_buffer("spatial_neighbor_rel", torch.tensor(neighbor_rel, dtype=torch.float32))
        self.register_buffer("spatial_neighbor_dist", torch.tensor(neighbor_dist, dtype=torch.float32))

    def _fetch_batch_codes(self, indices: np.ndarray) -> Optional[torch.Tensor]:
        if not self.batch_codes:
            return None
        tensors = [
            torch.tensor(code[indices], dtype=torch.long, device=self.device).unsqueeze(1)
            for code in self.batch_codes
        ]
        return torch.hstack(tensors)

    def _encode_batch_context(self, batch_codes: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        if not self.batch_keys or batch_codes is None:
            return torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)

        pieces: list[torch.Tensor] = []
        if self.batch_embedding == "embedding":
            for key_idx, embedding in enumerate(self.batch_embeddings):
                pieces.append(embedding(batch_codes[:, key_idx]))
        else:
            for key_idx, n_categories in enumerate(self.n_batch_keys):
                pieces.append(F.one_hot(batch_codes[:, key_idx], num_classes=n_categories).to(torch.float32))
        return torch.cat(pieces, dim=1)

    def _compute_horizontal_alignment_loss(
        self,
        shared_latent: torch.Tensor,
        batch_codes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if batch_codes is None or batch_codes.numel() == 0:
            return shared_latent.new_zeros(())

        primary_codes = batch_codes[:, 0]
        unique_codes = torch.unique(primary_codes)
        if unique_codes.numel() < 2:
            return shared_latent.new_zeros(())

        loss = shared_latent.new_zeros(())
        n_pairs = 0
        for left_idx in range(unique_codes.numel()):
            for right_idx in range(left_idx + 1, unique_codes.numel()):
                left_latent = shared_latent[primary_codes == unique_codes[left_idx]]
                right_latent = shared_latent[primary_codes == unique_codes[right_idx]]
                if left_latent.shape[0] <= 1 or right_latent.shape[0] <= 1:
                    continue
                loss = loss + LossFunction.mmd_loss_trvae(left_latent, right_latent)
                n_pairs += 1

        if n_pairs == 0:
            return shared_latent.new_zeros(())
        return loss / float(n_pairs)

    def decode(
        self,
        H: dict[str, torch.Tensor],
        lib_size: torch.Tensor,
        batch_codes: Optional[torch.Tensor] = None,
    ):
        batch_context = self._encode_batch_context(batch_codes, H["z_shared"].shape[0])
        hidden_st = self.decoder_st(torch.cat([H["z_shared"], H["z_st"], batch_context], dim=1))
        hidden_sm = self.decoder_sm(torch.cat([H["z_shared"], H["z_sm"], batch_context], dim=1))
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

    def forward_with_indices(
        self,
        X: torch.Tensor,
        indices: np.ndarray,
        reduction: str = "mean",
        batch_codes: Optional[torch.Tensor] = None,
    ):
        H = self.encode_with_indices(X, indices=indices)
        kldiv_loss = kld(
            Normal(H["q_mu"], H["q_var"].sqrt()),
            Normal(torch.zeros_like(H["q_mu"]), torch.ones_like(H["q_var"])),
        ).sum(dim=1)

        lib_size = H["X_st"].sum(1).clamp(min=1.0)
        R = self.decode(H, lib_size, batch_codes=batch_codes)
        horizontal_loss = self._compute_horizontal_alignment_loss(H["q_mu_shared"], batch_codes=batch_codes)

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
            "horizontal_loss": horizontal_loss,
        }

    def forward(
        self,
        X: torch.Tensor,
        reduction: str = "mean",
        indices: Optional[np.ndarray] = None,
        batch_codes: Optional[torch.Tensor] = None,
    ):
        if indices is None:
            if X.shape[0] != self.n_obs:
                raise ValueError("horizontal spatial branch requires explicit batch indices for mini-batch forward.")
            indices = self.indices
        return self.forward_with_indices(
            X,
            indices=np.asarray(indices),
            reduction=reduction,
            batch_codes=batch_codes,
        )

    def fit(self, *args, horizontal_weight: float = 1.0, horizontal_warmup_epochs: int = 16, **kwargs):
        self._horizontal_weight = float(horizontal_weight)
        self._horizontal_warmup_epochs = int(horizontal_warmup_epochs)
        return self._fit_horizontal(*args, **kwargs)

    def _fit_horizontal(
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
        spatial_contrastive_weight: float = 0.0,
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
        target_horizontal_weight = self._horizontal_weight
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
            "epoch_horizontal_loss_list": [],
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
                    "horizontal_loss",
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
            current_horizontal_weight = ramp_weight(
                target_horizontal_weight,
                self._horizontal_warmup_epochs,
                epoch_idx,
            )
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
                batch_codes = self._fetch_batch_codes(indices)
                H, _, losses = self.forward_with_indices(
                    X_batch,
                    indices=indices,
                    batch_codes=batch_codes,
                    reduction=reconstruction_reduction,
                )

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
                horizontal = losses["horizontal_loss"]

                task_losses = {
                    "shared": dec_weight * dec
                    + current_hete_weight * hete
                    + current_homo_weight * homo
                    + current_horizontal_weight * horizontal,
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
                spatial_consistency_loss = self.compute_spatial_consistency_loss(latent_for_consistency, indices)
                latent_for_contrastive = H["q_mu"] if spatial_contrastive_use_all_latent else H["q_mu_shared"]
                spatial_contrastive_loss = self.compute_spatial_contrastive_loss(latent_for_contrastive, indices)

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
                stats["horizontal_loss"] += horizontal.item()
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
                    "hori": f"{stats['horizontal_loss']:.2e}",
                    "w_sh": f"{stats['task_weight_shared']:.2f}",
                }
            )
            pbar.update(1)

            for key in history:
                stat_key = key.removeprefix("epoch_").removesuffix("_list")
                history[key].append(stats[stat_key])
            ran_epochs = epoch_idx + 1

        pbar.close()
        total_loss_history = history["epoch_total_loss_list"]
        self.fit_metadata = {
            "ran_epochs": int(ran_epochs),
            "min_total_loss_epoch": int(np.argmin(total_loss_history)) + 1 if total_loss_history else 0,
            "min_total_loss": float(min(total_loss_history)) if total_loss_history else float("nan"),
            "final_total_loss": float(total_loss_history[-1]) if total_loss_history else float("nan"),
            "balance_start_epoch": int(balance_start_epoch),
            "balance_ema": float(balance_ema),
            "task_weight_floor": float(task_weight_floor),
            "batch_keys": list(self.batch_keys),
            "batch_embedding": str(self.batch_embedding),
            "batch_hidden_dim": int(self.batch_hidden_dim),
            "horizontal_weight": float(target_horizontal_weight),
            "horizontal_warmup_epochs": int(self._horizontal_warmup_epochs),
            "kl_used": bool(kl_weight > 0),
            "spatial_branch": f"explicit_spatial_coord_token::{self.spatial_encoder_mode}",
            "spatial_context_k": int(self.actual_spatial_context_k),
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
            "spatial_gate_mean",
            "spatial_st_gate_mean",
            "spatial_sm_gate_mean",
            "spatial_token_scale",
        )
        extras: dict[str, list[np.ndarray]] = {key: [] for key in optional_keys}

        for batch_idx in self.as_dataloader(batch_size=n_per_batch, shuffle=False):
            indices = batch_idx[0].cpu().numpy()
            X_batch = self._fetch_rows(indices)
            batch_codes = self._fetch_batch_codes(indices)
            H, R, _ = self.forward_with_indices(
                X_batch,
                indices=indices,
                batch_codes=batch_codes,
                reduction="sum",
            )
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
