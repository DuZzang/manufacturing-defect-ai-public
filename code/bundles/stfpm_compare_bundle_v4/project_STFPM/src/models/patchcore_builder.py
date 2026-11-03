"""PatchCore model builder module using Anomalib API with OOM-safe memory management."""

import gc
import types
from typing import List, Optional

import torch
from tqdm import tqdm

from anomalib.models import Patchcore
from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy, SparseRandomProjection


def _safe_forward(self, input_tensor: torch.Tensor):
    """Memory-safe forward wrapper: offloads batch embeddings to CPU immediately during training."""
    output = self._orig_forward(input_tensor)
    if self.training and len(self.embedding_store) > 0:
        last_emb = self.embedding_store[-1]
        if last_emb.is_cuda:
            self.embedding_store[-1] = last_emb.detach().cpu()
            del last_emb
            torch.cuda.empty_cache()
    return output


def _safe_subsample_embedding(self, sampling_ratio: float, embeddings: torch.Tensor = None) -> None:
    """Memory-safe coreset subsampling: avoids OOM by streaming projection and running KCenterGreedy on GPU."""
    if len(self.embedding_store) == 0:
        if embeddings is not None:
            embeddings_cpu = embeddings.detach().cpu()
        else:
            raise ValueError("Embedding store is empty. Cannot perform coreset selection.")
    else:
        embeddings_cpu = torch.vstack(self.embedding_store)
        self.embedding_store.clear()
        gc.collect()

    n_samples, n_features = embeddings_cpu.shape
    coreset_size = int(n_samples * sampling_ratio)
    if coreset_size <= 0:
        raise ValueError(
            f"coreset_size must be a positive integer, got {coreset_size}. "
            f"Check sampling_ratio ({sampling_ratio}) and embedding size ({n_samples})."
        )

    # Determine execution device from model parameters
    try:
        device = next(self.parameters()).device
    except (StopIteration, Exception):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()

        # 1. Sparse random projection fitted with target dimension
        proj = SparseRandomProjection(eps=0.9)
        proj.n_components = proj._johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=proj.eps)
        proj.sparse_random_matrix = proj._sparse_random_matrix(n_features=n_features).to(device)

        # 2. Chunked GPU projection to prevent allocating 2.5GB + 2.3GB simultaneously
        features_gpu = torch.empty((n_samples, proj.n_components), dtype=embeddings_cpu.dtype, device=device)
        chunk_size = 200000
        for i in range(0, n_samples, chunk_size):
            chunk_cpu = embeddings_cpu[i : i + chunk_size]
            chunk_gpu = chunk_cpu.to(device)
            features_gpu[i : i + chunk_size] = chunk_gpu @ proj.sparse_random_matrix.T.type(chunk_gpu.dtype)
            del chunk_gpu
        torch.cuda.empty_cache()

        # 3. KCenterGreedy coreset selection on GPU features
        sampler = KCenterGreedy(embedding=embeddings_cpu, sampling_ratio=sampling_ratio)
        sampler.features = features_gpu
        sampler.reset_distances()

        idx = torch.randint(high=sampler.n_observations, size=(1,), device=features_gpu.device).squeeze()
        selected_coreset_idxs = [idx]

        for _ in tqdm(range(sampler.coreset_size - 1), desc="Selecting Coreset Indices (GPU Safe)"):
            sampler.update_distances(cluster_center=idx)
            idx = sampler.get_new_idx()
            sampler.min_distances.scatter_(0, idx.unsqueeze(0).unsqueeze(1), 0.0)
            selected_coreset_idxs.append(idx)

        idxs = torch.stack(selected_coreset_idxs).cpu().tolist()

        # Clean up projected GPU features immediately
        del features_gpu, sampler, proj
        torch.cuda.empty_cache()

        # 4. Final coreset patches to GPU buffer
        self.memory_bank = embeddings_cpu[idxs].to(device=device, dtype=torch.float32)
        del embeddings_cpu
        gc.collect()
        torch.cuda.empty_cache()

    else:
        # Fallback for CPU execution
        sampler = KCenterGreedy(embedding=embeddings_cpu, sampling_ratio=sampling_ratio)
        self.memory_bank = sampler.sample_coreset().to(device=device, dtype=torch.float32)
        del embeddings_cpu
        gc.collect()


def build_patchcore_model(
    backbone: str = "resnet34",
    layers: Optional[List[str]] = None,
    coreset_sampling_ratio: float = 0.1,
    num_neighbors: int = 9,
    pre_trained: bool = True,
    enable_safe_memory: bool = True,
) -> Patchcore:
    """Builds and initializes a PatchCore model with ResNet-34 backbone and safe memory handling.

    Args:
        backbone: Feature extractor backbone architecture (default: 'resnet34').
        layers: Layers used for feature extraction (default: ['layer2', 'layer3']).
        coreset_sampling_ratio: Subsampling ratio for memory bank reduction (default: 0.1).
        num_neighbors: Number of nearest neighbors for scoring (default: 9).
        pre_trained: Whether to use ImageNet pre-trained weights.
        enable_safe_memory: Whether to enable CPU embedding offloading and chunked projection
            to prevent GPU OOM on 6GB VRAM GPUs (GTX 1660 SUPER).

    Returns:
        Configured Patchcore model instance.
    """
    if layers is None:
        layers = ["layer2", "layer3"]

    model = Patchcore(
        backbone=backbone,
        layers=layers,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
        pre_trained=pre_trained,
    )

    if enable_safe_memory and hasattr(model, "model"):
        torch_model = model.model
        torch_model._orig_forward = torch_model.forward
        torch_model._orig_subsample_embedding = torch_model.subsample_embedding

        torch_model.forward = types.MethodType(_safe_forward, torch_model)
        torch_model.subsample_embedding = types.MethodType(_safe_subsample_embedding, torch_model)

    return model
