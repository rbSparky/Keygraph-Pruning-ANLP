import math
import torch
import torch.nn.functional as F
import faiss

# ---------- Torch-IVF helpers (ANN) ----------

@torch.no_grad()
def _faiss_ivf_flat_neighbors(
    phi: torch.Tensor,              # [N, r], arbitrary dtype/device
    tau: float,
    k: int,
    params: dict,
):
    """
    FAISS IVF-Flat (GPU if available). Cosine via L2-normalize + inner product.
    Returns (neighbors_idx [N,k_eff], neighbors_sim [N,k_eff]) on phi.device/dtype.
    Fallbacks: if FAISS or CUDA unavailable, raises RuntimeError (caller will route).
    """

    device = phi.device
    N, r = phi.shape
    k_eff = max(0, min(k, max(0, N - 1)))
    if N == 0 or k_eff == 0:
        return (torch.full((N, 0), -1, dtype=torch.long, device=device),
                torch.full((N, 0), float("-inf"), dtype=phi.dtype, device=device))

    # Defaults (match our Torch-IVF heuristics)
    nlist = params.get("nlist", None)
    if nlist is None:
        base  = max(1, int(math.sqrt(max(1, N))))
        scale = math.sqrt(max(1, r) / 16.0)   # more lists for higher r, fewer for small r
        # round to power of two, clamp
        nlist = 1
        while nlist < max(16, int(base * scale)):
            nlist <<= 1
        nlist = min(nlist, 1024)
    nprobe = int(params.get("nprobe", 16))
    nprobe = max(1, min(nprobe, nlist))

    train_subset = int(params.get("train_subset", min(N, 4096)))
    seed = int(params.get("seed", 42))

    # Normalize to cosine and cast to float32 (FAISS expects float32)
    # Keep data contiguous to avoid copies.
    phi_norm = torch.nn.functional.normalize(phi.to(dtype=torch.float32), dim=1)
    # We’ll build & search on GPU if possible.
    use_gpu = torch.cuda.is_available()
    cuda_dev = torch.cuda.current_device() if use_gpu else None

    # ---- Build IVF-Flat index (quantizer = FlatIP) ----
    # Create CPU index first, then move to GPU (FAISS canonical)
    quantizer = faiss.IndexFlatIP(r)  # inner product quantizer (cosine because unit-norm)
    cpu_index = faiss.IndexIVFFlat(quantizer, r, nlist, faiss.METRIC_INNER_PRODUCT)

    # Training data (subset for speed)
    if N <= train_subset:
        train_x = phi_norm
    else:
        g = torch.Generator(device="cpu"); g.manual_seed(seed)
        idx = torch.randperm(N, generator=g)[:train_subset].to(phi_norm.device)
        train_x = phi_norm.index_select(0, idx)

    if use_gpu:
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexIVFFlatConfig()
        cfg.device = cuda_dev
        gpu_index = faiss.index_cpu_to_gpu(res, cfg.device, cpu_index, cfg)
        # Train & add
        gpu_index.train(train_x.detach().contiguous().cpu().numpy() if not train_x.is_cuda
                        else train_x.detach().contiguous().cpu().numpy())  # conservative: numpy host
        # NOTE: For simplicity/portability we pass host numpy; FAISS uploads internally.
        #       If you want zero-copy Torch->FAISS later, we can use faiss.contrib.torch_utils.
        gpu_index.setNumProbes(nprobe)

        # Add DB
        gpu_index.add(phi_norm.detach().contiguous().cpu().numpy()
                      if not phi_norm.is_cuda else phi_norm.detach().contiguous().cpu().numpy())

        # Search with self-exclusion: get k+1 and drop self where present
        topk = min(k_eff + 1, max(1, N))  # at least 1
        D, I = gpu_index.search(phi_norm.detach().contiguous().cpu().numpy()
                                if not phi_norm.is_cuda else phi_norm.detach().contiguous().cpu().numpy(),
                                topk)
    else:
        # CPU fallback (still correct; slower). This is hit only if CUDA not available.
        cpu_index.train(train_x.detach().contiguous().cpu().numpy())
        cpu_index.nprobe = nprobe
        cpu_index.add(phi_norm.detach().contiguous().cpu().numpy())
        topk = min(k_eff + 1, max(1, N))
        D, I = cpu_index.search(phi_norm.detach().contiguous().cpu().numpy(), topk)

    # Convert outputs to torch on original device/dtype
    I = torch.from_numpy(I)  # [N, topk], int64
    D = torch.from_numpy(D)  # [N, topk], float32 (inner product)

    # Remove self (Q=DB). For each row, drop where I[row]==row, then take top k_eff.
    row_ids = torch.arange(N, dtype=I.dtype).unsqueeze(1)
    self_mask = (I == row_ids)
    # Build filtered top-k per row
    # Strategy: for each row, make a mask to keep the first k_eff non-self entries
    keep = ~self_mask
    # Guard: if no self present and topk == k_eff, this is a no-op
    # We still need exactly k_eff columns; pad with -1/-inf if fewer.
    idx_kept_list = []
    dist_kept_list = []
    for r_i in range(N):
        row_i = I[r_i][keep[r_i]]
        sim_i = D[r_i][keep[r_i]]
        if row_i.numel() >= k_eff:
            idx_kept_list.append(row_i[:k_eff])
            dist_kept_list.append(sim_i[:k_eff])
        else:
            pad = k_eff - row_i.numel()
            if pad > 0:
                row_i = torch.cat([row_i, -torch.ones(pad, dtype=row_i.dtype)], dim=0)
                sim_i = torch.cat([sim_i, torch.full((pad,), float("-inf"), dtype=sim_i.dtype)], dim=0)
            idx_kept_list.append(row_i)
            dist_kept_list.append(sim_i)

    neighbors_idx = torch.stack(idx_kept_list, dim=0).to(device=device, dtype=torch.long)
    neighbors_sim = torch.stack(dist_kept_list, dim=0).to(device=device, dtype=phi.dtype)

    # Apply tau threshold (cosine ~ inner product on unit-norm)
    if tau is not None and tau > float("-inf"):
        valid = neighbors_sim >= float(tau)
        neighbors_idx = torch.where(valid, neighbors_idx, torch.full_like(neighbors_idx, -1))
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=neighbors_sim.dtype)
        neighbors_sim = torch.where(valid, neighbors_sim, torch.full_like(neighbors_sim, neg_inf))

    return neighbors_idx, neighbors_sim


def _round_pow2(x: int, lo: int = 16, hi: int = 1024) -> int:
    x = max(lo, min(hi, x))
    p = 1
    while p < x:
        p <<= 1
    return p

def _round_pow2_down(x: int, lo: int = 1) -> int:
    x = max(lo, x)
    p = 1
    while (p << 1) <= x:
        p <<= 1
    return p


@torch.no_grad()
def _kmeans_train(phi: torch.Tensor, nlist: int, iters: int = 4, train_subset: int = 4096, seed: int = 42):
    """
    K-means (cosine) on GPU.
    phi: [N, r], assumed *unit-normalized* (we normalize again defensively).
    Returns unit-norm centroids [nlist, r] in phi.dtype/device.
    Accumulations are done in fp32 for stability.
    """
    device, dt = phi.device, phi.dtype
    N, r = phi.shape
    phi = F.normalize(phi, p=2, dim=1)

    # choose training subset
    if N <= train_subset:
        idx = torch.arange(N, device=device)
    else:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        # randperm on CPU to avoid CUDA generator constraints
        idx = torch.randperm(N, generator=g)[:train_subset].to(device)
    X = phi.index_select(0, idx)  # [Ns, r]
    Ns = X.shape[0]

    # --- kmeans++ init (cosine): first random, then by prob ~ (1 - max_sim)
    g2 = torch.Generator(device="cpu"); g2.manual_seed(seed + 1)
    first = idx[torch.randint(0, Ns, (1,), generator=g2).item()]
    C = phi.new_empty((nlist, r))
    C[0] = phi[first]
    # distance proxy: 1 - cosine to nearest chosen centroid
    dmin = (1 - (X @ C[0].unsqueeze(1)).squeeze(1)).clamp_min(0)  # [Ns]
    for c in range(1, nlist):
        probs = dmin / (dmin.sum() + 1e-8)
        # multinomial on CPU probs for determinism across setups
        pick = torch.multinomial(probs.cpu(), 1).item()
        C[c] = X[pick]
        # update dmin
        sim = (X @ C[c].unsqueeze(1)).squeeze(1)
        dmin = torch.minimum(dmin, (1 - sim).clamp_min(0))

    # --- Lloyd iterations (few iters), fp32 accum
    C = F.normalize(C, p=2, dim=1).to(device=device, dtype=dt)
    for _ in range(max(1, iters)):
        # assign training subset
        sims = X @ C.T  # [Ns, nlist]
        labels = sims.argmax(dim=1)  # [Ns]
        # accumulators
        sums = torch.zeros((nlist, r), device=device, dtype=torch.float32)
        counts = torch.bincount(labels, minlength=nlist).to(torch.long)  # [nlist]
        sums.index_add_(0, labels, X.to(torch.float32))
        # handle empty clusters by reseeding to random points
        empty = (counts == 0).nonzero(as_tuple=True)[0]
        if empty.numel():
            # pick random training points to replace empty centroids
            repl = idx[torch.randint(0, Ns, (empty.numel(),), generator=g2)]
            C[empty] = phi.index_select(0, repl)
            counts[empty] = 1
            sums[empty] = C[empty].to(torch.float32)

        C = F.normalize((sums / counts.clamp_min(1).view(-1, 1)).to(dtype=dt), p=2, dim=1)

    return C  # [nlist, r], unit-norm


@torch.no_grad()
def _ivf_build_invlists(phi: torch.Tensor, centroids: torch.Tensor):
    """
    Build inverted lists (CSR) for full dataset.
    Returns: list_ptr [nlist+1], list_idx [N], assign [N]
    """
    device = phi.device
    phi = F.normalize(phi, p=2, dim=1)
    sims = phi @ centroids.T  # [N, nlist]
    assign = sims.argmax(dim=1)  # [N] long
    nlist = centroids.shape[0]
    counts = torch.bincount(assign, minlength=nlist)  # [nlist]
    list_ptr = torch.empty(nlist + 1, dtype=torch.long, device=device)
    list_ptr[0] = 0
    list_ptr[1:] = torch.cumsum(counts, dim=0)

    # stable group-by using sort on (assign, idx)
    idx = torch.arange(phi.shape[0], device=device, dtype=torch.long)
    keys = assign * (phi.shape[0] + 1) + idx  # make (assign, idx) unique
    order = torch.argsort(keys)  # [N]
    list_idx = idx[order]        # [N]

    return list_ptr, list_idx, assign, sims  # sims reused for probe selection


@torch.no_grad()
def _ivf_select_probes(sims_centroid: torch.Tensor, nprobe: int):
    """
    sims_centroid: [N, nlist] = phi @ centroids.T
    Returns: probe_idx [N, nprobe_eff] long
    """
    N, nlist = sims_centroid.shape
    nprobe_eff = min(nprobe, nlist)
    if nprobe_eff <= 0:
        return sims_centroid.new_zeros((N, 0), dtype=torch.long)
    _, probe_idx = torch.topk(sims_centroid, k=nprobe_eff, dim=1)
    return probe_idx


@torch.no_grad()
def _ivf_knn_from_probes(
    phi: torch.Tensor,                # [N, r], unit-norm
    list_ptr: torch.Tensor,           # [nlist+1]
    list_idx: torch.Tensor,           # [N]
    probe_idx: torch.Tensor,          # [N, nprobe_eff]
    k: int,
    tau: float,
    self_ids: torch.Tensor | None = None,
    block_max_dots: int = 200_000,
):
    """
    Vectorized per-centroid candidate scoring with top-k merge.
    Returns neighbors_idx [N, k_eff], neighbors_sim [N, k_eff]
    """
    device, dt = phi.device, phi.dtype
    N = phi.shape[0]
    if N == 0:
        return (torch.empty(0, 0, dtype=torch.long, device=device),
                torch.empty(0, 0, dtype=dt, device=device))
    k_eff = max(0, min(k, N - 1))
    if k_eff == 0:
        return (torch.full((N, 0), -1, dtype=torch.long, device=device),
                torch.full((N, 0), float("-inf"), dtype=dt, device=device))

    nlist = list_ptr.numel() - 1
    neighbors_idx = torch.full((N, k_eff), -1, dtype=torch.long, device=device)
    neighbors_sim = torch.full((N, k_eff), float("-inf"), dtype=dt, device=device)

    # Pre-alloc reused structures
    # For each centroid, find queries that probe it
    for c in range(nlist):
        # members of centroid c
        start, end = list_ptr[c].item(), list_ptr[c + 1].item()
        if end <= start:
            continue
        members = list_idx[start:end]  # [m_c]
        m_c = members.numel()
        # queries that probe c
        q_mask = (probe_idx == c).any(dim=1)
        if not q_mask.any():
            continue
        q_idx = q_mask.nonzero(as_tuple=True)[0]  # [q_c]
        Q = phi.index_select(0, q_idx)            # [q_c, r]

        # chunk over members to bound memory
        # limit: q_c * m_block <= block_max_dots
        q_c = Q.shape[0]
        if q_c == 0:
            continue
        m_block = max(1, block_max_dots // max(1, q_c))
        for mb in range(0, m_c, m_block):
            mb_end = min(m_c, mb + m_block)
            mem_blk = members[mb:mb_end]             # [m_b]
            K = phi.index_select(0, mem_blk)         # [m_b, r]
            scores = Q @ K.T                          # [q_c, m_b]

            # self-exclusion if Q == DB
            if self_ids is not None:
                # mask where q_idx[i] == mem_blk[j]
                selfmask = (q_idx.view(-1, 1) == mem_blk.view(1, -1))
                if selfmask.any():
                    scores = scores.masked_fill(selfmask, float("-inf"))

            # merge into existing top-k for these q_idx rows
            prev_scores = neighbors_sim.index_select(0, q_idx)  # [q_c, k_eff]
            prev_idx    = neighbors_idx.index_select(0, q_idx)  # [q_c, k_eff]
            cand_idx    = mem_blk.view(1, -1).expand(q_c, -1)   # [q_c, m_b]

            all_scores = torch.cat([prev_scores, scores.to(dt)], dim=1)
            all_idx    = torch.cat([prev_idx,    cand_idx],     dim=1)
            top_scores, top_pos = torch.topk(all_scores, k=k_eff, dim=1)
            # gather idx by positions
            gather_idx = torch.gather(all_idx, 1, top_pos)
            # write back
            neighbors_sim.index_copy_(0, q_idx, top_scores)
            neighbors_idx.index_copy_(0, q_idx, gather_idx)

    # threshold by tau
    if tau is not None and tau > float("-inf"):
        valid = neighbors_sim >= float(tau)
        neighbors_idx = torch.where(valid, neighbors_idx, torch.full_like(neighbors_idx, -1))
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=dt)
        neighbors_sim = torch.where(valid, neighbors_sim, torch.full_like(neighbors_sim, neg_inf))

    return neighbors_idx, neighbors_sim


@torch.no_grad()
def _torch_ivf_neighbors(phi: torch.Tensor, tau: float, k: int, params: dict):
    """
    End-to-end IVF-Flat: train centroids, build lists, select probes, score candidates, return neighbors.
    """
    N, r = phi.shape
    device = phi.device
    phi = F.normalize(phi, p=2, dim=1)

    # defaults
    nlist = params.get("nlist", None)
    if nlist is None:
        base  = max(1, int(math.sqrt(max(1, N))))
        scale = math.sqrt(max(1, r) / 16.0)  # fewer lists for small r
        nlist = _round_pow2(int(max(1, base * scale)))
    nlist = max(16, min(1024, nlist))

    nprobe = int(params.get("nprobe", 16))
    nprobe = max(1, min(nprobe, nlist))

    # Enforce minimum candidate coverage for very low dims
    # r<=8 needs higher coverage; aim for >=0.75 coverage => nlist <= floor_pow2(nprobe / 0.75)
    if r <= 8:
        min_cov = 0.75
        target_max_nlist = max(1, int(nprobe / min_cov))  # e.g., 16 / 0.75 = 21 -> floor_pow2 -> 16
        nlist_cap = _round_pow2_down(target_max_nlist, lo=1)
        if nlist > nlist_cap:
            nlist = max(16, nlist_cap)
            # clamp nprobe again just in case
            nprobe = min(nprobe, nlist)

    iters = int(params.get("kmeans_iters", 4))
    if "kmeans_iters" not in params and r <= 8:
        iters = 6  # a bit more stable at tiny r

    train_subset   = int(params.get("train_subset", min(N, 4096)))
    seed           = int(params.get("seed", 42))
    block_max_dots = int(params.get("block_max_dots", 200_000))

    # train
    centroids = _kmeans_train(phi, nlist=nlist, iters=iters, train_subset=train_subset, seed=seed)  # [nlist, r]

    # lists + assign + centroid sims
    list_ptr, list_idx, assign, sims_centroid = _ivf_build_invlists(phi, centroids)

    # probes for all queries
    probe_idx = _ivf_select_probes(sims_centroid, nprobe=nprobe)  # [N, nprobe_eff]

    # neighbors via candidate scoring
    neighbors_idx, neighbors_sim = _ivf_knn_from_probes(
        phi, list_ptr, list_idx, probe_idx,
        k=k, tau=tau, self_ids=torch.arange(N, device=device, dtype=torch.long),
        block_max_dots=block_max_dots
    )
    return neighbors_idx, neighbors_sim
