"""
GalarModel — Dual-branch sequence model for GI endoscopy event detection.

Two branches share the same input feature sequence [B, T, feat_dim]:
  - Anatomy branch:   Windowed Attention → Dual GCN → Video GPS → Bi-Mamba → 8 classes
  - Pathology branch: Deviation signal + Dual GCN + Conv + Mamba → 9 classes

The pathology branch uses normal_patch_prototypes (per-anatomy EMA of healthy frames)
to compute a deviation signal before GCN processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layers import GraphConvolution, DistanceAdj
from mamba_ssm import Mamba


ANATOMY_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve",
]
PATHOLOGY_LABELS = [
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer",
]


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class WindowedAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_model * 4), QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop = nn.Dropout(dropout)
        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        m = self.attn_mask if self.attn_mask is not None else None
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=m)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class GalarModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        feat_dim   = config["model"]["feat_dim"]
        cls_dim    = feat_dim // 2
        patch_dim  = feat_dim // 2
        hidden_dim = config["model"]["hidden_dim"]
        n_heads    = config["model"]["n_heads"]
        n_layers   = config["model"]["n_layers"]
        window     = config["model"]["attn_window"]
        max_len    = config["model"]["max_seq_len"]
        dropout    = config["model"].get("dropout", 0.1)
        gcn_dim    = hidden_dim // 2

        self.cls_dim            = cls_dim
        self.patch_dim          = patch_dim
        self.sim_threshold      = config["model"].get("sim_threshold",      0.7)
        self.path_sim_threshold = config["model"].get("path_sim_threshold", 0.4)

        # ── ANATOMY BRANCH ────────────────────────────────────────────────────
        self.anat_input_proj = nn.Linear(feat_dim, hidden_dim)
        self.anat_input_ln   = nn.LayerNorm(hidden_dim)
        self.pos_embed       = nn.Embedding(max_len, hidden_dim)

        self.motion_proj_cls = nn.Sequential(
            nn.Linear(cls_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        attn_mask = self._build_window_mask(max_len, window)
        self.local_blocks = nn.ModuleList([
            WindowedAttentionBlock(hidden_dim, n_heads, attn_mask, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.dist_adj = DistanceAdj()
        self.gelu     = QuickGELU()

        self.gc_sim1 = GraphConvolution(hidden_dim, gcn_dim, residual=True)
        self.gc_sim2 = GraphConvolution(gcn_dim,    gcn_dim, residual=True)
        self.gc_dis1 = GraphConvolution(hidden_dim, gcn_dim, residual=True)
        self.gc_dis2 = GraphConvolution(gcn_dim,    gcn_dim, residual=True)
        self.anatomy_gcn_proj = nn.Linear(hidden_dim, hidden_dim)

        self.video_pos_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4), QuickGELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        mamba_d_state = config["model"].get("mamba_d_state", 32)
        mamba_expand  = config["model"].get("mamba_expand",  2)
        self.mamba_fwd     = Mamba(d_model=hidden_dim, d_state=mamba_d_state, d_conv=4, expand=mamba_expand)
        self.mamba_bwd     = Mamba(d_model=hidden_dim, d_state=mamba_d_state, d_conv=4, expand=mamba_expand)
        self.boundary_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        self.anatomy_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            QuickGELU(),
            nn.Linear(hidden_dim // 2, len(ANATOMY_LABELS)),
        )

        # ── PATHOLOGY BRANCH ──────────────────────────────────────────────────
        self.patch_ln = nn.LayerNorm(patch_dim)

        self.recen_proj = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.motion_proj_patch = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Pathology dual GCN: sim-graph (hidden→gcn_dim) + dis-graph (hidden→gcn_dim)
        self.gc_p_sim      = GraphConvolution(hidden_dim, gcn_dim, residual=True)
        self.gc_p_dis      = GraphConvolution(hidden_dim, gcn_dim, residual=True)
        self.path_gcn_proj = nn.Linear(hidden_dim, hidden_dim)  # cat(gcn_dim+gcn_dim)→hidden

        self.path_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

        self.mamba_path = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=1)

        self.anatomy_cond_proj = nn.Sequential(
            nn.Linear(len(ANATOMY_LABELS), hidden_dim // 4),
            QuickGELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.pathology_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            QuickGELU(),
            nn.Linear(hidden_dim // 2, len(PATHOLOGY_LABELS)),
        )

        # ── Prototypes ────────────────────────────────────────────────────────
        self.register_buffer("anatomy_prototypes",      torch.zeros(len(ANATOMY_LABELS), hidden_dim))
        self.register_buffer("normal_patch_prototypes", torch.zeros(len(ANATOMY_LABELS), patch_dim))
        self.anatomy_proto_initialized = False
        self.normal_proto_initialized  = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_window_mask(self, max_len, window):
        mask = torch.full((max_len, max_len), float("-inf"))
        for s in range(0, max_len, window):
            e = min(s + window, max_len)
            mask[s:e, s:e] = 0.0
        return mask

    def _similarity_adj(self, x, threshold):
        x_norm  = F.normalize(x, dim=-1, eps=1e-8)
        sim     = x_norm @ x_norm.transpose(-1, -2)
        sim     = F.threshold(sim, threshold, 0.0)
        T       = sim.shape[1]
        eye     = torch.eye(T, device=sim.device).unsqueeze(0)
        sim     = sim + eye
        row_sum = sim.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return sim / row_sum

    @staticmethod
    def _motion(feat):
        diff = torch.zeros_like(feat)
        diff[:, 1:] = feat[:, 1:] - feat[:, :-1]
        return diff

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        features:        torch.Tensor,
        raw_features:    torch.Tensor = None,
        valid_len=None,
        video_pos_ratio: torch.Tensor = None,
        return_features: bool = False,
        return_all:      bool = False,
    ):
        B, T, _ = features.shape
        device  = features.device

        cls_feat  = features[:, :, :self.cls_dim]
        raw_patch = (raw_features[:, :, self.cls_dim:]
                     if raw_features is not None
                     else features[:, :, self.cls_dim:])

        is_pad = (features.abs().sum(dim=-1) == 0)

        # ═══════════════════════════════════════════════════════════════════════
        # ANATOMY BRANCH
        # ═══════════════════════════════════════════════════════════════════════
        cls_diff = self._motion(cls_feat) * (~is_pad).float().unsqueeze(-1)

        x_a = self.anat_input_ln(self.anat_input_proj(features))
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x_a = x_a + self.pos_embed(pos) + self.motion_proj_cls(cls_diff)

        for block in self.local_blocks:
            x_a = block(x_a)

        x_pre_gcn = x_a

        sim_adj = self._similarity_adj(x_a, self.sim_threshold)
        dis_adj = self.dist_adj(B, T, device)

        g_sim = self.gelu(self.gc_sim1(x_a, sim_adj))
        g_sim = self.gelu(self.gc_sim2(g_sim, sim_adj))
        g_dis = self.gelu(self.gc_dis1(x_a, dis_adj))
        g_dis = self.gelu(self.gc_dis2(g_dis, dis_adj))
        x_gcn = self.gelu(self.anatomy_gcn_proj(torch.cat([g_sim, g_dis], dim=-1)))
        x_a   = x_gcn + x_pre_gcn

        if video_pos_ratio is not None:
            vp  = self.video_pos_proj(video_pos_ratio.float().view(B, 1, 1))
            x_a = x_a + vp

        h_fwd = self.mamba_fwd(x_a)
        h_bwd = self.mamba_bwd(x_a.flip(1)).flip(1)
        x_a   = self.boundary_proj(torch.cat([x_a, h_fwd, h_bwd], dim=-1))

        anatomy_logits = self.anatomy_head(x_a)

        # ═══════════════════════════════════════════════════════════════════════
        # PATHOLOGY BRANCH
        # ═══════════════════════════════════════════════════════════════════════
        raw_patch_norm = self.patch_ln(raw_patch)

        if self.normal_proto_initialized:
            anat_w          = torch.sigmoid(anatomy_logits.detach())
            normal_expected = anat_w @ self.normal_patch_prototypes
            deviation       = raw_patch - normal_expected
        else:
            deviation = raw_patch
        x_recen = self.recen_proj(deviation)

        patch_diff = self._motion(raw_patch) * (~is_pad).float().unsqueeze(-1)
        m_pat      = self.motion_proj_patch(patch_diff)
        x_pat      = self.patch_proj(raw_patch_norm) + m_pat

        # Dual GCN: sim + dis, ADD residual
        path_sim_adj = self._similarity_adj(x_pat, self.path_sim_threshold)
        path_dis_adj = self.dist_adj(B, T, device)
        g_p_sim  = self.gelu(self.gc_p_sim(x_pat, path_sim_adj))
        g_p_dis  = self.gelu(self.gc_p_dis(x_pat, path_dis_adj))
        x_gcn_p  = self.gelu(self.path_gcn_proj(torch.cat([g_p_sim, g_p_dis], dim=-1)))
        x_pat    = x_gcn_p + x_pat

        x_conv     = self.path_conv(x_pat.permute(0, 2, 1)).permute(0, 2, 1)
        x_mamba    = self.mamba_path(x_pat)
        x_pat_temp = x_conv + x_mamba + x_pat

        x_path = self.fusion_proj(torch.cat([x_recen, m_pat, x_pat_temp], dim=-1))

        anat_probs = torch.sigmoid(anatomy_logits.detach())
        x_path     = x_path + self.anatomy_cond_proj(anat_probs)

        pathology_logits = self.pathology_head(x_path)

        if return_all:
            return anatomy_logits, pathology_logits, x_a, x_recen
        if return_features:
            return anatomy_logits, pathology_logits, x_a
        return anatomy_logits, pathology_logits

    # ── Prototype Updates ─────────────────────────────────────────────────────

    @torch.no_grad()
    def update_normal_patch_prototypes(
        self,
        patch_features:   torch.Tensor,
        anatomy_labels:   torch.Tensor,
        pathology_labels: torch.Tensor = None,
        momentum: float = 0.99,
    ):
        if pathology_labels is not None:
            healthy = pathology_labels.sum(dim=-1) == 0
        else:
            healthy = torch.ones(patch_features.shape[:2], dtype=torch.bool,
                                 device=patch_features.device)
        for i in range(len(ANATOMY_LABELS)):
            mask = (anatomy_labels[:, :, i] == 1) & healthy
            if not mask.any():
                continue
            new_proto = patch_features[mask].mean(dim=0)
            if not self.normal_proto_initialized:
                self.normal_patch_prototypes[i] = new_proto
            else:
                self.normal_patch_prototypes[i] = (
                    momentum * self.normal_patch_prototypes[i]
                    + (1 - momentum) * new_proto
                )
        self.normal_proto_initialized = True

    @torch.no_grad()
    def update_anatomy_prototypes(
        self,
        x_a:              torch.Tensor,
        anatomy_labels:   torch.Tensor,
        pathology_labels: torch.Tensor = None,
        momentum: float = 0.99,
    ):
        if pathology_labels is not None:
            healthy = pathology_labels.sum(dim=-1) == 0
        else:
            healthy = torch.ones(x_a.shape[:2], dtype=torch.bool, device=x_a.device)
        for i in range(len(ANATOMY_LABELS)):
            mask = (anatomy_labels[:, :, i] == 1) & healthy
            if not mask.any():
                continue
            new_proto = x_a[mask].mean(dim=0)
            if not self.anatomy_proto_initialized:
                self.anatomy_prototypes[i] = new_proto
            else:
                self.anatomy_prototypes[i] = (
                    momentum * self.anatomy_prototypes[i]
                    + (1 - momentum) * new_proto
                )
        self.anatomy_proto_initialized = True
