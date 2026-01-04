import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualCentroidAlignment(nn.Module):
    """
    Implements the Attention-based Pooling mechanism (Eq. 5).
    Aggregates spatial visual tokens into a global visual centroid (c_v).
    """
    def __init__(self, visual_dim, hidden_dim):
        super().__init__()
        # W_v in the paper (d x d)
        self.W_v = nn.Linear(visual_dim, hidden_dim, bias=False)
        # w_a in the paper (d)
        self.w_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, visual_tokens):
        """
        Args:
            visual_tokens: (Batch, L_v, D)
        Returns:
            c_v: (Batch, D)
        """
        # 1. Project tokens: tanh(W_v * v_j)
        # Shape: (Batch, L_v, Hidden)
        projected = torch.tanh(self.W_v(visual_tokens))
        
        # 2. Calculate attention scores: w_a^T * ...
        # Shape: (Batch, L_v, 1) -> (Batch, L_v)
        attn_logits = self.w_a(projected).squeeze(-1)
        
        # 3. Softmax to get alpha_j
        attn_weights = F.softmax(attn_logits, dim=1)
        
        # 4. Weighted sum: sum(alpha_j * v_j)
        # (Batch, 1, L_v) @ (Batch, L_v, D) -> (Batch, 1, D)
        c_v = torch.bmm(attn_weights.unsqueeze(1), visual_tokens).squeeze(1)
        
        return c_v

class AdaptiveGatedFusion(nn.Module):
    """
    Implements the Adaptive Gating Module (Eq. 11 & 12).
    Dynamically weighs redundant and complementary components.
    """
    def __init__(self, input_dim):
        super().__init__()
        # W_g in Eq. 11. Input is concatenation of t_para + t_perp (2 * dim)
        self.gate_proj = nn.Linear(input_dim * 2, input_dim)
        
    def forward(self, t_para, t_perp, redundancy_score):
        """
        Args:
            t_para: Redundant component (Batch, D)
            t_perp: Complementary component (Batch, D)
            redundancy_score: Scalar rho (Batch, 1)
        Returns:
            z_final: Fused representation (Batch, 2*D + 1)
        """
        # Eq. 11: g = sigmoid(W_g [t_para; t_perp] + b_g)
        combined = torch.cat([t_para, t_perp], dim=-1)
        g = torch.sigmoid(self.gate_proj(combined))
        
        # Eq. 12: z_f = Concat(t_tilde, g * t_para, (1-g) * t_perp, rho)
        # Note: The paper says `z_f = Concat(t_tilde, ...)` but typically 
        # fusing the decomposed parts is sufficient. 
        # Here I implement the exact concatenation described in Eq 12.
        
        # Reconstruct original t (t_tilde = t_para + t_perp roughly)
        t_original = t_para + t_perp 
        
        weighted_para = g * t_para
        weighted_perp = (1 - g) * t_perp
        
        z_final = torch.cat([
            t_original,      # D
            weighted_para,   # D
            weighted_perp,   # D
            redundancy_score # 1
        ], dim=-1)
        
        return z_final
