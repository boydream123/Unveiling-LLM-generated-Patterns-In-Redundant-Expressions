import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import VisualCentroidAlignment, AdaptiveGatedFusion

class UMPIRE(nn.Module):
    def __init__(self, feature_dim=1024, projection_dim=512):
        super().__init__()
        
        # --- 1. Alignment Module ---
        self.visual_alignment = VisualCentroidAlignment(feature_dim, feature_dim)
        
        # Projection Heads (phi_v and phi_t in Eq. 7)
        # Maps backbone features to detection-specific metric space
        self.phi_v = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.phi_t = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # --- 2. Gated Fusion ---
        self.fusion = AdaptiveGatedFusion(projection_dim)
        
        # --- 3. Classifier (Eq. 13) ---
        # Input dim is 3*D + 1 based on Eq. 12 (Original + Weighted Para + Weighted Perp + Score)
        classifier_input_dim = 3 * projection_dim + 1
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )

    def orthogonal_decomposition(self, v_embed, t_embed):
        """
        Performs Eq. 8, 9, 10.
        v_embed: Normalized visual centroid (Batch, D)
        t_embed: Normalized textual summary (Batch, D)
        """
        # Eq. 8: Redundant Component (Projection of t onto v)
        # dot product (b, 1, d) @ (b, d, 1) -> (b, 1, 1)
        dot_prod = torch.bmm(t_embed.unsqueeze(1), v_embed.unsqueeze(2)).squeeze(2)
        t_para = dot_prod * v_embed # (Batch, D)
        
        # Eq. 9: Complementary Component (Residual)
        t_perp = t_embed - t_para
        
        # Eq. 10: CMR Score (Magnitude of projection)
        # Since v is normalized, the magnitude is just the absolute dot product
        # However, paper defines rho = ||t_para||. 
        rho = torch.norm(t_para, p=2, dim=-1, keepdim=True)
        
        return t_para, t_perp, rho

    def forward(self, visual_tokens, textual_tokens):
        """
        Args:
            visual_tokens: Output from Backbone Encoder (Batch, L_v, D)
            textual_tokens: Output from Backbone Decoder (Batch, L_t, D)
        Returns:
            prediction: Probability (Batch, 1)
            redundancy_score: (Batch, 1) for Loss calculation
        """
        
        # 1. Feature Extraction & Alignment
        # Visual Centroid (Eq. 5)
        c_v = self.visual_alignment(visual_tokens)
        
        # Textual Mean Pooling (Eq. 6)
        h_t = torch.mean(textual_tokens, dim=1)
        
        # Project to detection space (Eq. 7)
        v_proj = self.phi_v(c_v)
        t_proj = self.phi_t(h_t)
        
        # L2 Normalization (Eq. 7)
        v_norm = F.normalize(v_proj, p=2, dim=-1)
        t_norm = F.normalize(t_proj, p=2, dim=-1)
        
        # 2. Orthogonal Semantic Decomposition (Eq. 8-10)
        t_para, t_perp, rho = self.orthogonal_decomposition(v_norm, t_norm)
        
        # 3. Adaptive Gated Fusion (Eq. 11-12)
        z_final = self.fusion(t_para, t_perp, rho)
        
        # 4. Classification (Eq. 13)
        prediction = self.classifier(z_final)
        
        return prediction, rho
