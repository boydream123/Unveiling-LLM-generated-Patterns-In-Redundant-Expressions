import torch
import torch.nn as nn

class UMPIRELoss(nn.Module):
    """
    Combined loss: Binary Cross Entropy + L_LCRR (Eq. 16)
    """
    def __init__(self, lambda_reg=0.5, tau_high=0.8, tau_low=0.2):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.tau_high = tau_high  # Margin for LLM generated (High redundancy)
        self.tau_low = tau_low    # Margin for Human written (Low redundancy)
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions, labels, redundancy_scores):
        """
        Args:
            predictions: Model output probs (Batch, 1)
            labels: 1 for LLM, 0 for Human (Batch, 1)
            redundancy_scores: rho values (Batch, 1)
        """
        # 1. Classification Loss (L_cls - Eq. 14)
        l_cls = self.bce_loss(predictions, labels)
        
        # 2. Latent Contrastive Redundancy Regularization (L_LCRR - Eq. 15)
        # For LLM (y=1): max(0, tau_high - rho) -> Encourage rho > tau_high
        loss_llm = labels * torch.relu(self.tau_high - redundancy_scores)
        
        # For Human (y=0): max(0, rho - tau_low) -> Encourage rho < tau_low
        loss_human = (1 - labels) * torch.relu(redundancy_scores - self.tau_low)
        
        l_lcrr = torch.mean(loss_llm + loss_human)
        
        # 3. Total Loss (Eq. 16)
        total_loss = l_cls + self.lambda_reg * l_lcrr
        
        return total_loss, l_cls, l_lcrr
