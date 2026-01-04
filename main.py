import torch
from model import UMPIRE
from loss import UMPIRELoss

def main():
    # Hyperparameters
    BATCH_SIZE = 8
    FEATURE_DIM = 1024  # Assuming Qwen2-VL dimension
    SEQ_LEN_IMG = 196   # 14x14 patches
    SEQ_LEN_TXT = 50    # Caption length
    
    # 1. Instantiate Model and Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UMPIRE(feature_dim=FEATURE_DIM, projection_dim=512).to(device)
    criterion = UMPIRELoss(lambda_reg=0.5, tau_high=0.8, tau_low=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # 2. Simulate Data (In real usage, these come from the Multimodal Backbone)
    # y=1 (LLM), y=0 (Human)
    visual_feats = torch.randn(BATCH_SIZE, SEQ_LEN_IMG, FEATURE_DIM).to(device)
    text_feats = torch.randn(BATCH_SIZE, SEQ_LEN_TXT, FEATURE_DIM).to(device)
    labels = torch.randint(0, 2, (BATCH_SIZE, 1)).float().to(device)
    
    print("--- Training Step Simulation ---")
    model.train()
    
    # 3. Forward Pass
    preds, rho = model(visual_feats, text_feats)
    
    # 4. Calculate Loss
    loss, l_cls, l_lcrr = criterion(preds, labels, rho)
    
    # 5. Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Redundancy Scores (rho): {rho.squeeze().detach().cpu().numpy()}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"  - BCE Loss: {l_cls.item():.4f}")
    print(f"  - Geo Reg Loss: {l_lcrr.item():.4f}")

if __name__ == "__main__":
    main()
