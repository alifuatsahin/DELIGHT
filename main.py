from models import VAE, DDPM
from default_config import cfg

import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Initialize the model
    model = DDPM(cfg).to(device)

    # 2. Prepare your input
    batch_size = 4
    x = torch.randn(batch_size, 1, 512).to(device)  # [B, C, sequence_len]

    # 3. Prepare timesteps
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)  # [B]

    # 4. Forward pass
    output = model(x, timesteps)
    print(output.shape)  # Should be [batch_size, 512, 1]