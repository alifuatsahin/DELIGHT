from models.vae import VAE
from default_config import cfg
from utils.eval_helper import compute_NLL_metric

import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(cfg).to(device)
    model.eval()  # Set the model to evaluation mode

    B = 10
    input_dim = cfg.model.input_dim
    N = 2048  # Number of points, can be adjusted as needed

    p = torch.randn(B, N, input_dim, device=device)  # On CUDA if available
    g = torch.randn(B, N, input_dim, device=device).permute(0, 2, 1)  # Change to (B, input_dim, N)
    labels = torch.randint(0, 4, (B, N), device=device)  # Random labels for testing
    samples, labels = model.recont(g)

    # _, samples, labels, mixture_weights_logits = model.sample(n_sampled_points=N*2, n_samples=B)
    # print(f"Shape of the input point cloud: {p.shape}")
    # print(f"Shape of the generated samples: {samples.shape}")
    # print(f"Shape of the labels: {labels.shape}")
    # print(f"Shape of the mixture weights logits: {mixture_weights_logits.shape}")    

    print(f'Samples Shape: {samples.shape}, Labels Shape: {labels.shape}')

    # p = p.permute(0, 2, 1)  # Change to (B, N, input_dim)
    # g = g.permute(0, 2, 1)  # Change to
    # results = compute_NLL_metric(g, p, labels, device, batch_size=B, step=0, tag='test')

    # print("Results:", results)