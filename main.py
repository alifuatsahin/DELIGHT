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
    g = torch.randn(B, N, input_dim, device=device)
    labels = torch.randint(0, 4, (B, N), device=device)  # Random labels for testing
    output_encoder, output_decoder, mixture_weights_logits = model(p, g)

    # _, samples, labels, mixture_weights_logits = model.sample(n_sampled_points=N*2, n_samples=B)
    # print(f"Shape of the input point cloud: {p.shape}")
    # print(f"Shape of the generated samples: {samples.shape}")
    # print(f"Shape of the labels: {labels.shape}")
    # print(f"Shape of the mixture weights logits: {mixture_weights_logits.shape}")    

    print(f"Shape of the output encoder: {output_encoder['g_posterior_samples'].shape}")
    for i, t in enumerate(output_decoder[0]['p_prior_samples']):
        print(f"Shape of p_prior_samples[{i}]: {t.shape}")
    print(f"Shape of the mixture weights logits: {mixture_weights_logits.shape}")

    # p = p.permute(0, 2, 1)  # Change to (B, N, input_dim)
    # g = g.permute(0, 2, 1)  # Change to
    # results = compute_NLL_metric(g, p, labels, device, batch_size=B, step=0, tag='test')

    # print("Results:", results)