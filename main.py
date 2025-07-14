from models.vae import VAE
from default_config import cfg
from utils.eval_helper import compute_NLL_metric
# from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion

from models.encoder import Encoder

import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = 'vae_model.pth'  # Path to save the model state

    model = VAE(cfg).to(device)
    model.eval()
    # input = torch.rand((5, 2048, 3), device=device, dtype=torch.float32)  # Example input tensor

    # diff_model_config = {"target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
    #                 "params": {"dims": 1, "in_channels": 1, "model_channels": 320, "up_down_sampling": True,
    #                             "attention_resolutions": (2, 4, 8), "channel_mult": (1, 2, 4, 4), "num_res_blocks": 3}}

    # model = LatentDiffusion(diff_model_config=diff_model_config, conditioning_key=None).to(device)

    # torch.save(model, 'diffusion_model.pth')

    B = 10
    input_dim = cfg.model.input_dim
    N = 2048  # Number of points, can be adjusted as needed

    p = torch.randn(B, N, input_dim, device=device)  # On CUDA if available
    g = torch.randn(B, N, input_dim, device=device)  # Change to (B, input_dim, N)
    labels = torch.randint(0, 4, (B, N), device=device)  # Random labels for testing
    output = model(p, g)

    # print(f"Shape of the input point cloud: {p.shape}")
    # print(f"Generated output: {output}")

    # _, samples, labels, mixture_weights_logits = model.sample(n_sampled_points=N*2, n_samples=B)
    # print(f"Shape of the input point cloud: {p.shape}")
    # print(f"Shape of the generated samples: {samples.shape}")
    # print(f"Shape of the labels: {labels.shape}")
    # print(f"Shape of the mixture weights logits: {mixture_weights_logits.shape}")    

    # print(f'Losses: {output}')

    # p = p.permute(0, 2, 1)  # Change to (B, N, input_dim)
    # g = g.permute(0, 2, 1)  # Change to
    # results = compute_NLL_metric(g, p, labels, device, batch_size=B, step=0, tag='test')

    # print("Results:", results)