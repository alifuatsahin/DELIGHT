from model.vae import VAE
from default_config import cfg

import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(cfg, mode='training').to(device)
    model.eval()  # Set the model to evaluation mode

    B = 2
    input_dim = cfg.model.input_dim
    N = 2048  # Number of points, can be adjusted as needed

    p = torch.randn(B, input_dim, N, device=device)  # On CUDA if available
    g = torch.randn(B, input_dim, N, device=device)
    output_encoder, output_decoder, mixture_weights_logits = model(p, g)

    # print(f"Output Encoder: {output_encoder}")
    # print(f"Output Decoder: {output_decoder}")
    # print(f"Mixture Weights Logits: {mixture_weights_logits}")
    print(f"Shape of the output encoder: {output_encoder['g_posterior_samples'].shape}")
    for i, t in enumerate(output_decoder[0]['p_prior_samples']):
        print(f"Shape of p_prior_samples[{i}]: {t.shape}")
    print(f"Shape of the mixture weights logits: {mixture_weights_logits.shape}")