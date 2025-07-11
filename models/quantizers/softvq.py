import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Quantizer(nn.Module):
    def __init__(self, cfg, input_dim):
        super().__init__()
        self.n_e = cfg.model.soft_vq.n_e
        self.e_dim = cfg.model.latent_dim

        self.num_codebooks = cfg.model.num_codebooks
        self.learnable = cfg.model.soft_vq.learnable
        self.tau_min = cfg.model.soft_vq.tau_min
        self.tau_max = cfg.model.soft_vq.tau_max
        self.initial_tau = cfg.model.soft_vq.tau
        if self.learnable:
            self.log_tau = nn.Parameter(
                torch.log(torch.tensor(self.initial_tau, dtype=torch.float32)), 
                requires_grad=True
            )

        self.entropy_loss_ratio = cfg.model.soft_vq.entropy_loss_ratio
        self.show_usage = cfg.model.soft_vq.show_usage
        self.l2_norm = cfg.model.soft_vq.l2_norm

        self.mlp = nn.Linear(input_dim, self.e_dim)  # MLP to project input to codebook size
        
        # Initialize embedding as a learnable parameter
        self.embedding = nn.Parameter(torch.randn(self.n_e, self.e_dim))
        self.embedding.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        if self.l2_norm:
            self.embedding.data = F.normalize(self.embedding.data, p=2, dim=-1)

        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(65536))

    @property
    def tau(self):
        if self.learnable:
            return torch.clamp(torch.exp(self.log_tau), min=self.tau_min, max=self.tau_max)  # Ensure tau is positive and bounded
        else:
            return self.initial_tau

    def forward(self, features):
        """
        Args:
            z: Input tensor of shape (B, latent_dim).
        """

        assert features.dim() == 2, f"Expected input shape (B, input_dim), got {features.shape}"

        z = self.mlp(features)

        batch_size, embedding_dim = z.shape
        assert embedding_dim == self.e_dim, f"Expected input dimension {self.e_dim}, got {embedding_dim}"

        embedding = F.normalize(self.embedding.clone(), p=2, dim=-1)  # Add .clone()

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)

        logits = torch.einsum('be, ne -> bn', z, embedding.detach())  # Compute logits

        current_tau = self.tau
        probs = F.softmax(logits / current_tau, dim=-1)  # Compute probabilities

        z_q = torch.einsum('bn, ne -> be', probs, embedding)  # Quantize input
        z_q = z_q.view(batch_size, self.e_dim)  # Reshape back to original input shape

        # Calculate cosine similarity
        with torch.no_grad():
            zq_z_cos = F.cosine_similarity(z, z_q, dim=-1).mean()

        indices = torch.argmax(probs, dim=-1)  # Get indices of quantized vectors

        # Track codebook usage
        if self.show_usage and self.training:
            cur_len = indices.size(0)
            self.codebook_used[:-cur_len].copy_(self.codebook_used[cur_len:].clone())
            self.codebook_used[-cur_len:].copy_(indices)

        codebook_usage = torch.tensor([
            len(torch.unique(self.codebook_used[k])) / self.n_e 
            for k in range(self.num_codebooks)
        ]).mean() if self.show_usage else 0

        entropy_loss = self.entropy_loss_ratio * self.compute_entropy_loss(logits.view(-1, self.n_e), current_tau)

        avg_probs = torch.mean(probs, dim=-1).mean()  # Average probabilities
        max_probs = torch.max(probs, dim=-1)[0].mean()  # Maximum probabilities

        info = {
            "avg_probs": avg_probs,
            "max_probs": max_probs,
            "z_cos": zq_z_cos,
            "tau": current_tau.item() if self.learnable else current_tau,
            "codebook_usage": codebook_usage
        }

        return z_q, entropy_loss, info

    def compute_entropy_loss(self, affinity, tau=None):
        if tau is None:
            tau = self.tau
        flat_affinity = affinity.reshape(-1, affinity.shape[-1])
        flat_affinity = flat_affinity / tau
        probs = F.softmax(flat_affinity, dim=-1)
        log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)

        target_probs = probs

        avg_probs = torch.mean(target_probs, dim=0)
        avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
        sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
        loss = sample_entropy - avg_entropy
        
        return loss
    
    @torch.no_grad()
    def sample(self, batch_size, device=None):
        """Sample random codes from the codebook"""
        if device is None:
            device = self.device

        # Sample random indices
        indices = torch.randint(0, self.n_e, (batch_size,), device=device)
        
        # Get corresponding embeddings
        embedding = F.normalize(self.embedding, p=2, dim=-1) if self.l2_norm else self.embedding
        z_sampled = embedding[indices]  # [batch_size, e_dim]
        
        return z_sampled