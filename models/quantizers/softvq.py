import torch 
import torch.nn as nn 
import torch.nn.functional as F

from modules.pvcnn2 import PointNetSAModule

class Quantizer(nn.Module):
    def __init__(self, cfg, input_dim):
        super().__init__()
        self.n_e = cfg.soft_vq.n_e
        self.e_dim = cfg.soft_vq.e_dim

        assert cfg.latent_dim % self.e_dim == 0, \
            f"latent_dim ({cfg.latent_dim}) must be divisible by e_dim ({self.e_dim})"
        self.sequence_len = cfg.latent_dim // self.e_dim

        self.num_codebooks = cfg.soft_vq.num_codebooks
        self.learnable = cfg.soft_vq.learnable
        self.tau_min = cfg.soft_vq.tau_min
        self.tau_max = cfg.soft_vq.tau_max
        self.initial_tau = cfg.soft_vq.tau
        if self.learnable:
            self.log_tau = nn.Parameter(
                torch.log(torch.tensor(self.initial_tau, dtype=torch.float32)), 
                requires_grad=True
            )

        self.entropy_loss_ratio = cfg.soft_vq.entropy_loss_ratio
        self.show_usage = cfg.soft_vq.show_usage
        self.l2_norm = cfg.soft_vq.l2_norm

        self.pre_quant_layer = PointNetSAModule(
            num_centers=self.sequence_len, 
            radius=0.2, 
            num_neighbors=32, 
            in_channels=input_dim, 
            out_channels=[self.e_dim]
        )

        # Single embedding layer for all codebooks
        self.embedding = nn.Parameter(torch.randn(self.num_codebooks, self.n_e, self.e_dim))
        self.embedding.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        if self.l2_norm:
            self.embedding.data = F.normalize(self.embedding.data, p=2, dim=-1)

        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(self.num_codebooks, 65536))

    @property
    def tau(self):
        if self.learnable:
            return torch.clamp(torch.exp(self.log_tau), min=self.tau_min, max=self.tau_max)  # Ensure tau is positive and bounded
        else:
            return self.initial_tau

    def forward(self, features, xyz, *args, **kwargs):
        """
        Args:
            z: Input tensor.
        """
        features, _, _ = self.pre_quant_layer((features, xyz, None))  # Project input to codebook size
        z = features.transpose(1, 2).contiguous()  # (B, e_dim, sequence_len)

        # Handle different input shapes
        if z.dim() == 4:
            z = torch.einsum('b c h w -> b h w c', z).contiguous()
            z = z.view(z.size(0), -1, z.size(-1))
        
        batch_size, seq_length, _ = z.shape
        
        # Ensure sequence length is divisible by number of codebooks
        assert seq_length % self.num_codebooks == 0, \
            f"Sequence length ({seq_length}) must be divisible by number of codebooks ({self.num_codebooks})"
        
        segment_length = seq_length // self.num_codebooks
        z_segments = z.view(batch_size, self.num_codebooks, segment_length, self.e_dim)
        
        # Apply L2 norm if needed
        embedding = F.normalize(self.embedding, p=2, dim=-1) if self.l2_norm else self.embedding
        if self.l2_norm:
            z_segments = F.normalize(z_segments, p=2, dim=-1)
            
        z_flat = z_segments.permute(1, 0, 2, 3).contiguous().view(self.num_codebooks, -1, self.e_dim)
        
        logits = torch.einsum('nbe, nke -> nbk', z_flat, embedding.detach())

        current_tau = self.tau
        
        # Calculate probabilities
        probs = F.softmax(logits / current_tau, dim=-1)

        # Quantize
        z_q = torch.einsum('nbk, nke -> nbe', probs, embedding)
        
        # Reshape back
        z_q = z_q.view(self.num_codebooks, batch_size, segment_length, self.e_dim).permute(1, 0, 2, 3).contiguous()
        
        
        # Calculate cosine similarity
        with torch.no_grad():
            zq_z_cos = F.cosine_similarity(
                z_segments.view(-1, self.e_dim),
                z_q.view(-1, self.e_dim),
                dim=-1
            ).mean()
        
        # Get indices for usage tracking
        indices = torch.argmax(probs, dim=-1)  # (batch*segment_length, num_codebooks)
        
        # Track codebook usage
        if self.show_usage and self.training:
            for k in range(self.num_codebooks):
                cur_len = indices.size(1)
                self.codebook_used[k, :-cur_len].copy_(self.codebook_used[k, cur_len:].clone())
                self.codebook_used[k, -cur_len:].copy_(indices[k, :])
        
        # Calculate losses if training
        if self.training:
            entropy_loss = self.entropy_loss_ratio * self.compute_entropy_loss(logits.view(-1, self.n_e), tau=current_tau)
        else:
            entropy_loss = 0.0

        # Calculate codebook usage
        codebook_usage = torch.tensor([
            len(torch.unique(self.codebook_used[k])) / self.n_e 
            for k in range(self.num_codebooks)
        ]).mean() if self.show_usage else 0

        max_usage = torch.tensor([
            len(torch.unique(self.codebook_used[k])) / self.n_e 
            for k in range(self.num_codebooks)
        ]).max() if self.show_usage else 0

        min_usage = torch.tensor([
            len(torch.unique(self.codebook_used[k])) / self.n_e 
            for k in range(self.num_codebooks)
        ]).min() if self.show_usage else 0

        z_q = z_q.view(batch_size, -1, self.e_dim).contiguous()
        
        # Reshape back to match original input shape
        if len(z.shape) == 4:
            z_q = torch.einsum('b h w c -> b c h w', z_q)
        
        # Calculate average probabilities
        avg_probs = torch.mean(torch.mean(probs, dim=-1))
        max_probs = torch.mean(torch.max(probs, dim=-1)[0])

        info = {
            "avg_probs": avg_probs,
            "max_probs": max_probs,
            "z_cos": zq_z_cos,
            "tau": current_tau.item() if self.learnable else current_tau,
            "codebook_usage": codebook_usage,
            "max_codebook_usage": max_usage,
            "min_codebook_usage": min_usage,
        }

        return z_q.view(z.size(0), -1), entropy_loss, info

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
        if device is None:
            device = self.embedding.device

        sequence_len = self.sequence_len

        assert sequence_len % self.num_codebooks == 0, "sequence_len must be divisible by num_codebooks"
        segment_length = sequence_len // self.num_codebooks

        # Sample random indices for each codebook, batch, and segment
        indices = torch.randint(
            0, self.n_e, 
            (batch_size, self.num_codebooks, segment_length), 
            device=device
        )  # [B, num_codebooks, segment_length]

        embedding = F.normalize(self.embedding, p=2, dim=-1) if self.l2_norm else self.embedding  # [num_codebooks, n_e, e_dim]

        # Gather embeddings for each codebook, batch, and segment
        z_sampled = []
        for k in range(self.num_codebooks):
            # indices[:, k]: [B, segment_length]
            z_sampled.append(embedding[k][indices[:, k]])  # [B, segment_length, e_dim]
        z_sampled = torch.stack(z_sampled, dim=1)  # [B, num_codebooks, segment_length, e_dim]
        z_sampled = z_sampled.view(batch_size, sequence_len, self.e_dim)  # [B, sequence_len, e_dim]
        return z_sampled.view(batch_size, -1)