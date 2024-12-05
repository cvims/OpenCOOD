import torch
import torch.nn as nn
from opencood.visualization.vis_utils import plot_feature_map

class TemporalVisionTransformer(nn.Module):
    def __init__(self, num_heads, channels: int, patch_size: tuple, sequence_length=4):
        super(TemporalVisionTransformer, self).__init__()
        self.num_heads = num_heads
        self.patch_size_h, self.patch_size_w = patch_size

        self.dim = self.patch_size_h * self.patch_size_w

        # Linear embedding layer for patches
        self.patch_embedding = nn.Linear(self.dim * channels, self.dim)

        # Temporal position embedding
        self.temporal_pos_embedding = nn.Parameter(torch.randn(sequence_length, 1, self.dim))

        # Temporal attention
        self.attention = nn.MultiheadAttention(embed_dim=self.dim, num_heads=num_heads)

        # Final projection to reconstruct the spatial map
        self.reconstruction_proj = nn.Linear(self.dim, channels * self.dim)

    def extract_patches(self, x):
        """Extract patches from the spatial dimensions."""
        b, t, c, h, w = x.shape
        ph, pw = self.patch_size_h, self.patch_size_w

        # Ensure height and width are divisible by patch size
        assert h % ph == 0 and w % pw == 0, "Height and Width must be divisible by patch size."

        # Reshape to extract patches
        patches = x.unfold(3, ph, ph).unfold(4, pw, pw)  # [b, t, c, h/ph, w/pw, ph, pw]
        patches = patches.contiguous().view(b, t, c, -1, ph * pw)  # [b, t, c, num_patches, patch_area]
        patches = patches.permute(0, 3, 1, 2, 4)  # [b, num_patches, t, c, patch_area]

        # Flatten patch content and prepare for embedding
        patches = patches.flatten(-2, -1)  # [b, num_patches, t, patch_area * c]
        return patches

    def reconstruct_from_patches(self, patches, h, w):
        """
        Reconstruct the original image from extracted patches.

        Args:
            patches: Tensor of shape [b, num_patches, patch_area * c]
            h: Original height of the image.
            w: Original width of the image.

        Returns:
            Reconstructed image of shape [b, c, h, w].
        """
        b, num_patches, patch_area_c = patches.shape
        ph, pw = self.patch_size_h, self.patch_size_w
        c = patch_area_c // (ph * pw)  # Number of channels
        
        # Step 1: Reshape patch content back to [b, num_patches, t, c, ph, pw]
        patches = patches.view(b, num_patches, c, ph, pw)
        
        # Step 2: Place patches back in the spatial grid
        h_patches, w_patches = h // ph, w // pw  # Number of patches along height and width
        patches = patches.view(b, h_patches, w_patches, c, ph, pw)  # [b, h_patches, w_patches, c, ph, pw]
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()  # [b, c, h_patches, ph, w_patches, pw]

        # Step 3: Merge patches into the original spatial dimensions
        reconstructed = patches.view(b, c, h, w)  # [b, c, h, w]

        return reconstructed

    def forward(self, x):
        # x shape: [batch_size, sequence_length, channels, height, width]
        b, t, c, h, w = x.shaperec

        # Step 1: Extract patches
        patches = self.extract_patches(x)  # [b, num_patches, t, patch_area * c]
        b, num_patches, t, patch_dim = patches.shape

        # reconstructed_patches = self.reconstruct_from_patches(patches, h, w)

        # Flatten temporal and spatial patches for embedding
        patches = patches.view(-1, t, patch_dim)  # [b * num_patches, t, patch_dim]
        # patches = self.patch_embedding(patches)  # [b * num_patches, t, dim]

        # Step 2: Apply temporal attention
        patches = patches.permute(1, 0, 2)  # [t, b * num_patches, dim]

        # # Add temporal position embedding
        # patches = patches + self.temporal_pos_embedding

        # attn_output, _ = self.attention(patches, patches, patches)  # [t, b * num_patches, dim]
        # patches = attn_output.permute(1, 0, 2)  # [b * num_patches, t, dim]

        # to delete
        patches = patches.permute(1, 0, 2)  # [b * num_patches, t, dim]

        patches = patches[:, -1, :]  # Temporal fusion: [b * num_patches, dim] (we pick the latest frame)

        patches = patches.view(b, num_patches, -1)  # [b, num_patches, dim]
        
        # Reconstruct the spatial map
        reconstructed_patches = self.reconstruct_from_patches(patches, h, w)

        return reconstructed_patches


# Example usage
batch_size = 2
sequence_length = 4
channels = 256
height = 100
width = 352
patch_size = (20, 22)
num_heads = 8

data = torch.randn(batch_size, sequence_length, channels, height, width)  # Input tensor

model = TemporalVisionTransformer(num_heads=num_heads, channels=channels, patch_size=patch_size)
output = model(data)

print("Output shape:", output.shape)  # Expected: [2, 256, 100, 352]
