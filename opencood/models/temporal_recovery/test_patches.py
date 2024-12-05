import torch
from opencood.visualization.vis_utils import plot_feature_map


class MaskedTemporalVisionTransformer(torch.nn.Module):
    def __init__(self, patch_size: tuple, image_size: tuple):
        super(MaskedTemporalVisionTransformer, self).__init__()
        self.patch_size_h, self.patch_size_w = patch_size

        self.dim = self.patch_size_h * self.patch_size_w

        self.num_patches = (image_size[0] // self.patch_size_h) * (image_size[1] // self.patch_size_w)


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
        b, t, c, h, w = x.shape

        # Step 1: Extract patches
        patches = self.extract_patches(x)  # [b, num_patches, t, patch_area * c]
        b, num_patches, t, patch_dim = patches.shape

        patches = patches.view(-1, c, self.patch_size_h, self.patch_size_w) # [b * num_patches * t, c, patch_h, patch_w]
        patches = patches.min(dim=1, keepdim=True)[0]  # [b * num_patches * t, patch_h, patch_w]
        patches = patches.squeeze(1)
        patches = patches.view(b * num_patches, t, -1)  # [b * num_patches, t, dim]

        patches = patches.view(b * num_patches, t, self.patch_size_h, self.patch_size_w)  # [b * num_patches, t, h, w]

        patches = patches.view(b, num_patches, t * self.patch_size_h * self.patch_size_w)
        patches = patches.permute(1,0,2).contiguous()   # [num_patches, b, t * h * w]
        #patches = patches.view(num_patches, b, t * self.patch_size_h * self.patch_size_w)
        patches = patches.permute(1, 0, 2).contiguous()  # [b, num_patches, t * dim]
        patches = patches.view(num_patches * b, t, self.dim)  # [b * num_patches, t, dim]

        patches = patches[:, -1, :]  # Temporal fusion: [b * num_patches, dim] (we pick the latest frame)

        patches = patches.view(b, num_patches, -1)  # [b, num_patches, dim]

        # Reconstruct the spatial map
        reconstructed_patches = self.reconstruct_from_patches(patches, h, w)

        return reconstructed_patches


if __name__ == "__main__":

    image1 = torch.zeros(1, 3, 128, 128)
    image2 = torch.ones(1, 3, 128, 128)

    image2[0,:, 0:16, 0:16] = 0.0
    image2[0,:, 0:16, 32:48] = 0.0
    image2[0,:, 16:32, 0:16] = 0.0

    channels = 3
    patch_size = (16, 16)
    image_size = (128, 128)

    model = MaskedTemporalVisionTransformer(patch_size=patch_size, image_size=image_size)

    x1 = torch.stack([image1, image2], dim=1)

    # add batch dimension
    image3 = torch.zeros(1, 3, 128, 128) - 1.0
    image4 = torch.ones(1, 3, 128, 128) + 2.0

    image4[0,:, 32:48, 32:48] = 0.0

    x2 = torch.stack([image3, image4], dim=1)

    x = torch.cat([x1, x2], dim=0)

    y = model(x)

    print(y.shape)
