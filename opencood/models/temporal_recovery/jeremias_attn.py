from opencood.data_utils.feature_dataset.dataset import SCOPEFeatureDataset
from opencood.visualization.vis_utils import plot_feature_map
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple
import random


class CustomAttention(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        self.embed_dim = embed_dim

    
    def forward(self, query, key, value, sequences: int):
        p, b, dim = query.shape  # patches, batch, dim; dim = t * h * w

        q = query
        k = key
        v = value

        # q = query.view(p, b, 1, -1)  # [p, b, t, h * w]
        # k = key.view(p, b, sequences-1, -1)  # [p, b, t-1, h * w]
        v = value.view(p, b, sequences, -1)  # [p, b, t, h * w]

        # Normalize using the square root of the dimension
        scale = dim ** 0.5

        # # Compute attention weights
        # attn_weights = torch.einsum('pbqd,pbkd->pbd', q, k) / scale  # [p, b, d]

        # attn_weights = F.softmax(attn_weights, dim=-1)  # [p, b, h * w]

        # # Compute the attention output
        # attn_output = torch.einsum('pbd,pbtd->pbd', attn_weights, v)  # [p, b, h * w]

        # Compute attention weights
        attn_weights = torch.einsum('pbd,pbd->pbd', q, k) / scale  # [p, b, d]

        attn_weights = attn_weights.view(p, b, sequences, -1)  # [p, b, t-1, h * w]
        attn_weights = F.softmax(attn_weights, dim=2)  # [p, b, h * w]
        # attn_weights = attn_weights.view(p, b, -1)  # [p, b, t-1, h * w]

        # Compute the attention output
        attn_output = torch.einsum('pbtd,pbtd->pbd', attn_weights, v)  # [p, b, h * w]

        return attn_output, attn_weights


class MaskedTemporalVisionTransformer(nn.Module):
    def __init__(
            self, num_heads, input_feature_size_h: int, input_feature_size_w: int, input_feature_channels: int,
            patch_size_h: int, patch_size_w: int, sequence_length=4):
        super(MaskedTemporalVisionTransformer, self).__init__()
        self.num_heads = num_heads
        self.patch_size_h, self.patch_size_w = patch_size_h, patch_size_w
        self.num_patches = (input_feature_size_h // self.patch_size_h) * (input_feature_size_w // self.patch_size_w)

        self.channel_downsize = 1

        self.dim = self.patch_size_h * self.patch_size_w * self.channel_downsize
        self.patch_size_h = self.patch_size_h * self.channel_downsize
        self.patch_size_w = self.patch_size_w * self.channel_downsize        

        # input embedder
        self.feature_map_embedding = nn.Sequential(
            nn.Conv2d(input_feature_channels, input_feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_feature_channels, self.channel_downsize, kernel_size=1),
            nn.BatchNorm2d(self.channel_downsize),
            nn.ReLU(inplace=True),
        )

        # # Linear embedding layer for patches
        # self.patch_embedding = nn.Linear(self.dim * channels, self.dim)

        # Spatial position embedding
        self.spatial_pos_embedding = nn.Embedding(self.num_patches, self.dim)

        # Temporal position embedding
        self.temporal_pos_embedding = nn.Embedding(sequence_length, self.dim)

        # # Temporal attention
        # self.attention = nn.MultiheadAttention(
        #     embed_dim=sequence_length * self.dim,
        #     num_heads=1,
        #     dropout=0.75,
        #     batch_first=False)

        # Custom Attention
        self.attention = CustomAttention(embed_dim=sequence_length * self.dim)

        # conv encodings
        # self.q_conv = nn.Conv2d(1, 1, kernel_size=1) # only current frame
        # self.k_conv = nn.Conv2d(sequence_length-1, sequence_length-1, kernel_size=1) # all frames except current
        # self.v_conv = nn.Conv2d(sequence_length, sequence_length, kernel_size=1) # all frames
        self.q_conv = nn.Conv2d(sequence_length, sequence_length, kernel_size=1) # only current frame
        self.k_conv = nn.Conv2d(sequence_length, sequence_length, kernel_size=1) # all frames except current
        self.v_conv = nn.Conv2d(sequence_length, sequence_length, kernel_size=1) # all frames

        # Temporal Attention (Conv)
        # self.attention = ConvAttention(height=patch_size[0], width=patch_size[1], sequences=sequence_length, num_heads=num_heads)

        # Final projection to reconstruct the spatial map
        self.channel_upsampling = nn.Sequential(
            nn.Conv2d(1, input_feature_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_feature_channels // 2, input_feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_feature_channels),
            nn.ReLU(inplace=True),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

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

        x = x.view(b * t, c, h, w)

        fm_emb = self.feature_map_embedding(x)

        x = x.view(b, t, c, h, w)

        fm_emb = fm_emb.view(b, t, self.channel_downsize, h, w)

        # Step 1: Extract patches
        patches = self.extract_patches(fm_emb).contiguous()  # [b, num_patches, t, patch_area * c]
        patches = patches.squeeze(1)
        b, num_patches, t, patch_dim = patches.shape
        patches = patches.view(b * num_patches, t, -1)

        fm_emb = fm_emb.squeeze(2)

        # query = self.q_conv(fm_emb[:,-1:])  # [b, 1, h, w] - only latest frame
        # query = self.extract_patches(query.unsqueeze(dim=2)).contiguous()  # [b, num_patches, 1, h, w]
        # query = query.view(b * num_patches, 1, self.patch_size_h, self.patch_size_w)  # [b * num_patches, 1, dim]
        # key = self.k_conv(fm_emb[:,:-1])  # [b, t-1, h, w] - all frames except current
        # key = self.extract_patches(key.unsqueeze(dim=2)).contiguous()  # [b, num_patches, t-1, h, w]
        # key = key.view(b * num_patches, t-1, self.patch_size_h, self.patch_size_w)  # [b * num_patches, t-1, dim]
        # value = self.v_conv(fm_emb)  # [b, t, h, w] - all frames
        # value = self.extract_patches(value.unsqueeze(dim=2)).contiguous()  # [b, num_patches, t, h, w]
        # value = value.view(b * num_patches, t, self.patch_size_h, self.patch_size_w)  # [b * num_patches, t, dim]

        query = self.q_conv(fm_emb)  # [b, 1, h, w] - only latest frame
        query = self.extract_patches(query.unsqueeze(dim=2)).contiguous()  # [b, num_patches, 1, h, w]
        query = query.view(b * num_patches, t, self.patch_size_h, self.patch_size_w)  # [b * num_patches, 1, dim]
        key = self.k_conv(fm_emb)  # [b, t-1, h, w] - all frames except current
        key = self.extract_patches(key.unsqueeze(dim=2)).contiguous()  # [b, num_patches, t-1, h, w]
        key = key.view(b * num_patches, t, self.patch_size_h, self.patch_size_w)  # [b * num_patches, t-1, dim]
        value = self.v_conv(fm_emb)  # [b, t, h, w] - all frames
        value = self.extract_patches(value.unsqueeze(dim=2)).contiguous()  # [b, num_patches, t, h, w]
        value = value.view(b * num_patches, t, self.patch_size_h, self.patch_size_w)  # [b * num_patches, t, dim]

        # Step 2: Apply temporal attention
        # Add temporal position embedding
        temporal_embeddings = self.temporal_pos_embedding(torch.arange(t).to(device=patches.device))
        temporal_embeddings = temporal_embeddings.view(t, self.patch_size_h, self.patch_size_w)
        # query = query + temporal_embeddings[-1:, :, :]
        # key = key + temporal_embeddings[:-1, :, :]
        # value = value + temporal_embeddings
        query = query + temporal_embeddings
        key = key + temporal_embeddings
        value = value + temporal_embeddings

        # add spatial embedding
        spatial_embeddings = self.spatial_pos_embedding(torch.arange(num_patches).to(device=patches.device))
        spatial_embeddings = spatial_embeddings.view(1, num_patches, 1, self.patch_size_h, self.patch_size_w).repeat(b, 1, t, 1, 1)
        spatial_embeddings = spatial_embeddings.view(b * num_patches, t, self.patch_size_h, self.patch_size_w)
        # query = query + spatial_embeddings[:, -1:, :, :]
        # key = key + spatial_embeddings[:, :-1, :, :]
        # value = value + spatial_embeddings
        query = query + spatial_embeddings
        key = key + spatial_embeddings
        value = value + spatial_embeddings

        # Step 3: Attention
        # query = query.view(b, num_patches, 1 * self.patch_size_h * self.patch_size_w).permute(1,0,2).contiguous() # [num_patches, b, 1 * h * w]
        # key = key.view(b, num_patches, (t-1) * self.patch_size_h * self.patch_size_w).permute(1,0,2).contiguous()
        # value = value.view(b, num_patches, t * self.patch_size_h * self.patch_size_w).permute(1,0,2).contiguous()
        query = query.view(b, num_patches, t * self.patch_size_h * self.patch_size_w).permute(1,0,2).contiguous() # [num_patches, b, 1 * h * w]
        key = key.view(b, num_patches, t * self.patch_size_h * self.patch_size_w).permute(1,0,2).contiguous()
        value = value.view(b, num_patches, t * self.patch_size_h * self.patch_size_w).permute(1,0,2).contiguous()

        attn_output, _ = self.attention(query, key, value, sequences=t)  # [num_patches, b, t * dim]
        patches = attn_output.permute(1, 0, 2).contiguous()
        # patches = patches.view(num_patches * b, t, self.dim)  # [b * num_patches, t, dim]

        # patches = patches[:, -1, :]  # Temporal fusion: [b * num_patches, dim] (we pick the latest frame)

        # patches = patches.view(b, num_patches, -1)  # [b, num_patches, dim]

        # Reconstruct the spatial map
        reconstructed_patches = self.reconstruct_from_patches(patches, h, w)

        # Step 4: Channel upsampling (to original number of channels)
        upsampled = self.out_proj(reconstructed_patches)

        return upsampled


# class MaskedTemporalVisionTransformer(nn.Module):
#     def __init__(self, in_channels: int, patch_size: Tuple[int, int], embed_dim: int, num_heads: int, depth: int, seq_len: int, img_size: Tuple[int, int]):
#         super(MaskedTemporalVisionTransformer, self).__init__()

#         # check if img_size[0] is a multiple of 2
#         self.conditional_upsampler = None
#         self.conditional_downsampler = None
#         # log2 of img_size[0]
#         log2_img_height = int(img_size[0]).bit_length()
#         log2_img_width = int(img_size[1]).bit_length()

#         self.org_img_size = img_size

#         optimal_img_size = (2 ** log2_img_height, 2 ** log2_img_width)

#         # check if 2^log2_img_height == img_size[0]
#         if optimal_img_size[0] != img_size[0] or optimal_img_size[1] != img_size[1]:
#             # create conditional upsampler
#             self.conditional_upsampler = nn.Upsample(
#                 scale_factor=(optimal_img_size[0] / img_size[0], optimal_img_size[1] / img_size[1]),
#                 mode='bilinear',
#                 align_corners=True
#             )

#             self.conditional_downsampler = nn.Upsample(
#                 scale_factor=(img_size[0] / optimal_img_size[0], img_size[1] / optimal_img_size[1]),
#                 mode='bilinear',
#                 align_corners=True
#             )

#             img_size = optimal_img_size

#         self.patch_size = patch_size
#         self.seq_len = seq_len
#         self.h_patches = img_size[0] // patch_size[0]
#         self.w_patches = img_size[1] // patch_size[1]
#         self.num_patches = self.h_patches * self.w_patches
#         self.img_size = img_size
#         self.embed_dim = embed_dim

#         assert self.h_patches / 2 == self.h_patches // 2, 'Height must be divisible by 2'
#         assert self.w_patches / 2 == self.w_patches // 2, 'Width must be divisible by 2'
#         assert self.h_patches == self.w_patches, 'Height and width must have the same number of patches'

#         # Patch embedding
#         self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

#         # Position embeddings
#         self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, self.num_patches, embed_dim))
#         self.temporal_pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

#         # Transformer encoder
#         transformer_encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=embed_dim * 2,
#             dropout=0.1,
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             transformer_encoder_layer,
#             num_layers=depth,
#         )

#         # Gradual upsampling stages
#         self.upsample_steps = nn.ModuleList()
#         width_factor = self.patch_size[1] // self.patch_size[0]
#         height_factor = 1
#         quad_counter = 1
#         while quad_counter < self.h_patches:
#             if quad_counter > 1:
#                 height_factor = 2
#                 width_factor = 2
#             self.upsample_steps.append(self.upsample_layer(embed_dim, (height_factor, width_factor)))
#             quad_counter *= 2

#         # Final projection back to input channels
#         self.final_conv = nn.Sequential(
#             nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#         )

#         # Generate the mask
#         self.mask = self._generate_mask(self.h_patches, self.w_patches, self.seq_len)

#     def upsample_layer(self, channels, scale_factor):
#         _layer = nn.Sequential(
#             nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.LayerNorm(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=5, padding=2),
#             nn.LayerNorm(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=7, padding=3),
#             nn.LayerNorm(channels),
#             nn.ReLU(inplace=True),
#         )

#         return _layer

#     def _generate_mask(self, h_p, w_p, seq_len, radius=1):
#         """
#         Generates a mask to restrict attention to patches at the same position
#         across time and their neighbors.
#         """
#         num_patches = h_p * w_p
#         total_tokens = num_patches * seq_len

#         mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool)

#         for t in range(seq_len):
#             for p in range(num_patches):
#                 h_idx, w_idx = divmod(p, w_p)

#                 for dt in range(-radius, radius + 1):  # Temporal neighbors
#                     for dh in range(-radius, radius + 1):  # Spatial neighbors
#                         for dw in range(-radius, radius + 1):
#                             nh_idx, nw_idx = h_idx + dh, w_idx + dw
#                             nt = t + dt

#                             if 0 <= nh_idx < h_p and 0 <= nw_idx < w_p and 0 <= nt < seq_len:
#                                 neighbor_p = nh_idx * w_p + nw_idx
#                                 src_idx = t * num_patches + p
#                                 tgt_idx = nt * num_patches + neighbor_p
#                                 mask[src_idx, tgt_idx] = True

#         return ~mask  # Invert: True for invalid positions

#     def forward(self, feature_maps):
#         # Input: [BS, Seq, C, H, W]
#         bs, seq, c, h, w = feature_maps.shape
#         x = feature_maps.view(-1, c, h, w)  # Flatten temporal dimension

#         latest_feature_map = feature_maps[:, -1]

#         if self.conditional_upsampler is not None:
#             x = self.conditional_upsampler(x)
#             latest_feature_map = self.conditional_upsampler(latest_feature_map)
            

#         x = self.patch_embed(x)  # [BS * Seq, Embed_dim, H_patches, W_patches]

#         h_p, w_p = x.shape[2], x.shape[3]
#         x = x.flatten(2).transpose(1, 2)  # [BS * Seq, Num_patches, Embed_dim]

#         x = x.view(bs, seq, self.num_patches, -1)  # [BS, Seq, Num_patches, Embed_dim]
#         x = x + self.pos_embed
#         x = torch.reshape(x, (bs, seq * self.num_patches, -1))  # Flatten temporal and patch dims

#         # Apply Transformer layers with the mask
#         x = x.permute(1, 0, 2)  # [Seq * Num_patches, BS, Embed_dim]
#         x = self.transformer_encoder(x, src_key_padding_mask=None, mask=self.mask.to(device=x.device))
#         x = x.permute(1, 0, 2)  # [BS, Seq * Num_patches, Embed_dim]

#         # Reshape and reconstruct
#         # x = x.view(bs, seq, h_p, w_p, -1).mean(dim=1)  # Fuse temporal dimension
#         x = x.view(bs, seq, h_p, w_p, -1)[:,-1]
#         x = x.permute(0, 3, 1, 2)

#         # Gradual upsampling
#         for upsample in self.upsample_steps:
#             x = upsample(x)
        
#         # Final projection
#         x = self.final_conv(x)# + latest_feature_map

#         # Conditional downsampling
#         if self.conditional_downsampler is not None:
#             x = self.conditional_downsampler(x)

#         return x


if __name__ == "__main__":
    # Modellparameter
    in_channels = 256       # Eingabekanäle (z. B. RGB-Bilder)
    patch_size = (8, 32)    # Größe der Patches
    embed_dim = 128         # Dimension der eingebetteten Features
    num_heads = 4           # Anzahl der Attention-Köpfe
    depth = 3               # Anzahl der Transformer-Schichten
    seq_len = 4             # Anzahl der zeitlichen Frames
    img_size = (100, 352)   # Größe der Eingabebilder (Höhe und Breite)

    # Dummy-Input: Batchgröße 2, Sequenzlänge 8, 3-Kanal-Bilder von 64x64 Pixeln
    batch_size = 2
    dummy_input = torch.randn(batch_size, seq_len, in_channels, img_size[0], img_size[1])

    # Modellinitialisierung
    model = MaskedTemporalVisionTransformer(
        in_channels=in_channels,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        seq_len=seq_len,
        img_size=img_size,
    )

    # Number of parameters of my model
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Vorwärtsdurchlauf
    output = model(dummy_input)

    # Ausgabe
    print("Input Shape:", dummy_input.shape)  # [BS, Seq, C, H, W]
    print("Output Shape:", output.shape)      # [BS, Embed_dim, H_patches, W_patches]


# class CustomSpatialTemporalTransformer(nn.Module):
#     def __init__(
#         self,
#         *,
#         tensor_size: Tuple[int, int, int],
#         patch_size: Tuple[int, int, int],
#         positional_masking: Tuple[int, int, int],
#         embedding_dim: int,
#         sequence_length: int,
#         num_layers: int = 6,
#         num_heads: int = 8,
#         keep_input_shape: bool = True
#     ):
#         super().__init__()

#         self.keep_input_shape = keep_input_shape

#         # Unpack tensor and patch sizes
#         channels, height, width = tensor_size
#         patch_channels, patch_height, patch_width = patch_size

#         # Store original dimensions
#         self.original_channels = channels
#         self.original_height = height
#         self.original_width = width

#         self.sequence_length = sequence_length

#         # Store patch dimensions
#         self.patch_channels = patch_channels
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#         self.num_layers = num_layers

#         # Ensure image dimensions are divisible by patch sizes
#         assert (
#             height % patch_height == 0 and
#             width % patch_width == 0 and
#             channels % patch_channels == 0
#         ), 'Image dimensions must be divisible by the patch size.'

#         # Total number of patches
#         self.num_patches_c = channels // patch_channels
#         self.num_patches_h = height // patch_height
#         self.num_patches_w = width // patch_width
#         self.num_patches = self.num_patches_c * self.num_patches_h * self.num_patches_w

#         # Store positional masking
#         self.positonal_masking = positional_masking

#         # Dimension of each patch (flattened)
#         self.patch_dim = patch_channels * patch_height * patch_width

#         # Linear layers for embedding and inverse embedding
#         self.to_patch_embedding = nn.Linear(self.patch_dim, embedding_dim)
#         self.inverse_patch_embedding = nn.Linear(embedding_dim, self.patch_dim)

#         # Positional and temporal embeddings
#         self.pos_embedding = nn.Embedding(self.num_patches, embedding_dim)
#         self.temporal_embedding = nn.Embedding(sequence_length, embedding_dim)

#         # Transformer encoder
#         self.transformer_encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embedding_dim,
#             nhead=num_heads
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             self.transformer_encoder_layer,
#             num_layers=num_layers
#         )

#         self.attn_mask = self._create_attn_mask()

#     def _create_attn_mask(self):    
#         mask_c, mask_h, mask_w = self.positonal_masking

#         pos_w = torch.arange(self.num_patches_w)  # Shape: [W]
#         pos_h = torch.arange(self.num_patches_h)  # Shape: [H]
#         pos_c = torch.arange(self.num_patches_c)  # Shape: [C]

#         pos_w1, pos_w2 = torch.meshgrid(pos_w, pos_w, indexing='ij')  # Shape: [W, W]
#         pos_h1, pos_h2 = torch.meshgrid(pos_h, pos_h, indexing='ij')  # Shape: [H, H]
#         pos_c1, pos_c2 = torch.meshgrid(pos_c, pos_c, indexing='ij')  # Shape: [C, C]

#         mask_w_diff = (pos_w1 - pos_w2).abs() <= mask_w  # Shape: [W, W]
#         mask_h_diff = (pos_h1 - pos_h2).abs() <= mask_h  # Shape: [H, H]
#         mask_c_diff = (pos_c1 - pos_c2).abs() <= mask_c  # Shape: [C, C]

#         # Combine masks
#         pos_mask = mask_c_diff[:, None, None, :, None, None] & \
#            mask_h_diff[None, :, None, None, :, None] & \
#            mask_w_diff[None, None, :, None, None, :] # Shape: [C, H, W, C, H, W]
#         pos_mask = rearrange(pos_mask, 'c h w c1 h1 w1 -> (c h w) (c1 h1 w1)') # Shape: [C * H * W, C * H * W] [num_patches, num_patches]

#         # Expand mask for the sequence length
#         attn_mask = (~pos_mask.repeat(self.sequence_length, self.sequence_length)).float() # Shape: [sequence_length * num_patches, sequence_length * num_patches]

#         # Adapt masking to be compatible with the transformer   
#         attn_mask = attn_mask.masked_fill(attn_mask == 1, float('-inf'))
#         attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

#         return attn_mask
    
#     def forward(self, embedding_sequence: torch.Tensor):
#         # Input shape: [batch_size, sequence_length, channels, height, width]
#         batch_size, seq_len, channels, height, width = embedding_sequence.shape

#         # Rearrange input tensor into patches
#         # Resulting shape: [batch_size, sequence_length, num_patches, patch_dim]
#         patched_embeddings = rearrange(
#             embedding_sequence,
#             'b s (pc nc) (nh ph) (nw pw) -> b s (nc nh nw) (ph pw pc)', # ('b c (h p1) (w p2) -> b (h w) (p1 p2 c)'
#             ph=self.patch_height,
#             pw=self.patch_width,
#             pc=self.patch_channels,
#             nc=self.num_patches_c,
#             nh=self.num_patches_h,
#             nw=self.num_patches_w
#         )

#         # Project patches to embedding dimension
#         # Shape: [batch_size, sequence_length, num_patches, embedding_dim]
#         embedding_sequence = self.to_patch_embedding(patched_embeddings)

#         # Generate and add positional embeddings
#         pos_indices = torch.arange(self.num_patches, device=embedding_sequence.device)
#         pos_embedding = self.pos_embedding(pos_indices)
#         pos_embedding = pos_embedding.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, num_patches, embedding_dim]
#         pos_embedding = pos_embedding.expand(batch_size, seq_len, -1, -1)
#         embedding_sequence += pos_embedding

#         # Generate and add temporal embeddings
#         temporal_indices = torch.arange(seq_len, device=embedding_sequence.device)
#         temporal_embedding = self.temporal_embedding(temporal_indices)
#         temporal_embedding = temporal_embedding.unsqueeze(0).unsqueeze(2)  # Shape: [1, seq_len, 1, embedding_dim]
#         temporal_embedding = temporal_embedding.expand(batch_size, -1, self.num_patches, -1)
#         embedding_sequence += temporal_embedding

#         # Flatten sequence and patches into a single sequence dimension
#         # New shape: [batch_size, total_seq_len, embedding_dim]
#         total_seq_len = seq_len * self.num_patches
#         embedding_sequence = rearrange(embedding_sequence, 'b s p e -> b (s p) e')

#         # Rearrange for transformer input (expected shape: [total_seq_len, batch_size, embedding_dim])
#         embedding_sequence = embedding_sequence.permute(1, 0, 2)

#         # Apply transformer encoder
#         self.attn_mask = self.attn_mask.to(embedding_sequence.device)
#         transformed_sequence = self.transformer_encoder(embedding_sequence, mask=self.attn_mask)

#         # Rearrange back to original shape
#         transformed_sequence = transformed_sequence.permute(1, 0, 2)
#         transformed_sequence = transformed_sequence.view(batch_size, seq_len, self.num_patches, self.embedding_dim)

#         # Get embeddings from the last time step
#         last_sequence = transformed_sequence[:, -1]  # Shape: [batch_size, num_patches, embedding_dim]

#         if self.keep_input_shape:
#             # Inverse embedding to reconstruct patches
#             inverted_patches = self.inverse_patch_embedding(last_sequence)  # Shape: [batch_size, num_patches, patch_dim]

#             # Reconstruct the original image from patches
#             # Resulting shape: [batch_size, channels, height, width]
#             reconstructed_embeddings = rearrange(
#                 inverted_patches,
#                 'b (nc nh nw) (ph pw pc) -> b (pc nc) (nh ph) (nw pw)',
#                 ph=self.patch_height,
#                 pw=self.patch_width,
#                 pc=self.patch_channels,
#                 nc=self.num_patches_c,
#                 nh=self.num_patches_h,
#                 nw=self.num_patches_w
#             )
#         else:
#             reconstructed_embeddings = last_sequence

#         return reconstructed_embeddings
    

# class TemporalEnhancedEncoder(nn.Module):
#     def __init__(
#             self,
#             input_shape: Tuple[int, int, int],
#             patch_size: Tuple[int, int, int],
#             attention_embedding_dim: int,
#             sequence_length: int,
#             positional_masking: Tuple[int, int, int],
#             channel_downsample_factor: int = 2,
#             spatial_downsample_factor: int = 2,
#     ):
#         super(TemporalEnhancedEncoder, self).__init__()

#         c, h, w = input_shape

#         self.channel_downsample_factor = channel_downsample_factor
#         self.spatial_downsample_factor = spatial_downsample_factor

#         stride = (spatial_downsample_factor, spatial_downsample_factor)
        
#         down_channel_size = c // channel_downsample_factor

#         output_height = h // spatial_downsample_factor
#         output_width = w // spatial_downsample_factor

#         assert down_channel_size % patch_size[0] == 0, 'Channels must be divisible by patch size'
#         assert output_height % patch_size[1] == 0, 'Height must be divisible by patch size'
#         assert output_width % patch_size[2] == 0, 'Width must be divisible by patch size'

#         # conv downsampling
#         self.conv_encoder = nn.Sequential(
#             nn.Conv2d(c, down_channel_size, kernel_size=3, stride=stride, padding=1),
#             nn.BatchNorm2d(down_channel_size),
#             nn.ReLU(),
#             nn.Conv2d(down_channel_size, down_channel_size, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(down_channel_size),
#             nn.ReLU(),
#         )

#         # temporal attention
#         self.temporal_attention = CustomSpatialTemporalTransformer(
#             tensor_size=(down_channel_size, output_height, output_width),
#             patch_size=patch_size,
#             embedding_dim=attention_embedding_dim,
#             sequence_length=sequence_length,
#             positional_masking=positional_masking,
#             keep_input_shape=True
#         )

#         # upsample to original shape
#         spatial_upsample_factor = spatial_downsample_factor
        
#         # interpolate and conv
#         self.conv_decoder = nn.Sequential(
#             nn.Upsample(scale_factor=spatial_upsample_factor, mode='bilinear', align_corners=False),
#             nn.Conv2d(down_channel_size, c, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(c),
#             nn.ReLU(),
#             nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(c),
#             nn.ReLU(),
#         )

#     def forward(self, batched_sequences: torch.Tensor):
#         # Input shape: [batch_size, sequence_length, channels, height, width]
#         batch_size, sequence_length, channels, height, width = batched_sequences.shape
#         stacked_sequences = batched_sequences.view(-1, channels, height, width)
    
#         # downsampling
#         downsampled_sequences = self.conv_encoder(stacked_sequences)
#         downsampled_sequences = downsampled_sequences.view(batch_size, sequence_length, downsampled_sequences.shape[1], downsampled_sequences.shape[2], downsampled_sequences.shape[3])

#         # temporal attention
#         temporal_sequence = self.temporal_attention(downsampled_sequences)

#         # upsampling
#         upsampled_sequence = self.conv_decoder(temporal_sequence)

#         return upsampled_sequence


# if __name__ == "__main__":
#     tensor_size = (256, 100, 352)
#     sequence_length = 4
#     # gcd = math.gcd(tensor_size[1], tensor_size[2], tensor_size[0])
#     # divisors = [i for i in range(1, gcd + 1) if gcd % i == 0]

#     channel_downsample_factor = 8
#     spatial_downsample_factor = 2

#     patch_size = (tensor_size[0] // channel_downsample_factor, 5, 16)
#     positional_masking = (patch_size[0], 2, 2)
#     embedding_dim = 128

#     model = TemporalEnhancedEncoder(
#         input_shape=tensor_size,
#         sequence_length=sequence_length,
#         patch_size=patch_size,
#         attention_embedding_dim=embedding_dim,
#         positional_masking=positional_masking,
#         channel_downsample_factor=channel_downsample_factor,
#         spatial_downsample_factor=spatial_downsample_factor,
#     )

#     out = model(torch.randn(1, sequence_length, 256, 100, 352))



# class Conv2DAttention(nn.Module):
#     def __init__(self, in_channels, hidden_dim):
#         super(Conv2DAttention, self).__init__()
#         # Lineare Transformationen für Query, Key, Value
#         self.query = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
#         self.key = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
#         self.value = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)

#         self.output = nn.Sequential(
#             nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU()
#         )
    
#     def forward(self, maps):
#         """
#         maps: Tensor of shape [num_maps, batch_size, in_channels, height, width]
#         return: Fused map of shape [batch_size, in_channels, height, width]
#         """
#         num_maps, batch_size, in_channels, height, width = maps.shape

#         # Reshape maps for joint processing
#         maps = maps.view(num_maps * batch_size, in_channels, height, width)

#         # Calculate Query, Key, and Value
#         query = self.query(maps)
#         key = self.key(maps)
#         value = self.value(maps)

#         # Reshape back to separate maps
#         query = query.view(num_maps, batch_size, -1, height, width)
#         key = key.view(num_maps, batch_size, -1, height, width)
#         value = value.view(num_maps, batch_size, -1, height, width)

#         # Compute pairwise attention scores across maps
#         attention_scores = torch.einsum('nbcij,nbcij->nbc', query, key)

#         # Normalize attention scores
#         attention_scores = F.softmax(attention_scores, dim=-1)

#         # Apply attention to value maps
#         context = torch.einsum('nbc,nbcij->nbcij', attention_scores, value)

#         # Aggregate into a single map
#         aggregated_context = torch.mean(context, dim=0)

#         # Compute final map
#         output = self.output(aggregated_context)

#         return output


# class TemporalResidualConv2DAttention(nn.Module):
#     def __init__(self, in_channels, hidden_dim):
#         super(TemporalResidualConv2DAttention, self).__init__()
#         self.query = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
#         self.key = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
#         self.value = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)

#         # based on attention scores (conv layer)
#         # One learned weight per map
#         self.gamma = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

#         self.output = nn.Sequential(
#             nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU()
#         )

#     def forward(self, maps):
#         """
#         maps: Tensor of shape [num_maps, batch_size, in_channels, height, width].
#             First entry of maps is the current frame, the rest are historical frames.
#         return: Fused map of shape [batch_size, in_channels, height, width]
#         """
#         num_maps, batch_size, in_channels, height, width = maps.shape

#         # Reshape maps for joint processing
#         maps = maps.view(num_maps * batch_size, in_channels, height, width)

#         # Calculate Query, Key, and Value
#         query = self.query(maps)
#         key = self.key(maps)
#         value = self.value(maps)

#         # Reshape back to separate maps
#         query = query.view(num_maps, batch_size, -1, height, width)
#         key = key.view(num_maps, batch_size, -1, height, width)
#         value = value.view(num_maps, batch_size, -1, height, width)

#         # Compute pairwise attention scores across maps
#         attention_scores = torch.einsum('nbcij,nbcij->nbc', query, key)

#         # Normalize attention scores
#         attention_scores = F.softmax(attention_scores, dim=-1)

#         # Apply attention to value maps
#         context = torch.einsum('nbc,nbcij->nbcij', attention_scores, value)

#         # Aggregate into a single map
#         aggregated_context = torch.mean(context, dim=0)

#         # Compute final map
#         output = self.output(aggregated_context)

#         # Apply residual connection
#         difference_map = self.gamma(output)
#         output = difference_map + maps[0]

#         return output, difference_map


# class SpatialTemporalMaskModelAttention(torch.nn.Module):
#     # def __init__(self, scope_model):
#     def __init__(self):
#         super(SpatialTemporalMaskModelAttention, self).__init__()
#         # self.scope_model = scope_model
#         self.spatial_cav_fusion = Conv2DAttention(256, 64)
#         self.temporal_fusion = TemporalResidualConv2DAttention(256, 64)
#         self.mask_model = self.build_mask_model()
#         self.output_layer = self.build_output_layer()

#     def build_output_layer(self):
#         output_layer = torch.nn.Sequential(
#             torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             torch.nn.BatchNorm2d(256),
#             torch.nn.ReLU()
#         )

#         return output_layer

#     def build_mask_model(self):
#         mask_model = torch.nn.Sequential(
#             torch.nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             torch.nn.BatchNorm2d(64),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             torch.nn.BatchNorm2d(1),
#             torch.nn.Sigmoid(),
#         )
#         return mask_model

#     def forward(self, scope_intermediate_output, data_dict_list):
#         # scope_intermediate_output = self.scope_model(data_dict_list)

#         BS = data_dict_list[0]['ego']['object_bbx_center'].shape[0]
#         # [timesteps, CAVs*BS, 64, 200, 704]
#         feature_2d_list = scope_intermediate_output['feature_2d_list']

#         # cavs_per_timestamps shape for BS == 2: [[CAV_Batch1, CAV_Batch2], [CAV_Batch1, CAV_Batch2]]
#         # cavs_per_timestamps shape for BS == 1: [[CAV_Batch1], [CAV_Batch1]]
#         cavs_per_timestamps = [data['ego']['record_len'].tolist() for data in data_dict_list]

#         # to [timesteps * CAVs * BS, 64, 200, 704]
#         feature_batched = torch.cat([x for x in feature_2d_list], dim=0)

#         # format to [timesteps, CAVs*BS, 64, 200, 704]
#         # apply max pooling across the CAVs
#         temporal_features = []
#         for cav_counts in cavs_per_timestamps:
#             feature_list_batch = []
#             for cav_count in cav_counts:
#                 if len(feature_batched) == 0:
#                     print('hier')
#                 cav_fusion = self.spatial_cav_fusion(feature_batched[:cav_count].unsqueeze(1))
#                 feature_list_batch.append(cav_fusion)
#                 feature_batched = feature_batched[cav_count:]
#             temporal_features.append(torch.cat(feature_list_batch, dim=0))

#         temporal_features = torch.stack(temporal_features, dim=0)
#         temporal_fusion_output, difference_map = self.temporal_fusion(temporal_features)

#         mask_output = self.mask_model(difference_map)

#         mask_hist_fusion = self.output_layer(temporal_fusion_output)

#         mask_hist_fusion = mask_hist_fusion * mask_output

#         mask_output = torch.squeeze(mask_output, dim=1)

#         # if sum([v['temporal_recovered'] for v in data_dict_list[0]['ego']['object_detection_info_mapping'][0].values()]) > 0:
#         #     from opencood.loss.temporal_bce_loss import TemporalMaskBCELoss

#         #     bce_loss = TemporalMaskBCELoss(None)
#         #     bce_loss(
#         #         mask_output,
#         #         data_dict_list[0]['ego']['object_bbx_center'],
#         #         data_dict_list[0]['ego']['object_detection_info_mapping']
#         #     )

#         return mask_output, mask_hist_fusion
