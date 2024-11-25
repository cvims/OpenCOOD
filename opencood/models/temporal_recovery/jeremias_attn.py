from opencood.data_utils.feature_dataset.dataset import SCOPEFeatureDataset
from opencood.visualization.vis_utils import plot_feature_map
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


class CustomSpatialTemporalTransformer(nn.Module):
    def __init__(self, *, tensor_size: Tuple[int, int, int], patch_size: Tuple[int, int, int], embedding_dim: int, sequence_length: int):
        super().__init__()
        channels, height, width = tensor_size
        patch_channels, patch_height, patch_width = patch_size
        self.patch_channels = patch_channels
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.embedding_dim = embedding_dim

        assert [height % patch_height, width % patch_width, channels % patch_channels] == [0, 0, 0], 'Image dimensions must be divisible by the patch size.'

        num_patches = (height // patch_height) * (width // patch_width) * (channels // patch_channels)
        patch_dim = patch_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Linear(patch_dim, embedding_dim)

        self.pos_embedding = nn.Embedding(num_patches, embedding_dim)

        self.temporal_embedding = nn.Embedding(sequence_length, embedding_dim)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)

        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)
    
    def forward(self, cobevt_embedding_sequence: torch.Tensor):
        b, s, c, h, w = cobevt_embedding_sequence.shape
        patched_embeddings = rearrange(cobevt_embedding_sequence, 'b s (p3 c) (h p1) (w p2) -> b s (h w c) (p1 p2 p3)', p1 = self.patch_height, p2 = self.patch_width, p3 = self.patch_channels)
        
        embedding_sequence = self.to_patch_embedding(patched_embeddings)

        pos_embedding = self.pos_embedding(torch.arange(embedding_sequence.shape[2], device=embedding_sequence.device)).repeat(b, s, 1, 1)

        temporal_embedding = self.temporal_embedding(torch.arange(embedding_sequence.shape[1], device=embedding_sequence.device)).unsqueeze(1).repeat(b, 1, self.embedding_dim, 1)

        embedding_sequence = embedding_sequence + pos_embedding + temporal_embedding

        embedding_sequence = rearrange(embedding_sequence, 'b s n d -> b (s n) d')

        transformed_embedding_sequence = self.transformer_encoder(embedding_sequence)

        transformed_embedding_sequence = rearrange(transformed_embedding_sequence, 'b (s n) d -> b s n d', s=s)

        last_transformed_embedding_sequence = transformed_embedding_sequence[:, -1]

        return last_transformed_embedding_sequence

if __name__ == "__main__":
    tensor_size = (256, 100, 352)
    sequence_length = 4
    # gcd = math.gcd(tensor_size[1], tensor_size[2], tensor_size[0])
    # divisors = [i for i in range(1, gcd + 1) if gcd % i == 0]

    patch_size = (32, 25, 32)
    model = CustomSpatialTemporalTransformer(tensor_size=tensor_size, patch_size=patch_size, embedding_dim=352, sequence_length=sequence_length)
    out = model(torch.randn(1, sequence_length, 256, 100, 352))



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
