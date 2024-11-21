from opencood.data_utils.feature_dataset.dataset import SCOPEFeatureDataset
from opencood.visualization.vis_utils import plot_feature_map
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(Attention, self).__init__()
        # Lineare Transformationen für Query, Key, Value
        self.query = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.key = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.value = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)

        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, maps):
        """
        maps: Tensor der Form [num_maps, batch_size, in_channels, height, width]
        Rückgabe: Eine Differenzkarte [batch_size, in_channels, height, width]
        """
        num_maps, batch_size, in_channels, height, width = maps.shape

        # Reshape maps for joint processing
        maps = maps.view(num_maps * batch_size, in_channels, height, width)

        # Calculate Query, Key, and Value
        query = self.query(maps)  # [num_maps * batch_size, hidden_dim, height, width]
        key = self.key(maps)      # [num_maps * batch_size, hidden_dim, height, width]
        value = self.value(maps)  # [num_maps * batch_size, hidden_dim, height, width]

        # Reshape back to separate maps
        query = query.view(num_maps, batch_size, -1, height, width)  # [num_maps, batch_size, hidden_dim, height, width]
        key = key.view(num_maps, batch_size, -1, height, width)
        value = value.view(num_maps, batch_size, -1, height, width)

        # Compute pairwise attention scores across maps
        attention_scores = torch.einsum('nbcij,nbcij->nbc', query, key)  # [num_maps, batch_size, height, width]

        # Normalize attention scores
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Apply attention to value maps
        context = torch.einsum('nbc,nbcij->nbcij', attention_scores, value)  # [num_maps, batch_size, hidden_dim, height, width]

        
        # Aggregate differences into a single map (mean or learned combination)
        aggregated_context = torch.mean(context, dim=0)  # [batch_size, hidden_dim, height, width]

        # Compute final difference map
        output = self.output(aggregated_context)  # [batch_size, in_channels, height, width]
        return output


class TemporalMaskModelAttention(torch.nn.Module):
    # def __init__(self, scope_model):
    def __init__(self):
        super(TemporalMaskModelAttention, self).__init__()
        # self.scope_model = scope_model
        self.cav_fusion = Attention(256, 64)
        self.temporal_fusion = Attention(256, 64)
        self.mask_model = self.build_mask_model()
        self.output_layer = self.build_output_layer()

    def build_output_layer(self):
        output_layer = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        return output_layer

    def build_mask_model(self):
        mask_model = torch.nn.Sequential(
            torch.nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid(),
        )
        return mask_model
    
    def regroup(self, feature_batched, cavs_per_timestamps):
        feature_list = []
        for i, cavs in enumerate(cavs_per_timestamps):
            feature_list.append(feature_batched[:cavs].unsqueeze(0))
            feature_batched = feature_batched[cavs:]
        return

    def forward(self, scope_intermediate_output, data_dict_list):
        # scope_intermediate_output = self.scope_model(data_dict_list)

        timesteps = len(data_dict_list)
        BS = data_dict_list[0]['ego']['object_bbx_center'].shape[0]
        # [timesteps, CAVs*BS, 64, 200, 704]
        feature_list = scope_intermediate_output['feature_2d_list']

        # cavs_per_timestamps shape for BS == 2: [[CAV_Batch1, CAV_Batch2], [CAV_Batch1, CAV_Batch2]]
        # cavs_per_timestamps shape for BS == 1: [[CAV_Batch1], [CAV_Batch1]]
        cavs_per_timestamps = [data['ego']['record_len'].tolist() for data in data_dict_list]

        # to [timesteps * CAVs * BS, 64, 200, 704]
        feature_batched = torch.cat([x for x in feature_list], dim=0)

        # format to [timesteps, CAVs*BS, 64, 200, 704]
        # apply max pooling across the CAVs
        feature_list = []
        for i, cavs in enumerate(cavs_per_timestamps):
            feature_list_batch = []
            for cav_count in cavs:
                cav_fusion = self.cav_fusion(feature_batched[:cav_count].unsqueeze(1))
                feature_list_batch.append(cav_fusion)
                feature_batched = feature_batched[cav_count:]
            feature_list.append(torch.cat(feature_list_batch, dim=0))

        # historical_fusion (we ignore the latest [first in list] timestep and fuse the historical ones)
        hist_fusion_l = []
        feature_sub_list_bs = []
        for bs in range(BS):
            feature_list_bs = [feature_list[i+1][bs] for i in range(timesteps-1)]
            hist_fusion_l.append(torch.stack(feature_list_bs, dim=0))
            # subtract all historical features from the latest timestep
            feature_sub_list = [feature_list[bs][0] - feature_list[i+1][bs] for i in range(timesteps-1)]
            feature_sub_list_bs.append(torch.stack(feature_sub_list, dim=0))

        hist_fusion_output = torch.stack(hist_fusion_l, dim=0)
        hist_fusion_output = self.temporal_fusion(hist_fusion_output.permute(1, 0, 2, 3, 4))

        feature_sub_list_output = torch.stack(feature_sub_list_bs, dim=0)

        mask_output = self.mask_model(hist_fusion_output)

        # todo flow model
        mask_hist_fusion = self.output_layer(hist_fusion_output)

        mask_hist_fusion = mask_hist_fusion * mask_output

        mask_output = torch.squeeze(mask_output, dim=1)

        if sum([v['temporal_recovered'] for v in data_dict_list[0]['ego']['object_detection_info_mapping'][0].values()]) > 0:
            from opencood.loss.temporal_bce_loss import TemporalMaskBCELoss

            bce_loss = TemporalMaskBCELoss(None)
            bce_loss(
                mask_output,
                data_dict_list[0]['ego']['object_bbx_center'],
                data_dict_list[0]['ego']['object_detection_info_mapping']
            )

        return mask_output, mask_hist_fusion
