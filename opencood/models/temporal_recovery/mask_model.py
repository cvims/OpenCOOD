from opencood.data_utils.feature_dataset.dataset import SCOPEFeatureDataset
import torch
import einops


class TemporalMaskModel(torch.nn.Module):
    def __init__(self, scope_model):
        super(TemporalMaskModel, self).__init__()
        self.scope_model = scope_model
        self.cav_fusion = self.build_cav_fusion()
        self.historical_fusion = self.build_historical_fusion()
        self.mask_model = self.build_mask_model()

    def build_cav_fusion(self):
        # Conv layer (keep dimensions) [timesteps * CAVs, 64, 200, 704]
        return torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def build_historical_fusion(self):
        # Conv layer (keep dimensions) [timesteps, 64, 200, 704]
        return torch.nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def build_mask_model(self):
        return torch.nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    def regroup(self, feature_batched, cavs_per_timestamps):
        feature_list = []
        for i, cavs in enumerate(cavs_per_timestamps):
            feature_list.append(feature_batched[:cavs].unsqueeze(0))
            feature_batched = feature_batched[cavs:]
        return

    def forward(self, data_dict_list):
        scope_output = self.scope_model(data_dict_list)

        timesteps = len(data_dict_list)
        BS = data_dict_list[0]['ego']['object_bbx_center'].shape[0]
        # [timesteps, CAVs*BS, 64, 200, 704]
        feature_list = scope_output['feature_list']

        # cavs_per_timestamps shape for BS == 2: [[CAV_Batch1, CAV_Batch2], [CAV_Batch1, CAV_Batch2]]
        # cavs_per_timestamps shape for BS == 1: [[CAV_Batch1], [CAV_Batch1]]
        cavs_per_timestamps = [data['ego']['record_len'].tolist() for data in data_dict_list]
        cavs_per_timestamps_batched = [sum(x) for x in cavs_per_timestamps] # shape: [count1, count2, ...], list entry count are the timestamps

        # to [timesteps * CAVs * BS, 64, 200, 704]
        feature_batched = torch.cat([x for x in feature_list], dim=0)

        # cav_fusion Shape: [timesteps * CAVs * BS, 64, 200, 704]
        feature_batched_cav = self.cav_fusion(feature_batched)
        
        # format to [timesteps, CAVs*BS, 64, 200, 704]
        # apply max pooling across the CAVs
        feature_list = []
        for i, cavs in enumerate(cavs_per_timestamps):
            feature_list_batch = []
            for cav_count in cavs:
                max_pool = torch.max(feature_batched_cav[:cav_count], dim=0)[0]
                feature_list_batch.append(max_pool)
                feature_batched_cav = feature_batched_cav[cav_count:]
            feature_list.append(torch.stack(feature_list_batch, dim=0))

        # historical_fusion (we ignore the latest/first timestep and fuse the historical ones)
        hist_fusion = []
        for bs in range(BS):
            feature_list_bs = [feature_list[i+1][bs] for i in range(timesteps-1)]
            hist_fusion_out = self.historical_fusion(torch.cat(feature_list_bs, dim=0))
            hist_fusion.append(hist_fusion_out)
        
        hist_fusion_output = torch.stack(hist_fusion, dim=0)

        # mask_model
        mask_output = self.mask_model(torch.cat([feature_list[0], hist_fusion_output], dim=1))

        return mask_output
