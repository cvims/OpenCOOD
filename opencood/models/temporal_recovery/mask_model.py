from opencood.data_utils.feature_dataset.dataset import SCOPEFeatureDataset
from opencood.visualization.vis_utils import plot_feature_map
import torch


class TemporalMaskModel(torch.nn.Module):
    # def __init__(self, scope_model):
    def __init__(self):
        super(TemporalMaskModel, self).__init__()
        # self.scope_model = scope_model
        self.cav_fusion = self.build_cav_fusion()
        self.historical_fusion = self.build_historical_fusion()
        self.mask_model = self.build_mask_model()
        self.output_layer = self.build_output_layer()

    def build_cav_fusion(self):
        # Conv layer (keep dimensions) [timesteps * CAVs, 64, 200, 704]
        cav_fusion = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        return cav_fusion

    def build_historical_fusion(self):
        # Conv layer (keep dimensions) [timesteps, 64, 200, 704]
        hist_fusion = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        return hist_fusion

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

    def build_output_layer(self):
        output_layer = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        return output_layer
    
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

        # historical_fusion (we ignore the latest [first in list] timestep and fuse the historical ones)
        hist_fusion_l = []
        feature_sub_list_bs = []
        for bs in range(BS):
            feature_list_bs = [feature_list[i+1][bs] for i in range(timesteps-1)]
            # max pooling across the timesteps
            hist_fusion = torch.max(torch.stack(feature_list_bs, dim=0), dim=0)[0]
            hist_fusion_l.append(hist_fusion)
            # subtract all historical features from the latest timestep
            feature_sub_list = [feature_list[bs][0] - feature_list[i+1][bs] for i in range(timesteps-1)]
            feature_sub_list = torch.min(torch.stack(feature_sub_list, dim=0), dim=0)[0]
            feature_sub_list_bs.append(feature_sub_list)

        hist_fusion_output = torch.stack(hist_fusion_l, dim=0)
        hist_fusion_output = self.historical_fusion(hist_fusion_output)
        feature_sub_list_output = torch.stack(feature_sub_list_bs, dim=0)

        # todo flow model

        # mask_model
        # mask_output = self.mask_model(torch.cat([feature_list[0], hist_fusion_output], dim=1))
        # mask_output = self.mask_model(hist_fusion_output - feature_list[0])
        mask_output = self.mask_model(feature_sub_list_output)

        mask_hist_fusion = hist_fusion_output * mask_output

        mask_hist_fusion = self.output_layer(mask_hist_fusion)

        mask_output = torch.squeeze(mask_output, dim=1)

        if sum([v['temporal_recovered'] for v in data_dict_list[0]['ego']['object_detection_info_mapping'][0].values()]) > 0:
            print('hier')
            from opencood.loss.temporal_bce_loss import TemporalMaskBCELoss

            bce_loss = TemporalMaskBCELoss(None)
            bce_loss(
                mask_output,
                data_dict_list[0]['ego']['object_bbx_center'],
                data_dict_list[0]['ego']['object_detection_info_mapping']
            )

        return mask_output, mask_hist_fusion
