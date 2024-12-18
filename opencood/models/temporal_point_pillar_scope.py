import torch
import torch.nn as nn
from opencood.models.point_pillar_scope import PointPillarScope
from opencood.models.temporal_recovery.jeremias_attn import MaskedTemporalVisionTransformer


class TemporalPointPillarScope(PointPillarScope):
    def __init__(self, args):
        super().__init__(
            args,
            temporal_fusion_module=MaskedTemporalVisionTransformer(
                **args['fusion_args']['temporal_fusion']
            )
        )

    def forward(self, data_dict_list):
        batch_dict_list = [] 
        feature_list = []  
        feature_2d_list = []  
        matrix_list = []
        regroup_feature_list = []  
        regroup_feature_list_large = []

        # SCOPE loads the data from latest to oldest (this is different from the new implementation)
        # therefore we need to reverse the data_dict_list
        data_dict_list = data_dict_list[::-1]

        for origin_data in data_dict_list:  
            data_dict = origin_data['ego']
            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
            record_len = data_dict['record_len']

            pairwise_t_matrix = data_dict['pairwise_t_matrix']
            batch_dict = {'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points,
                        'record_len': record_len}
            batch_dict = self.pillar_vfe(batch_dict)
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)
            spatial_features_2d = batch_dict['spatial_features_2d']

            # check if record len matches the spatial features 2d
            if spatial_features_2d.shape[0] != record_len.sum():
                raise ValueError('Record len does not match spatial features 2d')
            
            # downsample feature to reduce memory
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            # compressor
            if self.compression:
                spatial_features_2d = self.naive_compressor(spatial_features_2d)
            # dcn
            if self.dcn:
                spatial_features_2d = self.dcn_net(spatial_features_2d)
                
            batch_dict_list.append(batch_dict)
            spatial_features = batch_dict['spatial_features']
            feature_list.append(spatial_features)
            feature_2d_list.append(spatial_features_2d)
            matrix_list.append(pairwise_t_matrix)  
            regroup_feature_list.append(self.regroup(spatial_features_2d,record_len))  
            regroup_feature_list_large.append(self.regroup(spatial_features,record_len))     
        
        spatial_features = feature_list[0]
        spatial_features_2d = feature_2d_list[0]
        batch_dict = batch_dict_list[0]
        record_len = batch_dict['record_len']
        
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)

        pairwise_t_matrix = matrix_list[0].clone().detach()  
        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix, 
                                            self.backbone,
                                            [self.shrink_conv, self.cls_head, self.reg_head])
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix)
                 
        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        psm_single_v = []
        psm_single_i = []
        rm_single_v = []
        rm_single_i = []
        for b in range(len(split_psm_single)):
            psm_single_v.append(split_psm_single[b][0:1])
            psm_single_i.append(split_psm_single[b][1:2])
            rm_single_v.append(split_rm_single[b][0:1])
            rm_single_i.append(split_rm_single[b][1:2])
        psm_single_v = torch.cat(psm_single_v, dim=0)
        psm_single_i = torch.cat(psm_single_i, dim=0)
        rm_single_v = torch.cat(rm_single_v, dim=0)
        rm_single_i = torch.cat(rm_single_i, dim=0)

        psm_cross = self.cls_head(fused_feature)

        ### Get all features instead of only the current frame features (default SCOPE)
        fused_features = [fused_feature.unsqueeze(dim=1)] # current frame already processed

        for i in range(1, len(data_dict_list)):
            spatial_features = feature_list[i]
            spatial_features_2d = feature_2d_list[i]
            batch_dict = batch_dict_list[i]
            record_len = batch_dict['record_len']
            psm_single = self.cls_head(spatial_features_2d)
            rm_single = self.reg_head(spatial_features_2d)
            pairwise_t_matrix = matrix_list[i].clone().detach()

            if self.multi_scale:
                fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features,
                                                psm_single,
                                                record_len,
                                                pairwise_t_matrix,
                                                self.backbone,
                                                [self.shrink_conv, self.cls_head, self.reg_head])
                if self.shrink_flag:
                    fused_feature = self.shrink_conv(fused_feature)
            else:
                fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d,
                                                psm_single,
                                                record_len,
                                                pairwise_t_matrix)

            fused_features.append(fused_feature.unsqueeze(dim=1))

        fused_features = torch.cat(fused_features, dim=1)

        # for transformer (from old to latest)
        fused_features = fused_features.flip(dims=[1])

        ### Temporal Mask Model
        temporal_output = self.temporal_mask_model(
            fused_features#, data_dict_list
        )

        psm_temporal = self.cls_head(temporal_output)
        # rm_temporal = self.reg_head(temporal_output)
        
        ego_feature_list = [x[0:1,:] for x in regroup_feature_list[0]]
        ego_feature = torch.cat(ego_feature_list,dim=0)
        final_feature = self.late_fusion([temporal_output,ego_feature,fused_feature],psm_temporal,psm_single_v,psm_cross)
        # print('fused_feature:{},final_feature:{}'.format(fused_feature.shape,final_feature.shape))
        
        psm = self.cls_head(final_feature)
        rm = self.reg_head(final_feature)

        # psm = self.cls_head_new(temporal_output)
        # rm = self.reg_head_new(temporal_output)

        output_dict = {'psm': psm,
                    'rm': rm
                    }
        output_dict.update(result_dict)
        # print("communication rate:",communication_rates)
        
        output_dict.update({
            'psm_single_v': psm_single_v,
            'psm_single_i': psm_single_i,
            'rm_single_v': rm_single_v,
            'rm_single_i': rm_single_i,
            'comm_rate': communication_rates,
            'temporal_features': temporal_output
        })

        return output_dict
