"""
Fusion for intermediate level (lidar)
"""
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points
from opencood.utils.temporal_utils import filter_vehicles_by_category, update_temporal_vehicles_list
from opencood.utils import box_utils
from opencood.data_utils.datasets.temporal.lidar.base_scenario_lidar_dataset import BaseScenarioLidarDataset


class LidarScenarioIntermediateFusionDataset(BaseScenarioLidarDataset):
    def __init__(
            self,
            params,
            visualize,
            train=True,
            validate=False,
            **kwargs):
        super(LidarScenarioIntermediateFusionDataset, self).__init__(
            params,
            visualize,
            train,
            validate,
            **kwargs
        )

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.proj_first = True
        if 'proj_first' in params['fusion']['args'] and \
            not params['fusion']['args']['proj_first']:
            self.proj_first = False
        
        # this is to simply load all bounding boxes without range filter
        self.full_object_range = [-99999, -99999, -3, 99999, 99999, 1]


    def __getitem__(self, idx):
        scenario_samples = self.get_sample_random(idx)

        scenario_processed = []

        prev_ego_id = -999

        all_processed_data = []
        all_in_range_vehicles = []
        all_visible_vehicles = []

        for s_idx, data_sample in enumerate(scenario_samples):
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = OrderedDict()

            ego_id = -999
            ego_lidar_pose = []
            ego_loc = []
            ego_vehicles = None

            # first find the ego vehicle's lidar pose
            for cav_id, ego_content in data_sample.items():
                if ego_content['ego']:
                    ego_id = cav_id
                    if prev_ego_id == -999:
                        prev_ego_id = ego_id
                    if ego_id != prev_ego_id:
                        print('Attention: Ego vehicle changed in the same scenario.')
                    prev_ego_id = ego_id
                    ego_lidar_pose = ego_content['params']['lidar_pose']
                    ego_loc = ego_content['params']['true_ego_pos']
                    ego_vehicles = ego_content['vehicles']
                    break
            assert cav_id == list(data_sample.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -999
            assert len(ego_lidar_pose) > 0

            pairwise_t_matrix = \
                self.get_pairwise_transformation(data_sample, self.params['train_params']['max_cav'], proj_first=False)
            
            # process all vehicles in ego perspective so that the temporal approach can simply restore them without recalculating
            processed_data = self.get_item_single_car(ego_content, ego_lidar_pose) #, range_filter=self.full_object_range)
            all_processed_data.append(processed_data)
            ego_range_vehicles = {v_id: ego_vehicles[v_id] for v_id in processed_data['object_ids']}

            processed_features = []
            object_stack = []
            object_id_stack = []

            # prior knowledge for time delay correction and indicating data type
            # (V2V vs V2i)
            velocity = []
            time_delay = []
            infra = []
            spatial_correction_matrix = []

            cav_ids = []

            prev_pose_offsets = []

            if self.visualize:
                projected_lidar_stack = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in data_sample.items():
                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)

                if cav_id != ego_id:
                    # add lidar hits of cav to in_range_vehicles
                    for v_id in ego_range_vehicles.keys():
                        if v_id == cav_id:
                            continue
                        ego_range_vehicles[v_id]['lidar_hits'] += selected_cav_base['vehicles'][v_id]['lidar_hits']

                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                processed_features.append(
                    selected_cav_processed['processed_features'])

                velocity.append(selected_cav_processed['velocity'])
                time_delay.append(float(selected_cav_base['time_delay']))
                # this is only useful when proj_first = True, and communication
                # delay is considered. Right now only V2X-ViT utilizes the
                # spatial_correction. There is a time delay when the cavs project
                # their lidar to ego and when the ego receives the feature, and
                # this variable is used to correct such pose difference (ego_t-1 to
                # ego_t)
                spatial_correction_matrix.append(
                    selected_cav_base['params']['spatial_correction_matrix'])
                infra.append(1 if int(cav_id) < 0 else 0)

                if self.visualize:
                    projected_lidar_stack.append(
                        selected_cav_processed['projected_lidar'])
                
                prev_pose_offsets.append(selected_cav_base['prev_pose_offset'])
                cav_ids.append(cav_id)
            
            all_in_range_vehicles.append(ego_range_vehicles)

            # exclude all repetitive objects
            unique_object_ids = list(set(object_id_stack))
            unique_indices = \
                [object_id_stack.index(x) for x in unique_object_ids]

            all_visible_vehicles.append(unique_object_ids)

            visible_vehicle_ids = self.create_temporal_gt_stack(
                all_in_range_vehicles
            )

            unique_indices = [processed_data['object_ids'].index(x) for x in visible_vehicle_ids]

            object_stack = processed_data['object_bbx_center'][unique_indices]
            object_id_stack = visible_vehicle_ids

            # object_stack = np.vstack(object_stack)
            # object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            cav_num = len(processed_features)
            merged_feature_dict = self.merge_features_to_dict(processed_features)

            # generate the anchor boxes
            anchor_box = self.post_processor.generate_anchor_box()

            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    mask=mask)

            # pad dv, dt, infra to max_cav
            velocity = velocity + (self.max_cav - len(velocity)) * [0.]
            time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
            infra = infra + (self.max_cav - len(infra)) * [0.]
            spatial_correction_matrix = np.stack(spatial_correction_matrix)
            padding_eye = np.tile(np.eye(4)[None],(self.max_cav - len(
                                                spatial_correction_matrix),1,1))
            spatial_correction_matrix = np.concatenate([spatial_correction_matrix,
                                                    padding_eye], axis=0)
        
            prev_pose_offsets = np.stack(prev_pose_offsets)

            processed_data_dict['ego'].update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': object_id_stack,
                'anchor_box': anchor_box,
                'processed_lidar': merged_feature_dict,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'velocity': velocity,
                'time_delay': time_delay,
                'infra': infra,
                'spatial_correction_matrix': spatial_correction_matrix,
                'pairwise_t_matrix': pairwise_t_matrix,
                'cav_ids': cav_ids,
                'prev_pose_offsets': prev_pose_offsets
            })

            scenario_processed.append(processed_data_dict)
    
        return scenario_processed


    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict


    def create_temporal_gt_stack(self, all_in_range_vehicles):
        category_filtered_vehicles = [filter_vehicles_by_category(vehicle_list, self.lidar_detection_criteria_threshold, False) for vehicle_list in all_in_range_vehicles]

        temporal_vehicles_list = update_temporal_vehicles_list(all_in_range_vehicles, category_filtered_vehicles)

        # return the object ids of the temporal vehicles
        return list(temporal_vehicles_list[-1].keys())
        

    def get_item_single_car(self, selected_cav_base, ego_pose, range_filter=None):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base], ego_pose, range_filter=range_filter)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                         transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # velocity
        velocity = selected_cav_base['params']['ego_speed']
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             'velocity': velocity})

        return selected_cav_processed
    

    def collate_batch(self, batch):
        # Intermediate fusion is different the other two
        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []

        # used for PriorEncoding
        velocity = []
        time_delay = []
        infra = []

        # used for correcting the spatial transformation between delayed timestamp
        # and current timestamp
        spatial_correction_matrix_list = []

        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        prior_encoding = []

        ego_transformation_matrix_list = []
        anchor_box_list = []

        origin_lidar_list = []

        cav_ids_batch = []

        vehicle_offsets_batch = []

        for i in range(len(batch)):
            scenarios = batch[i]

            object_bbx_center_scenario = []
            object_bbx_mask_scenario = []
            object_ids_scenario = []
            processed_lidar_list_scenario = []
            record_len_scenario = []
            label_dict_list_scenario = []

            velocity_scenario = []
            time_delay_scenario = []
            infra_scenario = []      

            spatial_correction_matrix_list_scenario = []
            pairwise_t_matrix_list_scenario = []

            spatial_correction_matrix_list_scenario = []

            prior_encoding_scenario = []

            # always identity matrix is added
            ego_transformation_matrix_scenario = []

            anchor_box_scenario = []
            
            origin_lidar_scenario = []

            cav_ids_scenario = []

            vehicle_offsets_scenario = []

            for j, scenario in enumerate(scenarios):
                ego_dict = scenario['ego']
                cav_ids = ego_dict['cav_ids']

                object_bbx_center_scenario.append(ego_dict['object_bbx_center'])
                object_bbx_mask_scenario.append(ego_dict['object_bbx_mask'])
                object_ids_scenario.append(ego_dict['object_ids'])

                processed_lidar_list_scenario.append(ego_dict['processed_lidar'])
                record_len_scenario.append(ego_dict['cav_num'])
                label_dict_list_scenario.append(ego_dict['label_dict'])

                velocity_scenario.append(ego_dict['velocity'])
                time_delay_scenario.append(ego_dict['time_delay'])
                infra_scenario.append(ego_dict['infra'])
                spatial_correction_matrix_list_scenario.append(ego_dict['spatial_correction_matrix'])
                pairwise_t_matrix_list_scenario.append(ego_dict['pairwise_t_matrix'])

                ego_transformation_matrix_scenario.append(torch.from_numpy(np.identity(4)).float())
                anchor_box_scenario.append(torch.from_numpy(np.array(ego_dict['anchor_box'])))

                if self.visualize:
                    origin_lidar_scenario.append(torch.from_numpy(ego_dict['origin_lidar']))

                cav_ids_scenario.append(cav_ids)
                vehicle_offsets_scenario.append(
                    {
                        cav_id: torch.from_numpy(ego_dict['prev_pose_offsets'][i]).float()
                        for i, cav_id in enumerate(cav_ids)
                    }
                )

            # convert to numpy, (B, max_num, 7)
            # convert to list of (1, max_num, 7)
            object_bbx_center_scenario = [torch.from_numpy(np.array(x)).unsqueeze(0) for x in object_bbx_center_scenario]
            object_bbx_mask_scenario = [torch.from_numpy(np.array(x)).unsqueeze(0) for x in object_bbx_mask_scenario]

            # processed_lidar_list_scenario = [[f_dict] for f_dict in processed_lidar_list_scenario]
            # processed_lidar_list_scenario = [self.merge_features_to_dict(f_scenario) for f_scenario in processed_lidar_list_scenario]
            # processed_lidar_list_scenario = [self.pre_processor.collate_batch(f_scenario) for f_scenario in processed_lidar_list_scenario]

            record_len_scenario = [torch.from_numpy(np.array([record_len], dtype=int)) for record_len in record_len_scenario]

            # label_dict_list_scenario = [self.post_processor.collate_batch([l_scenario]) for l_scenario in label_dict_list_scenario]

            # (B, max_cav)
            pairwise_t_matrix_list_scenario = [torch.from_numpy(np.array(pairwise_t_matrix)).unsqueeze(0) for pairwise_t_matrix in pairwise_t_matrix_list_scenario]
            velocity_scenario = [torch.from_numpy(np.array(velocity)).unsqueeze(0) for velocity in velocity_scenario]
            time_delay_scenario = [torch.from_numpy(np.array(time_delay)).unsqueeze(0) for time_delay in time_delay_scenario]
            infra_scenario = [torch.from_numpy(np.array(infra)).unsqueeze(0) for infra in infra_scenario]
            spatial_correction_matrix_list_scenario = [torch.from_numpy(np.array(spatial_correction_matrix)).unsqueeze(0) for spatial_correction_matrix in spatial_correction_matrix_list_scenario]

            # (B, max_cav, 3)
            prior_encoding_scenario = [torch.stack([velocity_scenario[i], time_delay_scenario[i], infra_scenario[i]], dim=-1).float() for i in range(len(velocity_scenario))]

            # if self.visualize:
            #     origin_lidar = \
            #         np.array(origin_lidar)
            #     origin_lidar_scenario.append(torch.from_numpy(origin_lidar))

            # add them to the lists
            object_bbx_center.append(object_bbx_center_scenario)
            object_bbx_mask.append(object_bbx_mask_scenario)
            object_ids.append(object_ids_scenario)
            processed_lidar_list.append(processed_lidar_list_scenario)
            record_len.append(record_len_scenario)
            label_dict_list.append(label_dict_list_scenario)
            velocity.append(velocity_scenario)
            time_delay.append(time_delay_scenario)
            infra.append(infra_scenario)
            spatial_correction_matrix_list.append(spatial_correction_matrix_list_scenario)
            pairwise_t_matrix_list.append(pairwise_t_matrix_list_scenario)
            prior_encoding.append(prior_encoding_scenario)
            ego_transformation_matrix_list.append(ego_transformation_matrix_scenario)
            anchor_box_list.append(anchor_box_scenario)
            origin_lidar_list.append(origin_lidar_scenario)

            cav_ids_batch.append(cav_ids_scenario)
            vehicle_offsets_batch.append(vehicle_offsets_scenario)

        # swap batch dim and scenario dim
        object_bbx_center = list(map(list, zip(*object_bbx_center)))
        object_bbx_center = [torch.cat(x, dim=0) for x in object_bbx_center]

        object_bbx_mask = list(map(list, zip(*object_bbx_mask)))
        object_bbx_mask = [torch.cat(x, dim=0) for x in object_bbx_mask]

        object_ids = list(map(list, zip(*object_ids)))

        record_len = list(map(list, zip(*record_len)))
        record_len = [torch.stack(record_len_scenario, dim=0) for record_len_scenario in record_len]

        label_dict_list = list(map(list, zip(*label_dict_list)))
        # iterate scenarios and process batches
        for i, batched_label_dict in enumerate(label_dict_list):
            label_dict_list[i] = self.post_processor.collate_batch(batched_label_dict)
            
        processed_lidar_list = list(map(list, zip(*processed_lidar_list)))
        for i, batched_processed_lidar in enumerate(processed_lidar_list):
            processed_lidar_list[i] = self.merge_features_to_dict(batched_processed_lidar)
            processed_lidar_list[i] = self.pre_processor.collate_batch(processed_lidar_list[i])

        # new_processed_lidar_list = []
        # for i in range(len(processed_lidar_list)):
        #     new_processed_lidar_dict = {}
        #     for feature_name in processed_lidar_list[i][0]:
        #         new_processed_lidar_dict[feature_name] = torch.cat([x[feature_name] for x in processed_lidar_list[i]], dim=0)
        #     new_processed_lidar_list.append(new_processed_lidar_dict)


        return dict(
            object_bbx_center=object_bbx_center,
            object_bbx_mask=object_bbx_mask,
            processed_lidar=processed_lidar_list,
            record_len=record_len,
            label_dict=label_dict_list,
            object_ids=object_ids,
            prior_encoding=prior_encoding,
            spatial_correction_matrix=spatial_correction_matrix_list,
            pairwise_t_matrix=pairwise_t_matrix_list,
            transformation_matrix=ego_transformation_matrix_list,
            anchor_box=anchor_box_list,
            origin_lidar=origin_lidar_list,
            cav_ids=cav_ids_batch,
            vehicle_offsets=vehicle_offsets_batch
        )





if __name__ == '__main__':
    from opencood.hypes_yaml.yaml_utils import load_yaml

    config_file = r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/hypes_yaml/aaa_test_lidar.yaml'
    params = load_yaml(config_file)
    params

    dataset = LidarScenarioIntermediateFusionDataset(params, visualize=False, train=True, validate=False)

    batch1 = dataset.__getitem__(200)
    batch2 = dataset.__getitem__(201)
    test = dataset.collate_batch([batch1, batch2])
