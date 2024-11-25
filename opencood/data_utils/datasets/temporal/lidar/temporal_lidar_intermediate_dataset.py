"""
Fusion for intermediate level (lidar)
"""
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.temporal_utils import filter_vehicles_by_category, \
    update_temporal_vehicles_list, filter_by_opv2v_original_visibility, \
    update_kitti_criteria
from opencood.utils import box_utils
from opencood.data_utils.datasets.temporal.lidar.base_temporal_lidar_dataset import BaseTemporalLidarDataset

from opencood.utils import eval_utils


class TemporalLidarIntermediateFusionDataset(BaseTemporalLidarDataset):
    def __init__(
            self,
            params,
            visualize,
            train=True,
            validate=False,
            preload_lidar_files=False,
            **kwargs):
        super(TemporalLidarIntermediateFusionDataset, self).__init__(
            params,
            visualize,
            train,
            validate,
            preload_lidar_files=preload_lidar_files,
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
        all_in_range_vehicles = []
        all_visible_vehicles = []

        ego_id = -999
        ego_lidar_pose = []
        ego_loc = []
        ego_vehicles = None

        # first find the ego vehicle's lidar pose (scenario_samples[-1] because it is the last frame)
        for i, data_sample in enumerate(scenario_samples):
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
                    ego_vehicles = ego_content['params']['vehicles']
                    break
            assert cav_id == list(data_sample.keys())[
                0], "The first element in the OrderedDict must be ego"

        assert ego_id != -999
        assert len(ego_lidar_pose) > 0

        # process all vehicles in ego perspective so that the temporal approach can simply restore them without recalculating
        processed_data = self.get_item_single_car(ego_content, ego_lidar_pose) #, range_filter=self.full_object_range)
        latest_frame_ego_range_vehicles = {v_id: ego_vehicles[v_id] for v_id in processed_data['object_ids']}

        for s_idx, data_sample in enumerate(scenario_samples):
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = OrderedDict()

            pairwise_t_matrix = \
                self.get_pairwise_transformation(data_sample, self.params['train_params']['max_cav'], proj_first=False)

            processed_features = []
            object_stack = []
            object_id_stack = []

            cav_object_stack = []

            # only for inference
            camera_lidar_transform = []

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
            
            selected_cav_processed_ego = None

            # loop over all CAVs to process information
            for i, (cav_id, selected_cav_base) in enumerate(data_sample.items()):
                if selected_cav_base['ego'] and i == 0:
                    # process all vehicles in ego perspective so that the temporal approach can simply restore them without recalculating
                    selected_cav_processed_ego = self.get_item_single_car(selected_cav_base, ego_lidar_pose, temporal=True) #, range_filter=self.full_object_range)
                    ego_range_vehicles = {v_id: selected_cav_base['params']['temporal_vehicles'][v_id] for v_id in selected_cav_processed_ego['object_ids']}
                    # delete if they are not in range anymore
                    ego_range_vehicles = {v_id: vehicle for v_id, vehicle in ego_range_vehicles.items() if v_id in latest_frame_ego_range_vehicles}
                    selected_cav_processed = selected_cav_processed_ego
                elif selected_cav_base['ego'] is False:
                    selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_lidar_pose)
                    if selected_cav_processed['projected_lidar'].shape[0] == 0:
                        # no lidar points in range of ego
                        continue
                else:
                    raise NotImplementedError('Ego has to be the first vehicle in the scenario.')

                if cav_id != ego_id:
                    # add lidar hits of cav to in_range_vehicles
                    for v_id in ego_range_vehicles.keys():
                        if v_id == cav_id:
                            continue
                        # Update KITTI criteria for cooperative perception
                        # CAVs can have easier visibility criteria than ego
                        if cav_id not in selected_cav_base['params']['vehicles']:
                            continue
                        updated_kitti_criteria = update_kitti_criteria(ego_range_vehicles[v_id], selected_cav_base['params']['vehicles'][v_id], self.kitti_detection_criteria)
                        ego_range_vehicles[v_id]['kitti_criteria'] = updated_kitti_criteria['kitti_criteria']
                        ego_range_vehicles[v_id]['kitti_criteria_props'] = updated_kitti_criteria['kitti_criteria_props']
                        # opv2v visible
                        if not ego_range_vehicles[v_id]['opv2v_visible'] and selected_cav_base['params']['vehicles'][v_id]['opv2v_visible']:
                            ego_range_vehicles[v_id]['opv2v_visible'] = selected_cav_base['params']['vehicles'][v_id]['opv2v_visible']

                cav_object_bbx_center = self.post_processor.generate_cav_object_center(selected_cav_base['params']['cav_vehicle'], ego_lidar_pose)
                cav_object_stack.append(cav_object_bbx_center)

                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                processed_features.append(
                    selected_cav_processed['processed_features'])

                velocity.append(selected_cav_processed['velocity'])
                time_delay.append(float(selected_cav_base['time_delay']))
                camera_lidar_transform.append(selected_cav_base['camera_params'])

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
                
                # prev_pose_offsets.append(selected_cav_base['prev_pose_offset'])
                cav_ids.append(cav_id)

            all_in_range_vehicles.append(ego_range_vehicles)

            # exclude all repetitive objects
            unique_object_ids = list(set(object_id_stack))
            unique_indices = \
                [object_id_stack.index(x) for x in unique_object_ids]

            all_visible_vehicles.append(unique_object_ids)

            # temporal gt stack vehicles
            temporal_gt_stack = self.create_temporal_gt_stack(
                all_in_range_vehicles
            )

            # all_opv2v_visible_vehicles = []
            # for v_id in all_in_range_vehicles[-1]:
            #     if all_in_range_vehicles[-1][v_id]['opv2v_visible']:
            #         all_opv2v_visible_vehicles.append(v_id)

            cav_object_stack = np.vstack(cav_object_stack)

            visible_vehicle_ids = list(set(temporal_gt_stack.keys()))

            unique_indices = [selected_cav_processed_ego['object_ids'].index(x) for x in visible_vehicle_ids]

            object_stack = selected_cav_processed_ego['object_bbx_center'][unique_indices]
            object_id_stack = visible_vehicle_ids

            temporal_unique_indices = []
            object_detection_info_mapping = dict()
            for v_id in visible_vehicle_ids:
                object_detection_info_mapping[v_id] = dict()
                object_detection_info_mapping[v_id]['kitti_criteria'] = temporal_gt_stack[v_id]['kitti_criteria']
                object_detection_info_mapping[v_id]['kitti_criteria_props'] = temporal_gt_stack[v_id]['kitti_criteria_props']
                object_detection_info_mapping[v_id]['temporal_kitti_criteria'] = temporal_gt_stack[v_id]['temporal_kitti_criteria']
                object_detection_info_mapping[v_id]['opv2v_visible'] = temporal_gt_stack[v_id]['opv2v_visible']
                object_detection_info_mapping[v_id]['temporal_recovered'] = temporal_gt_stack[v_id]['temporal_recovered']

                if temporal_gt_stack[v_id]['temporal_recovered']:
                    temporal_unique_indices.append(selected_cav_processed_ego['object_ids'].index(v_id))

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            # temporal bbx infos
            temporal_object_stack = selected_cav_processed_ego['object_bbx_center'][temporal_unique_indices]
            temp_object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            temp_mask = np.zeros(self.params['postprocess']['max_num'])
            # use the exact indices to put the temporal_obnject_stack into the temporal_object_bbx_center
            temp_object_bbx_center[[unique_indices.index(x) for x in temporal_unique_indices], :] = temporal_object_stack
            temp_mask[[unique_indices.index(x) for x in temporal_unique_indices]] = 1

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
            
            temporal_label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=temp_object_bbx_center,
                    anchors=anchor_box,
                    mask=temp_mask)
            
            # pad dv, dt, infra to max_cav
            velocity = velocity + (self.max_cav - len(velocity)) * [0.]
            time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
            infra = infra + (self.max_cav - len(infra)) * [0.]
            spatial_correction_matrix = np.stack(spatial_correction_matrix)
            padding_eye = np.tile(np.eye(4)[None],(self.max_cav - len(
                                                spatial_correction_matrix),1,1))
            spatial_correction_matrix = np.concatenate([spatial_correction_matrix,
                                                    padding_eye], axis=0)
        
            # prev_pose_offsets = np.stack(prev_pose_offsets)

            processed_data_dict['ego'].update({
                'cav_bbx_center': cav_object_stack,
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': object_id_stack,
                'object_detection_info_mapping': object_detection_info_mapping,
                'anchor_box': anchor_box,
                'processed_lidar': merged_feature_dict,
                'label_dict': label_dict,
                'temporal_label_dict': temporal_label_dict,
                'cav_num': cav_num,
                'velocity': velocity,
                'time_delay': time_delay,
                'camera_lidar_transform': camera_lidar_transform,
                'infra': infra,
                'spatial_correction_matrix': spatial_correction_matrix,
                'pairwise_t_matrix': pairwise_t_matrix,
                'cav_ids': cav_ids,
                'ego_pose': ego_loc,
                # 'prev_pose_offsets': prev_pose_offsets
            })

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                    np.vstack(
                        projected_lidar_stack)})

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
        # category_filtered_vehicles = [filter_vehicles_by_category(vehibcle_list, self.kitti_detection_criteria_threshold) for vehicle_list in all_in_range_vehicles]
        category_filtered_vehicles = [filter_by_opv2v_original_visibility(vehicle_list) for vehicle_list in all_in_range_vehicles]

        temporal_vehicles_list = update_temporal_vehicles_list(all_in_range_vehicles, category_filtered_vehicles)

        # return the object ids of the temporal vehicles
        return temporal_vehicles_list[-1]


    def get_item_single_car(self, selected_cav_base, ego_pose, range_filter=None, temporal: bool = False):
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
            self.post_processor.generate_object_center([selected_cav_base], ego_pose, range_filter=range_filter, temporal=True)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        # lidar_np = shuffle_points(lidar_np)
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
        output_dict_list = []
        # Iteration over the temporal (scenarios)
        for j in range(len(batch[0])):
            output_dict = {'ego': {}}

            ego_pose = []

            cav_bbx_center = []
            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            object_detection_info_mapping_list = []

            cav_ids = []

            processed_lidar_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            temporal_label_dict_list = []

            # used for PriorEncoding
            velocity = []
            time_delay = []
            infra = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # used for correcting the spatial transformation between delayed timestamp
            # and current timestamp
            spatial_correction_matrix_list = []

            if self.visualize:
                origin_lidar = []

            # Batch iteration itself
            for i in range(len(batch)):
                ego_dict = batch[i][j]['ego']
                ego_pose.append(ego_dict['ego_pose'])

                # cav_bbx_center.append(ego_dict['cav_bbx_center'])
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])

                # new
                object_detection_info_mapping_list.append(ego_dict['object_detection_info_mapping'])

                # cav ids
                cav_ids.append(ego_dict['cav_ids'])

                processed_lidar_list.append(ego_dict['processed_lidar'])
                record_len.append(ego_dict['cav_num'])
                label_dict_list.append(ego_dict['label_dict'])
                temporal_label_dict_list.append(ego_dict['temporal_label_dict'])

                velocity.append(ego_dict['velocity'])
                time_delay.append(ego_dict['time_delay'])
                infra.append(ego_dict['infra'])
                spatial_correction_matrix_list.append(
                    ego_dict['spatial_correction_matrix'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

            # convert to numpy, (B, max_num, 7)
            # cav_bbx_center = torch.from_numpy(np.array(cav_bbx_center))
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            # example: {'voxel_features':[np.array([1,2,3]]),
            # np.array([3,5,6]), ...]}
            merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(merged_feature_dict)
            # [2, 3, 4, ..., M]
            record_len = torch.from_numpy(np.array(record_len, dtype=int))

            coords = processed_lidar_torch_dict['voxel_coords']
            if record_len.sum() != (coords[:, 0].max().int().item() + 1):
                raise ValueError("Record length does not match the processed lidar's length.")
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)
            temporal_label_torch_dict = \
                self.post_processor.collate_batch(temporal_label_dict_list)

            # (B, max_cav)
            velocity = torch.from_numpy(np.array(velocity))
            time_delay = torch.from_numpy(np.array(time_delay))
            infra = torch.from_numpy(np.array(infra))
            spatial_correction_matrix_list = \
                torch.from_numpy(np.array(spatial_correction_matrix_list))
            # (B, max_cav, 3)
            prior_encoding = \
                torch.stack([velocity, time_delay, infra], dim=-1).float()
            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({
                # 'cav_bbx_center': cav_bbx_center,
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': object_bbx_mask,
                'processed_lidar': processed_lidar_torch_dict,
                'record_len': record_len,
                'label_dict': label_torch_dict,
                'temporal_label_dict': temporal_label_torch_dict,
                'object_ids': object_ids,
                'object_detection_info_mapping': object_detection_info_mapping_list,
                'cav_ids': cav_ids,
                'prior_encoding': prior_encoding,
                'spatial_correction_matrix': spatial_correction_matrix_list,
                'pairwise_t_matrix': pairwise_t_matrix,
                'ego_pose': ego_pose[0]})

            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            output_dict_list.append(output_dict)

        return output_dict_list
    
    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict_list = self.collate_batch(batch)

        # check if anchor box in the batch
        for i in range(len(batch[0])):
            if batch[0][i]['ego']['anchor_box'] is not None:
                output_dict_list[i]['ego'].update({'anchor_box':
                    torch.from_numpy(np.array(
                        batch[0][i]['ego'][
                            'anchor_box']))})
            
            output_dict_list[i]['ego']['camera_lidar_transform'] = batch[0][i]['ego']['camera_lidar_transform']
            # to torch
            for j, cav_transform in enumerate(output_dict_list[i]['ego']['camera_lidar_transform']):
                for cam_key in cav_transform:
                    for cam_attr in cav_transform[cam_key]:
                        if cam_attr == 'image_path':
                            continue
                        output_dict_list[i]['ego']['camera_lidar_transform'][j][cam_key][cam_attr] = \
                            torch.from_numpy(np.array(output_dict_list[i]['ego']['camera_lidar_transform'][j][cam_key][cam_attr]))

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()
            output_dict_list[i]['ego'].update({'transformation_matrix':
                                        transformation_matrix_torch})

        return output_dict_list
    
    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        
        gt_box_tensor, gt_object_ids = self.post_processor.generate_gt_bbx(data_dict)
        
        return pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids


    def post_process_cav_vehicle(self, object_bbx_center, transformation_matrix):
        return self.post_processor.generate_cav_bbx(object_bbx_center, transformation_matrix)


    def visualize_result(
            self,
            pred_box_tensor,
            gt_box_tensor,
            pcd,
            show_vis,
            save_path,
            dataset=None):
        # we need to convert the pcd from [n, 5] -> [n, 4]
        pcd = pcd[:, 1:]
        # visualize the model output
        self.post_processor.visualize(
            pred_box_tensor,
            gt_box_tensor,
            pcd,
            show_vis,
            save_path,
            dataset=dataset)
        
    def save_temporal_point_cloud(
            self,
            pred_box_tensor,
            gt_box_tensor,
            gt_object_ids_criteria,
            cav_box_tensor,
            pcd,
            save_path):
        """
        Save the point cloud with temporal information for visualization.
        """
        return self.post_processor.save_temporal_point_cloud(
            pred_box_tensor,
            gt_box_tensor,
            gt_object_ids_criteria,
            cav_box_tensor,
            pcd,
            save_path)



if __name__ == '__main__':
    from opencood.hypes_yaml.yaml_utils import load_yaml
    import torch
    import tqdm
    import time

    eval_utils.set_random_seed(0)

    use_scenarios_idx = [
        0, 5, 11, 14, 22, 24, 35, 40, 41, 42
    ]

    config_file = r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/hypes_yaml/aaa_test_lidar.yaml'
    params = load_yaml(config_file)

    params['root_dir'] = '/data/public_datasets/OPV2V/original/train'

    params['fusion']['args']['queue_length'] = 4
    params['fusion']['args']['temporal_ego_only'] = False
    # params['fusion']['args']['communication_dropout'] = 0.7

    dataset = TemporalLidarIntermediateFusionDataset(
        params, visualize=False, train=True, validate=False,
        use_scenarios_idx=use_scenarios_idx,
        preload_lidar_files=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_batch,
        num_workers=16)

    _start_time = time.time()
    for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # if i == 500:
        #     break
        pass
    _end_time = time.time()
    print(f"Time: {_end_time - _start_time}")

    # dataset = TemporalLidarIntermediateFusionDataset(
    #     params, visualize=False, train=True, validate=False,
    #     preload_lidar_files=False)

    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     collate_fn=dataset.collate_batch,
    #     num_workers=8)

    # _start_time = time.time()
    # for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
    #     if i == 500:
    #         break
    # _end_time = time.time()
    # print(f"Time: {_end_time - _start_time}")
