"""
Fusion for intermediate level (lidar)
"""
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import vehicle_in_bev_range
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
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
    

if __name__ == '__main__':
    from opencood.hypes_yaml.yaml_utils import load_yaml

    config_file = r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/hypes_yaml/aaa_test_lidar.yaml'
    params = load_yaml(config_file)

    dataset = LidarScenarioIntermediateFusionDataset(params, visualize=False, train=True, validate=False)

    test = dataset.__getitem__(200)
    test = dataset.collate_batch([test])
