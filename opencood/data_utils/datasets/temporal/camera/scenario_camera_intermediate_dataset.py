"""
Fusion for intermediate level (camera)
"""
from collections import OrderedDict

import numpy as np
import torch

from opencood.data_utils.datasets.temporal.camera.base_scenario_camera_dataset import BaseScenarioCameraDataset
from opencood.utils.bev_creation import create_bev


DETECTION_CRITERIA_RANKING = dict(
    easy=0,
    moderate=1,
    hard=2,
    very_hard=3,
    dontcare=4,
    none=5
)


def get_detection_criteria_rank(type: str):
    if type is None:
        type = 'none'
    else:
        type = type.lower()
    
    return DETECTION_CRITERIA_RANKING[type]


def get_hightest_detection_criteria_rank(detection_criteria: dict):
    highest_rank = 5
    for sensor_name in detection_criteria:
        criteria = detection_criteria[sensor_name]
        rank = get_detection_criteria_rank(criteria)
        if rank < highest_rank:
            highest_rank = rank
    
    return highest_rank


class CamScenarioIntermediateFusionDataset(BaseScenarioCameraDataset):
    def __init__(
            self,
            params,
            visualize,
            train=True,
            validate=False,
            **kwargs):
        super(CamScenarioIntermediateFusionDataset, self).__init__(
            params,
            visualize,
            train,
            validate,
            **kwargs
        )

        self.bev_image_size = params['fusion']['args']['bev_dim']  # px
        self.bev_width = params['fusion']['args']['bev_width']  # m
        self.bev_height = params['fusion']['args']['bev_height']  # m

        # TODO augmentation

        if 'minimum_detection_criteria' in params['fusion']['args']:
            self.minimum_detection_criteria = get_detection_criteria_rank(params['fusion']['args']['minimum_detection_criteria'])
        else:
            self.minimum_detection_criteria = get_detection_criteria_rank('very_hard')


    def __getitem__(self, idx):
        scenario_samples = self.get_sample_random(idx)

        scenario_processed = []

        prev_ego_id = -999

        # all objects before scenario_samples - 1
        # to filter objects that were never observed before
        observed_objects_before_gt = set()

        all_detected_temporal_vehicles = []

        for s_idx, data_sample in enumerate(scenario_samples):
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = OrderedDict()

            ego_id = -999
            ego_lidar_pose = []

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
                    break
            assert cav_id == list(data_sample.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -999
            assert len(ego_lidar_pose) > 0

            pairwise_t_matrix = \
                self.get_pairwise_transformation(data_sample,
                                                self.params['train_params']['max_cav'])

            # Final shape: (L, M, H, W, 3)
            camera_data = []
            # (L, M, 3, 3)
            camera_intrinsic = []
            # (L, M, 4, 4)
            camera2ego = []
            # (L, M, 4, 4)
            camera2self = []

            # (max_cav, 4, 4)
            transformation_matrix = []
            # (N, H, W)
            gt_static = []

            cav_ids = []

            prev_pose_offsets = []

            # save all detected vehicles
            cav_all_detected_vehicles = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in data_sample.items():
                # add all detected vehicles to the list (also vehicles detected by ego)
                cav_all_detected_vehicles.append(selected_cav_base['vehicles'])

                cav_id_before = cav_id
                # cav augmentation methods
                # cav_id, ego_content, selected_cav_base = self.cav_augmentor(ego_id, cav_id, ego_content, selected_cav_base, idx, s_idx)

                # from augmentations
                if cav_id is None and ego_id != cav_id_before:
                    continue

                prev_pose_offsets.append(selected_cav_base['prev_pose_offset'])

                # use full bev if this is the last scenario iteration
                if s_idx == len(scenario_samples) - 1:
                    selected_cav_processed = \
                        self.get_single_cav(selected_cav_base)
                else:
                    # add to observed objects
                    for obj_id in selected_cav_base['object_id']:
                        observed_objects_before_gt.add(obj_id)
                    selected_cav_processed = \
                        self.get_single_cav(selected_cav_base)

                camera_data.append(selected_cav_processed['camera']['data'])
                camera_intrinsic.append(
                    selected_cav_processed['camera']['intrinsic'])
                camera2ego.append(
                    selected_cav_processed['camera']['extrinsic'])
                camera2self.append(
                    selected_cav_processed['camera']['extrinsic_self'])
                transformation_matrix.append(
                    selected_cav_processed['transformation_matrix'])

                cav_ids.append(cav_id)

            # stack all agents together
            camera_data = np.stack(camera_data)
            camera_intrinsic = np.stack(camera_intrinsic)
            camera2ego = np.stack(camera2ego)
            camera2self = np.stack(camera2self)

            # padding
            transformation_matrix = np.stack(transformation_matrix)
            padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(
                                                transformation_matrix), 1, 1))
            transformation_matrix = np.concatenate(
                [transformation_matrix, padding_eye], axis=0)

            prev_pose_offsets = np.stack(prev_pose_offsets)

            # create empty gt_static (and expand dim 0)
            gt_static = np.zeros((1, self.bev_image_size, self.bev_image_size))

            processed_data_dict['ego'].update({
                'transformation_matrix': transformation_matrix,
                'pairwise_t_matrix': pairwise_t_matrix,
                'camera_data': camera_data,
                'camera_intrinsic': camera_intrinsic,
                'camera_extrinsic': camera2ego,
                'camera_extrinsic_self': camera2self,
                'gt_static': gt_static,
                'cav_ids': cav_ids,
                'prev_pose_offsets': prev_pose_offsets,
                'ego_id': int(ego_id),
                'vehicles': cav_all_detected_vehicles
            })

            all_detected_temporal_vehicles.append(cav_all_detected_vehicles)

            temporal_dynamic_gt, detected_vehicles, full_temporal_gt, full_detected_vehicles = self.create_temporal_gt(
                all_detected_temporal_vehicles, ego_content['true_ego_pos'], int(ego_id)
            )

            # expand dim 0
            temporal_dynamic_gt = np.expand_dims(temporal_dynamic_gt, axis=0)
            full_temporal_gt = np.expand_dims(full_temporal_gt, axis=0)

            processed_data_dict['ego'].update({
                'temporal_gt': temporal_dynamic_gt,
                'detected_vehicles': list(detected_vehicles.keys()),
                'detected_vehicles_dict': detected_vehicles,
                'full_temporal_gt': full_temporal_gt,
                'full_detected_vehicles': list(full_detected_vehicles.keys()),
                'full_detected_vehicles_dict': full_detected_vehicles,
                'true_ego_pos': np.asarray(ego_content['true_ego_pos']),
            })

            scenario_processed.append(processed_data_dict)

        return scenario_processed


    def create_temporal_gt(self, vehicles, ego_pos, ego_id, minimum_detection_criteria=None):
        # TODO depending on camera attributes

        # find all detected vehicles of all given vehicles lists (strategy?)
        # When is a vehicle considered visible?
        
        # we use the data from the latest frame list to build the ground truth
        gt_vehicles = dict()
        all_gt_vehicles = dict()
        for cav_vehicle in vehicles[-1]:
            for v_id, v_data in cav_vehicle.items():
                if v_id == ego_id:
                    # Ignore ego vehicle
                    continue
            
                v_data['detection_criteria'] = 'moderate'
                gt_vehicles[v_id] = v_data

                all_gt_vehicles[v_id] = v_data

        temporal_gt, visible_vehicles = create_bev(
            vehicles=gt_vehicles,
            t_ego_pos=ego_pos,
            bev_image_size=self.bev_image_size,
            bev_width=self.bev_width,
            bev_height=self.bev_height,
        )

        # only keep visible vehicles
        gt_vehicles = {k: v for k, v in gt_vehicles.items() if k in visible_vehicles}

        full_temporal_gt, visible_vehicles = create_bev(
            vehicles=all_gt_vehicles,
            t_ego_pos=ego_pos,
            bev_image_size=self.bev_image_size,
            bev_width=self.bev_width,
            bev_height=self.bev_height,
        )

        # only keep visible vehicles
        all_gt_vehicles = {k: v for k, v in all_gt_vehicles.items() if k in visible_vehicles}

        return temporal_gt, gt_vehicles, full_temporal_gt, all_gt_vehicles

    @staticmethod
    def get_pairwise_transformation(base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        # default are identity matrix
        pairwise_t_matrix[:, :] = np.identity(4)

        # return pairwise_t_matrix

        t_list = []

        # save all transformation matrix in a list in order first.
        for i, (cav_id, cav_content) in enumerate(base_data_dict.items()):
            if i >= max_cav:
                break
            t_list.append(cav_content['params']['transformation_matrix'])

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i == j:
                    continue
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix


    def get_single_cav(self, selected_cav_base):
        """
        Process the cav data in a structured manner for intermediate fusion.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        
        use_full_bev : bool
            The flag to indicate whether to use full bev or use self.visible flag

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = OrderedDict()

        # update the transformation matrix
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']
        selected_cav_processed.update({
            'transformation_matrix': transformation_matrix
        })

        all_camera_data = []
        all_camera_origin = []
        all_camera_intrinsic = []
        all_camera_extrinsic = []  # extrinsic to ego
        all_camera_extrinsic_self = []  # extrinsic to itself

        # preprocess the input rgb image and extrinsic params first
        for camera_id, camera_data in selected_cav_base['camera_np'].items():
            all_camera_origin.append(camera_data)
            camera_data = self.pre_processor.preprocess(camera_data)
            camera_intrinsic = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_intrinsic']
            cam2ego = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_extrinsic_to_ego']
            cam2self = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_extrinsic']

            all_camera_data.append(camera_data)
            all_camera_intrinsic.append(camera_intrinsic)
            all_camera_extrinsic.append(cam2ego)
            all_camera_extrinsic_self.append(cam2self)

        camera_dict = {
            'origin_data': np.stack(all_camera_origin),
            'data': np.stack(all_camera_data),
            'intrinsic': np.stack(all_camera_intrinsic),
            'extrinsic': np.stack(all_camera_extrinsic),
            'extrinsic_self': np.stack(all_camera_extrinsic_self)
        }

        selected_cav_processed.update({'camera': camera_dict})

        return selected_cav_processed


    def collate_batch(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """

        if not self.train:
            assert len(batch) == 1

        cam_rgb_all_batch = []
        cam_to_ego_all_batch = []
        cam_extrinsic_self_all_batch = []
        cam_intrinsic_all_batch = []

        gt_static_all_batch = []
        gt_dynamic_all_batch = []
        gt_dynamic_full_view_all_batch = []
        gt_dynamic_all_batch_non_corp = []

        transformation_matrix_all_batch = []
        pairwise_t_matrix_all_batch = []
        # used to save each scenario's agent number
        record_len = []

        cav_ids_batch = []

        vehicles_offsets_batch = []

        for i in range(len(batch)):
            scenarios = batch[i]

            cam_rgb_all_scenario = []
            cam_to_ego_all_scenario = []
            cam_extrinsic_self_all_scenario = []
            cam_intrinsic_all_scenario = []

            gt_static_all_scenario = []
            gt_dynamic_all_scenario = []
            gt_dynamic_full_view_all_scenario = []
            gt_dynamic_all_scenario_non_corp = []

            transformation_matrix_all_scenario = []
            pairwise_t_matrix_all_scenario = []
            # used to save each scenario's agent number
            record_len_scenario = []

            cav_ids_scenario = []

            vehicle_offsets_scenario = []
            
            for i, scenario in enumerate(scenarios):
                ego_dict = scenario['ego']
                cav_ids = ego_dict['cav_ids']

                camera_data = ego_dict['camera_data']
                camera_intrinsic = ego_dict['camera_intrinsic']
                camera_extrinsic = ego_dict['camera_extrinsic']
                camera_extrinsic_self = ego_dict['camera_extrinsic_self']

                assert camera_data.shape[0] == \
                    camera_intrinsic.shape[0] == \
                    camera_extrinsic.shape[0]

                record_len_scenario.append(torch.as_tensor(camera_data.shape[0]))

                cam_rgb_all_scenario.append(torch.from_numpy(camera_data).unsqueeze(1).float())
                cam_intrinsic_all_scenario.append(torch.from_numpy(camera_intrinsic).unsqueeze(1).float())
                cam_to_ego_all_scenario.append(torch.from_numpy(camera_extrinsic).unsqueeze(1).float())
                cam_extrinsic_self_all_scenario.append(torch.from_numpy(camera_extrinsic_self).unsqueeze(1).float())

                # ground truth
                gt_static_all_scenario.append(torch.from_numpy(ego_dict['gt_static']).long())
                gt_dynamic_all_scenario.append(torch.from_numpy(ego_dict['gt_dynamic']).long())
                gt_dynamic_full_view_all_scenario.append(torch.from_numpy(ego_dict['gt_dynamic_full_view']).long())
                gt_dynamic_all_scenario_non_corp.append(torch.from_numpy(ego_dict['gt_dynamic_non_corp']).long())

                # transformation matrix
                transformation_matrix_all_scenario.append(
                    torch.from_numpy(ego_dict['transformation_matrix']).float())
                # pairwise matrix
                pairwise_t_matrix_all_scenario.append(torch.from_numpy(ego_dict['pairwise_t_matrix']).float())

                cav_ids_scenario.append(cav_ids)

                vehicle_offsets_scenario.append(
                    {
                        cav_id: torch.from_numpy(ego_dict['prev_pose_offsets'][i]).float()
                        for i, cav_id in enumerate(cav_ids)
                    }
                )
        
            # append all scenarios to all batch lists
            cam_rgb_all_batch.append(cam_rgb_all_scenario)
            cam_intrinsic_all_batch.append(cam_intrinsic_all_scenario)
            cam_to_ego_all_batch.append(cam_to_ego_all_scenario)
            cam_extrinsic_self_all_batch.append(cam_extrinsic_self_all_scenario)
            
            gt_static_all_batch.append(gt_static_all_scenario)
            gt_dynamic_all_batch.append(gt_dynamic_all_scenario)
            gt_dynamic_full_view_all_batch.append(gt_dynamic_full_view_all_scenario)
            gt_dynamic_all_batch_non_corp.append(gt_dynamic_all_scenario_non_corp)

            transformation_matrix_all_batch.append(transformation_matrix_all_scenario)
            pairwise_t_matrix_all_batch.append(pairwise_t_matrix_all_scenario)
            record_len.append(record_len_scenario)

            cav_ids_batch.append(cav_ids_scenario)
            vehicles_offsets_batch.append(vehicle_offsets_scenario)

        # vehicle_location_offsets_batch = self.calculate_vehicle_offsets(cam_extrinsic_self_all_batch, cav_ids_batch)

        # reformat everything such that the batch size is the second dimension and scneario length is the first
        # now we have lists of [BS, Scenario_length, ...]
        # reformat them such that we have [Scenario_length, BS, tensors]
        cam_rgb_all_batch = list(map(list, zip(*cam_rgb_all_batch)))
        # now combine the batch size with the first dimension of tensors
        cam_rgb_all_batch = [torch.cat(cam_rgb_all_scenario, dim=0) for cam_rgb_all_scenario in cam_rgb_all_batch]
        # same with camera intrinsic
        cam_intrinsic_all_batch = list(map(list, zip(*cam_intrinsic_all_batch)))
        # now combine the batch size with the first dimension of tensors
        cam_intrinsic_all_batch = [torch.cat(cam_intrinsic_all_scenario, dim=0) for cam_intrinsic_all_scenario in cam_intrinsic_all_batch]
        # same with camera extrinsic
        cam_to_ego_all_batch = list(map(list, zip(*cam_to_ego_all_batch)))
        # now combine the batch size with the first dimension of tensors
        cam_to_ego_all_batch = [torch.cat(cam_to_ego_all_scenario, dim=0) for cam_to_ego_all_scenario in cam_to_ego_all_batch]
        # same with record len
        record_len = list(map(list, zip(*record_len)))
        # stack record lens
        record_len = [torch.stack(record_len_scenario, dim=0) for record_len_scenario in record_len]
        # same with vehicle offsets
        vehicles_offsets_batch = list(map(list, zip(*vehicles_offsets_batch)))
        # same with gt static
        gt_static_all_batch = list(map(list, zip(*gt_static_all_batch)))
        # stack gt static
        gt_static_all_batch = [torch.stack(gt_static_all_scenario) for gt_static_all_scenario in gt_static_all_batch]
        # same with gt dynamic
        gt_dynamic_all_batch = list(map(list, zip(*gt_dynamic_all_batch)))
        # stack gt dynamic
        gt_dynamic_all_batch = [torch.stack(gt_dynamic_all_scenario) for gt_dynamic_all_scenario in gt_dynamic_all_batch]
        # same with gt dynamic full view
        gt_dynamic_full_view_all_batch = list(map(list, zip(*gt_dynamic_full_view_all_batch)))
        # stack gt dynamic full view
        gt_dynamic_full_view_all_batch = [torch.stack(gt_dynamic_all_scenario) for gt_dynamic_all_scenario in gt_dynamic_full_view_all_batch]
        # same with gt dynamic non corp
        gt_dynamic_all_batch_non_corp = list(map(list, zip(*gt_dynamic_all_batch_non_corp)))
        # stack gt dynamic non corp
        gt_dynamic_all_batch_non_corp = [torch.stack(gt_dynamic_all_scenario) for gt_dynamic_all_scenario in gt_dynamic_all_batch_non_corp]
        # same with transformation matrix
        transformation_matrix_all_batch = list(map(list, zip(*transformation_matrix_all_batch)))
        # stack transformation matrix
        transformation_matrix_all_batch = [torch.stack(transformation_matrix_all_scenario) for transformation_matrix_all_scenario in transformation_matrix_all_batch]
        # same with pairwise matrix
        pairwise_t_matrix_all_batch = list(map(list, zip(*pairwise_t_matrix_all_batch)))
        # stack pairwise matrix
        pairwise_t_matrix_all_batch = [torch.stack(pairwise_t_matrix_all_scenario) for pairwise_t_matrix_all_scenario in pairwise_t_matrix_all_batch]
        # same with cav ids
        cav_ids_batch = list(map(list, zip(*cav_ids_batch)))


        # convert numpy arrays to torch tensor
        return {
            'inputs': cam_rgb_all_batch,
            'extrinsic': cam_to_ego_all_batch,
            'intrinsic': cam_intrinsic_all_batch,
            'vehicle_offsets': vehicles_offsets_batch,
            'gt_static': gt_static_all_batch,
            'gt_dynamic': gt_dynamic_all_batch,
            'gt_dynamic_full_view': gt_dynamic_full_view_all_batch,
            'gt_dynamic_non_corp': gt_dynamic_all_batch_non_corp,
            'transformation_matrix': transformation_matrix_all_batch,
            'pairwise_t_matrix': pairwise_t_matrix_all_batch,
            'record_len': record_len,
            'cav_ids': cav_ids_batch
        }

    def post_process(self, batch_dict, output_dict):
        output_dict = self.post_processor.post_process(batch_dict,
                                                       output_dict)

        return output_dict


if __name__ == '__main__':
    from opencood.hypes_yaml.yaml_utils import load_yaml

    config_file = r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/hypes_yaml/aaa_test.yaml'
    params = load_yaml(config_file)

    dataset = CamScenarioIntermediateFusionDataset(params, visualize=False, train=True, validate=False)

    test = dataset.__getitem__(0)
    test = dataset.collate_batch(test)
