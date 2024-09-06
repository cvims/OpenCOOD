"""
A plain dataset class for cameras
"""
from opencood.utils import box_utils, camera_utils
from opencood.data_utils.datasets.temporal.base_scenario_dataset import BaseScenarioDataset
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor

from opencood.utils.temporal_utils import filter_vehicles_by_category, update_temporal_vehicles_list
from opencood.utils.bev_creation import create_bev


class BaseScenarioCameraDataset(BaseScenarioDataset):
    def __init__(self, params, visualize, train=True, validate=False, timestamp_offset: int = 0, **kwargs):
        super(BaseScenarioCameraDataset, self).__init__(params, visualize, train,
                                                validate, timestamp_offset=timestamp_offset, **kwargs)
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = build_postprocessor(params['postprocess'], train)

    def get_sample_random(self, idx):
        return self.retrieve_base_data(idx, True, load_camera_data=True, load_lidar_data=False)
    
    def create_temporal_gt(self, vehicles, ego_pos, ego_id):
        # find all detected vehicles of all given vehicles lists (strategy?)
        # When is a vehicle considered visible?
        category_filtered_vehicles = [filter_vehicles_by_category(vehicle_list, self.camera_detection_criteria_threshold, True) for vehicle_list in vehicles]

        temporal_vehicles_list = update_temporal_vehicles_list(vehicles, category_filtered_vehicles)
        
        # we use the data from the latest frame list to build the ground truth
        temporal_visible_gt_vehicles = dict()
        all_gt_vehicles = dict()
        for veh_id, v_data in vehicles[-1].items():
            if veh_id == ego_id:
                # Ignore ego vehicle
                continue

            if veh_id in temporal_vehicles_list[-1]:
                temporal_visible_gt_vehicles[veh_id] = v_data

            all_gt_vehicles[veh_id] = v_data

        temporal_gt, _ = create_bev(
            vehicles=temporal_visible_gt_vehicles,
            t_ego_pos=ego_pos,
            bev_image_size=self.bev_image_size,
            bev_width=self.bev_width,
            bev_height=self.bev_height,
        )

        # only keep visible vehicles
        temporal_visible_gt_vehicles = {k: v for k, v in temporal_visible_gt_vehicles.items()}

        full_temporal_gt, _ = create_bev(
            vehicles=all_gt_vehicles,  # all vehicles in the last frame (except ego)
            t_ego_pos=ego_pos,
            bev_image_size=self.bev_image_size,
            bev_width=self.bev_width,
            bev_height=self.bev_height,
        )

        # only keep visible vehicles
        all_gt_vehicles = {k: v for k, v in all_gt_vehicles.items()}

        return temporal_gt, temporal_visible_gt_vehicles, full_temporal_gt, all_gt_vehicles

    def visualize_agent_camera_bbx(self, agent_sample,
                                   camera='camera0', draw_3d=True,
                                   color=(0, 255, 0), thickness=2):
        """
        Visualize bbx on the 2d image for a certain agent
        and a certain camera.

        Parameters
        ----------
        agent_sample : dict
            The dictionary contains a certain agent information at a certain
            timestamp.

        camera : str
            Which camera to visualize bbx.

        draw_3d : bool
            Draw 2d bbx or 3d bbx on image.

        color : tuple
            Bbx draw color.

        thickness : int
            Draw thickness.

        Returns
        -------
        The drawn image.
        """
        assert camera in ['camera0', 'camera1', 'camera2', 'camera3'], \
            'the camera has to be camera0, camera1, camera2 or camera3'

        # load camera params and rgb image
        camera_rgb = agent_sample['camera_np'][camera]
        camera_param = agent_sample['camera_params'][camera]
        camera_extrinsic = camera_param['camera_extrinsic']
        camera_intrinsic = camera_param['camera_intrinsic']

        # objects coordinate
        objects = agent_sample['object_bbx_cav']
        # convert to corner representation
        objects = box_utils.boxes_to_corners_3d(objects,
                                                self.post_processor.params[
                                                    'order'])
        # project objects coordinate from lidar space to camera space
        object_camera = camera_utils.project_3d_to_camera(objects,
                                                          camera_intrinsic,
                                                          camera_extrinsic)
        if draw_3d:
            draw_rgb = camera_utils.draw_3d_bbx(camera_rgb,
                                                object_camera,
                                                color,
                                                thickness)
        else:
            draw_rgb = camera_utils.draw_2d_bbx(camera_rgb,
                                                objects,
                                                color,
                                                thickness)
        return draw_rgb

    def visualize_agent_bbx(self, data_sample, agent, draw_3d=True,
                            color=(0, 255, 0), thickness=2):
        """
        Draw bbx on a certain agent's all cameras.

        Parameters
        ----------
        data_sample : dict
            The sample contains all information of all agents.

        agent : str
            The target agent.

        draw_3d : bool
            Draw 3d or 2d bbx.

        color : tuple
            Bbx draw color.

        thickness : int
            Draw thickness.

        Returns
        -------
        A list of drawn image.
        """
        agent_sample = data_sample[agent]
        draw_image_list = []

        for camera in ['camera0', 'camera1', 'camera2', 'camera3']:
            draw_image = self.visualize_agent_camera_bbx(agent_sample,
                                                         camera,
                                                         draw_3d,
                                                         color,
                                                         thickness)
            draw_image_list.append(draw_image)

        return draw_image_list

    def visualize_all_agents_bbx(self, data_sample,
                                 draw_3d=True,
                                 color=(0, 255, 0),
                                 thickness=2):
        """
        Visualize all agents and all cameras in a certain frame.
        """
        draw_image_list = []
        cav_id_list = []

        for cav_id, cav_content in data_sample.items():
            draw_image_list.append(self.visualize_agent_bbx(data_sample,
                                                            cav_id,
                                                            draw_3d,
                                                            color,
                                                            thickness))
            cav_id_list.append(cav_id)

        return draw_image_list, cav_id_list


if __name__ == '__main__':
    from opencood.hypes_yaml.yaml_utils import load_yaml
    import random

    random.seed(0)
    import pickle
    import os

    config_file = r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/hypes_yaml/aaa_test_camera.yaml'
    params = load_yaml(config_file)
    params['fusion']['args'] = {}
    params['fusion']['args']['queue_length'] = 4

    # CAMERA_CONTAINER = load_images_into_container(
    #     params['validate_dir'],
    #     pickle.load(open(os.path.join(params['validate_dir'], 'yamls.pkl'), 'rb')))

    dataset = BaseScenarioCameraDataset(params, visualize=False, train=True, validate=False, sensor_cache_container=CAMERA_CONTAINER)
    dataset.get_sample_random(0)
