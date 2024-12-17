import os
from collections import OrderedDict

import copy
import numpy as np
from torch.utils.data import Dataset
import pickle
import random
import torch
import concurrent.futures
import tqdm
from multiprocessing import Manager

from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.transformation_utils import x1_to_x2
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import pcd_to_np
from opencood.utils.camera_utils import load_rgb_from_files
from opencood.data_utils.datasets import COM_RANGE
from opencood.utils import common_utils
from opencood.utils.temporal_utils import categorize_by_kitti_criteria, KITTI_DETECTION_CATEGORY_ENUM


class BaseDataset(Dataset):
    def __init__(self, params: dict, visualize: bool, train=True, validate=False, preload_lidar_files=False, preload_camera_files=False, **kwargs):
        # For fast testing
        if 'use_scenarios_idx' in kwargs:
            self.use_scenarios_idx = kwargs['use_scenarios_idx']
        else:
            self.use_scenarios_idx = []

        self.params = params
        self.visualize = visualize
        self.train = train
        self.validate = validate

        self.pre_processor = None
        self.post_processor = None
        if 'data_augment' in params:
            self.data_augmentor = DataAugmentor(params['data_augment'], train)
        else:
            self.data_augmentor = None

        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed'] \
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0

        else:
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms

        if self.train and not self.validate:
            self.root_dir = params['root_dir']
        else:
            self.root_dir = params['validate_dir']

        if 'max_cav' not in params['train_params']:
            self.max_cav = 7
        else:
            self.max_cav = params['train_params']['max_cav']

        self.all_yamls = pickle.load(open(os.path.join(self.root_dir, 'additional', 'yamls.pkl'), 'rb'))

        # init scenario folders
        self.scenario_folders = sorted(list(self.all_yamls.keys()))

        if self.use_scenarios_idx:
            self.scenario_folders = [self.scenario_folders[i] for i in range(len(self.scenario_folders)) if i in self.use_scenarios_idx]

        # detection_criteria = params['detection_criteria']
        # self.camera_detection_criteria = detection_criteria['camera']
        # self.camera_detection_criteria_threshold = VISIBLITY_CATEGORY_ENUM[self.camera_detection_criteria['detection_criteria_threshold']]
        # self.camera_dection_criteria_config = self.camera_detection_criteria['config']
        # self.lidar_detection_criteria = detection_criteria['lidar']
        # self.lidar_detection_criteria_threshold = VISIBLITY_CATEGORY_ENUM[self.lidar_detection_criteria['detection_criteria_threshold']]
        # self.lidar_detection_criteria_config = self.lidar_detection_criteria['config']

        detection_criteria = params['kitti_detection']
        self.kitti_detection_criteria = detection_criteria['criteria']
        self.kitti_detection_criteria_threshold = KITTI_DETECTION_CATEGORY_ENUM[detection_criteria['criteria_threshold']]

        self.lidar_cache_container = None
        self.camera_cache_container = None

        self.reinitialize()

        self._preload_sensor_data(preload_lidar_files, preload_camera_files)

    
    def _preload_sensor_data(self, preload_lidar_files, preload_camera_files):
        # iterate scenario database to preload lidar and camera data
        # parallel loading. Use tqdm to show the progress bar
        if preload_lidar_files:
            self.lidar_cache_container = dict()
            for scenario in tqdm.tqdm(self.scenario_database.values(), desc='Preloading lidar files'):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for cav in scenario.values():
                        for timestamp in cav.keys():
                            if timestamp.isdigit():
                                executor.submit(pcd_to_np, cav[timestamp]['lidar'], self.lidar_cache_container)

        if preload_camera_files:
            self.camera_cache_container = dict()
            for scenario in tqdm.tqdm(self.scenario_database.values(), desc='Preloading camera files'):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for cav in scenario.values():
                        for timestamp in cav.keys():
                            if timestamp.isdigit():
                                executor.submit(load_rgb_from_files, cav[timestamp]['cameras'], self.camera_cache_container)

    def reinitialize(self):
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_list = sorted([x for x in self.all_yamls[scenario_folder].keys()])
            if self.train and not self.validate:
                # shuffle
                random.shuffle(cav_list)
            
            assert len(cav_list) > 0, f'No CAVs in {scenario_folder}'

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            ego_id = cav_list[0]
            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                timestamps = list(self.all_yamls[scenario_folder][cav_id].keys())

                cav_path = os.path.join(self.root_dir, scenario_folder, str(cav_id))
                for timestamp in timestamps:
                    ego_lidar_pose = self.all_yamls[scenario_folder][ego_id][timestamp]['lidar_pose']
                    cav_lidar_pose = self.all_yamls[scenario_folder][cav_id][timestamp]['lidar_pose']
                    # check if the cav is within the communication range with ego
                    distance = common_utils.cav_distance_cal(cav_lidar_pose, ego_lidar_pose)

                    if distance > COM_RANGE:
                        continue

                    self.scenario_database[i][cav_id][timestamp] = OrderedDict()

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = self.all_yamls[scenario_folder][cav_id][timestamp]
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = os.path.join(cav_path, f'{timestamp}.pcd')
                    self.scenario_database[i][cav_id][timestamp]['cameras'] = self.load_camera_files(cav_path, timestamp)
                    self.scenario_database[i][cav_id][timestamp]['scenario_folder'] = scenario_folder

                    # update the vehicles with the visibility categoryv
                    vehicles = self.scenario_database[i][cav_id][timestamp]['yaml']['vehicles']
                    # self.scenario_database[i][cav_id][timestamp]['yaml']['vehicles'] = categorize_vehicle_visibility_by_camera_props(vehicles, self.camera_dection_criteria_config)
                    # self.scenario_database[i][cav_id][timestamp]['yaml']['vehicles'] = categorize_vehicle_visibility_by_lidar_hits(vehicles, self.lidar_detection_criteria_config)
                    self.scenario_database[i][cav_id][timestamp]['yaml']['vehicles'] = categorize_by_kitti_criteria(vehicles, self.kitti_detection_criteria)

                if j == 0:
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False

    @staticmethod
    def load_camera_files(cav_path, timestamp):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + '_camera0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + '_camera1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + '_camera2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + '_camera3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        raise NotImplementedError


    def retrieve_by_idx(self, idx):
        """
        Retrieve the scenario index and timstamp by a single idx
        .
        Parameters
        ----------
        idx : int
            Idx among all frames.

        Returns
        -------
        scenario database and timestamp.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]

        return scenario_database, scenario_index, timestamp_index

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        timestamp_key = list(next(iter(scenario_database.values())).keys())[timestamp_index]

        return timestamp_key


    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if self.async_mode == 'real':
            # noise/time is in ms unit
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == 'sim':
            time_delay = np.abs(self.async_overhead)

        # todo: current 10hz, we may consider 20hz in the future
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0


    def reform_camera_param(self, cav_content, ego_content, timestamp):
        """
        Load camera extrinsic and intrinsic into a propoer format. todo:
        Enable delay and localization error.

        Returns
        -------
        The camera params dictionary.
        """
        camera_params = OrderedDict()

        cav_params = cav_content[timestamp]['yaml']
        ego_params = ego_content[timestamp]['yaml']
        ego_lidar_pose = ego_params['lidar_pose']
        ego_pose = ego_params['true_ego_pos']

        # load each camera's world coordinates, extrinsic (lidar to camera)
        # pose and intrinsics (the same for all cameras).

        for i in range(4):
            camera_coords = cav_params['camera%d' % i]['cords']
            camera_extrinsic = np.array(
                cav_params['camera%d' % i]['extrinsic'])
            camera_extrinsic_to_ego_lidar = x1_to_x2(camera_coords,
                                                     ego_lidar_pose)
            camera_extrinsic_to_ego = x1_to_x2(camera_coords,
                                               ego_pose)

            camera_intrinsic = np.array(
                cav_params['camera%d' % i]['intrinsic'])

            cur_camera_param = {'camera_coords': camera_coords,
                                'camera_extrinsic': camera_extrinsic,
                                'camera_intrinsic': camera_intrinsic,
                                'camera_extrinsic_to_ego_lidar':
                                    camera_extrinsic_to_ego_lidar,
                                'camera_extrinsic_to_ego':
                                    camera_extrinsic_to_ego,
                                'image_path': cav_content[timestamp]['cameras'][i]}
            camera_params.update({'camera%d' % i: cur_camera_param})

        return camera_params
    

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0],
                      pose[1] + xyz_noise[1],
                      pose[2] + xyz_noise[2],
                      pose[3],
                      pose[4] + ryp_std[1],
                      pose[5]]
        return noise_pose


    def reform_param(self, cav_content, ego_content, timestamp_cur,
                           timestamp_delay, cur_ego_pose_flag):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """

        cur_params = cav_content[timestamp_cur]['yaml']
        delay_params = copy.deepcopy(cav_content[timestamp_delay]['yaml'])

        cur_ego_params = ego_content[timestamp_cur]['yaml']
        delay_ego_params = copy.deepcopy(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params['lidar_pose']

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']

        if not cav_content['ego'] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose,
                                                      self.xyz_noise_std,
                                                      self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose,
                                                    self.xyz_noise_std,
                                                    self.ryp_noise_std)

        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose,
                                                 cur_ego_lidar_pose)
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,
                                            cur_ego_lidar_pose)

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_params['vehicles'] = cur_params['vehicles']
        delay_params['cav_vehicle'] = cur_params['cav_vehicle']
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = \
            gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix

        return delay_params


    def retrieve_base_data(self, idx, cur_ego_pose_flag=True, load_camera_data=False, load_lidar_data=False, load_ego_only=False):
        """
        Retrieves the base data for a given timestamp index or (scenario, timestamp) tuple.
        """
        # assert load_camera_data or load_lidar_data, 'At least one of the data should be loaded'

        # Determine scenario and timestamp index
        if isinstance(idx, int):
            scenario_database, scenario_index, timestamp_index = self.retrieve_by_idx(idx)
        elif isinstance(idx, tuple):
            scenario_database = self.scenario_database[idx[0]]
            timestamp_index = idx[1]
        else:
            raise ValueError('Index must be an int or tuple')

        # Retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)

        # Find the ego vehicle
        ego_cav_content = [cav_content for cav_content in scenario_database.values() if cav_content['ego']][0]

        data = OrderedDict()
        for cav_id, cav_content in scenario_database.items():
            if load_ego_only and not cav_content['ego']:
                continue

            # Calculate delay and adjusted timestamp
            timestamp_delay = self.time_delay_calculation(cav_content['ego'])
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(scenario_database, timestamp_index_delay)

            # check if CAV has data for the timestamp (otherwise it is possibly out of communication range)
            if timestamp_key not in cav_content:
                continue

            # Retrieve and structure CAV data
            cav_data = self.retrieve_cav_data(
                cav_content, ego_cav_content, timestamp_key, timestamp_key_delay, cur_ego_pose_flag, load_lidar_data, load_camera_data
            )
            
            if cav_data is None:
                continue

            data[cav_id] = cav_data

            # Store timestamp-related data
            data[cav_id]['timestamp_key'] = timestamp_key
            data[cav_id]['time_delay'] = timestamp_delay
            data[cav_id]['timestamp_key_delay'] = timestamp_key_delay
            data[cav_id]['scenario_folder'] = cav_content[timestamp_key]['scenario_folder']
        
        # check if ['params']['vehicles'].keys() of all entries are equal
        vehicle_ids = [list(data[cav_id]['params']['vehicles'].keys())[len(scenario_database.keys())-1:] for cav_id in data]

        return data, scenario_index

    def retrieve_cav_data(self, cav_content, ego_cav_content, timestamp_key, timestamp_key_delay, cur_ego_pose_flag, load_lidar_data, load_camera_data):
        """
        Helper function to retrieve and structure CAV data for a given timestamp.
        """
        if timestamp_key_delay not in cav_content or timestamp_key not in cav_content:
            return None

        cav_data = OrderedDict()
        cav_data['ego'] = cav_content['ego']
        cav_data['params'] = self.reform_param(cav_content, ego_cav_content, timestamp_key, timestamp_key_delay, cur_ego_pose_flag)
        cav_data['camera_params'] = self.reform_camera_param(cav_content, ego_cav_content, timestamp_key)
        
        if load_lidar_data:
            cav_data['lidar_np'] = pcd_to_np(cav_content[timestamp_key_delay]['lidar'], self.lidar_cache_container)
            # cav_data['lidar_np'] = pcd_to_np(cav_content[timestamp_key_delay]['lidar'], self.worker_lidar_cache)

        if load_camera_data:
            cav_data['camera_np'] = load_rgb_from_files(cav_content[timestamp_key_delay]['cameras'], self.camera_cache_container)
        
        return cav_data

    def get_pairwise_transformation(self, base_data_dict, max_cav, proj_first=False):
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
        pairwise_t_matrix[:, :] = np.identity(4)

        # if proj_first:
        #     # if lidar projected to ego first, then the pairwise matrix
        #     # becomes identity
        #     pairwise_t_matrix[:, :] = np.identity(4)
        # else:
        #     t_list = []

        #     # save all transformation matrix in a list in order first.
        #     for cav_id, cav_content in base_data_dict.items():
        #         t_list.append(cav_content['params']['transformation_matrix'])

        #     for i in range(len(t_list)):
        #         for j in range(len(t_list)):
        #             # identity matrix to self
        #             if i == j:
        #                 t_matrix = np.eye(4)
        #                 pairwise_t_matrix[i, j] = t_matrix
        #                 continue
        #             # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
        #             t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
        #             pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix


if __name__ == '__main__':
    config_file = r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/hypes_yaml/aaa_test_camera.yaml'
    params = load_yaml(config_file)

    dataset = BaseDataset(params, visualize=False, train=True, validate=False)
    # test
    import time
    start = time.time()
    dataset.retrieve_base_data(0)
    print(time.time() - start)
