"""
Basedataset class for lidar data pre-processing
"""
from collections import OrderedDict
import numpy as np
import random

from opencood.utils.pcd_utils import pcd_to_np
from opencood.utils.camera_utils import load_rgb_from_files
from opencood.utils.transformation_utils import calculate_prev_pose_offset
from opencood.data_utils.datasets.basedataset import BaseDataset


class BaseTemporalDataset(BaseDataset):
    """
    Base dataset for all kinds of fusion. Mainly used to assign correct
    index. Additionally, it samples multiple frames for each scenario ().

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the dataset is used for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    """

    def __init__(self, params, visualize, train=True, validate=False, preload_lidar_files=False, preload_camera_files=False, **kwargs):       
        if 'queue_length' in params['fusion']['args']:
            self.queue_length = max(1, params['fusion']['args']['queue_length'])
        else:
            self.queue_length = 1
        
        if 'timestamp_offset' in params['fusion']['args']:
            self.timestamp_offset = max(0, params['fusion']['args']['timestamp_offset'])
        else:
            self.timestamp_offset = 0
        
        if 'timestamp_offset_mean' in params['fusion']['args']:
            self.timestamp_offset_mean = params['fusion']['args']['timestamp_offset_mean']
            self.timestamp_offset_std = params['fusion']['args']['timestamp_offset_std']
        else:
            self.timestamp_offset_mean = 0
            self.timestamp_offset_std = 0
        
        # all previous timestamps are with ego data only (no cooperation)
        # only the current timestamp (last data point) has cooperation
        self.temporal_ego_only = params['fusion']['args']['temporal_ego_only'] if 'temporal_ego_only' in params['fusion']['args'] else False

        # CAV communication dropout
        self.comm_dropout = params['fusion']['args']['communication_dropout'] if 'communication_dropout' in params['fusion']['args'] else 0

        super(BaseTemporalDataset, self).__init__(params, visualize, train, validate, preload_lidar_files=preload_lidar_files, preload_camera_files=preload_camera_files, **kwargs)

        self._apply_communication_dropout()


    def _apply_communication_dropout(self):
        """
        We delete/drop the CAVs immediately before we use them.
        This ensures that historical frames have the same CAVs especially important for evaluation.
        """
        if self.comm_dropout <= 0:
            return

        # first CAV (dict entry) is the ego vehicle
        for scenario_id in self.scenario_database:
            # ego is first (we never drop the ego)
            cav_ids = list(self.scenario_database[scenario_id].keys())[1:]
            # randomly drop depending on the comm_dropout percentage
            for cav_id in cav_ids:
                # timestamps
                timestamp_to_del = []
                for timestamp_key in self.scenario_database[scenario_id][cav_id]:
                    if timestamp_key == 'ego':
                        continue
                    if random.random() < self.comm_dropout:
                        timestamp_to_del.append(timestamp_key)
                # delete the timestamps
                for timestamp_key in timestamp_to_del:
                    del self.scenario_database[scenario_id][cav_id][timestamp_key]

        
    def reinitialize(self):
        super().reinitialize()
        self._apply_communication_dropout()

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        raise NotImplementedError

    def retrieve_base_data_before(self, scenario_index, timestamp_index, cur_timestamp_key, cur_ego_pose_flag=True, load_camera_data=False, load_lidar_data=False, load_ego_only=True):
        """
        Retrieves base data for timestamps prior to the current one.
        """
        assert load_camera_data or load_lidar_data, 'At least one of the data should be loaded'

        scenario_database = self.scenario_database[scenario_index]
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)
        ego_cav_content = [cav_content for cav_content in scenario_database.values() if cav_content['ego']][0]

        data = OrderedDict()
        for cav_id, cav_content in scenario_database.items():
            if load_ego_only and not cav_content['ego']:
                continue

            # Calculate delay and adjusted timestamp
            timestamp_delay = self.time_delay_calculation(cav_content['ego'])
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(scenario_database, timestamp_index_delay)
          
            # we try to get the current timestamp for transformation matrix calculation
            # but if it does not exist, we try to get the latest timestamp possible
            if cur_timestamp_key not in cav_content:
                int_key_delay = int(timestamp_key_delay)
                latest_timestamp_key = int(cur_timestamp_key) -2  # 2 is the timestamp step size
                # all between latest and timestamp_key_delay (format to 6 digits with leading zeros)
                possible_timestamps = [f'{i:06d}' for i in range(latest_timestamp_key, int_key_delay, -2)]
                # check if the timestamps are available
                possible_timestamps = [pt for pt in possible_timestamps if pt in cav_content]

                if len(possible_timestamps) == 0:
                    cur_timestamp_key = timestamp_key_delay
                else:
                    cur_timestamp_key = possible_timestamps[-1]

            # Retrieve and structure CAV data
            cav_data = self.retrieve_cav_data(
                cav_content, ego_cav_content, cur_timestamp_key, timestamp_key_delay, cur_ego_pose_flag, load_lidar_data, load_camera_data
            )

            if cav_data is None:
                continue
        
            data[cav_id] = cav_data

            # Store timestamp-related data
            data[cav_id]['timestamp_key'] = timestamp_key
            data[cav_id]['time_delay'] = timestamp_delay
            data[cav_id]['timestamp_key_delay'] = timestamp_key_delay
            data[cav_id]['scenario_folder'] = cav_content[timestamp_key]['scenario_folder']
        
        return data

    def retrieve_temporal_data(self, idx, cur_ego_pose_flag=True, load_camera_data=False, load_lidar_data=False):
        assert load_camera_data or load_lidar_data, 'At least one of the data should be loaded'
        # if the queue length is set to 1, then the vehicle offset is already the connection to the previous car
        # otherwise it starts with the second vehicle (e.g. the first vehicle offset is the identity matrix)

        # we loop the accumulated length list to see get the scenario index
        if isinstance(idx, int):
            scenario_database, scenario_index, timestamp_indices, prev_bev_exists, timestamp_offset = self.retrieve_temporal_by_idx(idx)
        else:
            import sys
            sys.exit('Index has to be a int')

        last_frame_timestamp_key = self.return_timestamp_key(scenario_database, timestamp_indices[-1])
        
        data_queue = []
        # Load files for all timestamps
        for i, timestamp_index in enumerate(timestamp_indices):
            if i < len(timestamp_indices) - 1:
                data = self.retrieve_base_data_before(
                    scenario_index, timestamp_index, last_frame_timestamp_key, cur_ego_pose_flag,
                    load_camera_data, load_lidar_data,
                    load_ego_only=self.temporal_ego_only)
            else:
                scenario_idx_length = 0 if scenario_index == 0 else self.len_record[scenario_index - 1]
                true_idx = timestamp_index + scenario_idx_length
                data, _ = self.retrieve_base_data(
                    true_idx, cur_ego_pose_flag,
                    load_camera_data, load_lidar_data,
                    load_ego_only=False) # for the latest frame always all CAV
            
            # add temporal vehicle entry (the vehicles of the frame from the lidar data)
            timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)

            for v_id in data:
                data[v_id]['params']['temporal_vehicles'] = self.scenario_database[scenario_index][v_id][timestamp_key]['yaml']['vehicles']
                data[v_id]['params']['temporal_cav_vehicle'] = self.scenario_database[scenario_index][v_id][timestamp_key]['yaml']['cav_vehicle']

            data_queue.append(data)
        
        return data_queue

    @staticmethod
    def find_ego_pose(base_data_dict):
        """
        Find the ego vehicle id and corresponding LiDAR pose from all cavs.

        Parameters
        ----------
        base_data_dict : dict
            The dictionary contains all basic information of all cavs.

        Returns
        -------
        ego vehicle id and the corresponding lidar pose.
        """

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        return ego_id, ego_lidar_pose


    def retrieve_temporal_by_idx(self, idx):
        """
        Retrieve the scenario index and timstamp by a single idx
        .
        Parameters
        ----------
        idx : int
            Idx among all frames.
            We use the query length to get the previous (or subsequent) timestamps

        Returns
        -------
        scenario database and timestamp.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                _len_record = self.len_record[i - 1] if i > 0 else 0
                scenario_index = i
                timestamp_index = idx if i == 0 else idx - _len_record
                max_idx = self.len_record[i] - _len_record - 1
                break
        
        if self.timestamp_offset_mean > 0 and self.timestamp_offset_std > 0:
            timestamp_offset = max(0, int(np.random.normal(self.timestamp_offset_mean, self.timestamp_offset_std)))
        else:
            timestamp_offset = self.timestamp_offset
        
        # keep the timestamp indices between min_idx and max_idx
        span = (self.queue_length - 1) * timestamp_offset + self.queue_length

        if span > max_idx:
            timestamp_index = 0
            timestamp_offset = (max_idx - self.queue_length) // (self.queue_length - 1)
            span = (self.queue_length - 1) * timestamp_offset + self.queue_length

        # check if its in between min and max idx
        if span > 1:
            if (timestamp_index + span) > max_idx:
                timestamp_index = max(0, timestamp_index - span)

        scenario_database = self.scenario_database[scenario_index]

        timestamp_indices = [timestamp_index + i * (timestamp_offset + 1) for i in range(self.queue_length)]
        
        prev_bevs_exists = []
        if timestamp_index == 0:
            prev_bevs_exists.append(False)
        else:
            prev_bevs_exists.append(True)
        
        for i in range(self.queue_length - 1):
            prev_bevs_exists.append(True)

        return scenario_database, scenario_index, timestamp_indices, prev_bevs_exists, timestamp_offset

    def augment(self, **kwargs):
        """
        """
        raise NotImplementedError

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
        raise NotImplementedError
