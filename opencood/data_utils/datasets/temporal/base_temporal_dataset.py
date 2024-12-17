"""
Basedataset class for lidar data pre-processing
"""
from collections import OrderedDict
import numpy as np
import random
import tqdm

from opencood.utils.pcd_utils import pcd_to_np
from opencood.utils.camera_utils import load_rgb_from_files
from opencood.utils.transformation_utils import calculate_prev_pose_offset, calculate_rotation, x1_to_x2
from opencood.data_utils.datasets.basedataset import BaseDataset
from opencood.data_utils.datasets import GT_RANGE


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

        # Only load temporal potential data
        self.temporal_potential_only = params['fusion']['args']['temporal_potential_only'] if 'temporal_potential_only' in params['fusion']['args'] else False

        super(BaseTemporalDataset, self).__init__(params, visualize, train, validate, preload_lidar_files=preload_lidar_files, preload_camera_files=preload_camera_files, **kwargs)

        # self._apply_communication_dropout()
        # self._apply_temporal_potential_only(GT_RANGE)


    def _apply_temporal_potential_only(self, range_filter):
        """
        We delete data that does not have temporal potential.
        We use the center of the vehicles and the range filter (from center of ego)
        to determine if the vehicle is in the range.
        range_filter: [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.indices_with_temporal_potential = []
        self.adjusted_len_record = self.len_record

        if not self.temporal_potential_only:
            return

        def get_frame_opv2v_visible_vehicles(frame_data):
            # Create a mask by stacking conditions along the last axis and using np.all
            min_bounds = np.array(range_filter[:3])
            max_bounds = np.array(range_filter[3:])

            ego_id, ego_data = [(v_id, frame_data[v_id]) for v_id, v_data in frame_data.items() if v_data['ego']][0]

            # find ego lidar pose
            ego_pose = np.asarray(ego_data['params']['lidar_pose'], dtype=np.float32)

            # calculate the distance between the ego and the other vehicles
            all_vehicles = ego_data['params']['temporal_vehicles']

            # in_range_vehicles = [v_id for v_id in all_vehicles if v_id != ego_id]
            in_range_vehicles = set()
            for v_id in all_vehicles:
                if v_id == ego_id:
                    continue
                veh_loc = np.asarray(all_vehicles[v_id]['location'], dtype=np.float32)
                veh_center = np.asarray(all_vehicles[v_id]['center'], dtype=np.float32)
                veh_angle = np.asarray(all_vehicles[v_id]['angle'], dtype=np.float32)

                vehicle_pose = [
                    veh_loc[0] + veh_center[0],
                    veh_loc[1] + veh_center[1],
                    veh_loc[2] + veh_center[2],
                    veh_angle[0], veh_angle[1], veh_angle[2]
                ]

                loc_offset = x1_to_x2(vehicle_pose, ego_pose)[:,-1][:2]

                # Efficient masking by checking all conditions in one pass
                in_range = np.all((loc_offset[:2] >= min_bounds[:2]) & (loc_offset[:2] <= max_bounds[:2]))

                if in_range:
                    in_range_vehicles.add(v_id)

            # preset the opv2v_visible flag to False
            opv2v_visible_vehicles = {v_id: False for v_id in in_range_vehicles}

            # check "opv2v_visible" across all cavs (incl. ego)
            for cav_id, cav_data in frame_data.items():
                cav_vehicles = cav_data['params']['temporal_vehicles']
                for v_id in cav_vehicles:
                    if v_id not in opv2v_visible_vehicles:
                        continue
                    if cav_vehicles[v_id]['opv2v_visible'] == True:
                        opv2v_visible_vehicles[v_id] = True
            
            return opv2v_visible_vehicles


        for idx in tqdm.tqdm(range(self.len_record[-1]), desc='Filtering for temporal potential'):
            data_queue = self.retrieve_temporal_data(idx, cur_ego_pose_flag=True, load_camera_data=False, load_lidar_data=False)

            # objects ids of the current frame
            latest_frame_data = data_queue[-1]
            last_frame_opv2v_visible_vehicles = get_frame_opv2v_visible_vehicles(latest_frame_data)

            # get temporal potential vehicles from previous frames in data queue
            # find if opv2v_visible is True for any of the vehicles in the previous frames
            has_temporal_potential = False
            for i in range(len(data_queue) - 1):
                if has_temporal_potential:
                    break

                frame_data = data_queue[i]
                opv2v_visible_vehicles = get_frame_opv2v_visible_vehicles(frame_data)

                for v_id in last_frame_opv2v_visible_vehicles:
                    # if not visible in last_frame, we check if it is visible in the previous frame
                    last_frame_opv2v_visible = last_frame_opv2v_visible_vehicles[v_id]
                    if not last_frame_opv2v_visible:
                        # check if the vehicle was NOT visible in the previous frame
                        if v_id in opv2v_visible_vehicles and opv2v_visible_vehicles[v_id]:
                            has_temporal_potential = True
                            break
            
            if has_temporal_potential:
                self.indices_with_temporal_potential.append(idx)
        
        # update len_records
        last_index = 0
        new_len_records = [0 for _ in range(len(self.len_record))]
        for i, ele in enumerate(self.len_record):
            prev_len_record = new_len_records[i - 1] if i > 0 else 0
            new_len_records[i] = prev_len_record
            for j, idx in enumerate(self.indices_with_temporal_potential[last_index:]):
                if idx < ele:
                    new_len_records[i] += 1
                else:
                    last_index += j
                    break
                if j + last_index == len(self.indices_with_temporal_potential) - 1:
                    last_index += j + 1
                    break
        
        self.adjusted_len_record = new_len_records


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
        self._apply_temporal_potential_only(GT_RANGE)
    
    def get_corrected_idx(self, idx):
        if idx < len(self.indices_with_temporal_potential):
            return self.indices_with_temporal_potential[idx]
        else:
            return idx

    def __len__(self):
        return self.adjusted_len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        raise NotImplementedError

    def retrieve_base_data_before(self, scenario_index, timestamp_index, cur_timestamp_key, cur_ego_pose_flag=True, load_camera_data=False, load_lidar_data=False, load_ego_only=True):
        """
        Retrieves base data for timestamps prior to the current one.
        """
        # assert load_camera_data or load_lidar_data, 'At least one of the data should be loaded'

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
        # assert load_camera_data or load_lidar_data, 'At least one of the data should be loaded'
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
