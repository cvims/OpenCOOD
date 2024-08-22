import weakref
import numpy as np
import os
import cv2
import carla
from carla import ColorConverter as cc
from logreplay.sensors.base_sensor import BaseSensor
from logreplay.sensors.semantic_lidar import SemanticLidar
from logreplay.sensors.utils import get_camera_intrinsic
from opencood.hypes_yaml.yaml_utils import save_yaml_wo_overwriting


class DepthCamera(BaseSensor):
    def __init__(self, agent_id, vehicle, world, config, global_position):
        super().__init__(agent_id, vehicle, world, config, global_position)
        if vehicle is not None:
            world = vehicle.get_world()

        self.agent_id = agent_id

        blueprint = world.get_blueprint_library(). \
            find('sensor.camera.depth')
        blueprint.set_attribute('fov', str(config['fov']))
        blueprint.set_attribute('image_size_x', str(config['image_size_x']))
        blueprint.set_attribute('image_size_y', str(config['image_size_y']))

        relative_position = config['relative_pose']
        spawn_point = SemanticLidar.spawn_point_estimation(
            relative_position, global_position)
        self.name = 'depth_camera' + str(relative_position)

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        self.image = None
        self.timstamp = None
        self.frame = 0

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: DepthCamera._on_depth_image_event(
                weak_self, image))
    
    @staticmethod
    def _on_depth_image_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        
        image = np.frombuffer(event.raw_data, dtype=np.dtype("uint8"))
        image = np.reshape(image, (event.height, event.width, 4))
        image = image[:, :, :3]
        image = image.astype(np.float32)

        self.raw_image = image

        self.image = image
        self.frame = event.frame
        self.timestamp = event.timestamp

    def data_dump(self, output_root, cur_timestamp):
        # dump the image
        raw_output_label_name = os.path.join(
            output_root, cur_timestamp + '_depth_raw.png')       
        
        # raw image to disk
        cv2.imwrite(raw_output_label_name, self.raw_image)

        # dump the yaml
        save_yaml_name = os.path.join(
            output_root, cur_timestamp + '_additional.yaml')
        
        # intrinsic
        camera_intrinsic = get_camera_intrinsic(self.sensor)
        # pose under world coordinate system
        camera_transformation = self.sensor.get_transform()
        cords = [camera_transformation.location.x,
                camera_transformation.location.y,
                camera_transformation.location.z,
                camera_transformation.rotation.roll,
                camera_transformation.rotation.yaw,
                camera_transformation.rotation.pitch]

        bev_sem_cam_info = {self.name:
                                {
                                    'cords': cords,
                                    'extrinsic': camera_transformation,
                                    'intrinsic': camera_intrinsic
                                }}

        save_yaml_wo_overwriting(bev_sem_cam_info, save_yaml_name)
