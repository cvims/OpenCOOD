# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

# from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
# from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
# from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
# from opencood.data_utils.datasets.intermediate_fusion_dataset_v2 import IntermediateFusionDatasetV2

# the final range for evaluation
GT_RANGE = [-140, -40, -3, 140, 40, 1]
CAMERA_GT_RANGE = [-50, -50, -3, 50, 50, 1]
# The communication range for cavs
COM_RANGE = 70

from opencood.data_utils.datasets.temporal.camera.temporal_camera_intermediate_dataset import TemporalCameraBEVIntermediateFusionDataset
from opencood.data_utils.datasets.temporal.lidar.temporal_lidar_intermediate_dataset import TemporalLidarIntermediateFusionDataset

__all__ = {
    # 'LateFusionDataset': LateFusionDataset,
    # 'EarlyFusionDataset': EarlyFusionDataset,
    # 'IntermediateFusionDataset': IntermediateFusionDataset,
    # 'IntermediateFusionDatasetV2': IntermediateFusionDatasetV2,
    'TemporalCameraBEVIntermediateFusionDataset': TemporalCameraBEVIntermediateFusionDataset,
    'TemporalLidarIntermediateFusionDataset': TemporalLidarIntermediateFusionDataset
}


def build_dataset(dataset_cfg, visualize=False, train=True, **kwargs):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in [
        'LateFusionDataset', 'EarlyFusionDataset',
        'IntermediateFusionDataset', 'IntermediateFusionDatasetV2',
        'TemporalCameraBEVIntermediateFusionDataset',
        'TemporalLidarIntermediateFusionDataset'
        ], error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        **kwargs
    )

    return dataset
