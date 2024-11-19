import time
import os
import statistics

import torch
import tqdm
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils
from opencood.models.temporal_recovery.mask_model import TemporalMaskModel
from opencood.models.temporal_recovery.scope_backbone import PointPillarScopeCut


def main():
    eval_utils.set_random_seed(0)

    # runs/temporal_mask_model/{current_timestamp}
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    SAVE_PATH = os.path.join('runs', 'temporal_mask_model', timestamp)

    TRAIN_DATA_PATH = '/data/public_datasets/OPV2V/original/train'
    VALIDATE_DATA_PATH = '/data/public_datasets/OPV2V/original/validate'

    MODEL_DIR = r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/model_weights/SCOPE/weights/OPV2V'

    HYPES_YAML_FILE = os.path.join(MODEL_DIR, 'config.yaml')

    EPOCHS = 40
    
    TEMPORAL_STEPS = 4
    TEMPORAL_EGO_ONLY = False
    COMMUNICATION_DROPOUT = 0.25

    hypes = yaml_utils.load_yaml(HYPES_YAML_FILE, None)

    hypes['root_dir'] = TRAIN_DATA_PATH
    hypes['validate_dir'] = VALIDATE_DATA_PATH

    hypes['fusion']['args']['queue_length'] = TEMPORAL_STEPS
    hypes['fusion']['args']['temporal_ego_only'] = TEMPORAL_EGO_ONLY
    hypes['model']['args']['fusion_args']['communication_dropout'] = COMMUNICATION_DROPOUT

    hypes['model']['args']['fusion_args']['communication']['thre'] = 0
    hypes['postprocess']['target_args']['score_threshold'] = 0.23
    hypes['wild_setting']['xyz_std'] = 0.2
    hypes['wild_setting']['ryp_std'] = 0.2

    hypes['train_params']['batch_size'] = 1
    hypes['train_params']['frame'] = TEMPORAL_STEPS - 1
    hypes['model']['args']['fusion_args']['frame'] = TEMPORAL_STEPS - 1


    use_scenarios_idx = None

    print('Dataset Building')
    opencood_dataset_train = build_dataset(
        hypes, visualize=True, train=True,
        use_scenarios_idx=use_scenarios_idx,
        preload_lidar_files=False
    )

    data_loader_train = DataLoader(
        opencood_dataset_train,
        batch_size=2,
        num_workers=1,
        collate_fn=opencood_dataset_train.collate_batch,
        shuffle=True,
        pin_memory=False,
        drop_last=False
    )

    hypes['model']['args']['fusion_args']['communication_dropout'] = 0.0

    opencood_dataset_validate = build_dataset(
        hypes, visualize=True, train=False,
        use_scenarios_idx=use_scenarios_idx,
        preload_lidar_files=False
    )

    data_loader_validate = DataLoader(
        opencood_dataset_validate,
        batch_size=1,
        num_workers=1,
        collate_fn=opencood_dataset_validate.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

    print('Creating Model')
    model = PointPillarScopeCut(hypes['model']['args'])
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    _, model = train_utils.load_saved_model(MODEL_DIR, model)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # create temporalmaskmodel
    combined_model = TemporalMaskModel(model)

    if torch.cuda.is_available():
        combined_model = combined_model.cuda()
        combined_model.to(device)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, combined_model)
    # lr scheduler setup
    num_steps = len(data_loader_train)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    for epoch in range(EPOCHS):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        pbar_train = tqdm.tqdm(total=len(data_loader_train), leave=True)

        for i, batch_data_list in enumerate(data_loader_train):
            # the model will be evaluation mode during validation
            combined_model.train()
            combined_model.zero_grad()
            optimizer.zero_grad()

            batch_data = batch_data_list[-1]

            batch_data_list = train_utils.to_device(batch_data_list, device)
            batch_data = train_utils.to_device(batch_data, device)

            output_dict = combined_model(batch_data_list)

            # final_loss = 

            pbar_train.update(1)

            # final_loss.backward()
            optimizer.step()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(combined_model.state_dict(),
                os.path.join(SAVE_PATH, 'net_epoch%d.pth' % (epoch + 1)))
        

        # TODO EVALUATION LOOP
            
        

if __name__ == '__main__':
    main()
