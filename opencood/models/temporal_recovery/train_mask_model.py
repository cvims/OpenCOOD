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
# from opencood.models.temporal_recovery.scope_mask_model import TemporalPointPillarScope
from opencood.models.temporal_recovery.scope_new_temporal_model import TemporalPointPillarScope
from opencood.loss.temporal_bce_loss import TemporalMaskBCELoss
from opencood.loss.temporal_point_pillar_loss import TemporalPointPillarLoss


def create_temporal_result_stat_dict():
    result_stat = {0.3: {'hits': 0, 'no_hits': 0},
                   0.5: {'hits': 0, 'no_hits': 0},
                   0.7: {'hits': 0, 'no_hits': 0}}
    return result_stat


def main():
    eval_utils.set_random_seed(0)

    # runs/temporal_mask_model/{current_timestamp}
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    SAVE_PATH = os.path.join('runs', 'temporal_mask_model', timestamp)

    os.makedirs(SAVE_PATH, exist_ok=True)

    TRAIN_DATA_PATH = '/data/public_datasets/OPV2V/original/train'
    VALIDATE_DATA_PATH = '/data/public_datasets/OPV2V/original/validate'

    # MODEL_DIR = r'/home/dominik/Git_Repos/Private/OpenCOOD/opencood/model_weights/SCOPE/weights/OPV2V'
    MODEL_DIR = r'/home/dominik/Git_Repos/Private/OpenCOOD/runs/temporal_mask_model/20241126180425'

    HYPES_YAML_FILE = os.path.join(MODEL_DIR, 'config.yaml')

    EPOCHS = 40
    
    TEMPORAL_STEPS = 4
    TEMPORAL_EGO_ONLY = False
    COMMUNICATION_DROPOUT = 0.5

    hypes = yaml_utils.load_yaml(HYPES_YAML_FILE, None)

    hypes['root_dir'] = TRAIN_DATA_PATH
    hypes['validate_dir'] = VALIDATE_DATA_PATH

    hypes['fusion']['args']['queue_length'] = TEMPORAL_STEPS
    hypes['fusion']['args']['temporal_ego_only'] = TEMPORAL_EGO_ONLY
    hypes['fusion']['args']['communication_dropout'] = COMMUNICATION_DROPOUT

    hypes['model']['args']['fusion_args']['communication']['thre'] = 0
    hypes['postprocess']['target_args']['score_threshold'] = 0.23
    hypes['wild_setting']['xyz_std'] = 0.2
    hypes['wild_setting']['ryp_std'] = 0.2

    hypes['train_params']['batch_size'] = 1
    hypes['train_params']['frame'] = TEMPORAL_STEPS - 1
    hypes['model']['args']['fusion_args']['frame'] = TEMPORAL_STEPS - 1

    use_scenarios_idx = None
    use_scenarios_idx = [
        0, 5, 11, 14, 22, 24, 35, 40, 41, 42
    ]

    print('Dataset Building')
    opencood_dataset_train = build_dataset(
        hypes, visualize=False, train=True,
        use_scenarios_idx=use_scenarios_idx,
        preload_lidar_files=False
    )

    data_loader_train = DataLoader(
        opencood_dataset_train,
        batch_size=2,
        num_workers=16,
        collate_fn=opencood_dataset_train.collate_batch,
        shuffle=True,
        pin_memory=False,
        drop_last=False
    )

    hypes['fusion']['args']['communication_dropout'] = 0.0

    opencood_dataset_validate = build_dataset(
        hypes, visualize=False, train=False,
        # use_scenarios_idx=use_scenarios_idx,
        preload_lidar_files=False
    )

    data_loader_validate = DataLoader(
        opencood_dataset_validate,
        batch_size=1,
        num_workers=16,
        collate_fn=opencood_dataset_validate.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

    print('Creating Model')
    model = TemporalPointPillarScope(hypes['model']['args'])
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    _, model = train_utils.load_saved_model(MODEL_DIR, model)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    if torch.cuda.is_available():
        model = model.cuda()
        model.to(device)

    # freeze model weights
    for param in model.parameters():
        param.requires_grad = False
    
    # unfreeze model.temporal_mask_model
    for param in model.temporal_mask_model.parameters():
        param.requires_grad = True

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    num_steps = len(data_loader_train)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    mask_model_criterion = TemporalMaskBCELoss(
        hypes,
        pos_weight=100.0,
        neg_weight=1.0
    )

    hypes['loss']['args']['temporal_cls_weight'] = 1.0
    hypes['loss']['args']['temporal_reg'] = 10.0
    scope_default_criterion = TemporalPointPillarLoss(hypes['loss']['args'])

    for epoch in range(EPOCHS):
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        pbar_train = tqdm.tqdm(total=len(data_loader_train), leave=True)

        for i, batch_data_list in enumerate(data_loader_train):
            # the model will be evaluation mode during validation
            model.eval()
            model.temporal_mask_model.train()
            model.late_fusion.train()
            model.cls_head.train()
            model.reg_head.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = batch_data_list[-1]

            batch_data_list = train_utils.to_device(batch_data_list, device)
            batch_data = train_utils.to_device(batch_data, device)

            output = model(batch_data_list)

            # mask_model_loss = mask_model_criterion(
            #     output['temporal_mask'],
            #     batch_data['ego']['object_bbx_center'],
            #     batch_data['ego']['object_detection_info_mapping']
            # )

            scope_loss = scope_default_criterion(
                output,
                batch_data['ego']['label_dict'],
                batch_data['ego']['temporal_label_dict']
            )

            # final_loss = mask_model_loss + scope_loss
            final_loss = scope_loss

            # mask_model_criterion.logging(epoch, i, len(data_loader_train), None, pbar=pbar_train)
            scope_default_criterion.logging(epoch, i, len(data_loader_train), None, pbar=pbar_train)

            pbar_train.update(1)

            final_loss.backward()
            optimizer.step()
            scheduler.step(epoch)

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)


        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                os.path.join(SAVE_PATH, 'net_epoch%d.pth' % (epoch + 1)))
        

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            temporal_result_stats = create_temporal_result_stat_dict()

            pbar_val = tqdm.tqdm(total=len(data_loader_validate), leave=True)
            with torch.no_grad():
                for batch_data_list in data_loader_validate:
                    model.eval()
                    model.temporal_mask_model.eval()
                    model.late_fusion.eval()
                    model.cls_head.eval()
                    model.reg_head.eval()

                    batch_data = batch_data_list[-1]

                    batch_data_list = train_utils.to_device(batch_data_list, device)
                    batch_data = train_utils.to_device(batch_data, device)

                    output = model(batch_data_list)

                    # mask_model_loss = mask_model_criterion(
                    #     output['temporal_mask'],
                    #     batch_data['ego']['object_bbx_center'],
                    #     batch_data['ego']['object_detection_info_mapping']
                    # )

                    scope_loss = scope_default_criterion(
                        output,
                        batch_data['ego']['label_dict'],
                        batch_data['ego']['temporal_label_dict']
                    )

                    valid_ave_loss.append(scope_loss.item())

                    # temporal evaluation
                    pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                    opencood_dataset_validate.post_process(batch_data, {'ego': output})

                    _, gt_object_ids = opencood_dataset_validate.post_processor.generate_gt_bbx(batch_data)
                    gt_object_ids_criteria = batch_data['ego']['object_detection_info_mapping'][-1]
                    gt_object_ids_criteria = {o_id: gt_object_ids_criteria[o_id] for o_id in gt_object_ids}

                    for iou_thre in [0.3, 0.5, 0.7]:
                        eval_utils.calculate_temporal_recovered_hits(
                            pred_box_tensor,
                            pred_score,
                            gt_box_tensor,
                            temporal_result_stats,
                            iou_thre,
                            gt_object_ids_criteria)

                    pbar_val.update(1)
            
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            print('At epoch %d, the temporal evaluation is %s' % (epoch,
                                                              temporal_result_stats))


if __name__ == '__main__':
    main()
