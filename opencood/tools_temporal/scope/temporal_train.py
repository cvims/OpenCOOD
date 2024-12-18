# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
import statistics
import argparse

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    
    opt = parser.parse_args()
    return opt


def create_temporal_result_stat_dict():
    result_stat = {0.3: {'hits': 0, 'no_hits': 0},
                   0.5: {'hits': 0, 'no_hits': 0},
                   0.7: {'hits': 0, 'no_hits': 0}}
    return result_stat


def main():
    opt = train_parser()

    eval_utils.set_random_seed(0)

    MODEL_DIR = opt.model_dir
    HALF_PRECISION = opt.half

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    EPOCHS = hypes['train_params']['epoches']

    PRETRAINED_MODEL_PATH = hypes['train_params']['pretrained_model'] if 'pretrained_model' in hypes['train_params'] else None

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(
        hypes, visualize=False, train=True,
        preload_lidar_files=False,
        use_scenarios_idx=None)
    opencood_validate_dataset = build_dataset(
        hypes, visualize=False, train=False,
        preload_lidar_files=False,
        )

    train_loader = DataLoader(opencood_train_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=16,
                                collate_fn=opencood_train_dataset.collate_batch,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=1,
                            num_workers=16,
                            collate_fn=opencood_train_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if MODEL_DIR:
        saved_path = MODEL_DIR
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
    elif PRETRAINED_MODEL_PATH:
        init_epoch = 0
        model = train_utils.load_saved_weights(PRETRAINED_MODEL_PATH, model)
        saved_path = train_utils.setup_train(hypes)
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if HALF_PRECISION:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')

    for epoch in range(init_epoch, EPOCHS):
        if epoch > 0 and hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)

        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        pbar_train = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data_list in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = batch_data_list[-1]

            batch_data_list = train_utils.to_device(batch_data_list, device)
            batch_data = train_utils.to_device(batch_data, device)

            if not HALF_PRECISION:
                output_dict = model(batch_data_list)
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(
                    output_dict,
                    batch_data['ego']['label_dict'],
                    batch_data['ego']['temporal_label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    output_dict = model(batch_data_list)
                    final_loss = criterion(
                        output_dict,
                        batch_data['ego']['label_dict'],
                        batch_data['ego']['temporal_label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar_train)
            pbar_train.update(1)

            if not HALF_PRECISION:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)


        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        best_valid_loss = 1e10
        if (epoch > 0 or hypes['train_params']['eval_freq'] == 1) and epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            temporal_result_stats = create_temporal_result_stat_dict()

            pbar_val = tqdm.tqdm(total=len(val_loader), leave=True)
            with torch.no_grad():
                for i, batch_data_list in enumerate(val_loader):
                    model.eval()

                    batch_data = batch_data_list[-1]

                    batch_data_list = train_utils.to_device(batch_data_list, device)
                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict = model(batch_data_list)

                    final_loss = criterion(
                        output_dict,
                        batch_data['ego']['label_dict'],
                        batch_data['ego']['temporal_label_dict'])
                    valid_ave_loss.append(final_loss.item())

                    # temporal evaluation
                    pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                    opencood_validate_dataset.post_process(batch_data, {'ego': output_dict})

                    _, gt_object_ids = opencood_validate_dataset.post_processor.generate_gt_bbx(batch_data)
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

            if valid_ave_loss < best_valid_loss:
                best_valid_loss = valid_ave_loss
                torch.save(model_without_ddp.state_dict(),
                    os.path.join(saved_path, 'best_model.pth'))

            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            print('At epoch %d, the temporal evaluation is %s' % (epoch,
                                                              temporal_result_stats))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
