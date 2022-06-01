# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
import matplotlib.pyplot as plt
from opencood.visualization.vis_utils import draw_points_boxes_plt


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=6,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=False,
                              drop_last=True)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=6,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        # model = torch.nnDataParallel(model, device_ids=[0, 1])

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model, state_dict = train_utils.load_saved_model(saved_path, model)
        if 'optimizer' in state_dict:
            optimizer.load_state_dict(state_dict['optimizer'])
        if 'scheduler' in state_dict:
            scheduler.load_state_dict(state_dict['scheduler'])

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):

        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            # if batch_data['ego']['record_len'].sum() > 3:
            #     continue
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            ouput_dict = model(batch_data['ego'])
            # first argument is always your output dictionary,
            # second argument is always your label dictionary.
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, len(train_loader), writer)

            ##########
            vis_save_path = os.path.join(opt.model_dir, 'vis')
            if not os.path.exists(vis_save_path):
                os.makedirs(vis_save_path)
            vis_save_path = os.path.join(vis_save_path, 'tmp.png')
            ########PLOT###########

            points = batch_data['ego']['origin_lidar'].cpu().numpy()
            points = points[:, 1:]
            gt_boxes = batch_data['ego']['object_bbx_center'][0][batch_data['ego']['object_bbx_mask'][0].bool()].cpu().numpy()
            gt_boxes = gt_boxes[:, [0, 1, 2, 5, 4, 3, 6]]
            draw_points_boxes_plt(pc_range=[-140.8, -41.6, -3, 140.8, 41.6, 1],
                                  points=points, boxes_gt=gt_boxes, save_path=vis_save_path)

            # boxes_pred = pred_box_tensor.cpu().numpy()
            # boxes_gt = gt_box_tensor.cpu().numpy()
            # fig = plt.figure(figsize=(15, 6))
            # ax = fig.add_subplot(111)
            # ax.plot(points[:, 0], points[:, 1], '.y', markersize=0.1)
            # ax.axis('equal')
            # for p, g in zip(boxes_pred, boxes_gt):
            #     plt.plot(g[[0, 1, 2, 3, 0], 0], g[[0, 1, 2, 3, 0], 1], 'g', markersize=1)
            # for p, g in zip(boxes_pred, boxes_gt):
            #     plt.plot(p[[0, 1, 2, 3, 0], 0], p[[0, 1, 2, 3, 0], 1], 'r', markersize=0.1)
            # plt.savefig(vis_save_path)
            # plt.close()
            #######################

            # back-propagation
            final_loss.backward()
            optimizer.step()
            scheduler.step()

        # if epoch % hypes['train_params']['eval_freq'] == 0:
        #     valid_ave_loss = []
        #
        #     with torch.no_grad():
        #         for i, batch_data in enumerate(val_loader):
        #             model.eval()
        #
        #             batch_data = train_utils.to_device(batch_data, device)
        #             ouput_dict = model(batch_data['ego'])
        #
        #             final_loss = criterion(ouput_dict,
        #                                    batch_data['ego']['label_dict'])
        #             valid_ave_loss.append(final_loss.item())
        #     valid_ave_loss = statistics.mean(valid_ave_loss)
        #     print('At epoch %d, the validation loss is %f' % (epoch,
        #                                                       valid_ave_loss))
        #     writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
