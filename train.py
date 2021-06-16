from __future__ import print_function
import argparse
from math import log10

from libs.GANet.modules.GANet import MyLoss2
import sys
import numpy as np
import shutil
import os
import time
import skimage.io
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import WeightedRandomSampler
import wandb
import torchvision.utils as vutils
import torch.utils.data



# from models.GANet_deep import GANet
import torch.nn.functional as F
from dataloader.data import get_training_set, get_test_set
from torch.utils.tensorboard import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--threshold', type=float, default=3.0, help="threshold of error rates")
parser.add_argument('--val_height', type=int, default=576, help="val crop height")
parser.add_argument('--val_width', type=int, default=1344, help="val crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--cont', type=int, default=0, help="resume training from saved epoch")
parser.add_argument('--left_right', type=int, default=0, help="use right view for training. Default=False")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2048, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--eth3d', type=int, default=0, help='eth3d Dataset? Default=False')
parser.add_argument('--middlebury', type=int, default=0, help='middlebury dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
# parser.add_argument('--val_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
parser.add_argument('--training_list', type=str, default='./lists/sceneflow_train.list', help="training list")
parser.add_argument('--val_list', type=str, default='./lists/sceneflow_test_select.list', help="validation list")
parser.add_argument('--save_path', type=str, default='./checkpoint/', help="location to save models")
parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")
parser.add_argument('--visualize', type=int, default=0, help="Visualize Model")
parser.add_argument('--multi', type=int, default=0, help="Training on multiple datasets")
parser.add_argument('--eth3d_data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
parser.add_argument('--kitti_data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
parser.add_argument('--kitti2015_data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
parser.add_argument('--middlebury_data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
parser.add_argument('--eth3d_training_list', type=str, default='./lists/sceneflow_train.list',
                    help="eth3d training list")
parser.add_argument('--kitti_training_list', type=str, default='lists/kitti2012_train.list', help="kitti training list")
parser.add_argument('--kitti2015_training_list', type=str, default='lists/kitti2015_train.list',
                    help="kitti training list")
parser.add_argument('--middlebury_training_list', type=str, default='lists/middlebury_train.list',
                    help="kitti training list")
parser.add_argument('--multi_scale_loss', type=int, default=0, help='Loss to incorporate value for each scale')
parser.add_argument('--wbProj', type=str, default='ganet_ms', help="Wandb project name")
parser.add_argument('--wbRunName', type=str, default='ganet_ms', help="Wandb run name")
parser.add_argument('--wandb', type=int, default=0, help='Wandb logging enabled?')

opt = parser.parse_args()

print(opt)
if opt.model == 'GANet11':
    from models.GANet11 import GANet
elif opt.model == 'GANet_deep':
    from models.GANet_deep import GANet
elif opt.model == 'GANet_ms':
    from models.GANet_ms import GANet
else:
    raise Exception("No suitable model found ...")

cuda = opt.cuda
# cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
if opt.multi:

    print('===> Training on multiple datasets')
    train_set1 = get_training_set(opt.kitti_data_path, opt.kitti_training_list, [opt.crop_height, opt.crop_width],
                                  opt.left_right, True, opt.kitti2015, opt.eth3d, opt.middlebury, opt.shift)
    train_set2 = get_training_set(opt.eth3d_data_path, opt.eth3d_training_list, [opt.crop_height, opt.crop_width],
                                  opt.left_right, opt.kitti, opt.kitti2015, True, opt.middlebury, opt.shift)
    train_set3 = get_training_set(opt.middlebury_data_path, opt.middlebury_training_list,
                                  [opt.crop_height, opt.crop_width], opt.left_right, opt.kitti, opt.kitti2015,
                                  opt.eth3d, True, opt.shift)
    train_set4 = get_training_set(opt.kitti2015_data_path, opt.kitti2015_training_list,
                                  [opt.crop_height, opt.crop_width], opt.left_right, opt.kitti, True, opt.eth3d,
                                  opt.middlebury, opt.shift)
    test_set_multi = get_test_set(opt.kitti_data_path, opt.val_list, [384, 1344], opt.left_right, True, False, False,
                                  False)
    # test_set2 = get_test_set(opt.eth3d_data_path, opt.val_list, [576,960], opt.left_right, opt.kitti, opt.kitti2015, True, opt.middlebury)
    # test_set3 = get_test_set(opt.middlebury_data_path, opt.val_list, [576,960], opt.left_right, opt.kitti, opt.kitti2015, opt.eth3d, True)
    # test_set4 = get_test_set(opt.kitti2015_data_path, opt.val_list, [576,960], opt.left_right, opt.kitti, opt.kitti2015, opt.eth3d, True)

    train_set_multi = ConcatDataset((train_set1, train_set2, train_set3, train_set4))
    # test_set_multi = ConcatDataset((test_set1, test_set2, test_set3, test_set4))
    print("===> Total Size of Training Set: ", len(train_set_multi))

    sampler = WeightedRandomSampler([0.375, 0.125, 0.125, 0.375], train_set_multi.__len__(), True)
    training_data_loader = DataLoader(dataset=train_set_multi, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=False, sampler=sampler, drop_last=True)
    testing_data_loader = DataLoader(dataset=test_set_multi, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                     shuffle=False)

else:
    train_set = get_training_set(opt.data_path, opt.training_list, [opt.crop_height, opt.crop_width], opt.left_right,
                                 opt.kitti, opt.kitti2015, opt.eth3d, opt.middlebury, opt.shift)
    test_set = get_test_set(opt.data_path, opt.val_list, [opt.val_height, opt.val_width], opt.left_right, opt.kitti, opt.kitti2015,
                            opt.eth3d, opt.middlebury)

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True, drop_last=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                     shuffle=False)

print('===> Building model')
model = GANet(opt.max_disp)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('===> Total Trainable Parameters: ', pytorch_total_params)

if opt.visualize:
    writer = SummaryWriter("runs/" + str(opt.save_path))
    # writer.add_graph(model)
    # print('\n====> Added Tensorboard Graph')

# print('\n===> Model Details')
# print(model)


criterion = MyLoss2(thresh=3, alpha=2)
if cuda:
    model = torch.nn.DataParallel(model).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        try:
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        except:
            pretrained_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            updated_k = []
            for k, v in pretrained_dict.items():
                if k.startswith('module.cost_agg'):
                    updated_k.append(k)
            for i in updated_k:
                del pretrained_dict[i]
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    #        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))
        exit(0)


def disp_downsample(disp, s_fac, interpolation='bilinear'):
    if s_fac == 0:
        return disp
    disp = disp.unsqueeze(1)
    disp = F.interpolate(disp, size=[int((disp.size()[2] / s_fac)), int((disp.size()[3] / s_fac))], mode=interpolation)
    disp = disp.squeeze(1)
    disp = disp / s_fac
    return disp


def train(epoch):
    t0_epoch = time.time()
    epoch_loss = 0
    epoch_error0 = 0
    epoch_error1 = 0
    epoch_error2 = 0
    # epoch_error3 = 0
    epoch_ER = 0
    valid_iteration = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader):
        # print("====> Debug: in train() for loop")
        input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1],
                                                                                  requires_grad=True), Variable(
            batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()

        target = torch.squeeze(target, 1)
        # mask = target < 192
        mask = torch.logical_and(target >= 0.001, target <= 192)
        mask.detach_()
        valid = target[mask].size()[0]
        t0_iter = time.time()
        if valid > 0:
            optimizer.zero_grad()
            loss = 0

            if opt.model == 'GANet11':
                disp1, disp2 = model(input1, input2)
                disp0 = (disp1 + disp2) / 2.
                if opt.kitti or opt.kitti2015:
                    loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * criterion(
                        disp2[mask], target[mask])
                else:
                    loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * F.smooth_l1_loss(
                        disp2[mask], target[mask], reduction='mean')

            elif opt.model == 'GANet_deep':
                disp0, disp1, disp2 = model(input1, input2)

                if opt.kitti or opt.kitti2015:
                    loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(
                        disp1[mask], target[mask], reduction='mean') + criterion(disp2[mask], target[mask])
                else:
                    loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(
                        disp1[mask], target[mask], reduction='mean') + F.smooth_l1_loss(disp2[mask], target[mask],
                                                                                        reduction='mean')

            elif opt.model == 'GANet_ms':
                # disp0, disp1, disp2 = model(input1, input2)
                ms_disps = model(input1, input2)

                if opt.multi_scale_loss == 1:
                    for i, (d, w) in enumerate(zip(reversed(ms_disps), [1.0, 0.8, 0.6, 0.4])):
                        disp0, disp1, disp2 = d

                        if (i > 0):
                            if (opt.kitti or opt.kitti2015):
                                target_ds = disp_downsample(target, 2 ** i, interpolation='nearest')
                                mask_ds = target_ds < 192
                            else:
                                target_ds = disp_downsample(target, 2 ** i, interpolation='bilinear')
                                mask_ds = target_ds < 192
                        else:
                            target_ds = target
                            mask_ds = target < 192

                        if opt.kitti or opt.kitti2015:
                            loss += w * (0.2 * F.smooth_l1_loss(disp0[mask_ds], target_ds[mask_ds],
                                                                reduction='mean') + 0.6 * F.smooth_l1_loss(
                                disp1[mask_ds], target_ds[mask_ds], reduction='mean') + criterion(disp2[mask_ds],
                                                                                                  target_ds[mask_ds]))
                        else:
                            loss += w * (0.2 * F.smooth_l1_loss(disp0[mask_ds], target_ds[mask_ds],
                                                                reduction='mean') + 0.6 * F.smooth_l1_loss(
                                disp1[mask_ds], target_ds[mask_ds], reduction='mean') + F.smooth_l1_loss(disp2[mask_ds],
                                                                                                         target_ds[
                                                                                                             mask_ds],
                                                                                                         reduction='mean'))
                            # loss = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean')

                    disp0, disp1, disp2 = ms_disps[3]  # to get back the orig disp size for error calc

                else:  # without multi_scale loss
                    disp0, disp1, disp2 = ms_disps[-1]

                    if opt.kitti or opt.kitti2015:
                        loss = 0.2 * F.smooth_l1_loss(disp0[mask],
                                                                                                 target[mask],
                                                                                                 reduction='mean') + 0.6 * criterion(
                            disp1[mask], target[mask]) + criterion(
                            disp2[mask], target[mask])
                    else:
                        loss = 0.2 * F.smooth_l1_loss(disp0[mask],
                                                                                                 target[mask],
                                                                                                 reduction='mean') + 0.6 * F.smooth_l1_loss(
                            disp1[mask], target[mask], reduction='mean') + F.smooth_l1_loss(
                            disp2[mask], target[mask], reduction='mean')


            else:
                raise Exception("No suitable model found ...")

            loss.backward()
            optimizer.step()
            error0 = torch.mean(torch.abs(disp0[mask] - target[mask]))
            error1 = torch.mean(torch.abs(disp1[mask] - target[mask]))
            error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))
            # error3 = torch.mean(torch.abs(disp3[mask] - target[mask]))

            rate = torch.sum(torch.abs(disp2[mask] - target[mask]) > opt.threshold, dtype=torch.float) / torch.sum(mask)


            epoch_loss += loss.item()
            valid_iteration += 1
            epoch_error0 += error0.item()
            epoch_error1 += error1.item()
            epoch_error2 += error2.item()
            # epoch_error3 += error3.item()
            epoch_ER += rate.item()
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}, Error: ({:.4f} {:.4f} {:.4f}), ER: {:.4f}, Time: {:.1f}".format(epoch,
                                                                                                            iteration,
                                                                                                            len(
                                                                                                                training_data_loader),
                                                                                                            loss.item(),
                                                                                                            error0.item(),
                                                                                                            error1.item(),
                                                                                                            error2.item(),
                                                                                                            rate.item(),
                                                                                                            time.time() - t0_iter))
            # print("===> Epoch[{}]({}/{}): Loss: {:.4f}, Error: ({:.4f} )".format(epoch, iteration, len(training_data_loader), loss.item(), error0.item()))
            sys.stdout.flush()
            # if iteration % 10 == 0:
            #     writer.add_scalar('training loss',
            #                 epoch_loss / 1000,
            #                 iteration)

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Avg. Error: ({:.4f} {:.4f} {:.4f}), Avg. ER: {:.4f}, Time: {:.1f}".format(epoch,
                                                                                                               epoch_loss / valid_iteration,
                                                                                                               epoch_error0 / valid_iteration,
                                                                                                               epoch_error1 / valid_iteration,
                                                                                                               epoch_error2 / valid_iteration,
                                                                                                               epoch_ER / valid_iteration,
                                                                                                               time.time() - t0_epoch))
    writer.add_scalar('training loss',
                      epoch_loss / valid_iteration,
                      epoch)
    writer.add_scalar('Avg. Error 1',
                      epoch_error0 / valid_iteration,
                      epoch)
    writer.add_scalar('Avg. Error 2',
                      epoch_error1 / valid_iteration,
                      epoch)
    writer.add_scalar('Avg. Error 3',
                      epoch_error2 / valid_iteration,
                      epoch)
    # writer.add_scalar('Avg. Error 4',
    #                   epoch_error3 / valid_iteration,
    #                   epoch)
    wandb.log({"Training Loss": epoch_loss / valid_iteration, "epoch": epoch})
    wandb.log({"Avg. Error 1": epoch_error0 / valid_iteration, "Avg. Error 2": epoch_error1 / valid_iteration,
               "Avg. Error 3": epoch_error2 / valid_iteration, "Avg. ER" : epoch_ER/valid_iteration, "epoch": epoch})
    img_summary = dict()
    img_summary['disp1'] = disp0
    img_summary['disp2'] = disp1
    img_summary['disp3'] = disp2
    # img_summary['disp4'] = disp3
    img_summary['disp_error'] = disp_error_img(disp2, target)
    img_summary['gt_disp'] = target
    save_images(wandb, 'train', img_summary)


def val(epoch):
    t0 = time.time()
    epoch_error2 = 0
    epoch_ER = 0

    valid_iteration = 0
    model.eval()
    for iteration, batch in enumerate(testing_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=False), Variable(batch[1],
                                                                                   requires_grad=False), Variable(
            batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()
        target = torch.squeeze(target, 1)
        mask = torch.logical_and(target >= 0.001, target <= 192)
        mask.detach_()
        valid = target[mask].size()[0]
        if valid > 0:
            with torch.no_grad():
                disp2 = model(input1, input2)
                error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))
                rate = torch.sum(torch.abs(disp2[mask] - target[mask]) > opt.threshold, dtype=torch.float) / torch.sum(
                    mask)
                valid_iteration += 1
                epoch_error2 += error2.item()
                epoch_ER += rate.item()
                print("===> Test({}/{}): Error: ({:.4f}), ER: {:.4f},  Time: {:.1f}".format(iteration, len(testing_data_loader),
                                                                               error2.item(), rate.item(), time.time() - t0))

    print("===> Test: Avg. Error: ({:.4f}), Avg. ER: {:.4f}".format(epoch_error2 / valid_iteration, epoch_ER/valid_iteration))
    writer.add_scalar('validation_loss',
                      epoch_error2 / valid_iteration,
                      epoch)
    wandb.log({"Validation Loss": epoch_error2 / valid_iteration, "Validation ER": epoch_ER/valid_iteration, "epoch" : epoch})
    img_summary = dict()
    img_summary['disp3'] = disp2
    img_summary['disp_error'] = disp_error_img(disp2, target)
    img_summary['gt_disp'] = target
    save_images(wandb, 'val', img_summary)

    return epoch_error2 / valid_iteration, epoch_ER / valid_iteration


def save_checkpoint(save_path, epoch, state, is_best):
    if is_best:
        # filename = save_path + "_epoch_{} + _best.pth".format(epoch)
        filename = save_path + "_best.pth"
        torch.save(state, filename)
    else:
        filename = save_path + "_latest.pth"
        torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, save_path + '_best.pth')
    print("Checkpoint saved to {}".format(filename))


def adjust_learning_rate(optimizer, epoch, kitti, kitti2015):
    if kitti or kitti2015:
        if epoch <= 550:
            lr = opt.lr
        else:
            lr = opt.lr * 0.1
    else:
        if epoch <= 15:
            lr = opt.lr
        else:
            lr = opt.lr * 0.1
        print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_images(wb_logger, mode_tag, images_dict):
    images_dict = tensor2numpy(images_dict)
    for tag, values in images_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            if len(value.shape) == 3:
                value = value[:, np.newaxis, :, :]
            value = value[:1]
            value = torch.from_numpy(value)

            image_name = '{}/{}'.format(mode_tag, tag)
            if len(values) > 1:
                image_name = image_name + "_" + str(idx)
            if wb_logger is not None:
                wb_logger.log({image_name: [wb_logger.Image(vutils.make_grid(value, padding=0, nrow=1, normalize=True, scale_each=True), caption=image_name)]})


def tensor2numpy(var_dict):
    for key, vars in var_dict.items():
        if isinstance(vars, np.ndarray):
            var_dict[key] = vars
        elif isinstance(vars, torch.Tensor):
            var_dict[key] = vars.data.cpu().numpy()
        else:
            raise NotImplementedError("invalid input type for tensor2numpy")

    return var_dict

def gen_error_colormap():
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols


def disp_error_img(D_est_tensor, D_gt_tensor, abs_thres=3., rel_thres=0.05, dilate_radius=1):
    D_gt_np = D_gt_tensor.detach().cpu().numpy()
    D_est_np = D_est_tensor.detach().cpu().numpy()
    B, H, W = D_gt_np.shape
    # valid mask
    mask = D_gt_np > 0
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = np.minimum(error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres)
    # get colormap
    cols = gen_error_colormap()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]

    return torch.from_numpy(np.ascontiguousarray(error_image.transpose([0, 3, 1, 2])))


if __name__ == '__main__':
    error = 100
    rate = 100
    # print("====> Debug: in main before train for loop")
    if opt.cont == 1 and opt.resume:
        load_epoch = checkpoint['epoch'] + 1
        error = checkpoint['loss']
    else:
        load_epoch = 1

    if opt.wandb == 1:
        wandb.init(project=opt.wbProj, entity="KumailRaza")
        wandb.run.name = opt.wbRunName
        wandb.run.save()
        wandb.watch(model)

    for epoch in range(load_epoch, opt.nEpochs + 1):
        # if opt.kitti or opt.kitti2015:
        if epoch % 10 == 0:
            adjust_learning_rate(optimizer, epoch, opt.kitti, opt.kitti2015)
        # print("====> Debug: in Train for loop,  before train")
        train(epoch)
        # print("====> Debug: in Train for loop,  after train")
        is_best = False
        loss, ER = val(epoch)
        if loss < error and ER < rate:
           error = loss
           rate = ER
           is_best = True
        if opt.kitti or opt.kitti2015:
            if epoch >= 795 or is_best==True:
                save_checkpoint(opt.save_path, epoch, {
                    'epoch': epoch,
                    'loss': loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best)
        else:
            if epoch >= 8 or is_best == True:
                save_checkpoint(opt.save_path, epoch, {
                    'epoch': epoch,
                    'loss': loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best)

    save_checkpoint(opt.save_path, opt.nEpochs, {
        'epoch': opt.nEpochs,
        'loss': loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best)

