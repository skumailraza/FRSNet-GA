from __future__ import print_function
import argparse
# import skimage
# import skimage.io
# import skimage.transform
from PIL import Image
from math import log10
import math
import matplotlib.pyplot as plt
import sys
import csv
import cv2
import shutil
import os
import re
from struct import unpack
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# from dataloader.data import get_test_set
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--data_path', type=str, required=True, help="data root")
parser.add_argument('--test_list', type=str, required=True, help="training list")
parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")
parser.add_argument('--threshold', type=float, default=3.0, help="threshold of error rates")
parser.add_argument('--multi_gpu', type=int, default=0, help="multi_gpu choice")
parser.add_argument('--middlebury', type=int, default=0, help="middlebury dataset?")
parser.add_argument('--eth3d', type=int, default=0, help="ETH3D dataset?")
parser.add_argument('--benchmark', type=int, default=0, help="Kitti benchmark submission?")
parser.add_argument('--model', type=str, default='GANet_ms', help="model to use")

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

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# print('===> Loading datasets')
# test_set = get_test_set(opt.data_path, opt.test_list, [opt.crop_height, opt.crop_width], false, opt.kitti, opt.kitti2015)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = GANet(opt.max_disp)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def readPFM(file):
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            print(file, ': ** NOT a valid .pfm file ***')
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)
    return img, height, width


def readPNG(file):
    png_disp = Image.open(file)
    png_disp.load()
    png_disp = np.asarray(png_disp)

    # summarize some details about the image
    width = png_disp.shape[0]
    height = png_disp.shape[-1]

    return png_disp, height, width


def test_transform(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)
    print(np.shape(temp_data))
    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        print("in smaller crop width")
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3, crop_height, crop_width], 'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w


def disp_transform(temp_data, crop_height, crop_width):
    h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([crop_height, crop_width], 'float32')
        temp_data[crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([crop_height, crop_width], 'float32')
    left[:, :] = temp_data[:, :]
    return left


def load_data(leftname, rightname, c_height, c_width):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    s_fac = None
    dim = None

    if height > c_height or width > c_width:
        s_fac = (c_height / height, c_width / width)
        dim = (width, height)
        print('===> using resizing')
        new_size = (c_width, c_height)
        left = left.resize(new_size, resample=Image.BICUBIC)
        right = right.resize(new_size, resample=Image.BICUBIC)
        height = c_height
        width = c_width

    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]
    # r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return s_fac, dim, temp_data


def test(leftname, rightname, savename, disp_max):
    s_fac , dim, img_data = load_data(leftname, rightname, opt.crop_height, opt.crop_width)
    input1, input2, height, width = test_transform(img_data, opt.crop_height, opt.crop_width)
    input1 = Variable(input1, requires_grad=False)
    input2 = Variable(input2, requires_grad=False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        prediction = model(input1, input2)

    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
        # temp = temp[0, :, :]
    else:
        temp = temp[0, :, :]

    if s_fac is not None:
        temp = Image.fromarray(temp).resize(dim, resample=Image.LANCZOS)
        print("===> Scaling factor: ", s_fac[1])
        temp = np.asarray(temp) * 1/(s_fac[1])

    if opt.benchmark == 1:
        skimage.io.imsave(savename, (temp * 256).astype('uint16'))
    else:
        dmap_norm = (temp / disp_max) * 256
        # dmap_norm = cv2.normalize(temp.astype('uint8'), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        im_color = cv2.applyColorMap(dmap_norm.astype('uint8'), cv2.COLORMAP_JET)
        cv2.imwrite(savename, im_color)

    return temp


def apply_custom_colormap(image_gray, cmap=plt.get_cmap('jet')):
    assert image_gray.dtype == np.uint8, 'must be np.uint8 image'
    if image_gray.ndim == 3: image_gray = image_gray.squeeze(-1)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:, 0:3]  # color range RGBA => RGB
    color_range = (color_range * 255.0).astype(np.uint8)  # [0,1] => [0,255]
    color_range = np.squeeze(np.dstack([color_range[:, 2], color_range[:, 1], color_range[:, 0]]), 0)  # RGB => BGR

    # Apply colormap for each channel individually
    image_gray = cv2.normalize(image_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    channels = [cv2.LUT(image_gray, color_range[:, i]) for i in range(3)]
    return np.dstack(channels)


if __name__ == "__main__":
    file_path = opt.data_path
    file_list = opt.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()
    avg_error = 0
    avg_rate = 0
    skipped = 0
    for index in range(len(filelist)):
        current_file = filelist[index]
        if opt.kitti2015:
            leftname = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
            if opt.benchmark==1:
                dispname = "/ds-av/public_datasets/kitti2015/raw/training/" + 'disp_occ_0/' + current_file[0: len(current_file) - 1]
            else:
                dispname = file_path + 'disp_occ_0/' + current_file[0: len(current_file) - 1]
            savename = opt.save_path + current_file[0: len(current_file) - 1]
            disp = Image.open(dispname)
            disp = np.asarray(disp) / 256.0
        elif opt.kitti:
            leftname = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]
            dispname = file_path + 'disp_occ/' + current_file[0: len(current_file) - 1]
            savename = opt.save_path + current_file[0: len(current_file) - 1]
            disp = Image.open(dispname)
            disp = np.asarray(disp) / 256.0
        elif opt.middlebury:
            leftname = opt.data_path + current_file[0: len(current_file) - 1]
            try:
                rightname = opt.data_path + current_file[0: len(current_file) - 8] + 'im1.png'
                dispname = opt.data_path + current_file[0: len(current_file) - 8] + 'disp0GT.pfm'
                f = open(rightname)
                f.close()
                disp, height, width = readPFM(dispname)
            except:
                rightname = opt.data_path + current_file[0: len(current_file) - 10] + 'view5.png'
                dispname = opt.data_path + current_file[0: len(current_file) - 10] + 'disp1.png'
                disp, height, width = readPNG(dispname)
                # disp = Image.open(dispname)
                # disp = np.asarray(disp)

            savename = opt.save_path + str(index) + '.png'

            # try:
            #     with open(opt.data_path + current_file[0: len(current_file) - 8] + 'calib.txt') as f:
            #         d = {}
            #         for line in f:
            #             (key, val) = line.split('=')
            #             d[key] = val
            #     # print(d['cam0'].split(" ")[0][1:])
            #     disp = float(d['baseline']) * float(d['cam0'].split(" ")[0][1:]) / (disp + float(d['doffs']))
            #     # print(disp)
            # except:
            #     print("\nRunning without calib.txt...")

        elif opt.eth3d:
            leftname = opt.data_path + current_file[0: len(current_file) - 1]
            rightname = opt.data_path + current_file[0: len(current_file) - 8] + 'im1.png'
            dispname = opt.data_path + current_file[0: len(current_file) - 8] + 'disp0GT.pfm'
            savename = opt.save_path + str(index) + '.png'
            # print("Reading ETH3D images...", dispname)
            disp, height, width = readPFM(dispname)
            # print("Calculating Disparity...", dispname)
        else:
            leftname = opt.data_path + 'frames_finalpass/' + current_file[0: len(current_file) - 1]
            rightname = opt.data_path + 'frames_finalpass/' + current_file[
                                                              0: len(current_file) - 14] + 'right/' + current_file[len(
                current_file) - 9:len(current_file) - 1]
            dispname = opt.data_path + 'disparity/' + current_file[0: len(current_file) - 4] + 'pfm'
            savename = opt.save_path + str(index) + '.png'
            disp, height, width = readPFM(dispname)

        disp_max = disp.max()
        prediction = test(leftname, rightname, savename, disp_max)
        if opt.benchmark == 1:
            print("===> Frame {}: ".format(index) + current_file[0:len(
                current_file) - 1] )
            continue
        mask = np.logical_and(disp >= 0.001, disp <= 192)

        error = np.mean(np.abs(prediction[mask] - disp[mask]))
        rate = np.sum(np.abs(prediction[mask] - disp[mask]) > opt.threshold) / np.sum(mask)
        avg_error += error
        avg_rate += rate
        print("===> Frame {}: ".format(index) + current_file[0:len(
            current_file) - 1] + " ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))

        dmap_norm = cv2.normalize(disp.astype('uint8'), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        im_color = cv2.applyColorMap(dmap_norm, cv2.COLORMAP_JET)
        cv2.imwrite(savename[0: len(savename) - 3] + 'disp.png', im_color)
        # if disp.shape == prediction.shape:
        #     # mask = np.logical_and(disp >= 0.001, disp <= opt.max_disp)
        #     mask = np.logical_and(disp >= 0.001, disp <= 192)
        #     error = np.mean(np.abs(prediction[mask] - disp[mask]))
        #     # skimage.io.imsave(savename[0: len(savename)- 3] + 'disp.png', (disp * 256).astype('uint16'))
        #     dmap_norm = cv2.normalize(disp.astype('uint8'), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #     im_color = cv2.applyColorMap(dmap_norm, cv2.COLORMAP_JET)
        #     cv2.imwrite(savename[0: len(savename) - 3] + 'disp.png', im_color)



        # else:
        #     if opt.benchmark == 1:
        #         continue
        #     # new_size= (opt.crop_width, opt.crop_height)
        #     # disp = np.asarray(Image.fromarray(disp).resize(new_size), dtype=float)
        #     disp = disp_transform(disp, opt.crop_height, opt.crop_width)
        #     print("Scale Factor: ", opt.crop_width / width)
        #     disp = np.multiply(disp, opt.crop_width / width)
        #     # skimage.io.imsave(savename[0: len(savename)- 3] + 'disp.png', (disp * 256).astype('uint16'))
        #     dmap_norm = cv2.normalize(disp.astype('uint8'), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #     im_color = cv2.applyColorMap(dmap_norm, cv2.COLORMAP_JET)
        #     cv2.imwrite(savename[0: len(savename) - 3] + 'disp.png', im_color)
        #
        #     # print('Disp.shape: ', disp.shape)
        #
        #     # mask = np.logical_and(disp >= 0.001, disp <= opt.max_disp)
        #     mask = np.logical_and(disp >= 0.001, disp <= 192)
        #     error = np.mean(np.abs(prediction[mask] - disp[mask]))
        #
        # print('Disp.shape: ', disp.shape)
        # print('Prediction.shape: ', prediction.shape)
        # rate = np.sum(np.abs(prediction[mask] - disp[mask]) > opt.threshold) / np.sum(mask)
        # if math.isnan(rate):
        #     print('++> Prediction Mask False due to low max_disp value... Skipping frame.')
        #     skipped += 1
        #     continue
        #
        # # print("Rate: ", rate)
        # # exit(0)
        # avg_error += error
        # avg_rate += rate
        # print("===> Frame {}: ".format(index) + current_file[0:len(
        #     current_file) - 1] + " ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    avg_error = avg_error / len(filelist)
    avg_rate = avg_rate / len(filelist)
    print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Error Rate: {:.4f}, Skipped: {:}".format(len(filelist), avg_error, avg_rate, skipped))

