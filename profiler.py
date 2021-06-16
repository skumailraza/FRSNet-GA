from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10
#from GCNet.modules.GCNet import L1Loss
from torchprofile import profile_macs
from thop import profile
# from ptflops import get_model_complexity_info
import torchprof
import time


import sys
import shutil
import os
import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from models.GANet_deep import GANet
from dataloader.data import get_test_set
import numpy as np

# Training settings
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
parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")
parser.add_argument('--eth3d', type=int, default=0, help="ETH3D dataset?")

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
#cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

#torch.manual_seed(opt.seed)
#if cuda:
#    torch.cuda.manual_seed(opt.seed)
#print('===> Loading datasets')


print('===> Building model')
model = GANet(opt.max_disp)

if cuda:
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
       
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
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
    #r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data

def test(leftname, rightname, savename):
  #  count=0

    with torchprof.Profile(model, use_cuda=True) as prof:
        t0 = time.time()

        input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)

        
        input1 = Variable(input1, requires_grad = False)
        input2 = Variable(input2, requires_grad = False)

        model.eval()
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
        with torch.no_grad():

            prediction = model(input1, input2)
            t1 = time.time()
            # macs, params = get_model_complexity_info(model, (3, 192, 576), as_strings=True,
            #                                    print_per_layer_stat=True, verbose=True)
            flops, params = profile(model, inputs=[input1,input2])
            # inputs = [input1, input2]
            # macs = profile_macs(model, inputs)

            from thop import clever_format

            macs, params = clever_format([flops, params], "%.3f")

        temp = prediction.cpu()
        temp = temp.detach().numpy()
        if height <= opt.crop_height and width <= opt.crop_width:
            temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
        else:
            temp = temp[0, :, :]
        skimage.io.imsave(savename, (temp * 256).astype('uint16'))
    print(prof.display(show_events=False)) # equivalent to `print(prof)` and `print(prof.display())
    print("\nInference Time: {:.1f} sec".format(t1-t0))

    print("MACS / Params: ", macs, params)




   
if __name__ == "__main__":
    file_path = opt.data_path
    file_list = opt.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()
    
    for index in range(len(filelist)-1):
        current_file = filelist[index]
        if opt.kitti2015:
            leftname = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
        elif opt.kitti:
            leftname = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]
        elif opt.middlebury:
            
            leftname = file_path  + current_file[0: len(current_file) - 1]
            print('leftname: ', leftname)
            try:
                rightname = file_path + current_file[0: len(current_file) - 8] + 'im1.png'
                print('rightname: ',rightname)

                #dispname = opt.data_path + current_file[0: len(current_file) - 8] + 'disp0GT.pfm'
                f = open(rightname)
                f.close()
                #disp, height, width = readPFM(dispname)
            except:
                rightname = file_path + current_file[0: len(current_file) - 10] + 'view5.png'
                #dispname = opt.data_path + current_file[0: len(current_file) - 10] + 'disp1.png'
                #disp, height, width = readPNG(dispname)
                # disp = Image.open(dispname)
                # disp = np.asarray(disp)
        else:
            leftname = opt.data_path + 'frames_finalpass/' + current_file[0: len(current_file) - 1]
            rightname = opt.data_path + 'frames_finalpass/' + current_file[0: len(current_file) - 14] + 'right/' + current_file[len(current_file) - 9:len(current_file) - 1]
        
        savename = opt.save_path + 'file.png'
        test(leftname, rightname, savename)
        break






# import argparse
# import torch
# import torchprof
# from torch.utils.data import DataLoader
# from libs.GANet.modules.GANet import MyLoss2
# from torchprofile import profile_macs

# from math import log10
# import sys
# import shutil
# import os
# import time
# import skimage.io
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from dataloader.data import get_training_set, get_test_set
# from torch.utils.tensorboard import SummaryWriter

# # from models.GANet_ms import GANet


# parser = argparse.ArgumentParser(description='PyTorch GANet Example')
# parser.add_argument('--crop_height', type=int, required=True, help="crop height")
# parser.add_argument('--max_disp', type=int, default=192, help="max disp")
# parser.add_argument('--crop_width', type=int, required=True, help="crop width")
# parser.add_argument('--left_right', type=int, default=0, help="use right view for training. Default=False")
# parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
# parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
# parser.add_argument('--nEpochs', type=int, default=2048, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
# parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
# parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
# parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
# parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
# parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
# parser.add_argument('--eth3d', type=int, default=0, help='eth3d Dataset? Default=False')
# parser.add_argument('--middlebury', type=int, default=0, help='middlebury dataset? Default=False')
# parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
# parser.add_argument('--data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
# parser.add_argument('--training_list', type=str, default='./lists/sceneflow_train.list', help="training list")
# parser.add_argument('--val_list', type=str, default='./lists/sceneflow_test_select.list', help="validation list")
# parser.add_argument('--save_path', type=str, default='./checkpoint/', help="location to save models")
# parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")
# parser.add_argument('--visualize', type=int, default=0, help="Visualize Model")
# parser.add_argument('--multi', type=int, default=0, help="Training on multiple datasets")
# parser.add_argument('--eth3d_data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
# parser.add_argument('--kitti_data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
# parser.add_argument('--kitti2015_data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
# parser.add_argument('--middlebury_data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
# parser.add_argument('--eth3d_training_list', type=str, default='./lists/sceneflow_train.list', help="eth3d training list")
# parser.add_argument('--kitti_training_list', type=str, default='lists/kitti2012_train.list', help="kitti training list")
# parser.add_argument('--kitti2015_training_list', type=str, default='lists/kitti2015_train.list', help="kitti training list")
# parser.add_argument('--middlebury_training_list', type=str, default='lists/middlebury_train.list', help="kitti training list")

# opt = parser.parse_args()
# print(opt)

# if opt.model == 'GANet11':
#     from models.GANet11 import GANet
# elif opt.model == 'GANet_deep':
#     from models.GANet_deep import GANet
# elif opt.model == 'GANet_ms':
#     from models.GANet_ms import GANet
# else:
#     raise Exception("No suitable model found ...")

# model = GANet(opt.max_disp)
# device = 'cuda:0'

# cuda = opt.cuda
# #cuda = True
# if cuda and not torch.cuda.is_available():
#     raise Exception("No GPU found, please run without --cuda")

# torch.manual_seed(opt.seed)
# if cuda:
#     torch.cuda.manual_seed(opt.seed)

# criterion = MyLoss2(thresh=3, alpha=2)
# if cuda:
#     model = torch.nn.DataParallel(model).cuda()
# optimizer=optim.Adam(model.parameters(), lr=opt.lr,betas=(0.9,0.999))


# train_set = get_training_set(opt.data_path, opt.training_list, [opt.crop_height, opt.crop_width], opt.left_right, opt.kitti, opt.kitti2015, opt.eth3d, opt.middlebury, opt.shift)
# training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, drop_last=True)


# model.train()
# with torchprof.Profile(model, use_cuda=True) as prof:
#     for iteration, batch in enumerate(training_data_loader):
#         # print("====> Debug: in train() for loop")
#         input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), Variable(batch[2], requires_grad=False)
#         if cuda:
#             input1 = input1.cuda()
#             input2 = input2.cuda()
#             target = target.cuda()

#         target=torch.squeeze(target,1)
#         mask = target < opt.max_disp
#         mask.detach_()
#         valid = target[mask].size()[0]
#         t0_iter = time.time()
#         if valid > 0:
#             optimizer.zero_grad()
            
#             if opt.model == 'GANet11':
#                 disp1, disp2 = model(input1, input2)
#                 disp0 = (disp1 + disp2)/2.
#                 if opt.kitti or opt.kitti2015:
#                     loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * criterion(disp2[mask], target[mask])
#                 else:
#                     loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
#             elif opt.model == 'GANet_deep':
#                 disp0, disp1, disp2 = model(input1, input2)

#                 if opt.kitti or opt.kitti2015:
#                     loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') +  criterion(disp2[mask], target[mask])
#                 else:
#                     loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') +  F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
#             elif opt.model == 'GANet_ms':
#                 # disp0, disp1, disp2 = model(input1, input2)
#                 macs = profile_macs(model, (input1, input2))


                
#             #     if opt.kitti or opt.kitti2015:
#             #         loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') +  criterion(disp2[mask], target[mask])
#             #     else:
#             #         loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') +  F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
#             #         # loss = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean')
#             # loss.backward()
#             # optimizer.step()
#         break


# print(prof.display(show_events=False)) # equivalent to `print(prof)` and `print(prof.display())`
# # print('MACs / Params: ', macs)
# # trace, event_lists_dict = prof.raw()
# # print(trace[2])
# # print(event_lists_dict[trace[2].path][0])


