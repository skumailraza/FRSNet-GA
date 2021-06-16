import torch
import torch.nn as nn
import torch.nn.init as init
from libs.GANet.modules.GANet import DisparityRegression, GetCostVolume
from libs.GANet.modules.GANet import MyNormalize
from libs.GANet.modules.GANet import SGA
from libs.GANet.modules.GANet import LGA, LGA2, LGA3
# from libs.sync_bn.modules.sync_bn import BatchNorm2d, BatchNorm3d
from nets.feature import BasicBlock, BasicConv, Conv2x
from nets.refinement import StereoDRNetRefinement, HourglassRefinement
import apex
import torch.nn.functional as F
# from torch.autograd import Variable
# import numpy as np


# class BasicConv(nn.Module):
#
#     def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
#         super(BasicConv, self).__init__()
# #        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
#         self.relu = relu
#         self.use_bn = bn
#         if is_3d:
#             if deconv:
#                 self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
#             else:
#                 self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
#             self.bn = apex.parallel.SyncBatchNorm(out_channels)
#         else:
#             if deconv:
#                 self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
#             else:
#                 self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
#             self.bn = apex.parallel.SyncBatchNorm(out_channels)
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.use_bn:
#             x = self.bn(x)
#         if self.relu:
#             x = F.relu(x, inplace=True)
#         return x


# class Conv2x(nn.Module):
#
#     def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
#         super(Conv2x, self).__init__()
#         self.concat = concat
#
#         if deconv and is_3d:
#             kernel = (3, 4, 4)
#         elif deconv:
#             kernel = 4
#         else:
#             kernel = 3
#         self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)
#
#         if self.concat:
#             self.conv2 = BasicConv(out_channels*2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
#         else:
#             self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x, rem):
#         # print("==> Before x.size() ,  rem.size() : ", x.shape, rem.shape)
#         x = self.conv1(x)
#         # print("==> After x.size() ,  rem.size() : ", x.shape, rem.shape)
#         assert(x.size() == rem.size())
#         if self.concat:
#             x = torch.cat((x, rem), 1)
#         else:
#             x = x + rem
#         x = self.conv2(x)
#         return x

# class Conv2x(nn.Module):
#
#     def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True,
#                  mdconv=False):
#         super(Conv2x, self).__init__()
#         self.concat = concat
#
#         if deconv and is_3d:
#             kernel = (3, 4, 4)
#         elif deconv:
#             kernel = 4
#         else:
#             kernel = 3
#         self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
#                                stride=2, padding=1)
#
#         if self.concat:
#             if mdconv:
#                 self.conv2 = DeformConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1)
#             else:
#                 self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3,
#                                        stride=1, padding=1)
#         else:
#             self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
#                                    padding=1)
#
#     def forward(self, x, rem):
#         x = self.conv1(x)
#         assert (x.size() == rem.size())
#         if self.concat:
#             x = torch.cat((x, rem), 1)
#         else:
#             x = x + rem
#         x = self.conv2(x)
#         return x


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def forward(self, x, p=0):
        # print('++> in feature')
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x_48 = self.conv4b(x, rem4)

        x_24 = self.deconv4b(x_48, rem3)
        # if (p == -4):
            # return x
        x_12 = self.deconv3b(x_24, rem2)
        # if (p == -3):
            # return x
        x_6 = self.deconv2b(x_12, rem1)
        # if (p == -2):
            # return x
        x_3 = self.deconv1b(x_6, rem0)    
        
        return x_3, x_6, x_12, x_24, x_48

class Guidance(nn.Module):
    def __init__(self):
        super(Guidance, self).__init__()

        self.conv0 = BasicConv(64, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(
            BasicConv(16, 32, kernel_size=5, stride=3, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv2 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv3 = BasicConv(32, 32, kernel_size=3, padding=1)

#        self.conv11 = Conv2x(32, 48)
        self.conv11 = nn.Sequential(BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                                    BasicConv(48, 48, kernel_size=3, padding=1))
        self.conv12 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv13 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv14 = BasicConv(48, 48, kernel_size=3, padding=1)

        self.weight_sg1 = nn.Conv2d(32, 640, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg2 = nn.Conv2d(32, 640, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg3 = nn.Conv2d(32, 640, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_sg11 = nn.Conv2d(48, 960, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg12 = nn.Conv2d(48, 960, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg13 = nn.Conv2d(48, 960, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg14 = nn.Conv2d(48, 960, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_lg1 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1) ,bias=False))
        self.weight_lg2 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1) ,bias=False))

    def forward(self, x):
        x = self.conv0(x)
        rem = x
        x = self.conv1(x)
        sg1 = self.weight_sg1(x)
        x = self.conv2(x)
        sg2 = self.weight_sg2(x)
        x = self.conv3(x)
        sg3 = self.weight_sg3(x)

        x = self.conv11(x)
        sg11 = self.weight_sg11(x)
        x = self.conv12(x)
        sg12 = self.weight_sg12(x)
        x = self.conv13(x)
        sg13 = self.weight_sg13(x)
        x = self.conv14(x)
        sg14 = self.weight_sg14(x)

        lg1 = self.weight_lg1(rem)
        lg2 = self.weight_lg2(rem)
       
        return dict([
            ('sg1', sg1),
            ('sg2', sg2),
            ('sg3', sg3),
            ('sg11', sg11),
            ('sg12', sg12),
            ('sg13', sg13),
            ('sg14', sg14),
            ('lg1', lg1),
            ('lg2', lg2)])

class Disp(nn.Module):

    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)
#        self.conv32x1 = BasicConv(32, 1, kernel_size=3)
        self.conv32x1 = nn.Conv3d(32, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)

    def forward(self, x):
        x = F.interpolate(self.conv32x1(x), [self.maxdisp+1, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)

        return self.disparity(x)

class DispAgg(nn.Module):

    def __init__(self, maxdisp=192):
        super(DispAgg, self).__init__()
        self.maxdisp = maxdisp
        self.LGA3 = LGA3(radius=2)
        self.LGA2 = LGA2(radius=2)
        self.LGA = LGA(radius=2)
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)
        self.dropout = nn.Dropout(p=0.3)
#        self.conv32x1 = BasicConv(32, 1, kernel_size=3)
        self.conv32x1=nn.Conv3d(32, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)

    def lga(self, x, g):
        g = F.normalize(g, p=1, dim=1)
        x = self.LGA2(x, g)
        return x

    def forward(self, x, lg1, lg2):
        x = F.interpolate(self.conv32x1(x), [self.maxdisp+1, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        assert(lg1.size() == lg2.size())
        x = self.lga(x, lg1)
        x = self.softmax(x)
        x = self.lga(x, lg2)
        x = F.normalize(x, p=1, dim=1)
        return self.disparity(x)


class SGABlock(nn.Module):
    def __init__(self, channels=32, refine=False):
        super(SGABlock, self).__init__()
        self.refine = refine
        if self.refine:
            self.bn_relu = nn.Sequential(apex.parallel.SyncBatchNorm(channels),
                                         nn.ReLU(inplace=True))
            self.conv_refine = BasicConv(channels, channels, is_3d=True, kernel_size=3, padding=1, relu=False)
#            self.conv_refine1 = BasicConv(8, 8, is_3d=True, kernel_size=1, padding=1)
        else:
            self.bn = apex.parallel.SyncBatchNorm(channels)
        self.SGA=SGA()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, g):
        rem = x
        k1, k2, k3, k4 = torch.split(g, (x.size()[1]*5, x.size()[1]*5, x.size()[1]*5, x.size()[1]*5), 1)
        # print("===> k1.shape: ", k1.shape)
        k1 = F.normalize(k1.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k2 = F.normalize(k2.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k3 = F.normalize(k3.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k4 = F.normalize(k4.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        x = self.SGA(x, k1, k2, k3, k4)
        if self.refine:
            x = self.bn_relu(x)
            x = self.conv_refine(x)
        else:
            x = self.bn(x)
        assert(x.size() == rem.size())
        x += rem
        return self.relu(x)    
#        return self.bn_relu(x)


class CostAggregation(nn.Module):
    def __init__(self, maxdisp=192):
        super(CostAggregation, self).__init__()
        self.maxdisp = maxdisp
        self.conv_start = BasicConv(64, 32, is_3d=True, kernel_size=3, padding=1, relu=False)

        self.conv1a = BasicConv(32, 48, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, is_3d=True, kernel_size=3, stride=2, padding=1)
#        self.conv3a = BasicConv(64, 96, is_3d=True, kernel_size=3, stride=2, padding=1)

        self.deconv1a = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)
        self.deconv2a = Conv2x(64, 48, deconv=True, is_3d=True)
#        self.deconv3a = Conv2x(96, 64, deconv=True, is_3d=True)

        self.conv1b = Conv2x(32, 48, is_3d=True)
        self.conv2b = Conv2x(48, 64, is_3d=True)
#        self.conv3b = Conv2x(64, 96, is_3d=True)

        self.deconv1b = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)
        self.deconv2b = Conv2x(64, 48, deconv=True, is_3d=True)
#        self.deconv3b = Conv2x(96, 64, deconv=True, is_3d=True)
        self.deconv0b = Conv2x(8, 8, deconv=True, is_3d=True)
        
        self.sga1 = SGABlock(refine=True)
        self.sga2 = SGABlock(refine=True)
        self.sga3 = SGABlock(refine=True)

        self.sga11 = SGABlock(channels=48, refine=True)
        self.sga12 = SGABlock(channels=48, refine=True)
        self.sga13 = SGABlock(channels=48, refine=True)
        self.sga14 = SGABlock(channels=48, refine=True)

        self.disp0 = Disp(self.maxdisp)
        self.disp1 = Disp(self.maxdisp)
        self.disp2 = DispAgg(self.maxdisp)


    def forward(self, x, g):
        # print("==> x_in start: ", x.size())
        x = self.conv_start(x)

        x = self.sga1(x, g['sg1'])
        rem0 = x
       
        if self.training:
            disp0 = self.disp0(x)

        x = self.conv1a(x)


        x = self.sga11(x, g['sg11'])
        # print("==> x_before conv2a(rem1): ", x.size())
        rem1 = x
        x = self.conv2a(x)


        rem2 = x
#        x = self.conv3a(x)
#        rem3 = x

#        x = self.deconv3a(x, rem2)
#        rem2 = x
        # print('++> Came from CostAGG')
        # print("==> Before Deconv  x.size() ,  rem.size() : ", x.shape, rem1.shape)
        
        x = self.deconv2a(x, rem1)
        # print('++> out from deconv2a')
        # print("==> After Deconv  x.size() ,  rem.size() : ", x.shape, rem1.shape)

        x = self.sga12(x, g['sg12'])
        rem1 = x
        x = self.deconv1a(x, rem0)
        # print("==> After Deconv1a  x.size() ,  rem.size() : ", x.shape, rem0.shape)

        x = self.sga2(x, g['sg2'])
        rem0 = x
        if self.training:
            disp1 = self.disp1(x)

        x = self.conv1b(x, rem1)
        x = self.sga13(x, g['sg13'])
        rem1 = x
        x = self.conv2b(x, rem2)
#        rem2 = x
#        x = self.conv3b(x, rem3)

#        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.sga14(x, g['sg14'])
        x = self.deconv1b(x, rem0)
        x = self.sga3(x, g['sg3'])

        disp2 = self.disp2(x, g['lg1'], g['lg2'])
        if self.training:
            return disp0, disp1, disp2
        else:
            return disp2

class GANet(nn.Module):
    def __init__(self, maxdisp=192):
        super(GANet, self).__init__()
        self.maxdisp = maxdisp
        self.conv_start = nn.Sequential(BasicConv(3, 16, kernel_size=3, padding=1),
                                        BasicConv(16, 32, kernel_size=3, padding=1))

        self.conv_x = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_x_6 = BasicConv(48, 32, kernel_size=3, padding=1)
        self.conv_x_12 = BasicConv(64, 32, kernel_size=3, padding=1)
        self.conv_x_24 = BasicConv(96, 32, kernel_size=3, padding=1)
        self.conv_x_48 = BasicConv(128, 32, kernel_size=3, padding=1)
        self.conv_y = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_y_6 = BasicConv(48, 32, kernel_size=3, padding=1)
        self.conv_y_12 = BasicConv(64, 32, kernel_size=3, padding=1)
        self.conv_y_24 = BasicConv(96, 32, kernel_size=3, padding=1)
        self.conv_y_48 = BasicConv(128, 32, kernel_size=3, padding=1)

        self.conv_refine = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), bias=False)
        self.conv_refine_6 = nn.Conv2d(48, 32, (3, 3), (1, 1), (1,1), bias=False)


        self.bn_relu = nn.Sequential(apex.parallel.SyncBatchNorm(32),
                                     nn.ReLU(inplace=True))
        self.feature = Feature()
        self.guidance = Guidance()
        self.guidance_x6 = Guidance()
        self.guidance_x12 = Guidance()
        self.guidance_x24 = Guidance()
        self.guidance_x48 = Guidance()
        self.cost_agg = CostAggregation(self.maxdisp)
        self.cv = GetCostVolume(int(self.maxdisp / 3))
        self.hourglass_refinement = HourglassRefinement()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (apex.parallel.SyncBatchNorm, apex.parallel.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def conv_x_multi(self, x):
        rem = x
        if x.shape[1] == 32:
            x = self.conv_x(x)

        elif x.shape[1] == 48:
            x = self.conv_x_6(x)

        elif x.shape[1] == 64:
            x = self.conv_x_12(x)

        elif x.shape[1] == 96:
            x = self.conv_x_24(x)

        elif x.shape[1] == 128:
            x = self.conv_x_48(x)

        return x

    def conv_y_multi(self, x):
        rem = x
        if x.shape[1] == 32:
            x = self.conv_y(x)

        elif x.shape[1] == 48:
            x = self.conv_y_6(x)

        elif x.shape[1] == 64:
            x = self.conv_y_12(x)

        elif x.shape[1] == 96:
            x = self.conv_y_24(x)

        elif x.shape[1] == 128:
            x = self.conv_y_48(x)

        return x

    def conv_refine_multi(self, x, s):
        # if s == 0:
        #     x1 = self.conv_refine_48(x)
        # elif s == 1:
        #     x1 = self.conv_refine_24(x)
        # elif s == 2:
        #     x1 = self.conv_refine_12(x)
        if s == 3:
            x1 = self.conv_refine_6(x)
        else:
            x1 = self.conv_refine(x)

        x1 = F.interpolate(x1, [x1.size()[2] * 3, x1.size()[3] * 3], mode='bilinear', align_corners=False)
        return x1

    def ga_head(self, g, x, y, s, r):
        rem = r

        x = self.cv(x, y)
        # print("CV: ",x)
        # exit(0)
        x1 = self.conv_refine_multi(rem, s)

        x1 = self.bn_relu(x1)

        g = F.interpolate(g, [x1.size()[2], x1.size()[3]], mode='bilinear', align_corners=False)

        g = torch.cat((g, x1), 1)
        # print('Guidance subnet size: ', g.size())
        # exit(0)
        if s == 0:
            g = self.guidance_x48(g)
            # g = self.guidance(g)
            # print('s == 0')
        elif s == 1:
            # print('s == 1')
            # g = self.guidance(g)
            g = self.guidance_x24(g)
        elif s == 2:
            # print('s == 2')
            # g = self.guidance(g)
            g = self.guidance_x12(g)
        elif s == 3:
            # print('s == 3')
            # g = self.guidance(g)
            g = self.guidance_x6(g)
        else:
            # print('s == 4')
            g = self.guidance(g)


        if self.training:
            disp0, disp1, disp2 = self.cost_agg(x, g)
            return disp0, disp1, disp2

        else:
            return self.cost_agg(x, g)

    def disp_downsample(self, disp, s_fac):
        disp = disp.unsqueeze(1)
        disp = F.interpolate(disp, size=[int((disp.size()[2] / s_fac)), int((disp.size()[3] / s_fac))], mode='bilinear',
                             align_corners=False)
        disp = disp.squeeze(1)
        disp = disp / s_fac
        return disp

    
    def warp(self, x, disp):
        bs, ch, h, w = x.size()
        bg, hg, wg = torch.meshgrid(torch.arange(0,bs) , torch.arange(0,h), torch.arange(0,w))

        grid_b, grid_h, grid_w = bg.cuda(), hg.cuda(), wg.cuda()
        warped_gw = torch.sub(grid_w, disp)
        grid = torch.stack([warped_gw, grid_h.float()], dim=-1)
        grid_normalized = ((grid*2)/torch.Tensor([w,h]).cuda()) - 1
        output = F.grid_sample(x, grid_normalized, mode='bilinear', padding_mode='zeros')
        return output
    
    def forward(self, x, y):
        left_img = x
        right_img = y
        g = self.conv_start(x)
        x3, x6, x12, x24, x48 = self.feature(x)
        rem3 = x3
        rem6 = x6

        x3 = self.conv_x_multi(x3)
        x6 = self.conv_x_multi(x6)
        x12 = self.conv_x_multi(x12)
        x24 = self.conv_x_multi(x24)
        x48 = self.conv_x_multi(x48)

        rem12 = x12
        rem24 = x24
        rem48 = x48

        y3, y6, y12, y24, y48 = self.feature(y)
        y3 = self.conv_y_multi(y3)
        y6 = self.conv_y_multi(y6)
        y12 = self.conv_y_multi(y12)
        y24 = self.conv_y_multi(y24)
        y48 = self.conv_y_multi(y48)

        ms_disps = []

        ''' Resisizing multiscale disparities to add the residual disparity'''
        for i, (x, y, r) in enumerate(zip([x48, x24, x12, x6, x3], [y48, y24, y12, y6, y3], [rem48, rem24, rem12, rem6, rem3])):

            if i == 0:
                if self.training:
                    disp0, disp1, disp3 = self.ga_head(g, x, y, i, r)

                else:
                    disp3 = self.ga_head(g, x, y, i, r)

            else:
                if self.training:
                    disp_for_warping = self.disp_downsample(disp3, 1.5)
                    y_warped = self.warp(y, disp_for_warping)
                    disp0_res, disp1_res, disp2_res = self.ga_head(g, x, y_warped, i, r)
                    disp0 = self.disp_downsample(disp0, 0.5) + disp0_res
                    disp1 = self.disp_downsample(disp1, 0.5) + disp1_res
                    disp3 = self.disp_downsample(disp3, 0.5) + disp2_res

                    if i == 4:
                        disp3 = self.hourglass_refinement(disp3, left_img, right_img)

                    ms_disps.append([disp0, disp1, disp3])

                else:
                    disp_for_warping = self.disp_downsample(disp3, 1.5)
                    y_warped = self.warp(y, disp_for_warping)

                    disp2_res = self.ga_head(g, x, y_warped, i, r)
                    disp3 = self.disp_downsample(disp3, 0.5) + disp2_res


                    if i == 4:
                        disp3 = self.hourglass_refinement(disp3, left_img, right_img)

        if self.training:
            return ms_disps
        else:
            return disp3