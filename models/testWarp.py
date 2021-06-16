import cv2
import torch
import torch.nn.functional as F
import re
from struct import unpack
import numpy as np

def warp(x, disp):
        bs, ch, h, w = x.size()
        bg, hg, wg = torch.meshgrid(torch.arange(0,bs) , torch.arange(0,h), torch.arange(0,w))

        grid_b, grid_h, grid_w = bg.cuda(), hg.cuda(), wg.cuda()
        
        warped_gw = torch.sub(grid_w,disp)
        grid = torch.stack([warped_gw, grid_h.float()], dim=-1)
        grid_normalized = ((grid*2)/torch.Tensor([w,h]).cuda()) - 1
        output = F.grid_sample(x, grid_normalized, mode='bilinear', padding_mode='zeros')
        return output

def readPFM(file): 
    with open(file, "rb") as f:
            # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
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
#        quit()
    return img, height, width

x  = cv2.imread('/data/schuster/BMW_SceneFlow/DATA/Freiburg/FlyingThings3D/frames_finalpass/TRAIN/A/0000/left/0006.png')
y  = torch.from_numpy(np.asarray(cv2.imread('/data/schuster/BMW_SceneFlow/DATA/Freiburg/FlyingThings3D/frames_finalpass/TRAIN/A/0000/right/0006.png')).transpose(2,0,1))

disp, _, _ = readPFM('/data/schuster/BMW_SceneFlow/DATA/Freiburg/FlyingThings3D/disparity/TRAIN/A/0000/left/0006.pfm')
disp = torch.from_numpy(disp.copy())
# print(x.size(), disp.size())
output = warp(y.unsqueeze(dim=0).cuda().float(),disp.unsqueeze(dim=0).cuda().float())


cv2.imwrite('warped.png', output.squeeze().detach().cpu().numpy().transpose(1,2,0))
cv2.imwrite('left.png', x)