from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
import pdb
from misc import *
import models.derain_mulcmp as net

#from myutils.vgg16 import Vgg16
#from myutils import utils
import pdb
import torch.nn.functional as F
#from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from datasets.ct2pet import CustomAlignedDataset
import logging
from tools.interact import set_logger 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix_class',  help='')
parser.add_argument('--dataroot', required=False,
  default='', help='path to trn dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=512, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=512, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaGAN', type=float, default=0.01, help='lambdaGAN')
parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')

opt = parser.parse_args()
# print(opt)

# get logger
set_logger(opt.exp, 'test_log.txt')

create_exp_dir(opt.exp)
opt.manualSeed = random.randint(1, 10000)
# opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
opt.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

PET_MAX_PIXEL = 10.4 # 32767
CT_MAX_PIXEL = 2047

dataset_config = {
  'image_size': opt.imageSize,
  'max_pixel_input': PET_MAX_PIXEL,
  'max_pixel_corr': CT_MAX_PIXEL,
  'max_pixel_gt': PET_MAX_PIXEL
}

test_config = dataset_config
test_config['dataset_path'] = opt.dataroot
test_config['to_norm'] = True
test_config['to_aug'] = False

test_dataset = CustomAlignedDataset(dataset_config=test_config)

test_loader = DataLoader(test_dataset,
                          batch_size=opt.batchSize,
                          shuffle=False,
                          num_workers=8,
                          drop_last=False)

netG = net.UMRL()

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

netG.eval()
criterionCAE = torch.nn.SmoothL1Loss()

netG.cuda()
criterionCAE.cuda()

print('start testing...')
    
vlloss = 0 

os.makedirs(os.path.join(opt.exp, 'input_pet'), exist_ok=True)
os.makedirs(os.path.join(opt.exp, 'ct'), exist_ok=True)
os.makedirs(os.path.join(opt.exp, 'gt_pet'), exist_ok=True)
os.makedirs(os.path.join(opt.exp, 'pred_pet'), exist_ok=True)
os.makedirs(os.path.join(opt.exp, 'res_pet'), exist_ok=True)
os.makedirs(os.path.join(opt.exp, 'conf_pet'), exist_ok=True)

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img

def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())
    return norm_ip(t, t.min(), t.max())

for id, (input_pet_img, ct_img, gt_pet_img) in enumerate(test_loader):
    test_input = input_pet_img[0].to(opt.device)
    test_target = gt_pet_img[0].to(opt.device)

    with torch.no_grad():
        for idx in range(test_input.size(0)):
            single_img = test_input[idx,:,:,:].unsqueeze(0)
            count = 0
            [t1,t2,max_r,max_c] = single_img.size()
            N = 121
            test_inputv = single_img
            
            for row in range(0, max_r, 50):
                for col in range(0, max_c, 50):
                    test_inputv[:,:,:row,:col] = single_img[:,:,max_r-row:,max_c-col:]
                    test_inputv[:,:,row:,col:] = single_img[:,:,:max_r-row,:max_c-col]
                    test_inputv[:,:,row:,:col] = single_img[:,:,:max_r-row,max_c-col:]
                    test_inputv[:,:,:row,col:] = single_img[:,:,max_r-row:,:max_c-col]
            
                    test_inputv_128 = torch.nn.functional.interpolate(test_inputv,scale_factor=0.25)
                    test_inputv_256 = torch.nn.functional.interpolate(test_inputv,scale_factor=0.5)
        
                    residual_val, x_hat_val, x_hatlv128, x_hatvl256, c128, c256, c512 = netG(test_inputv, test_inputv_256, test_inputv_128)
                    tensor1 = x_hat_val.data.cpu()

                    tensor1 = norm_range(tensor1, None)

                    if count == 0:
                        tensor = (1.0/N)*tensor1
                    else : 
                        tensor[:,:,max_r-row:,max_c-col:] = tensor[:,:,max_r-row:,max_c-col:] + (1.0/N)*tensor1[:,:,:row,:col]
                        tensor[:,:,:max_r-row,:max_c-col] = tensor[:,:,:max_r-row,:max_c-col] + (1.0/N)*tensor1[:,:,row:,col:]
                        tensor[:,:,:max_r-row,max_c-col:] = tensor[:,:,:max_r-row,max_c-col:] + (1.0/N)*tensor1[:,:,row:,:col]
                        tensor[:,:,max_r-row:,:max_c-col] = tensor[:,:,max_r-row:,:max_c-col] + (1.0/N)*tensor1[:,:,:row,col:]
                        #print(count)
                    count += 1

            tensor = torch.squeeze(tensor)
            tensor = norm_range(tensor, None)     
            tensor = tensor.mul(PET_MAX_PIXEL).clamp(0, PET_MAX_PIXEL)  

            out_path = os.path.join(opt.exp, f'input_pet/{input_pet_img[1][idx]}')
            np.savez(out_path, input_pet_img[0][idx].permute(1, 2, 0).detach().cpu().numpy())
            
            out_path = os.path.join(opt.exp, f'ct/{ct_img[1][idx]}')
            np.savez(out_path, ct_img[0][idx].permute(1, 2, 0).detach().cpu().numpy())

            out_path = os.path.join(opt.exp, f'gt_pet/{gt_pet_img[1][idx]}')
            np.savez(out_path, gt_pet_img[0][idx].permute(1, 2, 0).detach().cpu().numpy())

            out_path = os.path.join(opt.exp, f'pred_pet/{gt_pet_img[1][idx]}')
            np.savez(out_path, tensor.permute(1, 2, 0).detach().cpu().numpy())
    