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
parser.add_argument('--valDataroot', required=False,
  default='', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=120, help='input batch size')
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
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')

opt = parser.parse_args()
# print(opt)

# get logger
set_logger(opt.exp, 'train_log.txt')

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

train_config = dataset_config
train_config['dataset_path'] = opt.dataroot
train_config['to_norm'] = True
train_config['to_aug'] = True

train_dataset = CustomAlignedDataset(dataset_config=train_config)

train_loader = DataLoader(train_dataset,
                          batch_size=opt.batchSize,
                          shuffle=True,
                          num_workers=8,
                          drop_last=True)

val_config = dataset_config
val_config['dataset_path'] = opt.valDataroot
val_config['to_norm'] = True
val_config['to_aug'] = False

val_dataset = CustomAlignedDataset(dataset_config=val_config)

val_loader = DataLoader(val_dataset,
                          batch_size=opt.valBatchSize,
                          shuffle=False,
                          num_workers=8,
                          drop_last=False)

netG = net.UMRL()

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

netG.train()
criterionCAE = torch.nn.SmoothL1Loss()

netG.cuda()
criterionCAE.cuda()

# Initialize VGG-16
# vgg = Vgg16()
#utils.init_vgg16('./models/')

# vgg.load_state_dict(torch.load(os.path.join('./models/', "vgg16.weight")))
vgg = models.vgg16(pretrained=True)
vgg.cuda()

lambdaIMG = opt.lambdaIMG
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.00005)
ganIterations = 0

print('start training...')

for epoch in range(opt.niter):
    if epoch > opt.annealStart:
        adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)
    
    for id, (input_pet_img, ct_img, gt_pet_img) in enumerate(train_loader):
        input_cyc = input_pet_img[0].to(opt.device)
        target_cyc = gt_pet_img[0].to(opt.device)

        width = target_cyc.size(2)
        height = target_cyc.size(3)

        row = 50*random.randint(0, width//50)
        col = 50*random.randint(0, height//50)

        input = input_cyc
        target = target_cyc

        input[:,:,:row,:col] = input_cyc[:,:,width-row:,height-col:]
        input[:,:,row:,col:] = input_cyc[:,:,:width-row,:height-col]
        input[:,:,row:,:col] = input_cyc[:,:,:width-row,height-col:]
        input[:,:,:row,col:] = input_cyc[:,:,width-row:,:height-col]

        target[:,:,:row,:col] = target_cyc[:,:,width-row:,height-col:]
        target[:,:,row:,col:] = target_cyc[:,:,:width-row,:height-col]
        target[:,:,row:,:col] = target_cyc[:,:,:width-row,height-col:]
        target[:,:,:row,col:] = target_cyc[:,:,width-row:,:height-col]
        
        input_256 = torch.nn.functional.interpolate(input,scale_factor=0.5)
        input_128 = torch.nn.functional.interpolate(input,scale_factor=0.25)
        target_256 = torch.nn.functional.interpolate(target,scale_factor=0.5)
        target_128 = torch.nn.functional.interpolate(target,scale_factor=0.25)

        x_hat1 = netG(input, input_256, input_128)

        residual, x_hat, x_hat128, x_hat256, conf_128, conf_256, conf_512 = x_hat1

        sng = 0.00000001

        netG.zero_grad() # start to update G
        
        lam_cmp = 0.1
        xeff = conf_512*x_hat+(1-conf_512)*target
        xeff_128 = conf_128*x_hat128+(1-conf_128)*target_128
        xeff_256 = conf_256*x_hat256+(1-conf_256)*target_256
        L_img_ = criterionCAE(xeff, target) + 0.25*criterionCAE(xeff_128, target_128) + 0.5*criterionCAE(xeff_256, target_256)

        tmp = torch.FloatTensor(1)
        tmp = Variable(tmp, False)
        
        #print(L_img_.data)
        with torch.no_grad():
            tmp = -(4.0/(width*height))*torch.sum(torch.log(conf_128+sng)) - (2.0/(width*height))*torch.sum(torch.log(conf_256+sng)) - (1.0/(width*height))*torch.sum(torch.log(conf_512+sng))
            tmp = tmp.cpu()

            if tmp.item() < 0.25:
                lam_cmp = 0.09 * lam_cmp * (np.exp(5.4 * tmp.item()) - 0.98) #0.09*lam_cmp/(np.exp(1.0*tmp.item()))
            
        L_img_ = L_img_ - (4.0*lam_cmp/(width*height))*torch.sum(torch.log(conf_128+sng)) - (2.0*lam_cmp/(width*height))*torch.sum(torch.log(conf_256+sng)) - (lam_cmp/(width*height))*torch.sum(torch.log(conf_512+sng))
        #L_img_ = L_img_ + (0.25*lam_cmp/(128*128))*torch.sum(1-conf_128) + (0.5*lam_cmp/(256*256))*torch.sum(1-conf_256) + (lam_cmp/(512*512))*torch.sum(1-conf_512)

        # L_res = lambdaIMG * L_res_
        L_img = lambdaIMG * L_img_

        if lambdaIMG != 0:
            L_img.backward(retain_graph=True) # in case of current version of pytorch
            # L_img.backward(retain_variables=True)
            # L_res.backward(retain_variables=True)

        # Perceptual Loss 1
        features_content = vgg(target)
        f_xc_c = Variable(features_content[1].data, requires_grad=False)
        features_content_128 = vgg(target_128)
        f_xc_c_128 = Variable(features_content_128[1].data, requires_grad=False)
        features_content_256 = vgg(target_256)
        f_xc_c_256 = Variable(features_content_256[1].data, requires_grad=False)

        features_y = vgg(x_hat)
        features_y128 = vgg(x_hat128)
        features_y256 = vgg(x_hat256)
        content_loss = 1.8*lambdaIMG*criterionCAE(features_y[1], f_xc_c) + 1.8*lambdaIMG*0.25*criterionCAE(features_y128[1], f_xc_c_128) + 1.8*lambdaIMG*0.50*criterionCAE(features_y256[1], f_xc_c_256)
        content_loss.backward(retain_graph=True)

        # Perceptual Loss 2
        features_content = vgg(target)
        f_xc_c = Variable(features_content[0].data, requires_grad=False)
        features_content_128 = vgg(target_128)
        f_xc_c_128 = Variable(features_content_128[0].data, requires_grad=False)
        features_content_256 = vgg(target_256)
        f_xc_c_256 = Variable(features_content_256[0].data, requires_grad=False)

        features_y = vgg(x_hat)
        features_y128 = vgg(x_hat128)
        features_y256 = vgg(x_hat256)
        content_loss1 = 1.8*lambdaIMG*criterionCAE(features_y[0], f_xc_c) + 1.8*lambdaIMG*0.25*criterionCAE(features_y128[0], f_xc_c_128) + 1.8*lambdaIMG*0.50*criterionCAE(features_y256[0], f_xc_c_256)
        content_loss1.backward(retain_graph=True)

        optimizerG.step()
        ganIterations += 1
        if ganIterations % 100 == 0:
            logging.info('[%d/%d][%d/%d] L_img: %f'
                    % (epoch, opt.niter, id, len(train_loader), L_img.item()))

    print('start validating...')

    vlloss = 0 
  
    os.makedirs(os.path.join(opt.exp, 'input_pet'), exist_ok=True)
    os.makedirs(os.path.join(opt.exp, 'ct'), exist_ok=True)
    os.makedirs(os.path.join(opt.exp, 'gt_pet'), exist_ok=True)
    os.makedirs(os.path.join(opt.exp, 'pred_pet'), exist_ok=True)
    os.makedirs(os.path.join(opt.exp, 'res_pet'), exist_ok=True)
    os.makedirs(os.path.join(opt.exp, 'conf_pet'), exist_ok=True)

    for id, (input_pet_img, ct_img, gt_pet_img) in enumerate(val_loader):
        val_inputv = input_pet_img[0].to(opt.device)
        val_targetv = gt_pet_img[0].to(opt.device)

        with torch.no_grad():
            val_inputv_128 = torch.nn.functional.interpolate(val_inputv,scale_factor=0.25)
            val_inputv_256 = torch.nn.functional.interpolate(val_inputv,scale_factor=0.5)
            val_targetv_128 = torch.nn.functional.interpolate(val_targetv,scale_factor=0.25)
            val_targetv_256 = torch.nn.functional.interpolate(val_targetv,scale_factor=0.5)
            
            #print(val_inputv_128.size())
            #print(val_inputv_256.size())
        ###  We use a random label here just for intermediate result visuliztion (No need to worry about the label here) ##
            residual_val, x_hat_val, x_hatlv128, x_hatvl256, c128, c256, c512 = netG(val_inputv,val_inputv_256,val_inputv_128)
            vl_loss = criterionCAE(x_hat_val, val_targetv) + 0.25*criterionCAE(x_hatlv128, val_targetv_128) + 0.5*criterionCAE(x_hatvl256, val_targetv_256)
            #print(vl_loss)
            vlloss += vl_loss.data

        n = x_hat_val.size(0)

        for b in range(n):
            out_path = os.path.join(opt.exp, f'input_pet/{input_pet_img[1][b]}')
            np.savez(out_path, input_pet_img[0][b].permute(1, 2, 0).detach().cpu().numpy())
            
            out_path = os.path.join(opt.exp, f'ct/{ct_img[1][b]}')
            np.savez(out_path, ct_img[0][b].permute(1, 2, 0).detach().cpu().numpy())

            out_path = os.path.join(opt.exp, f'gt_pet/{gt_pet_img[1][b]}')
            np.savez(out_path, gt_pet_img[0][b].permute(1, 2, 0).detach().cpu().numpy())

            out_path = os.path.join(opt.exp, f'pred_pet/{gt_pet_img[1][b]}')
            np.savez(out_path, x_hat_val[b].permute(1, 2, 0).detach().cpu().numpy())

            out_path = os.path.join(opt.exp, f'res_pet/{gt_pet_img[1][b]}')
            np.savez(out_path, residual_val[b].permute(1, 2, 0).detach().cpu().numpy())

            out_path = os.path.join(opt.exp, f'conf_pet/{gt_pet_img[1][b]}')
            np.savez(out_path, c512[b].permute(1, 2, 0).detach().cpu().numpy())

    vlloss /= len(val_loader)

    logging.info('Epoch: %d - Val loss: %f'%(epoch, vlloss))

    if epoch % 1 == 0:
        torch.save(netG.state_dict(), '%s/%d.pth' % (opt.exp, epoch))