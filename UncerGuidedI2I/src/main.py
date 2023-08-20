from ds import *
from trainer import *
from losses import *
from networks import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

ct_dir = '../../datasets/108/CT'
# ct_dir = '../../datasets/multimodal_slices/ct'
pet_dir = '../../datasets/108/PET'
# pet_dir = '../../datasets/multimodal_slices/pet'

# train_split_dir = '../../datasets/train_split.txt'
# val_split_dir = '../../datasets/val_split.txt'

train_flist = []
val_flist = []

NUM_TRAINING = 4000
NUM_VALIDATING = 500

for i in range(NUM_TRAINING):
    train_flist.append('{}.npy'.format(i))

for i in range(NUM_VALIDATING):
    val_flist.append('{}.npy'.format(NUM_TRAINING + i))

# with open(train_split_dir, 'r') as f:
#     train_flist = f.read().split('\n')

# with open(val_split_dir, 'r') as f:
#     val_flist = f.read().split('\n')

# train_flist = train_flist[:-1]
# val_flist = val_flist[:-1]

train_loader = PETnCTDataset(pet_dir, ct_dir, train_flist, train_flist)
val_loader = PETnCTDataset(pet_dir, ct_dir, val_flist, val_flist)

# init net and train
# netG_A = CasUNet_3head(1,1)
# netD_A = NLayerDiscriminator(1, n_layers=4)
# netG_A, netD_A = train_I2I_CasUNet3headGAN(
#     netG_A, netD_A,
#     train_loader, val_loader,
#     dtype=torch.cuda.FloatTensor,
#     device='cuda',
#     num_epochs=30,
#     init_lr=1e-4,
#     ckpt_path='../ckpt/I2I_CasUNet3headGAN',
# ) 

# init net and train
# netG_A1 = CasUNet_3head(1,1)
# netG_A1.load_state_dict(torch.load('../ckpt/I2I_CasUNet3headGAN_G_1_0.0001_best.pth'))
# netG_A2 = UNet_3head(4,1)

# netD_A = NLayerDiscriminator(1, n_layers=4)
# list_netG_A, list_netD_A = train_I2I_Sequence_CasUNet3headGAN(
#     [netG_A1, netG_A2], [netD_A],
#     train_loader, val_loader,
#     dtype=torch.cuda.FloatTensor,
#     device='cuda',
#     num_epochs=30,
#     init_lr=1e-4,
#     ckpt_path='../ckpt/I2I_Sequence_CasUNet3headGAN_Block2',
# )

# init net and train
netG_A1 = CasUNet_3head(1,1)
netG_A1.load_state_dict(torch.load('../ckpt/I2I_CasUNet3headGAN_G_1_0.0001_best.pth'))
netG_A2 = UNet_3head(4,1)
netG_A2.load_state_dict(torch.load('../ckpt/I2I_Sequence_CasUNet3headGAN_Block2_G_0.5_0.001_best.pth'))
netG_A3 = UNet_3head(4,1)

netD_A = NLayerDiscriminator(1, n_layers=4)
list_netG_A, list_netD_A = train_I2I_Sequence_CasUNet3headGAN(
    [netG_A1, netG_A2, netG_A3], [netD_A],
    train_loader, val_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=30,
    init_lr=1e-4,
    ckpt_path='../ckpt/I2I_Sequence_CasUNet3headGAN_Block3',
)