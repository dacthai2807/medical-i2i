import os
import sys
import logging
import numpy as np
import torch
import torch.utils.data as data
from models.ddpm import Model
from datasets import get_dataset,rescale,inverse_rescale,get_PairedPETnCT_dataset,unscale_image
import torchvision.utils as tvu
from functions.denoising import egsde_sample
from guided_diffusion.script_util import create_model,create_dse
from functions.resizer import Resizer
import models.derain_mulcmp as net
from myutils.vgg16 import Vgg16
from myutils import utils
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def truncation(global_step,ratio=0.5):
    part = int(global_step * ratio)
    weight_l = torch.zeros(part).reshape(-1, 1)
    weight_r = torch.ones(global_step - part).reshape(-1, 1)
    weight = torch.cat((weight_l, weight_r), dim=0)
    return weight

class EGSDE(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device


        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def egsde(self,):
        args, config = self.args, self.config
        #load SDE
        if args.diffusionmodel == 'ADM':
            model = create_model(image_size=config.data.image_size,
                                 num_class=config.model.num_class,
                                 num_channels=config.model.num_channels,
                                 num_res_blocks=config.model.num_res_blocks,
                                 learn_sigma=config.model.learn_sigma,
                                 class_cond=config.model.class_cond,
                                 attention_resolutions=config.model.attention_resolutions,
                                 num_heads=config.model.num_heads,
                                 num_head_channels=config.model.num_head_channels,
                                 num_heads_upsample=config.model.num_heads_upsample,
                                 use_scale_shift_norm=config.model.use_scale_shift_norm,
                                 dropout=config.model.dropout,
                                 resblock_updown=config.model.resblock_updown,
                                 use_fp16=config.model.use_fp16,
                                 use_new_attention_order=config.model.use_new_attention_order)
            states = torch.load(args.ckpt)
            model.load_state_dict(states)
            model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.eval()
        elif args.diffusionmodel == 'DDPM':
            model = Model(config)
            states = torch.load(self.args.ckpt)
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states, strict=True)
            model.eval()
        else:
            raise ValueError(f"unsupported diffusion model")

        #load domain-specific feature extractor
        dse = create_dse(image_size=config.data.image_size,
                         num_class=config.dse.num_class,
                         classifier_use_fp16=config.dse.classifier_use_fp16,
                         classifier_width=config.dse.classifier_width,
                         classifier_depth=config.dse.classifier_depth,
                         classifier_attention_resolutions=config.dse.classifier_attention_resolutions,
                         classifier_use_scale_shift_norm=config.dse.classifier_use_scale_shift_norm,
                         classifier_resblock_updown=config.dse.classifier_resblock_updown,
                         classifier_pool=config.dse.classifier_pool,
                         phase=args.phase)
        states = torch.load(args.dsepath)
        dse.load_state_dict(states)
        dse.to(self.device)
        dse = torch.nn.DataParallel(dse)
        dse.eval()

        #load domain-independent feature extractor
        shape = (args.batch_size, 3, config.data.image_size, config.data.image_size)
        shape_d = (
            args.batch_size, 3, int(config.data.image_size / args.down_N), int(config.data.image_size / args.down_N))
        down = Resizer(shape, 1 / args.down_N).to(self.device)
        up = Resizer(shape_d, args.down_N).to(self.device)
        die = (down, up)

        #create dataset
        # dataset = get_dataset(phase=args.phase,image_size= config.data.image_size, data_path = args.testdata_path)
        ct_dir = '../datasets/108/CT'
        pet_dir = '../datasets/108/PET'
        # ct_dir = '../datasets/multimodal_slices/ct'
        # pet_dir = '../datasets/multimodal_slices/pet'
        # train_split_dir = '../datasets/train_split.txt'
        # val_split_dir = '../datasets/val_split.txt'

        train_flist = []
        val_flist = []
        
        # with open(train_split_dir, 'r') as f:
        #     train_flist = f.read().split('\n')

        # train_flist = train_flist[:-1]

        # with open(val_split_dir, 'r') as f:
        #     val_flist = f.read().split('\n')

        # val_flist = val_flist[:-1]

        NUM_TRAINING = 4000
        NUM_VALIDATING = 500

        for i in range(NUM_TRAINING):
            train_flist.append('{}.npy'.format(i))

        for i in range(NUM_VALIDATING):
            val_flist.append('{}.npy'.format(NUM_TRAINING + i))

        dataset = get_PairedPETnCT_dataset(pet_dir, ct_dir, train_flist)
        val_dataset = get_PairedPETnCT_dataset(pet_dir, ct_dir, val_flist)
        
        data_loader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )

        val_data_loader = data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False
        )

        netG = net.UMRL()
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

        LR_G = 1e-5
        BETA_1 = 0.5
        NUM_EPOCH = 100
        LAMBDA_IMG = 1

        lambdaIMG = LAMBDA_IMG
        optimizerG = optim.Adam(netG.parameters(), lr = LR_G, betas = (BETA_1, 0.999), weight_decay=0.00005)
        ganIterations = 0

        def pre_sample(ct_img):
            y = ct_img
            n = y.size(0)
            y0 = y.to(self.device)
            #let x0 be source image
            x0 = y0
            #args.sample_step: the times for repeating EGSDE(usually set 1) (see Appendix A.2)
            for it in range(args.sample_step):
                e = torch.randn_like(y0)
                total_noise_levels = args.t
                a = (1 - self.betas).cumprod(dim=0)
                # the start point M: y ∼ qM|0(y|x0)
                y = y0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                for i in reversed(range(total_noise_levels)):
                    t = (torch.ones(n) * i).to(self.device)
                    #sample perturbed source image from the perturbation kernel: x ∼ qs|0(x|x0)
                    xt = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                    # egsde update (see VP-EGSDE in Appendix A.3)
                    y_ = egsde_sample(y=y, dse=dse,ls=args.ls,die=die,li=args.li,t=t,model=model,
                                        logvar=self.logvar,betas=self.betas,xt=xt,s1=args.s1,s2=args.s2, model_name = args.diffusionmodel)
                    y = y_
                y0 = y
            
            return y

        for epoch in range(NUM_EPOCH):
            for id, (pet_img, ct_img) in enumerate(data_loader):
                y = pre_sample(ct_img)
                
                #refine y to pet_img - ground truth
                input = y.to(self.device)
                target = pet_img.to(self.device)

                width = target.size(2)
                height = target.size(3)

                input_256 = torch.nn.functional.interpolate(input,scale_factor=0.5)
                input_128 = torch.nn.functional.interpolate(input,scale_factor=0.25)
                target_256 = torch.nn.functional.interpolate(target,scale_factor=0.5)
                target_128 = torch.nn.functional.interpolate(target,scale_factor=0.25)

                x_hat1 = netG(input,input_256,input_128)

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
                    
                L_img_ = L_img_ - (4.0*lam_cmp/(width*height))*torch.sum(torch.log(conf_128+sng))- (2.0*lam_cmp/(width*height))*torch.sum(torch.log(conf_256+sng)) - (lam_cmp/(width*height))*torch.sum(torch.log(conf_512+sng))
                #L_img_ = L_img_ + (0.25*lam_cmp/(128*128))*torch.sum(1-conf_128)+ (0.5*lam_cmp/(256*256))*torch.sum(1-conf_256) + (lam_cmp/(512*512))*torch.sum(1-conf_512)

                # L_res = lambdaIMG * L_res_
                L_img = lambdaIMG * L_img_

                if lambdaIMG != 0:
                    L_img.backward(retain_graph=True) # in case of current version of pytorch
                    #L_img.backward(retain_variables=True)
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
                content_loss = 1.8*lambdaIMG* criterionCAE(features_y[1], f_xc_c) + 1.8*lambdaIMG*0.25* criterionCAE(features_y128[1], f_xc_c_128) + 1.8*lambdaIMG*0.50* criterionCAE(features_y256[1], f_xc_c_256)
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
                content_loss1 = 1.8*lambdaIMG* criterionCAE(features_y[0], f_xc_c)+ 1.8*lambdaIMG*0.25* criterionCAE(features_y128[0], f_xc_c_128) + 1.8*lambdaIMG*0.50* criterionCAE(features_y256[0], f_xc_c_256)
                content_loss1.backward(retain_graph=True)

                optimizerG.step()
                ganIterations += 1
                if ganIterations % 10 == 0:
                    logging.info('[%d/%d][%d/%d] L_img: %f'
                            % (epoch, NUM_EPOCH, id, len(dataset), L_img.item()))

            print('start validating...')

            vlloss = 0 
            
            input_images = []
            sample_images = []
            res_images = []
            conf_images = []
            gt_images = []
            pred_images = []

            for id, (pet_img, ct_img) in enumerate(val_data_loader):
                val_inputv = pre_sample(ct_img).to(self.device)
                val_targetv = pet_img.to(self.device)

                with torch.no_grad():
                    val_inputv_128 = torch.nn.functional.interpolate(val_inputv,scale_factor=0.25)
                    val_inputv_256 = torch.nn.functional.interpolate(val_inputv,scale_factor=0.5)
                    val_targetv_128 = torch.nn.functional.interpolate(val_targetv,scale_factor=0.25)
                    val_targetv_256 = torch.nn.functional.interpolate(val_targetv,scale_factor=0.5)
                    
                    #print(val_inputv_128.size())
                    #print(val_inputv_256.size())
                ###  We use a random label here just for intermediate result visuliztion (No need to worry about the label here) ##
                    residual_val, x_hat_val, x_hatlv128, x_hatvl256,c128,c256,c512= netG(val_inputv,val_inputv_256,val_inputv_128)
                    vl_loss = criterionCAE(x_hat_val, val_targetv) + 0.25*criterionCAE(x_hatlv128, val_targetv_128) + 0.5*criterionCAE(x_hatvl256, val_targetv_256)
                    #print(vl_loss)
                    vlloss += vl_loss.data

                n = x_hat_val.size(0)

                for b in range(n):
                    input_images.append(ct_img[b].unsqueeze(0).permute(0, 2, 3, 1).cpu())
                    sample_images.append(val_inputv[b].unsqueeze(0).permute(0, 2, 3, 1).cpu())
                    res_images.append(residual_val[b].unsqueeze(0).permute(0, 2, 3, 1).cpu())
                    conf_images.append(conf_512[b].unsqueeze(0).permute(0, 2, 3, 1).cpu())
                    gt_images.append(val_targetv[b].unsqueeze(0).permute(0, 2, 3, 1).cpu())
                    pred_images.append(x_hat_val[b].unsqueeze(0).permute(0, 2, 3, 1).cpu())

            vlloss /= len(val_data_loader)

            logging.info('Epoch: %d - Val loss: %f\n'%(epoch, vlloss))

            if epoch % 1 == 0:
                input_images = np.concatenate(input_images, axis=0)
                sample_images = np.concatenate(sample_images, axis=0)
                res_images = np.concatenate(res_images, axis=0)
                conf_images = np.concatenate(conf_images, axis=0)
                gt_images = np.concatenate(gt_images, axis=0)
                pred_images = np.concatenate(pred_images, axis=0)

                os.makedirs(os.path.join(self.args.samplepath, str(epoch)), exist_ok=True)
                shape_str = "x".join([str(x) for x in input_images.shape])

                out_path = os.path.join(self.args.samplepath, str(epoch), f"input_{shape_str}.npz")
                np.savez(out_path, input_images)

                out_path = os.path.join(self.args.samplepath, str(epoch), f"sample_{shape_str}.npz")
                np.savez(out_path, sample_images)

                out_path = os.path.join(self.args.samplepath, str(epoch), f"res_{shape_str}.npz")
                np.savez(out_path, res_images)

                out_path = os.path.join(self.args.samplepath, str(epoch), f"conf_{shape_str}.npz")
                np.savez(out_path, conf_images)

                out_path = os.path.join(self.args.samplepath, str(epoch), f"gt_{shape_str}.npz")
                np.savez(out_path, gt_images)

                out_path = os.path.join(self.args.samplepath, str(epoch), f"pred_{shape_str}.npz")
                np.savez(out_path, pred_images)

                torch.save(netG.state_dict(), '%s/%d.pth' % (os.path.join(self.args.samplepath, str(epoch)), epoch))