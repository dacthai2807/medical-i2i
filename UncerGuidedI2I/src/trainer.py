import torch
from losses import *
from networks import *
import random
random.seed(0)

import torch
import torch.nn.functional as F
import torch.optim

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def train_I2I_CasUNet3headGAN(
    netG_A,
    netD_A,
    train_loader, val_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=50,
    init_lr=1e-4,
    ckpt_path='../ckpt/I2I_CasUNet3headGAN',
):
    netG_A.to(device)
    netG_A.type(dtype)
    ####
    netD_A.to(device)
    netD_A.type(dtype)
    ####
    optimizerG = torch.optim.Adam(list(netG_A.parameters()), lr=init_lr)
    optimizerD = torch.optim.Adam(list(netD_A.parameters()), lr=init_lr)
    ####
    list_num_epochs = [num_epochs, num_epochs, 100 + num_epochs]
    list_lambda1 = [1, 0.5, 0.1]
    list_lambda2 = [0.0001, 0.001, 0.01]

    best_mae = 1000
    n_early_stop = 10

    for num_epochs, lam1, lam2 in zip(list_num_epochs, list_lambda1, list_lambda2):
        n_patience = 0
        for eph in range(num_epochs):
            netG_A.train()
            netD_A.train()
            avg_tot_loss = 0
            print('start epoch {}'.format(eph))
            for i, batch in enumerate(train_loader):
                xA, xB = batch['ct'].to(device).type(dtype), batch['pet'].to(device).type(dtype)
                #calc all the required outputs
                rec_B, rec_alpha_B, rec_beta_B = netG_A(xA)

                #first gen
                netD_A.eval()
                total_loss = lam1*F.l1_loss(rec_B, xB) + lam2*bayeGen_loss(rec_B, rec_alpha_B, rec_beta_B, xB)
                t0 = netD_A(rec_B)
                t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                e5 = 0.001*F.mse_loss(t1, torch.ones(t1.size()).to(device).type(dtype))
                total_loss += e5
                optimizerG.zero_grad()
                total_loss.backward()
                optimizerG.step()

                #then discriminator
                netD_A.train()
                t0 = netD_A(xB)
                pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_real = 1*F.mse_loss(
                    pred_real_A, torch.ones(pred_real_A.size()).to(device).type(dtype)
                )
                t0 = netD_A(rec_B.detach())
                pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_pred = 1*F.mse_loss(
                    pred_fake_A, torch.zeros(pred_fake_A.size()).to(device).type(dtype)
                )
                loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5

                loss_D = loss_D_A
                optimizerD.zero_grad()
                loss_D.backward()
                optimizerD.step()

                avg_tot_loss += total_loss.item()

                if i % 100 == 0:
                    print(
                        'batch: [{}/{}] | cur_d_loss: {} | cur_tot_loss: {}'.format(
                            i, len(train_loader), loss_D.item(), total_loss.item()
                        )
                    )

            avg_tot_loss /= len(train_loader)

            print(
                'epoch: [{}/{}] | avg_tot_loss: {}'.format(
                    eph, num_epochs, avg_tot_loss
                )
            )

            netG_A.eval()
            netD_A.eval()
            avg_mae = 0
            print('validating...')
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    xA, xB = batch['ct'].type(dtype), batch['pet'].type(dtype)

                    rec_B, rec_alpha_B, rec_beta_B = netG_A(xA)

                    pet_img = xB.squeeze(0).squeeze(0).cpu().numpy()
                    pred_img = rec_B.squeeze(0).squeeze(0).cpu().numpy()

                    mae = compute_mae(pred_img, pet_img)

                    avg_mae += mae
                
            avg_mae /= len(val_loader)

            print(
                'epoch: [{}/{}] | avg_mae: {}'.format(
                    eph, num_epochs, avg_mae
                )
            )

            if avg_mae < best_mae: 
                best_mae = avg_mae
                n_patience = 0
                torch.save(netG_A.state_dict(), ckpt_path+'_G_{}_{}_best.pth'.format(lam1, lam2))
                torch.save(netD_A.state_dict(), ckpt_path+'_D_{}_{}_best.pth'.format(lam1, lam2))
            elif n_patience < n_early_stop:
                n_patience += 1
            else:
                break

    return netG_A, netD_A

def train_I2I_Sequence_CasUNet3headGAN(
    list_netG_A,
    list_netD_A,
    train_loader, val_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=50,
    init_lr=1e-4,
    ckpt_path='../ckpt/I2I_Sequence_CasUNet3headGAN',
):
    for nid, m1 in enumerate(list_netG_A):
        m1.to(device)
        m1.type(dtype)
        list_netG_A[nid] = m1
        
    for nid, m2 in enumerate(list_netD_A):
        m2.to(device)
        m2.type(dtype)
        list_netD_A[nid] = m2
    ####
    optimizerG = torch.optim.Adam(list(list_netG_A[-1].parameters()), lr=init_lr)
    optimizerD = torch.optim.Adam(list(list_netD_A[-1].parameters()), lr=init_lr)
    ####
    list_epochs = [num_epochs, num_epochs, 100 + num_epochs]
    list_lambda1 = [1, 0.5, 0.1]
    list_lambda2 = [0.0001, 0.001, 0.01]
    netG_A, netD_A = list_netG_A[-1], list_netD_A[-1]
    ####

    best_mae = 1000
    n_early_stop = 10

    for num_epochs, lam1, lam2 in zip(list_epochs, list_lambda1, list_lambda2):
        n_patience = 0
        for eph in range(num_epochs):
            netG_A.train()
            netD_A.train()
            avg_tot_loss = 0
            print('start epoch {}'.format(eph))
            for i, batch in enumerate(train_loader):
                xA, xB = batch['ct'].to(device).type(dtype), batch['pet'].to(device).type(dtype)
                #calc all the required outputs
                
                for nid, netG in enumerate(list_netG_A):
                    if nid == 0:
                        rec_B, rec_alpha_B, rec_beta_B = netG(xA)
                    else:
                        xch = torch.cat([rec_B, rec_alpha_B, rec_beta_B, xA], dim=1)
                        rec_B, rec_alpha_B, rec_beta_B = netG(xch)

                #first gen
                netD_A.eval()
                total_loss = lam1*F.l1_loss(rec_B, xB) + lam2*bayeGen_loss(rec_B, rec_alpha_B, rec_beta_B, xB)
                t0 = netD_A(rec_B)
                t1 = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                e5 = 0.001*F.mse_loss(t1, torch.ones(t1.size()).to(device).type(dtype))
                total_loss += e5
                optimizerG.zero_grad()
                total_loss.backward()
                optimizerG.step()

                #then discriminator
                netD_A.train()
                t0 = netD_A(xB)
                pred_real_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_real = 1*F.mse_loss(
                    pred_real_A, torch.ones(pred_real_A.size()).to(device).type(dtype)
                )
                t0 = netD_A(rec_B.detach())
                pred_fake_A = F.avg_pool2d(t0, t0.size()[2:]).view(t0.size()[0], -1)
                loss_D_A_pred = 1*F.mse_loss(
                    pred_fake_A, torch.zeros(pred_fake_A.size()).to(device).type(dtype)
                )
                loss_D_A = (loss_D_A_real + loss_D_A_pred)*0.5

                loss_D = loss_D_A
                optimizerD.zero_grad()
                loss_D.backward()
                optimizerD.step()

                avg_tot_loss += total_loss.item()

                if i % 100 == 0:
                    print(
                        'batch: [{}/{}] | cur_d_loss: {} | cur_tot_loss: {}'.format(
                            i, len(train_loader), loss_D.item(), total_loss.item()
                        )
                    )

            avg_tot_loss /= len(train_loader)

            print(
                'epoch: [{}/{}] | avg_tot_loss: {}'.format(
                    eph, num_epochs, avg_tot_loss
                )
            )

            netG_A.eval()
            netD_A.eval()
            avg_mae = 0
            print('validating...')
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    xA, xB = batch['ct'].to(device).type(dtype), batch['pet'].to(device).type(dtype)
                
                    for nid, netG in enumerate(list_netG_A):
                        if nid == 0:
                            rec_B, rec_alpha_B, rec_beta_B = netG(xA)
                        else:
                            xch = torch.cat([rec_B, rec_alpha_B, rec_beta_B, xA], dim=1)
                            rec_B, rec_alpha_B, rec_beta_B = netG(xch)


                    pet_img = xB.squeeze(0).squeeze(0).cpu().numpy()
                    pred_img = rec_B.squeeze(0).squeeze(0).cpu().numpy()

                    mae = compute_mae(pred_img, pet_img)

                    avg_mae += mae
                
            avg_mae /= len(val_loader)

            print(
                'epoch: [{}/{}] | avg_mae: {}'.format(
                    eph, num_epochs, avg_mae
                )
            )

            if avg_mae < best_mae: 
                best_mae = avg_mae
                n_patience = 0
                torch.save(netG_A.state_dict(), ckpt_path+'_G_{}_{}_best.pth'.format(lam1, lam2))
                torch.save(netD_A.state_dict(), ckpt_path+'_D_{}_{}_best.pth'.format(lam1, lam2))
            elif n_patience < n_early_stop:
                n_patience += 1
            else:
                break

    return list_netG_A, list_netD_A