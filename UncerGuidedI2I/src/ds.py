import torch.utils.data as data
import os.path
import numpy as np
import random
import torch
random.seed(0)

class PETnCTDataset(data.Dataset):
    '''
    can act as supervised or un-supervised based on flists
    '''
    def __init__(self, pet_root, ct_root, pet_flist, ct_flist, res=[128, 128], do_aug=True):
        self.pet_root = pet_root
        self.ct_root = ct_root
        self.pet_flist = pet_flist
        self.ct_flist = ct_flist
        self.res = res
        self.do_aug = do_aug
    def __getitem__(self, index):
        pet_impath = self.pet_flist[index]
        pet_img = np.load(os.path.join(self.pet_root, pet_impath))
        ct_impath = self.ct_flist[index]
        ct_img = np.load(os.path.join(self.ct_root, ct_impath))
        
        if self.do_aug:
            p1 = random.random()
            if p1 < 0.5:
                pet_img, ct_img = np.fliplr(pet_img), np.fliplr(ct_img)
            p2 = random.random()
            if p2 < 0.5:
                pet_img, ct_img = np.flipud(pet_img), np.flipud(ct_img)
        # crop
        from skimage.transform import resize
        ct_img = resize(ct_img, (self.res[0], self.res[1]))
        pet_img = resize(pet_img, (self.res[0], self.res[1]))
        # normalize
        pet_min = pet_img.min()
        pet_max = pet_img.max()
        pet_img = (pet_img - pet_min) / (pet_max - pet_min) 
        ct_min = ct_img.min()
        ct_max = ct_img.max()
        ct_img = (ct_img - ct_min) / (ct_max - ct_min) 

        ct_img = torch.from_numpy(ct_img).unsqueeze(0).unsqueeze(0)
        pet_img = torch.from_numpy(pet_img).unsqueeze(0).unsqueeze(0)

        return {
            'ct': ct_img,
            'pet': pet_img,
            'ct_min': ct_min,
            'ct_max': ct_max,
            'pet_min': pet_min,
            'pet_max': pet_max
        }
    def __len__(self):
        return len(self.ct_flist)