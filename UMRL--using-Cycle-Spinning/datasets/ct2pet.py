from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import os

def get_image_paths_from_dir(fdir):
    flist = os.listdir(fdir)
    flist.sort()
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, type, image_size=(256, 256), max_pixel=255.0, to_norm=True, to_aug=True):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.type = type
        
        # convert max_pixel to float
        self.max_pixel = float(max_pixel)
        self.to_norm = to_norm
        self.to_aug = to_aug

    def __len__(self):
        if self.to_aug:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0

        if index >= self._length:
            index -= self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        
        try:
            np_image = np.load(img_path, allow_pickle=True)

            if np_image.shape[2] == 3: # 3 channels to 1 channel
                np_image = (np_image[:, :, 0] + np_image[:, :, 1] + np_image[:, :, 2]) / 3

            if self.type == 'pet': 
                np_image = np.log1p(np_image)

            np_image = np_image / float(self.max_pixel)

            image = Image.fromarray(np_image)     
            image = transform(image) 
        except BaseException as e:
            print(e)

        if self.to_norm:
            # if self.type == 'pet':
                image = (image - 0.5) * 2.
        
        image = image.repeat(3, 1, 1)  # 1 channel to 3 channels
        
        # print(self.type, image.shape, image.min(), image.max())

        image_name = Path(img_path).stem
        return image, image_name
      
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config):
        super().__init__()
        self.image_size = (dataset_config['image_size'], dataset_config['image_size'])

        image_paths_input = get_image_paths_from_dir(os.path.join(dataset_config['dataset_path'], '200')) # sampled PET with BBDM
        image_paths_corr = get_image_paths_from_dir(os.path.join(dataset_config['dataset_path'], 'condition')) # corresponding CT
        image_paths_gt = get_image_paths_from_dir(os.path.join(dataset_config['dataset_path'], 'ground_truth')) # ground truth PET
        
        self.to_aug = dataset_config['to_aug']
        self.to_norm = dataset_config['to_norm']

        self.imgs_input = ImagePathDataset(image_paths_input, 'pet', self.image_size, dataset_config['max_pixel_input'], to_norm=self.to_norm, to_aug=self.to_aug) 
        self.imgs_corr = ImagePathDataset(image_paths_corr, 'ct', self.image_size, dataset_config['max_pixel_corr'], to_norm=self.to_norm, to_aug=self.to_aug) 
        self.imgs_gt = ImagePathDataset(image_paths_gt, 'pet', self.image_size, dataset_config['max_pixel_gt'], to_norm=self.to_norm, to_aug=self.to_aug) 
        
    def __len__(self):
        return len(self.imgs_gt)

    def __getitem__(self, index):
        return self.imgs_input[index], self.imgs_corr[index], self.imgs_gt[index]