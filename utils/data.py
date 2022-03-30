from sklearn.utils import resample
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image, ImageOps
import os, random

CROP_SIZE = 48

def get_path(dataset: str, scale_factor=0):
    path = os.getcwd() + "/data/" + dataset + "/"
    # Train dataset
    if dataset == "291Trainset":
        return path
    # Test dataset
    if not scale_factor:
        return path + "HR/"
    return path + "LR_bicubic/X" + str(scale_factor) + "/"
    

def load_img(img_path: str):
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    return img

def make_lr_img(hr_img: Image, scale_factor: int):
    hr_size = CROP_SIZE
    lr_size = CROP_SIZE // scale_factor
    lr_img = hr_img.resize((lr_size, lr_size), resample=Image.BICUBIC)
    lr_img = lr_img.resize((hr_size, hr_size), resample=Image.BICUBIC)

    return lr_img

class VDSRTrainDataset(Dataset):
    def __init__(self, scale_factor, rotate=False, flip=False):
        super().__init__()
        self.dataset = "291Trainset"
        self.train_image_path = get_path(self.dataset)
        self.train_images = sorted(os.listdir(self.train_image_path))

        self.scale_factor = scale_factor
        self.rotate = rotate
        self.flip = flip
        
    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        img = load_img(self.train_image_path + self.train_images[idx])
        
        transform = transforms.RandomCrop(CROP_SIZE)
        img = transform(img)

        if self.rotate:
            degree = random.randint(1, 3) # 90, 180, 270
            img = img.rotate(degree * 90)
        
        if self.flip:
            rd = random.randint(0, 1) # LEFT_RIGHT or TOP_BOTTOM
            img = img.transpose(Image.FLIP_TOP_BOTTOM if rd else Image.FLIP_LEFT_RIGHT)
        
        transform = transforms.ToTensor()
        hr_tensor = transform(img)

        lr_img = make_lr_img(img, self.scale_factor)
        lr_tensor = transform(lr_img)

        return lr_tensor, hr_tensor

class VDSRTestDataset(Dataset):
    def __init__(self, dataset: str, scale_factor: int):
        super().__init__()
        self.dataset = dataset
        self.scale_factor = scale_factor
        self.test_image_path = get_path(self.dataset)
        self.test_images = sorted(os.listdir(self.test_image_path))
        
    def __len__(self):
        return len(self.test_images)
    
    def __getitem__(self, idx):
        img = load_img(self.train_image_path + self.train_images[idx])
        
        transform = transforms.ToTensor()
        hr_tensor = transform(img)

        lr_img = make_lr_img(img, self.scale_factor)
        lr_tensor = transform(lr_img)

        return lr_tensor, hr_tensor
