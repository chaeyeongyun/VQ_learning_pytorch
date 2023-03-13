import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import os

class FolderDataset(Dataset):
    def __init__(self, data_dir, resize):
        self.images = glob(os.path.join(data_dir, '*.png'))
        self.resize  = resize
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = img.resize(self.resize, resample=Image.Resampling.BILINEAR)
        img = TF.to_tensor(img)
        return img
