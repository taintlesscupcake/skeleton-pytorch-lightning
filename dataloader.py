import torch
from torch.utils.data import Dataset
from PIL import Image
import os

def load_image(path):
    return Image.open(path).convert('RGB')

class YourDataset(Dataset):
    def __init__(self, mode, data_path, data_files, transform=None):
        # implement your own data
        self.data_path = data_path
        self.data_files = data_files
        self.transform = transform
        self.mode = mode
        if mode == 'train':
            # train data
            pass
        else:
            # test data
            pass
        
    def __getitem__(self, index):
        # implement your own datas.
        # it is only example.
        imgpath = os.path.join(self.data_path, self.data_files[index])
        img = load_image(imgpath)
        
        label = os.path.join(self.data_path, self.data_files[index].replace('jpg', 'txt'))
        label = open(label, 'r').read().split('\n')
        label = [float(i) for i in label]
        label = torch.tensor(label)
        
        img.resize((224, 224))
        
        # return datas will be batch.
        return img, label
    
    def __len__(self):
        return len(self.data_files)
    
# Replace YourDataset with your own dataset class
DefaultDataset = YourDataset