from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

class QCDataset(Dataset):
    def __init__(self, slide_dir, mask_dir, transform=None):
        #array of directories, mask dir is just the simple directory into masks
        self.slide_dir = slide_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_resolution = 320

    def __len__(self):
        return len(self.slide_dir)

    def __getitem__(self, index):
        slideLocation = self.slide_dir[index]
        detailedName = os.path.basename(slideLocation)
        detailedName = detailedName.split('_tile_', 1)
        slideName = detailedName[0]
        tileNumber = detailedName[1]
        tileNumber = tileNumber.replace('.png', '')

        fullMaskDir = self.mask_dir + slideName + '.csv'
        maskDF = pd.read_csv(fullMaskDir)
        tileNumber = tileNumber.split(' ', 1)
        tileNumber = tileNumber[0]
        tileRow =  maskDF[maskDF['Tile Number'] == int(tileNumber)]

        selectedLabel = int(tileRow['selectedLabel'])

        if selectedLabel == 9:
            selectedLabel = 1
        elif selectedLabel == 2:
            selectedLabel = 0
        elif selectedLabel == 7 or selectedLabel == 8:
            selectedLabel = 1
        

        img = Image.open(slideLocation).convert("RGB")

        imgNPArray = np.array(img)
        self.image_resolution = np.shape(imgNPArray)[0]
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.image_resolution, self.image_resolution)),
                        transforms.RandomHorizontalFlip(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            img = transform(img)

        y_label = torch.tensor(selectedLabel)

        return (img, y_label)
