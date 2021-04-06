import numpy as np
from PIL import Image
import glob
import torch
import pickle
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

'''
Pytorch uses datasets and has a very handy way of creating dataloaders in your main.py
Make sure you read enough documentation.
'''

class Data(Dataset):
    def __init__(self, data_dir):
    #gets the data from the directory
        batch_list = glob.glob(data_dir+'/*')
        self.image_list = []
        self.data = []
        for batch in batch_list:
            with open(batch, 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
            data = data_dict[b'data']
            for i in range(len(data)):
                self.image_list.append(data[i])

        self.data_len = 4 * len(self.image_list)

        for index in range(self.data_len//4):
            image = self.image_list[index]
            image_np = np.asarray(image)/255
            image_np = np.reshape(image_np,(3,32,32))
            image_np = np.transpose(image_np, (1,2,0))
            rot0 = image_np
            rot90 = np.rot90(image_np)
            rot180 = np.rot90(rot90)
            rot270 = np.rot90(rot180)
            rots = [rot0, rot90, rot180, rot270]
            for i in range(4):
                data = torch.from_numpy(np.transpose(rots[i], (2, 0, 1)).copy()).float()
                self.data.append((data, i))

    def __getitem__(self, index):
        img, label = self.data[index]
        return (img, label)

    def __len__(self):
        return self.data_len
