import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
from torchvision.transforms import Compose
from torchvision.transforms import functional 
import torchvision.transforms as transforms
from PIL import Image
import random

VALSPLIT = 60

class BowlDataset(Dataset):
    def __init__(self, train_image, train_mask, test_image, fold="train", transform=None):
        fold = fold.lower()

        self.train = self.test = self.val = False
        
        if fold == "train":
            self.train = True
        elif fold == "test":
            self.test = True
        elif fold == "val":
            self.val = True
        else:
            raise RuntimeError("Not train-val-test")

        self.train_image = train_image
        self.train_mask = train_mask
        self.test_image = test_image
        self.transform = transform
        
        if self.train or self.val:
            index = np.random.permutation(self.train_image.shape[0])
            self.val_imgs = self.train_image[index[:VALSPLIT]]
            self.train_imgs = self.train_image[index[VALSPLIT:]]
            
            self.val_masks = self.train_mask[index[:VALSPLIT]]
            self.train_masks = self.train_mask[index[VALSPLIT:]]


        elif self.test:

            self.test_imgs = self.test_image
            self.test_masks = None


    def __len__(self):
        if self.train:
            return self.train_imgs.shape[0]
        if self.val:
            return self.val_imgs.shape[0]
        if self.test:
            return self.test_imgs.shape[0]


    def __getitem__(self, idx):
        if self.train:
            img, mask = self.train_imgs[idx], self.train_masks[idx]
        elif self.test:
            img, mask = self.test_imgs[idx], None
        elif self.val:
            img, mask = self.val_imgs[idx], self.val_masks[idx]

        sample = {'image': img, 'mask': mask}
        if self.transform is not None:
            sample = self.transform({'image': img, 'mask': mask})

        return sample

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        return fmt_str

class Rescale(object):
    """Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #print("in Rescale", image.shape, mask.shape)

        if mask is not None:
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = cv2.resize(image, (new_h, new_w))
            mask = cv2.resize(mask, (new_h, new_w))
            mask = np.expand_dims(mask, axis=2)
        else:
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = cv2.resize(image, (new_h, new_w))
            mask = None
            #mask = np.zeros(img.shape)

        #print("out rescale", img.shape, mask.shape)


        return {'image': img, 'mask': mask}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #print("in crop", image.shape, mask.shape)

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w, :]

        mask = mask[top: top + new_h, left: left + new_w, :]

        #print("out crop", image.shape, mask.shape)

        return {'image': image, 'mask': mask}

class RandomHorizontallyFlip(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #print("in flip", image.shape, mask.shape)
        if random.random() < 0.5:
            image = cv2.flip(image, flipCode=1)
            mask = cv2.flip(mask, flipCode=1)
            mask = np.expand_dims(mask, axis=2)
        #print("out flip", image.shape, mask.shape)

        return {'image': image, 'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        #print("*****************in tensor********************", image.shape, mask.shape)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.transpose(image, [2, 0, 1]).astype(np.float32)
        if mask is not None:
            #print("tensor", image.shape, mask.shape)
            mask = np.transpose(mask, [2, 0, 1]).astype(np.float32)
            #print("out tensor", image.shape, mask.shape)
            return {'image': torch.from_numpy(image),'mask': torch.from_numpy(mask)}
        else:
            fake_mask = np.zeros(image.shape)
            return {'image': torch.from_numpy(image),'mask': torch.from_numpy(fake_mask)}

# test
if __name__ == '__main__':

    print("loading training and test data......")
    train_image = np.load("./data/train_image_small.npy")
    train_mask = np.load("./data/train_mask_small.npy")
    test_image = np.load("./data/test_image.npy")[0:2]

    print("compose transformation......")
    train_transform = Compose([Rescale(264), RandomCrop(256), RandomHorizontallyFlip(), ToTensor()])
    test_transform = Compose([ToTensor()])
   
    print("load BowlDataset ......")
    dataset = BowlDataset(train_image, train_mask, test_image, fold="test", transform=test_transform)

    print("data loader ......")
    data_loader = DataLoader(dataset, 1, True, num_workers=4)

    for i, batch_data in enumerate(data_loader):
        for i in range(batch_data['image'].size()[0]):
            img = batch_data['image'][i]
            img = img.numpy()       
            img = np.transpose(img, [1, 2, 0]).astype(np.float32)
            cv2.imwrite('color_img_test'+str(i)+'.jpg', img*255)


       

    # for idx, batch_data in enumerate(data_loader):
    #     # print(idx)
    #     # print(batch_data['image'].size())
    #     # print(batch_data['mask'].size())

    #     for i in range(batch_data['image'].size()[0]):
    #         img = batch_data['image'][i]
    #         mask = batch_data['mask'][i]

    #         img = img.numpy()
    #         mask = mask.numpy()

    #         img = np.transpose(img, [1, 2, 0]).astype(np.float32)
    #         mask = np.transpose(mask, [1, 2, 0]).astype(np.float32)

    #         mask = np.repeat(mask, 3, 2)
    #         concat_img = np.concatenate((img, mask), axis=1)

    #         cv2.imwrite('color_img_'+str(i)+'.jpg', concat_img*255)

        