# Copyright (c) Facebook, Inc. and its affiliates.

from factor_learning.dataio.DigitImageTfDataset import DigitImageTfDataset

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader, ConcatDataset

def data_loader(dataset_names, imgfeat_type="keypoint", datatype="train"):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_list = []
    for dataset_name in dataset_names:
        srcdir_dataset = "{0}/local/datasets/{1}".format(
            BASE_PATH, dataset_name)
        dataset_list.append(DigitImageTfDataset("{0}/{1}".format(srcdir_dataset, datatype), transform=transform,
                                                        imgfeat_type=imgfeat_type, downsample_imgs=False))

    dataset = ConcatDataset(dataset_list)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=2)

    return dataloader

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def main():

    # dataset_names = ["20200624_pushing-6in-disc-curves", "20200624_pushing-6in-disc-straight-line"]
    # dataset_names = ["20200928_rectangle-pushing-edges", "20200928_rectangle-pushing-corners"]
    dataset_names = ["20200624_pushing-6in-disc-curves", "20200624_pushing-6in-disc-straight-line",
                     "20200624_pushing-6in-disc-trial1", "20200928_rectangle-pushing-edges", "20200928_rectangle-pushing-corners"]

    # imgfeat_type: ellip, keypoint
    imgfeat_type = "ellip"

    dataloader = data_loader(dataset_names, imgfeat_type, datatype="all")

    mean_imgfeat, std_imgfeat = None, None
    mean_img, std_img = None, None
    for batch_idx, data in enumerate(dataloader):
        assert(batch_idx == 0)
        img, tf, imgfeat = data

        mean_img = torch.FloatTensor([torch.mean(img[:, 0, :, :]), torch.mean(img[:, 1, :, :]), 
                torch.mean(img[:, 2, :, :])])
        std_img = torch.FloatTensor([torch.std(img[:, 0, :, :]), torch.std(img[:, 1, :, :]), 
                torch.std(img[:, 2, :, :])])
        
        mean_imgfeat = torch.mean(imgfeat, 0)
        std_imgfeat = torch.std(imgfeat, 0)

    print("Using a dataset of {0} image,\n mean_img: {1},\n std_img: {2}".format(len(dataloader.dataset), mean_img, std_img))
    print("Using a dataset of {0} imagefeats,\n mean_imgfeat: {1},\n std_imgfeat: {2}".format(len(dataloader.dataset), mean_imgfeat, std_imgfeat))

if __name__ == "__main__":
    main()
