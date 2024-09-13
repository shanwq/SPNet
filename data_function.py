from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('./')
import time
import csv
import SimpleITK as sitk
import torch
from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler,LabelSampler,WeightedSampler
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from pathlib import Path

from hparam import hparams as hp

class MedData_train_mip_3d_sparse(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, mip_3d_sparse_train_dir):

        patch_size = hp.patch_size
            # patch_size = 512, 512, 46
        queue_length = 10
        samples_per_volume = 10

        self.subjects = []

        if (hp.in_class == 1) and (hp.out_class == 1) :

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))
            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(hp.fold_arch))
            mip_3d_sparse_train_dir = Path(mip_3d_sparse_train_dir)
            self.mip_3d_sparse_paths = sorted(mip_3d_sparse_train_dir.glob(hp.fold_arch))

            for (image_path, label_path, mip_3d_sparse_path) in zip(self.image_paths, self.label_paths, self.mip_3d_sparse_paths):

                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                    mip_3d_sparse_label=tio.ScalarImage(mip_3d_sparse_path),
                )
                self.subjects.append(subject)

        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        self.queue_dataset = Queue(
            self.training_set,
            queue_length,
            samples_per_volume,
            #UniformSampler(patch_size),
            # WeightedSampler(patch_size, 'label'),
            LabelSampler(patch_size),
        ) 

    def transform(self):


        training_transform = Compose([
        #ToCanonical(),
        #CropOrPad((hp.crop_or_pad_size, hp.crop_or_pad_size, hp.crop_or_pad_size), padding_mode='reflect'),
        #RandomMotion(),
        #RandomBiasField(),
        ZNormalization(),
        #RandomNoise(),
        # RandomFlip(axes=(0,)),
        #OneOf({
        #    RandomAffine(): 0.8,
        #    RandomElasticDeformation(): 0.2,
        #}),         
        ])
        return training_transform


