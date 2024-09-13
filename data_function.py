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

class Med_unlabel_train(torch.utils.data.Dataset):
    def __init__(self, unlabel_dir):

        if hp.mode == '3d':
            patch_size = hp.patch_size
            # patch_size = 128, 128, 64
        elif hp.mode == '2d':
            patch_size = hp.patch_size
        else:
            raise Exception('no such kind of mode!')

        queue_length = 5
        samples_per_volume = 5


        self.subjects = []



        unlabels_dir = Path(unlabel_dir)
        self.unlabel_paths = sorted(unlabels_dir.glob(hp.fold_arch))

        for (image_path) in zip(self.unlabel_paths):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
            )
            self.subjects.append(subject)
        

        self.transforms = self.transform1()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        self.queue_dataset = Queue(
            self.training_set,
            queue_length,
            samples_per_volume,
            UniformSampler(patch_size),
            # LabelSampler(patch_size),
        )

        self.transforms_2 = self.transform2()
        self.training_set_2 = tio.SubjectsDataset(self.subjects, transform=self.transforms_2)

        self.queue_dataset_2 = Queue(
            self.training_set_2,
            queue_length,
            samples_per_volume,
            UniformSampler(patch_size),
            # LabelSampler(patch_size),
        )


    def transform1(self):

        if hp.mode == '3d':
            # if hp.aug:
            training_transform = Compose([
            # ToCanonical(),
            # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
            #     # RandomMotion(),
            # RandomBiasField(),
            ZNormalization(),
            RandomNoise(),
            # RandomFlip(axes=(0,)),
            # OneOf({
            #     RandomAffine(): 0.8,
            #     RandomElasticDeformation(): 0.2,
            # })
            # ,])
            # else:
            # training_transform = Compose([
            # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
            # ZNormalization(),
            ])
        elif hp.mode == '2d':
            if hp.aug:
                training_transform = Compose([
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                # RandomMotion(),
                RandomBiasField(),
                ZNormalization(),
                RandomNoise(),
                RandomFlip(axes=(0,)),
                OneOf({
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }),])
            else:
                training_transform = Compose([
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                ZNormalization(),
                ])

        else:
            raise Exception('no such kind of mode!')


        return training_transform


    def transform2(self):

        if hp.mode == '3d':
            # if hp.aug:
            training_transform = Compose([
                # ToCanonical(),
            # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
            #     # RandomMotion(),
            # RandomBiasField(),
            # RandomNoise(),
            # RandomFlip(axes=(0,)),
            # OneOf({
            #     RandomAffine(): 0.8,
            #     RandomElasticDeformation(): 0.2,
            # }),])
            # else:
            # training_transform = Compose([
            # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
            ZNormalization(),
            ])
        elif hp.mode == '2d':
            if hp.aug:
                training_transform = Compose([
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                # RandomMotion(),
                RandomBiasField(),
                ZNormalization(),
                RandomNoise(),
                RandomFlip(axes=(0,)),
                OneOf({
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }),])
            else:
                training_transform = Compose([
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                ZNormalization(),
                ])

        else:
            raise Exception('no such kind of mode!')


        return training_transform



    def transform(self):

        if hp.mode == '3d':
            # if hp.aug:
            training_transform = Compose([
            # ToCanonical(),
            # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
            # RandomMotion(),
            # RandomBiasField(),
            ZNormalization(),
            RandomNoise(),
            # RandomFlip(axes=(0,)),
            # OneOf({
            #     RandomAffine(): 0.8,
            #     RandomElasticDeformation(): 0.2,
            # }),])
            # else:
            # training_transform = Compose([
            # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
            # ZNormalization(),
            ])
        elif hp.mode == '2d':
            if hp.aug:
                training_transform = Compose([
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                # RandomMotion(),
                RandomBiasField(),
                ZNormalization(),
                RandomNoise(),
                RandomFlip(axes=(0,)),
                OneOf({
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }),])
            else:
                training_transform = Compose([
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                ZNormalization(),
                ])

        else:
            raise Exception('no such kind of mode!')


        return training_transform


class My_unlabel(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
    def __getitem__(self, index):
        x1 = self.dataset1[index]
        x2 = self.dataset2[index]
        return x1,x2
    def __len__(self):
        return len(self.dataset1)

class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):

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
            

            for (image_path, label_path) in zip(self.image_paths, self.label_paths):
                # start_time = time.time()
                # source=tio.ScalarImage(image_path),
                # print(source)
                # source_arr = source[0].data
                # print(source_arr.shape)#, source_arr)
                # source_arr_transpose = torch.transpose(source_arr, 1,3)
                # print(source_arr_transpose.shape)
                # source[0].set_data(source_arr_transpose)
                # print(source)
                # print('耗时',time.time()-start_time)
                
                # aa = sitk.ReadImage(image_path)
                # aa_arr = sitk.GetArrayFromImage(aa)
                # aa_arr = np.expand_dims(aa_arr, axis=0)
                # print(aa_arr.shape,)
                # source[0].set_data(aa_arr)
                # print(source)
                
                # label=tio.LabelMap(label_path),
                # label_arr = label.data[0]
                # label_arr_transpose = label_arr.transpose(1,3)
                # label.set_data(label_arr_transpose)
                # print('耗时',time.time()-start_time)
                # label_arr = label.data[0]
                # label_arr_transpose = label_arr.transpose(1,3)
                # label.set_data(label_arr_transpose)
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                )
                self.subjects.append(subject)
            
     
        else:
            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))

            artery_labels_dir = Path(labels_dir+'/artery')
            self.artery_label_paths = sorted(artery_labels_dir.glob(hp.fold_arch))

            lung_labels_dir = Path(labels_dir+'/lung')
            self.lung_label_paths = sorted(lung_labels_dir.glob(hp.fold_arch))

            trachea_labels_dir = Path(labels_dir+'/trachea')
            self.trachea_label_paths = sorted(trachea_labels_dir.glob(hp.fold_arch))

            vein_labels_dir = Path(labels_dir+'/vein')
            self.vein_label_paths = sorted(vein_labels_dir.glob(hp.fold_arch))


            for (image_path, artery_label_path,lung_label_path,trachea_label_path,vein_label_path) in zip(self.image_paths, self.artery_label_paths,self.lung_label_paths,self.trachea_label_paths,self.vein_label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    atery=tio.LabelMap(artery_label_path),
                    lung=tio.LabelMap(lung_label_path),
                    trachea=tio.LabelMap(trachea_label_path),
                    vein=tio.LabelMap(vein_label_path),
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
                # print(image_path)
                # print(label_path)
                # print(mip_3d_sparse_path)
                # start_time = time.time()
                # source=tio.ScalarImage(image_path),
                # print(source)
                # source_arr = source[0].data
                # print(source_arr.shape)#, source_arr)
                # source_arr_transpose = torch.transpose(source_arr, 1,3)
                # print(source_arr_transpose.shape)
                # source[0].set_data(source_arr_transpose)
                # print(source)
                # print('耗时',time.time()-start_time)
                
                # aa = sitk.ReadImage(image_path)
                # aa_arr = sitk.GetArrayFromImage(aa)
                # aa_arr = np.expand_dims(aa_arr, axis=0)
                # print(aa_arr.shape,)
                # source[0].set_data(aa_arr)
                # print(source)
                
                # label=tio.LabelMap(label_path),
                # label_arr = label.data[0]
                # label_arr_transpose = label_arr.transpose(1,3)
                # label.set_data(label_arr_transpose)
                # print('耗时',time.time()-start_time)
                # label_arr = label.data[0]
                # label_arr_transpose = label_arr.transpose(1,3)
                # label.set_data(label_arr_transpose)
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                    mip_3d_sparse_label=tio.ScalarImage(mip_3d_sparse_path),
                )
                self.subjects.append(subject)
            
     
        else:
            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))

            artery_labels_dir = Path(labels_dir+'/artery')
            self.artery_label_paths = sorted(artery_labels_dir.glob(hp.fold_arch))

            lung_labels_dir = Path(labels_dir+'/lung')
            self.lung_label_paths = sorted(lung_labels_dir.glob(hp.fold_arch))

            trachea_labels_dir = Path(labels_dir+'/trachea')
            self.trachea_label_paths = sorted(trachea_labels_dir.glob(hp.fold_arch))

            vein_labels_dir = Path(labels_dir+'/vein')
            self.vein_label_paths = sorted(vein_labels_dir.glob(hp.fold_arch))


            for (image_path, artery_label_path,lung_label_path,trachea_label_path,vein_label_path) in zip(self.image_paths, self.artery_label_paths,self.lung_label_paths,self.trachea_label_paths,self.vein_label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    atery=tio.LabelMap(artery_label_path),
                    lung=tio.LabelMap(lung_label_path),
                    trachea=tio.LabelMap(trachea_label_path),
                    vein=tio.LabelMap(vein_label_path),
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





class MedData_test(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):


        self.subjects = []
        self.name_list = []

        if (hp.in_class == 1) and (hp.out_class == 1) :

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))
            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(hp.fold_arch))

            for (image_path, label_path) in zip(self.image_paths, self.label_paths):
                
                # source=tio.ScalarImage(image_path),
                # source_arr = source.data
                # source_arr_transpose = source_arr.transpose(1,3)
                # source.set_data(source_arr_transpose)
                
                # label=tio.LabelMap(label_path),
                # label_arr = label.data
                # label_arr_transpose = label_arr.transpose(1,3)
                # label.set_data(label_arr_transpose)
                
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                )
                self.subjects.append(subject)
                self.name_list.append(str(image_path).split('/')[-1])
                
        else:
            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))

            artery_labels_dir = Path(labels_dir+'/artery')
            self.artery_label_paths = sorted(artery_labels_dir.glob(hp.fold_arch))

            lung_labels_dir = Path(labels_dir+'/lung')
            self.lung_label_paths = sorted(lung_labels_dir.glob(hp.fold_arch))

            trachea_labels_dir = Path(labels_dir+'/trachea')
            self.trachea_label_paths = sorted(trachea_labels_dir.glob(hp.fold_arch))

            vein_labels_dir = Path(labels_dir+'/vein')
            self.vein_label_paths = sorted(vein_labels_dir.glob(hp.fold_arch))


            for (image_path, artery_label_path,lung_label_path,trachea_label_path,vein_label_path) in zip(self.image_paths, self.artery_label_paths,self.lung_label_paths,self.trachea_label_paths,self.vein_label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    atery=tio.LabelMap(artery_label_path),
                    lung=tio.LabelMap(lung_label_path),
                    trachea=tio.LabelMap(trachea_label_path),
                    vein=tio.LabelMap(vein_label_path),
                )
                self.subjects.append(subject)


        # self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=None)


    # def transform(self):

    #     training_transform = Compose([
    #     ZNormalization(),
    #     ])
        


    #     return training_transform
