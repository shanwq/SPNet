import os


import time
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
import torch.nn.functional as F

from copy import deepcopy
from torch import nn
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment

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
    Compose,
)
from medpy.io import load,save
from tqdm import tqdm
from torchvision import utils
from hparam import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
import json
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def weights_init_normal(m):
    classname = m.__class__.__name__
    gain = 0.02
    init_type = hp.init_type

    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
            m.reset_parameters()
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir
# unlabel_dir_train = hp.unlabel_dir

os.makedirs(hp.output_dir, exist_ok=True)

source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

output_int_dir = hp.output_int_dir
output_float_dir = hp.output_float_dir

hpparams_dict = {key: value for key, value in hp.__dict__.items() if not key.startswith('__') and not callable(key)}

print(hpparams_dict)

with open(hp.output_dir+'/experimental_settings.json', 'w', encoding='utf-8') as file:
    json.dump(hpparams_dict, file, ensure_ascii=False, indent=4)


def parse_training_args(parser):
    """	
    """
    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest_checkpoint_file', type=str, default='checkpoint_latest.pt', help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup') 

    training.add_argument('--epochs', type=int, default=300, help='Number of total epochs to run')   
    training.add_argument('--epochs_per_checkpoint', type=int, default=1, help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=1, help='batch-size')     #12
    training.add_argument('--sample', type=int, default=12, help='number of samples during training')    #12

    parser.add_argument("--ckpt",type=str, default=None, help="path to the checkpoints to resume training",)

    parser.add_argument("--init-lr", type=float, default=0.005, help="learning rate")   #0.001
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    training.add_argument('--amp-run', action='store_true', help='Enable AMPa')
    training.add_argument('--cudnn-enabled', default=True, help='En	able cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true', help='disable uniform initialization of batchnorm layer weight')

    return parser

def compute_loss(output, target, is_max=True, is_c2f=False, is_sigmoid=True, is_max_hungarian=True, is_max_ds=True, point_rend=False, num_point_rend=None, no_object_weight=None):
    total_loss, smooth, do_fg = None, 1e-5, False
    
    if isinstance(output, (tuple, list, dict)):
        len_ds = 1+len(output['aux_outputs']) if isinstance(output, dict) else len(output)

        max_ds_loss_weights = [1] * (len_ds) # previous had a bug with exp weight for 'v0' ..

    if is_max and is_max_hungarian:
        aux_outputs = output['aux_outputs'] # a list of dicts of ['pred_logits', 'pred_masks'], length is 3
        
        num_classes = 2
        target_onehot = torch.zeros_like(target.repeat(1, num_classes, 1, 1, 1), device=target.device)
        target_onehot.scatter_(1, target.long(), 1)
        target_sum = target_onehot.flatten(2).sum(dim=2) # (b, 3)
        targets = []
        for b in range(len(target_onehot)):
            target_mask = target_onehot[b][target_sum[b] > 0] # (K, D, H, W)
            target_label = torch.nonzero(target_sum[b] > 0).squeeze(1) # (K)
            targets.append({'labels':target_label, 'masks':target_mask})
        from loss_function import HungarianMatcher3D, compute_loss_hungarian
        cost_weight = [2.0, 5.0, 5.0]
        matcher = HungarianMatcher3D(
                cost_class=cost_weight[0], # 2.0
                cost_mask=cost_weight[1],
                cost_dice=cost_weight[2],
            )
        outputs_without_aux = {k: v for k, v in output.items() if k != "aux_outputs"}
        loss_list = []
        loss_final = compute_loss_hungarian(outputs_without_aux, targets, 0, matcher, 2, point_rend, num_point_rend, no_object_weight=no_object_weight, cost_weight=cost_weight)
        loss_list.append(max_ds_loss_weights[0] * loss_final)
        if is_max_ds and "aux_outputs" in output:
            for i, aux_outputs in enumerate(output["aux_outputs"][::-1]): # reverse order
                loss_aux = compute_loss_hungarian(aux_outputs, targets, i+1, matcher, 2, point_rend, num_point_rend, no_object_weight=no_object_weight, cost_weight=cost_weight)
                loss_list.append(max_ds_loss_weights[i+1] *loss_aux)

        total_loss = sum(loss_list) / len(loss_list)

        return total_loss


def train():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import MedData_train
    os.makedirs(args.output_dir, exist_ok=True)
    if hp.mode == '3d':
        
        from models.three_d.SPNet import SPNet
        model = SPNet()#input_channels=hp.in_class, out_channels=hp.out_class, init_features=8)#, base_n_filter=2)  #2

    model.apply(weights_init_normal)

    para = list(model.parameters())#+list(model_inv.parameters())
    optimizer = torch.optim.Adam(para, lr=0.0003, weight_decay=3e-05) # 0.0003

    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.8)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs= 20, max_epochs= args.epochs, warmup_start_lr=0.000003,eta_min= 0)
    
    if args.ckpt is not None:
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.cuda()

    writer = SummaryWriter(args.output_dir)
    print(source_train_dir, label_train_dir)
    train_dataset = MedData_train(source_train_dir,label_train_dir)
    train_loader = DataLoader(train_dataset.queue_dataset, 
                            batch_size=args.batch, 
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True)
    
    model.train()

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)

    print(args.output_dir)
    for epoch in range(1, epochs + 1):
        epoch += elapsed_epochs

        train_epoch_avg_loss = 0.0
        num_iters = 0

        for i, batch in enumerate(train_loader):
            t_start=time.time()

            optimizer.zero_grad()

            if (hp.in_class == 1) and (hp.out_class == 1) :
                x = batch['source']['data']
                y = batch['label']['data']
                mip_3D = batch['mpi_sparse']['data']
                x = torch.transpose(x, 2, 4)
                y = torch.transpose(y, 2, 4)
                mip_3D = torch.transpose(mip_3D, 2, 4)
                x = x.type(torch.FloatTensor).cuda()
                y = y.type(torch.FloatTensor).cuda()
                mip_3D = mip_3D.type(torch.FloatTensor).cuda()

            outputs = model(x, mip_3D)
            loss = compute_loss(outputs, y)#, True, False, True, True, is_max_ds, point_rend, num_point_rend, no_object_weight)
            print(f'Batch: {i}/{len(train_loader)} epoch {epoch}, loss:{loss}, lr:{scheduler._last_lr[0]}, time:{time.time()-t_start}')
            
            ##########################################################################################################################
            # num_iters += 1
            loss.backward()

            optimizer.step()
            iteration += 1
            writer.add_scalar('Training/Loss', loss.item(),iteration)

        scheduler.step()

        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )

        if epoch % args.epochs_per_checkpoint == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )

    writer.close()


# def test():

#     parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
#     parser = parse_training_args(parser)
#     args, _ = parser.parse_known_args()

#     args = parser.parse_args()


#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = args.cudnn_enabled
#     torch.backends.cudnn.benchmark = args.cudnn_benchmark

#     from data_function import MedData_test

#     os.makedirs(output_float_dir, exist_ok=True)

#     if hp.mode == '3d':
#         # from models.three_d.SPNet import SPNet
#         # model = SPNet()#(in_channels=hp.in_class, out_channels=hp.out_class, init_features=8)#, base_n_filter=2)
#     model = torch.nn.DataParallel(model)

#     print("load model:", args.ckpt)
#     print(os.path.join(args.output_dir, args.latest_checkpoint_file))
#     ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

#     model.load_state_dict(ckpt["model"])
#     model.cuda()
#     model.eval()


#     test_dataset = MedData_test(source_test_dir,label_test_dir)
#     znorm = ZNormalization()

#     if hp.mode == '3d':
#         patch_overlap = 4,4,4
#         # patch_size = 128, 128, 64


#     for i,subj in enumerate(test_dataset.subjects):
#         subj = znorm(subj)
#         grid_sampler = torchio.inference.GridSampler(
#                 subj,
#                 patch_size,
#                 patch_overlap,
#             )

#         patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
#         aggregator = torchio.inference.GridAggregator(grid_sampler)
#         aggregator_1 = torchio.inference.GridAggregator(grid_sampler)
#         aggregator_rec = torchio.inference.GridAggregator(grid_sampler)
#         aggregator_x = torchio.inference.GridAggregator(grid_sampler)
        
#         with torch.no_grad():
#             for patches_batch in tqdm(patch_loader):
#                 input_tensor = patches_batch['source'][torchio.DATA].to(device)
#                 mip_3d = patches_batch['mip_sparse'][torchio.DATA].to(device)
#                 locations = patches_batch[torchio.LOCATION]
#                 outputs = model(input_tensor,)
#                 logits = torch.sigmoid(outputs)

#                 labels = logits.clone()
#                 labels[labels>0.5] = 1
#                 labels[labels<=0.5] = 0

#                 aggregator.add_batch(logits, locations)
#                 aggregator_1.add_batch(labels, locations)
#                 aggregator_rec.add_batch(x_rec, locations)
#                 aggregator_x.add_batch(input_tensor, locations)
#         output_tensor = aggregator.get_output_tensor()
#         output_tensor_1 = aggregator_1.get_output_tensor()
#         rec_tensor = aggregator_rec.get_output_tensor()
#         x_tensor = aggregator_x.get_output_tensor()

#         affine = subj['source']['affine']
#         if i<9:
#             label_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
#             label_image.save(os.path.join(output_float_dir,"0"+str(i+1)+".mhd"))
#             output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy(), affine=affine)
#             output_image.save(os.path.join(output_int_dir,"0"+str(i+1)+".mhd"))
#             label_image = torchio.ScalarImage(tensor=x_tensor.numpy(), affine=affine)
#             label_image.save(os.path.join(output_generation_x,"0"+str(i+1)+".mhd"))
#             output_image = torchio.ScalarImage(tensor=rec_tensor.numpy(), affine=affine)
#             output_image.save(os.path.join(output_generation_rec,"0"+str(i+1)+".mhd"))

#         else:
#             label_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
#             label_image.save(os.path.join(output_float_dir,str(i+1)+".mhd"))
#             output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy(), affine=affine)
#             output_image.save(os.path.join(output_int_dir,str(i+1)+".mhd"))       
#             output_image = torchio.ScalarImage(tensor=x_tensor.numpy(), affine=affine)
#             output_image.save(os.path.join(output_generation_x,str(i+1)+"_x_input.mhd"))
#             output_image = torchio.ScalarImage(tensor=rec_tensor.numpy(), affine=affine)
#             output_image.save(os.path.join(output_generation_rec,str(i+1)+"_x_rec.mhd"))
   

if __name__ == '__main__':
    if hp.train_or_test == 'train':
        train()
    # elif hp.train_or_test == 'test':
        # test()
