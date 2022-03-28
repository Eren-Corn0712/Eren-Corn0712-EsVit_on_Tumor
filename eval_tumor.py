# Modified by Chunyuan Li (chunyl@microsoft.com)
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import json

import PIL.Image
import wandb
import torch
import torch.distributed as dist
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from plot import plot_augmentation
from PIL import Image
from pathlib import Path
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

import utils
import models.vision_transformer as vits
from models import build_model

from config import config
from config import update_config
from config import save_config

from utils import ResizePadding


class TumorDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.ground_truth = {'B': 0, 'M': 1}
        self.benign_mass = os.path.join(self.root, 'benign_mass')
        self.malignant_tumor = os.path.join(self.root, 'malignant_tumor')
        self.img_file_list = list(os.listdir(self.benign_mass)) + list(os.listdir(self.malignant_tumor))
        self.transform = transform

    def __len__(self):
        return len(self.img_file_list)

    def __getitem__(self, item):
        img_file_name = self.img_file_list[item]
        b_or_m = img_file_name.split('_')[0]
        patient_number = img_file_name.split('_')[1]
        if b_or_m == 'B':
            img_file_path = os.path.join(self.benign_mass, img_file_name)
        elif b_or_m == 'M':
            img_file_path = os.path.join(self.malignant_tumor, img_file_name)
        else:
            raise ValueError("Not a valid image")
        img = Image.open(img_file_path)

        if self.transform:
            img = self.transform(img)

        label = self.ground_truth[b_or_m]
        return img, label, patient_number


def eval_linear(args):
    # utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    train_transform = pth_transforms.Compose([
        ResizePadding(size=224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.RandomVerticalFlip(),
        pth_transforms.ToTensor(),
    ])
    val_transform = pth_transforms.Compose([
        ResizePadding(size=224),
        pth_transforms.ToTensor(),
    ])

    if args.zip_mode:

        from .zipdata import ZipData

        datapath_train = os.path.join(config.DATA.DATA_PATH, 'train.zip')
        data_map_train = os.path.join(config.DATA.DATA_PATH, 'train_map.txt')

        datapath_val = os.path.join(config.DATA.DATA_PATH, 'val.zip')
        data_map_val = os.path.join(config.DATA.DATA_PATH, 'val_map.txt')

        dataset_train = ZipData(datapath_train, data_map_train, train_transform)
        dataset_val = ZipData(datapath_val, data_map_val, val_transform)

    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)

    # TODO:Weight Sampler
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=None,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # plot_augmentation(val_loader)
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ create folder............. ============
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ============ building network ... ============
    # if the network is a 4-stage vision transformer (i.e. swin)
    if 'swin' in args.arch:
        update_config(config, args)
        model = build_model(config, is_teacher=True)

        swin_spec = config.MODEL.SPEC
        embed_dim = swin_spec['DIM_EMBED']
        depths = swin_spec['DEPTHS']
        num_heads = swin_spec['NUM_HEADS']

        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d

        print(num_features)
        num_features_linear = sum(num_features[-args.n_last_blocks:])

        print(f'num_features_linear {num_features_linear}')

        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)


    # if the network is a 4-stage vision transformer (i.e. longformer)
    elif 'vil' in args.arch:
        update_config(config, args)
        model = build_model(config, is_teacher=True)

        msvit_spec = config.MODEL.SPEC
        arch = msvit_spec.MSVIT.ARCH

        layer_cfgs = model.layer_cfgs
        num_stages = len(model.layer_cfgs)
        depths = [cfg['n'] for cfg in model.layer_cfgs]
        dims = [cfg['d'] for cfg in model.layer_cfgs]
        out_planes = model.layer_cfgs[-1]['d']
        Nglos = [cfg['g'] for cfg in model.layer_cfgs]

        print(dims)

        num_features = []
        for i, d in enumerate(depths):
            num_features += [dims[i]] * d

        print(num_features)
        num_features_linear = sum(num_features[-args.n_last_blocks:])

        print(f'num_features_linear {num_features_linear}')

        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)


    # if the network is a 4-stage vision transformer (i.e. CvT)
    elif 'cvt' in args.arch:
        update_config(config, args)
        model = build_model(config, is_teacher=True)

        cvt_spec = config.MODEL.SPEC
        embed_dim = cvt_spec['DIM_EMBED']
        depths = cvt_spec['DEPTH']
        num_heads = cvt_spec['NUM_HEADS']

        print(f'embed_dim {embed_dim} depths {depths}')
        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim[i])] * int(d)

        print(num_features)
        num_features_linear = sum(num_features[-args.n_last_blocks:])

        print(f'num_features_linear {num_features_linear}')

        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)



    # if the network is a vanilla vision transformer (i.e. deit_tiny, deit_small, vit_base)
    elif args.arch in vits.__dict__.keys():
        depths = []
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        linear_classifier = LinearClassifier(model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens)),
                                             args.num_labels)

    model.cuda()
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    state_dict = torch.load(args.linear_weights, map_location="cpu")
    state_dict = state_dict['state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    linear_classifier.load_state_dict(state_dict, strict=True)
    linear_classifier = linear_classifier.cuda()
    # linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    # ============ weight and bias logger......============
    if args.wandb_logger:
        wandb.login(key="6ca9efcfdd0230ae14f6160f01209f0ac93aff34")
        wandb_logger = wandb.init(project=args.wandb_project,
                                  config=vars(args),
                                  name=f"{args.wandb_name}_{args.checkpoint_key}",
                                  dir=args.output_dir)
    else:
        wandb_logger = None

    patient_state = dict()

    model.eval()
    linear_classifier.eval()
    with torch.no_grad():
        for idx, (image, label) in enumerate(dataset_val):
            file_name = dataset_val.samples[idx][0].split("\\")[-1]
            b_or_m, patient_number = file_name.split("_")[0], file_name.split("_")[1]
            patient_number = b_or_m + '_' + patient_number
            image = image.unsqueeze(0).cuda(non_blocking=True)
            # compute output
            output = model.forward_return_n_last_blocks(image, args.n_last_blocks, args.avgpool_patchtokens, depths)
            output = linear_classifier(output)

            preds = torch.argmax(output, dim=-1)

            if patient_number not in patient_state.keys():
                patient_state[patient_number] = [1, 0, 0]
            else:
                patient_state[patient_number][0] += 1

            if preds.item() == label:
                patient_state[patient_number][1] += 1
            else:
                patient_state[patient_number][2] += 1

        correct = 0
        error = 0
        for k, v in patient_state.items():
            correct += v[1]
            error += v[2]
            v.append(v[1] / v[0])

        patient_state = pd.DataFrame.from_dict(patient_state, orient='index',
                                               columns=['Total', 'Correct', 'Error', 'Acc'])
        patient_state.to_csv(Path(args.output_dir) / 'test.csv')


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str)

    parser.add_argument('--arch', default='deit_small', type=str,
                        choices=['cvt_tiny', 'swin_tiny', 'swin_small', 'swin_base', 'swin_large', 'swin', 'vil',
                                 'vil_1281', 'vil_2262', 'deit_tiny', 'deit_small', 'vit_base'] + torchvision_archs,
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""")

    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating DeiT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for DeiT-Small and to True with ViT-Base.""")

    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')

    # Dataset
    parser.add_argument('--zip_mode', type=utils.bool_flag, default=False, help="""Whether or not
        to use zip file.""")

    parser.add_argument('--num_labels', default=1000, type=int, help='number of classes in a dataset')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # Wandb Setting
    parser.add_argument('--wandb_logger', action="store_true")
    parser.add_argument('--wandb_project', default='None', type=str, help="Please enter the wandb project name")
    parser.add_argument('--wandb_name', default='None', type=str, help="Please enter the wandb exp name")

    # Load LinearClassifier Weight
    parser.add_argument('--linear_weights', default='', type=str, help=" ")
    parser.add_argument('--rank', default=0, type=int)
    args = parser.parse_args()
    eval_linear(args)
