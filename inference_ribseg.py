#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@File: main_partseg.py
@Time: 2021/7/20 7:49 PM
"""


from __future__ import print_function
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ShapeNetPart

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ribseg_dataset import RibSegDataset

from model import DGCNN_partseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from plyfile import PlyData, PlyElement
from tqdm import tqdm

global class_cnts
class_indexs = np.zeros((16,), dtype=int)
global visual_warning
visual_warning = True

# class_choices = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
seg_num = [25]
index_start = [0]

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name+'/'+'inference'):
        os.makedirs('outputs/'+args.exp_name+'/'+'inference')

def inference(args):
    time_samples = []
    # number of samples to run for timing
    dry_run = args.dry_run
    if dry_run:
        print("Dry run, not saving anything")
    batch_size = args.test_batch_size
    test_dataset = RibSegDataset(
        root=args.dataset_path, split="all", npoints=args.num_points, eval=True, binary_root = args.binary_dataset_path
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    seg_num_all = test_loader.dataset.seg_num_all
    if args.binary:
        print("binary network")
        seg_num_all = 2
    seg_start_index = 0
    if args.model == "dgcnn":
        model = DGCNN_partseg(args, seg_num_all).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    for fn, ct_list, label_list in tqdm(test_loader):
        if dry_run and (len(time_samples) > dry_run):
            break
        time_samples.append(0)
        all_data = []
        all_segs = []
        all_preds = []

        for batch_idx in tqdm(range(0, len(ct_list), batch_size), leave=False):
            data = ct_list[batch_idx : batch_idx + batch_size]
            data = torch.cat(data, dim=0)
            seg = label_list[batch_idx : batch_idx + batch_size]
            seg = torch.cat(seg, dim=0)

            label = np.zeros(seg.shape[0], dtype=int)
            seg = seg - seg_start_index
            if args.binary:
                seg = (seg>0).to(seg.dtype)
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = (
                data.to(device),
                label_one_hot.to(device),
                seg.to(device),
            )
            with torch.no_grad():
                data = data.permute(0, 2, 1)
                start = time.time()
                seg_pred = model(data, label_one_hot)
                end = time.time()
                time_samples[-1] += end - start
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()

            data_np = data.permute(0, 2, 1).cpu().numpy()
            data_np = data_np.reshape(-1, data_np.shape[2])
            seg_np = seg.cpu().numpy().reshape(-1)
            pred_np = seg_pred.detach().cpu().numpy().reshape(-1, seg_pred.shape[2])
            all_data.append(data_np)
            all_segs.append(seg_np)
            all_preds.append(pred_np)

        if not dry_run:
            name = fn[0].split("/")[-1].split(".")[0]
            np.savez(
                "outputs/" + args.exp_name + "/" + "inference/" + name + ".npz",
                data=np.concatenate(all_data, axis=0),
                seg=np.concatenate(all_segs, axis=0),
                pred=np.concatenate(all_preds, axis=0),
            )
    print(f"Average inference time: {np.mean(time_samples):.4f} seconds with std: {np.std(time_samples):.4f}")


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="Point Cloud Part Segmentation")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp",
        metavar="N",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dgcnn",
        metavar="N",
        choices=["dgcnn"],
        help="Model to use, [dgcnn]",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="batch_size",
        help="Size of batch)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=16,
        metavar="batch_size",
        help="Size of batch)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of episode to train ",
    )
    parser.add_argument("--use_sgd", type=bool, default=True, help="Use SGD")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001, 0.1 if using sgd)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cos",
        metavar="N",
        choices=["cos", "step"],
        help="Scheduler to use, [cos, step]",
    )
    parser.add_argument(
        "--no_cuda", type=bool, default=False, help="enables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"

    )
    parser.add_argument("--eval", type=bool, default=False, help="evaluate the model")
    parser.add_argument(
        "--num_points", type=int, default=2048, help="num of points to use"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument(
        "--emb_dims",
        type=int,
        default=1024,
        metavar="N",
        help="Dimension of embeddings",
    )
    parser.add_argument(
        "--k", type=int, default=40, metavar="N", help="Num of nearest neighbors to use"
    )
    parser.add_argument(
        "--model_path", type=str, default="", metavar="N", help="Pretrained model path"
    )
    parser.add_argument("--visu", type=str, default="", help="visualize the model")
    parser.add_argument(
        "--visu_format", type=str, default="ply", help="file format of visualization"
    )
    parser.add_argument(
        "--binary", action="store_true", help="Use binary classification"
    )
    parser.add_argument('--dataset_path', type=str, default='', metavar='N')
    parser.add_argument('--binary_dataset_path', type=str, default='', metavar='N')
    parser.add_argument('--dry_run', type=int, default=0, metavar='N')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        print(
            "Using GPU : "
            + str(torch.cuda.current_device())
            + " from "
            + str(torch.cuda.device_count())
            + " devices"
        )
        torch.cuda.manual_seed(args.seed)
    else:
        print("Using CPU")

    _init_()
    inference(args)

# python main_ribseg.py --exp_name=ribseg_8096_40_64 --num_points=8096 --k=40 --batch_size=64
# python inference_ribseg.py --exp_name=ribseg_2048_40_32 --num_points=2048 --k=40 --test_batch_size=32 --model_path="/data/adhinart/ribseg/dgcnn.pytorch/outputs/ribseg_2048_40_32/models/model.t7"
# python inference_ribseg.py --exp_name=ribseg_2048_40_32_binary --num_points=2048 --k=40 --test_batch_size=32 --model_path="/data/adhinart/ribseg/dgcnn.pytorch/outputs/ribseg_2048_40_32_binary/models/model.t7"
