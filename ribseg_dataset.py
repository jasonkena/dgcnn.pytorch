# *_*coding:utf-8 *_*

# adapted from https://github.com/M3DV/RibSeg/blob/main/data_utils/dataloader.py


import os
import torch
import numpy as np
from torch.utils.data import Dataset

from pytorch3d.ops import ball_query


def pc_normalize(pc, centroid=None, m=None):
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    pc = pc - centroid

    if m is None:
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m

    return pc, centroid, m


def ARPD(pc_full, pc_ds, npoints_abs=1000, npoints_relative=29, radius=0.05):
    """
    Input:
        pc_full: raw point data, [B, N, 3]
        pc_ds: input points position data, [B, npoints, 3]
    Return:
        arpd: absolute and relateive position data, [B, N, 3 + npoints_relative]
    """
    if pc_full.shape[0] > 100000:
        choice = np.random.choice(pc_full.shape[0], 100000, replace=False)
        pc_full = pc_full[choice]
    pc_full = torch.Tensor(np.expand_dims(pc_full, axis=0))
    pc_ds = torch.Tensor(np.expand_dims(pc_ds, axis=0))

    point_bq = ball_query(pc_ds, pc_full, K=npoints_relative, radius=radius)[2][
        0
    ].numpy()  # [30000,16,3]
    idx_zero = np.argwhere((point_bq == [0.0, 0.0, 0.0]).all(axis=2))
    pc_ds = pc_ds.numpy()[0]  # [30000, 3]

    point_bq -= pc_ds[:, None, :]
    for idx in idx_zero:
        dim1, dim2 = idx[0], idx[1]
        point_bq[dim1, dim2] = np.zeros(3)
    ct = np.concatenate((pc_ds, point_bq.reshape(30000, -1)), axis=1)
    return ct


class RibSegDataset(Dataset):
    def __init__(
        self, root, npoints=30000, split="train", flag_cl=False, flag_arpe=False
    ):
        ignore = [452, 485, 490]

        splits = {"train": range(1,421), "val": range(421,501), "test": range(501,661)}
        splits["trainval"] = list(splits["train"]) + list(splits["val"])
        assert split in splits
        for key in splits:
            splits[key] = [x for x in splits[key] if x not in ignore]

        self.npoints = npoints
        self.root = root
        self.flag_cl = flag_cl
        self.flag_arpe = flag_arpe
        # nclasses: 24 ribs + BG
        self.seg_num_all = 25

        ids = splits[split]

        self.datapath = [os.path.join(self.root, f"RibFrac{idx}.npz") for idx in ids]
        for file in self.datapath:
            if not os.path.exists(file):
                raise Warning(f"{file} not exists")

    def __getitem__(self, index):
        fn = self.datapath[index]
        data = np.load(fn)
        pc = data["seg"].astype(np.float32)
        pc[:, :3], centroid, m = pc_normalize(pc[:, :3])

        ct, label = pc[:, :3], pc[:, 3]

        choice = np.random.choice(ct.shape[0], self.npoints, replace=False)
        ct, label = ct[choice], label[choice]
        label = label.astype(int) # 25 classes, BG + 24 ribs
        dummy = np.zeros(1, dtype=int) # PC label

        if self.flag_arpe:
            ct = ARPD(pc[:, :3], ct)

        if self.flag_cl:
            cl = data["cl"].astype(np.float32)
            cl = cl.reshape(-1, 3)
            idx_list = []
            for i in range(24):
                if cl[i * 500][0] != -1:
                    idx_list.append(i)
            idx = []
            for x in idx_list:
                for a in range(x * 500, x * 500 + 500):
                    idx.append(a)
            if idx:
                cl[idx], _, _ = pc_normalize(cl[idx], centroid, m)
            cl = cl.reshape(-1, 500, 3)
            return ct, dummy, label, cl
        return ct, dummy, label

    def __len__(self):
        return len(self.datapath)

if __name__ == "__main__":
    dataset = RibSegDataset(root="../../ribseg_benchmark")
    a = dataset[0]
