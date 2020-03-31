import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


def Scale(pts, v):
    if random.random() > 0.5:
        v = -v
    v = v * 0.5
    return pts * v

def SheerX(pts, v):
    if random.random() > 0.5:
        v = -v
    v = v * 0.5
    pts[:, 0] = pts[:, 0] * v
    return pts

def SheerY(pts, v):
    if random.random() > 0.5:
        v = -v
    v = v * 0.5
    pts[:, 0] = pts[:, 1] * v
    return pts

def SheerZ(pts, v):
    if random.random() > 0.5:
        v = -v
    v = v * 0.5
    pts[:, 0] = pts[:, 2] * v
    return pts

def ShiftX(pts, v):
    if random.random() > 0.5:
        v = -v
    v = v * 0.5
    pts[:, 0] = pts[:, 0] + v
    return pts

def ShiftY(pts, v):
    if random.random() > 0.5:
        v = -v
    v = v * 0.5
    pts[:, 1] = pts[:, 1] + v
    return pts

def ShiftZ(pts, v):
    if random.random() > 0.5:
        v = -v
    v = v * 0.5
    pts[:, 2] = pts[:, 2] + v
    return pts

def Translate(pts, v):
    if random.random() > 0.5:
        v = -v
    v = v * 0.01
    translation = np.random.uniform(-v, v)
    pts[:, 0:3] += translation
    return pts

def Translate(pts, v):
    if random.random() > 0.5:
        v = -v
    v = v * 0.01
    translation = np.random.uniform(-v, v)
    pts[:, 0:3] += translation
    return pts

def Identity(pts, v):
    return pts

def RandomRotationMatrix(v):
    rand_choice = np.random.random()
    if rand_choice < 0.33:
        rotation_matrix = np.array([[np.cos(v), 0, np.sin(v)],
                                    [0, 1, 0],
                                    [-np.sin(v), 0, np.cos(v)]])
    elif 0.33 <= rand_choice < 0.66:
        rotation_matrix = np.array([[np.cos(v), -np.sin(v), 0 ],
                                    [np.sin(v),  np.cos(v), 0 ],
                                    [0, 0, 1]])
    else:
        rotation_matrix =  np.array([[1, 0, 0],
                                     [0, np.cos(v), -np.sin(v)],
                                     [0, np.sin(v),  np.cos(v)]])
    return rotation_matrix
                                     
def RandomRotate(pc, v):
    if random.random() > 0.5:
        v = -v
    v = v * 6
    normals = pc.shape[1] > 3
    rotation_matrix = torch.from_numpy(RandomRotationMatrix(v)).float()            
    cur_pc = torch.from_numpy(pc)
    if not normals:
        cur_pc = cur_pc @ rotation_matrix
    else:
        pc_xyz = cur_pc[:, 0:3]
        pc_normals = cur_pc[:, 3:]
        cur_pc[:, 0:3] = pc_xyz @ rotation_matrix
        cur_pc[:, 3:] = pc_normals @ rotation_matrix        
    pc = cur_pc.numpy()
    
    return pc
    
def SampleDrop(pc, v):
    assert 0 < v < 0.8
    bsize = pc.size()[0]
    for i in range(bsize):
        dropout_ratio = v
        drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            cur_pc = pc[i, :, :]
            cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
            pc[i, :, :] = cur_pc

    return pc
def Jitter(pc,v):
    assert 1 <= v <= 10
    std_ = 0.01 * v
    clip = 0.05 * v
                                   
    for i in range(pc.size()[0]):
        jittered_data = pc.new(pc.size(1), 3).normal_(
            mean=0.0, std=std_
        ).clamp_(-clip, clip)
        pc[i, :, 0:3] += jittered_data

    return pc                                    
                                     
def augment_list():
    l = [
        Identity,
        ShiftX,
        ShiftY,
        ShiftZ,
        Scale,
        SheerX,
        SheerY,
        SheerZ,
        Translate,
        RandomRotate,
        # SampleDrop,
        # Jitter,
    ]

    return l


class RandAugment3D:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.sample(self.augment_list, k=self.n)
        for op in ops:
            # val = (float(self.m) / 30) * float(maxval - minval) + minval
            val = self.m
            img = op(img, val)

        return img
