import sys
import os

parent_dir = os.path.dirname(os.path.dirname(__file__))

sys.path.append(os.path.join(parent_dir, 'dataset'))
sys.path.append(os.path.join(parent_dir, 'models'))

import wandb
import eval_utils
import torch
import torch.nn as nn
import torch.optim as optim
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
import segmentation_dataset
import minkunet
import argparse
from datetime import datetime


if __name__ == "__main__":

    net = minkunet.MinkUNet14A(in_channels=3, out_channels=23, D=3)

    optimizer = optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4)


    torch.save({
                'epoch': 0,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': 0
            }, os.path.join('/cluster/home/mertugrul/PMLR-Waymo', "checkpoint_0_epoch_0.pth"))


