import sys

sys.path.append('/cluster/home/lrabuzin/PMLR-Waymo/dataset')
sys.path.append('/cluster/home/lrabuzin/PMLR-Waymo/models')

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
import segmentation_dataset
import minkunet

if __name__ == "__main__":
    wandb.init(project="segmentation_test", entity="lrabuzin")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Device: {device}')
    train_dataset = segmentation_dataset.WaymoSegmentationDataset(root_dir='/cluster/scratch/lrabuzin/waymo_frames', device=device)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=5,
        collate_fn=ME.utils.SparseCollation(),
        num_workers=0,
        shuffle = True)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    net = minkunet.MinkUNet14A(in_channels=3, out_channels=23, D=3)
    net = net.to(device)

    optimizer = optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9)
    
    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 5,
        "momentum" : 0.9
    }

    wandb.watch(net)

    # print('Instantiated everything')
    for epoch in range(10):
        # print(f"Epoch {epoch+1}")
        train_iter = iter(train_dataloader)
        accum_loss = 0
        accum_iter = 0
        net.train()
        for i, data in enumerate(train_iter):
            # print(f"iter {i}")
            coords, feats, labels = data
            # print(f"got data")
            out = net(ME.SparseTensor(feats, coords, device = device))
            # print(f"forward pass")
            optimizer.zero_grad()
            loss = criterion(out.F.squeeze(), labels.long().to(device))
            loss.backward()
            # print(f"backward pass")
            optimizer.step()

            accum_loss += loss.item()
            accum_iter += 1

            print(f'Epoch:{epoch}, Iter:{i}, Loss:{accum_loss/accum_iter}')
