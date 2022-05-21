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


def validate_model(model, valid_dl, loss_func, no_classes=23, device='cpu'):
    "Compute performance of the model on the validation dataset"
    model.eval()

    val_loss = 0.
    total_intersection = [0]*no_classes
    total_union = [0]*no_classes

    with torch.inference_mode():
        valid_iter = iter(valid_dl)
        for i, data in enumerate(valid_iter):
            coords, feats, labels = data
            out = model(ME.SparseTensor(feats, coords, device = device))
            out_squeezed = out.F.squeeze()
            val_loss += loss_func(out_squeezed, labels.long().to(device)).item()

            for i in range(no_classes):
                intersection, union = eval_utils.iou_separate(out_squeezed.to(torch.device('cpu')), labels.to(torch.device('cpu')), i)
                total_intersection[i] += intersection
                total_union[i] += union


    iou_scores = [i/u for i,u in zip(total_intersection, total_union)]
    iou_dict = dict(enumerate(iou_scores))

    return val_loss / len(valid_dl), iou_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay',type=float, default=1e-4)
    parser.add_argument('--root_dir', default='/cluster/scratch/lrabuzin/waymo_frames')
    parser.add_argument('--checkpoint_location', default='/cluster/home/lrabuzin/PMLR-Waymo')

    hyperparams = parser.parse_args()

    wandb.init(project="pmlr-waymo", config=vars(hyperparams))

    config = wandb.config

    start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = segmentation_dataset.WaymoSegmentationDataset(root_dir=config["root_dir"], device=device)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=ME.utils.SparseCollation(),
        num_workers=0,
        shuffle = True)

    valid_dataset = segmentation_dataset.WaymoSegmentationDataset(root_dir=config["root_dir"], mode = 'validation', device=device)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        collate_fn=ME.utils.SparseCollation(),
        num_workers=0)

    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
    net = minkunet.MinkUNet14A(in_channels=3, out_channels=23, D=3)
    net = net.to(device)
    optimizer = optim.SGD(
        net.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"])

    wandb.watch(net)

    for epoch in range(config["max_epochs"]):
        train_iter = iter(train_dataloader)
        accum_loss = 0
        accum_iter = 0
        net.train()
        for i, data in enumerate(train_iter):
            coords, feats, labels = data
            out = net(ME.SparseTensor(feats, coords, device = device))
            optimizer.zero_grad()
            loss = criterion(out.F.squeeze(), labels.long().to(device))
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

            accum_loss += loss.item()
            accum_iter += 1

            print(f'Epoch:{epoch}, Iter:{i}, Loss:{accum_loss/accum_iter}')
        
        if (epoch+1)%2 == 0 or epoch == config["max_epochs"]-1:
            val_loss, iou_dict = validate_model(net, valid_dataloader, nn.CrossEntropyLoss(reduction='mean', ignore_index=0), device=device)
            
            wandb.log({"validation loss": val_loss, "IOUs": iou_dict})
            print(f'Validation loss: {val_loss}')
            print(f'IOUs per class: {iou_dict}')

            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, os.path.join(config["checkpoint_location"], f"checkpoint_{start_time}_epoch_{epoch}.pth"))


# bsub -n 12 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python scripts/training.py --max_epochs 10 --weight_decay 0.001 --batch_size 4 --root_dir /cluster/scratch/mertugrul/waymo_frames --checkpoint_location /cluster/home/mertugrul/PMLR-Waymo