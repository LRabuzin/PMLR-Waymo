import sys

sys.path.append('/cluster/home/lrabuzin/PMLR-Waymo/dataset')
sys.path.append('/cluster/home/lrabuzin/PMLR-Waymo/models')

import wandb
import eval_utils
import torch
import torch.nn as nn
import torch.optim as optim
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
import segmentation_dataset
import minkunet



def validate_model(model, valid_dl, loss_func, no_classes=22, device='cpu'):
    "Compute performance of the model on the validation dataset"
    model.eval()

    val_loss = 0.
    total_out = []
    total_labels = []

    with torch.inference_mode():
        valid_iter = iter(valid_dl)
        for i, data in enumerate(valid_iter):
            coords, feats, labels = data
            out = model(ME.SparseTensor(feats, coords, device = device))

            val_loss += loss_func(out.F.squeeze(), labels.long().to(device)).item()

            total_out.append(out.to(torch.device('cpu')))
            total_labels.append(labels.to(torch.device('cpu')))


    total_out = torch.cat(total_out)
    total_labels = torch.cat(total_labels)

    iou_scores = [eval_utils.iou_single_class(total_out, total_labels, class_id, no_classes) for class_id in range(no_classes)]
    iou_dict = dict(enumerate(iou_scores))

    return val_loss / len(valid_dl), iou_dict

if __name__ == "__main__":
    wandb.init(project="pmlr-waymo", entity="lrabuzin")
    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 5,
        "momentum" : 0.9
    }
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = segmentation_dataset.WaymoSegmentationDataset(root_dir='/cluster/scratch/lrabuzin/waymo_frames', device=device)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=ME.utils.SparseCollation(),
        num_workers=0,
        shuffle = True)

    valid_dataset = segmentation_dataset.WaymoSegmentationDataset(root_dir='/cluster/scratch/lrabuzin/waymo_frames', mode = 'validation', device=device)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        collate_fn=ME.utils.SparseCollation(),
        num_workers=0)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    net = minkunet.MinkUNet14A(in_channels=3, out_channels=23, D=3)
    net = net.to(device)
    optimizer = optim.SGD(
        net.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"])

    wandb.watch(net)

    for epoch in range(config["epochs"]):
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
        
        val_loss, iou_dict = validate_model(net, valid_dataloader, nn.CrossEntropyLoss(reduction='mean'), device=device)

        wandb.log({"validation loss": val_loss, "IOUs": iou_dict})
        print(f'Validation loss: {val_loss}')
        print(f'IOUs per class: {iou_dict}')
