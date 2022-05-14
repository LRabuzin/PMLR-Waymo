import torch
import torch.nn as nn
import torch.optim as optim
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from dataset import segmentation_dataset
from models import minkunet

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = segmentation_dataset.WaymoSegmentationDataset(root_dir='/cluster/scratch/lrabuzin/waymo_frames', device=device)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=5,
        collate_fn=ME.utils.SparseCollation(),
        num_workers=1,
        shuffle = True)

    criterion = nn.CrossEntropyLoss(reduce='sum')
    net = minkunet.MinkUNet14A(in_channels=3, out_channels=23, D=3)

    optimizer = optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9)


    for epoch in range(10):
        train_iter = iter(train_dataloader)
        accum_loss = 0
        accum_iter = 0
        for i, data in enumerate(train_iter):
            coords, feats, labels = data
            out = net(ME.SparseTensor(feats, coords))
            optimizer.zero_grad()
            loss = criterion(out.F.squeeze(), labels.long())
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_iter += 1

            print(f'Epoch:{epoch}, Iter:{i}, Loss:{accum_loss/accum_iter}')
