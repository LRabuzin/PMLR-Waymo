import os
import tensorflow.compat.v1 as tf
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import Dataset

tf.enable_eager_execution()


class WaymoSegmentationDataset(Dataset):

    def __init__(self, root_dir, mode='training', quantization_size = 0.2, transform=None, device = 'cpu'):
        self.root_dir = root_dir
        self.mode = mode
        self.quantization_size = quantization_size
        self.transform = transform
        self.device = device
        self.input_files = [os.path.join(root_dir, mode, f) for f in os.listdir(os.path.join(root_dir, mode))]
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):

        curr_file = self.input_files[idx]
        
        npzfile = np.load(curr_file)

        coordinates = npzfile['coordinates']
        features = npzfile['features']
        labels = npzfile['labels']

        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=coordinates,
            features=features,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=0)
        
        if self.transform:
            discrete_coords, unique_feats, unique_labels = self.transform(discrete_coords, unique_feats, unique_labels)

        return discrete_coords, unique_feats, unique_labels
        # return discrete_coords, torch.tensor(np.array(unique_feats)), torch.tensor(np.array(unique_labels)).type(torch.LongTensor)

if __name__=="__main__":
    dataset = WaymoSegmentationDataset(root_dir='/cluster/scratch/lrabuzin/waymo_frames')
    print(dataset[0])
