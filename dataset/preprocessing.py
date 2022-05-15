import os
import utils
import tensorflow.compat.v1 as tf
import numpy as np

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset

def split_per_frame(root_dir, mode, output_dir, discard_empty = True):
    input_files = [os.path.join(root_dir, mode, f) for f in os.listdir(os.path.join(root_dir, mode)) if os.path.isfile(os.path.join(root_dir, mode, f))]
    i = 0

    for raw_scene in input_files:
        scene = tf.data.TFRecordDataset(raw_scene, compression_type='')

        for data in scene:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            if discard_empty and not frame.lasers[0].ri_return1.segmentation_label_compressed:
                continue

            coordinates, features, labels = utils.extract_learnables(frame)

            np.savez(os.path.join(output_dir, f'{mode}_frame_{i:07d}.npz'), coordinates = coordinates, features = features, labels = labels)
            i += 1
    return



if __name__ == '__main__':
    split_per_frame('/cluster/scratch/lrabuzin/waymo_data_updated', 'training', '/cluster/scratch/lrabuzin/waymo_frames/training')
    split_per_frame('/cluster/scratch/lrabuzin/waymo_data_updated', 'validation', '/cluster/scratch/lrabuzin/waymo_frames/validation')
    split_per_frame('/cluster/scratch/lrabuzin/waymo_data_updated', 'testing', '/cluster/scratch/lrabuzin/waymo_frames/testing', discard_empty=False)