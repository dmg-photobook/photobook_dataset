import os
import json
import torch
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np

# Loads a PhotoBook segment data set object from file
class SegmentDataset(Dataset):
    def __init__(self, data_dir, segment_file, vectors_file, split='train'):

        self.data_dir = data_dir
        self.split = split
        self.segment_file = self.split + '_' + segment_file

        # Load a PhotoBook dialogue segment data set
        with open(os.path.join(self.data_dir, self.segment_file), 'r') as file:
            self.temp_dialogue_segments = json.load(file)

        self.dialogue_segments = []
        for d in self.temp_dialogue_segments:

            if d['segment'] == []:
                d['segment'] = [0] #pad empty segment
                d['length'] = 1


            self.dialogue_segments.append(d)

        # Load pre-defined image features
        with open(os.path.join(data_dir, vectors_file), 'r') as file:
            self.image_features = json.load(file)

    # Returns the length of the data set
    def __len__(self):
        return len(self.dialogue_segments)

    # Returns a PhotoBook Segment object at the given index
    def __getitem__(self, index):
        return self.dialogue_segments[index]

    @staticmethod
    def get_collate_fn(device):

        def collate_fn(data):

            #print('collate',data)
            max_src_length = max(d['length'] for d in data)
            max_target_images = max(len(d['targets']) for d in data)
            max_num_images = max([len(d['image_set']) for d in data])

            #print(max_src_length, max_target_images, max_num_images)

            batch = defaultdict(list)

            for sample in data:
                for key in data[0].keys():

                    if key == 'segment':
                        padded = sample['segment'] \
                            + [0] * (max_src_length-sample['length'])
                        #print('seg', padded)

                    elif key == 'image_set':

                        padded = [int(img) for img in sample['image_set']]

                        padded = padded \
                            + [0] * (max_num_images-len(sample['image_set']))
                        #print('img', padded)

                    elif key == 'targets':

                        #print(sample['targets'])
                        padded = np.zeros(max_num_images)
                        padded[sample['targets']] = 1

                        #print('tar', padded)

                    else:
                        #length of segment in number of words
                        padded = sample[key]

                    batch[key].append(padded)

            for key in batch.keys():
                #print(key, batch[key])
                batch[key] = torch.Tensor(batch[key]).long().to(device)

            return batch

        return collate_fn
