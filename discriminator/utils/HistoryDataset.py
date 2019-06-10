import os
import json
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


# Loads PhotoBook segments with relevant reference chains from file
class HistoryDataset(Dataset):
    def __init__(self, split, data_dir, chain_file, segment_file,
                 vectors_file):

        self.data_dir = data_dir
        self.split = split

        # Load a PhotoBook segment chain data set
        with open(os.path.join(self.data_dir, chain_file), 'r') as file:
            self.chains = json.load(file)

        # Load an underlying PhotoBook dialogue segment data set
        with open(os.path.join(self.data_dir, segment_file), 'r') as file:
            self.segments = json.load(file)

        # Load pre-defined image features
        with open(os.path.join(data_dir, vectors_file), 'r') as file:
            self.image_features = json.load(file)

        self.data = dict()

        self.img2chain = defaultdict(dict)

        for chain in self.chains:

            self.img2chain[chain['target']][chain['game_id']] = chain['segments']


        segments_added = []

        print('processing',self.split)

	# every segment in every chain, along with the relevant history
        for chain in self.chains:

            if len(chain['segments']) < 1:
                continue

            chain_segments = chain['segments']
            im_target = chain['target']
            game_id = chain['game_id']

            for s in range(len(chain_segments)):

                prev_chains = defaultdict(list)

                segment_id = chain_segments[s]

                if segment_id not in segments_added:

                    segments_added.append(segment_id)
                    cur_segment_obj = self.segments[segment_id]
                    cur_segment_text = cur_segment_obj['segment']

                    lengths = cur_segment_obj['length']

                    if cur_segment_text == []:
                        cur_segment_text = [0] #there is only one instance like this
                        lengths = 1

                    images = cur_segment_obj['image_set']
                    targets = cur_segment_obj['targets']

                    prev_lengths = defaultdict(int)

                    for im in images:

                        if game_id in self.img2chain[im]:
                            temp_chain = self.img2chain[im][game_id]

                            hist_segments = [t for t in temp_chain if t < segment_id]

                            if len(hist_segments) > 0:
                                for h in hist_segments:
                                    prev_chains[im].extend(self.segments[h]['segment']) #combine all prev histories

                            else:
                                #no prev reference to that image
                                prev_chains[im] = []

                        else:
                            # image is in the game but never referred to
                            prev_chains[im] = []


                        prev_lengths[im] = len(prev_chains[im])

                    self.data[len(self.data)] = {'segment': cur_segment_text,
                                                 'image_set': images,
                                                 'targets':targets,
                                                 'length': lengths,
                                                 'prev_histories': prev_chains,
                                                 'prev_history_lengths': prev_lengths
                                                 }

        name_sgj = 'segment_ids_' + self.split + '.json'
        with open(name_sgj, 'w') as f:
            json.dump(segments_added, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def get_collate_fn(device):

        def collate_fn(data):

            max_src_length = max(d['length'] for d in data)
            max_target_images = max(len(d['targets']) for d in data)
            max_num_images = max([len(d['image_set']) for d in data])
            max_prev_len = max([d['prev_history_lengths'][i] for d in data for i in d['image_set']])

            batch = defaultdict(list)

            for sample in data:

                for key in data[0].keys():

                    if key == 'segment':
                        padded = sample['segment'] \
                                 + [0] * (max_src_length - sample['length'])
                        # print('seg', padded)

                    elif key == 'image_set':

                        padded = [int(img) for img in sample['image_set']]

                        padded = padded \
                                 + [0] * (max_num_images - len(sample['image_set']))
                        # print('img', padded)

                    elif key == 'targets':

                        # print(sample['targets'])
                        padded = np.zeros(max_num_images)
                        padded[sample['targets']] = 1

                        # print('tar', padded)

                    elif key == 'prev_histories':

                        histories_per_img = []
                        len_per_img = []

                        for k in range(len(sample['image_set'])):
                            #keep the order of imgs
                            img_id = sample['image_set'][k]

                            len_per_img.append(len(sample[key][img_id]))
                            history = sample[key][img_id]

                            histories_per_img.append(history)

                        padded = histories_per_img

                    else:
                        # length of segment in number of words
                        padded = sample[key]

                    batch[key].append(padded)

            for key in batch.keys():
                if key != 'prev_histories' and key != 'prev_history_lengths':
                    batch[key] = torch.Tensor(batch[key]).long().to(device)


            return batch

        return collate_fn

