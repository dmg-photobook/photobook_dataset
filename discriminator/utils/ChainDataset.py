import os
import json
import torch
from collections import defaultdict

from torch.utils.data import Dataset


# Loads a PhotoBook segment chain data set from file
class ChainDataset(Dataset):
    def __init__(self, split, data_dir, chain_file, segment_file,
                 vectors_file, input='gold'):

        self.data_dir = data_dir
        self.split = split
        self.segment_file = self.split + '_' + segment_file

        # Load a PhotoBook segment chain data set
        with open(os.path.join(self.data_dir, chain_file), 'r') as file:
            self.chains = json.load(file)

        # Load an underlying PhotoBook dialogue segment data set
        with open(os.path.join(self.data_dir, self.segment_file), 'r') as file:
            self.segments = json.load(file)

        # Load pre-defined image features
        with open(os.path.join(data_dir, vectors_file), 'r') as file:
            self.image_features = json.load(file)

        self.data = dict()
        for chain_id, chain in enumerate(self.chains):

            if len(chain['segments']) <= 1:
                continue

            # load first source segment
            source_game_id = chain['game_id']
            source = self.load_segment(chain['segments'][0])

            reference_image = source['target_image']

            for segment_id in chain['segments'][1:]:

                source_language = source['language']
                source_length = source['length']

                target = self.load_segment(segment_id)
                target_language = target['language']
                target_length = target['length'] + 1  # +1 for sos/eos
                images = target['images']
                num_images = len(images)

                if len(target_language) == 0:
                    continue

                self.data[len(self.data)] = {
                    'game_id' : source_game_id,
                    'chain_id': chain_id,
                    'segment_id': segment_id,
                    'reference_image': reference_image,
                    'source_language': source_language,
                    'source_length': source_length,
                    'target_language': target_language,
                    'target_length': target_length,
                    'images': images,
                    'num_images': num_images,
                    'reference_image_features': self.image_features[reference_image]
                }

                source = self.merge_segments_language(source, target)

                if input == 'generation':
                    break

    def load_segment(self, index):
        segment = self.segments[index]
        language = segment['segment']
        length = segment['length']
        images = segment['image_set']
        target_image_id = segment['targets'][0]
        target_image = images[target_image_id]

        return {
            'language': language,
            'length': length,
            'images': images,
            'target_image_id': target_image_id,
            'target_image': target_image,
        }

    def merge_segments_language(self, old, new):
        # QUESTION: should we put some placeholder between segments?
        return {
            'language': old['language'] + new['language'],
            'length': old['length'] + new['length']
            }

    def get_segments(self, chain):
        return [self.dialogue_segments[segment_id] for segment_id in chain["chain"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def get_collate_fn(device, sos_token, eos_token):

        def collate_fn(data):

            max_src_length = max([d['source_length'] for d in data])
            max_tgt_length = max([d['target_length'] for d in data])
            max_num_images = max([d['num_images'] for d in data])

            batch = defaultdict(list)

            for sample in data:
                for key in data[0].keys():

                    if key == 'source_language':
                        padded = sample['source_language'] \
                            + [0] * (max_src_length-sample['source_length'])
                    elif key == 'target_language':

                        # input for teacher forcing
                        target_language_input = \
                            [sos_token] + sample['target_language'] \
                            + [0] * (max_tgt_length-sample['target_length'])

                        batch['target_language_input'].append(
                            target_language_input)

                        padded = sample['target_language'] + [eos_token] \
                            + [0] * (max_tgt_length-sample['target_length'])

                    elif key == 'images':
                        padded = sample['images'] \
                            + [0] * (max_num_images-sample['num_images'])
                    else:
                        padded = sample[key]

                    batch[key].append(padded)

            for key in batch.keys():

                if key in ['reference_image', 'images']:
                    continue

                try:
                    batch[key] = torch.Tensor(batch[key]).to(device)
                except:
                    print(key)
                    raise
                if key in ['reference_image_features']:
                    pass
                else:
                    batch[key] = batch[key].long()

            return batch

        return collate_fn
