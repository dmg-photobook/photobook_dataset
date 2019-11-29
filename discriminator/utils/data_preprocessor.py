import os
import json
import pickle
import argparse
from collections import defaultdict

import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from utils.Vocab import Vocab
from utils.SegmentBuilder import SegmentBuilder


def create_segment_datasets(dialogue_segments, vocab, chain_id=0, method="word2index", tokenization="word_tokenize", speaker_lables=True, lowercase=True, splitting=True):
    segment_builder = SegmentBuilder()

    segment_dataset = []
    chain_dataset = []
    for game_id, game_segments in dialogue_segments:
        if game_id in ["3Y9N9SS8LZ9I159GL3ACP8R77X5D3M3VW04L3ZLU48F9LBWSICQVJ3UAIXX8", "3PB5A5BD0W43E8KUP5EA8A6LTWT7GX38F5OAUN5OAHE4F59BWSTAIM8W67HW"]:
            continue
        target_segments = defaultdict(lambda: [])
        for round_segments, image_set in game_segments:
            for segment, targets in round_segments:
                segment_dataset.append(segment_builder.build(segment, targets, image_set, vocab, method, tokenization, speaker_lables, lowercase, splitting))
                for target in targets:
                    target_id = str(target.split('_')[-1].split('.')[0].lstrip('0'))
                    target_segments[target_id].append(len(segment_dataset)-1)

        for target, segments in target_segments.items():
            chain_dataset.append({"game_id": game_id, "chain_id": chain_id, "target": target, "segments": segments, "lengths": [segment_dataset[segment_id]["length"] for segment_id in segments]})
            chain_id += 1

    return segment_dataset, chain_dataset, chain_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="../data")
    parser.add_argument("-input_data", type=str, default="sections.pickle")
    parser.add_argument("-vocab", type=str, default="vocab.csv")
    parser.add_argument("-encoding", type=str, default="word2index")
    parser.add_argument("-tokenization", type=str, default="word_tokenize")
    parser.add_argument("-speaker_lables", type=bool, default=True)
    parser.add_argument("-lowercase", type=bool, default=True)
    parser.add_argument("-splitting", type=bool, default=True)
    parser.add_argument("-min_occ", type=int, default=3)
    args = parser.parse_args()

    vocab = Vocab.create(args.data_path, "train_" + args.input_data, args.vocab, args.tokenization, args.lowercase, args.splitting, args.min_occ)

    print("Creating new PhotoBook segment dataset...")
    chain_id = 0
    for set_name in ['dev', 'val', 'test', 'train']:
        with open(os.path.join(args.data_path, set_name + "_" + args.input_data), 'rb') as f:
            dialogue_segments = pickle.load(f)
        segment_dataset, chain_dataset, chain_id = create_segment_datasets(dialogue_segments, vocab, chain_id, args.encoding, args.tokenization, args.speaker_lables, args.lowercase, args.splitting)

        with open(os.path.join(args.data_path, set_name + "_segments.json"), 'w') as f:
            json.dump(segment_dataset, f)

        with open(os.path.join(args.data_path, set_name + "_chains.json"), 'w') as f:
            json.dump(chain_dataset, f)


    print("Done.")
