import os
import argparse

import sys
from collections import Counter
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from utils.SegmentDataset import SegmentDataset
from utils.ChainDataset import ChainDataset
from utils.Vocab import Vocab


# Tests the SegmentDataset and ChainDataset classes
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="../data")
    parser.add_argument("-segment_file", type=str, default="segments.json")
    parser.add_argument("-chains_file", type=str, default="val_chains.json")
    parser.add_argument("-vocab_file", type=str, default="vocab.csv")
    parser.add_argument("-vectors_file", type=str, default="vectors.json")
    parser.add_argument("-split", type=str, default="val")
    args = parser.parse_args()

    print("Loading the vocab...")
    vocab = Vocab(os.path.join(args.data_path, args.vocab_file), 3)

    print("Testing the SegmentDataset class initialization...")

    segment_val_set = SegmentDataset(
        data_dir=args.data_path,
        segment_file=args.segment_file,
        vectors_file=args.vectors_file,
        split=args.split
    )

    print("Testing the SegmentDataset class item getter...")
    print("Dataset contains {} segment samples".format(len(segment_val_set)))
    sample_id = 2
    sample = segment_val_set[sample_id]
    print("Segment {}:".format(sample_id))
    print("Image set: {}".format(sample["image_set"]))
    print("Target image index(es): {}".format(sample["targets"]))
    print("Target image Features: {}".format([segment_val_set.image_features[sample["image_set"][int(target)]] for target in sample["targets"]]))
    print("Encoded segment: {}".format(sample["segment"]))
    print("Decoded segment dialogue: {}".format(vocab.decode(sample["segment"])))
    print("Segment length: ", sample["length"])
    print("\nDone.")

    print("Testing the ChainDataset class initialization...")

    chain_val_set = ChainDataset(
        data_dir=args.data_path,
        segment_file=args.segment_file,
        chain_file=args.chains_file,
        vectors_file=args.vectors_file,
        split=args.split
    )

    print("Dataset contains {} cains.".format(len(chain_val_set.chains)))

    sample_id = 2
    sample = chain_val_set.chains[sample_id]

    print("Chain {}:".format(sample_id))
    print("Source Game ID: {}".format(sample["game_id"]))
    print("Target image index: {}".format(sample["target"]))
    print("Chain length: {}".format(len(sample["segments"])))
    print("Segment IDs: {}".format(sample["segments"]))
    print("Segment lengths: ", sample["lengths"])
    print("First segment encoding: {}".format(chain_val_set.segments[sample["segments"][0]]["segment"]))
    print("First segment decoded dialogue: {}".format(vocab.decode(chain_val_set.segments[sample["segments"][0]]["segment"])))

    print("Reference chain and segments' associated image sets:")
    for segment in sample["segments"]:
        print(vocab.decode(chain_val_set.segments[segment]["segment"]))
        print(chain_val_set.segments[segment]["image_set"])
    print("\nDone.")

    print("Testing the ChainDataset class item getter...")

    print("Dataset contains {} segment samples for batch processing".format(len(chain_val_set)))

    sample_id = 2
    sample = chain_val_set[sample_id]

    print("Segment {}:".format(sample_id))
    print("Source Game ID: {}".format(sample["game_id"]))
    print("Soruce Chain ID: {}".format(sample["chain_id"]))
    print("Segment ID: {}".format(sample["segment_id"]))
    print("Target image index: {}".format(sample["reference_image"]))
    print("Target")
    print("Encoded segment: {}".format(sample["source_language"]))
    print("Decoded segment dialogue: {}".format(vocab.decode(sample["source_language"])))
    print("Segment length: ", sample["source_length"])
    print("Source")
    print("Encoded segment: {}".format(sample["target_language"]))
    print("Decoded segment dialogue: {}".format(vocab.decode(sample["target_language"])))
    print("Segment length: ", sample["target_length"])
