import json
from collections import defaultdict
from utils.Vocab import Vocab

with open('data/test_segments.json', 'r') as file:
    test_sg = json.load(file)

with open('data/test_chains.json', 'r') as file:
    test = json.load(file)

vocab = Vocab('data/vocab.csv', 3)

# given an img, provides the chains for which it was the target

target2chains = defaultdict(list)

for ch in test:
    target_id = ch['target']
    segment_list = ch['segments']

    target2chains[target_id].append(segment_list)

id_list = []

# segments ids, in the order in which they were encountered in the chains in the whole dataset

for c in test:
    segments = c['segments']

    for s in segments:
        if s not in id_list:
            id_list.append(s)

with open('test_seg_ids.json', 'w') as file:
    json.dump(id_list, file)

seg2ranks = dict()

for c in test:

    segments = c['segments']
    for s in range(len(segments)):
        seg_id = segments[s]
        rank = s

        if seg_id in seg2ranks:
            seg2ranks[seg_id].append(rank)
        else:
            seg2ranks[seg_id] = [rank]

# all the ranks a segment is positioned
# in different chains (hence, there could be multiples of the same position)
with open('test_seg2ranks.json', 'w') as file:
    json.dump(seg2ranks, file)

# also for the val set

with open('data/val_segments.json', 'r') as file:
    val_sg = json.load(file)

with open('data/val_chains.json', 'r') as file:
    val = json.load(file)

vocab = Vocab('data/vocab.csv', 3)

target2chains = defaultdict(list)

for ch in val:
    target_id = ch['target']
    segment_list = ch['segments']

    target2chains[target_id].append(segment_list)

id_list = []

for c in val:
    segments = c['segments']

    for s in segments:
        if s not in id_list:
            id_list.append(s)

with open('val_seg_ids.json', 'w') as file:
    json.dump(id_list, file)

seg2ranks = dict()

for c in val:

    segments = c['segments']
    for s in range(len(segments)):
        seg_id = segments[s]
        rank = s

        if seg_id in seg2ranks:
            seg2ranks[seg_id].append(rank)
        else:
            seg2ranks[seg_id] = [rank]

with open('val_seg2ranks.json', 'w') as file:
    json.dump(seg2ranks, file)

