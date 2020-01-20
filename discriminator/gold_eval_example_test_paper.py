# Runs the models on the test set and writes correct-incorrect predictions into file for qualitative analysis
# indices for the history model require an additional step (as segments come in the order they occur in a given dialogue)

import json
import numpy as np

import torch
from torch import nn
from models.model_nohistory import DiscriminatoryModelBlind
from models.model_history import HistoryModelBlind

import train_nohistory
import train_history

import os

from utils.SegmentDataset import SegmentDataset
from utils.HistoryDataset import HistoryDataset

from utils.Vocab import Vocab

def get_f1(prec, recall, beta):

    if prec == 0 and recall == 0:
        return 'nan'
    else:
        return (1 + np.power(beta,2)) *((prec*recall)/(np.power(beta,2)*prec+recall))

def load_model(file, model, device):
    len_vocab = 3424
    embedding_dim = 512
    hidden_dim = 512
    img_dim = 2048
    model = model(len_vocab, embedding_dim, hidden_dim, img_dim).to(device)
    checkpoint = torch.load(file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    args_train = checkpoint['args']
    return model,epoch,loss,accuracy, args_train

def mask_attn(scores, actual_num_images, max_num_images, device):
    masks = []

    for n in range(len(actual_num_images)):
        mask = [1] * actual_num_images[n] + [0] * (max_num_images - actual_num_images[n])

        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).byte().to(device)

    scores.masked_fill_(1 - masks, float(-1e30))
    return scores

model_files = ['model_blind_accs_2019-02-17-21-18-7.pkl',
               'model_history_blind_accs_2019-02-20-14-22-23.pkl']

model_name_abbrv = {'model_history_blind_accs_2019-02-20-14-22-23.pkl':'History',
               'model_blind_accs_2019-02-17-21-18-7.pkl':'No history'}

for model_file in model_files:

    print(model_file)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if model_name_abbrv[model_file]=='No history':
        model,epoch,loss, accuracy, args_train = load_model(model_file, DiscriminatoryModelBlind, device)

    elif model_name_abbrv[model_file] == 'History':
        model, epoch, loss, accuracy, args_train = load_model(model_file, HistoryModelBlind, device)

    print(accuracy)

    args = args_train
    args.split = 'test'

    print("Loading the vocab...")
    vocab = Vocab(os.path.join(args.data_path, args.vocab_file), 3)
    print('vocab len', len(vocab))

    with open('test_seg2ranks.json', 'r') as file:
        seg2ranks = json.load(file)


    with open('test_seg_ids.json','r') as file:
        id_list = json.load(file)  #ordered IDs for history #segments come in a different order

    embedding_dim = 512
    hidden_dim = 512
    img_dim = 2048
    threshold = 0.5

    shuffle = args.shuffle
    normalize = args.normalize
    mask = args.mask
    weighting = args.weighting
    weight = args.weight
    breaking = args.breaking

    weights = torch.Tensor([weight]).to(device)

    learning_rate = args.learning_rate
    if weighting:
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights, reduction='sum')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

    batch_size = 1 #NO PADDING

    load_params = {'batch_size':batch_size,
            'shuffle': False,
                   'collate_fn': SegmentDataset.get_collate_fn(device)}

    load_params_hist = {'batch_size': batch_size,
                   'shuffle': False,
                   'collate_fn': HistoryDataset.get_collate_fn(device)}


    testset = SegmentDataset(
        data_dir=args.data_path,
        segment_file=args.segment_file,
        vectors_file=args.vectors_file,
        split='test'
    )

    load_params_test = {'batch_size': batch_size,
                        'shuffle': False, 'collate_fn': SegmentDataset.get_collate_fn(device)}

    test_loader = torch.utils.data.DataLoader(testset, **load_params_test)

    testset_hist = HistoryDataset(
        data_dir=args.data_path,
        segment_file='test_' + args.segment_file,
        vectors_file=args.vectors_file,
        chain_file='test_' + args.chains_file,
        split='test'
    )

    load_params_test_hist = {'batch_size': batch_size,
                        'shuffle': False, 'collate_fn': HistoryDataset.get_collate_fn(device)}


    test_hist_loader = torch.utils.data.DataLoader(testset_hist, **load_params_test_hist)

    with torch.no_grad():
        model.eval()
        print('\nGold Eval')

        # methods are in the training scripts

        if model_name_abbrv[model_file] == 'No history':
            train_nohistory.gold_evaluate_write(test_loader, testset, breaking, normalize, mask, img_dim, model, seg2ranks, device, criterion, threshold, weight)

        elif model_name_abbrv[model_file] == 'History':
            train_history.gold_evaluate_write(test_hist_loader, testset_hist, breaking, normalize, mask, img_dim, model, seg2ranks, id_list, device, criterion, threshold, weight)


