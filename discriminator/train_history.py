import torch
import pdb
import numpy as np

from torch import nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.utils.data

from models.model_history import HistoryModelBlind
import os
import argparse

import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from utils.HistoryDataset import HistoryDataset
from utils.Vocab import Vocab

import datetime

#print(torch.__version__) #0.4.1

def get_f1(prec, recall, beta):

    if prec == 0 and recall == 0:
        return 'nan'
    else:
        return (1 + np.power(beta,2)) *((prec*recall)/(np.power(beta,2)*prec+recall))


def mask_attn(scores, actual_num_images, max_num_images, device):
    masks = []

    for n in range(len(actual_num_images)):
        mask = [1] * actual_num_images[n] + [0] * (max_num_images - actual_num_images[n])
        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).byte().to(device)
    scores.masked_fill_(1 - masks, float(-1e30))
    return scores


def save_model(model, epoch, accuracy, loss, args, metric, timestamp):

    file_name = 'model_history_blind_' + metric + '_' + timestamp + '.pkl'

    print(file_name)

    torch.save({
        'accuracy': accuracy,
        'args': args,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }, file_name)


# method to evaluate the training and validation results
# since non-targets and pads are both 0, we take some measures against that
# there are hard-coded values regarding that

def evaluate(split_data_loader, dataset, breaking, normalize, mask, img_dim,\
             model, epoch, args, timestamp, best_accuracy, best_loss, isValidation, weight, device):

    losses = []
    count = 0

    matching = 0
    matching_0 = 0
    matching_1 = 0
    normalizer = 0
    normalizer_01 = 0
    normalizer_0 = 0
    normalizer_1 = 0

    non_match_1 = 0 #actually 0 but predicted as 1
    non_match_0 = 0  # actually 1 but predicted as 0

    all_non_pad = 59453 if isValidation else 267891

    for ii, data in enumerate(split_data_loader):

        if breaking and count == 5:
            break

        count += 1

        segments_text = torch.tensor(data['segment'])

        normalizer += segments_text.shape[0]

        image_set = data['image_set']
        no_images = image_set.shape[1]

        actual_no_images = []

        for s in range(image_set.shape[0]):
            actual_no_images.append(len(image_set[s].nonzero()))

        temp_batch_size = data['segment'].shape[0]

        context_sum = torch.zeros(temp_batch_size, img_dim).to(device)

        context_separate = torch.zeros(temp_batch_size, image_set.shape[1], img_dim).to(device)

        for b in range(image_set.shape[0]):

            non_pad_item = 0.0

            for i in range(image_set[b].shape[0]):

                img_id = str(image_set[b][i].item())

                if img_id != '0':

                    img_features = torch.Tensor(dataset.image_features[img_id]).to(device)

                    context_sum[b] += img_features

                    non_pad_item += 1

                else:

                    img_features = torch.zeros(img_dim).to(device)

                context_separate[b][i] = img_features

            context_sum[b] = context_sum[b] / non_pad_item #average #not used in this model

        lengths = data['length']
        targets = data['targets'].view(temp_batch_size, no_images, 1).float()
        prev_histories = data['prev_histories']

        out = model(segments_text, prev_histories, lengths, context_separate, context_sum, normalize, device)

        if mask:
            out = mask_attn(out, actual_no_images, no_images, device)

        sig_out = torch.sigmoid(out)

        loss = criterion(out, targets)

        #predictions according to the threshold
        preds = sig_out.data
        preds[preds >= threshold] = 1
        preds[preds < threshold] = 0

        for p in range(preds.shape[0]):

            if torch.all(torch.eq(preds[p], targets[p])):
                matching += 1

        for pi in range(preds.shape[0]):
            for pj in range(preds[pi].shape[0]):
                if targets[pi][pj] == 0:
                    normalizer_0 += 1
                    if preds[pi][pj] == 0:
                        matching_0 += 1

                    else:
                        non_match_1 += 1 #target 0 but predicted as 1

                elif targets[pi][pj] == 1:
                    normalizer_1 += 1
                    if preds[pi][pj] == 1:
                        matching_1 += 1

                    else:
                        non_match_0 += 1

                normalizer_01 += 1

        losses.append(loss.item())


    #subtract count of pads
    pad_0_count = normalizer_01 - all_non_pad
    normalizer_01 = normalizer_01 - pad_0_count
    normalizer_0 = normalizer_0 - pad_0_count
    matching_0 = matching_0 - pad_0_count

    print(pad_0_count, normalizer_01,normalizer_0,matching_0)

    print('Target 0 -', matching_0, normalizer_0)
    print('Target 1 -', matching_1, normalizer_1)
    print('Normalizer -', normalizer_01)

    print()

    check_accs = (matching_0 + matching_1) / normalizer_01
    check_accs_0 = matching_0 / normalizer_0
    check_accs_1 = matching_1 / normalizer_1
    print('0-matching acc:', check_accs_0)
    print('1-matching acc:', check_accs_1)
    print('All-matching acc:', check_accs)

    print()

    print('Fully matching:', matching)
    print('Normalizer:', normalizer)  # normalizer is batch size, matching is the matching of the whole pred & target
    accs = matching / normalizer

    mean_loss = np.mean(losses)
    print('Accuracy:', accs)
    print('Mean loss:', mean_loss)

    prec_1 = matching_1 / (matching_1 + non_match_1)

    recall_1 = check_accs_1

    prec_0 = matching_0 / (matching_0 + non_match_0)
    recall_0 = check_accs_0

    beta = 1

    f1_0 = get_f1(prec_0, recall_0, beta)
    f1_1 = get_f1(prec_1, recall_1, beta)

    f1 = (f1_0 + weight * f1_1) / (1 + weight)

    print()
    print('Precision 0:', prec_0)
    print('Precision 1:', prec_1)

    print('Recall 0:', recall_0)
    print('Recall 1:', recall_1)

    print()

    print('F1:', f1)
    print('F1_0:', f1_0)
    print('F1_1:', f1_1)

    accs_to_write = f1
    if isValidation:
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_model(model, epoch, accs_to_write, mean_loss, args, 'loss', timestamp)

        if accs_to_write > best_accuracy:
            best_accuracy = accs_to_write
            save_model(model, epoch, accs_to_write, mean_loss, args, 'accs', timestamp)

        return best_accuracy, best_loss, f1, mean_loss

# used in the gold_eval_test_*.py code to get the results on the test set
# input not batched
# not saving models
# provides extra details about the results (rank-related etc.)

def gold_evaluate(split_data_loader, dataset, breaking, normalize, mask, img_dim, model, seg2ranks, id_list, device, criterion, threshold, weight):

    losses = []
    count = 0

    matching = 0
    matching_0 = 0
    matching_1 = 0
    normalizer = 0
    normalizer_01 = 0
    normalizer_0 = 0
    normalizer_1 = 0

    non_match_1 = 0  # actually 0 but predicted as 1
    non_match_0 = 0  # actually 1 but predicted as 0

    segment_rank_res = dict()

    for ii, data in enumerate(split_data_loader):

        seg_id = id_list[ii]
        segment_ranks = seg2ranks[str(seg_id)]

        for segment_rank in segment_ranks:
            if segment_rank not in segment_rank_res:
                segment_rank_res[segment_rank] = {'matching_0': 0.0, 'matching_1': 0.0, \
                                                  'non_match_0': 0.0, 'non_match_1': 0.0, \
                                                  'normalizer_01': 0.0}

        if breaking and count == 5:
            break

        count += 1

        segments_text = torch.tensor(data['segment'])

        normalizer += segments_text.shape[0]

        image_set = data['image_set']
        no_images = image_set.shape[1]

        actual_no_images = []

        for s in range(image_set.shape[0]):
            actual_no_images.append(len(image_set[s].nonzero()))

        temp_batch_size = data['segment'].shape[0]

        context_sum = torch.zeros(temp_batch_size, img_dim).to(device)

        context_separate = torch.zeros(temp_batch_size, image_set.shape[1], img_dim).to(device)

        for b in range(image_set.shape[0]):

            non_pad_item = 0.0

            for i in range(image_set[b].shape[0]):

                img_id = str(image_set[b][i].item())

                if img_id != '0':

                    img_features = torch.Tensor(dataset.image_features[img_id]).to(device)

                    context_sum[b] += img_features

                    non_pad_item += 1

                else:

                    img_features = torch.zeros(img_dim).to(device)

                context_separate[b][i] = img_features

            context_sum[b] = context_sum[b] / non_pad_item  # average #not used in this model

        lengths = data['length']
        targets = data['targets'].view(temp_batch_size, no_images, 1).float()
        prev_histories = data['prev_histories']

        out = model(segments_text, prev_histories, lengths, context_separate, context_sum, normalize, device)

        if mask:
            out = mask_attn(out, actual_no_images, no_images, device)

        sig_out = torch.sigmoid(out)

        loss = criterion(out, targets)

        preds = sig_out.data
        preds[preds >= threshold] = 1
        preds[preds < threshold] = 0

        for p in range(preds.shape[0]):

            if torch.all(torch.eq(preds[p], targets[p])):
                matching += 1

        for pi in range(preds.shape[0]):
            for pj in range(preds[pi].shape[0]):
                if targets[pi][pj] == 0:
                    normalizer_0 += 1
                    if preds[pi][pj] == 0:
                        matching_0 += 1

                        for segment_rank in segment_ranks:
                            segment_rank_res[segment_rank]['matching_0'] += 1

                    else:
                        non_match_1 += 1  # target 0 but predicted as 1
                        for segment_rank in segment_ranks:
                            segment_rank_res[segment_rank]['non_match_1'] += 1

                elif targets[pi][pj] == 1:
                    normalizer_1 += 1
                    if preds[pi][pj] == 1:
                        matching_1 += 1
                        for segment_rank in segment_ranks:
                            segment_rank_res[segment_rank]['matching_1'] += 1
                    else:
                        non_match_0 += 1  # target 1 but predicted as 0
                        for segment_rank in segment_ranks:
                            segment_rank_res[segment_rank]['non_match_0'] += 1

                normalizer_01 += 1

                for segment_rank in segment_ranks:
                    segment_rank_res[segment_rank]['normalizer_01'] += 1

        losses.append(loss.item())

    print('Target 0 -', matching_0, normalizer_0)
    print('Target 1 -', matching_1, normalizer_1)
    print('Normalizer -', normalizer_01)

    print()

    check_accs = (matching_0 + matching_1) / normalizer_01
    check_accs_0 = matching_0 / normalizer_0
    check_accs_1 = matching_1 / normalizer_1
    print('0-matching acc:', check_accs_0)
    print('1-matching acc:', check_accs_1)
    print('All-matching acc:', check_accs)

    print()

    print('Fully matching:', matching)
    print('Normalizer:',
          normalizer)  # normalizer is batch size, matching is the matching of the whole pred & target
    accs = matching / normalizer

    mean_loss = np.mean(losses)
    print('Accuracy:', accs)
    print('Mean loss:', mean_loss)

    prec_1 = matching_1 / (matching_1 + non_match_1)

    recall_1 = check_accs_1

    prec_0 = matching_0 / (matching_0 + non_match_0)
    recall_0 = check_accs_0

    print()
    print('Precision 0:', prec_0)
    print('Precision 1:', prec_1)

    print('Recall 0:', recall_0)
    print('Recall 1:', recall_1)

    print()

    beta = 1

    f1_0 = get_f1(prec_0, recall_0, beta)
    f1_1 = get_f1(prec_1, recall_1, beta)

    f1_0_b = get_f1(prec_0, recall_0, 2)
    f1_1_b = get_f1(prec_1, recall_1, 2)

    w_f1 = (f1_0 + weight * f1_1) / (1 + weight)

    print()
    print('Weighted macro F1:', w_f1)
    print('F1_0:', f1_0)
    print('F1_1:', f1_1)

    print()

    rank_p_1 = []
    rank_r_1 = []
    rank_r_0 = []
    rank_p_0 = []

    for sr in segment_rank_res:

        rank_res = segment_rank_res[sr]

        print(sr)

        sr_prec_0 = rank_res['matching_0'] / (rank_res['matching_0'] + rank_res['non_match_0']) if (rank_res[
                                                                                                        'matching_0'] +
                                                                                                    rank_res[
                                                                                                        'non_match_0']) > 0 else 1
        sr_prec_1 = rank_res['matching_1'] / (rank_res['matching_1'] + rank_res['non_match_1']) if (rank_res[
                                                                                                        'matching_1'] +
                                                                                                    rank_res[
                                                                                                        'non_match_1']) > 0 else 1

        sr_norm_0 = rank_res['matching_0'] + rank_res['non_match_1']  # nm1 predicted as 1 but actually 0
        sr_norm_1 = rank_res['matching_1'] + rank_res['non_match_0']

        sr_rec_0 = rank_res['matching_0'] / sr_norm_0
        sr_rec_1 = rank_res['matching_1'] / sr_norm_1

        print(sr_norm_0, sr_norm_1)

        rank_p_1.append(sr_prec_1 * 100)
        rank_r_1.append(sr_rec_1 * 100)
        rank_p_0.append(sr_prec_0 * 100)
        rank_r_0.append(sr_rec_0 * 100)

        sr_f1_0 = get_f1(sr_prec_0, sr_rec_0, 1)
        sr_f1_1 = get_f1(sr_prec_1, sr_rec_1, 1)

        if sr_f1_0 == 'nan' or sr_f1_1 == 'nan':
            sr_w_f1 = 'nan'

        else:
            sr_w_f1 = (sr_f1_0 + weight * sr_f1_1) / (1 + weight)

        print(sr_prec_0, sr_prec_1, sr_rec_0, sr_rec_1, sr_f1_0, sr_f1_1, sr_w_f1)

    return rank_p_1, rank_r_1, rank_p_0, rank_r_0, segment_rank_res

# used in the gold_eval_example_test_paper.py to write the matching and non-matching
# results by the model, later to provide examples for qualitative analysis

def gold_evaluate_write(split_data_loader, dataset, breaking, normalize, mask, img_dim, model, seg2ranks, id_list, device, criterion, threshold, weight):

    count = 0

    matching = 0
    normalizer = 0

    with open('hard_match_hist.txt', 'w') as file:

        with open('hard_miss_hist.txt', 'w') as miss_file:

            for ii, data in enumerate(split_data_loader):

                seg_id = id_list[ii]
                segment_ranks = seg2ranks[str(seg_id)]

                if breaking and count == 5:
                    break

                count += 1

                segments_text = torch.tensor(data['segment'])

                normalizer += segments_text.shape[0]

                image_set = data['image_set']
                no_images = image_set.shape[1]

                actual_no_images = []

                for s in range(image_set.shape[0]):
                    actual_no_images.append(len(image_set[s].nonzero()))

                temp_batch_size = data['segment'].shape[0]

                context_sum = torch.zeros(temp_batch_size, img_dim).to(device)

                context_separate = torch.zeros(temp_batch_size, image_set.shape[1], img_dim).to(device)

                for b in range(image_set.shape[0]):

                    non_pad_item = 0.0

                    for i in range(image_set[b].shape[0]):

                        img_id = str(image_set[b][i].item())

                        if img_id != '0':

                            img_features = torch.Tensor(dataset.image_features[img_id]).to(device)

                            context_sum[b] += img_features

                            non_pad_item += 1

                        else:

                            img_features = torch.zeros(img_dim).to(device)

                        context_separate[b][i] = img_features

                    context_sum[b] = context_sum[b] / non_pad_item  # average

                lengths = data['length']
                targets = data['targets'].view(temp_batch_size, no_images, 1).float()
                prev_histories = data['prev_histories']

                out = model(segments_text, prev_histories, lengths, context_separate, context_sum, normalize, device)

                if mask:
                    out = mask_attn(out, actual_no_images, no_images, device)

                sig_out = torch.sigmoid(out)

                loss = criterion(out, targets)

                preds = sig_out.data
                preds[preds >= threshold] = 1
                preds[preds < threshold] = 0

                for p in range(preds.shape[0]):

                    if torch.all(torch.eq(preds[p], targets[p])):
                        matching += 1

                        file.write(str(ii))
                        file.write('\n')

                    else:

                        pred_str = ''
                        for v in preds[p]:
                            pred_str += str(int(v))
                            pred_str += ' '

                        tar_str = ''
                        for v in targets[p]:
                            tar_str += str(int(v))
                            tar_str += ' '


                        miss_str = str(ii) + '-' + pred_str + '-' + tar_str

                        miss_file.write(miss_str)
                        miss_file.write('\n')

            print(matching, normalizer)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="./data")
    parser.add_argument("-segment_file", type=str, default="segments.json")
    parser.add_argument("-chains_file", type=str, default="chains.json")
    parser.add_argument("-vocab_file", type=str, default="vocab.csv")
    parser.add_argument("-vectors_file", type=str, default="vectors.json")
    parser.add_argument("-split", type=str, default="train")
    parser.add_argument("-shuffle", action='store_true')
    parser.add_argument("-mask", action='store_true')
    parser.add_argument("-normalize", action='store_true')
    parser.add_argument("-weighting", action='store_true')
    parser.add_argument("-breaking", action='store_true')
    parser.add_argument("-weight", type=float, default=6.5)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-learning_rate", type=float, default=0.001)
    parser.add_argument("-seed", type=int, default=2)
    args = parser.parse_args()

    print(args)

    # prepare datasets and obtain the arguments

    t = datetime.datetime.now()
    timestamp = str(t.date()) + '-' + str(t.hour) + '-' + str(t.minute) + '-' + str(t.second)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Loading the vocab...")
    vocab = Vocab(os.path.join(args.data_path, args.vocab_file), 3)

    trainset = HistoryDataset(
        data_dir=args.data_path,
        segment_file='train_' +args.segment_file,
        vectors_file=args.vectors_file,
        chain_file='train_' + args.chains_file,
        split=args.split
    )

    testset = HistoryDataset(
        data_dir=args.data_path,
        segment_file='test_' + args.segment_file,
        vectors_file=args.vectors_file,
        chain_file='test_' + args.chains_file,
        split='test'
    )


    valset = HistoryDataset(
        data_dir=args.data_path,
        segment_file='val_' +args.segment_file,
        vectors_file=args.vectors_file,
        chain_file='val_' + args.chains_file,
        split='val'
    )

    print('vocab len', len(vocab))
    print('train len', len(trainset))
    print('test len', len(testset))
    print('val len', len(valset))

    embedding_dim = 512
    hidden_dim = 512
    img_dim = 2048
    threshold = 0.5

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    shuffle = args.shuffle
    normalize = args.normalize
    mask = args.mask
    weighting = args.weighting
    weight = args.weight
    breaking = args.breaking

    weights = torch.Tensor([weight]).to(device)

    model = HistoryModelBlind(len(vocab), embedding_dim, hidden_dim, img_dim).to(device)

    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if weighting:
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights, reduction='sum')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

    batch_size = args.batch_size

    # prepare dataloaders
    load_params = {'batch_size':batch_size,
            'shuffle': True,
                   'collate_fn': HistoryDataset.get_collate_fn(device)}

    load_params_test = {'batch_size': batch_size,
                   'shuffle': False,'collate_fn': HistoryDataset.get_collate_fn(device)}

    training_loader = torch.utils.data.DataLoader(trainset, **load_params)

    test_loader = torch.utils.data.DataLoader(testset, **load_params_test)

    val_loader = torch.utils.data.DataLoader(valset, **load_params_test)

    print('training history model')

    epochs = 100

    losses = []

    best_loss = 100000
    best_accuracy = -1

    prev_f1 = -1
    prev_loss = 100000

    for epoch in range(epochs):
         print('Epoch', epoch)
         print('Train')

         model.train()
         torch.enable_grad()

         count = 0

         for i, data in enumerate(training_loader):

            if breaking and count == 5:
                break

            count += 1

            segments_text = torch.tensor(data['segment'])

            image_set = data['image_set']
            no_images = image_set.shape[1]

            actual_no_images = []

            for s in range(image_set.shape[0]):
                actual_no_images.append(len(image_set[s].nonzero()))

            temp_batch_size = data['segment'].shape[0]

            context_sum = torch.zeros(temp_batch_size,img_dim).to(device)

            context_separate = torch.zeros(temp_batch_size, image_set.shape[1], img_dim).to(device)

            for b in range(image_set.shape[0]):

                non_pad_item = 0.0

                for i in range(image_set[b].shape[0]):

                    img_id = str(image_set[b][i].item())

                    if img_id != '0':

                        img_features = torch.Tensor(trainset.image_features[img_id]).to(device)

                        context_sum[b] += img_features

                        non_pad_item += 1
                    else:

                        img_features = torch.zeros(img_dim).to(device)

                    context_separate[b][i] = img_features

                context_sum[b] = context_sum[b] / non_pad_item

            lengths = data['length']
            targets = data['targets'].view(temp_batch_size, no_images, 1).float()
            prev_histories = data['prev_histories']

            out = model(segments_text, prev_histories, lengths, context_separate, context_sum, normalize, device)

            if mask:
                out = mask_attn(out, actual_no_images, no_images, device)

            model.zero_grad()

            loss = criterion(out, targets)

            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

         print('Train loss', np.mean(losses))

         #evaluation
         with torch.no_grad():
             model.eval()

             isValidation = False
             print('\nTrain Eval')
             evaluate(training_loader, trainset, breaking, normalize, mask, img_dim, \
                      model, epoch, args, timestamp, best_accuracy, best_loss, isValidation, weight, device)

             isValidation = True
             print('\nVal Eval')

             best_accuracy, best_loss, current_f1, current_loss = evaluate(val_loader, valset, breaking, normalize, mask, img_dim, \
                                                 model, epoch, args, timestamp, best_accuracy, best_loss,
                                                 isValidation, weight, device)

             print('\nBest', best_accuracy, best_loss) #validset
             print()

             # if the loss starts increasing, stop

             if prev_loss < current_loss:
                 break
             else:
                 prev_f1 = current_f1
                 prev_loss = current_loss
