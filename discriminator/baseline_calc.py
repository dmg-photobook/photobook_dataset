# calculates random baseline

import numpy as np

import torch
from torch import nn
from torch import optim
from models.model_nohistory import DiscriminatoryModelBlind

import os

from utils.SegmentDataset import SegmentDataset
from utils.Vocab import Vocab

def get_f1(prec, recall, beta):
    return (1 + np.power(beta,2)) *((prec*recall)/(np.power(beta,2)*prec+recall))

def load_model(file, model):
    len_vocab = 3424
    embedding_dim = 512
    hidden_dim = 512
    img_dim = 2048
    model = model(len_vocab, embedding_dim, hidden_dim, img_dim)
    optimizer = optim.Adam(model.parameters())
    checkpoint = torch.load(file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    args_train = checkpoint['args']
    return model,optimizer,epoch,loss,accuracy, args_train


def mask_attn(scores, actual_num_images, max_num_images, device):
    masks = []

    for n in range(len(actual_num_images)):
        mask = [1] * actual_num_images[n] + [0] * (max_num_images - actual_num_images[n])
        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).byte().to(device)
    scores.masked_fill_(1 - masks, float(-1e30))
    return scores

def evaluate(split_data_loader, dataset, breaking, normalize, mask, img_dim, model, epoch, args, isValidation):


    print(weight)
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

    for i, data in enumerate(split_data_loader):

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

            context_sum[b] = context_sum[b] / non_pad_item #average #NOT USED in the model


        lengths = data['length']
        targets = data['targets'].view(temp_batch_size, no_images, 1).float()

        out = model(segments_text, lengths, context_separate, context_sum, normalize)

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

                preds[pi][pj] = np.random.randint(2)

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

    #IF NO TP AND FP precision is 1

    prec_1 = matching_1 / (matching_1 + non_match_1) if (matching_1 + non_match_1) > 0 else 1

    recall_1 = check_accs_1

    prec_0 = matching_0 / (matching_0 + non_match_0) if (matching_0 + non_match_0) > 0 else 1
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

    f1 = (normalizer_0 * f1_0 + weight * normalizer_1 * f1_1) / (normalizer_0 + weight * normalizer_1)

    micro_f1 = 2 * (matching_0 + matching_1) / (matching_1 + non_match_1 + matching_0 + non_match_0 + normalizer_0 + normalizer_1)

    macro_f1 = (f1_0 + f1_1)/2

    f1_0_b = get_f1(prec_0, recall_0, 2)
    f1_1_b = get_f1(prec_1, recall_1, 2)

    beta_f1 = (normalizer_0 * f1_0_b + weight * normalizer_1 * f1_1_b) / (normalizer_0 + weight * normalizer_1)

    print()

    print('Weighted macro F1:', f1)
    print('Beta-regulated weighted F1:', beta_f1)
    print('Macro F1:', macro_f1)
    print('Micro F1:', micro_f1)

    print('F1:', f1)
    print('F1_0:', f1_0)
    print('F1_1:', f1_1)


model_file = 'model_blind_accs_2019-02-17-21-18-7.pkl' #"dummy.pkl"
model,optimizer,epoch,loss, accuracy, args_train = load_model(model_file, DiscriminatoryModelBlind)

args = args_train

print(args)

print("Loading the vocab...")
vocab = Vocab(os.path.join(args.data_path, args.vocab_file), 3)

trainset = SegmentDataset(
    data_dir=args.data_path,
    segment_file=args.segment_file,
    vectors_file=args.vectors_file,
    split=args.split
)

testset = SegmentDataset(
    data_dir=args.data_path,
    segment_file=args.segment_file,
    vectors_file=args.vectors_file,
    split='test'
)


valset = SegmentDataset(
    data_dir=args.data_path,
    segment_file=args.segment_file,
    vectors_file=args.vectors_file,
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

learning_rate = args.learning_rate
if weighting:
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
else:
    criterion = nn.BCEWithLogitsLoss()

batch_size = args.batch_size
batch_size = 1 #NO PADDING

load_params = {'batch_size':batch_size,
        'shuffle': False,
               'collate_fn': SegmentDataset.get_collate_fn(device)}

print('train vis context')

load_params_test = {'batch_size': batch_size,
               'shuffle': False,'collate_fn': SegmentDataset.get_collate_fn(device)}

training_loader = torch.utils.data.DataLoader(trainset, **load_params)

test_loader = torch.utils.data.DataLoader(testset, **load_params_test)

val_loader = torch.utils.data.DataLoader(valset, **load_params_test)


with torch.no_grad():
    model.eval()

    # 10 random runs
    for i in range(10):
        isValidation = False

        print('\nTest Eval')

        evaluate(test_loader, testset, breaking, normalize, mask, img_dim, model, epoch, args, isValidation)



