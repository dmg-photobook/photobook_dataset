import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class HistoryModelBlind(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, img_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.img_dim = img_dim

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx = 0)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)

        self.lstm_hist = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)

        self.linear = nn.Linear(self.img_dim, int(self.hidden_dim))

        self.linear_separate = nn.Linear(self.img_dim, self.hidden_dim)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.0) #no dropout

    def forward(self, segment, prev_histories, lengths, separate_images, visual_context, normalize, device):
        """
        @param segment (torch.LongTensor): Tensor of size [batch_size, sequence_length], segment text converted into indices
        @param prev_histories (list): List of size [batch_size], every item contains a list of size [image_set_size] variable depending on the visual context size per segment-to-be-resolved
                                      Each item of the inner lists is a concatenated reference chain per image, converted into indices
        @param lengths (torch.LongTensor): Tensor of size [batch_size], containing the segment lengths per batch
        @param separate_images (torch.FloatTensor): Tensor of size [batch_size, max_image_set_size, image_feature_dim], image feature vectors for all the images in the context per batch
        @param visual_context (torch.FloatTensor): NOT USED IN THIS MODEL, Tensor of size [batch_size, image_feature_dim], average image feature vector for all the images in the context per batch
        @param normalize (bool): whether to normalize the candidate images or not
        @param device (torch.device): device to map the history per image into
        """

        batch_size = segment.shape[0]

        embeds_words = self.embedding(segment) #b, l, d

        embeds_words = self.dropout(embeds_words)

        # pack sequence
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        embeds_words = embeds_words[sorted_idx]

        packed_input = nn.utils.rnn.pack_padded_sequence(embeds_words, sorted_lengths, batch_first=True)

        # rnn forward
        packed_outputs, hidden = self.lstm(packed_input, hx = None)

        # re-pad sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # un-sort
        _, reversed_idx = torch.sort(sorted_idx)
        outputs = outputs[reversed_idx]

        batch_out_hidden = hidden[0][:, reversed_idx]


        #print(batch_out_hidden.shape, separate_images.shape)

        separate_images = self.linear_separate(separate_images)
        separate_images = self.dropout(separate_images)

        #get linguistic history per image specific to that game up to that segment


        for b in range(batch_size):

            prev_hist = prev_histories[b]

            for p in range(len(prev_hist)):

                cur_img_hist = torch.tensor(prev_hist[p]).long().to(device)

                if len(prev_hist[p]) > 0:

                    #encode linguistic background

                    hist_embeds = self.embedding(cur_img_hist).view(1,-1,self.embedding_dim)

                    hist_out, hist_hidden = self.lstm_hist(hist_embeds)

                    separate_images[b][p] += hist_hidden[0].squeeze()

                else:
                    #no linguistic background, just use image features
                    pass

        if normalize:
            separate_images = self.relu(separate_images)

            separate_images = F.normalize(separate_images, p=2, dim=2)


        dot = torch.bmm(separate_images, batch_out_hidden.view(batch_size, self.hidden_dim,1))

        return dot

