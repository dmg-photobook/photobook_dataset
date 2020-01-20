# Photobook Discriminator

Project developed with

* Python 3.6
* PyTorch 0.4
* NLTK
* NumPy 1.16

Download the pre-trained models from https://github.com/dmg-photobook/photobook_acl_models.

Put the models in the discriminator folder.

Unzip [discriminator_data_full.zip](https://github.com/dmg-photobook/photobook_dataset/blob/master/discriminator_data_full.zip), move the 'data' folder under the discriminator folder.

If you would like to run the gold evaluation scripts, you would need 2 additional files:
* test_seg2ranks.json
* test_seg_ids.json

These will help you get the ordered IDs of segments for history models (the order of encounter with the segments as they come in the chains).

To generate these files, you can use the segment_ranks_ids.py script.

## Utils

**Vocab.py** is used to load the vocabulary, which also includes \<pad\> and \<unk\> tokens. It stores tokens with a frequency lower than the minimum occurrence limit as \<unk\>.
  
**SegmentDataset.py** provides the models with segment information as training/test/validation data.
For each segment, it retrieves the tokenized segment text and its length, the image set for the segment and the target image/images out of this set. This class uses the segment file and the image feature file. Pads the segments and image sets, when necessary in batching.

**HistoryDataset.py** provides the models with segment information and chain histories.
For each segment, it retrieves the tokenized segment text and its length, the image set for the segment and the target image/images out of this set. In addition to the functionalities of SegmentDataset, this class also supplies the concatenated previous reference segments for each image in the image set up to that point in the game. This class uses the segment file, chain file and the image feature file. Pads the dialogue segments and image sets, when necessary in batching.

**segment_ranks_ids.py**
Script to get all the ranks a segments is positioned in different chains (hence, there could be multiples of the same position).

## Training

The training scripts take the arguments below:

* `-data_path` the location of the data folder
* `-segment_file` the name of the .json file containing the dialogue segments
* `-chains_file` the name of the .json file containing the dialogue chains
* `-vocab_file` the name of the vocabulary used in the models (.csv file)
* `-vectors_file` the name of the .json file containing the image feature vectors
* `-split` the split name
* `-shuffle` flag to indicate whether to shuffle the data (not used, train takes True, val and test take False
* `-mask` flag to indicate whether to mask pads
* `-normalize` flag to indicate whether to normalize target features (depending on the model: image, image + history, history)
* `-weighting` flag to indicate whether to give separate weights for the target vs. non-target items
* `-weight` weight of the positive class to be fed into the loss criterion
* `-learning_rate` learning rate given to the optimizer
* `-seed` value of the random seed for PyTorch and NumPy 

**train_history.py** trains the History model.
**train_nohistory.py** trains the No history model.
**train_history_noimg.py** trains the History - No image model.

These scripts also include the methods for evaluation, both during training and later for the analysis of the results obtained using the test set.


## Models

### No History
Takes a batch of segments, segment lengths, image set for the segment, visual context (not used in this model) and a flag to normalize or not. The details of the architecture can be found in our paper.
Returns the dot product between the segment representation and the image representations for each instance in the batch.
               
### History
Takes a batch of segments, segment lengths, image set for the segment, current history for the images in the image set, visual context (not used in this model), a flag to normalize or not and the device to move the history per image representations to. The details of the architecture can be found in our paper.

Returns the dot product between the segment representation and the representation combining the image and its history for each instance in the batch.
        
### History - No image
Takes a batch of segments, segment lengths, image set for the segment (not used in this model), current history for the images in the image set, visual context (not used in this model) and a flag to normalize or not. The details of the architecture can be found in our paper.

Returns the dot product between the segment representation and the linguistic representation per image in the image set (excluding the image features).

## Evaluation

**baseline_calc.py** runs a random baseline for the task of correctly predicting the target images given a segment and an image set. At each decision point, the model makes a random decision. It can also be changed to predict target or non-target all the time. You can provide a dummy model pickle for this purpose.

**gold_eval_test_paper.py** runs the given models using the test set and writes correct-incorrect predictions into file for qualitative analysis. The indices for the history model require an additional step (as unique segments come in the order they occur the first time in the whole dialogue set), care should be taken when looking at the files.

**gold_eval_test.py** plots segment rank vs. precision 0, precision 1, recall 0, recall 1 for the provided models using the test set.

**gold_eval_test_f1.py** plots segment rank vs. F1_0, F1_1 and overall F1 for the provided models using the test set.

**gold_eval_example_test_paper.py** plots precision and recall of the target items as illustrated in the paper, prints out detailed statistics regarding the test set performance of the provided models.


