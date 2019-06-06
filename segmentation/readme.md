# Photobook-Reference-Chains

Project developed with

* Python 3.6
* PyTorch 0.4
* NLTK



## Preprocessing

In order to pre-train the dialogue segment generator and discriminator we need to convert the PhotoBook
transcripts into labeled dialogue segments. This is done during a number of data pre-processing steps which will be
described in detail in the following sections:

### Step 1: Data split

In order to obtain representative train, test and validation splits of the data, the collected transcripts are
allocated to one of those sets based on their domain id. The PhotoBook task contains 30 sets of images that each
produce 2 different game setups with mutually exclusive sets of target images. As a result, the 2504 games collected are
spread over 60 different so-called domains, with 20-40 plays per domain.

**dialogue_segmentatation.py** runs the data split. It has arguments

* `-data_path` to specify the location of the data folder,
* `-new_split` to force a new split and
* `-split` to specify the ratio of splits.

By default, the data is plit in 70%/15%/15% Train/Val/Test. 62 games are split off of the validation set and are used for
fine-tuning of the segmentation heuristics as dev set.

Function `generate_game_sets()` allocates a specified number of games to a new split by adding games with respect to
the frequency of plays in their domain. As a result, the dev set for example contains at least one game per domain. Splits
are saved as lists of game_ids in .json files and can be loaded to retain a specific split. The doalogue segmentation scripts
attempts to load a pre-specified split by default.

### Step 2: Generating Dialogue Segments

`dialogue_segmentation()` contains a 32-case heuristics to segment the PhotoBook game transcripts. It returns a list of
tuples containing the game_id of a sample and the sample data.

The sample data is a tuple containing a list of five round sample objects and
a list of the images shown during that game round. Round objects in turn each
contain a variable-length list of the segments produced in that respective game round.

A segment finally is a tuple
containing a list of Message objects and a list of target image identifiers. For details about the Message objects please
refer to the documentation of the PhotoBook Log interface and subclasses.

### Step 3: Generating Segment Chain Training Samples

**data_preprocessor.py** uses the generated dialogue segments and the data split information to produce stripped-down data
sets for specific training requirements. In this case the data contains samples of dialogue segments and chains of segments for the
different target images extracted from a transcript

The data preprocessor takes up to 10 arguments:

* `-data_path` the location of the data folder,
* `-input_data` the name of suffix of the segment files
* `-output_data` the suffix of the output dataset files
* `-vocab` the name of vocabulary to use or to generate
* `-encoding` the message encoding to be used
* `-tokenization` the tokenization method
* `-speaker_lables` whether or not to include speaker labels in the segments
* `-lowercase` whether or not to lowercase the text input
* `-splitting` wether or not to split hypenated and slashed compounds
* `-min_occ` to specify the minimum occurrence count for words to be added to the vocabulary

The data preprocessor creates two files per data split:

The `_segments.json` file contains a list of all dialogue segments in the split.
Its fields are

* `segment`, an encoding of a segmen's utterances
* `image_set`, a list of image IDs for the pictures shown to any of the participants in the round from which the sample was taken
* `targets`, the index(es) of the target image(s) in the image set
* `length`, the number of tokens in the segment

The `_chains.json` file contains a list of all dialogue segment chains with their respective target images. It's fields are

* `target`, the target image ID
* `chain`, a list of indexes of dialogue segments associated with the target ID in a specific game
* `lengths` a list of the segment lengths in the chain. `chain` and `lengths` are of equal length and identical sorting

### Re-creating the Data

To re-create the data sets, run

`python dialogue_segmentation.py`

and

`python data_preprocessor.py`

without any arguments. As default, the preprocessing tool will use the NLTK `word_tokenize` method on lowercased tokens split on hyphons or slashes. The minimum occurrence count for words to be added to the vocabulary is 3 by default.

## Data Loader

The SegmentDataset class is introduced to supply models with training data.
It initialises a dataset object that contains the training samples as well
as the dialogue and image encodings. It provides an iterator to
access training samples through `__getitem()__` and the total number of samples
through `__len()__`.

See **data_tester.py** for example usage.
