from utils.UtteranceTokenizer import UtteranceTokenizer

class SegmentBuilder():
    def __init__(self, ):
        self.tokenizer = UtteranceTokenizer()

    def build(self, messages, targets, image_set, vocab, method="word2index", tokenization="word_tokenize", speaker_lables=True, lowercase=True, splitting=True):
        dialogue_segment = {}

        dialogue_segment["segment"] = self.flatten_and_encode(messages, method, vocab, tokenization, speaker_lables, lowercase, splitting)
        dialogue_segment["image_set"] = [str(target.split('_')[-1].split('.')[0].lstrip('0')) for target in image_set]
        target_ids = [str(target.split('_')[-1].split('.')[0].lstrip('0')) for target in targets]
        dialogue_segment["targets"] = [dialogue_segment["image_set"].index(target) for target in target_ids]
        dialogue_segment["length"] = len(dialogue_segment["segment"])

        return dialogue_segment

    def flatten_and_encode(self, messages, method="word2index", vocab=None, tokenization="word_tokenize", speaker_lables=True, lowercase=True, splitting=True):
        """
        Concatenates the utterances of a dialogue segment and encodes them in the specified manner
        :param messages: list. List of Message objects
        :param method: String. Specifies the desired word encoding scheme
        :param vocab: Vocabulary object. Vocabulary for word encoding
        :param speaker_lables: bool. Set to False to disable adding speaker labels in the output string
        :return: list. A vector representation of the encoded dialogue segment
        """
        assert vocab, print("Warning: No vocabulary given!")

        last_speaker = None
        segment = []
        for message in messages:
            if message.type == "text":
                speaker = message.speaker
                if speaker_lables and last_speaker != speaker:
                    if tokenization == "word_tokenize":
                        segment.extend(vocab.encode(["-" + speaker + "-"]))
                    else:
                        segment.extend(vocab.encode(["<" + speaker + ">"]))
                    last_speaker = speaker
                segment.extend(vocab.encode(self.tokenizer.tokenize_utterance(message.text, tokenization, lowercase, splitting)))
        if method == "word2index":
            pass
        else:
            print("Warning: Encoding method not implemented.")
        return segment



