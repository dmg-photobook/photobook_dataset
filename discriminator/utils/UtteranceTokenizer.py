from nltk import word_tokenize
from nltk import TweetTokenizer


class UtteranceTokenizer():
    def __init__(self, ):
        self.tweet_tokenizer = TweetTokenizer()

    def tokenize_utterance(self, utterance, method="word_tokenize", lowercase=True, splitting=True):
        """
        Tokenises a given utterance
        :param utterance: String. Utterance to be tokenised
        :param method: String. Method indicato; options: "word_tokenize" and "tweet_tokenize"
        :param lowercase: bool. Set True to lowercase the utterance before processing it
        :param splitting: bool. Set True to split hyphenated and slashed composites and *-marked corrections
        :return: list. A list of word tokens from the utterance
        """
        if lowercase: utterance = utterance.lower()
        tokens = []
        if method == "word_tokenize":
            tokens = word_tokenize(utterance)
        elif method == "tweet_tokenize":
            tokens = self.tweet_tokenizer.tokenize(utterance)
        else:
            print("Warning: Tokenization method not implemented!")

        if splitting:
            output = []
            for t in tokens:
                if "-" in t[1:]:
                    output.extend(t.split("-"))
                elif "/" in t[1:]:
                    output.extend(t.split("/"))
                elif "*" == t[0]:
                    output.append(t.split("*")[1])
                else:
                    output.append(t)
            return output
        else:
            return tokens



