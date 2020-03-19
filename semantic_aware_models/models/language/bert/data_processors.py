import sys
import csv
import logging
import xml.etree.ElementTree as ET
import pandas as pd

from .inputs import InputExample, InputFeatures
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)

class DataProcessor(object):

    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):

        """Gets a collection of `InputExample`s for the train set."""
        pass

    def get_dev_examples(self, data_dir):

        """Gets a collection of `InputExample`s for the dev set."""
        pass

    def get_test_examples(self, data_dir, data_file_name, size=-1):

        """Gets a collection of `InputExample`s for the test set."""
        pass

    def get_labels(self):

        """Gets the list of labels for this data set."""
        pass

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):

        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class SAProcessor(DataProcessor):

    """ Processor for our data set """

    def get_train_examples(self, file_path):
        return self._create_examples(file_path, "train")

    def get_dev_examples(self, file_path):
        return self._create_examples(file_path, "dev")

    def get_labels(self):
        return ["P", "NEU", "NONE", "N"]

    def _create_examples(self, filepath, type):

        """ Creates examples for training and dev sets """

        examples = []
        tweets = ET.parse(filepath)
        for tweet in tweets.findall('/tweet'):
            content = tweet.find('content')
            tweetid = tweet.find('tweetid')
            sentiment = tweet.find('sentiment').find('polarity').find('value')
            # print("Content = {0} --> Tag = {1}".format(content.text, sentiment.text))

            id = tweetid.text
            sentence = content.text
            tag = sentiment.text
            examples.append(InputExample(guid=id, text_a=sentence, text_b=None, label=tag))
        return examples


class SAProcessorCSV5(DataProcessor):

    """ Processor for our data set """

    def get_train_examples(self, file_path):
        return self._create_examples(file_path, "train")

    def get_dev_examples(self, file_path):
        return self._create_examples(file_path, "dev")

    def get_labels(self):
        return ["P+", "P", "NEU", "N", "N+"]

    def _create_examples(self, filepath, type):

        """ Creates examples for training and dev sets """

        examples = []
        niter = 0
        with open(filepath, encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                words, tags = [], []
                if niter == 0:
                    print("CSV Header: ", row)
                else:
                    sentence, tag = row[0], row[1]
                    #tokens = sentence.strip().split(' ')  #Tokenization
                    examples.append(InputExample(guid=niter, text_a=sentence, text_b=None, label=tag))
                niter += 1

        return examples


class CBRSProcessor(DataProcessor):

    """ Processor for our data set """

    def get_train_examples(self, descriptions, ratings, user_id):
        return self._create_examples(descriptions, ratings, user_id)

    def get_dev_examples(self, descriptions, ratings, user_id):
        return self._create_examples(descriptions, ratings, user_id)

    def get_labels(self):
        return [1.0, 2.0, 3.0, 4.0, 5.0]

    def _create_examples(self, descriptions, ratings, user_id):

        """ Creates examples for training and dev sets """

        examples = []
        ratings_user = ratings.get_item_ids_from_user(user_id=user_id)
        for i in ratings_user:
            print('Item: ', i)
            description = descriptions.get_description_from_id(item_id=i)
            rating =  ratings.get_preference_value(user_id=user_id, item_id=i)
            examples.append(InputExample(guid=i, text_a=description, text_b=None, label= rating))

        return examples


processors = {
    "sa": SAProcessor,
    "sa_csv": SAProcessorCSV5,
    "cbrs": CBRSProcessor,
}

output_modes = {
    "sa": "classification",
    "sa_csv": "classification",
    "cbrs": "classification",
}

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)} #label2id
    id2label = {i: label for i, label in enumerate(label_list)} #id2label
    mlb = None

    features = []
    for (ex_index, example) in enumerate(examples):
        #if ex_index % 10000 == 0:
        #    logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        elif output_mode == "prediction":
            label_id = 0.0
        elif output_mode == "multiclassification":
            labels_ids = []
            mlb = MultiLabelBinarizer(classes=label_list)
            multi_label_id = mlb.fit_transform([example.label])
            # label_id = multi_label_id
            for l in multi_label_id[0]:
                labels_ids.append(float(l))
            label_id = labels_ids
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
                InputFeatures(tokens=tokens,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    if mlb:
        return features, id2label, mlb
    else:
        return features, id2label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):

    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


