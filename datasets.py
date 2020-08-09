import os
import torch
import tqdm
import logging
import csv
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Optional
from dataclasses import dataclass
from filelock import FileLock


logger = logging.getLogger(__name__)

@dataclass
class DatasetInputExample:
    
    contents: str
    label: Optional[int]
        

@dataclass
class DatasetInputFeature:
    
    input_ids: List[int]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]
            
            
class ClassificationDataset(Dataset):

    features: List[DatasetInputFeature]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: str = "train",
    ):

        processor = ClassificationProcessor()

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(mode, tokenizer.__class__.__name__, str(max_seq_length), task,),
        )
        
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")

                if mode == "dev":
                    examples = processor.get_dev_examples(data_dir)
                elif mode == "test":
                    examples = processor.get_test_examples(data_dir)
                else:
                    examples = processor.get_train_examples(data_dir)

                logger.info("Training examples: {}".format(len(examples)))
                self.features = convert_examples_to_features(examples, max_seq_length, tokenizer,)
                logger.info("Saving features into cached file {}".format(cached_features_file))
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> DatasetInputExample:
        return self.features[i]

    
class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()
    
    
class ClassificationProcessor(DataProcessor):
    
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        train = self._read_csv(data_dir, "train")
        
        return self._create_examples(train)

    
    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        dev = self._read_csv(data_dir, "dev")
        
        return self._create_examples(dev)

    
    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        test = self._read_csv(data_dir, "test")
        
        return self._create_examples(test)

    
    def _read_csv(self, input_dir, set_type):
        corpus = []
        with open("{}/{}.csv".format(input_dir, set_type), 'r', encoding='utf-8') as f:
            data = csv.reader(f)
            for line in data:
                if len(line) == 1:
                    corpus.append([line[0]])
                else:
                    corpus.append([line[1], int(line[0])])

        return corpus

    
    def _create_examples(self, corpus):
        """Creates examples for the training and dev sets."""
        examples = []
        for data in corpus:
            examples.append(
                DatasetInputExample(
                    contents=data[0],
                    label=data[1] if len(data) == 2 else None
                )
            )
        return examples

        
def convert_examples_to_features(
    examples: List[DatasetInputExample], max_length: int, tokenizer: PreTrainedTokenizer,
) -> List[DatasetInputFeature]:
    """
    Loads a data file into a list of `DatasetInputExample`
    """

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("\nWriting example {} of {}".format(ex_index, len(examples)))

        inputs = tokenizer(
            example.contents,
            max_length=max_length,
            padding="max_length",
            truncation=True
        )

        features.append(
            DatasetInputFeature(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask if "attention_mask" in inputs else None,
                token_type_ids=None,
                label=example.label
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: {}".format(f))

    return features
