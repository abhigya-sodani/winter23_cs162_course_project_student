import os
import sys
import json
import csv
import glob
import pprint
import numpy as np
import random
import argparse
from tqdm import tqdm
from .utils import DataProcessor
from .utils import Coms2SenseSingleSentenceExample
from transformers import (
    AutoTokenizer,
)


class Com2SenseDataProcessor(DataProcessor):
    """Processor for Com2Sense Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

    def __init__(self, data_dir=None, args=None, **kwargs):
        """Initialization."""
        self.args = args
        self.data_dir = data_dir

        # TODO: Label to Int mapping, dict type.
        self.label2int = {"True": 1, "False": 0}

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

    def _read_data(self, data_dir=None, split="train"):
        """Reads in data files to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir

        examples = []   # Store your examples in this list

        ##################################################
        # TODO:
        # Some instructions for reading data:
        # 1. Use json python package to load the data properly.
        # 2. Use the provided class `Coms2SenseSingleSentenceExample` 
        # in `utils.py` for creating examples
        # 3. Store the two complementary statements as two 
        # individual examples 
        # e.g. example_1 = ...
        #      example_2 = ...
        #      examples.append(example_1)
        #      examples.append(example_2)
        # 4. Make sure that the order is maintained.
        # i.e. sent_1 in the data is stored/appended first and
        # sent_2 in the data is stored/appened after it.
        # 5. For the guid, simply use the row number (0-
        # indexed) for each data instance.
        # Use the same guid for statements from the same complementary pair.
        # 6. Make sure to handle if data do not have labels field.
        # This is useful for loading test data
        
        with open(data_dir + '/' + split + '.json', 'r') as file:
            data = json.load(file)
        
        if split=="dev":
            for i, row in enumerate(data):
                if row["scenario"] == "causal":
                    example_1 = Coms2SenseSingleSentenceExample(
                        guid=i,
                        text=row["sent_1"],
                        domain=row["domain"],
                        scenario=row["scenario"],
                        numeracy=row["numeracy"] == 'True',
                    )
                    if "label_1" in row:
                        example_1.label = int(row["label_1"] == 'True')
                    example_2 = Coms2SenseSingleSentenceExample(
                        guid=i,
                        text=row["sent_2"],
                        domain=row["domain"],
                        scenario=row["scenario"],
                        numeracy=row["numeracy"] == 'True',
                    )
                    if "label_2" in row:
                        example_2.label = int(row["label_2"] == 'True')
                    examples.append(example_1)
                    examples.append(example_2)
                else:
                    pass

        else:
            for i, row in enumerate(data):
                example_1 = Coms2SenseSingleSentenceExample(
                    guid=i,
                    text=row["sent_1"],
                    domain=row["domain"],
                    scenario=row["scenario"],
                    numeracy=row["numeracy"] == 'True',
                )
                if "label_1" in row:
                    example_1.label = int(row["label_1"] == 'True')
                example_2 = Coms2SenseSingleSentenceExample(
                    guid=i,
                    text=row["sent_2"],
                    domain=row["domain"],
                    scenario=row["scenario"],
                    numeracy=row["numeracy"] == 'True',
                )
                if "label_2" in row:
                    example_2.label = int(row["label_2"] == 'True')
                examples.append(example_1)
                examples.append(example_2)
        # End of TODO.
        ##################################################

        return examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="train")

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="dev")

    def get_test_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="test")


if __name__ == "__main__":

    # Test loading data.
    proc = Com2SenseDataProcessor(data_dir="datasets/com2sense")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print()
    for i in range(3):
        print(test_examples[i])
    print()
