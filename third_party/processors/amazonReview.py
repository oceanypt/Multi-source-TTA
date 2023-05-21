# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" XNLI utils (dataset loading and evaluation) """


import logging
import os

from transformers import DataProcessor
from .utils import InputExample

logger = logging.getLogger(__name__)


class AmazonReviewProcessor(DataProcessor):
    """Processor for the amazon review dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_examples(self, data_dir, language='en', split='train'):
      """See base class."""
      examples = []
      for lg in language.split(','):
        lines = self._read_tsv(os.path.join(data_dir, "{}.{}".format(lg, split)))
        
        for (i, line) in enumerate(lines):
          guid = "%s-%s-%s" % (split, lg, i)
          text = line[1]
          label = line[0]

          assert isinstance(text, str)  and isinstance(label, str)
          examples.append(InputExample(guid=guid, text_a=text, label=label, language=lg))
      return examples

    def get_train_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='train')

    def get_dev_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='dev')

    def get_test_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='test')

   
    def get_labels(self):
        """See base class."""
        #return ["contradiction", "entailment", "neutral"]
        #return ["1", '2', '3', '4', '5']
        #return ["negative", 'neural', 'postive']
        return ["0", '1', '2']




amazonReview_processors = {
    "amazonReview": AmazonReviewProcessor,
}

amazonReview_output_modes = {
    "amazonReview": "classification",
}

amazonReview_tasks_num_labels = {
    "amazonReview": 3,
}
