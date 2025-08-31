# Copyright 2024 THU-BPM MarkLLM.
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

# ===========================================
# dataset.py
# Description: Dataset classes for evaluation
# ===========================================

import json
import numpy as np

class BaseDataset:
    """Base class for dataset."""

    def __init__(self, max_samples: int = 200) -> None:
        """
        Initialize the dataset.
        
        Parameters:
            max_samples (int): Maximum number of samples to load. Default is 200.
        """
        self.max_samples = max_samples
        self.prompts = []
        self.natural_texts = []
        # begin new
        self.watermarked_texts = []
        self.unwatermarked_texts = []
        self.indexing_legacy = False
        # end new
        self.references = []

    @property
    def prompt_nums(self):
        """Return the number of prompts."""
        return len(self.prompts)

    @property
    def natural_text_nums(self):
        """Return the number of natural texts."""
        return len(self.natural_texts)

    @property
    def reference_nums(self):
        """Return the number of references."""
        return len(self.references)

    # begin new
    def get_prompt(self, index):
        """Return the prompt at the specified index."""
        return [self.prompts[i] for i in index] if isinstance (index, range) else self.prompts[index]

    def get_natural_text(self, index):
        """Return the natural text at the specified index."""
        return [self.natural_texts[i] for i in index] if isinstance (index, range) else self.natural_texts[index]
        
    def get_watermarked_text(self, index):
        """Return the watermarked text at the specified index."""
        return [self.watermarked_texts[i] for i in index] if isinstance (index, range) else self.watermarked_texts[index]
    
    def get_unwatermarked_text(self, index):
        """Return the unwatermarked text at the specified index."""
        return [self.unwatermarked_texts[i] for i in index] if isinstance (index, range) else self.unwatermarked_texts[index]
    # end new

    def get_reference(self, index):
        """Return the reference at the specified index."""
        return self.references[index]

    def load_data(self):
        """Load and process data to populate prompts, natural_texts, and references."""
        pass


class C4Dataset(BaseDataset):
    """Dataset class for C4 dataset."""

    def __init__(self, data_source: str, max_samples: int = 200, indexing_legacy: bool = False, 
                 watermark_idx: str = "watermarked_sample",
                 unwatermarked_idx: str = "unwatermarked_sample") -> None:
        """
            Initialize the C4 dataset.

            Parameters:
                data_source (str): The path to the C4 dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.indexing_legacy = indexing_legacy
        self.watermark_idx = watermark_idx
        self.unwatermarked_idx = unwatermarked_idx
        self.load_data()
        
    # begin new
    def load_data(self):
        """Load data from the C4 dataset file."""
        with open(self.data_source, 'r') as f:
           data = json.load(f)
        prompt_idx = "prompt"
        natural_text_idx = "natural_text"
        watermarked_text_idx = self.watermark_idx
        unwatermarked_text_idx = self.unwatermarked_idx
        if self.indexing_legacy:
            prompt_idx = "model_prompt"
            natural_text_idx = "gold"
            watermarked_text_idx = "wm"
            unwatermarked_text_idx = "unwm"
        for example in data[:self.max_samples]:
            self.prompts.append(example[prompt_idx])
            self.natural_texts.append(example[natural_text_idx])
            self.watermarked_texts.append(example[watermarked_text_idx])
            self.unwatermarked_texts.append(example[unwatermarked_text_idx])
    # end new

class WMT16DE_ENDataset(BaseDataset):
    """Dataset class for WMT16 DE-EN dataset."""

    def __init__(self, data_source: str, max_samples: int = 200) -> None:
        """
            Initialize the WMT16 DE-EN dataset.

            Parameters:
                data_source (str): The path to the WMT16 DE-EN dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the WMT16 DE-EN dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['de'])
            self.references.append(item['en'])


class HumanEvalDataset(BaseDataset):
    """Dataset class for HumanEval dataset."""

    def __init__(self, data_source: str, max_samples: int = 200) -> None:
        """
            Initialize the HumanEval dataset.

            Parameters:
                data_source (str): The path to the HumanEval dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the HumanEval dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            # process prompt
            prompt = item['prompt']
            sections = prompt.split(">>>")
            prompt = sections[0]
            if len(sections) > 1:
                prompt += '\"\"\"'

            self.prompts.append(prompt)
            self.references.append({'task': prompt, 'test': item['test'], 'entry_point': item['entry_point']})


if __name__ == '__main__':
    d1 = C4Dataset('dataset/c4/processed_c4.json', max_samples=100)
    d2 = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl', max_samples=100)
    d3 = HumanEvalDataset('dataset/HumanEval/test.jsonl', max_samples=100)
