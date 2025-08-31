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

# ============================================
# exp.py
# Description: Implementation of EXP algorithm
# ============================================

import torch
import scipy
from math import log
from ..base import BaseWatermark
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from exceptions.exceptions import AlgorithmNameMismatchError
from visualize.data_for_visualization import DataForVisualization
from transformers.cache_utils import HybridCache, DynamicCache

class EXPConfig:
    """Config class for EXP algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/EXP.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'EXP':
            raise AlgorithmNameMismatchError('EXP', config_dict['algorithm_name'])

        self.prefix_length = config_dict['prefix_length']
        self.hash_key = config_dict['hash_key']
        self.threshold = config_dict['threshold']
        self.sequence_length = config_dict['sequence_length']
        self.top_k = config_dict['top_k']

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class EXPUtils:
    """Utility class for EXP algorithm, contains helper functions."""

    def __init__(self, config: EXPConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP utility class.

            Parameters:
                config (EXPConfig): Configuration for the EXP algorithm.
        """
        self.config = config
        self.rng = torch.Generator()

    def seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed the random number generator with the last `prefix_length` tokens of the input."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        prev_token = time_result % self.config.vocab_size
        self.rng.manual_seed(self.config.hash_key * prev_token)
        return
    
    def exp_sampling(self, probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Sample a token from the vocabulary using the exponential sampling method."""
        
        # If top_k is not specified, use argmax
        if self.config.top_k <= 0:
            return torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)
        
        # Ensure top_k is not greater than the vocabulary size
        top_k = min(self.config.top_k, probs.size(-1))
    
        # Get the top_k probabilities and their indices
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
    
        # Perform exponential sampling on the top_k probabilities
        sampled_indices = torch.argmax(u.gather(-1, top_indices) ** (1 / top_probs), dim=-1)
    
        # Map back the sampled indices to the original vocabulary indices
        return top_indices.gather(-1, sampled_indices.unsqueeze(-1))
    
    def _value_transformation(self, value):
        """Transform the value to a range between 0 and 1."""
        return value/(value + 1)
    

class EXP(BaseWatermark):
    """Top-level class for the EXP algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the EXP algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        # new begin
        super().__init__(algorithm_config, transformers_config, **kwargs)
        # new end
        self.config = EXPConfig(algorithm_config, transformers_config)
        self.utils = EXPUtils(self.config)
    
    # new begin
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        # Encode prompts
        instruction = kwargs.get("instruction", None)
        if instruction:
            # add messages for instruction-following models
            prompts = []
            for p in prompt:
                messages = [
                        {"role": "user", "content": p}
                    ]
                encoded_prompt = self.config.generation_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(encoded_prompt)
            encoded_prompts = self.config.generation_tokenizer(prompts, return_tensors="pt", truncation=True, padding='max_length', max_length=self.prompt_tokens).to(self.config.device) # mansour recently changed padding to 'max_length' and added max_length=self.prompt_tokens
        else:
            # encoded_prompts = self.config.generation_tokenizer.batch_encode_plus(
            #     prompt, return_tensors="pt", add_special_tokens=True, padding="max_length", truncation=True, max_length=self.prompt_tokens).to(self.config.device)
            encoded_prompts = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True, padding="max_length", truncation=True, max_length=self.prompt_tokens).to(self.config.device)
            
        # Initialize
        inputs = encoded_prompts['input_ids']
        attn = encoded_prompts['attention_mask']
        new_inputs = {'input_ids': inputs, 'attention_mask': attn}
        max_generated_length = inputs.shape[1] + self.config.sequence_length
        # only for model with cache implementation support - new transformers
        past = None
        cache_position = None
        cache_implementation = None
        if hasattr(self.config.generation_model.config, 'cache_implementation'):
            cache_implementation = self.config.generation_model.config.cache_implementation
            cache_position = torch.arange(inputs.shape[1], dtype=torch.int64, device=self.config.device)
            if 'hybrid' in cache_implementation.lower():
                past = HybridCache(config=self.config.generation_model.config,
                                max_batch_size=inputs.shape[0],
                                max_cache_len=max_generated_length,
                                device=self.config.device,
                                dtype=self.config.generation_model.dtype,)
            elif 'dynamic' in cache_implementation.lower():
                past = DynamicCache(config=self.config.generation_model.config,
                                max_batch_size=inputs.shape[0],
                                max_cache_len=max_generated_length,
                                device=self.config.device,
                                dtype=self.config.generation_model.dtype,)
            else:
                raise NotImplementedError(f"Cache implementation {cache_implementation} not supported.")

        # Generate tokens
        for i in range(self.config.sequence_length):
            with torch.no_grad():
                output = self.config.generation_model(**new_inputs, past_key_values=past, cache_position=cache_position, use_cache=True)
            # Get probabilities
            probs = torch.nn.functional.softmax(output.logits[:, -1, :self.config.vocab_size], dim=-1).cpu()
            
            
            # Generate r1, r2,..., rk
            tokens = []
            for j in range(inputs.size(0)):
                self.utils.seed_rng(inputs[j])
                random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.rng)
            
                # Sample token to add watermark
                token = self.utils.exp_sampling(probs[j].unsqueeze(0), random_numbers).to(self.config.device)
                tokens.append(token)
                            
            # add new tokens to the end of the input
            inputs = torch.cat([inputs, torch.stack(tokens, dim=0).view(-1, 1)], dim=-1)

            # Update attention mask
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

            new_inputs = {'input_ids': inputs, 'attention_mask': attn}

            if cache_implementation:
                cache_position = torch.cat([cache_position, cache_position[-1:] + 1], dim=-1)
                
        # some watermarked texts are fewer than others, make sure all are of size sequence length or new_tokens
        watermarked_texts = self.config.generation_tokenizer.batch_decode(inputs[:, self.prompt_tokens:], skip_special_tokens=True)

        return watermarked_texts
    # new end

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> dict:
        """Detect watermark in the text."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).numpy()[0]

        # Calculate the number of tokens to score, excluding the prefix
        num_scored = len(encoded_text) - self.config.prefix_length
        total_score = 0

        for i in range(self.config.prefix_length, len(encoded_text)):
            # Seed RNG with the prefix of the encoded text
            self.utils.seed_rng(encoded_text[:i])

            # Generate random numbers for each token in the vocabulary
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.rng)

            # Calculate score for the current token
            r = random_numbers[encoded_text[i]]
            total_score += log(1 / (1 - r))

        # Calculate p_value
        p_value = scipy.stats.gamma.sf(total_score, num_scored, loc=0, scale=1)

        # Determine if the computed score exceeds the threshold for watermarking
        is_watermarked = p_value < self.config.threshold

        # Return results based on the `return_dict` flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": p_value}
        else:
            return (is_watermarked, p_value)
        
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        """Get data for visualization."""

        # Encode the text into tokens using the configured tokenizer
        encoded_text = self.config.generation_tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).numpy()[0]

        # Initialize the list of values with None for the prefix length
        highlight_values = [None] * self.config.prefix_length

        # Calculate the value for each token beyond the prefix
        for i in range(self.config.prefix_length, len(encoded_text)):
            # Seed the random number generator using the prefix of the encoded text
            self.utils.seed_rng(encoded_text[:i])
            random_numbers = torch.rand(self.config.vocab_size, generator=self.utils.rng)
            r = random_numbers[encoded_text[i]]
            v = log(1 / (1 - r))
            v = self.utils._value_transformation(v)
            highlight_values.append(v)

        # Decode each token id to its corresponding string token
        decoded_tokens = [self.config.generation_tokenizer.decode([token_id]) for token_id in encoded_text]
        
        return DataForVisualization(decoded_tokens, highlight_values)
        
    