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

# =======================================================
# text_quality_analyzer.py
# Description: Analyze text quality using various metrics
# =======================================================

import math
import random
import re
import torch
import numpy as np
import sacrebleu
from utils.openai_utils import OpenAIAPI
from exceptions.exceptions import CodeExecutionError, InvalidAnswerError


class TextQualityAnalyzer:
    """Base class for text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str):
        pass


class DirectTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for direct text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str):
        pass


class ReferencedTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for referenced text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference):
        pass


class ExternalDiscriminatorTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for external discriminator text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text1: str, text2: str, description: str):
        pass


class PPLCalculator(DirectTextQualityAnalyzer):
    """Perplexity calculator for text quality analysis."""

    def __init__(self, model, tokenizer, device='cuda') -> None:
        """
            Initialize the perplexity calculator.

            Parameters:
                model: The language model for perplexity calculation.
                tokenizer: The tokenizer for the language model.
                device (str): The device to use for the calculation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def analyze(self, text: str):
        """Calculate the perplexity of the given text."""
        criterion = torch.nn.CrossEntropyLoss()
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        logits = self.model(torch.unsqueeze(encoded_text, 0), return_dict=True).logits[0]
        loss = criterion(logits[:-1], encoded_text[1:])
        ppl = torch.exp(loss)
        return ppl.item()
    
    # def analyze(self, text: str, stride: int = 512):
    #     """
    #     Calculate perplexity with sliding window for longer texts.
        
    #     Args:
    #         text (str): Input text
    #         stride (int): Stride for sliding window
        
    #     Returns:
    #         float: Perplexity score
    #     """
    #     self.model.eval()
    #     encodings = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
    #     seq_len = encodings.input_ids.size(1)
        
    #     nlls = []
    #     prev_end_loc = 0
    #     total_tokens = 0
        
    #     try:
    #         for begin_loc in range(0, seq_len, stride):
    #             end_loc = min(begin_loc + self.model.config.max_position_embeddings, seq_len)
    #             trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                
    #             input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
    #             target_ids = input_ids.clone()
    #             target_ids[:, :-trg_len] = -100  # ignore non-target tokens
                
    #             with torch.no_grad():
    #                 outputs = self.model(input_ids, labels=target_ids)

    #                 # loss is calculated using CrossEntropyLoss which averages over valid labels
    #                 # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
    #                 # to the left by 1.
    #                 neg_log_likelihood = outputs.loss
                
    #             nlls.append(neg_log_likelihood)
    #             total_tokens += trg_len
    #             prev_end_loc = end_loc
                
    #             if end_loc == seq_len:
    #                 break
                    
    #         # Calculate average negative log likelihood
    #         nll = torch.stack(nlls).mean()
            
    #         # Calculate perplexity
    #         ppl = torch.exp(nll)
            
    #         return ppl.item()
        
    #     except Exception as e:
    #         print(f"Error calculating perplexity: {str(e)}")
    #         return None
        
class LogDiversityAnalyzer(DirectTextQualityAnalyzer):
    """Log diversity analyzer for text quality analysis."""
    
    def __init__(self) -> None:
        super().__init__()

    def _eval_text(self, text: str, ngram: int):
        """Evaluate text to compute the number of unique and total n-grams."""
        tokens = text.split()
        ngram_set = set()
        total_ngrams = 0

        for i in range(len(tokens) - ngram + 1):
            ngram_set.add(" ".join(tokens[i:i + ngram]))
            total_ngrams += 1

        return len(ngram_set), total_ngrams

    def _eval_one_instance(self, text: str, ngram_list: list):
        """Evaluate a single text instance for multiple n-gram lengths."""
        results = {}
        for n in ngram_list:
            unique, total = self._eval_text(text, n)
            results[n] = {"unique": unique, "total": total}
        unique_tokens = set(text.split())
        return results, unique_tokens

    def analyze(self, text: str):
        """Analyze text to compute log diversity based on n-gram uniqueness."""
        ngram_list = [2, 3, 4]
        prediction_results = {n: {"unique": 0, "total": 0} for n in ngram_list}
        unique_token_set = set()

        stripped_text = text.strip()
        ngram_results, unique_tokens = self._eval_one_instance(stripped_text, ngram_list)

        unique_token_set.update(unique_tokens)

        for n in ngram_list:
            prediction_results[n]["unique"] += ngram_results[n]["unique"]
            prediction_results[n]["total"] += ngram_results[n]["total"]

        # Compute diversity scores for each n-gram length
        diversity_scores = [
            1 - (prediction_results[n]["unique"] / prediction_results[n]["total"])
            for n in ngram_list
        ]

        # Overall diversity is the product of individual n-gram diversities
        overall_diversity = (1 - diversity_scores[0] / 100) * (1 - diversity_scores[1] / 100) * (1 - diversity_scores[2] / 100)
        log_diversity = -math.log(max(1 - overall_diversity, math.exp(-20)))

        return log_diversity


class BLEUCalculator(ReferencedTextQualityAnalyzer):
    """BLEU calculator for text quality analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference: str):
        """Calculate the BLEU score of the given text with the reference."""
        b = sacrebleu.corpus_bleu([text], [[reference]]).score
        return b


class PassOrNotJudger(ReferencedTextQualityAnalyzer):
    """Pass or not judger for text quality analysis."""
    def __init__(self) -> None:
        pass

    def _check_correctness(self, prompt: str, completion: str, test: str, entry_point: str):
        """Check the correctness of the code.""" 
        check_program = (
            prompt + '\n' + completion + "\n" +
            test + "\n" +
            f"check({entry_point})"
        )
        # print(check_program)
        try:
            exec_globals = {}
            exec(check_program, exec_globals)
            return 1
        except BaseException as e:
            return 0

    def analyze(self, text: str, reference: dict):
        """Check if the text passes the correctness test."""
        passed = self._check_correctness(reference['task'], text, reference['test'], reference['entry_point'])
        return passed
    

class GPTTextDiscriminator(ExternalDiscriminatorTextQualityAnalyzer):
    """GPT text discriminator for text quality analysis."""

    def __init__(self, openai_model: str, task_description: str) -> None:
        """
            Initialize the GPT text discriminator.

            Parameters:
                openai_model (str): The OpenAI model to use for text discrimination.
                task_description (str): The description of the task for text discrimination.
        """
        self.openai_model = openai_model
        self.task_description = task_description
    
    def _get_query(self, text1: str, text2: str, question: str):
        """Get the query for text discrimination."""

        query = f"Task Description: {self.task_description}\n"
        query += f"Question: {question}\n"
        query += f"Answer 1: {text1}\n"
        query += f"Answer 2: {text2}\n"
        query += f"Which anwser is better? Only return a number."
        query += f"Return 1 if the first text is better, 2 if the second text is better, 0 if they are equal."
        return query

    def analyze(self, text1: str, text2: str, question: str):
        """Analyze the text to determine which one is better."""
        openai_util = OpenAIAPI(model=self.openai_model, temperature=0.2, 
                                system_content="You are a helpful assistant to determine which of the two answers is better based on the given task description.")
        query = self._get_query(text1, text2, question)
        answer = openai_util.get_result(query)
        # validate answer
        if answer not in ['0', '1', '2']:
            raise InvalidAnswerError
        return eval(answer)

class SinghZouJudge(ExternalDiscriminatorTextQualityAnalyzer):
    """GPT text discriminator for text quality analysis."""

    def __init__(self, openai_model: str, task_description: str) -> None:
        """
            Initialize the GPT text discriminator.

            Parameters:
                openai_model (str): The OpenAI model to use for text discrimination.
                task_description (str): The description of the task for text discrimination.
        """
        self.openai_model = openai_model
        self.task_description = task_description
    
    def _get_query(self, a_output: str, b_output: str, prompt: str):
        """Get the query for text discrimination."""

        query = """
        [Prompt]
        {prompt}
        
        [The Start of LLM A’s Answer]
        {a_output}
        [The End of LLM A’s Answer]
        
        [The Start of LLM B’s Answer]
        {b_output}
        [The End of LLM B’s Answer]""".format(prompt=prompt, a_output=a_output, b_output=b_output)
        return query
    
    # gpt-judger prompt and score extraction
    def gpt_judge(self, prompt, w_output, uw_output, is_randomized = None):

        if is_randomized is None:
            is_randomized = random.choice([True, False])
        
        # randomizing the order
        a_output, b_output = (uw_output, w_output) if is_randomized else (w_output, uw_output)
        num_creteria = 7
        system = """
        Please act as an impartial judge and evaluate the quality of the text completions provided by two large language models to the prompt displayed below. 
        Assess each response according to the criteria outlined, using a 1-5 Likert scale where 1 indicates strong disagreement or the lowest quality, and 5 indicates strong agreement or the highest quality. 
        After scoring each criterion, provide a short summary for each response, including specific examples that influenced your scoring. 
        Additionally, don't let the length of the responses influence your evaluation. 
        Be as objective as possible and ensure that the order in which the responses are presented does not affect your decision.
        
        Criteria:
        1. Relevance to the prompt
        2. Depth of detail
        3. Clarity of writing
        4. Coherence and logical flow
        5. Originality and insight
        6. Use of specific examples
        7. Accuracy of information
        
        Write the scores for both assistants to create two lists strictly following this format:

        "((A)): [list of scores]"
        "((B)): [list of scores]".

        At the very end of your response, declare your verdict by choosing one of the choices below, strictly following the given format:
        
        "[[A]]: [list of scores]" if assistant A is better, 
        "[[B]]: [list of scores]" if assistant B is better, or 
        "[[C]]" for a tie."""
        
        openai_util = OpenAIAPI(model=self.openai_model, temperature=0.2, system_content=system)
        judge_prompt = self._get_query(a_output, b_output, prompt)
        judge_output = openai_util.get_result(judge_prompt)
        
        # search for a tie first
        tie_pattern = r'\[\[C\]\]'
        tie_match = re.search(tie_pattern, judge_output)
        
        # extract the summaries now
        summary_pattern = r'\(\(([AB])\)\): (?:\[)?([5, 4, 3, 2, 1, ]+)(?:\])?'
        summaries_match = re.findall(summary_pattern, judge_output)
        
        if summaries_match:
            llm_1_output, llm_2_output = summaries_match
            llm_1, scores_1 = llm_1_output
            llm_2, scores_2 = llm_2_output
            scores_1 = eval(scores_1)
            scores_2 = eval(scores_2)
            if is_randomized:
                summaries = [{"Unwatermarked": (llm_1, scores_1)}, {"Watermarked": (llm_2, scores_2)}]
            else:
                summaries = [{"Watermarked": (llm_1, scores_1)}, {"Unwatermarked": (llm_2, scores_2)}]
        else:
            summaries = [{"Unwatermarked": ("Model Failure", (0,)*num_creteria)}, {"Watermarked": ("Model Failure", (0)*num_creteria)}]
        
        if tie_match:
            judge_choice = "C"
            final_verdict = "Tie"
            scores = []
        else:
            # pattern to match the verdict and the scores for A or B
            verdict_pattern = r'\[\[([AB])\]\]: (?:\[)?([5, 4, 3, 2, 1, ]+)(?:\])?'
            matches = re.findall(verdict_pattern, judge_output)
            
            if matches:
                # extract the last match which will have the choice and the corresponding scores
                judge_choice, scores_str = matches[-1]
                
                # remove square brackets if they exist, strip whitespace, and split by comma
                # new begin
                scores = eval(scores_str)
                # scores_str = scores_str.replace('[', '').replace(']', '').strip()
                # scores = [float(score) for score in scores_str.split(',')]
                
                # determine verdict based on the judge choice
                if is_randomized:
                    final_verdict = "Unwatermarked" if judge_choice == 'A' else "Watermarked"
                else:
                    final_verdict = "Watermarked" if judge_choice == 'A' else "Unwatermarked"
            else:
                final_verdict = "Model Failure"
                scores = []
        
        return summaries, final_verdict, scores, judge_output
    
    def analyze(self, text1: str, text2: str, prompt: str):
        """Analyze the text to determine which one is better."""
        summaries, final_verdict, scores, judge_output = self.gpt_judge(prompt, text1, text2)
        return [summaries, final_verdict, scores, judge_output]
        
    # function to tally judger results
    def extract_and_count_choices(self, data_list, category):
        
        # extract all 'category' entries from the list of dictionaries
        choices = [entry[category] for entry in data_list]
        
        # count occurrences of each 'judge_choice'
        count = {}
        for choice in choices:
            if choice in count:
                count[choice] += 1
            else:
                count[choice] = 1

        count_norm = count.copy()
        for choice in count_norm:
            count_norm[choice] /= len(choices)
            count_norm[choice] *= 100
            count_norm[choice] = np.round(count_norm[choice], 3)
            
        return count, count_norm

    # A helper function to convert probabilities to binary outcomes based on a threshold
    def prob_to_binary(self, probabilities, threshold=0.5):
        return (probabilities > threshold).astype(int)