import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import anthropic
from openai import OpenAI

import tqdm as tqdm
import random
from typing import List, Dict, Any
import time
import json
import re
from argparse import ArgumentParser
import copy
from collections import defaultdict

from exceptions.exceptions import OpenAIModelConfigurationError
from utils.openai_utils import OpenAIAPI

def judge_different_texts():
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=str,
        default="42 123",
        help="The seeds to use for the OpenAI API.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="The model to use for the OpenAI API.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature to use for the OpenAI API.",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=2,
        help="The number of tests to run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/malghanim/ml_security/cross-lingual-watermarking/outputs/llm-as-a-judge/",
        help="The directory to save the output files.",
    )
    parser.add_argument(
        "--dir_path",
        type=str,
        default="/home/malghanim/ml_security/cross-lingual-watermarking/data/",
        help="The directory containing the data files.",
    )
    parser.add_argument(
        "--perturbation_type",
        type=str,
        default="translation",
        help="The type of perturbation to use: translation or paraphrase.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="en ar zh de fa ja id",
        help="The languages to use for the OpenAI API.",
    )
    args = parser.parse_args()

    # Set the parameters
    model = args.model
    temperature = args.temperature
    num_test = args.num_test
    dir_path = args.dir_path
    output_dir = args.output_dir
    seeds = args.seeds.split()
    perturbation_type = args.perturbation_type
    seeds = [int(seed) for seed in seeds]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    languages = args.languages.split()
    
    texts = {}
    for lang in languages:
        with open(f'{dir_path}/{perturbation_type}_versions/gpt-3.5-turbo/{lang}_c4.jsonl', 'r') as f:
            texts.update({lang: json.load(f)})

    en_file = dir_path + 'c4_500_processed.json'

    system_content = """
    Please act as an impartial judge and evaluate the quality of the text completions provided in these languages but not necessarily in the same order [English, Arabic, Chinese, German, Indonesian, Farsi, Japanese]. 
        Assess each text according to the criteria outlined, using a likert scale from 1 to 5, where 1 is the lowest and 5 is the highest quality.
        Don't let the length of the responses or the language influence your evaluation. 
        Be as objective as possible and ensure that the order in which the responses are presented does not affect your decision.
        
        Criteria:
        1. Depth of detail
        2. Clarity of writing
        3. Coherence and logical flow
        4. Originality and insight
        5. Use of specific examples
        6. Accuracy of information
        
        Start with a brief statement about which text you think is superior. Then, for each text and criterion, provide a score, followed by a brief justification for that score.
        At the very end of your response, declare your verdict by choosing the winner language, strictly following the given format:
        
        "[[EN]]: [list of scores]" if English text is better,
        "[[AR]]: [list of scores]" if Arabic text is better,
        "[[ZH]]: [list of scores]" if Chinese text is better,
        "[[DE]]: [list of scores]" if German text is better,
        "[[FA]]: [list of scores]" Farsi text is better,
        "[[ID]]: [list of scores]" if Indonesian text is better,
        "[[JA]]: [list of scores]" if Japanese text is better, or
        "[[TIE]]" if all texts are equally good.
        """
    if perturbation_type == 'paraphrase':
        system_content = """
        Please act as an impartial judge and evaluate the quality of the text completions provided in one of these languages but not necessarily in the same order [English, Arabic, Chinese, German, Indonesian, Farsi, Japanese]. 
            Assess each text according to the criteria outlined, using a likert scale from 1 to 5, where 1 is the lowest and 5 is the highest quality.
            Don't let the length of the responses or the language influence your evaluation. 
            Be as objective as possible and ensure that the order in which the responses are presented does not affect your decision.
            
            Criteria:
            1. Depth of detail
            2. Clarity of writing
            3. Coherence and logical flow
            4. Originality and insight
            5. Use of specific examples
            6. Accuracy of information
            
            Start with a brief statement about which text you think is superior. Then, for each text and criterion, provide a score, followed by a brief justification for that score.
            At the very end of your response, declare your verdict by choosing the winner text, strictly following the given format:
            
            "[[A]]: [list of scores]" if text A is better,
            "[[B]]: [list of scores]" if text B is better, or
            "[[TIE]]" if both texts are equally good.
        """
    openai_util = OpenAIAPI('gpt-4o-mini', 0, system_content=system_content)

    for seed in seeds:
        random.seed(seed)
        
        print(f"Running tests with seed {seed}...")
        copied_texts = copy.deepcopy(texts)
        for language in languages:
            random.shuffle(copied_texts[language])
            # put texts from each language in a tuple
        
        if perturbation_type == 'translation':
            index = 0
            outputs = []
            with tqdm.tqdm(total=num_test, desc="Processing tests") as pbar:
                while index < num_test:
                    # shuffle the texts from the languages list and save the new langauges order
                    languages = random.sample(languages, len(languages))                
                    texts_list = [copied_texts[lang][index]['natural_text'] for lang in languages]

                    # TODO: recently added. check if it works correctly
                    prompt = [f"""{i+1}. {texts_list[i]}""" for i in range(len(texts_list))]
                    prompt = "\n".join(prompt)
                    prompt = f"""Texts:
                    {prompt}
                    """
                    
                    result = openai_util.get_result(prompt)
                    # print(f"Result for index {index}: {result}")
                    tie_pattern = re.compile(r'\[\[TIE\]\]')
                    tie_match = tie_pattern.search(result)
                    final_verdict = None
                    if tie_match:
                        final_verdict = "TIE"
                    else:
                        pattern = re.compile(r'\[\[(EN|AR|ZH|DE|FA|JA|ID)\]\]')
                        matches = re.findall(pattern, result)
                        if matches:
                            final_verdict = matches[-1]
                        else:
                            print(f"Error: No match found in result: {result}")
                            final_verdict = "Model Failure"

                    # TODO: recently added. check if it works correctly
                    texts_outputs = {'index': index, 'shuffled_languages': languages}
                    texts_outputs.update({f'text{i+1}': texts_list[i] for i in range(len(texts_list))})
                    texts_outputs.update({'result': result, 'verdict': final_verdict})
                    outputs.append(texts_outputs)
                    index += 1
                    pbar.update(1)

            with open(f'{output_dir}/gpt_bias_results_seed_{seed}.jsonl', 'a') as f:
                json.dump(outputs, f, indent=4, ensure_ascii=False)
            print(f"Results saved to {output_dir}/gpt_bias_results_seed_{seed}.jsonl")

        elif perturbation_type == 'paraphrase':
            index = 0
            outputs = defaultdict(list)
            with tqdm.tqdm(total=num_test, desc="Processing tests") as pbar:
                while index < num_test:
                    inner_outputs = []

                    # shuffle the texts from the languages list and save the new langauges order to avoid order bias
                    languages = random.sample(languages, len(languages))
                    texts_list = [(copied_texts[lang][index]['natural_text'],
                                   copied_texts[lang][index]['perturbed_natural_text']) for lang in languages] # use a list to ensure the order of languages
                    is_randomized = random.choice([True, False])
                    if is_randomized:
                        prompts = [f"""Texts:
                                A. {t[1]}
                                B. {t[0]}
                                """ for t in texts_list]
                    else:
                        prompts = [f"""Texts:
                                A. {t[0]}
                                B. {t[1]}
                                """ for t in texts_list]
                    
                    results = []
                    for i, prompt in enumerate(prompts):
                        result = openai_util.get_result(prompt)

                        # print(f"Result for index {index}: {result}")
                        tie_pattern = re.compile(r'\[\[TIE\]\]')
                        tie_match = tie_pattern.search(result)
                        final_verdict = None
                        if tie_match:
                            final_verdict = "TIE"
                        else:
                            pattern = re.compile(r'\[\[(A|B)\]\]')
                            matches = re.findall(pattern, result)
                            if matches:
                                judge_choice = matches[-1]
                                if is_randomized:
                                    final_verdict = "perturbed_text" if judge_choice == 'A' else "natural_text"
                                else:
                                    final_verdict = "natural_text" if judge_choice == 'A' else "perturbed_text"
                            else:
                                print(f"Error: No match found in result: {result}")
                                final_verdict = "Model Failure"
                        inner_outputs.append({
                            'index': index,
                            'shuffled_languages': languages,
                            'natural_text': texts_list[i][0],
                            'perturbed_natural_text': texts_list[i][1],
                            'result': result,
                            'verdict': final_verdict
                        })
                    
                    index += 1
                    for i, lang in enumerate(languages):
                        outputs[lang].append(inner_outputs[i])
                    pbar.update(1)
            # Save the results to a file
            for lang, lang_outputs in outputs.items():
                with open(f'{output_dir}/{lang}_gpt_bias_results_seed_{seed}_paraphrase.jsonl', 'a') as f:
                    json.dump(lang_outputs, f, indent=4, ensure_ascii=False)
                print(f"Results saved to {output_dir}/{lang}_gpt_bias_results_seed_{seed}_paraphrase.jsonl")
           
    print("All done!")