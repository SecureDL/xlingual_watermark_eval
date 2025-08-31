import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import copy
import random
import argparse
import json


# openai
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

langs_acronyms = {
    'Arabic': 'ar',
    'English': 'en',
    'French': 'fr',
    'German': 'de',
    'Spanish': 'es',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Indonesian': 'id',
    'Persian': 'fa',
    'Hebrew': 'he',
    'Hindi': 'hi',
    "Turkish": "tr",
}

# write code to paraphrase a text using OpenAI API
def paraphrase_chunk(chunk, model='gpt-3.5-turbo', dest_language='Arabic'):
    response = client.responses.create(
    model=model,
    input=[
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": f"""Please act as a professional writer and rewrite this {dest_language} text according to the following criteria.
                    
                    Criteria:
                    1. Do not change abbreviations, names, or commands.
                    2. Do not include any unrelated information in the paraphrase.
                    3. Do not translate to English or any other language. Just paraphrase the text in {dest_language}.
                    3. Change as many words as possible in the paraphrased version while keeping the meaning of the text."""
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": chunk
                }
            ]
        }
    ],
    text={
        "format": {
            "type": "text"
            }
        },
    reasoning={},
    tools=[],
    temperature=0.5,
    max_output_tokens=2048,
    top_p=1,
    )
    # breakpoint()
    result = response.output[0].content[0].text.strip()
    result = result.replace('"""', '') # remove the double quotes, as we used them to surround the text
    return result

# This code was taken from the OpenAI API documentation
def translate_chunk(chunk, model='gpt-3.5-turbo',
                    dest_language='Arabic',
                    sample_translation=("Why everything in the universe experience aging or deteriorate like the earth or human beings... Except atoms?"
                                        "لماذا كل شيء في الكون يتعرض للشيخوخة أو التدهور مثل الأرض أو البشر... إلا الذرات؟",
                                        "Atoms also experience this. Larger atoms experience radioactive decay. Smaller atoms still experience decreasing entropy. They lose energy over time. The time scale is large and the energy loss is tiny so we do not observe it as easily as radioactive decay. Not ELI5 but Look up the second law of thermodynamics and the heat death of the universe."
                                        "تختبر الذرات هذه الظاهرة أيضًا. الذرات الأكبر حجمًا تختبر التحلل الإشعاعي. الذرات الأصغر حجمًا ما زالت تختبر نقصان الإنتروبيا. تفقد الطاقة مع مرور الزمن. رغم أنّ الفترة الزمنية كبيرة وفقدان الطاقة ضئيل، لذا لا نلاحظه بسهولة كما نلاحظ التحلل الإشعاعي. لا أشرح لك كمبتدئ ولكن ابحث عن القانون الثاني للديناميكا الحرارية ونهاية الكون الحرارية.",
                                        "Why snitches get stitches?"
                                        "لماذا يتعرض الواشون للأذى؟",)
                    ):
    
    response = client.responses.create(
    model=model,
    input=[
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": f"""Please act as a professional translator and translate the following text into {dest_language}.
                    Translate the chunks according to the following criteria.
                    
                    Criteria:
                    1. Do not translate abbreviations, names, or commands.
                    2. Translate quoted texts with extra care.
                    3. Do not include any unrelated information in the translation.
                    4. Remove any unrelated extra characters from original text before translating. For example, if there 'Q:', 'q:', 'A:', or 'a:', then remove them from translations."""
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": chunk
                }
            ]
        }
    ],
    text={
        "format": {
            "type": "text"
            }
        },
    reasoning={},
    tools=[],
    temperature=0.5,
    max_output_tokens=2048,
    top_p=1,
    )
    # breakpoint()
    result = response.output[0].content[0].text.strip()

    result = result.replace('"""', '') # remove the double quotes, as we used them to surround the text
    return result
    

def translate_lfqa_openGen_dataset(data=None, model='gpt-3.5-turbo',
                           dest_language='Arabic', num_test=2,
                           dest_dir="",
                           seed=42,
                           task="lfqa",
                           length=100,
                           tokenizer=None):
    # shuffled_data = copy.deepcopy(data)
    # random.seed(seed)
    # random.shuffle(shuffled_data)
    # shuffled_data = shuffled_data[:num_test]
    translated_data = []
    for idx, cur_data in tqdm(enumerate(data)):#, total=min(len(shuffled_data), num_test)):
        if len(translated_data) == num_test:
            break
        if "gold_completion" not in cur_data and 'targets' not in cur_data:
            continue
        elif "gold_completion" in cur_data:
            prefix = cur_data['prefix'].replace("@", "")
            target = cur_data['gold_completion'].replace("@", "")
        else:
            prefix = cur_data['prefix'].replace("@", "")
            target = cur_data['targets'][0].replace("@", "")
        # tokens = tokenizer.encode(target, return_tensors='pt', truncation=True)[0]
        # if len(tokens) < length + 20: # 20 buffer tokens for different languages
        #     continue
        prefix = process_spaces(prefix)
        target = process_spaces(target)
        translated_prefix = process_spaces(translate_chunk(prefix, model=model, dest_language=dest_language))
        translated_target = process_spaces(translate_chunk(target, model=model, dest_language=dest_language))
        translated_data.append(
            dict(**cur_data, prompt=translated_prefix, natural_text=translated_target)
        )
            
    with open(os.path.join(dest_dir, f"{langs_acronyms[dest_language]}_{task}.jsonl"), 'w') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()

def read_lfqa_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]
    
# read json file where dicts are in a list
def read_json_file(filename):
    with open(filename, "r") as f:
        return json.load(f)
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--data_file", type=str, default="", help="The directory of the dataset")
    args.add_argument("--output_dir", type=str, default="", help="The directory to save the translated dataset")
    args.add_argument("--num_examples", type=int, default=2, help="The number of test samples to translate")
    args.add_argument("--seeds", type=str, default="42 100", help="The seeds to use for the translation")
    args.add_argument("--task", type=str, default="lfqa", help="The task to translate")
    args.add_argument("--tokenizer", type=str, default="CohereForAI/c4ai-command-r7b-12-2024", help="The tokenizer to use")
    args.add_argument("--dest_language", type=str, default="Arabic", help="The destination language to translate to")
    args.add_argument("--perturbation_type", type=str, default="translation", help="The type of perturbation to use: translation or paraphrase")
    args.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The model to use for translation")
    args.add_argument("--T", type=int, default=200, help="The length of the text to consider for translation")
    args = args.parse_args()
    # nicely print the arguments
    print("Arguments:")
    print(f"  data_file: {args.data_file}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  num_examples: {args.num_examples}")
    print(f"  seeds (not really used): {args.seeds}")
    print(f"  task: {args.task}")
    print(f"  dest_language: {args.dest_language}")
    print(f"  model: {args.model}")
    print(f"  tokenizer: {args.tokenizer}")
    print(f"  T: {args.T}")


    seeds = args.seeds.split(" ") # for lfqa and opengen
    seeds = [int(seed) for seed in seeds]
    data_file = args.data_file
    output_dir = args.output_dir
    num_examples = args.num_examples
    perturbation_type = args.perturbation_type
    T = args.T

    tokenizer_path = args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id # recently added

    input_examples = None
    if "lfqa" in args.task or "opengen" in args.task:
        input_examples = read_lfqa_file(data_file)
        translate_lfqa_openGen_dataset(data=input_examples,
                                        model=args.model,
                                        dest_dir=output_dir,
                                        dest_language=args.dest_language,
                                        num_test=num_examples,
                                        task=args.task,
                                        length=T,
                                        tokenizer=tokenizer)
    
    elif "c4" in args.task:
        # import and translate 500 c4 dataset with different seeds
        # load c4 streamlined from hf datasets
        collected_examples = []
        dataset = None
        dataset = read_json_file(data_file)
        for idx in range(num_examples):
            example = dataset[idx]
            prompt = process_spaces(example['prompt'])
            natural_text = process_spaces(example['natural_text'])
            perturbed_prompt = None
            perturbed_natural_text = None
            if perturbation_type == "translation":
                perturbed_prompt = translate_chunk(prompt, model=args.model,
                                                    dest_language=args.dest_language)
                perturbed_natural_text = translate_chunk(natural_text, model=args.model,
                                                        dest_language=args.dest_language)
            
            elif perturbation_type == "paraphrase":
                perturbed_natural_text = paraphrase_chunk(natural_text, model=args.model, dest_language=args.dest_language)
            
            if perturbed_prompt:
                process_spaces(perturbed_prompt)
            process_spaces(perturbed_natural_text)
            
            if perturbation_type == "translation":
                collected_examples.append({
                    "index": idx,
                    "prompt": perturbed_prompt,
                    "natural_text": perturbed_natural_text
                    })
            elif perturbation_type == "paraphrase":
                collected_examples.append({
                    "index": idx,
                    "natural_text": natural_text,
                    "perturbed_natural_text": perturbed_natural_text
                    })
        with open(os.path.join(output_dir, f"{langs_acronyms[args.dest_language]}_c4.jsonl"), 'w') as f:
            json.dump(collected_examples, f, ensure_ascii=False, indent=4)