### Pytorch ###
import torch

### General ###
import json
import argparse
import os
import gc
import copy
import sys
from collections import defaultdict
import time
from translate import Translator
from tqdm import tqdm
from requests.exceptions import ReadTimeout

### Scientific ###
import numpy as np
from scipy.spatial.distance import cosine
import torch.nn.functional as F


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

### Watermarking ###
from watermark.auto_watermark import AutoWatermark

# attacks
from evaluation.tools.text_editor import (
    TruncatePromptTextEditor,
    TruncateTaskTextEditor,
    WordDeletion,
    SynonymSubstitution,
    ContextAwareSynonymSubstitution,
    GPTParaphraser,
    DipperParaphraser,
    AnthropicTranslationTextEditor,
    TranslationTextEditor,
    OPUSTranslationTextEditor,
)

# Quality analysis
from evaluation.dataset import (
    C4Dataset,
    WMT16DE_ENDataset,
    HumanEvalDataset,
    )
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from evaluation.tools.text_quality_analyzer import (
    PPLCalculator,
    LogDiversityAnalyzer,
    BLEUCalculator,
    PassOrNotJudger,
    SinghZouJudge,
    )
from evaluation.pipelines.detection import (
    WatermarkedTextDetectionPipeline,
    UnWatermarkedTextDetectionPipeline,
    DetectionPipelineReturnType,
    )
from evaluation.pipelines.quality_analysis import (
    DirectTextQualityAnalysisPipeline,
    ReferencedTextQualityAnalysisPipeline,
    ExternalDiscriminatorTextQualityAnalysisPipeline,
    QualityPipelineReturnType,
    )
from utils.utils import process_spaces


### Huggingface ###
from utils.transformers_config import TransformersConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MarianTokenizer,
    MarianMTModel,
    BitsAndBytesConfig,
    LlamaTokenizer,
    BertTokenizer,
    BertForMaskedLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# Remove this function and use the one in the text_editor.py
def get_mt_model_tokenizer(src_lang, tar_lang):
    mt_model_name = None
    if src_lang == "english" and tar_lang == "arabic":
        mt_model_name = "Helsinki-NLP/opus-mt-en-ar"
    elif src_lang == "arabic" and tar_lang == "english":
        mt_model_name = "Helsinki-NLP/opus-mt-ar-en"
    elif src_lang == "english" and tar_lang == "German":
        mt_model_name = "Helsinki-NLP/opus-mt-en-de"
    elif src_lang == "german" and tar_lang == "english":
        mt_model_name = "Helsinki-NLP/opus-mt-de-en"
    elif src_lang == "english" and tar_lang == "chinese":
        mt_model_name = "Helsinki-NLP/opus-mt-en-zh"
    elif src_lang == "chinese" and tar_lang == "english":
        mt_model_name = "Helsinki-NLP/opus-mt-zh-en"
    elif src_lang == "english" and tar_lang == "Japanese":
        mt_model_name = "Helsinki-NLP/opus-mt-en-ja"
    elif src_lang == "japanese" and tar_lang == "english":
        mt_model_name = "Helsinki-NLP/opus-mt-ja-en"
    elif src_lang == "english" and tar_lang == "vietnamese":
        mt_model_name = "Helsinki-NLP/opus-mt-en-vi"
    elif src_lang == "vietnamese" and tar_lang == "english":
        mt_model_name = "Helsinki-NLP/opus-mt-vi-en"
    elif src_lang == "english" and tar_lang == "indonesian":
        mt_model_name = "Helsinki-NLP/opus-mt-en-id"
    elif src_lang == "indonesian" and tar_lang == "english":
        mt_model_name = "Helsinki-NLP/opus-mt-id-en"
    else:
        raise ValueError(f"Unsupported language pair: {src_lang} to {tar_lang}")
    mt_tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
    mt_model = MarianMTModel.from_pretrained(mt_model_name).to(device)
    return mt_model, mt_tokenizer

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    results = defaultdict(dict)

    parser = argparse.ArgumentParser(description="Experiment Settings")
    parser.add_argument('--experiment',default="original",type=str, help="One of original, attack, or quality checks")
    # Original args
    parser.add_argument('--method',default="transform",type=str)
    parser.add_argument('--task',default="c4",type=str)
    parser.add_argument('--model_path',default="facebook/opt-1.3b",type=str)
    parser.add_argument('--data_dir',default=None,type=str)
    parser.add_argument('--is_processed_prompts', default=False, type=str2bool, help="Whether the prompts are already processed.")
    parser.add_argument('--save',default="",type=str)
    parser.add_argument('--batch_size',default=1,type=int)
    parser.add_argument('--language',default="english",type=str)
    parser.add_argument("--output_dir", type=str, default="./outputs", help="The unique name for the run.",)
    parser.add_argument('--truncate_vocab',default=8,type=int)
    parser.add_argument('--num_examples',default=20,type=int)
    parser.add_argument('--T',default=200,type=int) # length of the text in which we are aiming to detect the watermark
    parser.add_argument('--prompt_tokens',default=50,type=int)
    parser.add_argument('--buffer_tokens',default=20,type=int)
    parser.add_argument('--delta',default=2.0,type=float)
    parser.add_argument('--gamma',default=0.5,type=float)
    parser.add_argument('--instruction_following', default=False, type=str2bool, help="Whether to use instruction following models.")
    parser.add_argument('--temperature', default=0.7, type=float, help="Temperature for sampling.")

    # Quality args
    parser.add_argument(
            "--oracle_model_name",
            type=str,
            default="EleutherAI/gpt-j-6B",
            help="PPL scoring, or oracle model, path to pretrained model or model identifier from huggingface.co/models.",
        )
    parser.add_argument("--quality_analyzer", type=str, default="PPL", help="The quality analyzer to use. One of PPL, GPTDiscriminator.")

    # Attack args
    parser.add_argument('--rt_translate', action='store_true')
    parser.add_argument('--translate', default=False, type=str2bool)
    parser.add_argument('--target_language',default="arabic",type=str)
    parser.add_argument('--translation_aware', default=False, type=str2bool, help="Use equal close portions of English and Arabic tokens.")
    parser.add_argument('--adversary_scenario',default="",type=str)
    parser.add_argument('--mapping_name', default=None, type=str)
    parser.add_argument('--deletion',default=0.0,type=float)
    parser.add_argument('--insertion',default=0.0,type=float)
    parser.add_argument('--substitution',default=0.0,type=float)
    parser.add_argument('--paraphrase', default=False, type=str2bool)
    
    args = parser.parse_args()
    results['args'] = copy.deepcopy(args)

    print(args)
    gamma = args.gamma
    delta = args.delta
    temperature = args.temperature
    method = args.method
    oracle_model_path = args.oracle_model_name #"FreedomIntelligence/AceGPT-7B"
    model_path = args.model_path # "bigscience/bloomz-7b1-mt", "inceptionai/jais-family-6p7b", 
    output_dir = args.output_dir
    data_dir = args.data_dir
    task = args.task
    is_processed_prompts = args.is_processed_prompts
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert method in ['KGW', 'Unigram', 'SWEET', 'EWD', 'SIR', 'XSIR', 'UPV', 'EXP', 'EXPEdit', 'SynthID', 'PF']

    T = args.T                                # length of generated watermarked text
    num_examples = args.num_examples                  # number of examples to generate
    batch_size = args.batch_size
    prompt_tokens = args.prompt_tokens        # number of tokens in the prompt
    n_batches = int(np.ceil(num_examples / batch_size))  # number of batches
    max_new_tokens = T    # number of tokens to generate
    buffer_tokens = args.buffer_tokens

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id # recently added

    # set padding side to left for decoder-only models.
    # tokenizer.padding_side = "left"
    
    # if "r7b" in model_path.lower():
        # use bfloat16 for r7b models
        # model = model.bfloat16()
        # tokenizer.pad_token = "[PAD]"
        # tokenizer.padding_side = "left"

    # Transformers config
    model_prefix = "jais"
    if "baichuan" in model_path:
        model_prefix = "baichuan"
    elif "sail" in model_path:
        model_prefix = "sail"
    elif "qwen" in model_path:
        model_prefix = "qwen"
    elif "r7b" in model_path:
        model_prefix = "r7b"
    change_gen_config = False
    if "acegpt" in model_path.lower():
        change_gen_config = True
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    draft_model = None
    draft_tokenizer = None
    enhancement_kwargs = {}
    
    if model_prefix == "r7b":
        # make half precision, probabilities of next scores gives inf or nan if we don't use half
        model = model.half()
        
    if change_gen_config:
        if hasattr(model,"generation_config"):
            model.generation_config.do_sample = True
            model.generation_config.temperature = temperature
            model.generation_config.top_k = 0
            model.generation_config.top_p = 0.9
    
    
    gen_kwargs = {
        "max_new_tokens": T,
        "min_length": T + prompt_tokens, # 250
        "do_sample": True,
        "no_repeat_ngram_size": 4,
        "temperature": temperature, # recently added
    }
    transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            device=device,
                                            **gen_kwargs)
    experiment = args.experiment
    language = args.language

    mt_model = None
    mt_tokenizer = None
    adversary_scenario = args.adversary_scenario
    if adversary_scenario == "naive":
        src_lang = language.lower()
        tar_lang = args.target_language.lower()
        if experiment == "backtrans_attack":
            src_lang, tar_lang = tar_lang, src_lang
        src_lang_flag = False
        mt_model_name = None
        mt_model, mt_tokenizer = get_mt_model_tokenizer(src_lang, tar_lang)

    translation_aware = args.translation_aware
    english_tokens = []
    other_lang_tokens = []
    unknown_tokens = []
    mapping_clusters_to_vocab = {}
    all_token_ids = list(tokenizer.get_vocab().values())
    vocab = all_token_ids

    truncate_vocab = 8
    vocab_size = len(all_token_ids)
    eff_vocab_size = vocab_size - truncate_vocab

    method_config_file = None
    if method == "KGW" or method == "Unigram" or method == "EWD":
        method_config_file = f'config/{method}_{gamma}_{delta}.json'
    elif method == "XSIR" or method == "SIR":
        method_config_file = f'config/{method}_{delta}.json'
    elif method == "EXP" or method == "EXPGumbel":
        method_config_file = f'config/{method}.json'
        with open(method_config_file) as f:
            method_config = json.load(f)
        method_config.update({"sequence_length": T})
        with open(method_config_file, 'w') as f:
            json.dump(method_config, f, indent=4)
    else: # SynthID or PF # different sampling or decoding methods
        method_config_file = f'config/{method}.json'
    assert method_config_file is not None, f"Method config file not found for {method}"

    kwargs = {"prompt_tokens": prompt_tokens, "new_tokens": T,
              "mapping_name": args.mapping_name, # for XSIR
              "mapping": mapping_clusters_to_vocab,
              "vocab": vocab}
    
    myWatermark = AutoWatermark.load(f'{method}', algorithm_config=method_config_file, transformers_config=transformers_config, **kwargs)
    dataset_path = os.path.join(args.output_dir, f"{args.save}.json")
    my_dataset = None

    if experiment == "original":
        generate_watermarked = lambda prompt,instruction=False, draft_model=None, draft_tokenizer=None : \
            myWatermark.generate_watermarked_text(
                prompt, instruction=instruction, draft_model=draft_model, draft_tokenizer=draft_tokenizer
            )
        generate_unwatermarked = lambda prompt,instruction=False, draft_model=None, draft_tokenizer=None : \
            myWatermark.generate_unwatermarked_text(
                prompt, instruction=instruction, draft_model=draft_model, draft_tokenizer=draft_tokenizer
            )
        detect_watermark = lambda text: myWatermark.detect_watermark(text) 

        def load_dataset_with_retry(dataset_name, config_name=None, split=None, streaming=False, retries=5, delay=5):
            for attempt in range(retries):
                try:
                    if config_name:
                        return load_dataset(dataset_name, config_name, split=split, streaming=streaming)
                    else:
                        return load_dataset(dataset_name, split=split, streaming=streaming)
                except Exception:
                    if attempt < retries - 1:
                        print(f"ReadTimeout or HfHubHTTPError occurred. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        raise

        if data_dir is not None:
            # read json file
            with open(data_dir) as f:
                dataset = json.load(f)
        else: # If no data provided use c4 dataset from huggingface
            if language.lower() == "english":
                if task == "c4":
                    dataset = load_dataset_with_retry("allenai/c4", "realnewslike", split="train", streaming=True)

            elif language.lower() == "german":
                if task == "c4":
                    dataset = load_dataset_with_retry("allenai/c4", "de", split="train", streaming=True)

            elif language.lower() == "arabic":
                if task == "c4":
                    dataset = load_dataset_with_retry("allenai/c4", "ar", split="train", streaming=True)

            elif language.lower() == "chinese":
                if task == "c4":
                    dataset = load_dataset_with_retry("allenai/c4", "zh", split="train", streaming=True)

            elif language.lower() == "indonesian":
                if task == "c4":
                    dataset = load_dataset_with_retry("allenai/c4", "id", split="train", streaming=True)

            elif language.lower() == "japanese":
                if task == "c4":
                    dataset = load_dataset_with_retry("allenai/c4", "ja", split="train", streaming=True)

            elif language.lower() == "persian":
                if task == "c4":
                    dataset = load_dataset_with_retry("allenai/c4", "fa", split="train", streaming=True)
            elif language.lower() == "hindi":
                if task == "c4":
                    dataset = load_dataset_with_retry("allenai/c4", "hi", split="train", streaming=True)
            elif language.lower() == "turkish":
                if task == "c4":
                    dataset = load_dataset_with_retry("allenai/c4", "tr", split="train", streaming=True)

            else: # raise not supported language error
                raise ValueError(f"Unsupported language: {language}")
        
        ds_iterator = iter(dataset)
        item = 0
        complete_examples = []
        complete_examples_c4 = []
        prompts = []
        natural_completions = []
        # if we are using c4 directly use the following code. Otherwise.
        # we have the prompt ready in the data_dir file. We just need to
        # save the prompts and completions in their respective lists.
        if data_dir is None:
            while item < num_examples:
                example = next(ds_iterator)
                text = example["text"]
                text = process_spaces(text)
                tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
                if len(tokens) < prompt_tokens + T:
                    continue
                complete_examples_c4.append({
                    "index": item,
                    "text": text,
                })
                complete_examples.append(tokens)
                prompt = tokens[-(T+prompt_tokens):-T]
                completion = tokens[-T:]
                prompts.append(prompt)
                natural_completions.append(tokenizer.decode(completion, skip_special_tokens=True))
                item += 1
        else:
            if is_processed_prompts: # prompts and completions already processed from old translations or experiments
                prompt_key = "prompt"
                completion_key = "natural_text"
                if task == "lfqa" and language.lower() == "english":
                    prompt_key = "prefix"
                    completion_key = "gold_completion"

                while item < num_examples:
                    example = next(ds_iterator)
                    prompt = example[prompt_key]
                    completion = example[completion_key]
                    prompt = tokenizer.encode(prompt, return_tensors='pt', truncation=True, padding="max_length", max_length=prompt_tokens)[0]
                    prompts.append(prompt)
                    natural_completions.append(completion)
                    complete_examples.append(prompts + natural_completions)
                    item += 1
            else: # dataset save to disk and we read text from their - no pre-processing has been done
                while item < num_examples:
                    example = next(ds_iterator)
                    text = example["text"]
                    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
                    complete_examples.append(tokens)
                    prompt = tokens[-(T+prompt_tokens):-T]
                    completion = tokens[-T:]
                    prompts.append(prompt)
                    natural_completions.append(tokenizer.decode(completion, skip_special_tokens=True))
                    item += 1
        
        prompts = torch.vstack(prompts)

        unwatermarked_samples = []
        watermarked_samples = []
        tokenized_unwm_samples = []
        tokenized_wm_samples = []
        decoded_prompts_batched = []
        
        # first issue: input is not batched so we might need to batch it.
        enhancement_kwargs.update(
            {
                'instruction': args.instruction_following,
            }
        )
        for batch in range(n_batches):
            idx = torch.arange(batch * batch_size,min(num_examples,(batch + 1) * batch_size))
            
            decoded_prompts = tokenizer.batch_decode(prompts[idx], skip_special_tokens=True) #[tokenizer.decode(p, skip_special_tokens=True) for p in prompts[idx]]
            decoded_prompts_batched.append(decoded_prompts)
            
            output = generate_unwatermarked(decoded_prompts, **enhancement_kwargs)
            
            encoded_output = tokenizer(output, return_tensors='pt', truncation=True, padding='max_length', max_length=T)['input_ids']
            unwatermarked_samples.append(output)
            tokenized_unwm_samples.append(encoded_output)

            output = generate_watermarked(decoded_prompts, **enhancement_kwargs)
            encoded_output = tokenizer(output, return_tensors='pt', truncation=True, padding='max_length', max_length=T)['input_ids']
            watermarked_samples.append(output)
            tokenized_wm_samples.append(encoded_output)
            

        tokenized_unwm_samples = torch.vstack(tokenized_unwm_samples)
        tokenized_wm_samples = torch.vstack(tokenized_wm_samples)

        watermarked_samples = [item for sublist in watermarked_samples for item in sublist]
        unwatermarked_samples = [item for sublist in unwatermarked_samples for item in sublist]
        decoded_prompts = [item for sublist in decoded_prompts_batched for item in sublist]
        
        tokenized_unwm_samples = torch.clip(tokenized_unwm_samples,max=eff_vocab_size-1)
        tokenized_wm_samples = torch.clip(tokenized_wm_samples,max=eff_vocab_size-1)

        pvals_watermark = []
        pvals_unwatermark = []
        pvals_gold = []

        scores = []
        z_scores = {}
        translated_wm_samples = []
        corrupted_wm_samples = []
        paraphrased_wm_samples = []

        pbar = tqdm(total=num_examples)
        for itm in range(num_examples):
            watermarked_sample = tokenized_wm_samples[itm]

            if len(watermarked_sample) < T + 1:
                watermarked_sample = torch.nn.functional.pad(watermarked_sample,(T-len(watermarked_sample),0),"constant",0)
            else:
                watermarked_sample = watermarked_sample[1:T+1]

            watermarked_sample = tokenizer.decode(watermarked_sample, skip_special_tokens=True)
            scores.append(detect_watermark(natural_completions[itm]))
            scores.append(detect_watermark(unwatermarked_samples[itm]))
            scores.append(detect_watermark(watermarked_sample))
            z_scores[itm] = scores
            scores=[]
            pbar.update(1)

        pbar.close()
        scoring_name = "z_scores" if method != "EXP" else "pvals"

        results_file = []
        for i in range(num_examples):
            results_file.append({
                "index": i,
                "prompt": decoded_prompts[i],
                "natural_text": natural_completions[i],
                "unwatermarked_sample": unwatermarked_samples[i],
                "watermarked_sample": watermarked_samples[i],
                scoring_name: [z_scores[i][0]['score'],z_scores[i][1]['score'], z_scores[i][2]['score']],
            })

        # dump results to json
        json_file_dump = dataset_path
        with open(json_file_dump, "w") as f:
            json.dump(results_file, f, ensure_ascii=False, indent=4)

    elif experiment == "quality":
        # Evaluate PPL and GPT judger
        my_dataset = C4Dataset(dataset_path, num_examples, indexing_legacy=False)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_path, quantization_config=bnb_config)
        oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_path)
        
        oracle_tokenizer.truncation_side = 'left' # we want truncation to happen on the left side for PPL calculation
        ppl_analyzer = PPLCalculator(model=oracle_model,
                                tokenizer=oracle_tokenizer,
                                device=device)

        ppl_quality_pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset,
                                        watermarked_text_editor_list=[],
                                        unwatermarked_text_editor_list=[],
                                        analyzer=ppl_analyzer,
                                        unwatermarked_text_source='generated', show_progress=True,
                                        is_processed_data=True,
                                        return_type=QualityPipelineReturnType.SCORES)
        
        singhZou_analyzer = SinghZouJudge(openai_model='gpt-4o-mini', task_description="Singh and Zou GPT judger")
        singhZou_quality_pipeline = ExternalDiscriminatorTextQualityAnalysisPipeline(dataset=my_dataset,
                                        watermarked_text_editor_list=[],
                                        unwatermarked_text_editor_list=[],
                                        show_progress=True,
                                        is_processed_data=True,
                                        analyzer=singhZou_analyzer,
                                        return_type=QualityPipelineReturnType.SCORES)
    
        print(f"PPL Evaluation ...")
        ppl_eval_results = ppl_quality_pipeline.evaluate(myWatermark)
        print(f"SinghZou Evaluation ...")
        singhZou_eval_results = singhZou_quality_pipeline.evaluate(myWatermark)
    
        # read json file
        print("saving results ...")
        with open(dataset_path) as f:
            data = json.load(f)

        for idx, d in enumerate(data[:num_examples]):
            d.update({'ppl': [ppl_eval_results[idx]['unwatermarked'], ppl_eval_results[idx]['watermarked']]})
            uwm_results = singhZou_eval_results[idx]['unwatermarked']
            wm_results = singhZou_eval_results[idx]['watermarked']
            uwm_final_verdict = uwm_results.get('final_verdict', None)
            wm_final_verdict = wm_results.get('final_verdict', None)
            judge_output = wm_results["judge_output"] # for now judge output is in the watermarked results
            final_verdict = "Model Failure"
            if uwm_final_verdict is not None and wm_final_verdict is not None: # it's a tie
                final_verdict = "Tie"
            elif uwm_final_verdict is not None:
                final_verdict = uwm_final_verdict
            elif wm_final_verdict is not None:
                final_verdict = wm_final_verdict
            uwm_scores = uwm_results.get('scores', None)
            wm_scores = wm_results.get('scores', None)

            d.update({"final_verdict" : final_verdict, \
                      'singhZouGptJudge': [uwm_scores, wm_scores], \
                        'judge_output': judge_output})
        
        json_file_dump = dataset_path.replace("original", "quality")
        # get the number after examples= using regex and replace with the num_examples
        # import re
        # num_examples_match = re.search(r'examples=(\d+)', json_file_dump)
        # if num_examples_match:
        #     num_examples_str = num_examples_match.group(1)
        #     json_file_dump = json_file_dump.replace(num_examples_str, str(num_examples))
        with open(json_file_dump, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    elif experiment == "attack":
        ### Attacks ###
        attack1 = "translation"
        attack2 = "translation->paraphrase"
        attack3 = "paraphrase->translation"
        attacks = [attack1, attack2, attack3]
        is_processed_data = True
        if translation_aware:
            is_processed_data = False # we would like to match mywatermark settings.
        kwargs.update({"is_processed_data": is_processed_data})
        
        # 1. Translation
        trans_editor = OPUSTranslationTextEditor(mt_tokenizer, mt_model)
        # for Anthropic translations use:
        '''trans_editor = AnthropicTranslationTextEditor(anthropic_model="claude-3-5-haiku-20241022",
                                                      from_lang=language.lower(),
                                                      to_lang=args.target_language.lower())'''

        # 1. backtranslation
        # for backtranslation src_lang becomes target_lang
        mt_model_back, mt_tokenizer_back = get_mt_model_tokenizer(args.target_language.lower(), language.lower())
        trans_editor_back = OPUSTranslationTextEditor(mt_tokenizer_back, mt_model_back)

        paraphras_editor = GPTParaphraser(openai_model='gpt-4o-mini',
                                         prompt='Please rewrite the following text: ')
            
        my_dataset = C4Dataset(dataset_path, num_examples, indexing_legacy=False)

        # for Fair ROC curve scores, we translated and paraphrased the unwatermarked text to get
        # the correct z-scores.
        unwm_pipeline1 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[trans_editor],
                                                       show_progress=True,
                                                       return_type=DetectionPipelineReturnType.FULL,
                                                       **kwargs)
        
        unwm_pipeline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[trans_editor, paraphras_editor],
                                                       show_progress=True,
                                                       return_type=DetectionPipelineReturnType.FULL,
                                                       **kwargs)
        
        pipeline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[trans_editor],
                                                    show_progress=True,
                                                    return_type=DetectionPipelineReturnType.FULL,
                                                    **kwargs)
        
        pipeline2 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[trans_editor, paraphras_editor],
                                                    show_progress=True,
                                                    return_type=DetectionPipelineReturnType.FULL,
                                                    **kwargs)
        
        unwm_pipeline1_results = unwm_pipeline1.evaluate(myWatermark)
        unwm_pipeline2_results = unwm_pipeline2.evaluate(myWatermark)

        pipeline1_results = pipeline1.evaluate(myWatermark) # You could pass batch size as well (default=10)
        pipeline2_results = pipeline2.evaluate(myWatermark)


        # pipeline1 results
        unwm_pipeline1_unmodified_wm_texts = [detect_res.generated_or_retrieved_text for detect_res in unwm_pipeline1_results]
        unwm_pipeline1_edited_texts = [detect_res.edited_text for detect_res in unwm_pipeline1_results]
        unwm_pipeline1_z_scores = [detect_res.detect_result['score'] for detect_res in unwm_pipeline1_results]

        # pipeline2 results
        unwm_pipeline2_unmodified_wm_texts = [detect_res.generated_or_retrieved_text for detect_res in unwm_pipeline2_results]
        unwm_pipeline2_edited_texts = [detect_res.edited_text for detect_res in unwm_pipeline2_results]
        unwm_pipeline2_z_scores = [detect_res.detect_result['score'] for detect_res in unwm_pipeline2_results]

        # pipeline1 results
        pipeline1_unmodified_wm_texts = [detect_res.generated_or_retrieved_text for detect_res in pipeline1_results]
        pipeline1_edited_texts = [detect_res.edited_text for detect_res in pipeline1_results]
        pipeline1_z_scores = [detect_res.detect_result['score'] for detect_res in pipeline1_results]

        # pipeline2 results
        pipeline2_unmodified_wm_texts = [detect_res.generated_or_retrieved_text for detect_res in pipeline2_results]
        pipeline2_edited_texts = [detect_res.edited_text for detect_res in pipeline2_results]
        pipeline2_z_scores = [detect_res.detect_result['score'] for detect_res in pipeline2_results]

        
        # read json file
        with open(dataset_path) as f:
            data = json.load(f)

        for idx, d in enumerate(data[:num_examples]):
            d.update({'unwm_translated': unwm_pipeline1_edited_texts[idx],
                      'unwm_translated_score': unwm_pipeline1_z_scores[idx],
                      'unwm_translated_paraphrased': unwm_pipeline2_edited_texts[idx],
                      'unwm_translated_paraphrased_score': unwm_pipeline2_z_scores[idx]})
            
            d.update({attack1: attacks[0],
                    'original_wm1': pipeline1_unmodified_wm_texts[idx],
                    'attacked_wm1': pipeline1_edited_texts[idx],
                    'attacked_score1': pipeline1_z_scores[idx]})
            
            d.update({attack2: attacks[1],
                      'original_wm2': pipeline2_unmodified_wm_texts[idx],
                      'attacked_wm2': pipeline2_edited_texts[idx],
                      'attacked_score2': pipeline2_z_scores[idx]})
        
        json_file_dump = dataset_path.replace("original", "attack")
        json_file_dump = json_file_dump.replace(f"_{language}_{model_prefix}",f"_src={language}_{model_prefix}_tgt={args.target_language}")

        # write back the json file after updating
        with open(json_file_dump, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        # paraphrase->translate
        dataset_path = json_file_dump
        my_dataset = C4Dataset(dataset_path, num_examples, indexing_legacy=False, watermark_idx="attacked_wm2")
        pipeline3 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[trans_editor_back],
                                                    show_progress=True,
                                                    return_type=DetectionPipelineReturnType.FULL,
                                                    **kwargs)
        pipeline3_results = pipeline3.evaluate(myWatermark)
        pipeline3_unmodified_wm_texts = [detect_res.generated_or_retrieved_text for detect_res in pipeline3_results]
        pipeline3_edited_texts = [detect_res.edited_text for detect_res in pipeline3_results]
        pipeline3_z_scores = [detect_res.detect_result['score'] for detect_res in pipeline3_results]

        # read json file
        with open(dataset_path) as f:
            data = json.load(f)

        for idx, d in enumerate(data[:num_examples]):
            d.update({attack3: attacks[2],
                      'original_wm3': pipeline3_unmodified_wm_texts[idx],
                      'attacked_wm3': pipeline3_edited_texts[idx],
                      'attacked_score3': pipeline3_z_scores[idx]})
            
        with open(dataset_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    elif experiment == "low_entropy_test":
        # Evaluate PPL and GPT judger
        # my_dataset = C4Dataset(dataset_path, num_examples, indexing_legacy=False)
        generate_watermarked = lambda prompt : myWatermark.generate_watermarked_text(prompt)
        generate_unwatermarked = lambda prompt : myWatermark.generate_unwatermarked_text(prompt)
        detect_watermark = lambda text: myWatermark.detect_watermark(text)

        unwatermarked_samples = []
        watermarked_samples = []

        # Open the json file and read the data into their specific lists
        with open(dataset_path) as f:
            data = json.load(f)
        for idx, d in enumerate(data[:num_examples]):
            prompt = d['prompt']
            natural_text = d['natural_text']
            unwatermarked_sample = d['unwatermarked_sample']
            watermarked_sample = d['watermarked_sample']

            unwatermarked_samples.append(unwatermarked_sample)
            watermarked_samples.append(watermarked_sample)
        
        wm_z_scores = []
        unwm_z_scores = []
        natural_z_scores = []

        for idx in range(num_examples):
            unwatermarked_sample = unwatermarked_samples[idx]
            watermarked_sample = watermarked_samples[idx]
            wm_z_scores.append(detect_watermark(watermarked_sample))
            unwm_z_scores.append(detect_watermark(unwatermarked_sample))
            natural_z_scores.append(detect_watermark(natural_text))

        # Update the json file with the z scores
        for idx, d in enumerate(data[:num_examples]):
            d.update({"z_scores_ewd": [natural_z_scores[idx]['score'], unwm_z_scores[idx]['score'], wm_z_scores[idx]['score']]})

        json_file_dump = dataset_path.replace("original", "quality")
        
        with open(json_file_dump, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    elif experiment == "token_length":
        # token length experiments, just retrieve the watermarked and unwatermarked samples
        # and truncate the watermarked samples to the specified lengths then call detect score
        # read json file
        with open(dataset_path) as f:
            data = json.load(f)
        
        detect_watermark = lambda text: myWatermark.detect_watermark(text)

        for idx, d in enumerate(data[:num_examples]):
            prompt = d['prompt']
            natural_text = d['natural_text']
            unwatermarked_sample = d['unwatermarked_sample']
            watermarked_sample = d['watermarked_sample']
            unwatermarked_sample = tokenizer.encode(unwatermarked_sample, return_tensors='pt', truncation=True, max_length=T+prompt_tokens)[0]
            watermarked_sample = tokenizer.encode(watermarked_sample, return_tensors='pt', truncation=True, max_length=T+prompt_tokens)[0]
            natural_sample = tokenizer.encode(natural_text, return_tensors='pt', truncation=True, max_length=T+prompt_tokens)[0]
            # truncate the watermarked sample to the specified lengths
            wm_z_scores = []
            unwm_z_scores = []
            natural_z_scores = []
            # try lengths 50, 100, 150, 200
            for length in [25, 50, 100, 150, 200]:
                wm_sample = watermarked_sample[:length]
                unwm_sample = unwatermarked_sample[:length]
                natural_sample = natural_sample[:length]
                wm_z_scores.append(detect_watermark(tokenizer.decode(wm_sample, skip_special_tokens=True)))
                unwm_z_scores.append(detect_watermark(tokenizer.decode(unwm_sample, skip_special_tokens=True)))
                natural_z_scores.append(detect_watermark(tokenizer.decode(natural_sample, skip_special_tokens=True)))

            # Update the json file with the z scores
            d.update({"z_scores_25": [natural_z_scores[0]['score'], unwm_z_scores[0]['score'], wm_z_scores[0]['score']],
                      "z_scores_50": [natural_z_scores[1]['score'], unwm_z_scores[1]['score'], wm_z_scores[1]['score']],
                      "z_scores_100": [natural_z_scores[2]['score'], unwm_z_scores[2]['score'], wm_z_scores[2]['score']],
                      "z_scores_150": [natural_z_scores[3]['score'], unwm_z_scores[3]['score'], wm_z_scores[3]['score']],
                      "z_scores_200": [natural_z_scores[4]['score'], unwm_z_scores[4]['score'], wm_z_scores[4]['score']],})
        # write back the json file after updating
        json_file_dump = dataset_path.replace("original", "token_length")
        if not os.path.exists(json_file_dump):
            os.makedirs(os.path.dirname(json_file_dump), exist_ok=True)

        with open(json_file_dump, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        

    else: # mitigation or attacks with defenses. Only apply if you have a defense method. Also these codes are not reviewed.
        ### Attacks with mitigation###
        # attack_name = 'Doc-P(Dipper)'
        attack1 = "translation"
        attack2 = "translation->paraphrase"
        attack3 = "paraphrase->translation"
        attacks = [attack1, attack2, attack3]
        is_processed_data = True
        if translation_aware: # For mitigation we need to process the data so this must be True
            is_processed_data = False # we would like to match mywatermark settings.
        kwargs.update({"is_processed_data": is_processed_data})
        
        # 1. Translation
        trans_editor = TranslationTextEditor(mt_tokenizer, mt_model)

       
        paraphras_editor = GPTParaphraser(openai_model='gpt-4o-mini',
                                         prompt='Please rewrite the following text: ')
            
        my_dataset = C4Dataset(dataset_path, num_examples, indexing_legacy=False)
        pipeline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[trans_editor],
                                                    show_progress=True,
                                                    return_type=DetectionPipelineReturnType.FULL,
                                                    **kwargs)
        
        pipeline2 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[trans_editor, paraphras_editor],
                                                    show_progress=True,
                                                    return_type=DetectionPipelineReturnType.FULL,
                                                    **kwargs)
        pipeline3 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[trans_editor, paraphras_editor, trans_editor_back],
                                                    show_progress=True,
                                                    return_type=DetectionPipelineReturnType.FULL,
                                                    **kwargs)
        
        pipeline1_results = pipeline1.evaluate(myWatermark) # You could pass batch size as well (default=10)
        pipeline2_results = pipeline2.evaluate(myWatermark)
        pipeline3_results = pipeline3.evaluate(myWatermark)

        # pipeline1 results
        pipeline1_unmodified_wm_texts = [detect_res.generated_or_retrieved_text for detect_res in pipeline1_results]
        pipeline1_edited_texts = [detect_res.edited_text for detect_res in pipeline1_results]
        pipeline1_z_scores = [detect_res.detect_result['score'] for detect_res in pipeline1_results]

        # pipeline2 results
        pipeline2_unmodified_wm_texts = [detect_res.generated_or_retrieved_text for detect_res in pipeline2_results]
        pipeline2_edited_texts = [detect_res.edited_text for detect_res in pipeline2_results]
        pipeline2_z_scores = [detect_res.detect_result['score'] for detect_res in pipeline2_results]

        # pipeline3 results
        pipeline3_unmodified_wm_texts = [detect_res.generated_or_retrieved_text for detect_res in pipeline3_results]
        pipeline3_edited_texts = [detect_res.edited_text for detect_res in pipeline3_results]
        pipeline3_z_scores = [detect_res.detect_result['score'] for detect_res in pipeline3_results]
        
        # read json file
        with open(dataset_path) as f:
            data = json.load(f)

        for idx, d in enumerate(data[:num_examples]):
            d.update({attack1: attacks[0],
                      'original_wm1': pipeline1_unmodified_wm_texts[idx],
                      'attacked_wm1': pipeline1_edited_texts[idx],
                      'attacked_score1': pipeline1_z_scores[idx]})
            
            d.update({attack2: attacks[1],
                      'original_wm2': pipeline2_unmodified_wm_texts[idx],
                      'attacked_wm2': pipeline2_edited_texts[idx],
                      'attacked_score2': pipeline2_z_scores[idx]})
            
            d.update({attack3: attacks[2],
                      'original_wm3': pipeline3_unmodified_wm_texts[idx],
                      'attacked_wm3': pipeline3_edited_texts[idx],
                      'attacked_score3': pipeline3_z_scores[idx]})
            
        # json_file_dump = dataset_path.replace("quality", "attack")
        json_file_dump = dataset_path.replace("original", "mitigation")
        json_file_dump = json_file_dump.replace(f"_{language}_{model_prefix}",f"_src={language}_{model_prefix}_tgt={args.target_language}_hash=4")
        # write back the json file after updating
        with open(json_file_dump, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)