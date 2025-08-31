#!/bin/bash
num_examples=$1
model=$2
perturbation_type=$3
if [ -z "$num_examples" ] || [ -z "$model" ] || [ -z "$perturbation_type" ]; then
    echo "Usage: $0 <num_examples> <model> <perturbation_type>"
    exit 1
fi
output_dir=data/${perturbation_type}_versions/$model
# if the output directory does not exist, create it
if [ ! -d "$output_dir" ]; then
    mkdir -p $output_dir
fi
# if the logs directory does not exist, create it
if [ ! -d "$output_dir/logs" ]; then
    mkdir -p $output_dir/logs
fi

# if perturbation_type is translation then do the following for loop
# else, it's a paraphrase, do the next for loop.
if [ "$perturbation_type" == "translation" ]; then
    for task in lfqa c4
    do
        if [ "$task" == "lfqa" ]; then
            data_file=data/lfqa.jsonl
        else
            data_file=data/c4_500_processed.json
        fi
        for dest_language in Hindi Turkish Indonesian Arabic Persian German Chinese Japanese
        do
            time python3 experiments/text_perturbation_by_llm.py \
            --data_file $data_file \
            --output_dir $output_dir \
            --num_examples $num_examples \
            --task $task \
            --dest_language $dest_language \
            --model $model \
            --perturbation_type $perturbation_type \
            &> ${output_dir}/logs/${perturbation_type}_${task}_${dest_language}_${num_examples}.log
            if [ $? -ne 0 ]; then
                echo "Error occurred during translation for $task in $dest_language"
                exit 1
            fi
            echo "Translated $task data to $dest_language"
        done
    done
else

# only do it for c4
# create a mapping of the languages name to their respective acronyms, please only
# include languages that are needed for the experiments in the paper.
declare -A language_map=(
    ["zh"]="Chinese"
    ["fa"]="Persian"
    ["id"]="Indonesian"
    ["ja"]="Japanese"
    ["hi"]="Hindi"
    ["tr"]="Turkish"
    ["de"]="German"
    ["ar"]="Arabic"
    ["en"]="English"
)

    for language in "${!language_map[@]}"
    do
        # echo language acronym
        echo "$language"
        # get data file which ends with the language acronym. for example, for Arabic, it should be data/.../ar_c4.jsonl
        data_file=data/translation_versions/$model/${language}_c4.jsonl
        time python3 experiments/text_perturbation_by_llm.py \
        --data_file $data_file \
        --output_dir $output_dir \
        --num_examples $num_examples \
        --task c4 \
        --dest_language ${language_map[$language]} \
        --model $model \
        --perturbation_type $perturbation_type \
        &> ${output_dir}/logs/${perturbation_type}_c4_${language}_${num_examples}.log
        if [ $? -ne 0 ]; then
            echo "Error occurred during paraphrasing for c4 in ${language_map[$language]}"
            exit 1
        fi
        echo "Paraphrased c4 data to ${language_map[$language]}"
    done
fi
