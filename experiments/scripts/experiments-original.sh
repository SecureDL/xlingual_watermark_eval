#!/bin/bash
run=$1
examples=$2
task=$3
data_dir=$4
instructive=$5
is_processed_prompts=$6
experiment=original
sampling=multinomial
T=200

for method in KGW Unigram
do
    for language in English Arabic Chinese Indonesian Turkish Hindi
    do
        model_prefix="r7b"
        model="CohereForAI/c4ai-command-r7b-12-2024"
        # to re-produce the results, add to the if statements
        # model_prefix="jais" and model="inceptionai/jais-family-6p7b" for Arabic,
        # and model_prefix="sail" and model="sail/Sailor2-1B" for Chinses and Indonesian.
        # remove data_dir flag if you want to read from C4 directly. For LFQA you still need data_dir flag.
        if [ "$language" == "Arabic" ]; then
            data=${data_dir}ar_${task}.jsonl
        elif [ "$language" == "Chinese" ]; then
            data=${data_dir}zh_${task}.jsonl
        elif [ "$language" == "English" ]; then
            data=${data_dir}en_${task}.jsonl # any language prefix can work here since all files have the English versions of the text
        elif [ "$language" == "Indonesian" ]; then
            data=${data_dir}id_${task}.jsonl
        elif [ "$language" == "Turkish" ]; then
            data=${data_dir}tr_${task}.jsonl
        else # Hindi
            data=${data_dir}hi_${task}.jsonl
        fi
        run=1
        for gamma in 0.1 0.5 0.9
        do
            for delta in 2.0 5.0 10.0
            do
                # if this is the first run, remove flag --data_dir
                echo run: $run
                time python3 experiments/all-experiments.py \
                    --method $method \
                    --batch_size 10 \
                    --T $T \
                    --delta $delta \
                    --gamma $gamma \
                    --model_path $model \
                    --save $experiment/${method}/${language}/run_${run}_${task}_examples=${examples}_${language}_${model_prefix}_${method}_delta=${delta}_gamma=${gamma}_T=${T}_${sampling} \
                    --num_examples $examples \
                    --language $language  \
                    --experiment $experiment \
                    --data_dir $data \
                    --is_processed_prompts $is_processed_prompts \
                    --instruction_following $instructive \
                    &> outputs/${experiment}/${method}/logs/${run}_${task}_examples_${examples}_${language}_${model_prefix}_${method}_${delta}_${gamma}_T_${T}_${sampling}.log
                run=$((run+1))
            done
        done
    done
done

for method in EXP
do
    for language in English Arabic Chinese Indonesian Turkish Hindi
    do
        model_prefix="r7b"
        model="CohereForAI/c4ai-command-r7b-12-2024"
        if [ "$language" == "Arabic" ]; then
            data=${data_dir}ar_${task}.jsonl
        elif [ "$language" == "Chinese" ]; then
            data=${data_dir}zh_${task}.jsonl
        elif [ "$language" == "English" ]; then
            data=${data_dir}en_${task}.jsonl # any language prefix can work here since all files have the English versions of the text
        elif [ "$language" == "Indonesian" ]; then
            data=${data_dir}id_${task}.jsonl
        elif [ "$language" == "Turkish" ]; then
            data=${data_dir}tr_${task}.jsonl
        else # Hindi
            data=${data_dir}hi_${task}.jsonl
        fi
        run=1
        echo run: $run
        time python3 experiments/all-experiments.py \
            --method $method \
            --batch_size 10 \
            --T 200 \
            --model_path $model \
            --save $experiment/${method}/${language}/run_${run}_${task}_examples=${examples}_${language}_${model_prefix}_${method}_T=${T}_${sampling} \
            --num_examples $examples \
            --language $language  \
            --experiment $experiment \
            --data_dir $data \
            --is_processed_prompts $is_processed_prompts \
            --instruction_following $instructive \
            &> outputs/${experiment}/${method}/logs/${run}_${task}_examples_${examples}_${language}_${model_prefix}_${method}_T_${T}_${sampling}.log
        run=$((run+1))
    done
done

for method in XSIR
do
    for language in English Arabic Chinese Indonesian Persian Japanese Turkish Hindi
    do
        model_prefix="r7b"
        model="CohereForAI/c4ai-command-r7b-12-2024"
        mapping_name="watermark/xsir/mapping/300_mapping_r7b-12-2024.json"
        if [ "$language" == "Arabic" ]; then
            data=${data_dir}ar_${task}.jsonl
        elif [ "$language" == "Chinese" ]; then
            data=${data_dir}zh_${task}.jsonl
        elif [ "$language" == "English" ]; then
            data=${data_dir}en_${task}.jsonl # any language prefix can work here since all files have the English versions of the text
        elif [ "$language" == "Indonesian" ]; then
            data=${data_dir}id_${task}.jsonl
        elif [ "$language" == "Turkish" ]; then
            data=${data_dir}tr_${task}.jsonl
        else # Hindi
            data=${data_dir}hi_${task}.jsonl
        fi
        run=1
        for delta in 2.0 5.0 10.0
        do
            echo run: $run
            time python3 experiments/all-experiments.py \
                --method $method \
                --batch_size 10 \
                --T 200 \
                --delta $delta \
                --model_path $model \
                --save $experiment/${method}/${language}/run_${run}_${task}_examples=${examples}_${language}_${model_prefix}_${method}_delta=${delta}_T=${T}_${sampling} \
                --num_examples $examples \
                --language $language  \
                --experiment $experiment \
                --data_dir $data \
                --mapping_name $mapping_name \
                --is_processed_prompts $is_processed_prompts \
                --instruction_following $instructive \
                &> outputs/${experiment}/${method}/logs/${run}_${task}_examples_${examples}_${language}_${model_prefix}_${method}_${delta}_T_${T}_${sampling}.log
            run=$((run+1))
        done
    done
done