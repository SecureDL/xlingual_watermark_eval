#!/bin/bash
examples=$1
task=$2
output_dir=$3
experiment=attack
T=200
sampling=multinomial
language=English
delta=2.0
gamma=0.5

for method in KGW Unigram
do
    run=4 # please use the correct run number for corresponding gamma and delta values.
    for targetlanguage in Arabic Chinese Indonesian
    do
        # model_prefix="r7b"
        # model="CohereForAI/c4ai-command-r7b-12-2024"

        # if single model is used for all languages, uncomment the above two lines and comment the below lines.
        # We use different models for different languages so the correct file must be used. for example, when
        # translating English to Arabic, we need to use the the generated English file from Arabic model, not
        # from the other languages' models.
        if [ "$targetlanguage" == "Arabic" ]; then
            model_prefix="jais"
            model="inceptionai/jais-family-6p7b"
            
        else
            model_prefix="sail"
            model="sail/Sailor2-1B"
        fi
        echo run: $run
        time python3 experiments/all-experiments.py \
            --method $method \
            --batch_size 10 \
            --T $T \
            --delta $delta \
            --gamma $gamma \
            --model_path $model \
            --save original/${method}/${language}/run_${run}_${task}_examples=${examples}_${language}_${model_prefix}_${method}_delta=${delta}_gamma=${gamma}_T=${T}_${sampling} \
            --output_dir $output_dir \
            --num_examples $examples \
            --language $language  \
            --experiment $experiment \
            --adversary_scenario naive \
            --target_language $targetlanguage \
            &> ${output_dir}/${experiment}/${method}/logs/${run}_${task}_examples_${examples}_src_${language}_${model_prefix}_tgt_${targetlanguage}_${method}_${delta}_${gamma}_T_${T}_${sampling}.log
    done
done

targetlanguage=English
for method in KGW Unigram
do
    run=4
    for language in Arabic Chinese Indonesian
    do
        # model_prefix="r7b"
        # model="CohereForAI/c4ai-command-r7b-12-2024"
        if [ "$language" == "Arabic" ]; then
            model_prefix="jais"
            model="inceptionai/jais-family-6p7b"
            
        else
            model_prefix="sail"
            model="sail/Sailor2-1B"
        fi
        echo run: $run
        time python3 experiments/all-experiments.py \
            --method $method \
            --batch_size 10 \
            --T $T \
            --delta $delta \
            --gamma $gamma \
            --model_path $model \
            --save original/${method}/${language}/run_${run}_${task}_examples=${examples}_${language}_${model_prefix}_${method}_delta=${delta}_gamma=${gamma}_T=${T}_${sampling} \
            --output_dir $output_dir \
            --num_examples $examples \
            --language $language  \
            --experiment $experiment \
            --adversary_scenario naive \
            --target_language $targetlanguage \
            &> ${output_dir}/${experiment}/${method}/logs/${run}_${task}_examples_${examples}_src_${language}_${model_prefix}_tgt_${targetlanguage}_${method}_${delta}_${gamma}_T_${T}_${sampling}.log
    done
done

language=English
method=EXP
run=1
for targetlanguage in Arabic Chinese Indonesian
do
    # model_prefix="r7b"
    # model="CohereForAI/c4ai-command-r7b-12-2024"
    if [ "$targetlanguage" == "Arabic" ]; then
        model_prefix="jais"
        model="inceptionai/jais-family-6p7b"
    else
        model_prefix="sail"
        model="sail/Sailor2-1B"
    fi
    echo run: $run
    time python3 experiments/all-experiments.py \
        --method $method \
        --batch_size 10 \
        --T $T \
        --model_path $model \
        --output_dir $output_dir \
        --save original/${method}/${language}/run_${run}_${task}_examples=${examples}_${language}_${model_prefix}_${method}_T=${T}_${sampling} \
        --num_examples $examples \
        --language $language  \
        --experiment $experiment \
        --adversary_scenario naive \
        --target_language $targetlanguage \
        &> ${output_dir}/${experiment}/${method}/logs/${run}_${task}_examples_${examples}_src_${language}_${model_prefix}_tgt_${targetlanguage}_${method}_T_${T}_${sampling}.log
done

targetlanguage=English
for language in Arabic Chinese Indonesian
do
#     model_prefix="r7b"
#     model="CohereForAI/c4ai-command-r7b-12-2024"
    if [ "$language" == "Arabic" ]; then
        model_prefix="jais"
        model="inceptionai/jais-family-6p7b"
    else
        model_prefix="sail"
        model="sail/Sailor2-1B"
    fi
    echo run: $run
    time python3 experiments/all-experiments.py \
        --method $method \
        --batch_size 10 \
        --T $T \
        --model_path $model \
        --output_dir $output_dir \
        --save original/${method}/${language}/run_${run}_${task}_examples=${examples}_${language}_${model_prefix}_${method}_T=${T}_${sampling} \
        --num_examples $examples \
        --language $language  \
        --experiment $experiment \
        --adversary_scenario naive \
        --target_language $targetlanguage \
        &> ${output_dir}/${experiment}/${method}/logs/${run}_${task}_examples_${examples}_src_${language}_${model_prefix}_tgt_${targetlanguage}_${method}_delta_${delta}_T_${T}_${sampling}.log
done

language=English
method=XSIR
run=1
for targetlanguage in Arabic Chinese Indonesian
do
    # model_prefix="r7b"
    # model="CohereForAI/c4ai-command-r7b-12-2024"
    # mapping_name="watermark/xsir/mapping/300_mapping_r7b-12-2024.json"
    if [ "$targetlanguage" == "Arabic" ]; then
        model_prefix="jais"
        model="inceptionai/jais-family-6p7b"
        mapping_name="watermark/xsir/mapping/300_mapping_jais-family-6p7b-84992.json"
    else
        model_prefix="sail"
        model="sail/Sailor2-1B"
        mapping_name="watermark/xsir/mapping/300_mapping_sailor2-1B-151665.json"
    fi
    echo run: $run
    time python3 experiments/all-experiments.py \
        --method $method \
        --batch_size 10 \
        --T $T \
        --delta $delta \
        --model_path $model \
        --output_dir $output_dir \
        --save original/${method}/${language}/run_${run}_${task}_examples=${examples}_${language}_${model_prefix}_${method}_delta=${delta}_T=${T}_${sampling} \
        --num_examples $examples \
        --language $language  \
        --experiment $experiment \
        --adversary_scenario naive \
        --target_language $targetlanguage \
        --mapping_name $mapping_name \
        &> ${output_dir}/${experiment}/${method}/logs/${run}_${task}_examples_${examples}_src_${language}_${model_prefix}_tgt_${targetlanguage}_${method}_${delta}_T_${T}_${sampling}.log
done

targetlanguage=English
for language in Arabic Chinese Indonesian
do
    # model_prefix="r7b"
    # model="CohereForAI/c4ai-command-r7b-12-2024"
    # mapping_name="watermark/xsir/mapping/300_mapping_r7b-12-2024.json"
    if [ "$language" == "Arabic" ]; then
        model_prefix="jais"
        model="inceptionai/jais-family-6p7b"
        mapping_name="watermark/xsir/mapping/300_mapping_jais-family-6p7b-84992.json"
    else
        model_prefix="sail"
        model="sail/Sailor2-1B"
        mapping_name="watermark/xsir/mapping/300_mapping_sailor2-1B-151665.json"
    fi
    echo run: $run
    time python3 experiments/all-experiments.py \
        --method $method \
        --batch_size 10 \
        --T $T \
        --delta $delta \
        --model_path $model \
        --output_dir $output_dir \
        --save original/${method}/${language}/run_${run}_${task}_examples=${examples}_${language}_${model_prefix}_${method}_delta=${delta}_T=${T}_${sampling} \
        --num_examples $examples \
        --language $language  \
        --experiment $experiment \
        --adversary_scenario naive \
        --target_language $targetlanguage \
        --mapping_name $mapping_name \
        &> ${output_dir}/${experiment}/${method}/logs/${run}_${task}_examples_${examples}_src_${language}_${model_prefix}_tgt_${targetlanguage}_${method}_${delta}_T_${T}_${sampling}.log
done