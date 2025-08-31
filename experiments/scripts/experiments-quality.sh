#!/bin/bash
run=$1 # the run number is important to find the correct original file
examples=$2
task=$3
oracle_model=$4
experiment=quality
sampling=multinomial
T=200

for method in KGW Unigram
do
    for language in English Arabic Chinese Indonesian Turkish Hindi
    do
        model_prefix="r7b"
        model="CohereForAI/c4ai-command-r7b-12-2024"
        # simple if-else could be used if different models are used for different languages.
        # no need for data_dir flag here.

        # Please specify the correct run number for corresponding gamma and delta values.
        run=1
        for gamma in 0.1 0.5 0.9
        do
            for delta in 2.0 5.0 10.0
            do
                echo run: $run
                time python3 experiments/all-experiments.py \
                    --method $method \
                    --batch_size 10 \
                    --T $T \
                    --delta $delta \
                    --gamma $gamma \
                    --model_path $model \
                    --oracle_model_name $oracle_model \
                    --save original/${method}/${language}/run_${run}_${task}_examples=500_${language}_${model_prefix}_${method}_delta=${delta}_gamma=${gamma}_T=${T}_${sampling} \
                    --num_examples $examples \
                    --language $language  \
                    --experiment $experiment \
                    &> outputs/${experiment}/${method}/logs/${run}_${task}_examples_${examples}_${language}_${model_prefix}_${method}_${delta}_${gamma}_T_${T}_${sampling}.log
                # run=$((run+1))
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
        # simple if-else could be used if different models are used for different languages
        run=1
        echo run: $run
        time python3 experiments/all-experiments.py \
            --method $method \
            --batch_size 10 \
            --T 200 \
            --model_path $model \
            --oracle_model_name $oracle_model \
            --save original/${method}/${language}/run_${run}_${task}_examples=500_${language}_${model_prefix}_${method}_T=${T}_${sampling} \
            --num_examples $examples \
            --language $language  \
            --experiment $experiment \
            &> outputs/${experiment}/${method}/logs/${run}_${task}_examples_${examples}_${language}_${model_prefix}_${method}_T_${T}_${sampling}.log
        # run=$((run+1))
    done
done

for method in XSIR
do
    for language in English Arabic Chinese Indonesian Turkish Hindi
    do
        model_prefix="r7b"
        model="CohereForAI/c4ai-command-r7b-12-2024"
        mapping_name="watermark/xsir/mapping/300_mapping_r7b-12-2024.json"

        # simple if-else could be used if different models are used for different languages
        # for example, if [ "$language" == "Arabic" ]; then
        #     model_prefix="jais"
        #     model="inceptionai/jais-family-6p7b"
        #     mapping_name="watermark/xsir/mapping/300_mapping_jais-family-6p7b-84992.json"
        # elif [ "$language" == "Chinese" ]; then
        #     model_prefix="sail"
        #     model="sail/Sailor2-1B"
        #     mapping_name="watermark/xsir/mapping/300_mapping_sailor2-1B-151665.json"
        # ...
        
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
                --oracle_model_name $oracle_model \
                --save original/${method}/${language}/run_${run}_${task}_examples=500_${language}_${model_prefix}_${method}_delta=${delta}_T=${T}_${sampling} \
                --num_examples $examples \
                --language $language  \
                --experiment $experiment \
                --mapping_name $mapping_name \
                &> outputs/${experiment}/${method}/logs/${run}_${task}_examples_${examples}_${language}_${model_prefix}_${method}_${delta}_T_${T}_${sampling}.log
            # run=$((run+1))
        done
    done
done