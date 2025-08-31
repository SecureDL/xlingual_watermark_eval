#!/bin/bash
run=$1
oracle_model=$2
examples=$3

# attack
# these experiments are not reported in the paper, but we included descriptions
# of the results at the end of the paper.
experiment=attack
language=English
delta=2.0
T=200
sampling=multinomial
method=XSIR
run=1
for targetlanguage in Arabic Indonesian
do
    if [ "$targetlanguage" == "Arabic" ]; then
        model_prefix="jais"
        model="inceptionai/jais-family-6p7b"
        mapping_name="watermark/xsir/mapping/300_mapping_84992.json"
    else
        model_prefix="sail"
        model="sail/Sailor2-1B"
        mapping_name="watermark/xsir/mapping/300_mapping_151665.json"
    fi
    echo run: $run
    time python3 experiments/all-experiments.py \
        --method $method \
        --batch_size 10 \
        --T $T \
        --delta $delta \
        --model_path $model \
        --save original/${method}/${language}/cluster_ablation_${run}_examples=100_${language}_${model_prefix}_${method}_delta=${delta}_T=${T}_${sampling} \
        --num_examples $examples \
        --language $language  \
        --experiment $experiment \
        --adversary_scenario naive \
        --target_language $targetlanguage \
        --mapping_name $mapping_name \
        &> outputs/${experiment}/${method}/logs/cluster_ablation_${run}_c4_examples_${examples}_src_${language}_${model_prefix}_tgt_${targetlanguage}_${method}_${delta}_T_${T}_${sampling}.log
done

targetlanguage=English
for language in Arabic Indonesian
do
    if [ "$language" == "Arabic" ]; then
        model_prefix="jais"
        model="inceptionai/jais-family-6p7b"
        mapping_name="watermark/xsir/mapping/300_mapping_84992.json"
    else
        model_prefix="sail"
        model="sail/Sailor2-1B"
        mapping_name="watermark/xsir/mapping/300_mapping_151665.json"
    fi
    echo run: $run
    time python3 experiments/all-experiments.py \
        --method $method \
        --batch_size 10 \
        --T $T \
        --delta $delta \
        --model_path $model \
        --save original/${method}/${language}/cluster_ablation_${run}_examples=100_${language}_${model_prefix}_${method}_delta=${delta}_T=${T}_${sampling} \
        --num_examples $examples \
        --language $language  \
        --experiment $experiment \
        --adversary_scenario naive \
        --target_language $targetlanguage \
        --mapping_name $mapping_name \
        &> outputs/${experiment}/${method}/logs/cluster_ablation_${run}_c4_examples_${examples}_src_${language}_${model_prefix}_tgt_${targetlanguage}_${method}_${delta}_T_${T}_${sampling}.log
done