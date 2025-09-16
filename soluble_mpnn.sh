#!/bin/bash
set -ex

binder=$1
model=$2

name=dimer_trop2_human_binder_${binder}_model_${model}
input_path=./${name}_in.json
output_path=./out_$name
fix_res_file=./mpnn_max/interface/${name}_hotspots

python run.py \
        --model_type "soluble_mpnn" \
        --seed 420 \
        --pdb_path_multi "$input_path" \
        --out_folder "$output_path" \
        --save_stats 1 \
        --fixed_residues "$(<$fix_res_file)" \
        --batch_size 5 \
        --number_of_batches 20 \
        --chains_to_design "C" \
        --pack_side_chains 1 \
        --number_of_packs_per_design 1 \
