#!/usr/bin/env bash

dataset="femnist"
pc=0.5
ps=1.0
epochs=20
num_rounds=2000
per_round=20
attack_type=3
actors=16
gpus=2
lambda=10000.0
seed=0
setup_clients=1.0
batch_size=64
mkdir -p results/${dataset} log/${dataset}
pushd ../

pushd models/ || exit
#method=$1
for method in "arfl" "cfl" "mkrum" "rfa"
for seed in 0 1 2 3 4
    do
        output_name=${dataset}_$1_${pc}_${ps}_${attack_type}_${lambda}_${seed}
            python main.py --dataset ${dataset} --model cnn --setup_clients ${setup_clients} --num_actors ${actors} \
               --batch-size ${batch_size} --num_gpus ${gpus} --seed ${seed} --clients-per-round ${per_round} \
               --reg_weight ${lambda} --num-epochs ${epochs} --num-rounds ${num_rounds} --method ${method} -pc ${pc} -ps ${ps} \
               --attack_type ${attack_type} --metrics-dir ../experiments/results/${dataset} --metrics_name \
              ${output_name} > ../experiments/log/${dataset}/${output_name}.txt
    done

#for attack_type in 0 1 2
#  do
#    for pc in 0.3 0.5
#    do
#      for seed in 0 1 2 3 4
#          do
#            output_name=${dataset}_$1_${pc}_${ps}_${attack_type}_${lambda}_${seed}
#            python main.py --dataset ${dataset} --model cnn --setup_clients ${setup_clients} --num_actors ${actors} \
#               --batch-size ${batch_size} --num_gpus ${gpus} --seed ${seed} --clients-per-round ${per_round} \
#               --reg_weight ${lambda} --num-epochs ${epochs} --num-rounds ${num_rounds} --method ${method} -pc ${pc} -ps ${ps} \
#               --attack_type ${attack_type} --metrics-dir ../experiments/results/${dataset} --metrics_name \
#              ${output_name} > ../experiments/log/${dataset}/${output_name}.txt
#          done
#    done
#  done

popd || exit
popd || exit
