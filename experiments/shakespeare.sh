#!/usr/bin/env bash

model="lstm"
dataset="shakespeare"
pc=0.5
ps=1.0
epochs=5
seed=0
actors=16
gpus=2
batch_size=64
num_rounds=100
per_round=16
setup_clients=1.0
lr=0.6
attack_type=3
lambda=100.0
mkdir -p results/${dataset} log/${dataset}
pushd ../

pushd models/ || exit

method=$1


for attack_type in 3
  do
    for pc in 0.3
    do
      for seed in 0
          do
            output_name=${dataset}_$1_${pc}_${ps}_${attack_type}_${lambda}_${seed}
            python main.py --dataset ${dataset} --model ${model} --lr ${lr} --setup_clients ${setup_clients} --num_actors ${actors} \
               --batch-size ${batch_size} --num_gpus ${gpus} --seed ${seed} --clients-per-round ${per_round} \
               --reg_weight ${lambda} --num-epochs ${epochs} --num-rounds ${num_rounds} --method ${method} -pc ${pc} -ps ${ps} \
               --attack_type ${attack_type} --metrics-dir ../experiments/results/${dataset} --metrics_name \
              ${output_name} > ../experiments/log/${dataset}/${output_name}.txt
          done
    done
  done

popd || exit
popd || exit