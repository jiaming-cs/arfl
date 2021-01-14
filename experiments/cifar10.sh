#!/usr/bin/env bash

dataset="cifar10"
epochs=5
num_rounds=2000
per_round=16
actors=16
gpus=2
ps=1.0
method=$1
lr=0.01
seed=0
setup_clients=1.0
batch_size=64

pushd ../data/cifar10/preprocess || exit
  python get_cifar10.py
popd || exit

mkdir -p results/${dataset} log/${dataset}
pushd ../models || exit


for attack_type in 2
  do
    for pc in 0.3
    do
      for seed in 0 1 2 3 4
          do
            output_name=${dataset}_$1_${pc}_${ps}_${attack_type}_${seed}
            python main.py --dataset ${dataset} --model cnn --setup_clients ${setup_clients} --lr ${lr} --num_actors \
            ${actors} --batch-size ${batch_size} --num_gpus ${gpus} --seed ${seed} --clients-per-round ${per_round} \
            --reg_weight 1.0  --num-epochs ${epochs} --num-rounds ${num_rounds} --method ${method} -pc ${pc} -ps ${ps} \
             --attack_type ${attack_type} --metrics-dir ../experiments/results/${dataset} --metrics_name \
              ${output_name} > ../experiments/log/${dataset}/${output_name}.txt
          done
    done
  done

popd || exit