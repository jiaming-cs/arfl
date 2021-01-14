#!/usr/bin/env bash

dataset="cifar10"
epochs=1
num_rounds=1000
per_round=10
actors=10
gpus=2
ps=1.0
method=$1
lr=0.01
seed=0
setup_clients=1.0
batch_size=64
attack_type=0
pc=0.3
seed=1

mkdir -p results/${dataset} log/${dataset}
pushd ../models || exit

pushd ../data/cifar10/preprocess || exit
  python get_cifar10.py --num_clients 10 --noniid
popd || exit

for lambda in 2.5 5.0
  do
      output_name=${dataset}_$1_${pc}_${ps}_${attack_type}_${lambda}
      python main.py --dataset ${dataset} --model cnn --setup_clients ${setup_clients} --lr ${lr} --num_actors \
      ${actors} --batch-size ${batch_size} --num_gpus ${gpus} --seed ${seed} --clients-per-round ${per_round} \
      --reg_weight ${lambda} --num-epochs ${epochs} --num-rounds ${num_rounds} --method ${method} -pc ${pc} -ps ${ps} \
       --attack_type ${attack_type} --metrics-dir ../experiments/results/${dataset} --metrics_name \
        ${output_name} > ../experiments/log/${dataset}/${output_name}.txt
  done

popd || exit