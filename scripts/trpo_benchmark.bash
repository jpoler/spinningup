#!/bin/bash

for env in "HalfCheetah-v3" "Ant-v3" "Humanoid-v3"; do
    python -m spinup.run trpo_mytorch \
           --seed 0 10 20 \
           --steps_per_epoch 2048 \
           --train_v_iters 10 \
           --epochs 1465 \
           --gamma 0.99 \
           --delta 0.01 \
           --lam 0.95 \
           --vf_lr 0.001 \
           --damping_coeff 0.1 \
           --cg_iters 10 \
           --backtrack_iters 10 \
           --backtrack_coeff 0.8 \
           --hid "[64,64]" \
           --env "$env" \
           --exp_name "trpo-benchmark-mytorch-$(echo $env | tr '[:upper:]' '[:lower:]')-$(date +%s)" \
           --use_gpu
done

for env in "Swimmer-v3" "Hopper-v3" "Walker2d-v3"; do
    python -m spinup.run trpo_mytorch \
           --seed 0 10 20 \
           --steps_per_epoch 2048 \
           --train_v_iters 80 \
           --epochs 1465 \
           --gamma 0.99 \
           --delta 0.01 \
           --lam 0.95 \
           --vf_lr 0.001 \
           --damping_coeff 0.1 \
           --cg_iters 10 \
           --backtrack_iters 10 \
           --backtrack_coeff 0.8 \
           --hid "[64,64]" \
           --env "$env" \
           --exp_name "trpo-benchmark-mytorch-$(echo $env | tr '[:upper:]' '[:lower:]')-$(date +%s)" \
           --use_gpu
done
