#!/bin/bash

# for env in "HalfCheetah-v3" "Swimmer-v3" "Hopper-v3" "Walker2d-v3" "Ant-v3"; do
#     python -m spinup.run ppo_mytorch \
#            --seed 0 10 20 \
#            --steps_per_epoch 2048 \
#            --train_iters 10 \
#            --train_minibatches 32 \
#            --epochs 1465 \
#            --gamma 0.99 \
#            --lam 0.95 \
#            --clip_ratio 0.2 \
#            --pi_lr 0.0003 \
#            --vf_lr 0.001 \
#            --target_kl 0.01  \
#            --entropy_bonus_coef 0.0 \
#            --hid "[64,64]" \
#            --env "$env" \
#            --exp_name "ppo-benchmark-mytorch-$(echo $env | tr '[:upper:]' '[:lower:]')-$(date +%s)"
# done

for env in "Humanoid-v3"; do
    python -m spinup.run ppo_mytorch \
           --seed 0 10 20 \
           --steps_per_epoch 10000 \
           --train_iters 10 \
           --train_minibatches 10 \
           --epochs 10 \
           --gamma 0.99 \
           --lam 0.95 \
           --clip_ratio 0.2 \
           --pi_lr 0.001 \
           --vf_lr 0.001 \
           --target_kl 0.01  \
           --entropy_bonus_coef 0.0 \
           --hid "[64,64]" \
           --env "$env" \
           --exp_name "ppo-benchmark-mytorch-$(echo $env | tr '[:upper:]' '[:lower:]')-$(date +%s)" \
           --use_gpu
done
