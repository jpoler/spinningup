 #!/bin/bash

for env in "HalfCheetah-v3" "Swimmer-v3" "Hopper-v3" "Walker2d-v3" "Ant-v3" "Humanoid-v3"; do
    python -m spinup.run vpg_mytorch \
           --seed 0 10 20 \
           --steps_per_epoch 2048 \
           --train_iters 10 \
           --train_minibatches 4 \
           --epochs 1465 \
           --gamma 0.99 \
           --lam 0.97 \
           --pi_lr 0.0003 \
           --vf_lr 0.001 \
           --hid "[64,64]" \
           --env "$env" \
           --exp_name "vpg-benchmark-mytorch-$(echo $env | tr '[:upper:]' '[:lower:]')-$(date +%s)"
done
