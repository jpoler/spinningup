# for env in "HalfCheetah-v3"; do
#     time python -m spinup.run sac_mytorch \
#            --seed 0 10 20 30 40 \
#            --epochs 50 \
#            --steps_per_epoch 10000 \
#            --max_ep_len 1000  \
#            --replay_size 1000000  \
#            --gamma 0.99 \
#            --polyak 0.995 \
#            --pi_lr 0.001 \
#            --q_lr 0.001 \
#            --alpha 0.2 \
#            --batch_size 100 \
#            --start_steps 10000 \
#            --update_after 1000 \
#            --update_every 50 \
#            --num_test_episodes 10 \
#            --hid "[256,256]" \
#            --env "$env" \
#            --exp_name "sac-benchmark-mytorch-$(echo $env | tr '[:upper:]' '[:lower:]')-$(date +%s)"
# done

# for env in "HalfCheetah-v3" "Swimmer-v3" "Hopper-v3" "Walker2d-v3" "Ant-v3"; do
for env in "Swimmer-v3" "Hopper-v3" "Walker2d-v3" "Ant-v3"; do
    time python -m spinup.run sac_mytorch \
         --seed 0 10 20 \
         --epochs 300 \
         --steps_per_epoch 10000 \
         --max_ep_len 1000  \
         --replay_size 1000000  \
         --gamma 0.99 \
         --polyak 0.995 \
         --pi_lr 0.001 \
         --q_lr 0.001 \
         --alpha 0.2 \
         --batch_size 100 \
         --start_steps 10000 \
         --update_after 1000 \
         --update_every 50 \
         --num_test_episodes 10 \
         --hid "[256,256]" \
         --env "$env" \
         --exp_name "sac-benchmark-mytorch-$(echo $env | tr '[:upper:]' '[:lower:]')-$(date +%s)"
done

for env in "Humanoid-v3"; do
    python -m spinup.run sac_mytorch \
           --seed 0 10 20 \
           --epochs 300 \
           --steps_per_epoch 10000 \
           --max_ep_len 1000  \
           --replay_size 1000000  \
           --gamma 0.99 \
           --polyak 0.995 \
           --pi_lr 0.0003 \
           --q_lr 0.0003 \
           --alpha 0.05 \
           --batch_size 100 \
           --start_steps 10000 \
           --update_after 1000 \
           --update_every 50 \
           --num_test_episodes 10 \
           --hid "[256,256]" \
           --env "$env" \
           --exp_name "sac-benchmark-mytorch-$(echo $env | tr '[:upper:]' '[:lower:]')-$(date +%s)"
done
