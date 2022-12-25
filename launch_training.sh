 python train.py \
        --device mps \
#        --track \
        --wandb-entity justadreamer2007 \
        --wandb-project-name dino-dqn \
        --algo dqn \
        --env DinoEnv-v0 \
        --env-kwargs mark_action:False accelerate:False \
        --tensorboard-log tblogs

#        --trained-agent logs/dqn/DinoEnv-v0_7/best_model.zip
