 python train.py \
        --device mps \
        --wandb-entity justadreamer2007 \
        --wandb-project-name dino-dqn \
        --algo dqn \
        --env DinoEnv-v0 \
        --tensorboard-log tblogs \
        --trained-agent logs/dqn/DinoEnv-v0_7/best_model.zip
