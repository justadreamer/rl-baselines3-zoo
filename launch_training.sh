git pull
python train.py \
      --device mps \
      --wandb-entity justadreamer2007 \
      --wandb-project-name dino-dqn \
      --algo dqn \
      --env DinoEnv-v0 \
      --env-kwargs mark_action:True accelerate:True \
      --eval-freq 4000 \
      --tensorboard-log tblogs \
      --track \
#      --trained-agent logs/dqn/DinoEnv-v0_40/best_model.zip
