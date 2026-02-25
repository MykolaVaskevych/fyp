# DQN×Pong n_envs mismatch

Seeds 1-7 of DQN×Pong were trained with `n_envs=8` (eval checkpoints at exact 25,000-step intervals).
Seeds 8-10 (and all other algo×env combos) use `n_envs=16` (checkpoints at ~24,992-step intervals).

evaluate.py truncates to min checkpoint count, so metrics still work. But if the mismatch
is noticeable in learning curves, rerun the 7 completed DQN×Pong seeds with n_envs=16:

    rm -rf results/dqn/pong/
    uv run python train.py --algo dqn --env Pong --device cuda
