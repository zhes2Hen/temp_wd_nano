# config for training GPT-2 (124M)
# launch as the following (e.g. in a screen session)
# $ python train_new.py config/train_long_gpt2_v256_new.py

wandb_log = False
out_dir = 'out-long-gpt2-v256-new'

# 16 batch size * 256 block size * 16 gradaccum * 1 GPU = 65,536 ~0.06M
batch_size = 16
block_size = 256
gradient_accumulation_steps = 16

# learning rate schedule
learning_rate = 3e-3
min_lr = 3e-4

max_iters = 50000
lr_decay_iters = 50000
warmup_iters = 1000

# eval stuff
eval_interval = 25
eval_iters = 128
log_interval = 100

# weight decay
weight_decay = 0.02
suffix = f'_wd{weight_decay}' if learning_rate == 6e-4 else f'_wd{weight_decay}_lr{learning_rate}'
