# config for training GPT-2 (124M)
# launch as the following (e.g. in a screen session)
# $ python train_norm_decay.py config/train_long_gpt2_norm_decay_v256.py

wandb_log = False
out_dir = 'out-long-gpt2-norm-decay-v256'

# 16 batch size * 256 block size * 16 gradaccum * 1 GPU = 65,536 ~0.06M
batch_size = 16
block_size = 256
gradient_accumulation_steps = 16

# learning rate schedule
learning_rate = 6e-4
min_lr = 6e-5
assert abs(learning_rate/10 - min_lr) < 1e-9

max_iters = 50000
lr_decay_iters = 50000
warmup_iters = 1000

# eval stuff
eval_interval = 25
eval_iters = 128
log_interval = 100 # this value is deprecated

# sphere projected regularization
begin_norm_decay_step = 5000 # the iteration to add a sphere projected regularization
norm_decay_percent = 4.0

_origin_suffix = f'_norm_decay_{begin_norm_decay_step}steps_{norm_decay_percent}'
suffix = _origin_suffix if learning_rate == 6e-4 else _origin_suffix+f'_lr{learning_rate}'
