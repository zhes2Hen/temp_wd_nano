# you can compose any regularization method in different intervals

"""
several regularization methods:

weight decay {'method':'wd','interval':[0,30000],'wd':0.1}
Sphere Proj norm schedule {'method':'norm_schedule','interval':[30001,50000],'end_point_ratio':1.3,('start_point_ratio':None,'schedule_method':'cosine')}

"""

import os
import time
import math
import pickle
import inspect
from contextlib import nullcontext

import numpy as np
import torch
#from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
suffix = '_compose'  # filename suffix
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# regularization composition settings
composed_method=[{'method':'wd','interval':[0,600000],'wd':0.1}]
"""
another example

composed_method=[
    {
        'method':'wd',
        'interval':[0,30000],
        'wd':0.1,
    },
    {
        'method':'norm_schedule',
        'interval':[30001,50000],
        'end_point_ratio':1.3,
    }
]"""

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    """init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size"""
    raise ValueError('DDP mode is not supported.')  # we never use ddp mode
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

assert init_from=='scratch'  # we only use scratch mode
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
"""elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, f'ckpt{suffix}.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)"""
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
@torch.no_grad()
def get_c_dict():
    norms_dict = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() >= 2:
            norm_val = torch.norm(p, p='fro').item()
            norms_dict[name] = norm_val
    unwanted_prefix = '_orig_mod.'
    for name,_ in list(norms_dict.items()):
        if name.startswith(unwanted_prefix):
            norms_dict[name[len(unwanted_prefix):]] = norms_dict.pop(name)
    return norms_dict

method_time=[]
for method_idx in range(len(composed_method)):
    method=composed_method[method_idx]

    left_time,right_time = method['interval']
    if method_idx==0:
        assert left_time==0
    else:
        assert left_time==composed_method[method_idx-1]['interval'][1]+1
    if method_idx==len(composed_method)-1:
        assert right_time==max_iters
    method_time.append(left_time)

    if method['method']=='wd':
        assert method['wd']>=0
    elif method['method']=='norm_schedule':
        assert method['end_point_ratio']>=0
        if method.get('schedule_method') is not None:
            assert method['schedule_method']=='cosine'
        if method.get('start_point_ratio') is not None:
            assert method['start_point_ratio']>=0
    else:
        raise ValueError('Unknown regularization method')

temp_used_norm_dict=(None,)

def set_weight_decay_for_params(optimizer, new_wd):
    assert len(optimizer.param_groups) == 2
    success_num = 0
    for g in optimizer.param_groups:
        if g['params'][0].dim() >=2:
            g['weight_decay'] = new_wd
            success_num += 1
    if success_num == 1:
        return
    else:
        raise ValueError('can not successfully set weight_decay')

def change_method(optimizer):
    global temp_used_norm_dict
    if iter_num in method_time:
        method_index = method_time.index(iter_num)
        method = composed_method[method_index]
        if method['method']=='wd':
            temp_used_norm_dict=('wd',)

            new_wd=method['wd']
            set_weight_decay_for_params(optimizer, new_wd)

        elif method['method']=='norm_schedule':
            set_weight_decay_for_params(optimizer, 0.0)
            _norm_dict=get_c_dict()

            start_point_ratio=method.get('start_point_ratio')
            if start_point_ratio is None:
                start_point_ratio=1.0
            end_point_ratio=method['end_point_ratio']

            _optimizer_state={'interval':method['interval'],'point_ratio':(start_point_ratio, end_point_ratio),'schedule_method':'cosine'}
            temp_used_norm_dict=('norm_schedule',_norm_dict,_optimizer_state)
        else:
            raise ValueError('Unknown regularization method')


def create_sphere_constrained_adam(
    model: torch.nn.Module,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str
) -> torch.optim.Optimizer:
    named = list(model.named_parameters())
    decay = [(n, p) for n, p in named if p.requires_grad and p.dim() >= 2]
    nodecay = [(n, p) for n, p in named if p.requires_grad and p.dim() <  2]

    optim_groups = [
        {'params': [p for _, p in decay],   'weight_decay': 0.0},
        {'params': [p for _, p in nodecay], 'weight_decay': 0.0},
    ]

    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = {'fused': True} if use_fused else {}

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    orig_step = optimizer.step

    def sphere_step(closure=None):
        loss = orig_step(closure) if closure is not None else orig_step()
        # sphere proj
        if temp_used_norm_dict[0]=='norm_schedule':
            _,_norm_dict,_optimizer_state=temp_used_norm_dict
            interval_begins,interval_ends=_optimizer_state['interval']
            interval_begins-=1  # to make the length of the interval great
            start_point_ratio, end_point_ratio=_optimizer_state['point_ratio']
            assert _optimizer_state['schedule_method']=='cosine'

            _norm_decay_ratio=(iter_num-interval_begins)/(interval_ends-interval_begins)
            _norm_coeff=end_point_ratio+(1.0+math.cos(math.pi*_norm_decay_ratio))*(start_point_ratio-end_point_ratio)/2

            with torch.no_grad():
                for name, param in decay:
                    assert param.grad is not None
                    norm = torch.norm(param, p='fro')
                    if norm > 0:
                        c = _norm_dict[name]
                        param.mul_(_norm_coeff * c / norm)

        return loss

    optimizer.step = sphere_step

    return optimizer

optimizer = create_sphere_constrained_adam(model,learning_rate, (beta1, beta2), device_type)

"""if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])"""
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

"""# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])"""

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

assert not wandb_log  # we never use wandb
"""# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)"""

# datas to be saved
if master_process:
    train_loss_data=[]
    train_lr=[]

    eval_iterations=[]
    eval_train_loss_data=[]
    eval_val_loss_data=[]

    params_norms=[]

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
assert iter_num==local_iter_num
assert max_iters%eval_interval==0
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

while True:
    # change the optimizer's state
    change_method(optimizer)

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        """if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })"""
        # save data
        eval_iterations.append(iter_num)
        eval_train_loss_data.append(losses['train'].item())
        eval_val_loss_data.append(losses['val'].item())

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']

            # if iter_num > 0:
            if iter_num == max_iters:  # only save checkpoint in the last iter
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt{suffix}.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    if master_process:
        raw_loss=0.0

    for micro_step in range(gradient_accumulation_steps):
        """if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)"""
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        if master_process:
            raw_loss += loss.item()

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # we never estimate mfu
    """# timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")"""


    # save datas
    if master_process:
        norms_dict=get_c_dict()
        params_norms.append(norms_dict)

        train_lr.append(lr)
        train_loss_data.append(raw_loss)

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break


# save datas
if master_process:
    with open(os.path.join(out_dir, f'eval_data{suffix}.pkl'), 'wb') as f:
        datas = {
            "train": eval_train_loss_data,
            "val": eval_val_loss_data,
            "iteration": eval_iterations,
        }
        pickle.dump(datas, f)

    with open(os.path.join(out_dir, f'train_data{suffix}.pkl'), 'wb') as f:
        datas = {
            "lr": train_lr,
            "train": train_loss_data,
        }
        pickle.dump(datas, f)

    with open(os.path.join(out_dir, f'norms_data{suffix}.pkl'), 'wb') as f:
        pickle.dump(params_norms, f)


"""if ddp:
    destroy_process_group()"""

