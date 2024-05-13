import argparse
import os
import time
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a GPT model on the OpenWebText dataset.")
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory for checkpoints and logs.')
    parser.add_argument('--eval_interval', type=int, default=2000, help='Number of iterations between evaluations.')
    parser.add_argument('--log_interval', type=int, default=1, help='Number of iterations between logging.')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of iterations to run during evaluation.')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only, no training.')
    parser.add_argument('--always_save_checkpoint', action='store_true', default=True, help='Always save a checkpoint after each eval.')
    parser.add_argument('--init_from', type=str, default='scratch', choices=['scratch', 'resume', 'gpt2'], help='Initialize model from scratch, resume, or use GPT-2 weights.')
    parser.add_argument('--wandb_log', action='store_true', help='Enable logging to Weights & Biases.')
    parser.add_argument('--wandb_project', type=str, default='owt', help='Weights & Biases project name.')
    parser.add_argument('--wandb_run_name', type=str, default='gpt2', help='Weights & Biases run name.')
    parser.add_argument('--dataset', type=str, default='openwebtext', help='Dataset to train on.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=40, help='Number of gradient accumulation steps.')
    parser.add_argument('--batch_size', type=int, default=12, help='Mini-batch size.')
    parser.add_argument('--block_size', type=int, default=1024, help='Block size for training.')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of layers in the model.')
    parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads in the model.')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension size.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--bias', action='store_true', help='Use bias in LayerNorm and Linear layers.')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate.')
    parser.add_argument('--max_iters', type=int, default=600000, help='Maximum number of training iterations.')
    parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay for optimization.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 hyperparameter for Adam optimizer.')
    parser.add_argument('--beta2', type=float, default=0.95, help='Beta2 hyperparameter for Adam optimizer.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value.')
    parser.add_argument('--decay_lr', action='store_true', help='Enable learning rate decay.')
    parser.add_argument('--warmup_iters', type=int, default=2000, help='Number of warmup iterations for learning rate.')
    parser.add_argument('--lr_decay_iters', type=int, default=600000, help='Iterations to decay the learning rate over.')
    parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate.')
    parser.add_argument('--backend', type=str, default='nccl', help='Backend for distributed training.')
    parser.add_argument('--device', type=str, default='cuda', help='Device for training.')
    parser.add_argument('--compile', action='store_true', help='Compile the model for faster training.')
    return parser.parse_args()

def initialize_ddp_environment(args):
    ddp = int(os.environ.get('RANK', -1)) != -1  # Detect if launched under a DDP context
    ddp_rank, ddp_local_rank, ddp_world_size = -1, -1, 1
    if ddp:
        init_process_group(backend=args.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        args.device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(args.device)
    master_process = ddp_rank == 0 or not ddp
    seed_offset = ddp_rank if ddp else 0
    return ddp, master_process, seed_offset, ddp_local_rank, ddp_world_size

def setup_logging_and_checkpoints(args, master_process, seed_offset):
    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)

def get_model_and_optimizer(args, ddp=False, ddp_local_rank=None):
    if args.init_from == 'scratch':
        print("Initializing a new model from scratch.")
        model_config = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                                 bias=args.bias, vocab_size=50304, dropout=args.dropout)  # Assuming GPT-2 vocab size
        model = GPT(model_config)
    elif args.init_from == 'resume':
        print(f"Resuming training from {args.out_dir}")
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        model_config = GPTConfig(**checkpoint['model_args'])
        model = GPT(model_config)
        model.load_state_dict(checkpoint['model'])
    elif args.init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {args.init_from}")
        model = GPT.from_pretrained(args.init_from)
    model.to(args.device)
    if args.compile:
        print("Compiling the model... (may take a minute)")
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay)
    return model, optimizer

def get_batch(dataset, args, split='train'):
    data_dir = os.path.join('data', dataset)
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
    return x.to(args.device), y.to(args.device)

def evaluate(model, args):
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        X, Y = get_batch(args.dataset, args, split='val')
        with torch.no_grad():
            logits, loss = model(X, Y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_one_epoch(model, optimizer, args, scheduler=None, ddp=False, ddp_world_size=1):
    start_time = time.time()
    running_mfu = -1.0  # Model Forward Usage, example metric for monitoring
    iterator = tqdm(range(args.max_iters), desc="Training", total=args.max_iters)

    for step in iterator:
        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at the start of a new accumulation cycle

        for _ in range(args.gradient_accumulation_steps):
            X, Y = get_batch(args.dataset, args)
            logits, loss = model(X, Y)
            loss = loss / args.gradient_accumulation_steps  # Normalize loss to account for accumulation
            loss.backward()
            total_loss += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if scheduler:
            scheduler.step()

        # Time and performance metrics calculation
        elapsed_time = time.time() - start_time
        start_time = time.time()  # Reset start time after each log

        if step % args.log_interval == 0:
            if running_mfu == -1.0:
                running_mfu = elapsed_time  # Initialize if it hasn't been set
            else:
                running_mfu = 0.9 * running_mfu + 0.1 * elapsed_time  # Exponential moving average

            iterator.set_description(
                f"iter {step}: loss {total_loss:.4f}, time {elapsed_time * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
            )

        if step % args.eval_interval == 0 and (ddp and step % (args.eval_interval * ddp_world_size) == 0):
            val_loss = evaluate(model, args)
            print(f"Evaluation Loss at step {step}: {val_loss}")

        if args.eval_only:
            break

    if ddp:
        destroy_process_group()


def main():
    args = parse_arguments()
    ddp, master_process, seed_offset, ddp_local_rank, ddp_world_size = initialize_ddp_environment(args)
    setup_logging_and_checkpoints(args, master_process, seed_offset)
    model, optimizer = get_model_and_optimizer(args, ddp, ddp_local_rank)
    from hiq.vis import print_model
    print_model(model)

    # Set up a learning rate scheduler if needed
    if args.decay_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_decay_iters,
                                                               eta_min=args.min_lr)
    else:
        scheduler = None

    # Training loop
    try:
        train_one_epoch(model, optimizer, args, scheduler, ddp, ddp_world_size)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        if ddp:
            destroy_process_group()


if __name__ == "__main__":
    main()