from regex import F
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
    MixedPrecision,
    ShardingStrategy,
    FullyShardedDataParallel as FSDP
)
from torchvision.utils import make_grid
from datetime import timedelta, datetime
import torch.distributed as dist
from omegaconf import OmegaConf
from functools import partial
import numpy as np
import random
import torch
import os


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def init_logging_folder(args):
    import wandb
    date = str(datetime.now()).replace(" ", "-").replace(":", "-")
    output_path = os.path.join(
        args.output_path,
        f"{date}_seed{args.seed}"
    )
    os.makedirs(output_path, exist_ok=False)

    os.makedirs(args.output_path, exist_ok=True)
    wandb.login(host=args.wandb_host, key=args.wandb_key)
    run = wandb.init(config=OmegaConf.to_container(args, resolve=True), dir=args.output_path, **
                     {"mode": "online", "entity": args.wandb_entity, "project": args.wandb_project})
    wandb.run.log_code(".")
    wandb.run.name = args.wandb_name
    print(f"run dir: {run.dir}")
    wandb_folder = run.dir
    os.makedirs(wandb_folder, exist_ok=True)

    return output_path, wandb_folder


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def cycle(dl):
    while True:
        for data in dl:
            yield data


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def barrier():
    if dist.is_initialized():
        dist.barrier()


def prepare_for_saving(tensor, fps=16, caption=None):
    import wandb
    # Convert range [-1, 1] to [0, 1]
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1).detach()

    if tensor.ndim == 4:
        # Assuming it's an image and has shape [batch_size, 3, height, width]
        tensor = make_grid(tensor, 4, padding=0, normalize=False)
        return wandb.Image((tensor * 255).cpu().numpy().astype(np.uint8), caption=caption)
    elif tensor.ndim == 5:
        # Assuming it's a video and has shape [batch_size, num_frames, 3, height, width]
        return wandb.Video((tensor * 255).cpu().numpy().astype(np.uint8), fps=fps, format="webm", caption=caption)
    else:
        raise ValueError("Unsupported tensor shape for saving. Expected 4D (image) or 5D (video) tensor.")

def validate_tensor(tensor, expected_shape, expected_dtype, expected_device):
    """验证单个tensor的形状、dtype和device"""
    try:
        if not isinstance(tensor, torch.Tensor):
            raise Exception("tensor is not a torch.Tensor")
        if tensor.shape != expected_shape:
            raise Exception("tensor shape is not expected")
        if tensor.dtype != expected_dtype:
            raise Exception("tensor dtype is not expected")
        if tensor.device != expected_device:
            raise Exception("tensor device is not expected")
    except Exception as e:
        # import pdb;pdb.set_trace()
        print(f"error: {e}")
        return False
    return True


def check_input(noise_input,timestep,arg_c):
    """
    验证输入参数的tensor是否符合预期规格
    
    Args:
        arg_c: 包含各种tensor的字典
        
    Returns:
        bool: 所有验证通过返回True，否则返回False
    """
    try:
        bs = noise_input.__len__()      # batch_size
        noise_input = noise_input[0]      # [c,f,h,w]
        f = noise_input.shape[1]       # frames  
        h = noise_input.shape[2]       # height
        w = noise_input.shape[3]       # width
        motion_frames_rgb = arg_c["motion_frames"][0]
        motion_frames_latent = arg_c["motion_frames"][1]
        
        expected_dtype = noise_input.dtype
        expected_device = noise_input.device
        
        validation_configs = [
            ("cond_states", (bs, 16, f, h, w), True),
            ("ref_latents", (bs, 16,1, h, w), True),
            ("motion_latents", (bs, 16, motion_frames_latent, h, w), False),
            ("audio_input", (bs, 25, 1024, 4*f), True),#数据集里 1的部分在 motion_frames，后面都是 4n
            ("context", (bs, 512, 4096), True),
        ]
        if validate_tensor(timestep, (bs, f), expected_dtype, expected_device) is False:
            print(f"input timestep, shape {timestep.shape}, dtype {timestep.dtype}, device {timestep.device} is not valid")
            print(f"expected shape {bs, f}, expected dtype {expected_dtype}, expected device {expected_device}")
            return False

        for key, expected_shape, is_required in validation_configs:
            if key in arg_c:
                if not validate_tensor(arg_c[key], expected_shape, expected_dtype, expected_device):
                    print(f"input {key}, shape {arg_c[key].shape}, dtype {arg_c[key].dtype}, device {arg_c[key].device} is not valid")
                    print(f"expected shape {expected_shape}, expected dtype {expected_dtype}, expected device {expected_device}")
                    return False
            elif is_required:
                print(f"input {key} is required, but not found")
                return False
                print(f"input {key} is required, but not found")
    except Exception as e:
        print(f"error: {e}")
        return False
 
    return True