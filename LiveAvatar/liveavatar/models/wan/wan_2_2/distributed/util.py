# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist


# Global variable to store the sequence parallel group
_SP_GROUP = None

_GLOBAL_SEQ_LEN = None


def init_distributed_group():
    """r initialize sequence parallel group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')


def get_or_create_sp_group(sp_size=None):
    """
    Get or create sequence parallel process group based on sp_size.
    
    For example, if world_size=4 and sp_size=2:
    - rank 0,1 form one group
    - rank 2,3 form another group
    
    Args:
        sp_size: Size of sequence parallel group. If None, use world_size.
    
    Returns:
        ProcessGroup for sequence parallel communication
    """
    global _SP_GROUP
    
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # If sp_size is None or equals world_size, use the default group
    if sp_size is None or sp_size == world_size:
        return None  # None means use default group
    
    # If already created and stored, return it
    # Note: This assumes sp_size doesn't change during runtime
    if _SP_GROUP is not None:
        return _SP_GROUP
    
    # Create process groups: divide world into groups of sp_size
    # e.g., world_size=4, sp_size=2: groups are [0,1] and [2,3]
    assert world_size % sp_size == 0, f"world_size {world_size} must be divisible by sp_size {sp_size}"
    
    num_groups = world_size // sp_size
    for i in range(num_groups):
        ranks = list(range(i * sp_size, (i + 1) * sp_size))
        group = dist.new_group(ranks=ranks)
        # Store the group if current rank belongs to it
        if rank in ranks:
            _SP_GROUP = group
    
    return _SP_GROUP


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def all_to_all(x, scatter_dim, gather_dim, group=None, sp_size=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = sp_size if sp_size is not None else get_world_size()
    if world_size > 1:
        # If group is not provided, get or create the sp group
        if group is None and sp_size is not None:
            group = get_or_create_sp_group(sp_size)
        
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_gather(tensor, sp_size=None, group=None):
    world_size = sp_size if sp_size is not None else dist.get_world_size()
    if world_size == 1:
        return [tensor]
    
    # If group is not provided, get or create the sp group
    if group is None and sp_size is not None:
        group = get_or_create_sp_group(sp_size)
    
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor, group=group)
    return tensor_list


def gather_forward(input, dim, sp_size=None, group=None):
    # skip if world_size == 1
    world_size = sp_size if sp_size is not None else dist.get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input, sp_size=sp_size, group=group)
    output = torch.cat(output, dim=dim).contiguous()
    if _GLOBAL_SEQ_LEN is not None:
        output = torch.narrow(output, dim, 0, _GLOBAL_SEQ_LEN)
    return output


def pad_chunk(tensor, sp_size, dim, pad_value=0):
    global _GLOBAL_SEQ_LEN
    orig_len = tensor.size(dim)
    _GLOBAL_SEQ_LEN = orig_len
    pad_len = (sp_size - orig_len % sp_size) % sp_size
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_len
    pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    padded = torch.cat([tensor, pad_tensor], dim=dim)
    chunks = torch.chunk(padded, sp_size, dim=dim)
    sp_rank = get_rank() % sp_size
    return chunks[sp_rank], orig_len
