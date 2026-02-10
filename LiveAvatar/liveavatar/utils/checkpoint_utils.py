"""
独立的检查点保存工具，不依赖 Lightning 框架
"""
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)


def save_fsdp_checkpoint_with_topk(
    model,
    save_dir,
    global_step,
    only_trainable=True,
    top_k=50,
    saved_checkpoints_list=None,
    prefix="model"
):
    """
    保存 FSDP 模型检查点并管理 top_k
    
    Args:
        model: FSDP wrapped model
        save_dir: 保存目录
        global_step: 当前训练步数
        only_trainable: 是否只保存可训练参数
        top_k: 保留的最大检查点数量
        saved_checkpoints_list: 已保存检查点的列表 [(step, path), ...]
        prefix: 文件名前缀
    
    Returns:
        updated saved_checkpoints_list
    """
    if saved_checkpoints_list is None:
        saved_checkpoints_list = []
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用 FSDP 的 state_dict API 收集模型状态
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        consolidated_model_state_dict = model.state_dict()
        
        if only_trainable:
            # 获取可训练参数的名称（去掉 FSDP 包装的前缀）
            trainable_param_names = {
                name.replace('_fsdp_wrapped_module.', '') 
                for name, param in model.named_parameters() 
                if param.requires_grad
            }
            # 过滤出可训练参数
            filtered_state_dict = {
                name: tensor 
                for name, tensor in consolidated_model_state_dict.items() 
                if name in trainable_param_names
            }
        else:
            filtered_state_dict = consolidated_model_state_dict
        
        # 只有 rank 0 保存检查点
        is_rank_0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if is_rank_0:
            ckpt_path = os.path.join(save_dir, f"step_{global_step}_{prefix}.pt")
            torch.save(filtered_state_dict, ckpt_path)
            print(f"[Checkpoint] Saved: {ckpt_path}")

            # 管理保存的文件，仅保留 top_k 个
            saved_checkpoints_list.append((global_step, ckpt_path))
            saved_checkpoints_list.sort(reverse=True)  # 最新的在前

            if len(saved_checkpoints_list) > top_k:
                # 删除最旧的
                _, old_path = saved_checkpoints_list.pop()
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"[Checkpoint] Removed old: {old_path}")
        
        # 清理内存
        del consolidated_model_state_dict
        if only_trainable:
            del filtered_state_dict
    
    # 同步所有进程
    if dist.is_initialized():
        dist.barrier()
    
    return saved_checkpoints_list

