import os
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from PIL import Image


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


def save_frames_to_png_mp4(tensor, save_dir,video_save_dir=None):
    img_list=[]
    os.makedirs(save_dir, exist_ok=True)
    for i in range(tensor.size(0)):
        frame = tensor[i]
        normalized = frame

        uint8_frame = (normalized * 255).type(torch.uint8)
        img = Image.fromarray(uint8_frame.numpy(), mode='L')
        # resize from 45*30 to 720*480
        img = img.resize((720, 480), Image.Resampling.NEAREST)
        img_path = f"{save_dir}/frame_{i:02d}.png"
        img.save(img_path)
        for i in range(4 if i!=0 else 1):
            img_list.append(img)
    # save these pngs merged into mp4
    if video_save_dir is None:
        video_save_dir=save_dir+"/output.mp4"
    imageio.mimsave(video_save_dir, img_list, fps=25)


def draw_routing_logit(routing_logits, base_dir="assets/output/tempfile", suffix="",video_save_dir=None,use_softmax=True):
    routing_logit=routing_logits[1] # bs,bs1 is uncond, not meaningful, take the second batch i.e. full condition
    torch.save(routing_logit, base_dir+"routing_logit"+suffix+".pt")
    if use_softmax:
        routing_logit = torch.softmax(routing_logit.float(), dim=-1).to(routing_logit.dtype)
    routing_logit = routing_logit.squeeze(0).cpu().float()
    logit_0 = routing_logit[:, 0].view(13, 30, 45)
    logit_1 = routing_logit[:, 1].view(13, 30, 45)
    prefix="routing_logit_0"
    video_save_dir_0 = video_save_dir.replace(".mp4", "_0.mp4")
    save_frames_to_png_mp4(logit_0, base_dir+prefix+suffix,video_save_dir_0)
    prefix="routing_logit_1"
    video_save_dir_1 = video_save_dir.replace(".mp4", "_1.mp4")
    save_frames_to_png_mp4(logit_1, base_dir+prefix+suffix,video_save_dir_1)

def process_single_mask_dir(mask_dir):
    """Process masks from a single directory"""
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    dense_masks_dict = []

    for frame in range(len(mask_files)):
        mask_path = os.path.join(mask_dir, f"annotated_frame_{int(frame):05d}.png")
        mask_array = np.array(Image.open(mask_path))
        binary_mask = np.where(mask_array > 0, 1, 0).astype(np.uint8)
        dense_masks_dict.append(binary_mask)

    # Convert to tensor
    dense_masks = torch.stack([torch.from_numpy(m) for m in dense_masks_dict])  # [T, H, W]
    dense_masks = dense_masks.unsqueeze(0)  # [B=1, T, H, W]

    return dense_masks


def process_masks_to_routing_logits(base_dir, shape=None):
    """Process masks to teacher routing logits following the original codebase exactly
    
    Args:
        base_dir: Directory containing subdirectories '1', '2', '3', ... with mask files
        shape: Optional tuple (B, T, H, W) specifying target shape. 
               If None, uses default (1, 13, 60, 90)
    """

    # ----------------- Part 1: Auto-detect actor directories and load masks -----------------
    # 自动检测有多少个角色目录（"1", "2", "3", ...）
    actor_dirs = []
    actor_id = 1
    while True:
        dir_path = os.path.join(base_dir, str(actor_id))
        if os.path.exists(dir_path):
            actor_dirs.append(dir_path)
            actor_id += 1
        else:
            break

    if len(actor_dirs) < 1:
        raise ValueError(f"No actor directories found in {base_dir}")

    num_actors = len(actor_dirs)
    print(f"Found {num_actors} actor(s) in {base_dir}")

    # 加载所有角色的 mask
    all_dense_masks = []
    for idx, dir_path in enumerate(actor_dirs, 1):
        print(f"Loading masks for actor {idx} from {dir_path}")
        dense_mask = process_single_mask_dir(dir_path)
        all_dense_masks.append(dense_mask)

    # ----------------- Part 2: Process masks (from train.py) -----------------
    if shape is None:
        B = 1  # Fixed batch size
        T = 13  # Fixed temporal length
        H = 60  # Fixed height
        W = 90  # Fixed width
        assert False, "shape is not provided"
    else:
        B, T, H, W = shape
    
    p = 2   # Fixed patch size

    # Create fake latent tensor
    fake_latent = torch.zeros((B, 1, T, H//p, W//p))

    # Initialize index_mask as background (-1)
    index_mask = torch.full((B, 1, T, H//p, W//p), -1, dtype=torch.long)

    # 处理所有 mask
    all_resized_masks = []
    for idx, dense_mask in enumerate(all_dense_masks, 1):
        print(f"Processing mask for actor {idx}")
        current_mask = dense_mask.to(memory_format=torch.contiguous_format).float()
        current_mask = current_mask.unsqueeze(1)  # [B, 1, T, H, W]
        
        # Resize mask
        resized_mask = resize_mask(current_mask, fake_latent, process_first_frame_only=False)
        binary_mask = (resized_mask > 0.5).long()
        all_resized_masks.append(binary_mask)

    # 为每个角色分配标签 (0, 1, 2, ...)
    # Fill index_mask with labels:
    # - background (-1) -> will remain -1 if not covered by any mask
    # - actor 1 (mask 0) -> 0
    # - actor 2 (mask 1) -> 1
    # - actor 3 (mask 2) -> 2
    # - ...
    for actor_idx, binary_mask in enumerate(all_resized_masks):
        index_mask = torch.where(binary_mask == 1, torch.tensor(actor_idx, dtype=torch.long), index_mask)

    # Remove channel dimension and reshape
    index_mask = index_mask.squeeze(1)  # [B, T, H//p, W//p]
    index_mask = index_mask.reshape(B, -1)  # [B, len]

    # ----------------- Part 3: Generate teacher routing logits -----------------
    # 动态设置 routing_logits 维度，最后一维是角色数
    routing_logit_shape = (1, index_mask.shape[1], num_actors)  # [1, T*H//p*W//p, num_actors]
    teacher_routing_logit = torch.zeros(routing_logit_shape)

    # Set logits according to mask values:
    # background (-1) -> [0, 0, ..., 0]
    # actor 1 (0) -> [1, 0, ..., 0]
    # actor 2 (1) -> [0, 1, ..., 0]
    # actor 3 (2) -> [0, 0, 1, ..., 0]
    # ...
    for actor_idx in range(num_actors):
        teacher_routing_logit[0, index_mask[0] == actor_idx, actor_idx] = 1
        print(f"Actor {actor_idx + 1}: {(index_mask[0] == actor_idx).sum().item()} pixels assigned")

    return teacher_routing_logit


def get_routing_logits_from_tracking_mask_results(tracking_mask_results_dir,is_draw_video=False,video_save_dir="assets/output/tempfile/temp_mask.mp4"):
    routing_logits = process_masks_to_routing_logits(tracking_mask_results_dir)

    if is_draw_video:
        routing_logits_draw=[ None ,routing_logits]
        video_dir = os.path.dirname(video_save_dir)
        if video_dir and not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        draw_routing_logit(routing_logits=routing_logits_draw,suffix="test1",video_save_dir=video_save_dir,use_softmax=False)

    return routing_logits
