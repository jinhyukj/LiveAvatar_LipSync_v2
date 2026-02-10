import imageio, os, torch, warnings, torchvision, argparse, json
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs



class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100) # ms
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")   
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)             
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames
    
    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, 
                 save_full_checkpoint_steps=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.save_full_checkpoint_steps = save_full_checkpoint_steps


    def on_step_end(self, accelerator, model, save_steps=None, optimizer=None, scheduler=None, 
                    epoch_id=0, wandb_run_id=None):
        self.num_steps += 1
        
        # Save trainable-only checkpoint
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        
        # Save full checkpoint (less frequently)
        if (self.save_full_checkpoint_steps is not None and 
            self.num_steps % self.save_full_checkpoint_steps == 0):
            self.save_full_checkpoint(accelerator, model, optimizer, scheduler, 
                                     epoch_id, wandb_run_id, f"step-{self.num_steps}-full.pt")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None, optimizer=None, scheduler=None,
                       epoch_id=0, wandb_run_id=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        
        # Always save final full checkpoint
        if self.save_full_checkpoint_steps is not None:
            self.save_full_checkpoint(accelerator, model, optimizer, scheduler,
                                     epoch_id, wandb_run_id, f"step-{self.num_steps}-full.pt")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


    def save_full_checkpoint(self, accelerator, model, optimizer, scheduler, 
                            epoch_id, wandb_run_id, file_name):
        """Save complete training state including optimizer and scheduler."""
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Get model state dict (trainable params only)
            state_dict = accelerator.get_state_dict(model)
            trainable_state = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            trainable_state = self.state_dict_converter(trainable_state)
            
            # Build full checkpoint
            checkpoint = {
                'model': trainable_state,
                'optimizer': optimizer.state_dict() if optimizer else None,
                'scheduler': scheduler.state_dict() if scheduler else None,
                'epoch': epoch_id,
                'global_step': self.num_steps,
            }
            
            # Optional: save W&B run_id for continuation
            if wandb_run_id is not None:
                checkpoint['wandb_run_id'] = wandb_run_id
            
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            torch.save(checkpoint, path)
            print(f"[Checkpoint] Saved full training state to {file_name}")


def load_model_checkpoint(checkpoint_path, model):
    """
    Load model weights from checkpoint (LoRA + audio layers).
    Called during WanTrainingModule.__init__ after model construction.
    
    Args:
        checkpoint_path: Path to .safetensors or .pt checkpoint
        model: WanTrainingModule instance with pipe.dit initialized
        
    Returns:
        dict with 'checkpoint_data' for later training state loading
    """
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[Resume] Checkpoint not found: {checkpoint_path}")
    
    print(f"[Resume] Loading model checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state = load_file(checkpoint_path)
        checkpoint_data = {'is_full': False, 'path': checkpoint_path}
    else:
        try:
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            state = torch.load(checkpoint_path, map_location='cpu')
        is_full = isinstance(state, dict) and 'optimizer' in state
        checkpoint_data = {'is_full': is_full, 'path': checkpoint_path, 'full_state': state if is_full else None}
        
        # Extract model weights
        if is_full:
            state = state['model']
    
    # Load into pipe.dit (CausalWan model with LoRA + audio)
    missing, unexpected = model.pipe.dit.load_state_dict(state, strict=False)
    
    # Filter out expected missing/unexpected keys
    expected_missing = [k for k in missing if 'lora' not in k and 'audio' not in k]
    if len(expected_missing) > 0:
        print(f"[Resume] Warning: Missing non-trainable keys: {len(expected_missing)}")
    
    print(f"[Resume] Loaded model weights: {len(state)} keys from checkpoint")
    print(f"[Resume] - LoRA + audio layers restored from checkpoint")
    
    return checkpoint_data


def load_training_state(checkpoint_data, optimizer, scheduler, resume_mode="auto"):
    """
    Load optimizer and scheduler state from checkpoint.
    Called in training loop after optimizer/scheduler are created.
    
    Args:
        checkpoint_data: dict returned by load_model_checkpoint
        optimizer: Optimizer instance to restore state into
        scheduler: Scheduler instance to restore state into
        resume_mode: 'auto', 'full', or 'model_only'
        
    Returns:
        dict with keys: global_step, epoch, wandb_run_id
    """
    metadata = {'global_step': 0, 'epoch': 0, 'wandb_run_id': None}
    
    if not checkpoint_data['is_full']:
        print(f"[Resume] Trainable-only checkpoint, skipping optimizer/scheduler load")
        return metadata
    
    if resume_mode == "model_only":
        print(f"[Resume] resume_mode='model_only', skipping optimizer/scheduler load")
        return metadata
    
    # Load full state
    state = checkpoint_data['full_state']
    
    if 'optimizer' in state and optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        print(f"[Resume] Loaded optimizer state")
    
    if 'scheduler' in state and scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
        print(f"[Resume] Loaded scheduler state")
    
    metadata['global_step'] = state.get('global_step', 0)
    metadata['epoch'] = state.get('epoch', 0)
    metadata['wandb_run_id'] = state.get('wandb_run_id', None)
    
    print(f"[Resume] Training state: global_step={metadata['global_step']}, epoch={metadata['epoch']}")
    
    return metadata

def collate_dict_batch(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    if len(batch) == 1:
        return batch[0]
    
    first_sample = batch[0]
    has_tensors = any(isinstance(v, torch.Tensor) for v in first_sample.values())
    
    if not has_tensors:
        return batch
    
    result = {}
    for key in first_sample.keys():
        values = [b[key] for b in batch if key in b]
        if len(values) == 0:
            continue
            
        first_val = values[0]
        if isinstance(first_val, torch.Tensor):
            if first_val.dim() >= 1 and first_val.shape[0] == 1:
                result[key] = torch.cat(values, dim=0)
            else:
                result[key] = torch.stack(values, dim=0)
        elif isinstance(first_val, (list, tuple)) and len(first_val) > 0 and isinstance(first_val[0], torch.Tensor):
            result[key] = [torch.stack([v[i] for v in values], dim=0) for i in range(len(first_val))]
        else:
            result[key] = values
    
    return result


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    batch_size: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        batch_size = getattr(args, 'batch_size', 1)
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collate_dict_batch, 
        num_workers=num_workers,
        drop_last=(batch_size > 1),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in tqdm(enumerate(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data, return_inputs=True)
                torch.save(data, save_path)



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument(
        "--val_num_frames",
        type=int,
        default=None,
        help="Max number of frames per video for validation-only runs (--run_val_mode/--run_val_audio_cfg). "
             "If None, falls back to --num_frames.",
    )
    parser.add_argument(
        "--val_output_dir",
        type=str,
        default=None,
        help="Output directory for validation-only videos (val_standard_*/val_audio_cfg_*). "
             "If None, videos are written in the current working directory.",
    )
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    # Precomputed inputs (optional)
    parser.add_argument("--use_precomputed_context", default=False, action="store_true", help="Use precomputed text embeddings and skip prompt encoder.")
    parser.add_argument("--use_precomputed_latents", default=False, action="store_true", help="Use precomputed VAE latents and skip VAE encoding.")
    parser.add_argument("--precomputed_context_key", type=str, default="context_path", help="Metadata key containing path to precomputed text embeddings (.pt).")
    parser.add_argument("--precomputed_latents_key", type=str, default="vae_latents_path", help="Metadata key containing path to precomputed VAE latents (.pt).")
    parser.add_argument(
        "--precomputed_negative_context_path",
        type=str,
        default="/mnt/dataset1/jinhyuk/Hallo3/cropped_only_10K_preprocessed/text_emb/negative_embeddings.pt",
        help="Absolute or dataset-relative path to a precomputed negative/unconditional text embedding (.pt). Used when text dropout triggers while using precomputed context.",
    )
    # Self-Forcing discrete timestep support
    parser.add_argument("--sf_restrict_timesteps", default=False, action="store_true", help="Restrict training timesteps to a warped denoising_step_list like Self-Forcing.")
    parser.add_argument("--sf_denoising_step_list", type=str, default="1000,750,500,250", help="Comma-separated denoising steps (e.g., 1000,750,500,250).")
    parser.add_argument("--sf_warp_denoising_step", default=True, action="store_true", help="Apply Self-Forcing warp: use scheduler.timesteps[1000 - step].")
    parser.add_argument("--sf_timestep_shift", type=float, default=5.0, help="Scheduler shift used to compute timesteps (should match Self-Forcing config).")
        # Weights & Biases (optional)
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Enable Weights & Biases logging (main process only).")
    parser.add_argument("--wandb_project", type=str, default="DiffSynth", help="W&B project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team) name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name.")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags.")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="W&B API key to force login for this run.")
    parser.add_argument("--wandb_log_every", type=int, default=10, help="Log every N steps.")
    # Memory / precision
    parser.add_argument("--enable_gc", default=False, action="store_true", help="Enable gradient checkpointing for DiT/CausalWan")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Accelerate mixed precision mode (default: bf16)",
    )
    parser.add_argument("--audio_frames_per_block", type=int, default=3, help="Frames per causal block in audio path (memory knob)")
    # Optional: integrate external causal WAN model
    parser.add_argument("--use_causal_wan", default=False, action="store_true", help="Replace DiT with external CausalWanModel.")
    parser.add_argument("--causal_wan_model_file", type=str, default=None, help="Path to causal_model.py defining CausalWanModel.")
    parser.add_argument("--causal_wan_config", type=str, default=None, help="Optional: path to config.json for CausalWanModel.")
    parser.add_argument("--causal_wan_weights", type=str, default=None, help="Path to pretrained weights (.pt/.pth/.safetensors) for CausalWanModel.")
    parser.add_argument("--causal_wan_kwargs", type=str, default=None, help="JSON string of extra kwargs for CausalWanModel (e.g., {\"use_audio\":true,\"in_dim\":33}).")
    # Optional: LoRA for external CausalWanModel via PEFT
    parser.add_argument("--causal_wan_lora_rank", type=int, default=None, help="If set, apply LoRA with this rank to CausalWanModel.")
    parser.add_argument("--causal_wan_lora_alpha", type=float, default=64.0, help="LoRA alpha for CausalWanModel.")
    parser.add_argument("--causal_wan_lora_targets", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Comma list of target module names for LoRA.")
    parser.add_argument("--causal_wan_lora_init", type=str, default="kaiming", help="LoRA weight init scheme.")
    # Optional: control whether to use LoRA or full fine-tuning for CausalWan
    parser.add_argument("--causal_wan_use_lora", default=True, action="store_true", 
        help="Use LoRA for CausalWan parameter-efficient fine-tuning (default: True). Only applies when --causal_wan_lora_rank is set.")
    parser.add_argument("--causal_wan_full_finetune", default=False, action="store_true",
        help="Full fine-tune CausalWan model weights instead of LoRA (disables LoRA, overrides --causal_wan_use_lora)")
    # New: condition dropout controls (CFG training)
    parser.add_argument("--enable_text_dropout", default=False, action="store_true", help="Enable classifier-free text dropout during training.")
    parser.add_argument("--text_dropout_prob", type=float, default=0.0, help="Probability to drop text (set prompt to empty string).")
    parser.add_argument("--enable_audio_dropout", default=False, action="store_true", help="Enable classifier-free audio dropout during training.")
    parser.add_argument("--audio_dropout_prob", type=float, default=0.0, help="Probability to drop audio (zero audio_emb).")
    parser.add_argument("--enable_image_dropout", default=False, action="store_true", help="Enable classifier-free image/CLIP dropout during training.")
    parser.add_argument("--image_dropout_prob", type=float, default=0.0, help="Probability to drop image condition (zero clip_feature).")
    # New: warm-start audio modules from OmniAvatar ckpt
    parser.add_argument("--init_audio_from_omni", default=False, action="store_true", help="Initialize audio modules from OmniAvatar checkpoint.")
    parser.add_argument("--omni_ckpt_path", type=str, default="/home/work/.local/Self-Forcing-Omniavatar/OmniAvatar/pretrained_models/OmniAvatar-1.3B/pytorch_model.pt", help="Path to OmniAvatar audio checkpoint.")
    parser.add_argument("--patch_embedding_trainable", default=True, action="store_true", help="Train patch embedding.")
    # Wav2Vec online audio feature extraction
    parser.add_argument(
        "--extract_audio_embeddings_online",
        default=False,
        action="store_true",
        help="Extract audio embeddings online using Wav2Vec2 instead of loading precomputed .pt files."
    )
    parser.add_argument(
        "--wav2vec_checkpoint_path",
        type=str,
        default="/mnt/data1/hyunbin/_from_dataset2/Self-Forcing_audio_conditioning/assets/checkpoints/wav2vec/wav2vec2-base-960h",
        help="Path to Wav2Vec2 checkpoint directory for online audio feature extraction."
    )
    parser.add_argument(
        "--audio_sample_rate",
        type=int,
        default=16000,
        help="Audio sample rate for Wav2Vec2 feature extraction (default: 16000 Hz)."
    )
    parser.add_argument(
        "--use_stableavatar_audio",
        default=False,
        action="store_true",
        help="Use StableAvatar-style audio embeddings (~2:1 token-to-frame ratio, 768-dim). "
             "Requires 'total_frames' column in metadata CSV."
    )
    # Lipsync toggles
    parser.add_argument("--lipsync_use_latent_masking", default=False, action="store_true", help="Use latent-space masking (old path) instead of default RGB masking + VAE re-encode.")
    parser.add_argument("--lipsync_use_RGB_masking", default=False, action="store_true", help="Use RGB masking instead of latest masking method.")
    parser.add_argument("--lipsync_use_wan_masking", default=False, action="store_true", help="Use Wan masking instead of latest masking method.")
    parser.add_argument("--lipsync_use_VAE_masking", default=False, action="store_true", help="Use VAE masking instead of latest masking method.")
    parser.add_argument("--lipsync_loss_mouth_weight", type=float, default=1.0,
        help="Weight multiplier for mouth region in loss computation. 1.0=no weighting, 3.0=3x weight for mouth. "
             "Only active when --lipsync_use_VAE_masking is enabled.")

    # LatentSync Stage 2 Training Arguments
    parser.add_argument("--latentsync_stage2", default=False, action="store_true",
        help="Enable LatentSync stage 2 training with auxiliary losses (LPIPS, TREPA, Sync)")
    parser.add_argument("--lipsync_use_VAE_masking_latentsync", default=False, action="store_true",
        help="Use LatentSync-style VAE masking with fixed mouth mask")
    parser.add_argument("--latentsync_mask_path", type=str, default="/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png",
        help="Path to custom LatentSync mask PNG (optional, defaults to bundled mask)")

    # Auxiliary Model Paths for LatentSync
    parser.add_argument("--syncnet_config_path", type=str, default="/mnt/data1/hyunbin/LatentSync/configs/syncnet/syncnet_16_pixel_attn.yaml",
        help="Path to SyncNet config YAML file")
    parser.add_argument("--syncnet_checkpoint_path", type=str, default="/mnt/data1/hyunbin/LatentSync/checkpoints/stable_syncnet.pt",
        help="Path to pretrained SyncNet checkpoint (.pt file)")
    parser.add_argument("--trepa_checkpoint_path", type=str, default="/mnt/data1/hyunbin/LatentSync/checkpoints/auxiliary/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
        help="Path to TREPA VideoMAE checkpoint")

    # LatentSync Loss Weights
    parser.add_argument("--latentsync_recon_weight", type=float, default=1.0,
        help="Weight for reconstruction (MSE) loss in LatentSync stage 2")
    parser.add_argument("--latentsync_sync_weight", type=float, default=0.05,
        help="Weight for audio-visual sync loss in LatentSync stage 2")
    parser.add_argument("--latentsync_lpips_weight", type=float, default=0.1,
        help="Weight for LPIPS perceptual loss in LatentSync stage 2")
    parser.add_argument("--latentsync_trepa_weight", type=float, default=10.0,
        help="Weight for TREPA temporal consistency loss in LatentSync stage 2")

    # LatentSync Memory Management
    parser.add_argument("--latentsync_sync_len", type=int, default=5,
        help="Number of LATENT frames to decode for LatentSync losses (default: 5 → ~17 RGB frames). "
             "Reduce this value to save memory during VAE decoding.")
    parser.add_argument("--use_vae_gradient_checkpointing", default=False, action="store_true",
        help="Enable gradient checkpointing in VAE decoder to reduce memory usage during backprop. "
             "Reduces memory from ~48GB to ~10-15GB for 5 latent frames, with minimal speed impact.")

    # Chunked Sync Loss Arguments
    parser.add_argument("--use_chunked_sync_loss", default=False, action="store_true",
        help="Enable chunked sync loss to supervise more frames (default: False, uses single-chunk sync loss on first 16 frames)")
    parser.add_argument("--sync_chunk_size", type=int, default=16,
        help="Number of frames per chunk for sync loss (SyncNet requirement)")
    parser.add_argument("--sync_chunk_stride", type=int, default=8,
        help="Stride between chunks for sync loss (8 = 50%% overlap)")
    parser.add_argument("--sync_num_supervised_frames", type=int, default=80,
        help="Total number of frames to supervise with sync loss (out of 81 RGB frames)")

    parser.add_argument("--kv_cache_size", type=int, default=21, help="KV cache size in latent frames. e.g. 21 ")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per GPU. Default 1 for backward compatibility.")
    # Checkpoint resumption arguments
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
        help="Path to checkpoint (.safetensors or .pt) to resume training from")
    parser.add_argument("--resume_mode", type=str, default="auto", 
        choices=["auto", "full", "model_only"],
        help="Resume mode: auto (load all available), full (require optimizer/scheduler), model_only (weights only)")
    parser.add_argument("--resume_training", default=False, action="store_true",
        help="Continue global_step counter from checkpoint (default: reset to 0)")
    parser.add_argument("--save_full_checkpoint_steps", type=int, default=1000,
        help="Save full training state (model+optimizer+scheduler) every N steps")
    parser.add_argument("--causal_wan_adapter_weights", type=str, default=None,
        help="Path to adapter checkpoint (.safetensors) for CausalWan LoRA/audio weights (loaded during model init)")
    parser.add_argument("--training_stage", type=int, default=1, choices=[1, 2, 3],
        help="Training stage: 1 (teacher forcing), 2 (self-forcing lite), 3 (full self-forcing with trajectory)")
    parser.add_argument("--use_stableavatar", default=False, action="store_true",
        help="Use StableAvatar instead of OmniAvatar")
    parser.add_argument("--init_from_stableavatar", default=False, action="store_true",
        help="Initialize audio and cross-attention weights from StableAvatar checkpoint")
    parser.add_argument("--stableavatar_ckpt_path", type=str, default=None,
        help="Path to StableAvatar checkpoint (.safetensors) to load audio weights from")
    parser.add_argument("--clip_model_path", type=str, default=None,
        help="Path to CLIP model (.pth) to load image encoder from. Optional, only needed for StableAvatar.")
    # validation
    parser.add_argument("--validation_dataset_metadata_path", type=str, default=None,
        help="Path to the metadata file of the validation dataset.")
    parser.add_argument("--validation_steps", type=int, default=1000,
        help="Number of steps to validate the model.")
    parser.add_argument("--replace_gt", default=False, action="store_true",
        help="Replace GT background during validation (keep only generated mouth region).")
    parser.add_argument("--use_new_forward", default=False, action="store_true",
        help="Use new forward pass with 9-frame repeat for first frame.")
    parser.add_argument("--match_audio_length", default=False, action="store_true",
        help="Match audio length with video length.")
    parser.add_argument("--use_frame_directories", default=False, action="store_true",
        help="Load video frames from directories instead of video files. Each 'video' entry in metadata should be a directory path containing numbered frame images (0.jpg, 1.jpg, ...).")
    parser.add_argument("--composite_validation", default=False, action="store_true",
        help="Composite generated faces into original full frames during validation.")
    parser.add_argument("--original_frames_base_path", type=str, default=None,
        help="Base path for original full frames (e.g., /path/to/images). If not set, derived from dataset_base_path.")
    parser.add_argument("--long_video", default=False, action="store_true",
        help="Use long video for validation.")
    # Profiling arguments
    parser.add_argument("--profile", default=False, action="store_true",
        help="Enable CUDA profiling for inference timing measurements.")
    parser.add_argument("--profile_output_csv", type=str, default="profiling_results.csv",
        help="Output path for profiling CSV results.")

    # Per-timestep validation arguments
    parser.add_argument(
        "--run_val_from_timestep",
        action="store_true",
        help="Run validation using lipsync_validation_from_timestep (single-step denoising from GT at each timestep).",
    )
    parser.add_argument(
        "--val_timestep_indices",
        type=str,
        default=None,
        help="Comma-separated list of timestep indices to validate (e.g., '0,1,2,3'). If None, validates all indices in denoising_steps.",
    )

    # Sync metrics validation arguments
    parser.add_argument(
        "--val_recon_metadata",
        type=str,
        default=None,
        help="Metadata CSV for reconstruction validation (video_id == audio_id)"
    )
    parser.add_argument(
        "--val_mixed_metadata",
        type=str,
        default=None,
        help="Metadata CSV for generalization validation (video_id != audio_id)"
    )
    parser.add_argument(
        "--enable_sync_metrics",
        action="store_true",
        help="Enable Sync-C and Sync-D metric computation during validation"
    )
    parser.add_argument(
        "--syncnet_model_path",
        type=str,
        default="/mnt/data1/jinhyuk/LatentSync/checkpoints/auxiliary/syncnet_v2.model",
        help="Path to SyncNet model checkpoint"
    )
    parser.add_argument(
        "--s3fd_model_path",
        type=str,
        default="/mnt/data1/hyunbin/LatentSync/checkpoints/auxiliary/sfd_face.pth",
        help="Path to S3FD face detector model checkpoint"
    )
    # LatentSync inference arguments
    parser.add_argument(
        "--latentsync_inference",
        action="store_true",
        help="Enable LatentSync-style preprocessing and compositing during validation"
    )
    parser.add_argument(
        "--original_video_dir",
        type=str,
        default=None,
        help="Base directory containing original videos (named {video_id}.mp4)"
    )
    parser.add_argument(
        "--latentsync_resolution",
        type=int,
        default=480,
        help="Resolution for LatentSync face detection (default: 480)"
    )
    parser.add_argument(
        "--latentsync_device",
        type=str,
        default="cuda",
        help="Device for LatentSync face detection (default: cuda)"
    )
    parser.add_argument(
        "--add_audio_to_composited_videos",
        action="store_true",
        help="Merge audio from original video into composited LatentSync outputs"
    )
    parser.add_argument(
        "--repeat_first_frame",
        action="store_true",
        help="Repeat first frame 9 times for first frame"
    )
    parser.add_argument(
        "--capture_vocal_attn",
        action="store_true",
        help="Capture vocal attention"
    )

    # ═══════════════════════════════════════════════════════════════════════════════
    # Streaming Inference Arguments
    # ═══════════════════════════════════════════════════════════════════════════════
    parser.add_argument(
        "--streaming_inference",
        action="store_true",
        default=False,
        help="Enable streaming inference mode with progressive frame generation."
    )
    parser.add_argument(
        "--streaming_output_dir",
        type=str,
        default=None,
        help="Directory to save intermediate frames during streaming (optional)."
    )
    parser.add_argument(
        "--streaming_skip_warmup_frames",
        type=int,
        default=3,
        help="Number of frames to skip from first block due to VAE warmup artifacts."
    )

    # ═══════════════════════════════════════════════════════════════════════════════
    # Performance Optimization Arguments
    # ═══════════════════════════════════════════════════════════════════════════════
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        default=False,
        help="Enable torch.compile for DiT model (10-30%% speedup after warmup). First inference will be slow."
    )
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="max-autotune-no-cudagraphs",
        choices=["max-autotune-no-cudagraphs", "max-autotune", "reduce-overhead", "default"],
        help="torch.compile mode. 'max-autotune-no-cudagraphs' recommended for inference."
    )

    # ═══════════════════════════════════════════════════════════════════════════════
    # Profiling Arguments
    # ═══════════════════════════════════════════════════════════════════════════════
    parser.add_argument(
        "--enable_profiling",
        action="store_true",
        default=False,
        help="Enable comprehensive profiling for inference comparison."
    )
    parser.add_argument(
        "--profiling_output_path",
        type=str,
        default=None,
        help="Path to save profiling results (JSON format)."
    )
    parser.add_argument(
        "--profiling_compare_baseline",
        type=str,
        default=None,
        help="Path to baseline profiling results for comparison."
    )

    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    # (Flux parser intentionally does not include CFG/OmniAudio flags.)
    return parser



def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--enable_fp8_training", default=False, action="store_true", help="Whether to enable FP8 training. Only available for LoRA training on a single GPU.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    return parser
