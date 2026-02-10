"""Model construction scaffold.

Goal: isolate model wiring, LoRA setup, and training-stage configuration.
"""

from __future__ import annotations

from examples.wanvideo.model_training.train import WanTrainingModule  # noqa: E402


def build_model(args):
    return WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        dataset_base_path=args.dataset_base_path,
        sf_restrict_timesteps=args.sf_restrict_timesteps,
        sf_denoising_step_list=args.sf_denoising_step_list,
        sf_warp_denoising_step=args.sf_warp_denoising_step,
        sf_timestep_shift=args.sf_timestep_shift,
        use_causal_wan=getattr(args, "use_causal_wan", False),
        causal_wan_model_file=getattr(args, "causal_wan_model_file", None),
        causal_wan_config=getattr(args, "causal_wan_config", None),
        causal_wan_kwargs=getattr(args, "causal_wan_kwargs", None),
        causal_wan_weights=getattr(args, "causal_wan_weights", None),
        causal_wan_lora_rank=getattr(args, "causal_wan_lora_rank", None),
        causal_wan_lora_alpha=getattr(args, "causal_wan_lora_alpha", 64.0),
        causal_wan_lora_targets=getattr(args, "causal_wan_lora_targets", "q,k,v,o,ffn.0,ffn.2"),
        causal_wan_lora_init=getattr(args, "causal_wan_lora_init", "kaiming"),
        causal_wan_use_lora=not getattr(args, "causal_wan_full_finetune", False) and getattr(args, "causal_wan_use_lora", True),
        audio_frames_per_block=getattr(args, "audio_frames_per_block", 3),
        enable_text_dropout=getattr(args, "enable_text_dropout", False),
        text_dropout_prob=getattr(args, "text_dropout_prob", 0.0),
        enable_audio_dropout=getattr(args, "enable_audio_dropout", False),
        audio_dropout_prob=getattr(args, "audio_dropout_prob", 0.0),
        enable_image_dropout=getattr(args, "enable_image_dropout", False),
        image_dropout_prob=getattr(args, "image_dropout_prob", 0.0),
        init_audio_from_omni=getattr(args, "init_audio_from_omni", False),
        omni_ckpt_path=getattr(args, "omni_ckpt_path", None),
        patch_embedding_trainable=getattr(args, "patch_embedding_trainable", False),
        kv_cache_size=getattr(args, "kv_cache_size", None),
    )
