"""Dataset construction scaffold.

Goal: isolate dataset and dataloader setup from training loop.
"""

from __future__ import annotations

from diffsynth.trainers.unified_dataset import UnifiedDataset


def build_main_data_operator(args):
    use_reference_frames = getattr(args, "use_reference_frames", False) or getattr(args, "use_latentsync_audio", False)
    if use_reference_frames:
        return UnifiedDataset.default_video_operator_with_reference(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        )
    return UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=args.num_frames,
        time_division_factor=4,
        time_division_remainder=1,
        use_frame_directories=getattr(args, "use_frame_directories", False),
    )


def build_train_dataset(args):
    main_data_operator = build_main_data_operator(args)
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=main_data_operator,
    )


def build_val_dataset(args):
    if not getattr(args, "validation_dataset_metadata_path", None):
        return None

    val_num_frames = args.num_frames
    if (getattr(args, "run_val_mode", None) is not None or getattr(args, "run_val_audio_cfg", False)) and getattr(args, "val_num_frames", None) is not None:
        val_num_frames = args.val_num_frames

    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.validation_dataset_metadata_path,
        repeat=1,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=val_num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            use_frame_directories=getattr(args, "use_frame_directories", False),
        ),
    )
