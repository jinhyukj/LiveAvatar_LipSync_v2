import argparse
import os
import tqdm
from statistics import fmean
from diffsynth.models.syncnet.syncnet import SyncNetEval
from diffsynth.models.syncnet.syncnet_detect import SyncNetDetector
from diffsynth.models.syncnet.utils import red_text
import torch


def syncnet_eval(syncnet, syncnet_detector, video_path, temp_dir, detect_results_dir="detect_results"):
    """Run SyncNet detection + scoring on one video and average over all tracks."""
    syncnet_detector(video_path=video_path, min_track=32)
    crop_videos = os.listdir(os.path.join(detect_results_dir, "crop"))
    if crop_videos == []:
        raise Exception(red_text(f"Face not detected in {video_path}"))

    av_offset_list, min_dist_list, conf_list = [], [], []

    for video in crop_videos:
        av_offset, min_dist, conf = syncnet.evaluate(
            video_path=os.path.join(detect_results_dir, "crop", video), temp_dir=temp_dir
        )
        av_offset_list.append(av_offset)
        min_dist_list.append(min_dist)
        conf_list.append(conf)

    return (
        int(fmean(av_offset_list)),
        fmean(min_dist_list),  # Sync-D
        fmean(conf_list),  # Sync-C
    )


def main():
    parser = argparse.ArgumentParser(description="SyncNet")
    parser.add_argument("--initial_model", type=str, default="checkpoints/auxiliary/syncnet_v2.model", help="")
    parser.add_argument("--video_path", type=str, default=None, help="")
    parser.add_argument("--videos_dir", type=str, default="/root/processed")
    parser.add_argument("--temp_dir", type=str, default="temp", help="")
    parser.add_argument(
        "--debug_syncnet_dir",
        type=str,
        default="",
        help="Optional directory to dump SyncNet crop frames (per video + track)",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    syncnet = SyncNetEval(device=device)
    syncnet.loadParameters(args.initial_model)

    syncnet_detector = SyncNetDetector(
        device=device,
        detect_results_dir="detect_results",
        debug_crop_dir=args.debug_syncnet_dir or None,
    )

    if args.video_path is not None:
        av_offset, min_dist, conf = syncnet_eval(syncnet, syncnet_detector, args.video_path, args.temp_dir)
        print(
            f"Input video: {args.video_path}\n"
            f"SyncNet min distance (Sync-D): {min_dist:.2f}\n"
            f"SyncNet confidence (Sync-C): {conf:.2f}\n"
            f"AV offset: {av_offset}"
        )
    else:
        sync_d_list = []
        sync_c_list = []
        video_names = sorted([f for f in os.listdir(args.videos_dir) if f.endswith('.mp4')])
        for video_name in tqdm.tqdm(video_names):
            try:
                _, min_dist, conf = syncnet_eval(
                    syncnet, syncnet_detector, os.path.join(args.videos_dir, video_name), args.temp_dir
                )
                sync_d_list.append(min_dist)
                sync_c_list.append(conf)
                print(f"{video_name}: Sync-D {min_dist:.2f}, Sync-C {conf:.2f}")
            except Exception as e:
                print(e)

        if sync_d_list:
            print(f"Mean SyncNet Min Distance (Sync-D): {fmean(sync_d_list):.02f}")
        else:
            print("No videos were processed for Sync-D.")

        if sync_c_list:
            print(f"Mean SyncNet Confidence (Sync-C): {fmean(sync_c_list):.02f}")
        else:
            print("No videos were processed for Sync-C.")


if __name__ == "__main__":
    main()
