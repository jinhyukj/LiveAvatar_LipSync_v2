import math
import os
import subprocess
import cv2
import torch
from tqdm import tqdm
import lpips
from glob import glob
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from pytorch_fid import fid_score
from tempfile import TemporaryDirectory
from utils.sync_net.run_pipeline import run_syncnet
from transformers import AutoModelForCausalLM
from PIL import Image
# LPIPS模型初始化
lpips_fn = lpips.LPIPS(net='alex')

def calculate_psnr_ssim_lpips(gen_frames, real_frames):
    """计算每一帧的PSNR、SSIM、LPIPS"""
    psnr_values = []
    ssim_values = []
    lpips_values = []

    for gen_frame, real_frame in zip(gen_frames, real_frames):
        # PSNR
        psnr_value = psnr(real_frame, gen_frame)
        psnr_values.append(psnr_value)

        # SSIM
        ssim_value = ssim(real_frame, gen_frame, multichannel=True, channel_axis=2)
        ssim_values.append(ssim_value)

        # LPIPS
        gen_frame_tensor = torch.tensor(gen_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        real_frame_tensor = torch.tensor(real_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        lpips_value = lpips_fn(gen_frame_tensor, real_frame_tensor).item()
        lpips_values.append(lpips_value)
    # 返回平均值
    return {
        "PSNR": np.mean(psnr_values),
        "SSIM": np.mean(ssim_values),
        "LPIPS": np.mean(lpips_values),
    }

def calculate_fid(gen_frames, real_frames):
    """计算FID指标"""
    # 保存图片到文件夹
    with TemporaryDirectory() as tmp: 
        gen_img_path = f'{tmp}/gen'
        real_img_path = f'{tmp}/real'
        os.makedirs(gen_img_path)
        os.makedirs(real_img_path)
        for i, gen_frame in enumerate(gen_frames):
            cv2.imwrite(f'{gen_img_path}/{i}.png', gen_frame[..., ::-1])
        for i, real_frame in enumerate(real_frames):
            cv2.imwrite(f'{real_img_path}/{i}.png', real_frame[..., ::-1])

        # 使用pytorch-fid库，提取帧并计算FID
        score = fid_score.calculate_fid_given_paths([gen_img_path, real_img_path], batch_size=16, device='cuda' if torch.cuda.is_available() else 'cpu', dims=2048)
    return score

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, device, method='styleganv'):
    """计算FVD指标"""

    if method == 'styleganv':
        from utils.fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from utils.fvd.videogpt.fvd import load_i3d_pretrained
        from utils.fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from utils.fvd.videogpt.fvd import frechet_distance

    # videos [batch_size, timestamps, channel, h, w]
    assert videos1.shape == videos2.shape
    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = {}

    # for calculate FVD, each clip_timestamp must >= 10
    for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):
       
        # get a video clip
        # videos_clip [batch_size, channel, timestamps[:clip], h, w]
        videos_clip1 = videos1[:, :, : clip_timestamp]
        videos_clip2 = videos2[:, :, : clip_timestamp]

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
      
        # calculate FVD when timestamps[:clip]
        fvd_results[clip_timestamp] = frechet_distance(feats1, feats2)

    return fvd_results

def load_video_with_fps(path, fps=30):
    with TemporaryDirectory() as tmp:
        video_cap = cv2.VideoCapture(path)
        
        original_fps = math.ceil(video_cap.get(cv2.CAP_PROP_FPS))
        if original_fps != fps:
            tmp_path = os.path.join(tmp, 'tmp.mp4')
            subprocess.run(['ffmpeg', '-i', path, '-r', str(fps), tmp_path, '-v', 'quiet', '-y'])
            video_cap.release()
            video_cap = cv2.VideoCapture(tmp_path)
        
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        # frame_interval = int(original_fps / fps)  # 计算每隔多少帧读取一帧
        frame_idx = 0  # 当前帧的索引
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
            # if frame_idx % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            frame_idx += 1
        video_cap.release()
    
    return frames

def evaluate_video_metrics(pred_video_path, gt_video_path, fps=30):
    """计算所有评估指标"""
    gen_frames = load_video_with_fps(pred_video_path, fps)
    real_frames = load_video_with_fps(gt_video_path, fps)
    if gen_frames[0].shape != real_frames[0].shape:
        for i in range(len(gen_frames)):
            gen_frames[i] = cv2.resize(gen_frames[i], real_frames[0].shape[1::-1]) #gen_frames[i][:real_frames[0].shape[0], :real_frames[0].shape[1]]
    h = gen_frames[0].shape[0]
    w = gen_frames[0].shape[1]

    # if real_frames[0].shape[0] != h and real_frames[0].shape[1] != w:
    #     print(f'image size not match: {image_path}, gt: {real_frames[0].shape}, pred: {gen_frames[0].shape}')
    #     for i in range(len(real_frames)):
    #         # 对原视频进行中心裁剪
    #         real_frames[i] = cv2.resize(real_frames[i], (w, h))
    if len(gen_frames) != len(real_frames):
        print("Generated and real videos must have the same number of frames")
        real_frames = real_frames[:len(gen_frames)]
        gen_frames = gen_frames[:len(real_frames)]
    # 计算 PSNR, SSIM, LPIPS
    frame_metrics = calculate_psnr_ssim_lpips(gen_frames, real_frames)

    # 计算 FID
    fid_value = calculate_fid(gen_frames, real_frames)
    
    # 计算 FVD (这里假设有实际的 FVD 计算函数)
    gen_tensor = torch.from_numpy(np.array(gen_frames)).permute(0, 3, 1, 2)[None, :] / 255.
    real_tensor = torch.from_numpy(np.array(real_frames)).permute(0, 3, 1, 2)[None, :] / 255.
    fvd_value = calculate_fvd(gen_tensor, real_tensor, device='cuda:0' if torch.cuda.is_available() else 'cpu')

    # 汇总结果
    metrics = {
        "PSNR": frame_metrics["PSNR"],
        "SSIM": frame_metrics["SSIM"],
        "LPIPS": frame_metrics["LPIPS"],
        "FID": fid_value,
        "FVD": np.mean([score for key, score in fvd_value.items()])
    }

    # lipsync
    with TemporaryDirectory() as tmpdir:
        try:
            offset, sync_c, sync_d = run_syncnet(pred_video_path, tmpdir)
        except:
            offset, sync_c, sync_d = None, None, None
    if offset is not None:
        metrics['Sync-C'] = sync_c.item()
        metrics['Sync-D'] = sync_d.item()

    # qalign
    qua, aes = cal_qalign(gen_frames)
    metrics['IQA'] = qua
    metrics['Aesthe'] = aes

    # VBench

    return metrics

def evaluate_gt(gt_video_path):
    real_frames = load_video_with_fps(gt_video_path, 30)
    metrics = {}
    # lipsync
    with TemporaryDirectory() as tmpdir:
        offset, sync_c, sync_d = run_syncnet(gt_video_path, tmpdir)
    if offset is not None:
        metrics['Sync-C'] = sync_c.item()
        metrics['Sync-D'] = sync_d.item()

    # qalign
    qua, aes = cal_qalign(real_frames)
    metrics['IQA'] = qua
    metrics['Aesthe'] = aes

    return metrics

qalign_model = None
def cal_qalign(frames):
    global qalign_model
    if qalign_model is None:
        qalign_model = AutoModelForCausalLM.from_pretrained("/tmp/pretrained/one-align", trust_remote_code=True,torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    aes = []
    qua = []
    qua = qalign_model.score([Image.fromarray(frame) for frame in frames], task_="quality", input_="image")
    aes = qalign_model.score([Image.fromarray(frame) for frame in frames], task_="aesthetic", input_="image")
    return qua.mean().item(), aes.mean().item()

def eval_path(out_path, gt_path, fps=30):
    # pred_path=/mnt/bn/foundation-ads2/user/renyi/interns_env/ganqijun/Moore-AnimateAnyone/output/20240912/1931--seed_42-576x1024/00000_ref_00000_1024x576_3_1931_res.mp4
    # gt_path = ['/mnt/bn/foundation-ads2/user/renyi/interns_env/ganqijun/mega_avatar/sample_videos', '/mnt/bn/foundation-ads2/user/renyi/interns_env/ganqijun/Mega_testset', '/mnt/bn/foundation-ads2/user/renyi/interns_env/ganqijun/TikTok_testset']
    # out_path = '/mnt/bn/foundation-ads2/user/renyi/interns_env/ganqijun/mega_avatar/demo/0905_all_s2_bilibili/2024-09-13T07-58-03-debug-10-seed42-1.0-ema-zeromf/videos'
    metric_dict = {}
    video_names = [i[:-4] for i in os.listdir(gt_path) if i[-4:] == '.mp4']
    psnr_tmp, ssim_tmp, lpips_tmp, fid_tmp, fvd_tmp = 0.0, 0.0, 0.0, 0.0, 0.0
    count = 0
    for video_name in video_names:
        tmp_video_paths = glob(f'{out_path}/{video_name}_*')
        if len(tmp_video_paths) == 0:
            print('Not exist generated video:', f'{out_path}/{video_name}_*')
            continue   
        metrics = evaluate_video_metrics(tmp_video_paths[0], fps)
        metric_dict[video_name] = metrics
        psnr_tmp += metrics['PSNR']
        ssim_tmp += metrics['SSIM']
        lpips_tmp += metrics['LPIPS']
        fid_tmp += metrics['FID']
        fvd_tmp += metrics['FVD']
        count += 1
        print(video_name, '\n', metrics)
    print('--------------------------------------\n')
    print(f'{os.path.basename(gt_path)}: PSNR: {psnr_tmp / count}, SSIM: {ssim_tmp / count}, LPIPS: {lpips_tmp / count}, FID:{fid_tmp / count}, FVD:{fvd_tmp / count}')
    psnr_total, ssim_total, lpips_total, fid_total, fvd_total = [], [], [], [], []
    for video_name in metric_dict.keys():
        psnr_total.append(metric_dict[video_name]['PSNR'])
        ssim_total.append(metric_dict[video_name]['SSIM'])
        lpips_total.append(metric_dict[video_name]['LPIPS'])
        fid_total.append(metric_dict[video_name]['FID'])
        fvd_total.append(metric_dict[video_name]['FVD'])
    print(f'Total: PSNR: {sum(psnr_total) / len(metric_dict)}, SSIM: {sum(ssim_total) / len(metric_dict)}, LPIPS: {sum(lpips_total) / len(metric_dict)}, FID:{sum(fid_total) / len(metric_dict)}, FVD:{sum(fvd_total) / len(metric_dict)}')
