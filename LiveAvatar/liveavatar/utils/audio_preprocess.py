import os
import subprocess
from audio_separator.separator import Separator

def add_silence_to_audio_ffmpeg(audio_path, tmp_audio_path, silence_duration_s=0.5):
    # 使用 ffmpeg 命令在音频前加上静音
    command = [
        'ffmpeg', 
        '-i', audio_path,  # 输入音频文件路径
        '-f', 'lavfi',  # 使用 lavfi 虚拟输入设备生成静音
        '-t', str(silence_duration_s),  # 静音时长，单位秒
        '-i', 'anullsrc=r=16000:cl=stereo',  # 创建静音片段（假设音频为 stereo，采样率 44100）
        '-filter_complex', '[1][0]concat=n=2:v=0:a=1[out]',  # 合并静音和原音频
        '-map', '[out]',  # 输出合并后的音频
        '-y', tmp_audio_path,  # 输出文件路径
        '-loglevel', 'quiet'
    ]
    
    subprocess.run(command, check=True)

# 人声分离
audio_separator = None
def separate_audio(input_audio_path, output_path):
    global audio_separator
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    if audio_separator is None:
        audio_separator = Separator(
                    output_dir=out_dir,
                    output_single_stem="vocals",
                    model_file_dir='pretrained_models/audio_separator',
                )
        audio_separator.load_model('Kim_Vocal_2.onnx')
    outputs = audio_separator.separate(input_audio_path)
    if len(outputs) <= 0:
        raise RuntimeError("Audio separate failed.")

    vocal_audio_file = outputs[0]
    vocal_audio_name, _ = os.path.splitext(vocal_audio_file)
    vocal_audio_file = os.path.join(audio_separator.output_dir, vocal_audio_file)
    subprocess.run(['mv', vocal_audio_file, output_path], check=True)
    return output_path