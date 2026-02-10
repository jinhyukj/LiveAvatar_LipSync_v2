import argparse
import subprocess
import os

def merge_audio_files(input_file1, input_file2, output_file):
    """
    Merge two WAV audio files using ffmpeg.
    The merged audio will contain sounds from both original audio files.
    If the input audio durations are different, the output audio duration will match the shorter one.

    Parameters:
    input_file1 (str): Path to the first input WAV file.
    input_file2 (str): Path to the second input WAV file.
    output_file (str): Path to the output WAV file.
    """
    if not os.path.exists(input_file1):
        print(f"Error: Input file {input_file1} does not exist.")
        return
    if not os.path.exists(input_file2):
        print(f"Error: Input file {input_file2} does not exist.")
        return

    command = [
        'ffmpeg',
        '-i', input_file1,
        '-i', input_file2,
        '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=shortest[aout]',
        '-map', '[aout]',
        '-y',
        output_file
    ]

    try:
        print(f"Executing ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Audio files successfully merged to {output_file}")
        if result.stdout:
            print("ffmpeg output:")
            print(result.stdout)
        if result.stderr:
            print("ffmpeg error output (may contain warnings or detailed information):")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg command execution failed. Return code: {e.returncode}")
        print("ffmpeg output (stdout):")
        print(e.stdout)
        print("ffmpeg error output (stderr):")
        print(e.stderr)
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure ffmpeg is installed and added to your system PATH.")
    except Exception as e:
        print(f"Unknown error occurred while merging audio: {e}")


def merge_multiple_audio_files(input_files_list, output_file):
    """
    Merge multiple WAV audio files using ffmpeg.
    The merged audio will contain sounds from all original audio files.
    If the input audio durations are different, the output audio duration will match the shortest one.

    Parameters:
    input_files_list (list): List of paths to input WAV files.
    output_file (str): Path to the output WAV file.
    """
    if len(input_files_list) == 0:
        print("Error: No input files provided.")
        return
    
    if len(input_files_list) == 1:
        print(f"Only one input file provided, copying to output: {output_file}")
        command = ['cp', input_files_list[0], output_file]
        subprocess.run(command, check=True)
        return
    
    # Check if all input files exist
    for input_file in input_files_list:
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} does not exist.")
            return
    
    # Build ffmpeg command
    command = ['ffmpeg']
    
    # Add all input files
    for input_file in input_files_list:
        command.extend(['-i', input_file])
    
    # Build filter_complex string
    # Format: [0:a][1:a]...[N-1:a]amix=inputs=N:duration=shortest[aout]
    filter_inputs = ''.join([f'[{i}:a]' for i in range(len(input_files_list))])
    filter_complex = f'{filter_inputs}amix=inputs={len(input_files_list)}:duration=shortest[aout]'
    
    command.extend([
        '-filter_complex', filter_complex,
        '-map', '[aout]',
        '-y',
        output_file
    ])
    
    try:
        print(f"Executing ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Audio files successfully merged to {output_file}")
        if result.stdout:
            print("ffmpeg output:")
            print(result.stdout)
        if result.stderr:
            print("ffmpeg error output (may contain warnings or detailed information):")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg command execution failed. Return code: {e.returncode}")
        print("ffmpeg output (stdout):")
        print(e.stdout)
        print("ffmpeg error output (stderr):")
        print(e.stderr)
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure ffmpeg is installed and added to your system PATH.")
    except Exception as e:
        print(f"Unknown error occurred while merging audio: {e}")

if __name__ == "__main__":
    input_file1 = "Causvid/examples/talk_padded.wav"
    input_file2 = "Causvid/examples/zero_shot_prompt_padded.wav"
    output_file = "Causvid/examples/talk_sync.wav"

    merge_audio_files(input_file1, input_file2, output_file)
