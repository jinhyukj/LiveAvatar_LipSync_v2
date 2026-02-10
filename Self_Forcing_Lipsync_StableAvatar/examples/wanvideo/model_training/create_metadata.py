import os
import csv

# --- Step 1: CONFIGURE YOUR PATHS HERE ---

# Path to the folder containing your video files (e.g., .../videos_cfr)
video_folder_path = "/mnt/dataset1/jinhyuk/Hallo3/cropped_only_10K_preprocessed/videos_cfr"

# Path to the folder containing your caption .txt files
captions_folder_path = "/mnt/dataset1/jinhyuk/Hallo3/cropped_only_10K_preprocessed/captions"

audio_folder_path = "/mnt/dataset1/jinhyuk/Hallo3/cropped_only_10K_preprocessed/audio_emb_omniavatar_aligned"

# Where to save the final metadata.csv file
output_csv_path = "./metadata_audio.csv"

# --- Step 2: THE SCRIPT WILL DO THE REST ---

# A list to hold all our data rows
metadata = []

# Get a list of all video files in the video folder
try:
    video_files = [f for f in os.listdir(video_folder_path) if os.path.isfile(os.path.join(video_folder_path, f))]
except FileNotFoundError:
    print(f"Error: The video directory was not found at '{video_folder_path}'")
    exit()

print(f"Found {len(video_files)} video files. Processing...")

# Loop through each video file
for video_filename in video_files:
    # Get the base name of the file without the extension (e.g., "text_cfr25")
    video_base_name = os.path.splitext(video_filename)[0]
    
    # Remove the "_cfr25" suffix to get the name for the caption file (e.g., "text") # <-- MODIFIED LOGIC
    caption_base_name = video_base_name.removesuffix('_cfr25')
    
    # The corresponding text file should have this new base name
    caption_filename = f"{caption_base_name}.txt"
    caption_filepath = os.path.join(captions_folder_path, caption_filename)

    audio_filename = f"{caption_base_name}.pt"
    audio_filepath = os.path.join(audio_folder_path, audio_filename)
    
    # Check if the caption file actually exists
    if os.path.exists(caption_filepath):
        # Open and read the prompt from the text file
        with open(caption_filepath, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
            
            # Add the video filename and its prompt to our list
            metadata.append([video_filename, prompt, audio_filepath])
    else:
        print(f"Warning: No caption file found for video '{video_filename}'. Looked for '{caption_filename}'. Skipping.")

# Now, write all the collected data to the CSV file
with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write the header row
    csv_writer.writerow(['video', 'prompt', 'audio_emb'])
    
    # Write all the data rows
    csv_writer.writerows(metadata)

print(f"\n✅ Successfully created metadata file at: {output_csv_path}")
print(f"Total entries written: {len(metadata)}")