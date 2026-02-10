import os
PATH = '/home/work/.cache'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

from natsort import natsorted
from tqdm import tqdm


import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def test(args):
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto", attn_implementation="flash_attention_2"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    # breakpoint()

    # Messages containing a local video path and a text query
    
    fps = 16
    
    video_base_path = args.video_base_path
    output_base_path = args.output_base_path
    
    names = natsorted(os.listdir(video_base_path))
    names = [name.split('.')[0] for name in names if name.endswith('.mp4')]
    if args.batch!=  -1:
        data_len = len(names)
        # breakpoint()
        names = names[int(args.batch/args.total_batch* data_len):int((args.batch+1)/args.total_batch* data_len)]
    os.makedirs(output_base_path, exist_ok=True)
    # for name in names:
    
    for name in tqdm(names):
        output_path = os.path.join(output_base_path, f'{name}.txt')
        if os.path.exists(output_path):
            print(f"Caption for {name} already exists. Skipping...")
            continue
        
        video_name = f'{name}.mp4'
        video_path = os.path.join(video_base_path, video_name)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{video_path}",
                        "max_pixels": 480 * 768,
                        "fps": 4.0,
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]

        #In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=fps,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)
        with open(output_path, 'w') as f:
            f.write(output_text[0].replace('\n', ' '))
                




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")
    parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
    parser.add_argument('--batch', type=int, default=-1, help='Current batch index for splitting dataset')
    parser.add_argument('--total_batch', type=int, default=10, help='Total number of batch splits')
    parser.add_argument('--video_base_path', type=str, default='', help='Path to the directory containing video files')
    parser.add_argument('--output_base_path', type=str, default='', help='Path to the directory to save captions')
    args = parser.parse_args()
    test(args)
    