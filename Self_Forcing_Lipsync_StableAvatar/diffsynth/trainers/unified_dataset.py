import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image



class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor, use_direct_resize=False):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.use_direct_resize = use_direct_resize

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
    
    def direct_resize(self, image, target_height, target_width):
        """Direct resize using cv2 like Wav2Lip - may distort aspect ratio but preserves all content."""
        import cv2
        import numpy as np
        # Convert PIL to numpy (RGB)
        img_np = np.array(image)
        # cv2.resize uses (width, height) order
        img_resized = cv2.resize(img_np, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        # Convert back to PIL
        return Image.fromarray(img_resized)
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        target_height, target_width = self.get_height_width(data)
        if self.use_direct_resize:
            return self.direct_resize(data, target_height, target_width)
        else:
            return self.crop_and_resize(data, target_height, target_width)



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames



class LoadVideoWithReference(DataProcessingOperator):
    """Load video with non-overlapping GT and reference frame sequences.

    GT frames: frames 0 to (num_frames-1) - sequential from start
    Ref frames: random contiguous segment from remaining frames (no overlap with GT)

    Requires video length >= 2 * num_frames (e.g., 162 for num_frames=81)

    Returns a dict with:
        - gt_frames: list of PIL Images for target frames
        - ref_frames: list of PIL Images for reference frames
        - ref_start_idx: starting index of reference segment (for debugging)
    """

    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1,
                 frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor

    def __call__(self, data: str):
        import random

        reader = imageio.get_reader(data)
        total_frames = int(reader.count_frames())

        min_required = 2 * self.num_frames
        if total_frames < min_required:
            reader.close()
            raise ValueError(f"Video too short: {total_frames} < {min_required} required for reference frames")

        # GT frames: 0 to num_frames-1 (sequential from start)
        gt_frames = []
        for frame_id in range(self.num_frames):
            frame = Image.fromarray(reader.get_data(frame_id))
            gt_frames.append(self.frame_processor(frame))

        # Reference frames: random contiguous segment from remaining frames
        # Valid start range: [num_frames, total_frames - num_frames]
        ref_start_min = self.num_frames
        ref_start_max = total_frames - self.num_frames
        ref_start = random.randint(ref_start_min, ref_start_max)

        ref_frames = []
        for frame_id in range(ref_start, ref_start + self.num_frames):
            frame = Image.fromarray(reader.get_data(frame_id))
            ref_frames.append(self.frame_processor(frame))

        reader.close()

        return {
            "gt_frames": gt_frames,
            "ref_frames": ref_frames,
            "ref_start_idx": ref_start,
        }



class LoadFramesFromDirectory(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        
    def get_sorted_frame_files(self, directory):
        """Get numerically sorted list of image files in directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        files = [f for f in os.listdir(directory) 
                if os.path.splitext(f)[1].lower() in valid_extensions]
        # Numeric sort: 0.jpg, 1.jpg, 2.jpg, ... 10.jpg, 11.jpg (not alphabetical)
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        return files
        
    def get_num_frames(self, num_available):
        """Determine actual number of frames to load."""
        num_frames = min(self.num_frames, num_available)
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        """Load frames from directory path."""
        if not os.path.isdir(data):
            raise ValueError(f"Expected directory path, got: {data}")
            
        frame_files = self.get_sorted_frame_files(data)
        num_frames = self.get_num_frames(len(frame_files))
        
        frames = []
        for i in range(num_frames):
            frame_path = os.path.join(data, frame_files[i])
            frame = Image.open(frame_path).convert("RGB")
            frame = self.frame_processor(frame)
            frames.append(frame)
            
        return frames



class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)



class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        use_frame_directories=False,
    ):
        # When using frame directories (wav2lip crops), use direct resize to preserve
        # all content for proper compositing. Otherwise use crop+resize for better
        # aspect ratio handling with general videos.
        frame_processor = ImageCropAndResize(
            height, width, max_pixels, height_division_factor, width_division_factor,
            use_direct_resize=use_frame_directories
        )
        
        if use_frame_directories:
            return RouteByType(operator_map=[
                (str, ToAbsolutePath(base_path) >> LoadFramesFromDirectory(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=frame_processor,
                )),
            ])
        else:
            return RouteByType(operator_map=[
                (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                    (("jpg", "jpeg", "png", "webp"), LoadImage() >> frame_processor >> ToList()),
                    (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> frame_processor),
                    (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                        num_frames, time_division_factor, time_division_remainder,
                        frame_processor=frame_processor,
                    )),
                ])),
            ])

    @staticmethod
    def default_video_operator_with_reference(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        """Video operator that returns both GT and non-overlapping reference frames.

        Use this for LatentSync-style training where reference frames come from
        a different temporal segment than the target (GT) frames.

        Requires videos with at least 2 * num_frames (e.g., 162 frames for num_frames=81).

        Returns a dict with 'gt_frames', 'ref_frames', and 'ref_start_idx'.
        """
        frame_processor = ImageCropAndResize(
            height, width, max_pixels, height_division_factor, width_division_factor
        )
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideoWithReference(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=frame_processor,
                )),
            ])),
        ])

    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            # Preserve original file paths / IDs in a lightweight meta dict
            # before applying any loading/transformation operators.
            meta = data.get("meta", None)
            if not isinstance(meta, dict):
                if meta is None:
                    meta = {}
                else:
                    meta = {"_orig_meta": meta}
            for key in self.data_file_keys:
                if key in data:
                    original_value = data[key]
                    if isinstance(original_value, str):
                        # Remember original path for this key (e.g., 'video')
                        meta_key_path = f"{key}_path"
                        if meta_key_path not in meta:
                            meta[meta_key_path] = original_value
                        # For videos, also derive a simple video_id from filename.
                        if key == "video" and "video_id" not in meta:
                            base = os.path.basename(original_value)
                            if base.endswith("_cfr25.mp4"):
                                meta["video_id"] = base[:-len("_cfr25.mp4")]
                            else:
                                meta["video_id"] = os.path.splitext(base)[0]
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key]
                    elif key in self.data_file_keys:
                        result = self.main_data_operator(data[key])
                        # Handle dict return from LoadVideoWithReference
                        if isinstance(result, dict) and "gt_frames" in result:
                            data[key] = result["gt_frames"]
                            data["ref_frames"] = result["ref_frames"]
                            meta["ref_start_idx"] = result.get("ref_start_idx")
                        else:
                            data[key] = result
            # Also preserve audio_emb path if present (not in data_file_keys but used for lipsync)
            if "audio_emb" in data and isinstance(data["audio_emb"], str):
                if "audio_emb_path" not in meta:
                    meta["audio_emb_path"] = data["audio_emb"]
                # Also derive audio_id from filename
                if "audio_id" not in meta:
                    base = os.path.basename(data["audio_emb"])
                    if base.endswith(".pt"):
                        meta["audio_id"] = base[:-3]
                    else:
                        meta["audio_id"] = os.path.splitext(base)[0]
            if meta:
                data["meta"] = meta
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
