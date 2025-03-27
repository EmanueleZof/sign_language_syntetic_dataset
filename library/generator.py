import gc
import os
import math
import json
import yaml
import torch.jit
import numpy as np

from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image

from MimicMotion.mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from MimicMotion.mimicmotion.utils.loader import create_pipeline
from MimicMotion.mimicmotion.utils.utils import save_to_mp4
from MimicMotion.mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

class Generator:
    def __init__(self,
                 output_dir = "outputs/",
                 temp_dir = "temp/",
                 use_float16 = True
                 ):
        self.output_dir = output_dir
        self.temp = temp
        self.use_float16 = use_float16

        self.ASPECT_RATIO = 9 / 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.default_video = {
                "ref_video_path": "MimicMotion/assets/example_data/videos/pose1.mp4",
                "ref_image_path": "MimicMotion/assets/example_data/images/demo1.jpg",
                "num_frames": 72,
                "resolution": 576,
                "frames_overlap": 6,
                "num_inference_steps": 25,
                "noise_aug_strength": 0,
                "guidance_scale": 2.0,
                "sample_stride": 2,
                "fps": 15,
                "seed": 42,
            }
        self.default_config = {
            "base_model_path": "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
            "ckpt_path": "models/MimicMotion_1-1.pth",
            "list" : [self.default_video]
        }

    def _output_dir(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _flush_ram(self):
        gc.collect()

    def _load_images(self, images_json):
        images_file = open(images_json)
        images_list = json.load(images_file)
        return images_list

    def _preprocess(self, video_path, image_path, resolution=576, sample_stride=2):
        """preprocess ref image pose and video pose

        Args:
            video_path (str): input video pose path
            image_path (str): reference image path
            resolution (int, optional):  Defaults to 576.
            sample_stride (int, optional): Defaults to 2.
        """
        image_pixels = pil_loader(image_path)
        image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
        h, w = image_pixels.shape[-2:]

        # compute target h/w according to original aspect ratio
        if h>w:
            w_target, h_target = resolution, int(resolution / self.ASPECT_RATIO // 64) * 64
        else:
            w_target, h_target = int(resolution / self.ASPECT_RATIO // 64) * 64, resolution

        h_w_ratio = float(h) / float(w)

        if h_w_ratio < h_target / w_target:
            h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
        else:
            h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target

        image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
        image_pixels = center_crop(image_pixels, [h_target, w_target])
        image_pixels = image_pixels.permute((1, 2, 0)).numpy()

        # get image&video pose value
        image_pose = get_image_pose(image_pixels)
        video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)
        pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
        image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))

        return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1

    def _run_pipeline(self, pipeline: MimicMotionPipeline, image_pixels, pose_pixels, device, task_config):
        image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
        generator = torch.Generator(device=device)
        generator.manual_seed(task_config.seed)
        frames = pipeline(
            image_pixels,
            image_pose=pose_pixels,
            num_frames=pose_pixels.size(0),
            tile_size=task_config.num_frames,
            tile_overlap=task_config.frames_overlap,
            height=pose_pixels.shape[-2],
            width=pose_pixels.shape[-1],
            fps=7,
            noise_aug_strength=task_config.noise_aug_strength,
            num_inference_steps=task_config.num_inference_steps,
            generator=generator, min_guidance_scale=task_config.guidance_scale,
            max_guidance_scale=task_config.guidance_scale,
            decode_chunk_size=8,
            output_type="pt",
            device=device
        ).frames.cpu()
        video_frames = (frames * 255.0).to(torch.uint8)

        for vid_idx in range(video_frames.shape[0]):
            # deprecated first frame because of ref image
            _video_frames = video_frames[vid_idx, 1:]

        return _video_frames

    def build_config(self,
                     num_video = 1,
                     model = 72,
                     frames_overlap = 6,
                     video_path = "",
                     images_path = ""):

        main_config = self.default_config

        if (video_path != "" and images_path != ""):
          videos = []
          images_list = self._load_images(images_path)

          for i in range(num_video):
            videos.append({
                "ref_video_path": "test_video",
                "ref_image_path": images_list[np.random.randint(0,len(images_list))],
                "num_frames": model,
                "resolution": self.default_video["resolution"],
                "frames_overlap": frames_overlap,
                "num_inference_steps": self.default_video["num_inference_steps"],
                "noise_aug_strength": self.default_video["noise_aug_strength"],
                "guidance_scale": self.default_video["guidance_scale"],
                "sample_stride": self.default_video["sample_stride"],
                "fps": self.default_video["fps"],
                "seed": self.default_video["seed"],
            })
          main_config["list"] = videos

        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        with open("temp/config.yaml", "w") as yaml_file:
          yaml.dump(main_config, yaml_file, default_flow_style=False)

    @torch.no_grad()
    def generate(self, inference_config="configs/default.yaml"):
        self._output_dir()

        if self.use_float16 :
            torch.set_default_dtype(torch.float16)

        infer_config = OmegaConf.load(inference_config)
        pipeline = create_pipeline(infer_config, self.device)

        for task in infer_config.list:
            # Pre-process data
            pose_pixels, image_pixels = self._preprocess(
                task.ref_video_path, 
                task.ref_image_path, 
                task.resolution, 
                task.sample_stride
                )

            # Run MimicMotion pipeline
            _video_frames = self._run_pipeline(
                pipeline, 
                image_pixels, 
                pose_pixels, 
                self.device, 
                task
                )

            # Save results to output folder
            save_to_mp4(
                _video_frames, 
                f"{self.output_dir}/{os.path.basename(task.ref_video_path).split('.')[0]}" \
                f"_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4",
                fps=task.fps,
            )