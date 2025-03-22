import torch.jit

from pathlib import Path
from omegaconf import OmegaConf
from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

class GENERATOR:
    def __init__(self,
                 inference_config = "configs/default.yaml",
                 output_dir = "outputs/",
                 use_float16 = True
                 ):
        self.inference_config = inference_config
        self.output_dir = output_dir
        self.use_float16 = use_float16

        self.ASPECT_RATIO = 9 / 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._main()

    def _output_dir(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _preprocess(video_path, image_path, resolution=576, sample_stride=2):
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

    def _main(self):        
        self._output_dir()

        if self.use_float16 :
            torch.set_default_dtype(torch.float16)
        
        infer_config = OmegaConf.load(self.inference_config)
        pipeline = create_pipeline(infer_config, self.device)

        for task in infer_config.base:
            pose_pixels, image_pixels = self._preprocess(
                task.ref_video_path, 
                task.ref_image_path, 
                resolution=task.resolution, 
                sample_stride=task.sample_stride
                )