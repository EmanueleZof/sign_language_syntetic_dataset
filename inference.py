import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.jit
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image


from MimicMotion.mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

from MimicMotion.constants import ASPECT_RATIO

from MimicMotion.mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from MimicMotion.mimicmotion.utils.loader import create_pipeline
from MimicMotion.mimicmotion.utils.utils import save_to_mp4
from MimicMotion.mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = {
    "log_file": None,
    "inference_config": "configs/test.yaml",
    "output_dir": "outputs/",
    "no_use_float16": False
}

#

Path(args['output_dir']).mkdir(parents=True, exist_ok=True)