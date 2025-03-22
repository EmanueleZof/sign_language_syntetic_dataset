import torch.jit

from pathlib import Path
from omegaconf import OmegaConf

class GENERATOR:
    def __init__(self,
                 inference_config = "configs/test.yaml",
                 output_dir = "outputs/",
                 use_float16 = True
                 ):
        self.inference_config = inference_config
        self.output_dir = output_dir
        self.use_float16 = use_float16

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._main()

    def _output_dir(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _main(self):        
        self._output_dir()

        if self.use_float16 :
            torch.set_default_dtype(torch.float16)
        
        infer_config = OmegaConf.load(self.inference_config)
        print(infer_config)