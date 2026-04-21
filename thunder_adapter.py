import contextlib
import os

import torch
from thunder.models.pretrained_models import PretrainedModel
from torchvision import transforms

from model import NanoPathFM


class NanoPathThunderModel(PretrainedModel):
    def __init__(self):
        super().__init__()
        checkpoint = torch.load(os.environ["NANOPATH_THUNDER_CKPT"], map_location="cpu", weights_only=False)
        self.cfg = checkpoint["config"]
        self.model = NanoPathFM(self.cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.name = os.environ["NANOPATH_THUNDER_MODEL_NAME"]
        self.emb_dim = int(self.cfg["model"]["dim"])
        self.vlm = False
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.cfg["data"]["mean"], std=self.cfg["data"]["std"]),
            ]
        )

    def get_transform(self):
        return self.transform

    def get_linear_probing_embeddings(self, x):
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if x.device.type == "cuda" else contextlib.nullcontext()
        with torch.inference_mode(), autocast:
            return self.model.probe_features(x).float()

    def get_segmentation_embeddings(self, x):
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if x.device.type == "cuda" else contextlib.nullcontext()
        with torch.inference_mode(), autocast:
            tokens, _ = self.model.encode_image(x, checkpoint=False)
            return tokens[:, self.model.registers :].float()
