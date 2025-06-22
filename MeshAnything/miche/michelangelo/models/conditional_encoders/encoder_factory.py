# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPModel, CLIPTokenizer
from collections import OrderedDict

from MeshAnything.miche.michelangelo.data.transforms import RandomResize
import clip
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPProcessor
from utils import get_device, is_mps_device


class AbstractEncoder(nn.Module):
    embedding_dim: int

    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class"):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class FrozenCLIPTextEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        tokenizer_version=None,
        device=None,
        max_length=77,
        zero_embedding_radio: float = 0.1,
    ):
        super().__init__()
        
        # Use device detection if not provided
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_version if tokenizer_version is not None else version)
        self.transformer = CLIPModel.from_pretrained(version)
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.max_length = max_length
        self.zero_embedding_radio = zero_embedding_radio
        self._move_flag = False

    @property
    def clip(self):
        return self.transformer

    def move(self):
        if not self._move_flag:
            self.transformer.to(self.device)
            self._move_flag = True

    def unconditional_embedding(self, batch_size):
        return torch.zeros(batch_size, 77, 768, device=self.device, dtype=self.transformer.text_model.embeddings.token_embedding.weight.dtype)

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            tokens = batch_encoding["input_ids"].to(self.device)
            # Use last_hidden_state instead of hidden_states for newer transformers
            if hasattr(self.transformer.text_model, 'embeddings'):
                outputs = self.transformer.text_model(input_ids=tokens)
                z = outputs.last_hidden_state
            else:
                z = self.transformer.text_model(input_ids=tokens).last_hidden_state

        return z

    def encode(self, text):
        self.move()
        return self(text)


class FrozenAlignedCLIPTextEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        tokenizer_version=None,
        device=None,
        max_length=77,
        zero_embedding_radio: float = 0.1,
    ):
        super().__init__()
        
        # Use device detection if not provided
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_version if tokenizer_version is not None else version)
        self.transformer = CLIPModel.from_pretrained(version)
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.max_length = max_length
        self.zero_embedding_radio = zero_embedding_radio
        self._move_flag = False

    @property
    def clip(self):
        return self.transformer

    def move(self):
        if not self._move_flag:
            self.transformer.to(self.device)
            self._move_flag = True

    def unconditional_embedding(self, batch_size):
        return torch.zeros(batch_size, 768, device=self.device, dtype=self.transformer.text_model.embeddings.token_embedding.weight.dtype)

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            tokens = batch_encoding["input_ids"].to(self.device)
            # Use last_hidden_state for compatibility
            if hasattr(self.transformer.text_model, 'embeddings'):
                outputs = self.transformer.text_model(input_ids=tokens)
                z = outputs.last_hidden_state[:, -1]  # Take the last token
            else:
                z = self.transformer.text_model(input_ids=tokens).last_hidden_state[:, -1]

        return z

    def encode(self, text):
        self.move()
        return self(text)


class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
            self,
            version="openai/clip-vit-large-patch14",
            device=None,
            zero_embedding_radio=0.1,
            normalize_embedding=True,
            num_projection_vector=0,
            linear_mapping_bias=True,
            reverse_visual_projection=False,
    ):
        super().__init__()
        
        # Use device detection if not provided
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)

        self.clip_dict = OrderedDict()
        self.clip_name = os.path.split(version)[-1]

        clip_model = CLIPModel.from_pretrained(version)
        clip_model.text_model = None
        clip_model.text_projection = None
        clip_model = clip_model.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.clip_dict[self.clip_name] = clip_model

        self.transform = transforms.Compose(
            [
                transforms.Resize(224, transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(224),  # crop a (224, 224) square
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.zero_embedding_radio = zero_embedding_radio

        self.num_projection_vector = num_projection_vector
        self.reverse_visual_projection = reverse_visual_projection
        self.normalize_embedding = normalize_embedding

        embedding_dim = (
            clip_model.visual_projection.in_features
            if reverse_visual_projection
            else clip_model.visual_projection.out_features
        )
        self.embedding_dim = embedding_dim
        if self.num_projection_vector > 0:
            self.projection = nn.Linear(
                embedding_dim,
                clip_model.visual_projection.out_features * num_projection_vector,
                bias=linear_mapping_bias,
            )
            nn.init.normal_(self.projection.weight, std=embedding_dim ** -0.5)

        self._move_flag = False

    @property
    def clip(self):
        return self.clip_dict[self.clip_name]

    def unconditional_embedding(self, batch_size):
        zero = torch.zeros(
            batch_size,
            1,
            self.embedding_dim,
            device=self.device,
            dtype=self.clip.visual_projection.weight.dtype,
        )
        if self.num_projection_vector > 0:
            zero = self.projection(zero).view(batch_size, self.num_projection_vector, -1)
        return zero

    def forward(self, image, value_range=(-1, 1), zero_embedding_radio=0):
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = image.to(self.device, dtype=self.clip.visual_projection.weight.dtype)

        if self.reverse_visual_projection:
            z = self.clip.vision_model(self.transform(image))[1]
        else:
            z = self.clip.get_image_features(self.transform(image))

        if self.normalize_embedding:
            z = z / z.norm(dim=-1, keepdim=True)
        if z.ndim == 2:
            z = z.unsqueeze(dim=-2)

        if zero_embedding_radio > 0:
            mask = torch.rand((len(image), 1, 1), device=z.device, dtype=z.dtype) < zero_embedding_radio
            z = z * mask.to(z)

        if self.num_projection_vector > 0:
            z = self.projection(z).view(len(image), self.num_projection_vector, -1)

        return z

    def move(self):
        if not self._move_flag:
            self.clip_dict[self.clip_name].to(self.device)
            self._move_flag = True

    def encode(self, image):
        self.move()
        return self(image, zero_embedding_radio=self.zero_embedding_radio)


class FrozenCLIPImageGridEmbedder(AbstractEncoder):

    def __init__(
            self,
            version="openai/clip-vit-large-patch14",
            device=None,
            zero_embedding_radio=0.1,
    ):
        super().__init__()
        
        # Use device detection if not provided
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)

        self.clip_dict = OrderedDict()
        self.clip_name = os.path.split(version)[-1]

        clip_model: CLIPModel = CLIPModel.from_pretrained(version)
        clip_model.text_model = None
        clip_model.text_projection = None
        clip_model = clip_model.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.clip_dict[self.clip_name] = clip_model

        self.transform = transforms.Compose(
            [
                transforms.Resize(224, transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(224),  # crop a (224, 224) square
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.zero_embedding_radio = zero_embedding_radio
        self.embedding_dim = clip_model.vision_embed_dim

        self._move_flag = False

    @property
    def clip(self):
        return self.clip_dict[self.clip_name]

    def move(self):
        if not self._move_flag:
            self.clip_dict[self.clip_name].to(self.device)
            self._move_flag = True

    def unconditional_embedding(self, batch_size):
        zero = torch.zeros(
            batch_size,
            self.clip.vision_model.embeddings.num_positions,
            self.embedding_dim,
            device=self.device,
            dtype=self.clip.visual_projection.weight.dtype,
        )
        return zero

    def forward(self, image, value_range=(-1, 1), zero_embedding_radio=0):
        self.move()

        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = image.to(self.device, dtype=self.clip.visual_projection.weight.dtype)

        z = self.clip.vision_model(self.transform(image)).last_hidden_state

        if zero_embedding_radio > 0:
            mask = torch.rand((len(image), 1, 1), device=z.device, dtype=z.dtype) >= zero_embedding_radio
            z = z * mask.to(z)

        return z

    def encode(self, image):
        self.move()
        return self(image, zero_embedding_radio=self.zero_embedding_radio)


class MoECLIPImageEncoder(nn.Module):
    def __init__(
            self,
            versions,
            hidden_state_dim,
            num_projection_vector=8,
            zero_embedding_radio=0.1,
            device=None,
            precision="fp16",
            normalize=False,
            clip_max=0,
            transform_type="base",
            argument_p=0.2,
    ):
        super().__init__()

        # Use device detection if not provided
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.hidden_state_dim = hidden_state_dim
        self.zero_embedding_radio = zero_embedding_radio
        self.num_projection_vector = num_projection_vector
        
        # Adjust precision for MPS compatibility
        if is_mps_device(self.device) and precision == "fp16":
            precision = "fp32"  # MPS works better with fp32
            
        self.dtype = dict(fp16=torch.float16, fp32=torch.float32, bf16=torch.bfloat16)[precision]
        self.normalize = normalize
        self.clip_max = clip_max

        if transform_type == "base":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(224, transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop(224),  # crop a (224, 224) square
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        elif transform_type == "crop_blur_resize":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(224, transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop(224),  # crop a (224, 224) square
                    transforms.RandomApply(
                        transforms=[
                            transforms.RandomResizedCrop(
                                size=224,
                                scale=(0.8, 1.0),
                                ratio=(0.99, 1.01),
                                interpolation=transforms.InterpolationMode.BICUBIC,
                            ),
                        ],
                        p=argument_p,
                    ),
                    transforms.RandomApply(
                        transforms=[
                            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 5)),
                        ],
                        p=argument_p,
                    ),
                    transforms.RandomApply(
                        transforms=[
                            RandomResize(size=224, resize_radio=(0.2, 1)),
                        ],
                        p=argument_p,
                    ),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        else:
            raise ValueError(f"invalid {transform_type=}")

        if isinstance(versions, str):
            versions = (versions,)

        # 如果直接把clips定位为当前类的子module，1. 会在保存ckp时存无用的多个权重。 2. pl会调用to，导致layer_norm的权重也被转换成fp16
        clips = OrderedDict()

        for v in versions:
            # Load clips on CPU first to avoid device allocation issues
            clips[v], _ = clip.load(name=v, device="cpu", jit=False, download_root=None)
            delattr(clips[v], "transformer")
            clips[v].eval()
            clips[v].requires_grad_(False)

        self.clips_hidden_dim = sum(clips[v].ln_final.weight.size(0) for v in clips)

        if self.num_projection_vector == 0:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(self.clips_hidden_dim, hidden_state_dim * self.num_projection_vector, bias=True)
            self.projection.to(dtype=self.dtype)
            nn.init.normal_(self.projection.weight, std=self.clips_hidden_dim ** -0.5)

        self.clips = clips

        self._move_flag = False

    def move(self):
        if not self._move_flag:
            for k in self.clips:
                self.clips[k].to(self.device)
            self._move_flag = True

    def unconditional_embedding(self, batch_size=None):
        zero = torch.zeros(
            batch_size,
            self.clips_hidden_dim,
            device=self.device,
            dtype=self.dtype,
        )
        if self.num_projection_vector > 0:
            zero = self.projection(zero).view(batch_size, self.num_projection_vector, -1)
        return zero

    def convert_embedding(self, z):
        if self.num_projection_vector > 0:
            z = self.projection(z.type(self.projection.weight.dtype)).view(len(z), self.num_projection_vector, -1)
        return z

    def forward(self, image, value_range=(-1, 1), zero_embedding_radio=0):
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = self.transform(image)

        with torch.no_grad():
            embs = []
            for v in self.clips:
                x = self.clips[v].encode_image(image)
                if self.normalize:
                    x = x / x.norm(p=2, dim=-1, keepdim=True) * (x.size(-1) ** 0.5)
                    # clip_max only works with normalization
                    if self.clip_max > 0:
                        x = x.clamp(-self.clip_max, self.clip_max)
                embs.append(x)

            z = torch.cat(embs, dim=-1)
            if self.normalize:
                z /= z.size(-1) ** 0.5

        if zero_embedding_radio > 0:
            mask = torch.rand((len(image), 1, 1), device=z.device, dtype=z.dtype) >= zero_embedding_radio
            z = z + mask.to(z)

        if self.num_projection_vector > 0:
            z = self.projection(z).view(len(image), self.num_projection_vector, -1)
        return z

    def encode(self, image):
        self.move()
        return self(image, zero_embedding_radio=self.zero_embedding_radio)
