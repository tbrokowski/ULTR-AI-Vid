"""
Shared dataset utilities used by the Benin and South Africa adapters.

Both adapters expose the same public API:
- `PatientLevelDataset`
- `collate_patient_batch`
- `LungUltrasoundDataModule`

The dataset-specific files keep the metadata parsing and site/label mapping
logic that differs between Benin and SA, while this module keeps the shared
video/image transform primitives in one place.
"""

import random
from typing import Sequence

import torch
import torchvision.transforms.functional as F
from PIL import ImageEnhance, ImageFilter


NUM_PATH_CLASSES = 4


class UltrasoundPreprocessing(object):
    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        return img


class UltrasoundNoiseAugment(object):
    def __call__(self, tensor):
        speckle = torch.randn_like(tensor) * 0.2
        tensor = tensor * (1 + speckle)
        tensor = torch.clamp(tensor, 0, 1)
        return tensor


class TemporallyConsistentTransforms:
    def __init__(
        self,
        resize_size=(224, 224),
        degrees=25,
        translate=(0.15, 0.15),
        scale=(0.65, 1.45),
        brightness=0.3,
        contrast=0.3,
        blur_kernel_size=3,
        blur_sigma=(0.1, 0.5),
        noise_std=0.2,
        augment_prob=0.5,
        blur_prob=0.2,
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
    ):
        self.resize_size = resize_size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.brightness = brightness
        self.contrast = contrast
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std
        self.augment_prob = augment_prob
        self.blur_prob = blur_prob
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, video_frames: Sequence):
        if not video_frames:
            return torch.empty(0, 3, *self.resize_size)

        augment_params = self._sample_augmentation_parameters()

        transformed_frames = []
        for frame in video_frames:
            frame = self._apply_resize(frame)
            frame = self._apply_contrast_enhancement(frame, augment_params)
            frame = self._apply_affine_transform(frame, augment_params)
            frame = self._apply_color_jitter(frame, augment_params)
            frame = self._apply_gaussian_blur(frame, augment_params)

            frame_tensor = F.to_tensor(frame)
            transformed_frames.append(frame_tensor)

        video_tensor = torch.stack(transformed_frames)
        video_tensor = self._apply_noise(video_tensor, augment_params)
        video_tensor = self._apply_normalization(video_tensor)

        return video_tensor

    def _sample_augmentation_parameters(self):
        params = {}

        if random.random() < self.augment_prob:
            params["apply_affine"] = True
            params["angle"] = random.uniform(-self.degrees, self.degrees)
            params["translate"] = (
                random.uniform(-self.translate[0], self.translate[0]),
                random.uniform(-self.translate[1], self.translate[1]),
            )
            params["scale"] = random.uniform(self.scale[0], self.scale[1])
        else:
            params["apply_affine"] = False

        params["brightness_factor"] = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        params["contrast_factor"] = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)

        if random.random() < self.blur_prob:
            params["apply_blur"] = True
            params["blur_sigma"] = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
        else:
            params["apply_blur"] = False

        params["noise_multiplier"] = torch.randn(1, 1, 1, 1) * self.noise_std
        return params

    def _apply_resize(self, frame):
        return F.resize(frame, self.resize_size)

    def _apply_contrast_enhancement(self, frame, params):
        enhancer = ImageEnhance.Contrast(frame)
        return enhancer.enhance(1.2)

    def _apply_affine_transform(self, frame, params):
        if not params["apply_affine"]:
            return frame

        width, height = frame.size
        translate_pixels = (
            int(params["translate"][0] * width),
            int(params["translate"][1] * height),
        )

        return F.affine(
            frame,
            angle=params["angle"],
            translate=translate_pixels,
            scale=params["scale"],
            shear=0,
            fill=0,
        )

    def _apply_color_jitter(self, frame, params):
        frame = F.adjust_brightness(frame, params["brightness_factor"])
        frame = F.adjust_contrast(frame, params["contrast_factor"])
        return frame

    def _apply_gaussian_blur(self, frame, params):
        if not params["apply_blur"]:
            return frame
        return frame.filter(ImageFilter.GaussianBlur(radius=params["blur_sigma"]))

    def _apply_noise(self, video_tensor, params):
        speckle = params["noise_multiplier"] * torch.randn_like(video_tensor)
        video_tensor = video_tensor * (1 + speckle)
        return torch.clamp(video_tensor, 0, 1)

    def _apply_normalization(self, video_tensor):
        return (video_tensor - self.mean) / self.std


class SimpleVideoTransforms:
    def __init__(
        self,
        resize_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.resize_size = resize_size
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, video_frames: Sequence):
        if not video_frames:
            return torch.empty(0, 3, *self.resize_size)

        transformed_frames = []
        for frame in video_frames:
            frame = F.resize(frame, self.resize_size)
            frame_tensor = F.to_tensor(frame)
            transformed_frames.append(frame_tensor)

        video_tensor = torch.stack(transformed_frames)
        return (video_tensor - self.mean) / self.std
