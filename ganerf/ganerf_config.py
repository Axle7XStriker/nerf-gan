from __future__ import annotations

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from ganerf.ganerf import GANerfModelConfig
from ganerf.ganerf_pipeline import GanerfPipelineConfig

PATCH_SIZE = 128

ganerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="ganerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=GanerfPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                ),
            ),
            discriminator_datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=PATCH_SIZE*PATCH_SIZE,
                eval_num_rays_per_batch=PATCH_SIZE*PATCH_SIZE,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                ),
                patch_size=PATCH_SIZE,
                pixel_sampler=PatchPixelSamplerConfig(keep_full_image=True, patch_size=PATCH_SIZE),
            ),
            model=GANerfModelConfig(eval_num_rays_per_chunk=1 << 15, patch_size=PATCH_SIZE),
            train_generator=True,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="GANerf implementation"

)