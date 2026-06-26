# models/backbone.py
#
# Adapted from Nadeem Mohamed's IntentNetViT
# Original repo: https://github.com/Nadeem202020/VisionTransformer-Intention-Prediction
# Original file: model_vit.py
#
# Modifications:
#   1. Copied TwoStreamViTBackbone and its helper classes
#      (BasicBlock, conv3x3_for_basic, conv1x1_for_basic) verbatim.
#      Updated import paths to match new repo structure.
#   2. Added SwinBackbone class — new for V3.
#      Uses Swin-Tiny pretrained on ImageNet as a drop-in replacement
#      for TwoStreamViTBackbone. Produces same output shape [B, 512, 50, 90].
#      SOURCED: Swin Transformer — Liu et al., ICCV 2021 Best Paper.
#      SOURCED: ImageNet pretrained ViT on LiDAR — RangeViT, Ando et al.
#      CVPR 2023 — proves ImageNet pretrained ViTs transfer to LiDAR data.
#
# Usage:
#   V1, V2: from models.backbone import TwoStreamViTBackbone
#   V3:     from models.backbone import SwinBackbone

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import warnings
import numpy as np

from utils.constants import (
    LIDAR_TOTAL_CHANNELS,  # 290 — 10 sweeps × 29 height channels
    MAP_CHANNELS,          # 9  — HD map layers
    GRID_HEIGHT_PX,        # 400
    GRID_WIDTH_PX,         # 720
)


# =============================================================================
# Helper functions and BasicBlock
# Copied verbatim from Nadeem's model_vit.py
# SOURCED: Nadeem thesis Section 3.3
# =============================================================================

def conv3x3_for_basic(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    kernel_size: int = 3
) -> nn.Conv2d:
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )


def conv1x1_for_basic(
    in_planes: int,
    out_planes: int,
    stride: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    """
    ResNet BasicBlock for the fusion layer.
    Copied verbatim from Nadeem's model_vit.py.
    SOURCED: Nadeem thesis Section 3.3
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        kernel_size: int = 3
    ):
        super().__init__()
        self.conv1 = conv3x3_for_basic(inplanes, planes, stride, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_for_basic(planes, planes, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# =============================================================================
# TwoStreamViTBackbone
# Copied verbatim from Nadeem's model_vit.py
# SOURCED: Nadeem thesis Section 3.3
# Only change: import path updated from constants → utils.constants
# =============================================================================

class TwoStreamViTBackbone(nn.Module):
    """
    Two-stream Vision Transformer backbone for LiDAR BEV + HD map.

    Copied verbatim from Nadeem's model_vit.py.
    SOURCED: Nadeem thesis Section 3.3

    Architecture:
        LiDAR stream: vit_small_patch8_224, in_chans=290
        Map stream:   vit_small_patch8_224, in_chans=9
        Both → adapter layers (192ch each)
        → concatenate (384ch)
        → fusion block (2× BasicBlock → 512ch)

    Output: [B, 512, 50, 90]
        50 = 400 / 8  (height / patch_size)
        90 = 720 / 8  (width / patch_size)
    """

    def __init__(
        self,
        lidar_input_channels: int = LIDAR_TOTAL_CHANNELS,
        map_input_channels: int = MAP_CHANNELS,
        vit_model_name_lidar: str = 'vit_small_patch8_224',
        vit_model_name_map: str = 'vit_small_patch8_224',
        pretrained_lidar: bool = False,
        pretrained_map: bool = False,
        img_size: tuple = (GRID_HEIGHT_PX, GRID_WIDTH_PX),
        drop_path_rate_lidar: float = 0.1,
        drop_path_rate_map: float = 0.1,
        lidar_adapter_out_channels: int = 192,
        map_adapter_out_channels: int = 192,
        fusion_block_planes: int = 512,
        fusion_block_layers: int = 2,
        fusion_block_kernel_size: int = 3,
        fusion_block_stride: int = 1,
        res_block_type: type = BasicBlock
    ):
        super().__init__()
        self.img_size = img_size
        self.lidar_adapter_out_channels = lidar_adapter_out_channels
        self.map_adapter_out_channels = map_adapter_out_channels

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.vit_lidar = timm.create_model(
                vit_model_name_lidar,
                pretrained=pretrained_lidar,
                in_chans=lidar_input_channels,
                img_size=self.img_size,
                drop_path_rate=drop_path_rate_lidar
            )
        self.vit_lidar.head = nn.Identity()
        self.lidar_embed_dim = self.vit_lidar.embed_dim
        self.lidar_num_prefix_tokens = getattr(
            self.vit_lidar, 'num_prefix_tokens',
            1 if hasattr(self.vit_lidar, 'cls_token') and
            self.vit_lidar.cls_token is not None else 0
        )
        self.lidar_grid_size, _ = self._get_patch_info(self.vit_lidar, "LiDAR")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.vit_map = timm.create_model(
                vit_model_name_map,
                pretrained=pretrained_map,
                in_chans=map_input_channels,
                img_size=self.img_size,
                drop_path_rate=drop_path_rate_map
            )
        self.vit_map.head = nn.Identity()
        self.map_embed_dim = self.vit_map.embed_dim
        self.map_num_prefix_tokens = getattr(
            self.vit_map, 'num_prefix_tokens',
            1 if hasattr(self.vit_map, 'cls_token') and
            self.vit_map.cls_token is not None else 0
        )
        self.map_grid_size, _ = self._get_patch_info(self.vit_map, "Map")

        if (self.lidar_grid_size != self.map_grid_size and
                self.lidar_grid_size is not None and
                self.map_grid_size is not None):
            warnings.warn(
                f"LiDAR patch grid {self.lidar_grid_size} and "
                f"Map patch grid {self.map_grid_size} differ."
            )

        self.feature_map_grid_h = (
            self.lidar_grid_size[0] if self.lidar_grid_size
            else (self.map_grid_size[0] if self.map_grid_size else 0)
        )
        self.feature_map_grid_w = (
            self.lidar_grid_size[1] if self.lidar_grid_size
            else (self.map_grid_size[1] if self.map_grid_size else 0)
        )

        self.adapter_lidar = nn.Sequential(
            nn.LayerNorm(self.lidar_embed_dim),
            nn.Linear(self.lidar_embed_dim, self.lidar_adapter_out_channels),
            nn.GELU()
        )
        self.adapter_map = nn.Sequential(
            nn.LayerNorm(self.map_embed_dim),
            nn.Linear(self.map_embed_dim, self.map_adapter_out_channels),
            nn.GELU()
        )

        self.fusion_input_channels = (
            self.lidar_adapter_out_channels + self.map_adapter_out_channels
        )
        self.fusion_block_stride = fusion_block_stride

        self.fusion_block = self._make_fusion_layer(
            res_block_type,
            fusion_block_planes,
            fusion_block_layers,
            stride=self.fusion_block_stride,
            current_inplanes=self.fusion_input_channels,
            kernel_size_for_block=fusion_block_kernel_size
        )
        self.final_feature_channels = fusion_block_planes * res_block_type.expansion

        print(f"TwoStreamViTBackbone Initialized:")
        print(f"  LiDAR ViT: {vit_model_name_lidar}, Adapter Out: {self.lidar_adapter_out_channels}")
        print(f"  Map ViT: {vit_model_name_map}, Adapter Out: {self.map_adapter_out_channels}")
        print(f"  Output: {self.final_feature_channels} channels, {self.feature_map_grid_h}×{self.feature_map_grid_w}")

    def _get_patch_info(self, vit_model, stream_name=""):
        grid_size, num_patches = None, 0
        try:
            patch_embed = vit_model.patch_embed
            if hasattr(patch_embed, 'grid_size') and patch_embed.grid_size is not None:
                grid_size = tuple(patch_embed.grid_size)
                num_patches = grid_size[0] * grid_size[1]
            elif hasattr(patch_embed, 'num_patches'):
                num_patches = patch_embed.num_patches
                if self.img_size and hasattr(patch_embed, 'patch_size'):
                    patch_h, patch_w = (
                        patch_embed.patch_size
                        if isinstance(patch_embed.patch_size, tuple)
                        else (patch_embed.patch_size, patch_embed.patch_size)
                    )
                    gs_h = self.img_size[0] // patch_h
                    gs_w = self.img_size[1] // patch_w
                    if gs_h * gs_w == num_patches:
                        grid_size = (gs_h, gs_w)
        except AttributeError:
            print(f"Error ({stream_name}): Could not access patch_embed attributes.")
        return grid_size, num_patches

    def _process_stream(self, x, vit_stream, num_prefix_tokens, grid_size, adapter, stream_name):
        tokens_all = vit_stream.forward_features(x)
        patch_tokens = tokens_all[:, num_prefix_tokens:]
        adapted_tokens = adapter(patch_tokens)
        B, N, C = adapted_tokens.shape
        if grid_size and N == grid_size[0] * grid_size[1]:
            Hf, Wf = grid_size
            return adapted_tokens.permute(0, 2, 1).contiguous().view(B, C, Hf, Wf)
        else:
            print(f"ERROR ({stream_name}): Token count {N} or grid_size {grid_size} issue.")
            return None

    def _make_fusion_layer(self, block, planes, num_blocks, stride=1, current_inplanes=0, kernel_size_for_block=3):
        downsample = None
        out_channels_block = planes * block.expansion
        if stride != 1 or current_inplanes != out_channels_block:
            downsample = nn.Sequential(
                conv1x1_for_basic(current_inplanes, out_channels_block, stride),
                nn.BatchNorm2d(out_channels_block)
            )
        layers = [block(current_inplanes, planes, stride, downsample, kernel_size=kernel_size_for_block)]
        for _ in range(1, num_blocks):
            layers.append(block(planes * block.expansion, planes, kernel_size=kernel_size_for_block))
        return nn.Sequential(*layers)

    def forward(self, lidar_bev: torch.Tensor, map_bev: torch.Tensor) -> torch.Tensor:
        lidar_fm = self._process_stream(
            lidar_bev, self.vit_lidar,
            self.lidar_num_prefix_tokens,
            self.lidar_grid_size,
            self.adapter_lidar, "LiDAR"
        )
        if lidar_fm is None:
            return torch.zeros(
                lidar_bev.shape[0], self.final_feature_channels,
                self.feature_map_grid_h or 1, self.feature_map_grid_w or 1,
                device=lidar_bev.device
            )

        map_fm = self._process_stream(
            map_bev, self.vit_map,
            self.map_num_prefix_tokens,
            self.map_grid_size,
            self.adapter_map, "Map"
        )
        if map_fm is None:
            return torch.zeros(
                map_bev.shape[0], self.final_feature_channels,
                self.feature_map_grid_h or 1, self.feature_map_grid_w or 1,
                device=map_bev.device
            )

        if lidar_fm.shape[2:] != map_fm.shape[2:]:
            map_fm = F.interpolate(
                map_fm, size=lidar_fm.shape[2:],
                mode='bilinear', align_corners=False
            )

        fused = torch.cat([lidar_fm, map_fm], dim=1)
        return self.fusion_block(fused)
        # [B, 512, 50, 90]


# =============================================================================
# SwinBackbone
# NEW — for V3
# =============================================================================

class SwinBackbone(nn.Module):
    """
    Swin Transformer backbone for LiDAR BEV + HD map.

    NEW for V3 — replaces TwoStreamViTBackbone with a pretrained
    hierarchical ViT that produces richer multi-scale features.

    Key differences from TwoStreamViTBackbone:
        1. Hierarchical feature pyramid (4 stages) vs single-scale ViT
        2. Windowed local attention (linear cost) vs global attention (quadratic)
        3. ImageNet pretrained weights → faster convergence on limited AV2 data
        4. Patch size 4 (not 8) but with window_size=5 to fit 400×720 input

    SOURCED: Swin Transformer — Liu et al., ICCV 2021 Best Paper
        "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
        https://arxiv.org/abs/2103.14030

    SOURCED: ImageNet pretrained ViT on LiDAR data transfers well despite
    domain gap — RangeViT (Ando et al., CVPR 2023).

    SOURCED: Swin used as backbone in BEV autonomous driving models —
    BEVFusion (Liu et al., 2022), BEVerse (Zhang et al., 2022).

    Input channel adaptation:
        Swin pretrained on ImageNet uses 3-channel RGB input.
        For 290-channel LiDAR BEV, timm replaces patch_embed.proj
        with Conv2d(290, embed_dim, ...) and initialises it by
        averaging the 3-channel pretrained weights and repeating.
        SOURCED: timm library handles in_chans != 3 automatically
        using mean-and-repeat initialisation. This preserves the
        spatial filter structure from ImageNet pretraining.

    Window size adaptation:
        swin_tiny_patch4_window7_224 uses window_size=7 for 224×224 input.
        Your BEV is 400×720. With patch_size=4, the feature map is
        100×180. Window size 7 does not divide 100 evenly (100/7 = 14.28).
        We use window_size=5: 100/5=20, 180/5=36 — both divide evenly.
        NEEDS TEST: verify Swin initialises correctly with window_size=5.
        Alternative: use window_size=4 (100/4=25, 180/4=45).

    Output:
        Same as TwoStreamViTBackbone: [B, 512, 50, 90]
        Achieved by taking Swin's last stage output [B, 768, 13, 23]
        and upsampling + projecting to [B, 512, 50, 90].
        ASSUMED: upsampling from 13×23 to 50×90 to match existing
        heads. Alternative: redesign heads for new feature map size.

    Args:
        lidar_input_channels: 290 (10 sweeps × 29 height channels)
        map_input_channels:   9  (HD map layers)
        pretrained:           True for V3 (ImageNet pretrained)
        out_channels:         512 to match TwoStreamViTBackbone output
        img_size:             (400, 720) — BEV dimensions
    """

    def __init__(
        self,
        lidar_input_channels: int = LIDAR_TOTAL_CHANNELS,
        # SOURCED: 290 from constants.py
        map_input_channels: int = MAP_CHANNELS,
        # SOURCED: 9 from constants.py
        pretrained: bool = True,
        # True for V3 — ImageNet pretrained weights
        # SOURCED: benefit of pretraining shown in RangeViT (CVPR 2023)
        out_channels: int = 512,
        # Must match TwoStreamViTBackbone.final_feature_channels
        # so that all downstream heads work without modification
        # SOURCED: 512 from Nadeem thesis Section 3.3
        img_size: tuple = (GRID_HEIGHT_PX, GRID_WIDTH_PX),
        # (400, 720) — BEV image size
        window_size: int = 5,
        # NEEDS TEST: must divide feature map dimensions evenly
        # feature map = img_size / patch_size = 400/4=100, 720/4=180
        # window_size=5: 100/5=20 ✓, 180/5=36 ✓
        # window_size=4: 100/4=25 ✓, 180/4=45 ✓
        # window_size=7: 100/7=14.28 ✗ — does not divide evenly
        swin_model_name: str = 'swin_tiny_patch4_window7_224',
        # Base model — window size overridden by window_size param
        # SOURCED: swin_tiny chosen for balance of performance and compute
        # Swin-S would be stronger but ~1.5× more parameters
    ) -> None:
        super().__init__()

        self.lidar_input_channels = lidar_input_channels
        self.map_input_channels = map_input_channels
        self.out_channels = out_channels
        self.img_size = img_size

        # =====================================================================
        # LiDAR stream — Swin-Tiny for 290-channel BEV
        # =====================================================================
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.swin_lidar = timm.create_model(
                swin_model_name,
                pretrained=pretrained,
                in_chans=lidar_input_channels,
                # timm handles channel mismatch automatically:
                # averages pretrained 3ch weights → repeats for 290ch
                # SOURCED: timm source code — in_chans adaptation
                img_size=img_size,
                window_size=window_size,
                # NEEDS TEST: verify correct initialisation
                num_classes=0,
                # Remove classification head — we use features only
            )

        # =====================================================================
        # Map stream — Swin-Tiny for 9-channel map BEV
        # =====================================================================
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.swin_map = timm.create_model(
                swin_model_name,
                pretrained=pretrained,
                in_chans=map_input_channels,
                img_size=img_size,
                window_size=window_size,
                num_classes=0,
            )

        # Get Swin output channels (last stage)
        # swin_tiny: stages produce 96, 192, 384, 768 channels
        # We take the last stage: 768 channels
        # SOURCED: Swin-Tiny architecture — Liu et al., ICCV 2021
        swin_out_channels = self._get_swin_out_channels()

        # =====================================================================
        # Fusion projection
        # Two Swin streams → concatenate → project to out_channels
        #
        # Both streams output [B, swin_out_channels, H', W'] where
        # H' = img_size[0] / (patch_size × 2^(num_stages-1))
        #    = 400 / (4 × 8) = 400 / 32 = 12.5 → ~13
        # W' = img_size[1] / 32 = 720 / 32 = 22.5 → ~23
        #
        # We upsample from [B, 2×768, 13, 23] → [B, 512, 50, 90]
        # to match TwoStreamViTBackbone output shape exactly.
        # ASSUMED: bilinear upsampling — simple and avoids redesigning
        # heads. A learnable deconvolution would be more principled.
        # =====================================================================
        fusion_in_channels = swin_out_channels * 2
        # 768 × 2 = 1536 (LiDAR stream + Map stream concatenated)

        self.fusion_proj = nn.Sequential(
            # Point-wise convolution to reduce channels
            nn.Conv2d(fusion_in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second conv for additional mixing
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.final_feature_channels = out_channels
        # Matches TwoStreamViTBackbone.final_feature_channels = 512

        # Target output spatial size — must match heads' expectations
        # SOURCED: 50×90 from Nadeem's heads (22500 = 5 × 50 × 90)
        self.target_h = GRID_HEIGHT_PX // 8  # 50
        self.target_w = GRID_WIDTH_PX // 8   # 90

        print(f"SwinBackbone Initialized:")
        print(f"  Model: {swin_model_name}, window_size={window_size}")
        print(f"  Pretrained: {pretrained}")
        print(f"  LiDAR in_chans: {lidar_input_channels}")
        print(f"  Map in_chans: {map_input_channels}")
        print(f"  Swin output channels: {swin_out_channels}")
        print(f"  Fusion output: {out_channels} channels, {self.target_h}×{self.target_w}")

    def _get_swin_out_channels(self) -> int:
        """
        Get the number of output channels from Swin's last stage.
        swin_tiny produces 768 at the last stage.
        SOURCED: Swin-Tiny architecture — Liu et al., ICCV 2021
        """
        try:
            # timm Swin models expose num_features
            return self.swin_lidar.num_features
        except AttributeError:
            # Fallback — swin_tiny last stage = 768
            return 768

    def forward(
        self,
        lidar_bev: torch.Tensor,
        map_bev: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of SwinBackbone.

        Args:
            lidar_bev: [B, 290, 400, 720]
            map_bev:   [B,   9, 400, 720]

        Returns:
            feature_map: [B, 512, 50, 90]
            Same shape as TwoStreamViTBackbone output — all downstream
            heads (DetectionHead, IntentionHead, TrajectoryHead) work
            without modification.
        """
        # --- LiDAR stream ---
        # timm Swin's forward_features returns [B, H', W', C] or [B, C]
        # depending on the model version
        lidar_feat = self.swin_lidar.forward_features(lidar_bev)
        # Expected: [B, H', W', C] where H'≈13, W'≈23, C=768

        # --- Map stream ---
        map_feat = self.swin_map.forward_features(map_bev)
        # [B, H', W', C]

        # --- Handle timm output format ---
        # Some timm Swin versions output [B, H*W, C] (flattened)
        # Others output [B, H, W, C]
        # We normalise to [B, C, H, W] for conv operations
        lidar_feat = self._to_spatial(lidar_feat)
        map_feat = self._to_spatial(map_feat)
        # Both: [B, C, H', W']

        # --- Align spatial dimensions if they differ ---
        if lidar_feat.shape[2:] != map_feat.shape[2:]:
            map_feat = F.interpolate(
                map_feat,
                size=lidar_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # --- Concatenate and project ---
        fused = torch.cat([lidar_feat, map_feat], dim=1)
        # [B, 1536, H', W']

        fused = self.fusion_proj(fused)
        # [B, 512, H', W']

        # --- Upsample to target size [B, 512, 50, 90] ---
        # ASSUMED: bilinear upsampling to match existing head expectations
        # A learnable upsampling (deconv) would be more principled
        # but adds parameters and complexity
        if fused.shape[2:] != (self.target_h, self.target_w):
            fused = F.interpolate(
                fused,
                size=(self.target_h, self.target_w),
                mode='bilinear',
                align_corners=False
            )
        # [B, 512, 50, 90]

        return fused

    def _to_spatial(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Convert timm Swin output to [B, C, H, W] format.

        timm Swin can output:
            [B, H, W, C] — spatial with channels last
            [B, H*W, C]  — flattened spatial
            [B, C]        — global pooled (if num_classes > 0, shouldn't happen)

        We convert all to [B, C, H, W].
        """
        if feat.dim() == 4:
            # [B, H, W, C] — permute to [B, C, H, W]
            return feat.permute(0, 3, 1, 2).contiguous()
        elif feat.dim() == 3:
            # [B, H*W, C] — need to reshape to [B, C, H, W]
            B, N, C = feat.shape
            # Infer spatial dimensions
            # For swin_tiny with patch4 and 400×720 input:
            # H' ≈ 13, W' ≈ 23, H'×W' ≈ 299
            # We find H', W' such that H'×W' = N
            H = self.target_h  # Use target as approximation
            W = N // H
            if H * W != N:
                # Can't reshape cleanly — use square root approximation
                H = int(N ** 0.5)
                W = N // H
            feat = feat.permute(0, 2, 1).contiguous()
            # [B, C, H*W]
            feat = feat.view(B, C, H, W)
            return feat
        else:
            raise ValueError(
                f"Unexpected Swin output shape: {feat.shape}. "
                f"Expected 3D or 4D tensor."
            )