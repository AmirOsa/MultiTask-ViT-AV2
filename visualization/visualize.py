# visualization/visualize.py
#
# IntentTrajNet — Defense Demo Visualization Script
# Author: Amir Osama — Bachelor Thesis, GUC 2025
#
# Generates a side-by-side video:
#   Left:  Front camera view from AV2 Sensor log
#   Right: BEV grid with detections (boxes), intentions (labels), trajectories (arrows)
#
# Usage:
#   python visualization/visualize.py \
#       --log_dir /path/to/av2/sensor/train/LOG_ID \
#       --weights /path/to/MultiTask_V3_epoch9.pth \
#       --output  visualization/output/demo.mp4 \
#       --num_frames 200
#
# Requirements:
#   pip install torch torchvision timm opencv-python numpy pandas
#   pip install pyarrow scipy av2
#
# The script imports your existing model and preprocessing code.
# Run from the repo root so imports resolve correctly.
#   cd /path/to/your/repo
#   python visualization/visualize.py ...

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import cv2
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path
from scipy.spatial.transform import Rotation as R_scipy

# ── Your existing project imports ────────────────────────────────────────────
from models.model_mt import IntentNetViT_MT
from utils.utils import (
    create_intentnet_lidar_bev,
    rasterize_map_ego_centric,
    decode_box_predictions,
    apply_nms,
    load_ego_poses,
    transform_points,
)
from utils.constants import (
    GRID_HEIGHT_PX, GRID_WIDTH_PX, VOXEL_SIZE_M,
    BEV_PIXEL_OFFSET_X, BEV_PIXEL_OFFSET_Y,
    Z_MIN, Z_MAX, LIDAR_HEIGHT_CHANNELS, LIDAR_SWEEPS,
    INTENTIONS_MAP_REV, VEHICLE_CATEGORIES,
    TRAJECTORY_NUM_MODES, TRAJECTORY_FUTURE_STEPS,
)

# ── Visualization constants ───────────────────────────────────────────────────
INTENTION_COLORS = {
    0: (0, 200, 0),      # KEEP_LANE       green
    1: (255, 100, 0),    # TURN_LEFT       orange
    2: (0, 100, 255),    # TURN_RIGHT      blue
    3: (200, 200, 0),    # LEFT_CHANGE     yellow
    4: (200, 0, 200),    # RIGHT_CHANGE    purple
    5: (0, 0, 255),      # STOPPING        red
    6: (128, 128, 128),  # PARKED          gray
    7: (255, 255, 255),  # OTHER           white
}

TRAJ_COLOR  = (0, 255, 255)   # cyan for trajectory lines
BOX_COLOR   = (0, 255, 0)     # green for bounding boxes
EGO_COLOR   = (255, 255, 0)   # yellow for ego vehicle marker

CONF_THRESH = 0.4
NMS_THRESH  = 0.2
MAX_DETS    = 10               # max detections to draw per frame
VIDEO_FPS   = 10

# ── BEV render helpers ────────────────────────────────────────────────────────

def ego_to_bev_pixel(cx_m: float, cy_m: float):
    """Convert ego-frame metres (cx, cy) to BEV pixel (col, row)."""
    col = int(BEV_PIXEL_OFFSET_X + cy_m / VOXEL_SIZE_M)
    row = int(BEV_PIXEL_OFFSET_Y - cx_m / VOXEL_SIZE_M)
    return col, row


def draw_rotated_box(img: np.ndarray, cx_m, cy_m, w_m, l_m, heading_rad, color, thickness=1):
    """Draw an oriented bounding box on the BEV image."""
    col, row = ego_to_bev_pixel(cx_m, cy_m)

    # Convert dimensions from metres to pixels
    w_px = w_m / VOXEL_SIZE_M
    l_px = l_m / VOXEL_SIZE_M

    # Four corners in local frame (length along X, width along Y)
    corners_local = np.array([
        [ l_px/2,  w_px/2],
        [-l_px/2,  w_px/2],
        [-l_px/2, -w_px/2],
        [ l_px/2, -w_px/2],
    ])

    # Rotate
    cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners_rot = corners_local @ R.T

    # BEV pixel convention: col = X (right), row = Y (down), but ego X = forward = -row
    # col offset: +y_ego direction
    # row offset: -x_ego direction
    corners_px = np.zeros_like(corners_rot, dtype=np.int32)
    corners_px[:, 0] = col + (corners_rot[:, 1] / VOXEL_SIZE_M).astype(int)  # col += Δy
    corners_px[:, 1] = row - (corners_rot[:, 0] / VOXEL_SIZE_M).astype(int)  # row -= Δx

    pts = corners_px.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def draw_trajectory(img: np.ndarray, cx_m, cy_m, heading_rad,
                    traj_agent_local: np.ndarray, color):
    """
    Draw trajectory arrow on BEV.
    traj_agent_local: [T, 2] positions in agent-local frame (x forward, y left)
    We rotate back to ego frame for drawing.
    """
    if traj_agent_local.shape[0] == 0:
        return

    cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
    R_back = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

    # Transform agent-local → ego frame
    traj_ego = traj_agent_local @ R_back.T
    traj_ego[:, 0] += cx_m
    traj_ego[:, 1] += cy_m

    prev_col, prev_row = ego_to_bev_pixel(cx_m, cy_m)

    step = max(1, len(traj_ego) // 10)   # draw every Nth point for clarity
    for i in range(0, len(traj_ego), step):
        col, row = ego_to_bev_pixel(traj_ego[i, 0], traj_ego[i, 1])
        if (0 <= col < GRID_WIDTH_PX) and (0 <= row < GRID_HEIGHT_PX):
            cv2.line(img, (prev_col, prev_row), (col, row), color, 1)
            prev_col, prev_row = col, row


def render_bev_frame(lidar_bev: np.ndarray) -> np.ndarray:
    """
    Convert the 290-channel LiDAR BEV into an 8-bit RGB image for display.
    We max-pool across all height channels to get a single intensity map.
    """
    # Max over all channels → [H, W]
    intensity = lidar_bev.max(axis=0)
    # Normalize to 0-255
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
    intensity_uint8 = (intensity * 255).astype(np.uint8)
    # Convert to BGR
    bev_img = cv2.cvtColor(intensity_uint8, cv2.COLOR_GRAY2BGR)
    return bev_img


# ── Data loading ──────────────────────────────────────────────────────────────

def load_lidar_sweep(sweep_dir: Path, timestamp_ns: int):
    """Load a single LiDAR sweep feather file."""
    fname = sweep_dir / f"{timestamp_ns}.feather"
    if not fname.exists():
        return None, None
    df = feather.read_feather(fname)
    xyz = df[['x', 'y', 'z']].values.astype(np.float32)
    intensity = df['intensity'].values.astype(np.float32) if 'intensity' in df.columns else np.ones(len(xyz), np.float32)
    return xyz, intensity


def get_sorted_timestamps(log_dir: Path):
    """Return sorted list of LiDAR timestamps from the sweep directory."""
    sweep_dir = log_dir / "sensors" / "lidar"
    if not sweep_dir.exists():
        raise FileNotFoundError(f"LiDAR sweep directory not found: {sweep_dir}")
    ts_list = sorted([int(f.stem) for f in sweep_dir.glob("*.feather")])
    return ts_list, sweep_dir


def load_camera_frame(log_dir: Path, timestamp_ns: int) -> np.ndarray | None:
    """Load the front-center camera frame closest to the given timestamp."""
    cam_dir = log_dir / "sensors" / "cameras" / "ring_front_center"
    if not cam_dir.exists():
        # Try alternative camera name
        cam_dir = log_dir / "sensors" / "cameras" / "ring_front_center"
        if not cam_dir.exists():
            return None

    # Find all available timestamps for this camera
    imgs = sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.png"))
    if not imgs:
        return None

    # Pick the image with the closest timestamp
    ts_available = [int(p.stem) for p in imgs]
    closest_idx = int(np.argmin(np.abs(np.array(ts_available) - timestamp_ns)))
    img_path = imgs[closest_idx]

    img = cv2.imread(str(img_path))
    return img


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(weights_path: str, device: torch.device) -> IntentNetViT_MT:
    """Load the V3 IntentTrajNet model from a checkpoint."""
    model = IntentNetViT_MT(
        backbone_type='vit',
        backbone_cfg={
            'vit_model_name_lidar': 'vit_small_patch8_224',
            'vit_model_name_map':   'vit_small_patch8_224',
            'pretrained_lidar': False,
            'pretrained_map':   False,
            'lidar_adapter_out_channels': 192,
            'map_adapter_out_channels':   192,
            'fusion_block_planes': 512,
            'fusion_block_layers': 2,
            'fusion_block_kernel_size': 3,
            'fusion_block_stride': 1,
        },
        use_trajectory=True,
        decoder_type='social_mlp',
        trajectory_head_cfg={
            'mlp_dropout': 0.0,
        },
    )

    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded from {weights_path}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, lidar_bev_np, map_bev_np, device):
    """Run a single forward pass and return decoded detections + trajectories."""
    lidar_t = torch.from_numpy(lidar_bev_np).unsqueeze(0).float().to(device)
    map_t   = torch.from_numpy(map_bev_np).unsqueeze(0).float().to(device)

    outputs = model(
        lidar_bev=lidar_t,
        map_bev=map_t,
        gt_list=None,
        use_gt_boxes_for_traj=False,
        run_traj_head=False,   # inference uses GT boxes — skip traj head here
    )

    # Detection post-processing
    cls_logits = outputs['det_cls_logits'][0]     # [22500, 1]
    box_preds  = outputs['det_box_preds'][0]      # [22500, 6]
    intent_log = outputs['intention_logits'][0]   # [22500, 8]
    anchors    = outputs['anchors']               # [22500, 5]

    scores = torch.sigmoid(cls_logits).squeeze(-1)   # [22500]
    mask   = scores > CONF_THRESH
    scores_f = scores[mask]
    boxes_f  = decode_box_predictions(box_preds[mask], anchors[mask])
    intent_f = intent_log[mask].argmax(dim=-1)

    if scores_f.shape[0] == 0:
        return [], [], [], None, None

    keep = apply_nms(boxes_f, scores_f, iou_threshold=NMS_THRESH)
    keep = keep[:MAX_DETS]

    det_boxes   = boxes_f[keep].cpu().numpy()     # [K, 5]
    det_scores  = scores_f[keep].cpu().numpy()    # [K]
    det_intents = intent_f[keep].cpu().numpy()    # [K]

    return det_boxes, det_scores, det_intents, outputs, anchors


@torch.no_grad()
def run_traj_inference(model, lidar_bev_np, map_bev_np, det_boxes_np, device):
    """
    Run trajectory head using detected boxes as input (inference mode).
    det_boxes_np: [K, 5] in ego-frame metres
    """
    if det_boxes_np.shape[0] == 0:
        return None, None

    lidar_t = torch.from_numpy(lidar_bev_np).unsqueeze(0).float().to(device)
    map_t   = torch.from_numpy(map_bev_np).unsqueeze(0).float().to(device)

    # Build gt_list using detected boxes (teacher-forcing disabled at inference)
    gt_boxes_t = torch.from_numpy(det_boxes_np).float().to(device)
    gt_list = [{'boxes_xywha': gt_boxes_t}]

    outputs = model(
        lidar_bev=lidar_t,
        map_bev=map_t,
        gt_list=gt_list,
        use_gt_boxes_for_traj=True,   # sample features at det locations
        run_traj_head=True,
    )

    y_hat = outputs['y_hat']   # [F, N, T, 4] or None
    pi    = outputs['pi']      # [N, F] or None

    return y_hat, pi


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_log(args):
    log_dir   = Path(args.log_dir)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(args.weights, device)

    # Load ego poses
    ego_poses_df = load_ego_poses(log_dir)
    ego_poses_df = ego_poses_df.sort_values('timestamp_ns').reset_index(drop=True)

    # Load map JSON path
    map_json_candidates = list(log_dir.glob("map/*.json"))
    if not map_json_candidates:
        map_json_candidates = list(log_dir.glob("*.json"))
    map_json_path = str(map_json_candidates[0]) if map_json_candidates else None
    print(f"Map JSON: {map_json_path}")

    timestamps, sweep_dir = get_sorted_timestamps(log_dir)
    print(f"Found {len(timestamps)} LiDAR sweeps in log")

    if args.num_frames > 0:
        # Start from the middle of the log for more interesting scenes
        start_idx = max(LIDAR_SWEEPS, len(timestamps) // 4)
        timestamps = timestamps[start_idx:start_idx + args.num_frames]

    print(f"Processing {len(timestamps)} frames...")

    # Video writer — we'll set size after first frame
    writer = None
    frame_count = 0

    for ts_idx, ts_ns in enumerate(timestamps):
        if ts_idx % 20 == 0:
            print(f"  Frame {ts_idx}/{len(timestamps)}")

        # ── 1. Load ego pose for this timestamp ──────────────────────────────
        ego_row_mask = ego_poses_df['timestamp_ns'] == ts_ns
        if not ego_row_mask.any():
            # Use nearest available pose
            idx = (ego_poses_df['timestamp_ns'] - ts_ns).abs().idxmin()
            ego_pose = ego_poses_df.iloc[idx]
        else:
            ego_pose = ego_poses_df[ego_row_mask].iloc[0]

        # ── 2. Load 10 LiDAR sweeps ──────────────────────────────────────────
        all_ts = sorted(ego_poses_df['timestamp_ns'].tolist())
        ts_pos = all_ts.index(ts_ns) if ts_ns in all_ts else 0
        sweep_ts_list = all_ts[max(0, ts_pos - LIDAR_SWEEPS + 1): ts_pos + 1]

        # Get current ego transform for ego-motion compensation
        ego_q_curr = ego_pose[['qx', 'qy', 'qz', 'qw']].values
        ego_t_curr = ego_pose[['tx_m', 'ty_m', 'tz_m']].values
        T_world_to_ego_curr = np.eye(4)
        T_world_to_ego_curr[:3, :3] = R_scipy.from_quat(ego_q_curr).as_matrix().T
        T_world_to_ego_curr[:3, 3] = -T_world_to_ego_curr[:3, :3] @ ego_t_curr

        points_list    = []
        intensity_list = []

        for sweep_ts in sweep_ts_list:
            pts, inten = load_lidar_sweep(sweep_dir, sweep_ts)
            if pts is None:
                points_list.append(None)
                intensity_list.append(None)
                continue

            # Ego-motion compensation: transform sweep to current ego frame
            if sweep_ts != ts_ns:
                sweep_ego_mask = ego_poses_df['timestamp_ns'] == sweep_ts
                if sweep_ego_mask.any():
                    sweep_pose = ego_poses_df[sweep_ego_mask].iloc[0]
                    sq = sweep_pose[['qx', 'qy', 'qz', 'qw']].values
                    st = sweep_pose[['tx_m', 'ty_m', 'tz_m']].values
                    T_ego_sweep_to_world = np.eye(4)
                    T_ego_sweep_to_world[:3, :3] = R_scipy.from_quat(sq).as_matrix()
                    T_ego_sweep_to_world[:3, 3] = st
                    T_sweep_to_curr = T_world_to_ego_curr @ T_ego_sweep_to_world
                    pts = transform_points(pts, T_sweep_to_curr)

            # Filter by Z range
            z_mask = (pts[:, 2] >= Z_MIN) & (pts[:, 2] < Z_MAX)
            points_list.append(pts[z_mask])
            intensity_list.append(inten[z_mask])

        # Pad to LIDAR_SWEEPS if needed
        while len(points_list) < LIDAR_SWEEPS:
            points_list.insert(0, None)
            intensity_list.insert(0, None)

        lidar_bev_np = create_intentnet_lidar_bev(points_list, intensity_list)

        # ── 3. Rasterize HD map ───────────────────────────────────────────────
        if map_json_path:
            map_bev_np = rasterize_map_ego_centric(map_json_path, ego_pose)
        else:
            map_bev_np = np.zeros((9, GRID_HEIGHT_PX, GRID_WIDTH_PX), dtype=np.float32)

        # ── 4. Run detection + intention inference ────────────────────────────
        det_boxes, det_scores, det_intents, _, _ = run_inference(
            model, lidar_bev_np, map_bev_np, device
        )

        # ── 5. Run trajectory inference on detected boxes ─────────────────────
        y_hat, pi = None, None
        if len(det_boxes) > 0:
            det_boxes_np_arr = np.array(det_boxes)
            y_hat, pi = run_traj_inference(
                model, lidar_bev_np, map_bev_np, det_boxes_np_arr, device
            )

        # ── 6. Render BEV frame ───────────────────────────────────────────────
        bev_img = render_bev_frame(lidar_bev_np)

        # Draw ego vehicle marker
        ego_col, ego_row = ego_to_bev_pixel(0, 0)
        cv2.circle(bev_img, (ego_col, ego_row), 5, EGO_COLOR, -1)
        cv2.putText(bev_img, "EGO", (ego_col + 6, ego_row - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, EGO_COLOR, 1)

        # Draw trajectories first (behind boxes)
        if y_hat is not None and pi is not None:
            y_hat_np = y_hat.cpu().numpy()   # [F, N, T, 4]
            pi_np    = pi.cpu().numpy()       # [N, F]
            best_modes = pi_np.argmax(axis=1) # [N]

            for i, box in enumerate(det_boxes):
                cx_m, cy_m, w_m, l_m, heading = box
                mode_idx = best_modes[i]
                traj = y_hat_np[mode_idx, i, :, :2]   # [T, 2] — µx, µy in agent-local
                draw_trajectory(bev_img, cx_m, cy_m, heading, traj, TRAJ_COLOR)

        # Draw detection boxes and intention labels
        for i, (box, score, intent_idx) in enumerate(zip(det_boxes, det_scores, det_intents)):
            if int(intent_idx) == 7:  # skip OTHER
                continue            
            cx_m, cy_m, w_m, l_m, heading = box
            intent_name  = INTENTIONS_MAP_REV.get(int(intent_idx), "OTHER")
            intent_color = INTENTION_COLORS.get(int(intent_idx), (255, 255, 255))

            draw_rotated_box(bev_img, cx_m, cy_m, w_m, l_m, heading, intent_color, thickness=2)

            # Label
            col, row = ego_to_bev_pixel(cx_m, cy_m)
            if 0 <= col < GRID_WIDTH_PX and 0 <= row < GRID_HEIGHT_PX:
                short_name = intent_name.replace("_", " ")
                cv2.putText(bev_img, short_name, (col + 4, row - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, intent_color, 1)

        # Stats overlay
        cv2.putText(bev_img, f"Dets: {len(det_boxes)}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(bev_img, f"Frame: {frame_count}", (5, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Intention legend
        for j, (cls_idx, cls_name) in enumerate(INTENTIONS_MAP_REV.items()):
            color = INTENTION_COLORS.get(cls_idx, (255, 255, 255))
            y_legend = GRID_HEIGHT_PX - 10 - j * 14
            cv2.rectangle(bev_img, (5, y_legend - 8), (13, y_legend), color, -1)
            cv2.putText(bev_img, cls_name.replace("_", " "),
                        (16, y_legend), cv2.FONT_HERSHEY_SIMPLEX,
                        0.28, color, 1)

        # ── 7. Load camera frame ──────────────────────────────────────────────
        cam_img = load_camera_frame(log_dir, ts_ns)

        # ── 8. Compose side-by-side frame ─────────────────────────────────────
        target_h = GRID_HEIGHT_PX  # 400

        if cam_img is not None:
            # Resize camera to same height as BEV
            cam_h, cam_w = cam_img.shape[:2]
            scale = target_h / cam_h
            new_w = int(cam_w * scale)
            cam_resized = cv2.resize(cam_img, (new_w, target_h))

            # Add label
            cv2.putText(cam_resized, "Camera (Front)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add BEV label
            cv2.putText(bev_img, "BEV | Detection + Intention + Trajectory",
                        (5, GRID_HEIGHT_PX - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            combined = np.hstack([cam_resized, bev_img])
        else:
            # No camera — show BEV only
            combined = bev_img

        # ── 9. Write frame ────────────────────────────────────────────────────
        if writer is None:
            h, w = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_path), fourcc, VIDEO_FPS, (w, h))
            print(f"Video size: {w}×{h}, writing to {out_path}")

        writer.write(combined)
        frame_count += 1

    if writer is not None:
        writer.release()
    print(f"\nDone! Written {frame_count} frames to {out_path}")
    print(f"Video duration: {frame_count / VIDEO_FPS:.1f} seconds at {VIDEO_FPS} fps")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="IntentTrajNet — Defense Demo Visualization"
    )
    parser.add_argument(
        '--log_dir', type=str, required=True,
        help='Path to one AV2 Sensor log directory (e.g. .../train/LOG_ID)'
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='Path to V3 model weights (.pth)'
    )
    parser.add_argument(
        '--output', type=str,
        default='visualization/output/demo.mp4',
        help='Output video path'
    )
    parser.add_argument(
        '--num_frames', type=int, default=150,
        help='Number of frames to process (0 = all frames)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device: auto / cpu / cuda'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_log(args)