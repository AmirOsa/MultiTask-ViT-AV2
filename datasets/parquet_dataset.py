# datasets/parquet_dataset.py
#
# New file for IntentTrajNet-AV2 V4/V5
# Author: Amir — Bachelor Thesis, GUC 2025
#
# ParquetTrajectoryDataset — loads trajectory training data from
# converted MF-format parquet files (created by sensor_to_mf.py).
#
# Used alongside ArgoverseIntentNetDataset in dual-dataset training:
#   - ArgoverseIntentNetDataset → detection + intention batches
#   - ParquetTrajectoryDataset  → trajectory batches
#
# For each parquet scenario this dataset loads:
#   1. LiDAR BEV + map BEV from the matching sensor sequence
#      (same log, same timestamp as parquet current_ts)
#   2. Agent history [N, 50, 5] for ALL agents from parquet
#      (x, y, vx, vy, heading) per timestep
#   3. GT future trajectory for focal agent only [60, 2]
#      (positions in city frame, transformed to agent-local at eval)
#   4. GT future trajectory for ALL agents [N, 60, 2]
#      (for all-agent minADE comparison with V2/V3)
#   5. Focal agent index within the N agents
#
# SOURCED: history features — Abdulbaki thesis Section 3.6
# SOURCED: future steps 60 — Abdulbaki thesis Section 3.1
# SOURCED: agent-local evaluation — Abdulbaki thesis Section 3.7

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
from scipy.spatial.transform import Rotation as R

from utils.constants import (
    LIDAR_SWEEPS,
    GRID_HEIGHT_PX,
    GRID_WIDTH_PX,
    VOXEL_SIZE_M,
    BEV_X_MIN, BEV_X_MAX,
    BEV_Y_MIN, BEV_Y_MAX,
    BEV_PIXEL_OFFSET_X,
    BEV_PIXEL_OFFSET_Y,
    Z_MIN, Z_MAX,
    LIDAR_HEIGHT_CHANNELS,
    LIDAR_TOTAL_CHANNELS,
    MAP_CHANNELS,
    VEHICLE_CATEGORIES,
    TRAJECTORY_FUTURE_STEPS,
    AGENT_HISTORY_STEPS,
    AGENT_HISTORY_FEATURES,
)
from utils.utils import (
    load_ego_poses,
    transform_points,
    create_intentnet_lidar_bev,
    rasterize_map_ego_centric,
    prepare_gt_for_frame,
)


class ParquetTrajectoryDataset(Dataset):
    """
    Dataset for trajectory training from MF-format parquet files.

    Used in V4/V5 dual-dataset training for the trajectory head.
    Each scenario provides:
        - LiDAR BEV and map BEV (from sensor data, matched by timestamp)
        - Agent history [N, 50, 5] for all agents
        - GT future trajectory for focal agent [60, 2]
        - GT future trajectory for all agents [N, 60, 2]
        - Focal agent index
        - GT boxes and intentions (for detection/intention GT)

    The dual-dataset training loop uses this alongside the sensor
    dataset — sensor batches train detection+intention, parquet batches
    train trajectory. Both update the shared backbone.

    Args:
        parquet_dir:    path to converted parquet scenarios
                        (output of sensor_to_mf.py)
        sensor_dir:     path to original AV2 sensor logs
                        (needed to load LiDAR BEV and map)
        num_sweeps:     number of LiDAR sweeps for BEV (default 10)
        is_train:       True for training, False for validation
    """

    def __init__(
        self,
        parquet_dir: str,
        sensor_dir: str,
        num_sweeps: int = LIDAR_SWEEPS,
        is_train: bool = False,
    ):
        self.parquet_dir = Path(parquet_dir)
        self.sensor_dir = Path(sensor_dir)
        self.num_sweeps = num_sweeps
        self.is_train = is_train

        # Find all scenario folders
        self.scenarios = sorted([
            d for d in self.parquet_dir.iterdir()
            if d.is_dir() and list(d.glob("*.parquet"))
        ])

        if not self.scenarios:
            raise ValueError(
                f"No parquet scenarios found in {parquet_dir}. "
                "Run sensor_to_mf.py first."
            )

        print(
            f"ParquetTrajectoryDataset: "
            f"{'Train' if is_train else 'Val'} — "
            f"{len(self.scenarios)} scenarios"
        )

        # Cache for sensor log data (ego poses, map)
        self._log_cache = {}

    def __len__(self) -> int:
        return len(self.scenarios)

    def _get_log_cache(self, log_id: str) -> dict | None:
        """Load and cache ego poses and map path for a sensor log."""
        if log_id not in self._log_cache:
            log_dir = self.sensor_dir / log_id
            if not log_dir.is_dir():
                print(f"WARNING: sensor log not found: {log_dir}")
                self._log_cache[log_id] = None
                return None

            try:
                ego_poses = load_ego_poses(log_dir)

                map_dir = log_dir / "map"
                map_files = list(map_dir.glob("log_map_archive_*.json"))
                if not map_files:
                    print(f"WARNING: no map JSON found for {log_id}")
                    self._log_cache[log_id] = None
                    return None

                self._log_cache[log_id] = {
                    "ego_poses": ego_poses,
                    "map_json_path": str(map_files[0]),
                    "log_dir": log_dir,
                }
            except Exception as e:
                print(f"ERROR loading log cache for {log_id}: {e}")
                self._log_cache[log_id] = None

        return self._log_cache.get(log_id)

    def _load_lidar_bev(
        self,
        log_dir: Path,
        current_ts_ns: int,
        ego_poses: pd.DataFrame,
    ) -> np.ndarray | None:
        """
        Load LiDAR BEV for the current timestamp.
        Same logic as ArgoverseIntentNetDataset._create_sequences
        and __getitem__ LiDAR loading section.
        SOURCED: Nadeem's dataset.py — unchanged logic
        """
        lidar_dir = log_dir / "sensors" / "lidar"
        if not lidar_dir.is_dir():
            return None

        # Get all available timestamps
        available_ts = sorted([
            int(p.stem) for p in lidar_dir.glob("*.feather")
        ])

        if len(available_ts) < self.num_sweeps:
            return None

        # Find index of current_ts_ns
        try:
            current_idx = available_ts.index(current_ts_ns)
        except ValueError:
            # Find closest timestamp
            diffs = [abs(ts - current_ts_ns) for ts in available_ts]
            current_idx = diffs.index(min(diffs))

        # Get sweep timestamps (num_sweeps ending at current)
        start_idx = max(0, current_idx - self.num_sweeps + 1)
        sweep_ts_list = available_ts[start_idx:current_idx + 1]

        # Pad with earliest if not enough sweeps
        while len(sweep_ts_list) < self.num_sweeps:
            sweep_ts_list = [sweep_ts_list[0]] + sweep_ts_list

        # Get current ego pose for coordinate transform
        current_ego_row = ego_poses[
            ego_poses['timestamp_ns'] == current_ts_ns
        ]
        if current_ego_row.empty:
            # Find closest
            idx = (ego_poses['timestamp_ns'] - current_ts_ns).abs().argsort().iloc[0]
            current_ego_row = ego_poses.iloc[[idx]]

        current_ego = current_ego_row.iloc[0]
        try:
            rot_mat = R.from_quat([
                current_ego['qx'], current_ego['qy'],
                current_ego['qz'], current_ego['qw']
            ]).as_matrix()
        except ValueError:
            return None

        world_SE3_ego = np.eye(4)
        world_SE3_ego[:3, :3] = rot_mat
        world_SE3_ego[:3, 3] = [
            current_ego['tx_m'], current_ego['ty_m'], current_ego['tz_m']
        ]
        ego_SE3_world = np.linalg.inv(world_SE3_ego)

        # Load sweeps
        points_list, intensity_list = [], []
        for ts_sweep in sweep_ts_list:
            sweep_path = lidar_dir / f"{ts_sweep}.feather"
            if not sweep_path.is_file():
                points_list.append(None)
                intensity_list.append(None)
                continue

            try:
                import pyarrow.feather as feather
                sweep_df = pd.read_feather(
                    sweep_path, columns=['x', 'y', 'z', 'intensity']
                )
                if sweep_df.empty:
                    points_list.append(None)
                    intensity_list.append(None)
                    continue
            except Exception:
                points_list.append(None)
                intensity_list.append(None)
                continue

            pts = sweep_df[['x', 'y', 'z']].values
            intensity = sweep_df['intensity'].values.astype(np.float32)

            # Transform to current ego frame
            sweep_ego_row = ego_poses[
                ego_poses['timestamp_ns'] == ts_sweep
            ]
            if sweep_ego_row.empty:
                points_list.append(None)
                intensity_list.append(None)
                continue

            sw = sweep_ego_row.iloc[0]
            try:
                sw_rot = R.from_quat([
                    sw['qx'], sw['qy'], sw['qz'], sw['qw']
                ]).as_matrix()
            except ValueError:
                points_list.append(None)
                intensity_list.append(None)
                continue

            sw_tf = np.eye(4)
            sw_tf[:3, :3] = sw_rot
            sw_tf[:3, 3] = [sw['tx_m'], sw['ty_m'], sw['tz_m']]
            rel_tf = ego_SE3_world @ sw_tf
            pts_ego = transform_points(pts, rel_tf)

            points_list.append(pts_ego)
            intensity_list.append(intensity)

        if all(p is None for p in points_list):
            return None

        return create_intentnet_lidar_bev(points_list, intensity_list)

    def _extract_agent_history(
        self,
        df: pd.DataFrame,
        track_id: str,
        hist_steps: int = AGENT_HISTORY_STEPS,
        feat_dim: int = AGENT_HISTORY_FEATURES,
    ) -> np.ndarray:
        """
        Extract history [50, 5] for one agent from parquet dataframe.

        History features: (x, y, vx, vy, heading)
        SOURCED: Abdulbaki thesis Section 3.6
            f^t_i = [p^t_i, v^t_i, h^t_i] ∈ R^5
            position (x,y) + velocity (vx,vy) + heading (1)

        Observed=True rows are the history (past 50 timesteps).
        Zero-pad if agent has fewer than 50 observed timesteps.

        Args:
            df:       parquet scenario dataframe
            track_id: agent UUID string
            hist_steps: 50 — SOURCED: Abdulbaki Section 3.1
            feat_dim:   5  — SOURCED: Abdulbaki Section 3.6

        Returns:
            history: [50, 5] — zero padded where not observed
        """
        history = np.zeros((hist_steps, feat_dim), dtype=np.float32)

        agent_hist = df[
            (df['track_id'] == track_id) &
            (df['observed'] == True)
        ].sort_values('timestep')

        if agent_hist.empty:
            return history

        # Take last hist_steps observed timesteps
        agent_hist = agent_hist.tail(hist_steps)
        n_obs = len(agent_hist)

        # Fill from end — most recent at position hist_steps-1
        # Zero padding at start if fewer than 50 timesteps observed
        start_pos = hist_steps - n_obs

        history[start_pos:, 0] = agent_hist['position_x'].values
        history[start_pos:, 1] = agent_hist['position_y'].values
        history[start_pos:, 2] = agent_hist['velocity_x'].values
        history[start_pos:, 3] = agent_hist['velocity_y'].values
        history[start_pos:, 4] = agent_hist['heading'].values

        return history
        # [50, 5]

    def _extract_future_trajectory(
        self,
        df: pd.DataFrame,
        track_id: str,
        future_steps: int = TRAJECTORY_FUTURE_STEPS,
    ) -> tuple:
        """
        Extract future trajectory GT [60, 2] and mask [60] for one agent.

        Future positions come from observed=False rows in parquet.
        SOURCED: future_steps=60 — Abdulbaki thesis Section 3.1

        Returns:
            traj: [60, 2] — (x, y) positions, zero where not available
            mask: [60]    — True where position data exists
        """
        traj = np.zeros((future_steps, 2), dtype=np.float32)
        mask = np.zeros(future_steps, dtype=bool)

        agent_future = df[
            (df['track_id'] == track_id) &
            (df['observed'] == False)
        ].sort_values('timestep').head(future_steps)

        if agent_future.empty:
            return traj, mask

        n_future = len(agent_future)
        traj[:n_future, 0] = agent_future['position_x'].values
        traj[:n_future, 1] = agent_future['position_y'].values
        mask[:n_future] = True

        return traj, mask

    def __getitem__(self, idx: int) -> dict | None:
        """
        Load one parquet scenario.

        Returns dict with:
            lidar_bev:         [290, 400, 720] — LiDAR BEV tensor
            map_bev:           [9,   400, 720] — map BEV tensor
            agent_history:     [N, 50, 5]      — history for all agents
            gt_traj_focal:     [60, 2]          — focal agent GT future
            gt_mask_focal:     [60]             — focal agent mask
            gt_traj_all:       [N, 60, 2]       — all agents GT future
            gt_mask_all:       [N, 60]          — all agents mask
            focal_idx:         int              — index of focal agent in N
            focal_track_id:    str              — focal agent UUID
            gt_boxes:          [N, 5]           — GT boxes for teacher forcing
            gt_intentions:     [N]              — intention labels
            track_ids:         list             — agent UUIDs
            scenario_id:       str              — scenario identifier
            current_ts_ns:     int              — current timestamp
        """
        scenario_dir = self.scenarios[idx]

        try:
            # Load parquet file
            pq_files = list(scenario_dir.glob("*.parquet"))
            if not pq_files:
                return None

            df = pd.read_parquet(pq_files[0])
            scenario_id = scenario_dir.name

            # Extract log_id and window index from scenario_id
            # Format: {log_id}_w{window_idx:03d}
            parts = scenario_id.rsplit('_w', 1)
            if len(parts) != 2:
                return None
            log_id = parts[0]

            # Get focal agent
            focal_track_id = df['focal_track_id'].iloc[0]

            # Get current timestamp (end of history = timestep HIST_STEPS-1)
            # In parquet: timestep 49 = last observed step = current
            current_step = AGENT_HISTORY_STEPS - 1  # = 49
            current_rows = df[df['timestep'] == current_step]
            if current_rows.empty:
                return None

            # Get actual timestamp_ns for matching sensor sequence
            current_ts_ns = int(current_rows['timestamp_ns'].iloc[0])

            # Load sensor log data
            log_cache = self._get_log_cache(log_id)
            if log_cache is None:
                return None

            ego_poses = log_cache['ego_poses']
            map_json_path = log_cache['map_json_path']
            log_dir = log_cache['log_dir']

            # Get ego pose at current timestamp for map rasterization
            current_ego_row = ego_poses[
                ego_poses['timestamp_ns'] == current_ts_ns
            ]
            if current_ego_row.empty:
                # Find closest
                idx_close = (
                    ego_poses['timestamp_ns'] - current_ts_ns
                ).abs().argsort().iloc[0]
                current_ego_pose = ego_poses.iloc[idx_close]
            else:
                current_ego_pose = current_ego_row.iloc[0]

            # Load LiDAR BEV
            lidar_bev_np = self._load_lidar_bev(
                log_dir, current_ts_ns, ego_poses
            )
            if lidar_bev_np is None:
                return None

            # Load map BEV
            map_bev_np = rasterize_map_ego_centric(
                map_json_path, current_ego_pose
            )

            # Get all agents with sufficient history
            # Use agents with object_category > 0 (not unknown)
            # category 3 = focal, category 1 = scored, category 0 = other
            all_track_ids = df['track_id'].unique().tolist()

            # Filter to agents present at current timestep
            # and have sufficient history
            valid_agents = []
            for tid in all_track_ids:
                agent_rows = df[df['track_id'] == tid]
                obs_rows = agent_rows[agent_rows['observed'] == True]
                if len(obs_rows) >= 5:
                    # At least 5 observed timesteps
                    valid_agents.append(tid)

            if not valid_agents:
                return None

            N = len(valid_agents)

            # Find focal agent index
            if focal_track_id not in valid_agents:
                # Focal agent filtered out — skip scenario
                return None
            focal_idx = valid_agents.index(focal_track_id)

            # Extract history for all agents [N, 50, 5]
            all_histories = np.zeros(
                (N, AGENT_HISTORY_STEPS, AGENT_HISTORY_FEATURES),
                dtype=np.float32
            )
            for i, tid in enumerate(valid_agents):
                all_histories[i] = self._extract_agent_history(df, tid)

            # Extract future trajectories for all agents [N, 60, 2]
            all_trajs = np.zeros(
                (N, TRAJECTORY_FUTURE_STEPS, 2),
                dtype=np.float32
            )
            all_masks = np.zeros(
                (N, TRAJECTORY_FUTURE_STEPS),
                dtype=bool
            )
            for i, tid in enumerate(valid_agents):
                traj, mask = self._extract_future_trajectory(df, tid)
                all_trajs[i] = traj
                all_masks[i] = mask

            # Focal agent trajectory
            gt_traj_focal = all_trajs[focal_idx]
            # [60, 2]
            gt_mask_focal = all_masks[focal_idx]
            # [60]

            # Build GT boxes from parquet current timestep positions
            # Use position_x, position_y, heading from parquet
            # Width and length not available in parquet — use defaults
            # This is used for teacher forcing in trajectory head
            gt_boxes_list = []
            for tid in valid_agents:
                agent_current = df[
                    (df['track_id'] == tid) &
                    (df['timestep'] == current_step)
                ]
                if agent_current.empty:
                    # Use last observed position
                    agent_obs = df[
                        (df['track_id'] == tid) &
                        (df['observed'] == True)
                    ].sort_values('timestep')
                    if agent_obs.empty:
                        gt_boxes_list.append([0., 0., 2., 4.5, 0.])
                        continue
                    agent_current = agent_obs.tail(1)

                row = agent_current.iloc[0]

                # Convert city frame position to ego frame
                # Parquet positions are in city frame
                # We need ego frame for BEV coordinate system
                ego_tx = float(current_ego_pose['tx_m'])
                ego_ty = float(current_ego_pose['ty_m'])
                try:
                    ego_yaw = R.from_quat([
                        float(current_ego_pose['qx']),
                        float(current_ego_pose['qy']),
                        float(current_ego_pose['qz']),
                        float(current_ego_pose['qw'])
                    ]).as_euler('xyz')[2]
                except Exception:
                    ego_yaw = 0.0

                # City frame → ego frame
                dx = float(row['position_x']) - ego_tx
                dy = float(row['position_y']) - ego_ty
                cos_yaw = np.cos(-ego_yaw)
                sin_yaw = np.sin(-ego_yaw)
                cx_ego = cos_yaw * dx - sin_yaw * dy
                cy_ego = sin_yaw * dx + cos_yaw * dy
                heading_ego = float(row['heading']) - ego_yaw

                gt_boxes_list.append([
                    cx_ego, cy_ego,
                    2.0, 4.5,
                    # Default vehicle dimensions
                    # Width=2m, Length=4.5m
                    # ASSUMED: parquet doesn't store box dimensions
                    heading_ego
                ])

            gt_boxes_np = np.array(gt_boxes_list, dtype=np.float32)
            # [N, 5]

            # Intention labels — not available in parquet
            # Use zeros (KEEP_LANE=0) as placeholder
            # Detection/intention training uses sensor dataset not parquet
            gt_intentions_np = np.zeros(N, dtype=np.int64)

            return {
                "lidar_bev": torch.from_numpy(lidar_bev_np).float(),
                # [290, 400, 720]

                "map_bev": torch.from_numpy(map_bev_np).float(),
                # [9, 400, 720]

                "agent_history": torch.from_numpy(all_histories).float(),
                # [N, 50, 5] — history for all agents
                # SOURCED: Abdulbaki thesis Section 3.6

                "gt_traj_focal": torch.from_numpy(gt_traj_focal).float(),
                # [60, 2] — focal agent future positions
                # SOURCED: Abdulbaki thesis Section 3.1

                "gt_mask_focal": torch.from_numpy(gt_mask_focal).bool(),
                # [60] — True where focal trajectory is valid

                "gt_traj_all": torch.from_numpy(all_trajs).float(),
                # [N, 60, 2] — all agents future positions
                # Used for all-agent minADE (comparable to V2/V3)

                "gt_mask_all": torch.from_numpy(all_masks).bool(),
                # [N, 60]

                "focal_idx": focal_idx,
                # int — index of focal agent in N agents

                "focal_track_id": focal_track_id,
                # str — focal agent UUID

                "gt_boxes": torch.from_numpy(gt_boxes_np).float(),
                # [N, 5] — for teacher forcing in trajectory head

                "gt_intentions": torch.from_numpy(gt_intentions_np).long(),
                # [N] — placeholder zeros (not used for training)

                "track_ids": valid_agents,
                # list of N agent UUID strings

                "scenario_id": scenario_id,
                # str — for debugging

                "current_ts_ns": current_ts_ns,
                # int — timestamp for V3 re-evaluation matching
            }

        except Exception as e:
            print(
                f"ERROR in ParquetTrajectoryDataset __getitem__ "
                f"idx={idx} scenario={scenario_dir.name}: {e}"
            )
            traceback.print_exc()
            return None


def parquet_collate_fn(batch: list) -> dict | None:
    """
    Custom collate for ParquetTrajectoryDataset.

    N varies per scenario so we keep items as lists.
    Only lidar_bev and map_bev are stacked into tensors.

    Returns None if all items in batch failed.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Stack BEV tensors — fixed size [290/9, 400, 720]
    lidar_bevs = torch.stack([item["lidar_bev"] for item in batch])
    map_bevs = torch.stack([item["map_bev"] for item in batch])

    # Keep variable-N items as lists
    return {
        "lidar_bev":      lidar_bevs,
        # [B, 290, 400, 720]

        "map_bev":        map_bevs,
        # [B, 9, 400, 720]

        "agent_history":  [item["agent_history"] for item in batch],
        # list of B tensors, each [N_b, 50, 5]

        "gt_traj_focal":  [item["gt_traj_focal"] for item in batch],
        # list of B tensors, each [60, 2]

        "gt_mask_focal":  [item["gt_mask_focal"] for item in batch],
        # list of B tensors, each [60]

        "gt_traj_all":    [item["gt_traj_all"] for item in batch],
        # list of B tensors, each [N_b, 60, 2]

        "gt_mask_all":    [item["gt_mask_all"] for item in batch],
        # list of B tensors, each [N_b, 60]

        "focal_idx":      [item["focal_idx"] for item in batch],
        # list of B ints

        "focal_track_id": [item["focal_track_id"] for item in batch],
        # list of B strings

        "gt_boxes":       [item["gt_boxes"] for item in batch],
        # list of B tensors, each [N_b, 5]

        "gt_intentions":  [item["gt_intentions"] for item in batch],
        # list of B tensors, each [N_b]

        "track_ids":      [item["track_ids"] for item in batch],
        # list of B lists of strings

        "scenario_id":    [item["scenario_id"] for item in batch],
        # list of B strings

        "current_ts_ns":  [item["current_ts_ns"] for item in batch],
        # list of B ints
    }