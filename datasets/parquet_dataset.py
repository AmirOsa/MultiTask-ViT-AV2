# datasets/parquet_dataset.py
#
# New file for IntentTrajNet-AV2 V4/V5
# Author: Amir — Bachelor Thesis, GUC 2025
#
# ParquetTrajectoryDataset — loads trajectory training data from
# converted MF-format parquet files (created by sensor_to_mf.py).
#
# COORDINATE FRAME NOTE:
#   Parquet files store ALL positions in city frame (absolute coordinates).
#   This dataset converts everything to ego frame for consistency with
#   the BEV-based detection/intention pipeline:
#
#   GT boxes:        city frame → ego frame (for BEV feature sampling)
#   Trajectory GT:   city frame → ego frame (for loss computation)
#   Agent history:   city frame → relative frame (agent-centric, for GRU)
#
#   City → ego frame transformation:
#       dx = city_x - ego_tx
#       dy = city_y - ego_ty
#       ego_x = cos(-ego_yaw) * dx - sin(-ego_yaw) * dy
#       ego_y = sin(-ego_yaw) * dx + cos(-ego_yaw) * dy
#
#   Agent history additionally normalized to agent-centric:
#       Subtract current position so GRU sees relative displacements
#       rather than absolute coordinates. This makes history
#       independent of absolute location — only motion matters.
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
)


def _city_to_ego(
    city_x: np.ndarray,
    city_y: np.ndarray,
    ego_tx: float,
    ego_ty: float,
    ego_yaw: float,
) -> tuple:
    """
    Convert city frame positions to ego frame.

    Transformation:
        dx = city_x - ego_tx
        dy = city_y - ego_ty
        ego_x = cos(-ego_yaw) * dx - sin(-ego_yaw) * dy
        ego_y = sin(-ego_yaw) * dx + cos(-ego_yaw) * dy

    Args:
        city_x, city_y: positions in city frame
        ego_tx, ego_ty: ego vehicle position in city frame
        ego_yaw:        ego vehicle heading in city frame (radians)

    Returns:
        ego_x, ego_y: positions in ego frame
    """
    dx = city_x - ego_tx
    dy = city_y - ego_ty
    cos_yaw = np.cos(-ego_yaw)
    sin_yaw = np.sin(-ego_yaw)
    ego_x = cos_yaw * dx - sin_yaw * dy
    ego_y = sin_yaw * dx + cos_yaw * dy
    return ego_x, ego_y


class ParquetTrajectoryDataset(Dataset):
    """
    Dataset for trajectory training from MF-format parquet files.

    Used in V4/V5 dual-dataset training for the trajectory head.
    Each scenario provides:
        - LiDAR BEV and map BEV (from sensor data, matched by timestamp)
        - Agent history [N, 50, 5] in agent-centric relative frame
        - GT future trajectory for focal agent [60, 2] in ego frame
        - GT future trajectory for all agents [N, 60, 2] in ego frame
        - Focal agent index
        - GT boxes in ego frame [N, 5]

    All positions converted from parquet city frame to ego frame.
    History additionally normalized to agent-centric (relative to
    each agent's current position) so GRU learns motion patterns
    independent of absolute location.

    Args:
        parquet_dir:    path to converted parquet scenarios
        sensor_dir:     path to original AV2 sensor logs
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
        self.sensor_dir  = Path(sensor_dir)
        self.num_sweeps  = num_sweeps
        self.is_train    = is_train

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
                map_dir   = log_dir / "map"
                map_files = list(map_dir.glob("log_map_archive_*.json"))
                if not map_files:
                    print(f"WARNING: no map JSON found for {log_id}")
                    self._log_cache[log_id] = None
                    return None

                self._log_cache[log_id] = {
                    "ego_poses":     ego_poses,
                    "map_json_path": str(map_files[0]),
                    "log_dir":       log_dir,
                }
            except Exception as e:
                print(f"ERROR loading log cache for {log_id}: {e}")
                self._log_cache[log_id] = None

        return self._log_cache.get(log_id)

    def _get_ego_pose_at(
        self,
        ego_poses: pd.DataFrame,
        ts_ns: int,
    ) -> tuple | None:
        """
        Get ego pose (tx, ty, yaw) at a specific timestamp.
        Returns None if timestamp not found.
        """
        row = ego_poses[ego_poses['timestamp_ns'] == ts_ns]
        if row.empty:
            # Find closest timestamp
            idx = (ego_poses['timestamp_ns'] - ts_ns).abs().argsort().iloc[0]
            row = ego_poses.iloc[[idx]]

        r = row.iloc[0]
        try:
            yaw = R.from_quat([
                float(r['qx']), float(r['qy']),
                float(r['qz']), float(r['qw'])
            ]).as_euler('xyz')[2]
        except Exception:
            return None

        return float(r['tx_m']), float(r['ty_m']), yaw

    def _load_lidar_bev(
        self,
        log_dir: Path,
        current_ts_ns: int,
        ego_poses: pd.DataFrame,
    ) -> np.ndarray | None:
        """
        Load LiDAR BEV for the current timestamp.
        SOURCED: Nadeem's dataset.py — same logic as ArgoverseIntentNetDataset
        """
        lidar_dir = log_dir / "sensors" / "lidar"
        if not lidar_dir.is_dir():
            return None

        available_ts = sorted([int(p.stem) for p in lidar_dir.glob("*.feather")])
        if len(available_ts) < self.num_sweeps:
            return None

        try:
            current_idx = available_ts.index(current_ts_ns)
        except ValueError:
            diffs = [abs(ts - current_ts_ns) for ts in available_ts]
            current_idx = diffs.index(min(diffs))

        start_idx    = max(0, current_idx - self.num_sweeps + 1)
        sweep_ts_list = available_ts[start_idx:current_idx + 1]
        while len(sweep_ts_list) < self.num_sweeps:
            sweep_ts_list = [sweep_ts_list[0]] + sweep_ts_list

        # Get current ego pose for coordinate transform
        ego_pose = self._get_ego_pose_at(ego_poses, current_ts_ns)
        if ego_pose is None:
            return None
        ego_tx, ego_ty, ego_yaw = ego_pose

        rot_mat = R.from_euler('z', ego_yaw).as_matrix()
        world_SE3_ego        = np.eye(4)
        world_SE3_ego[:3, :3] = rot_mat
        world_SE3_ego[:3, 3]  = [ego_tx, ego_ty, 0.0]
        ego_SE3_world         = np.linalg.inv(world_SE3_ego)

        points_list, intensity_list = [], []
        for ts_sweep in sweep_ts_list:
            sweep_path = lidar_dir / f"{ts_sweep}.feather"
            if not sweep_path.is_file():
                points_list.append(None)
                intensity_list.append(None)
                continue

            try:
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

            pts       = sweep_df[['x', 'y', 'z']].values
            intensity = sweep_df['intensity'].values.astype(np.float32)

            sweep_ego_pose = self._get_ego_pose_at(ego_poses, ts_sweep)
            if sweep_ego_pose is None:
                points_list.append(None)
                intensity_list.append(None)
                continue

            sw_tx, sw_ty, sw_yaw = sweep_ego_pose
            sw_rot = R.from_euler('z', sw_yaw).as_matrix()
            sw_tf  = np.eye(4)
            sw_tf[:3, :3] = sw_rot
            sw_tf[:3, 3]  = [sw_tx, sw_ty, 0.0]
            rel_tf  = ego_SE3_world @ sw_tf
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
        ego_tx: float,
        ego_ty: float,
        ego_yaw: float,
        hist_steps: int = AGENT_HISTORY_STEPS,
        feat_dim: int = AGENT_HISTORY_FEATURES,
    ) -> np.ndarray:
        """
        Extract agent-centric history [50, 5] for one agent.

        History features: (rel_x, rel_y, vx_ego, vy_ego, heading_ego)
        where rel_x, rel_y are positions relative to agent's current position
        in ego frame. This makes the GRU learn motion patterns independent
        of absolute location.

        SOURCED: Abdulbaki thesis Section 3.6
            f^t_i = [p^t_i, v^t_i, h^t_i] ∈ R^5

        Coordinate conversion:
            1. City frame → ego frame (subtract ego position, rotate by -ego_yaw)
            2. Ego frame → agent-centric (subtract agent's current position)

        Args:
            ego_tx, ego_ty, ego_yaw: ego pose at current timestamp
        """
        history = np.zeros((hist_steps, feat_dim), dtype=np.float32)

        agent_hist = df[
            (df['track_id'] == track_id) &
            (df['observed'] == True)
        ].sort_values('timestep')

        if agent_hist.empty:
            return history

        agent_hist = agent_hist.tail(hist_steps)
        n_obs      = len(agent_hist)
        start_pos  = hist_steps - n_obs

        # Step 1: Convert city frame positions → ego frame
        city_x = agent_hist['position_x'].values
        city_y = agent_hist['position_y'].values
        ego_x, ego_y = _city_to_ego(city_x, city_y, ego_tx, ego_ty, ego_yaw)

        # Step 2: Convert velocities city frame → ego frame
        city_vx = agent_hist['velocity_x'].values
        city_vy = agent_hist['velocity_y'].values
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        ego_vx  = cos_yaw * city_vx - sin_yaw * city_vy
        ego_vy  = sin_yaw * city_vx + cos_yaw * city_vy

        # Step 3: Convert heading city frame → ego frame
        city_heading = agent_hist['heading'].values
        ego_heading  = city_heading - ego_yaw

        # Step 4: Normalize positions to agent-centric
        # Use the last observed position (current position) as origin
        # This makes the GRU see relative displacements, not absolute coords
        if n_obs > 0:
            current_ego_x = ego_x[-1]
            current_ego_y = ego_y[-1]
            ego_x = ego_x - current_ego_x
            ego_y = ego_y - current_ego_y

        # Fill history array (zero-padded at start for missing timesteps)
        history[start_pos:, 0] = ego_x
        history[start_pos:, 1] = ego_y
        history[start_pos:, 2] = ego_vx
        history[start_pos:, 3] = ego_vy
        history[start_pos:, 4] = ego_heading

        return history

    def _extract_future_trajectory(
        self,
        df: pd.DataFrame,
        track_id: str,
        ego_tx: float,
        ego_ty: float,
        ego_yaw: float,
        future_steps: int = TRAJECTORY_FUTURE_STEPS,
    ) -> tuple:
        """
        Extract future trajectory GT [60, 2] in ego frame.

        Converts from city frame (parquet) to ego frame (BEV-consistent).
        This ensures trajectory GT matches the model's prediction frame.

        SOURCED: future_steps=60 — Abdulbaki thesis Section 3.1

        Args:
            ego_tx, ego_ty, ego_yaw: ego pose at current timestamp
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

        # Convert city frame → ego frame
        city_x = agent_future['position_x'].values
        city_y = agent_future['position_y'].values
        ego_x, ego_y = _city_to_ego(city_x, city_y, ego_tx, ego_ty, ego_yaw)

        traj[:n_future, 0] = ego_x
        traj[:n_future, 1] = ego_y
        mask[:n_future]    = True

        return traj, mask

    def __getitem__(self, idx: int) -> dict | None:
        """
        Load one parquet scenario.

        All positions converted from city frame to ego frame.
        History additionally normalized to agent-centric.
        """
        scenario_dir = self.scenarios[idx]

        try:
            pq_files = list(scenario_dir.glob("*.parquet"))
            if not pq_files:
                return None

            df          = pd.read_parquet(pq_files[0])
            scenario_id = scenario_dir.name

            parts = scenario_id.rsplit('_w', 1)
            if len(parts) != 2:
                return None
            log_id = parts[0]

            focal_track_id = df['focal_track_id'].iloc[0]

            # Get current timestamp (timestep 49 = last observed step)
            current_step  = AGENT_HISTORY_STEPS - 1  # = 49
            current_rows  = df[df['timestep'] == current_step]
            if current_rows.empty:
                return None
            current_ts_ns = int(current_rows['timestamp_ns'].iloc[0])

            # Load sensor log
            log_cache = self._get_log_cache(log_id)
            if log_cache is None:
                return None

            ego_poses      = log_cache['ego_poses']
            map_json_path  = log_cache['map_json_path']
            log_dir        = log_cache['log_dir']

            # Get ego pose at current timestamp
            ego_pose = self._get_ego_pose_at(ego_poses, current_ts_ns)
            if ego_pose is None:
                return None
            ego_tx, ego_ty, ego_yaw = ego_pose

            # Get ego pose as pandas Series for map rasterization
            current_ego_row = ego_poses[
                ego_poses['timestamp_ns'] == current_ts_ns
            ]
            if current_ego_row.empty:
                idx_close = (
                    ego_poses['timestamp_ns'] - current_ts_ns
                ).abs().argsort().iloc[0]
                current_ego_pose = ego_poses.iloc[idx_close]
            else:
                current_ego_pose = current_ego_row.iloc[0]

            # Load LiDAR BEV
            lidar_bev_np = self._load_lidar_bev(log_dir, current_ts_ns, ego_poses)
            if lidar_bev_np is None:
                return None

            # Load map BEV
            map_bev_np = rasterize_map_ego_centric(map_json_path, current_ego_pose)

            # Get valid agents
            all_track_ids = df['track_id'].unique().tolist()
            valid_agents  = []
            for tid in all_track_ids:
                obs_rows = df[(df['track_id'] == tid) & (df['observed'] == True)]
                if len(obs_rows) >= 5:
                    valid_agents.append(tid)

            if not valid_agents:
                return None

            N = len(valid_agents)

            if focal_track_id not in valid_agents:
                return None
            focal_idx = valid_agents.index(focal_track_id)

            # Extract history [N, 50, 5] — agent-centric, ego frame
            all_histories = np.zeros(
                (N, AGENT_HISTORY_STEPS, AGENT_HISTORY_FEATURES),
                dtype=np.float32
            )
            for i, tid in enumerate(valid_agents):
                all_histories[i] = self._extract_agent_history(
                    df, tid, ego_tx, ego_ty, ego_yaw
                )

            # Extract future trajectories [N, 60, 2] — ego frame
            all_trajs = np.zeros((N, TRAJECTORY_FUTURE_STEPS, 2), dtype=np.float32)
            all_masks = np.zeros((N, TRAJECTORY_FUTURE_STEPS), dtype=bool)
            for i, tid in enumerate(valid_agents):
                traj, mask = self._extract_future_trajectory(
                    df, tid, ego_tx, ego_ty, ego_yaw
                )
                all_trajs[i] = traj
                all_masks[i] = mask

            gt_traj_focal = all_trajs[focal_idx]   # [60, 2]
            gt_mask_focal = all_masks[focal_idx]   # [60]

            # Build GT boxes in ego frame [N, 5]
            gt_boxes_list = []
            for tid in valid_agents:
                agent_current = df[
                    (df['track_id'] == tid) &
                    (df['timestep'] == current_step)
                ]
                if agent_current.empty:
                    agent_obs = df[
                        (df['track_id'] == tid) & (df['observed'] == True)
                    ].sort_values('timestep')
                    if agent_obs.empty:
                        gt_boxes_list.append([0., 0., 2., 4.5, 0.])
                        continue
                    agent_current = agent_obs.tail(1)

                row = agent_current.iloc[0]

                # Convert city frame → ego frame
                cx_ego, cy_ego = _city_to_ego(
                    np.array([float(row['position_x'])]),
                    np.array([float(row['position_y'])]),
                    ego_tx, ego_ty, ego_yaw
                )
                heading_ego = float(row['heading']) - ego_yaw

                gt_boxes_list.append([
                    float(cx_ego[0]),
                    float(cy_ego[0]),
                    2.0, 4.5,
                    heading_ego
                ])

            gt_boxes_np      = np.array(gt_boxes_list, dtype=np.float32)
            gt_intentions_np = np.zeros(N, dtype=np.int64)

            return {
                "lidar_bev":      torch.from_numpy(lidar_bev_np).float(),
                "map_bev":        torch.from_numpy(map_bev_np).float(),
                "agent_history":  torch.from_numpy(all_histories).float(),
                # [N, 50, 5] — agent-centric relative positions, ego frame velocities/heading
                "gt_traj_focal":  torch.from_numpy(gt_traj_focal).float(),
                # [60, 2] — focal agent future in EGO FRAME
                "gt_mask_focal":  torch.from_numpy(gt_mask_focal).bool(),
                "gt_traj_all":    torch.from_numpy(all_trajs).float(),
                # [N, 60, 2] — all agents future in EGO FRAME
                "gt_mask_all":    torch.from_numpy(all_masks).bool(),
                "focal_idx":      focal_idx,
                "focal_track_id": focal_track_id,
                "gt_boxes":       torch.from_numpy(gt_boxes_np).float(),
                # [N, 5] — in EGO FRAME
                "gt_intentions":  torch.from_numpy(gt_intentions_np).long(),
                "track_ids":      valid_agents,
                "scenario_id":    scenario_id,
                "current_ts_ns":  current_ts_ns,
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
    N varies per scenario so variable-N items kept as lists.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    lidar_bevs = torch.stack([item["lidar_bev"] for item in batch])
    map_bevs   = torch.stack([item["map_bev"]   for item in batch])

    return {
        "lidar_bev":      lidar_bevs,
        "map_bev":        map_bevs,
        "agent_history":  [item["agent_history"]  for item in batch],
        "gt_traj_focal":  [item["gt_traj_focal"]  for item in batch],
        "gt_mask_focal":  [item["gt_mask_focal"]  for item in batch],
        "gt_traj_all":    [item["gt_traj_all"]    for item in batch],
        "gt_mask_all":    [item["gt_mask_all"]    for item in batch],
        "focal_idx":      [item["focal_idx"]      for item in batch],
        "focal_track_id": [item["focal_track_id"] for item in batch],
        "gt_boxes":       [item["gt_boxes"]       for item in batch],
        "gt_intentions":  [item["gt_intentions"]  for item in batch],
        "track_ids":      [item["track_ids"]      for item in batch],
        "scenario_id":    [item["scenario_id"]    for item in batch],
        "current_ts_ns":  [item["current_ts_ns"]  for item in batch],
    }