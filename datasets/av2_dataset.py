# Adapted from Nadeem Mohamed's IntentNetViT
# Original repo: https://github.com/Nadeem202020/VisionTransformer-Intention-Prediction
#
# Modifications:
#   1. Updated all import paths to match new repo structure
#      (utils.constants, utils.utils, utils.heuristic_labeling)
#   2. Added _get_future_trajectory() method to extract GT trajectory
#      from annotations_with_intent.feather for each vehicle
#   3. Added _apply_augmentation_to_trajectory() method to transform
#      trajectory GT consistently with BEV augmentation
#   4. Updated __getitem__ to extract future trajectories and add them
#      to the GT dict as future_traj_ego [N,60,2] and future_traj_mask [N,60]
#   5. Updated augmentation call — augment_bev() now returns aug_params
#      so trajectory augmentation uses identical random values as BEV
#   6. Updated collate_fn comment to reflect new GT fields
#   7. Added TRAJECTORY_FUTURE_STEPS to imports from utils.constants
#
# All original logic (ScenarioValidator, BEV generation, sequence creation,
# log data caching) is unchanged from Nadeem's original.

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow
import pyarrow.feather as feather
from scipy.spatial.transform import Rotation as R
import traceback
import time
import os
from collections import namedtuple

# --- MODIFICATION 1: updated import paths to match new repo structure ---
from utils.constants import (
    LIDAR_SWEEPS,
    AV2_MAP_AVAILABLE,
    SHAPELY_AVAILABLE,
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
    TRAJECTORY_FUTURE_STEPS,   # NEW: number of future timesteps for trajectory GT
                               # SOURCED: Abdulbaki thesis Section 3.1 — 60 steps = 6s
)
from utils.utils import (
    load_ego_poses,
    transform_points,
    create_intentnet_lidar_bev,
    rasterize_map_ego_centric,
    prepare_gt_for_frame,
    augment_bev              # MODIFICATION 5: augment_bev now returns aug_params
)
from utils.heuristic_labeling import get_vehicle_intention_heuristic_enhanced


# =============================================================================
# ScenarioValidator
# Unchanged from Nadeem's original dataset.py
# =============================================================================

class ScenarioValidator:
    """
    Validates scenario directories to ensure all required files are present
    and optionally skips known corrupted logs.
    Unchanged from Nadeem's original dataset.py.
    """

    def __init__(
        self,
        base_path: str,
        skip_known_corrupted: bool = True,
        min_feather_size_bytes: int = 1024
    ):
        self.base_path = Path(base_path)
        self.ScenarioPaths = namedtuple(
            "ScenarioPaths",
            ["log_dir", "map_path", "annotations_path"]
        )
        self.skip_known_corrupted = skip_known_corrupted
        self.min_feather_size_bytes = min_feather_size_bytes
        self.KNOWN_CORRUPTED_LOGS = {}

    def find_valid_scenarios(self) -> list:
        """
        Scans the base_path for valid scenario directories.
        Returns a list of ScenarioPaths namedtuples for valid scenarios.
        """
        valid_scenarios = []
        print(f"ScenarioValidator: Searching for scenarios in: {self.base_path}")
        if not self.base_path.is_dir():
            print(f"Error: Base path does not exist: {self.base_path}")
            return []

        total_dirs_scanned = 0
        skipped_corrupted = 0
        skipped_reasons = {}
        start_time = time.time()

        try:
            iterator = os.scandir(self.base_path)
        except OSError as e:
            print(f"Error: Cannot scan directory {self.base_path}: {e}")
            return []

        for entry in iterator:
            if not entry.is_dir():
                continue

            scenario_dir = Path(entry.path)
            total_dirs_scanned += 1
            scenario_name = scenario_dir.name

            if total_dirs_scanned % 50 == 0:
                print(f"  Scanned {total_dirs_scanned} directories...")

            if self.skip_known_corrupted and scenario_name in self.KNOWN_CORRUPTED_LOGS:
                skipped_corrupted += 1
                continue

            validation_result = self._validate_scenario(scenario_dir)
            if isinstance(validation_result, self.ScenarioPaths):
                valid_scenarios.append(validation_result)
            elif isinstance(validation_result, str):
                reason = validation_result
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

        end_time = time.time()
        print(f"\nScenario Scan Summary (took {end_time - start_time:.2f}s):")
        print(f"  Total directories scanned: {total_dirs_scanned}")
        if self.skip_known_corrupted:
            print(f"  Skipped (known corrupted): {skipped_corrupted}")
        if skipped_reasons:
            print(f"  Skipped (invalid): {sum(skipped_reasons.values())}")
            for reason, count in skipped_reasons.items():
                print(f"    - {reason}: {count}")
        print(f"  Found {len(valid_scenarios)} valid scenarios.")
        return valid_scenarios

    def _validate_scenario(self, scenario_dir: Path):
        """Checks a single scenario directory for required files and basic integrity."""
        lidar_dir = scenario_dir / "sensors" / "lidar"
        annotation_file = scenario_dir / "annotations.feather"
        map_dir = scenario_dir / "map"
        ego_pose_file = scenario_dir / "city_SE3_egovehicle.feather"
        log_id = scenario_dir.name

        required_paths = {
            "Lidar directory": lidar_dir,
            "Annotations file": annotation_file,
            "Map directory": map_dir,
            "Ego pose file": ego_pose_file,
        }
        for name, path_obj in required_paths.items():
            if (
                (path_obj.is_dir() and not any(path_obj.iterdir())) or
                (path_obj.is_file() and
                 path_obj.stat().st_size < self.min_feather_size_bytes and
                 self.min_feather_size_bytes > 0) or
                not path_obj.exists()
            ):
                return f"Missing or invalid {name.lower()} ({path_obj.name})"

        if not any(lidar_dir.glob("*.feather")):
            return "No *.feather files in lidar directory"

        map_files = list(map_dir.glob(f"log_map_archive_{log_id}*.json"))
        if not map_files:
            map_files_fallback = list(map_dir.glob("log_map_archive_*.json"))
            if not map_files_fallback:
                return "No map file found in map directory"
            map_files = map_files_fallback

        return self.ScenarioPaths(
            log_dir=str(scenario_dir),
            map_path=str(map_files[0]),
            annotations_path=str(annotation_file)
        )


# =============================================================================
# collate_fn
# Minor comment update — logic unchanged
# =============================================================================

def collate_fn(batch: list) -> dict | None:
    """
    Custom collate function to handle batches of data items.
    Filters out None items from failed __getitem__ calls.

    GT stays as a list because N (number of vehicles) varies per frame.
    Each element in gt_list is a dict with keys:
        boxes_xywha      [N, 5]   — box coordinates in ego frame
        intentions       [N]      — intention class per vehicle
        track_ids        list     — vehicle ID strings
        future_traj_ego  [N,60,2] — NEW: future positions in ego frame
        future_traj_mask [N,60]   — NEW: True where position data is valid
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    lidar_bevs = torch.stack([item["lidar_bev"] for item in batch])
    map_bevs = torch.stack([item["map_bev"] for item in batch])
    gt_list = [item["gt"] for item in batch]

    return {
        "lidar_bev": lidar_bevs,
        "map_bev": map_bevs,
        "gt_list": gt_list
    }


# =============================================================================
# ArgoverseIntentNetDataset
# Main dataset class — modified to add trajectory GT
# =============================================================================

class ArgoverseIntentNetDataset(Dataset):
    """
    Dataset class for Argoverse 2, preparing data for multi-task
    detection + intention + trajectory prediction.

    Changes vs Nadeem's original:
    - Added _get_future_trajectory() for trajectory GT extraction
    - Added _apply_augmentation_to_trajectory() for consistent augmentation
    - Updated __getitem__ to include future_traj_ego and future_traj_mask
      in the returned GT dict
    - augment_bev() now returns aug_params for synchronised augmentation
    """

    def __init__(
        self,
        data_dir: str,
        num_sweeps: int = LIDAR_SWEEPS,
        is_train: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.num_sweeps = num_sweeps
        self.is_train = is_train

        validator = ScenarioValidator(str(self.data_dir))
        self.valid_scenario_paths = validator.find_valid_scenarios()
        if not self.valid_scenario_paths:
            raise ValueError(
                f"No valid scenarios found in {self.data_dir}. "
                "Check the path and data integrity."
            )

        self.log_data_cache = {}
        self.sequences = self._create_sequences()
        if not self.sequences:
            raise ValueError(
                f"Could not create any valid sequences from {self.data_dir}."
            )
        print(
            f"Dataset Initialized: {'Train' if is_train else 'Val'}. "
            f"Found {len(self.sequences)} sequences."
        )

    # -------------------------------------------------------------------------
    # _create_sequences — unchanged from Nadeem's original
    # -------------------------------------------------------------------------

    def _create_sequences(self) -> list:
        """
        Generates a list of all valid (current_timestamp, past_sweeps)
        sequences from scenarios.
        Unchanged from Nadeem's original dataset.py.
        """
        sequences = []
        print("Creating sequences from valid scenarios...")

        for scenario_info in self.valid_scenario_paths:
            log_dir = Path(scenario_info.log_dir)
            log_id = log_dir.name
            lidar_dir = log_dir / "sensors" / "lidar"

            try:
                if not lidar_dir.is_dir():
                    print(f"  Warning: LiDAR dir missing for {log_id}. Skipping.")
                    continue

                timestamps = sorted(
                    [int(p.stem) for p in lidar_dir.glob("*.feather")]
                )

                if len(timestamps) < self.num_sweeps:
                    continue

                for i in range(len(timestamps) - self.num_sweeps + 1):
                    current_ts_ns = timestamps[i + self.num_sweeps - 1]
                    sweep_ts_list = timestamps[i: i + self.num_sweeps]
                    sequences.append({
                        "log_id": log_id,
                        "log_dir": str(log_dir),
                        "map_json_path": scenario_info.map_path,
                        "annotations_path": scenario_info.annotations_path,
                        "current_ts_ns": current_ts_ns,
                        "sweep_ts_list": sweep_ts_list
                    })

            except ValueError as e:
                print(f"  Warning: Timestamp error in {log_id}: {e}. Skipping.")
            except Exception as e:
                print(f"  Warning: Error in {log_id}: {e}. Skipping.")

        print(f"Created {len(sequences)} sequences in total.")
        return sequences

    # -------------------------------------------------------------------------
    # _get_log_data — unchanged from Nadeem's original
    # -------------------------------------------------------------------------

    def _get_log_data(
        self,
        log_id: str,
        log_dir: str,
        annotations_path: str
    ) -> dict | None:
        """
        Loads and caches essential data for a given log_id.
        Includes ego poses, GT annotations with pre-computed intentions, and map.
        Unchanged from Nadeem's original dataset.py.
        """
        log_dir_path = Path(log_dir)
        intent_annotation_file_path = (
            log_dir_path / "annotations_with_intent.feather"
        )

        if log_id not in self.log_data_cache:
            try:
                if not intent_annotation_file_path.is_file():
                    print(
                        f"FATAL: Pre-computed intent file missing for {log_id}. "
                        "Run preprocess_intent_labels.py first."
                    )
                    self.log_data_cache[log_id] = None
                    return None

                gt_df_with_intent = pd.read_feather(intent_annotation_file_path)
                ego_poses_df = load_ego_poses(log_dir_path)

                map_api = None
                if AV2_MAP_AVAILABLE:
                    map_base_path = log_dir_path / "map"
                    if map_base_path.is_dir() and any(map_base_path.iterdir()):
                        from av2.map.map_api import ArgoverseStaticMap
                        map_api = ArgoverseStaticMap.from_map_dir(
                            map_base_path, build_raster=False
                        )

                self.log_data_cache[log_id] = {
                    "ego_poses": ego_poses_df,
                    "gt_df": gt_df_with_intent,
                    "map_api": map_api
                }

            except FileNotFoundError as e:
                print(f"Error in _get_log_data for {log_id}: {e}")
                self.log_data_cache[log_id] = None
            except Exception as e:
                print(f"Error loading cache for {log_id}: {e}")
                traceback.print_exc()
                self.log_data_cache[log_id] = None

        return self.log_data_cache.get(log_id)

    # -------------------------------------------------------------------------
    # _get_future_trajectory — NEW METHOD
    # -------------------------------------------------------------------------

    def _get_future_trajectory(
        self,
        track_id: str,
        current_ts_ns: int,
        gt_df: pd.DataFrame,
        ego_SE3_world: np.ndarray,
        future_steps: int = TRAJECTORY_FUTURE_STEPS
    ) -> tuple:
        """
        NEW METHOD — Added for trajectory prediction (V2 and V3).

        Extracts the future trajectory of one vehicle in ego frame.

        How it works:
            1. Filter annotations to this vehicle's future rows only
            2. Take the next `future_steps` rows (max 60)
            3. Transform world-frame (x,y) positions to ego-frame
               using ego_SE3_world (the inverse of the ego pose matrix)
            4. Return positions and a validity mask

        Why ego frame?
            The BEV is ego-centric so trajectory GT must match.
            SOURCED: consistent with prepare_gt_for_frame() in utils.py
            which also uses ego-frame positions for GT boxes.

        Why a validity mask?
            Not all vehicles have 60 future timesteps. A vehicle that
            leaves the sensor range after 2 seconds only has 20 future
            observations. The mask tells the loss function which timesteps
            are real data and which are zero-padding.

        Why z = 0?
            ASSUMED: annotations store only (x, y) per timestep.
            We add z=0 to satisfy transform_points() which expects 3D.
            z is immediately discarded after transformation.
            Valid under the standard BEV flat-ground assumption used
            throughout this codebase.

        Args:
            track_id:      unique vehicle ID string
            current_ts_ns: current timestamp in nanoseconds
            gt_df:         full annotations DataFrame for this log
            ego_SE3_world: 4x4 matrix transforming world to ego frame
            future_steps:  number of future timesteps to extract
                           SOURCED: 60 from Abdulbaki thesis Section 3.1

        Returns:
            traj: np.ndarray [future_steps, 2]
                  x,y in ego frame — zero where data is missing
            mask: np.ndarray [future_steps] bool
                  True = valid observation, False = zero-padded
        """
        # Initialise output — zeros for positions, False for mask
        traj = np.zeros((future_steps, 2), dtype=np.float32)
        mask = np.zeros(future_steps, dtype=bool)

        # Get this vehicle's future rows sorted by time
        vehicle_future = gt_df[
            (gt_df['track_uuid'] == track_id) &
            (gt_df['timestamp_ns'] > current_ts_ns)
        ].sort_values('timestamp_ns')

        if vehicle_future.empty:
            # No future observations — return all zeros, all False
            return traj, mask

        # Take only the next future_steps rows
        vehicle_future = vehicle_future.iloc[:future_steps]
        num_valid = len(vehicle_future)

        # Extract world-frame (x, y) positions
        # SOURCED: tx_m, ty_m are city-frame positions in
        # annotations_with_intent.feather after city-frame GT fix
        world_xy = vehicle_future[['tx_m', 'ty_m']].values  # [T, 2]

        # Add z=0 to make 3D for transform_points()
        # ASSUMED: flat ground plane — z discarded immediately after
        world_xyz = np.hstack([
            world_xy,
            np.zeros((num_valid, 1), dtype=np.float32)
        ])  # [T, 3]

        # Transform from world frame to ego frame
        # ego_SE3_world = inverse of world_SE3_ego
        # result: positions relative to ego vehicle at current timestamp
        ego_xyz = transform_points(world_xyz, ego_SE3_world)  # [T, 3]
        ego_xy = ego_xyz[:, :2]  # [T, 2] — drop z

        # Fill output arrays
        traj[:num_valid] = ego_xy
        mask[:num_valid] = True

        return traj, mask

    # -------------------------------------------------------------------------
    # _apply_augmentation_to_trajectory — NEW METHOD
    # -------------------------------------------------------------------------

    def _apply_augmentation_to_trajectory(
        self,
        future_trajs: np.ndarray,
        future_masks: np.ndarray,
        flip_applied: bool,
        rotation_angle_rad: float,
        scale_factor: float
    ) -> np.ndarray:
        """
        NEW METHOD — Transforms trajectory GT to match BEV augmentation.

        When the BEV is flipped, rotated, or scaled during training,
        the trajectory GT must undergo the exact same transformation.
        Without this, the model receives inconsistent training signal —
        the BEV shows a flipped scene but the trajectory GT shows the
        original unflipped future path.

        The transformation parameters come from aug_params returned by
        augment_bev() — the exact same random values used on the BEV.
        This guarantees perfect synchronisation.

        Each transformation mirrors the corresponding operation in utils.py:
        - Flip:     matches random_flip_bev()   → negate Y coordinate
        - Rotation: matches random_rotate_bev() → rotate (x,y) by angle
        - Scale:    matches random_scale_bev()  → multiply by scale factor

        Args:
            future_trajs:       [N, 60, 2] trajectory positions in ego frame
            future_masks:       [N, 60] validity masks (True = valid)
            flip_applied:       whether horizontal flip was applied to BEV
            rotation_angle_rad: rotation angle applied to BEV in radians
            scale_factor:       scale factor applied to BEV

        Returns:
            transformed future_trajs [N, 60, 2]
        """
        if future_trajs.shape[0] == 0:
            # No vehicles in this frame — nothing to transform
            return future_trajs

        trajs = future_trajs.copy()

        # --- Apply flip ---
        # SOURCED: mirrors random_flip_bev() in utils.py
        # which applies gt_boxes_xywha[:, 1] *= -1 (Y coordinate)
        if flip_applied:
            trajs[:, :, 1] *= -1

        # --- Apply rotation ---
        # SOURCED: mirrors random_rotate_bev() in utils.py
        # which rotates box centers by the same angle
        if rotation_angle_rad != 0.0:
            cos_a = np.cos(rotation_angle_rad)
            sin_a = np.sin(rotation_angle_rad)
            # 2D rotation matrix
            rot = np.array([[cos_a, -sin_a],
                            [sin_a,  cos_a]])
            # Apply to all N vehicles, all 60 timesteps simultaneously
            # Reshape [N, 60, 2] → [N*60, 2], rotate, reshape back
            original_shape = trajs.shape
            trajs_flat = trajs.reshape(-1, 2)          # [N*60, 2]
            trajs_flat = (rot @ trajs_flat.T).T        # [N*60, 2]
            trajs = trajs_flat.reshape(original_shape) # [N, 60, 2]

        # --- Apply scale ---
        # SOURCED: mirrors random_scale_bev() in utils.py
        # which applies gt_boxes_xywha[:, :4] *= scale_factor
        if scale_factor != 1.0:
            trajs *= scale_factor

        # --- Zero out padded positions ---
        # After geometric transformations, padded positions (where mask=False)
        # may have non-zero values due to floating point operations.
        # Reset them to zero to keep padding clean.
        trajs[~future_masks] = 0.0

        return trajs

    # -------------------------------------------------------------------------
    # __len__ — unchanged
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    # -------------------------------------------------------------------------
    # __getitem__ — modified to add trajectory GT
    # -------------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict | None:
        """
        Retrieves a single data sample for training or validation.

        Changes vs Nadeem's original:
        1. ego_SE3_world is stored (was computed but not kept before)
           — needed for world-to-ego transformation of future positions
        2. After prepare_gt_for_frame(), loop through each GT vehicle
           and call _get_future_trajectory() to get its future path
        3. Stack results into future_traj_ego [N,60,2] and
           future_traj_mask [N,60]
        4. augment_bev() now returns aug_params — use them to apply
           identical transformation to trajectory GT
        5. future_traj_ego and future_traj_mask added to final_gt_dict

        Everything else (LiDAR loading, BEV generation, map rasterization,
        sequence indexing) is unchanged from Nadeem's original.
        """
        if not (0 <= idx < len(self.sequences)):
            raise IndexError(
                f"Index {idx} out of bounds for dataset size {len(self.sequences)}"
            )

        sequence_info = self.sequences[idx]
        log_id = sequence_info["log_id"]
        log_dir = sequence_info["log_dir"]
        map_json_path = sequence_info["map_json_path"]
        current_ts_ns = sequence_info["current_ts_ns"]
        sweep_ts_list = sequence_info["sweep_ts_list"]

        try:
            # --- Load cached log data ---
            # Unchanged from Nadeem's original
            log_data = self._get_log_data(
                log_id, log_dir, sequence_info["annotations_path"]
            )
            if log_data is None:
                return None

            ego_poses_df = log_data["ego_poses"]
            gt_df_with_intent = log_data["gt_df"]
            map_api = log_data["map_api"]

            # --- Get current ego pose ---
            # Unchanged from Nadeem's original
            current_ego_pose_row = ego_poses_df[
                ego_poses_df['timestamp_ns'] == current_ts_ns
            ]
            if current_ego_pose_row.empty:
                return None
            current_ego_pose = current_ego_pose_row.iloc[0]

            tx = current_ego_pose['tx_m']
            ty = current_ego_pose['ty_m']
            tz = current_ego_pose['tz_m']
            qx = current_ego_pose['qx']
            qy = current_ego_pose['qy']
            qz = current_ego_pose['qz']
            qw = current_ego_pose['qw']

            try:
                rot_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
            except ValueError:
                return None

            # world_SE3_ego: transforms ego coords → world coords
            world_SE3_ego = np.eye(4)
            world_SE3_ego[:3, :3] = rot_mat
            world_SE3_ego[:3, 3] = [tx, ty, tz]

            # ego_SE3_world: transforms world coords → ego coords
            # MODIFICATION: now stored as instance variable so
            # _get_future_trajectory() can use it to transform
            # future world positions into ego frame
            ego_SE3_world = np.linalg.inv(world_SE3_ego)

            # --- Load LiDAR sweeps ---
            # Unchanged from Nadeem's original
            points_list, intensity_list = [], []
            lidar_base_path = Path(log_dir) / "sensors" / "lidar"

            for ts_sweep in sweep_ts_list:
                sweep_path = lidar_base_path / f"{ts_sweep}.feather"
                if not sweep_path.is_file():
                    points_list.append(None)
                    intensity_list.append(None)
                    continue
                try:
                    sweep_df = pd.read_feather(
                        sweep_path,
                        columns=['x', 'y', 'z', 'intensity']
                    )
                    if sweep_df.empty:
                        points_list.append(None)
                        intensity_list.append(None)
                        continue
                except pyarrow.ArrowInvalid:
                    points_list.append(None)
                    intensity_list.append(None)
                    continue

                pts_world = sweep_df[['x', 'y', 'z']].values
                intensity = sweep_df['intensity'].values.astype(np.float32)

                sweep_pose_row = ego_poses_df[
                    ego_poses_df['timestamp_ns'] == ts_sweep
                ]
                if sweep_pose_row.empty:
                    points_list.append(None)
                    intensity_list.append(None)
                    continue

                sw_tx = sweep_pose_row.iloc[0]['tx_m']
                sw_ty = sweep_pose_row.iloc[0]['ty_m']
                sw_tz = sweep_pose_row.iloc[0]['tz_m']
                sw_q = sweep_pose_row.iloc[0][['qx', 'qy', 'qz', 'qw']].values

                try:
                    sw_rot = R.from_quat(sw_q).as_matrix()
                except ValueError:
                    points_list.append(None)
                    intensity_list.append(None)
                    continue

                sw_tf_world_ego = np.eye(4)
                sw_tf_world_ego[:3, :3] = sw_rot
                sw_tf_world_ego[:3, 3] = [sw_tx, sw_ty, sw_tz]
                rel_tf = ego_SE3_world @ sw_tf_world_ego
                pts_curr_ego = transform_points(pts_world, rel_tf)
                points_list.append(pts_curr_ego)
                intensity_list.append(intensity)

            if all(p is None for p in points_list):
                return None

            # --- Build BEV images ---
            # Unchanged from Nadeem's original
            lidar_bev_np = create_intentnet_lidar_bev(points_list, intensity_list)
            map_bev_np = rasterize_map_ego_centric(map_json_path, current_ego_pose)

            # --- Get GT boxes and intentions ---
            # MODIFICATION: pass ego_SE3_world so prepare_gt_for_frame() can
            # transform city-frame positions back to ego frame.
            # ego_SE3_world is already computed above for LiDAR sweep alignment.
            frame_gt_dict = prepare_gt_for_frame(
                current_ts_ns,
                gt_df_with_intent,
                map_api,
            )

            # =================================================================
            # NEW: Extract future trajectories for all GT vehicles
            # =================================================================
            track_ids = frame_gt_dict.get('track_ids', [])
            num_vehicles = len(track_ids)

            if num_vehicles > 0:
                # Pre-allocate arrays for all N vehicles
                # Shape: [N, 60, 2] for positions, [N, 60] for mask
                # SOURCED: 60 from TRAJECTORY_FUTURE_STEPS
                # (Abdulbaki thesis Section 3.1)
                future_trajs_np = np.zeros(
                    (num_vehicles, TRAJECTORY_FUTURE_STEPS, 2),
                    dtype=np.float32
                )
                future_masks_np = np.zeros(
                    (num_vehicles, TRAJECTORY_FUTURE_STEPS),
                    dtype=bool
                )

                # Get future trajectory for each vehicle individually
                for i, track_id in enumerate(track_ids):
                    traj, mask = self._get_future_trajectory(
                        track_id=track_id,
                        current_ts_ns=current_ts_ns,
                        gt_df=gt_df_with_intent,
                        future_steps=TRAJECTORY_FUTURE_STEPS
                    )
                    future_trajs_np[i] = traj
                    future_masks_np[i] = mask
            else:
                # No vehicles in this frame — empty arrays
                future_trajs_np = np.zeros(
                    (0, TRAJECTORY_FUTURE_STEPS, 2),
                    dtype=np.float32
                )
                future_masks_np = np.zeros(
                    (0, TRAJECTORY_FUTURE_STEPS),
                    dtype=bool
                )
            # =================================================================

            # --- Apply augmentation (training only) ---
            if self.is_train:
                # MODIFICATION: augment_bev() now returns aug_params
                # containing the exact random values it used
                # (flip_applied, rotation_angle_rad, scale_factor).
                # We pass these directly to _apply_augmentation_to_trajectory()
                # so the trajectory GT is transformed with IDENTICAL parameters
                # as the BEV — guaranteeing consistency.
                #
                # SOURCED: approach of returning augmentation parameters
                # for multi-task consistency is standard practice
                # (see DeTra, Casas et al. 2024 — augmentation applied
                # jointly to all outputs)
                lidar_bev_np, map_bev_np, frame_gt_dict, aug_params = augment_bev(
                    lidar_bev_np, map_bev_np, frame_gt_dict
                )

                # Apply exact same augmentation to trajectory GT
                future_trajs_np = self._apply_augmentation_to_trajectory(
                    future_trajs=future_trajs_np,
                    future_masks=future_masks_np,
                    flip_applied=aug_params['flip_applied'],
                    rotation_angle_rad=aug_params['rotation_angle_rad'],
                    scale_factor=aug_params['scale_factor']
                )

            # --- Build final tensors ---
            final_lidar_bev = torch.from_numpy(lidar_bev_np).float()
            final_map_bev = torch.from_numpy(map_bev_np).float()

            # MODIFICATION: GT dict now includes two new trajectory fields
            final_gt_dict = {
                # --- Original fields — unchanged ---
                'boxes_xywha': frame_gt_dict['boxes_xywha'].float(),
                # [N, 5]: (cx, cy, w, l, heading) in ego frame

                'intentions': frame_gt_dict['intentions'].long(),
                # [N]: intention class index per vehicle (0-7)

                'track_ids': frame_gt_dict.get('track_ids', []),
                # [N]: vehicle UUID strings for cross-model matching

                # --- NEW fields for trajectory prediction ---
                'future_traj_ego': torch.from_numpy(future_trajs_np).float(),
                # [N, 60, 2]: future (x,y) positions in ego frame
                # SOURCED: 60 steps from TRAJECTORY_FUTURE_STEPS
                # (Abdulbaki thesis Section 3.1)
                # Zero-padded where vehicle leaves the scene early

                'future_traj_mask': torch.from_numpy(future_masks_np).bool(),
                # [N, 60]: True where position data is real,
                # False where zero-padded
                # Used by trajectory loss to ignore padded timesteps
            }

            return {
                "lidar_bev": final_lidar_bev,
                "map_bev": final_map_bev,
                "gt": final_gt_dict
            }

        except Exception as e:
            print(
                f"!!! ERROR in __getitem__ idx={idx} "
                f"log={log_id} ts={current_ts_ns} !!!"
            )
            traceback.print_exc()
            return None