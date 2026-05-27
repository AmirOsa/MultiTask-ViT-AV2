#USE THIS ONE
#Same as V1 but with  heading change to city frame and new focal agent selection
#Might need future changes like  5-second heading change

import numpy as np
import shutil
import glob
from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

SENSOR_DIR   = Path("/content/drive/MyDrive/Bachelor Thesis/MultiTask_V2/av2/sensor/val")
OUTPUT_BASE = Path("/content/local_mf/val")

HIST_STEPS   = 50
FUTURE_STEPS = 60
WINDOW_SIZE  = HIST_STEPS + FUTURE_STEPS
STEP_SIZE    = 25
MIN_OBSERVED_STEPS = 5

MF_MAP = {
    "REGULAR_VEHICLE": "vehicle",
    "LARGE_VEHICLE": "vehicle",
    "BOX_TRUCK": "vehicle",
    "TRUCK": "vehicle",
    "TRUCK_CAB": "vehicle",
    "VEHICULAR_TRAILER": "vehicle",
    "BUS": "bus",
    "SCHOOL_BUS": "bus",
    "ARTICULATED_BUS": "bus",
    "PEDESTRIAN": "pedestrian",
    "DOG": "pedestrian",
    "ANIMAL": "pedestrian",
    "OFFICIAL_SIGNALER": "pedestrian",
    "BICYCLIST": "cyclist",
    "BICYCLE": "cyclist",
    "WHEELED_RIDER": "cyclist",
    "MOTORCYCLE": "motorcyclist",
    "MOTORCYCLIST": "motorcyclist",
    "BOLLARD": "static",
    "CONSTRUCTION_CONE": "static",
    "CONSTRUCTION_BARREL": "static",
    "STOP_SIGN": "static",
    "MOBILE_PEDESTRIAN_SIGN": "static",
    "SIGN": "static",
    "WHEELED_DEVICE": "static",
    "STROLLER": "static",
    "WHEELCHAIR": "static",
    "MESSAGE_BOARD_TRAILER": "static",
    "RAILED_VEHICLE": "static",
    "TRAFFIC_LIGHT_TRAILER": "static",
}

def quaternion_to_yaw(qw, qx, qy, qz):
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy**2 + qz**2)))

def neighbor_count(row, frame, radius=10.0):
    dx = frame["tx_m"] - row["tx_m"]
    dy = frame["ty_m"] - row["ty_m"]
    return np.sum(np.sqrt(dx*dx + dy*dy) < radius)

def process_log(LOG_DIR):

    OUTPUT_DIR = OUTPUT_BASE

    ann_path = LOG_DIR / "annotations.feather"
    ego_path = LOG_DIR / "city_SE3_egovehicle.feather"

    if not ann_path.exists() or not ego_path.exists():
        print(f"  ⚠️  Skipping {LOG_DIR.name} — missing files")
        return

    map_dir = LOG_DIR / "map"
    map_json_matches = list(map_dir.glob("log_map_archive_*.json")) if map_dir.exists() else []

    if not map_json_matches:
        print(f"  ⚠️  Skipping {LOG_DIR.name} — missing log_map_archive JSON in map/")
        return

    map_json_path = map_json_matches[0]

    # ── Extract city name from map JSON filename ──────────────
    map_stem    = map_json_path.stem
    city_abbrev = map_stem.split('____')[1].split('_city_')[0].upper()
    city_map = {
        'ATX': 'austin',
        'DTW': 'detroit',
        'MIA': 'miami',
        'PAO': 'palo-alto',
        'PIT': 'pittsburgh',
        'WDC': 'washington-dc'
    }
    city_name = city_map.get(city_abbrev, city_abbrev.lower())
    print(f"  City: {city_name}")

    # ── STEP 1: Load annotations and ego poses ────────────────
    df  = pd.read_feather(ann_path)
    ego = pd.read_feather(ego_path).sort_values('timestamp_ns').reset_index(drop=True)

    df = df[df['category'].isin(MF_MAP.keys())].copy()
    df = df.sort_values('timestamp_ns').reset_index(drop=True)

    # ── Build ego pose lookup ─────────────────────────────────
    ego_lookup = ego.set_index('timestamp_ns')

    # ── STEP 2: Build sorted timestamp list ───────────────────
    all_timestamps = sorted(df['timestamp_ns'].unique())
    n_ts           = len(all_timestamps)

    if n_ts < WINDOW_SIZE:
        print(f"  ⚠️  Skipping {LOG_DIR.name} — too short ({n_ts} timesteps)")
        return

    # ── STEP 3: Compute sliding window start indices ──────────
    window_starts = list(range(0, n_ts - WINDOW_SIZE + 1, STEP_SIZE))
    n_windows     = len(window_starts)

    # ── STEP 4: Pre-compute per-agent trajectory lookup ───────
    agent_dfs = {}
    for uuid, group in df.groupby('track_uuid'):
        agent_dfs[uuid] = group.set_index('timestamp_ns').sort_index()

    # ── STEP 5: Process each window ───────────────────────────
    for w_idx, start_idx in enumerate(window_starts):

        end_idx   = start_idx + WINDOW_SIZE
        window_ts = all_timestamps[start_idx:end_idx]
        ts_to_idx = {ts: i for i, ts in enumerate(window_ts)}

        hist_ts     = set(window_ts[:HIST_STEPS])
        start_ts_ns = window_ts[0]
        end_ts_ns   = window_ts[-1]
        current_ts  = window_ts[HIST_STEPS - 1]

        # ── Find ego position at current timestamp ────────────
        ego_at_current = ego[ego['timestamp_ns'] == current_ts]
        if ego_at_current.empty:
            closest_idx = (ego['timestamp_ns'] - current_ts).abs().argsort().iloc[0]
            ego_x = ego.iloc[closest_idx]['tx_m']
            ego_y = ego.iloc[closest_idx]['ty_m']
        else:
            ego_x = ego_at_current.iloc[0]['tx_m']
            ego_y = ego_at_current.iloc[0]['ty_m']

        # ── Find focal agent ──────────────────────────────────
        current_frame  = df[df['timestamp_ns'] == current_ts].copy()
        if current_frame.empty:
            continue

        future_ts_list = window_ts[HIST_STEPS:]

        # ── Compute future displacement in city frame ─────────
        future_disp_map = {}
        for uuid, agent_df in agent_dfs.items():
            future_positions_city = []
            for ts in future_ts_list:
                if ts in agent_df.index:
                    row = agent_df.loc[ts]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]

                    if ts in ego_lookup.index:
                        ego_row = ego_lookup.loc[ts]
                        if isinstance(ego_row, pd.DataFrame):
                            ego_row = ego_row.iloc[0]
                        ego_tx  = float(ego_row['tx_m'])
                        ego_ty  = float(ego_row['ty_m'])
                        ego_yaw = quaternion_to_yaw(
                            float(ego_row['qw']), float(ego_row['qx']),
                            float(ego_row['qy']), float(ego_row['qz'])
                        )
                        cos_yaw     = np.cos(ego_yaw)
                        sin_yaw     = np.sin(ego_yaw)
                        agent_ego_x = float(row['tx_m'])
                        agent_ego_y = float(row['ty_m'])
                        pos_x = ego_tx + cos_yaw * agent_ego_x - sin_yaw * agent_ego_y
                        pos_y = ego_ty + sin_yaw * agent_ego_x + cos_yaw * agent_ego_y
                        future_positions_city.append((pos_x, pos_y))

            if len(future_positions_city) < 2:
                future_disp_map[uuid] = 0.0
                continue

            total_disp = sum(
                np.sqrt(
                    (future_positions_city[i+1][0] - future_positions_city[i][0])**2 +
                    (future_positions_city[i+1][1] - future_positions_city[i][1])**2
                )
                for i in range(len(future_positions_city) - 1)
            )
            future_disp_map[uuid] = total_disp

        # ── FIX 2: Compute heading change in city frame ───────
        # Uses position-vector method on city-frame coords —
        # ego yaw is already baked in so the delta is correct.
        # Only first 30 future steps (3s) used for scoring.
        heading_change_map = {}
        for uuid, agent_df in agent_dfs.items():
            future_city_positions = []
            for ts in future_ts_list[:30]:
                if ts in agent_df.index:
                    row = agent_df.loc[ts]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    if ts in ego_lookup.index:
                        ego_row = ego_lookup.loc[ts]
                        if isinstance(ego_row, pd.DataFrame):
                            ego_row = ego_row.iloc[0]
                        ego_tx  = float(ego_row['tx_m'])
                        ego_ty  = float(ego_row['ty_m'])
                        ego_yaw = quaternion_to_yaw(
                            float(ego_row['qw']), float(ego_row['qx']),
                            float(ego_row['qy']), float(ego_row['qz'])
                        )
                        ax = float(row['tx_m'])
                        ay = float(row['ty_m'])
                        cx = ego_tx + np.cos(ego_yaw) * ax - np.sin(ego_yaw) * ay
                        cy = ego_ty + np.sin(ego_yaw) * ax + np.cos(ego_yaw) * ay
                        future_city_positions.append((cx, cy))

            if len(future_city_positions) >= 6:
                start_vec = (
                    np.array(future_city_positions[5]) -
                    np.array(future_city_positions[0])
                )
                end_vec = (
                    np.array(future_city_positions[-1]) -
                    np.array(future_city_positions[-6])
                )
                if np.linalg.norm(start_vec) > 0.1 and np.linalg.norm(end_vec) > 0.1:
                    h1 = np.arctan2(start_vec[1], start_vec[0])
                    h2 = np.arctan2(end_vec[1], end_vec[0])
                    dh = np.arctan2(np.sin(h2 - h1), np.cos(h2 - h1))
                    heading_change_map[uuid] = abs(np.degrees(dh))
                else:
                    heading_change_map[uuid] = 0.0
            else:
                heading_change_map[uuid] = 0.0

        # ── Interaction and proximity ─────────────────────────
        current_frame["dist_to_ego"] = np.sqrt(
            (current_frame["tx_m"] - ego_x)**2 +
            (current_frame["ty_m"] - ego_y)**2
        )
        current_frame["interaction"] = current_frame.apply(
            lambda r: neighbor_count(r, current_frame), axis=1
        )

        current_frame["future_disp"]    = current_frame["track_uuid"].map(future_disp_map).fillna(0.0)
        current_frame["heading_change"] = current_frame["track_uuid"].map(heading_change_map).fillna(0.0)

        # ── FIX 2: Updated scoring with heading change ────────
        current_frame["score"] = (
            0.4 * current_frame["future_disp"] +
            0.3 * current_frame["heading_change"] +
            0.15 * current_frame["interaction"] +
            0.15 * (1.0 / (1.0 + current_frame["dist_to_ego"]))
        )

        VEHICLE_CATEGORIES = {
            "REGULAR_VEHICLE", "LARGE_VEHICLE", "BOX_TRUCK", "TRUCK",
            "TRUCK_CAB", "VEHICULAR_TRAILER", "BUS", "SCHOOL_BUS", "ARTICULATED_BUS"
        }
        vehicle_frame = current_frame[current_frame['category'].isin(VEHICLE_CATEGORIES)]
        if vehicle_frame.empty:
            continue
        scoring_frame = vehicle_frame
        current_frame_sorted = scoring_frame.sort_values("score", ascending=False)

        def is_in_bev(uuid):
            """Check if agent is within BEV grid at current timestamp."""
            if uuid not in agent_dfs:
                return False
            agent_df = agent_dfs[uuid]
            if current_ts not in agent_df.index:
                return False
            row = agent_df.loc[current_ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            x = float(row['tx_m'])
            y = float(row['ty_m'])
            return (-20.0 <= x <= 60.0) and (-72.0 <= y <= 72.0) 

        focal_uuid = None
        for _, candidate_row in current_frame_sorted.iterrows():
            candidate_uuid = candidate_row["track_uuid"]
            if candidate_uuid in agent_dfs:
                candidate_df = agent_dfs[candidate_uuid]
                obs_count    = sum(ts in set(candidate_df.index) for ts in window_ts)
                if (obs_count >= MIN_OBSERVED_STEPS and
                        future_disp_map.get(candidate_uuid, 0.0) > 10.0 and
                        is_in_bev(candidate_uuid)):  # ADD THIS
                    focal_uuid = candidate_uuid
                    break

        # Fallback
        if focal_uuid is None:
            for _, candidate_row in current_frame_sorted.iterrows():
                candidate_uuid = candidate_row["track_uuid"]
                if candidate_uuid in agent_dfs:
                    candidate_df = agent_dfs[candidate_uuid]
                    obs_count    = sum(ts in set(candidate_df.index) for ts in window_ts)
                    if (obs_count >= MIN_OBSERVED_STEPS and
                            future_disp_map.get(candidate_uuid, 0.0) > 5.0 and
                            is_in_bev(candidate_uuid)):  # ADD THIS
                        focal_uuid = candidate_uuid
                        break

        if focal_uuid is None:
            continue  # no valid in-BEV focal agent for this window

        window_ts_set = set(window_ts)
        agent_window_counts = {
            uuid: sum(ts in window_ts_set for ts in agent_df.index)
            for uuid, agent_df in agent_dfs.items()
        }

        # ── Build rows for all agents in this window ──────────
        rows = []

        for uuid, agent_df in agent_dfs.items():

            agent_ts_set    = set(agent_df.index)
            agent_window_ts = [ts for ts in window_ts if ts in agent_ts_set]

            obs_count = sum(ts in agent_ts_set for ts in window_ts)
            if obs_count < MIN_OBSERVED_STEPS:
                continue

            category    = agent_df.iloc[0]['category']
            object_type = MF_MAP.get(category, None)
            if object_type is None:
                continue

            # ── Velocity computation ──────────────────────────
            agent_ts  = agent_window_ts
            positions = []
            for ts_i in agent_ts:
                row = agent_df.loc[ts_i]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                positions.append((float(row['tx_m']), float(row['ty_m'])))

            smoothed = []
            for i in range(len(positions)):
                if 0 < i < len(positions) - 1:
                    x = (positions[i-1][0] + positions[i][0] + positions[i+1][0]) / 3
                    y = (positions[i-1][1] + positions[i][1] + positions[i+1][1]) / 3
                else:
                    x, y = positions[i]
                smoothed.append((x, y))

            velocities = {}
            for i, ts_i in enumerate(agent_ts):
                x_i, y_i = smoothed[i]
                if 0 < i < len(agent_ts) - 1:
                    prev_t         = agent_ts[i-1]
                    next_t         = agent_ts[i+1]
                    prev_x, prev_y = smoothed[i-1]
                    next_x, next_y = smoothed[i+1]
                    dt    = (next_t - prev_t) * 1e-9
                    vel_x = (next_x - prev_x) / dt if dt > 0 else 0.0
                    vel_y = (next_y - prev_y) / dt if dt > 0 else 0.0
                elif i > 0:
                    prev_t         = agent_ts[i-1]
                    prev_x, prev_y = smoothed[i-1]
                    dt    = (ts_i - prev_t) * 1e-9
                    vel_x = (x_i - prev_x) / dt if dt > 0 else 0.0
                    vel_y = (y_i - prev_y) / dt if dt > 0 else 0.0
                elif i < len(agent_ts) - 1:
                    next_t         = agent_ts[i+1]
                    next_x, next_y = smoothed[i+1]
                    dt    = (next_t - ts_i) * 1e-9
                    vel_x = (next_x - x_i) / dt if dt > 0 else 0.0
                    vel_y = (next_y - y_i) / dt if dt > 0 else 0.0
                else:
                    vel_x, vel_y = 0.0, 0.0
                velocities[ts_i] = (vel_x, vel_y)

            avg_speed = np.mean([
                np.sqrt(velocities[ts][0]**2 + velocities[ts][1]**2)
                for ts in agent_window_ts if ts in velocities
            ]) if any(ts in velocities for ts in agent_window_ts) else 0.0

            if uuid == focal_uuid:
                object_category = 3
            elif agent_window_counts[uuid] == WINDOW_SIZE and avg_speed > 0.5:
                object_category = 1
            else:
                object_category = 0

            # ── Build one row per timestep ────────────────────
            for ts in window_ts:
                is_observed = ts in hist_ts

                if ts in agent_df.index:
                    row_data = agent_df.loc[ts]
                    if isinstance(row_data, pd.DataFrame):
                        row_data = row_data.iloc[0]

                    if ts in ego_lookup.index:
                        ego_row = ego_lookup.loc[ts]
                        if isinstance(ego_row, pd.DataFrame):
                            ego_row = ego_row.iloc[0]

                        ego_tx  = float(ego_row['tx_m'])
                        ego_ty  = float(ego_row['ty_m'])
                        ego_yaw = quaternion_to_yaw(
                            float(ego_row['qw']),
                            float(ego_row['qx']),
                            float(ego_row['qy']),
                            float(ego_row['qz'])
                        )

                        agent_ego_x = float(row_data['tx_m'])
                        agent_ego_y = float(row_data['ty_m'])

                        cos_yaw = np.cos(ego_yaw)
                        sin_yaw = np.sin(ego_yaw)

                        pos_x = ego_tx + cos_yaw * agent_ego_x - sin_yaw * agent_ego_y
                        pos_y = ego_ty + sin_yaw * agent_ego_x + cos_yaw * agent_ego_y

                        vel_x, vel_y = velocities.get(ts, (0.0, 0.0))
                        vel_x_city = cos_yaw * vel_x - sin_yaw * vel_y
                        vel_y_city = sin_yaw * vel_x + cos_yaw * vel_y
                        vel_x, vel_y = vel_x_city, vel_y_city

                        agent_heading_ego = quaternion_to_yaw(
                            float(row_data['qw']),
                            float(row_data['qx']),
                            float(row_data['qy']),
                            float(row_data['qz'])
                        )
                        heading = agent_heading_ego + ego_yaw

                    else:
                        pos_x   = float(row_data['tx_m'])
                        pos_y   = float(row_data['ty_m'])
                        vel_x, vel_y = velocities.get(ts, (0.0, 0.0))
                        heading = quaternion_to_yaw(
                            float(row_data['qw']),
                            float(row_data['qx']),
                            float(row_data['qy']),
                            float(row_data['qz'])
                        )

                    timestep_0indexed = ts_to_idx[ts]

                    rows.append({
                        'observed':        bool(is_observed),
                        'track_id':        uuid,
                        'object_type':     object_type,
                        'object_category': object_category,
                        'timestep':        timestep_0indexed,
                        'position_x':      pos_x,
                        'position_y':      pos_y,
                        'heading':         heading,
                        'velocity_x':      vel_x,
                        'velocity_y':      vel_y,
                        'timestamp_ns':    int(ts),        # FIX 1: added
                        'scenario_id':     f"{LOG_DIR.name}_w{w_idx:03d}",
                        'start_timestamp': start_ts_ns,
                        'end_timestamp':   end_ts_ns,
                        'num_timestamps':  WINDOW_SIZE,
                        'focal_track_id':  focal_uuid,
                        'city':            city_name,
                    })

        if len(rows) == 0:
            continue

        # ── Add AV (ego vehicle) ──────────────────────────────
        av_rows = []
        for ts in window_ts:
            is_observed = ts in hist_ts
            if ts in ego_lookup.index:
                ego_row = ego_lookup.loc[ts]
                if isinstance(ego_row, pd.DataFrame):
                    ego_row = ego_row.iloc[0]

                ego_tx  = float(ego_row['tx_m'])
                ego_ty  = float(ego_row['ty_m'])
                ego_yaw = quaternion_to_yaw(
                    float(ego_row['qw']),
                    float(ego_row['qx']),
                    float(ego_row['qy']),
                    float(ego_row['qz'])
                )

                av_rows.append({
                    'observed':        bool(is_observed),
                    'track_id':        'AV',
                    'object_type':     'vehicle',
                    'object_category': 1,
                    'timestep':        ts_to_idx[ts],
                    'position_x':      ego_tx,
                    'position_y':      ego_ty,
                    'heading':         ego_yaw,
                    'velocity_x':      0.0,
                    'velocity_y':      0.0,
                    'timestamp_ns':    int(ts),            # FIX 1: added
                    'scenario_id':     f"{LOG_DIR.name}_w{w_idx:03d}",
                    'start_timestamp': start_ts_ns,
                    'end_timestamp':   end_ts_ns,
                    'num_timestamps':  WINDOW_SIZE,
                    'focal_track_id':  focal_uuid,
                    'city':            city_name,
                })

        # Compute AV velocities from positions
        if len(av_rows) > 1:
            for i in range(len(av_rows)):
                if 0 < i < len(av_rows) - 1:
                    dt = (window_ts[av_rows[i+1]['timestep']] -
                          window_ts[av_rows[i-1]['timestep']]) * 1e-9
                    dx = av_rows[i+1]['position_x'] - av_rows[i-1]['position_x']
                    dy = av_rows[i+1]['position_y'] - av_rows[i-1]['position_y']
                    av_rows[i]['velocity_x'] = dx / dt if dt > 0 else 0.0
                    av_rows[i]['velocity_y'] = dy / dt if dt > 0 else 0.0

        rows.extend(av_rows)

        scenario_df = pd.DataFrame(rows)

        col_order = [
            'observed', 'track_id', 'object_type', 'object_category',
            'timestep', 'position_x', 'position_y', 'heading',
            'velocity_x', 'velocity_y', 'timestamp_ns',   # FIX 1: added
            'scenario_id', 'start_timestamp', 'end_timestamp',
            'num_timestamps', 'focal_track_id', 'city'
        ]
        scenario_df = scenario_df[col_order]

        scenario_id     = f"{LOG_DIR.name}_w{w_idx:03d}"
        scenario_folder = OUTPUT_DIR / scenario_id
        scenario_folder.mkdir(parents=True, exist_ok=True)

        out_path = scenario_folder / f"{scenario_id}.parquet"
        scenario_df.to_parquet(out_path, index=False)

        map_dest = scenario_folder / map_json_path.name
        if not map_dest.exists():
            shutil.copy2(map_json_path, map_dest)

        print(f"    Window {w_idx+1}/{n_windows} → {scenario_id}/ "
              f"({len(scenario_df)} rows, "
              f"{scenario_df['track_id'].nunique()} agents)")

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

log_dirs = sorted([d for d in SENSOR_DIR.iterdir() if d.is_dir()])
print(f"Found {len(log_dirs)} logs in {SENSOR_DIR}\n")

for log_idx, log_dir in enumerate(log_dirs):
    print(f"[{log_idx+1}/{len(log_dirs)}] Processing {log_dir.name}...")
    process_log(log_dir)

print(f"\n✅ Done. All logs processed. Output saved to {OUTPUT_BASE}")