import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import DataLoader

# ── Add repo to path ──────────────────────────────────────────────────────────
REPO_DIR = "/content/HiVT-av2"
sys.path.insert(0, REPO_DIR)

from datasets import ArgoverseV2Dataset
from models.hivt import HiVT
from av2.map.map_api import ArgoverseStaticMap

# ── Config ────────────────────────────────────────────────────────────────────
MF_DATA_ROOT = "/content/drive/MyDrive/Amir_Dataset/HiVT-project_SMF/av2/motion-forecasting"
CKPT_PATH    = "/content/drive/MyDrive/Amir_Dataset/HiVT_av2_checkpoints/epoch=63-step=4032.ckpt"
OUTPUT_CSV   = "/content/drive/MyDrive/Amir_Dataset/HiVT-project_Confidence/hivt_focal_inspection.csv"
BATCH_SIZE   = 1
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Constants matching Nadeem's heuristic ─────────────────────────────────────
MIN_SPEED_STOPPED               = 0.5
MIN_SPEED_MOVING                = 1.0
PARKED_MAX_DISP_M               = 0.5
HEADING_CHANGE_THRESH_TURN      = 20.0
HEADING_CHANGE_THRESH_LANE_KEEP = 5.0
KEEP_LANE_MAX_LAT_DIST          = 0.5

# ── Helper: load map ──────────────────────────────────────────────────────────
def load_static_map(raw_dir):
    map_files = list(Path(raw_dir).glob("log_map_archive_*.json"))
    if not map_files:
        return None
    try:
        return ArgoverseStaticMap.from_json(map_files[0])
    except Exception:
        return None

# ── Helper: infer intention from predicted trajectory (agent-centric, no map) ─
def infer_intention_from_trajectory(traj):
    """
    traj: [N, 2] agent-centric local frame.
    No map check — map polygons don't apply in agent-centric frame.
    """
    if traj.shape[0] < 5:
        return "OTHER"

    speeds       = np.linalg.norm(np.diff(traj, axis=0), axis=1) / 0.1
    avg_speed    = speeds.mean()
    displacement = np.linalg.norm(traj[-1] - traj[0])

    if avg_speed < MIN_SPEED_STOPPED:
        return "PARKED" if displacement < PARKED_MAX_DISP_M else "STOPPING_STOPPED"

    start_vec = traj[min(5, len(traj)-1)] - traj[0]
    end_vec   = traj[-1] - traj[max(0, len(traj)-6)]

    if np.linalg.norm(start_vec) < 0.1 or np.linalg.norm(end_vec) < 0.1:
        return "KEEP_LANE"

    initial_heading    = np.arctan2(start_vec[1], start_vec[0])
    final_heading      = np.arctan2(end_vec[1], end_vec[0])
    heading_change_deg = np.degrees(final_heading - initial_heading)
    heading_change_deg = (heading_change_deg + 180) % 360 - 180

    if avg_speed > MIN_SPEED_MOVING and abs(heading_change_deg) > HEADING_CHANGE_THRESH_TURN:
        return "TURN_LEFT" if heading_change_deg > 0 else "TURN_RIGHT"

    if abs(heading_change_deg) > HEADING_CHANGE_THRESH_LANE_KEEP:
        return "LEFT_CHANGE_LANE" if heading_change_deg > 0 else "RIGHT_CHANGE_LANE"

    if abs(heading_change_deg) <= HEADING_CHANGE_THRESH_LANE_KEEP:
        direction = traj[-1] - traj[0]
        dir_norm  = np.linalg.norm(direction)
        if dir_norm > 0.01:
            direction_unit = direction / dir_norm
            perp           = np.array([-direction_unit[1], direction_unit[0]])
            lateral_devs   = np.abs((traj - traj[0]) @ perp)
            if lateral_devs.max() <= KEEP_LANE_MAX_LAT_DIST:
                return "KEEP_LANE"

    return "OTHER"

# ── Helper: GT intention from parquet (city frame, with map) ─────────────────
def get_ground_truth_intention(scenario_parquet, track_id, static_map):
    """
    GT from city-frame parquet positions + annotated headings.
    Uses Nadeem's exact heuristic logic including shapely polygon check.
    GT window: timesteps 50-79 (3s at 10Hz).
    """
    if scenario_parquet is None:
        return "UNKNOWN"

    agent_df = scenario_parquet[
        (scenario_parquet['track_id'] == track_id) &
        (scenario_parquet['timestep'] >= 50) &
        (scenario_parquet['timestep'] < 80)
    ].sort_values('timestep')

    if len(agent_df) < 5:
        return "UNKNOWN"

    traj = agent_df[['position_x', 'position_y']].to_numpy()

    # ── Speed from velocity columns ───────────────────────────────────────────
    speeds       = np.sqrt(agent_df['velocity_x']**2 + agent_df['velocity_y']**2).values
    avg_speed    = speeds.mean()
    displacement = np.linalg.norm(traj[-1] - traj[0])

    # ── PARKED / STOPPING_STOPPED ─────────────────────────────────────────────
    if avg_speed < MIN_SPEED_STOPPED:
        return "PARKED" if displacement < PARKED_MAX_DISP_M else "STOPPING_STOPPED"

    # ── Heading from annotated column ─────────────────────────────────────────
    start_h            = float(agent_df.iloc[0]['heading'])
    end_h              = float(agent_df.iloc[-1]['heading'])
    heading_change_deg = np.degrees(np.arctan2(
        np.sin(end_h - start_h), np.cos(end_h - start_h)
    ))

    # ── City-frame position at first future step for map checks ───────────────
    city_pos = np.array([
        float(agent_df.iloc[0]['position_x']),
        float(agent_df.iloc[0]['position_y'])
    ])

    # ── TURN LEFT / TURN RIGHT ────────────────────────────────────────────────
    if avg_speed > MIN_SPEED_MOVING and abs(heading_change_deg) > HEADING_CHANGE_THRESH_TURN:
        return "TURN_LEFT" if heading_change_deg > 0 else "TURN_RIGHT"

    # ── Map context: intersection check ──────────────────────────────────────
    is_intersection      = False
    map_context_available = False
    best_lane_id         = None

    if static_map is not None:
        try:
            if abs(heading_change_deg) <= HEADING_CHANGE_THRESH_TURN and avg_speed >= MIN_SPEED_MOVING:
                nearby = static_map.get_nearby_lane_segments(city_pos, 5.0)
                if nearby:
                    best_lane_id = min(nearby, key=lambda x: x[1])[0]
                    lane_obj     = static_map.vector_lane_segments.get(best_lane_id)
                    if lane_obj:
                        is_intersection       = lane_obj.is_intersection
                        map_context_available = True
        except Exception:
            map_context_available = False

    # ── If in intersection with small heading change → KEEP_LANE ─────────────
    if map_context_available and is_intersection and avg_speed >= MIN_SPEED_MOVING:
        return "KEEP_LANE" if abs(heading_change_deg) <= HEADING_CHANGE_THRESH_LANE_KEEP else "OTHER"

    # ── LEFT / RIGHT CHANGE LANE ──────────────────────────────────────────────
    if (avg_speed >= MIN_SPEED_MOVING and
            not (map_context_available and is_intersection) and
            abs(heading_change_deg) > HEADING_CHANGE_THRESH_LANE_KEEP and
            abs(heading_change_deg) <= HEADING_CHANGE_THRESH_TURN):
        return "LEFT_CHANGE_LANE" if heading_change_deg > 0 else "RIGHT_CHANGE_LANE"

    # ── KEEP_LANE with shapely polygon check (Nadeem's exact method) ──────────
    if avg_speed >= MIN_SPEED_MOVING and abs(heading_change_deg) <= HEADING_CHANGE_THRESH_LANE_KEEP:
        can_check_polygons = False
        points_stay_in_lane = False

        if static_map is not None and map_context_available and not is_intersection:
            try:
                from shapely.geometry import Point
                lane_id_for_check = best_lane_id
                if lane_id_for_check is None:
                    nearby = static_map.get_nearby_lane_segments(city_pos, 5.0)
                    if nearby:
                        lane_id_for_check = min(nearby, key=lambda x: x[1])[0]

                if lane_id_for_check is not None:
                    successor_ids  = static_map.get_lane_segment_successor_ids(lane_id_for_check) or set()
                    valid_lane_ids = {lane_id_for_check}.union(successor_ids)
                    valid_lane_polys = [
                        p for p in [
                            static_map.get_lane_segment_polygon(lid)
                            for lid in valid_lane_ids
                        ]
                        if p is not None and p.is_valid
                    ]

                    if valid_lane_polys:
                        can_check_polygons = True
                        future_points = [
                            Point(float(row['position_x']), float(row['position_y']))
                            for _, row in agent_df.iterrows()
                        ]
                        points_stay_in_lane = all(
                            any(poly.contains(pt) for poly in valid_lane_polys)
                            for pt in future_points
                        )
            except Exception:
                can_check_polygons = False

        if can_check_polygons and points_stay_in_lane:
            return "KEEP_LANE"
        elif not can_check_polygons:
            # Fallback: lateral distance check
            direction = traj[-1] - traj[0]
            dir_norm  = np.linalg.norm(direction)
            if dir_norm > 0.01:
                direction_unit = direction / dir_norm
                perp           = np.array([-direction_unit[1], direction_unit[0]])
                lateral_devs   = np.abs((traj - traj[0]) @ perp)
                if lateral_devs.max() <= KEEP_LANE_MAX_LAT_DIST:
                    return "KEEP_LANE"

    return "OTHER"

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading HiVT model...")
model = HiVT.load_from_checkpoint(
    checkpoint_path=CKPT_PATH,
    map_location=DEVICE,
    strict=False
)
model.eval()
model = model.to(DEVICE)
print("✅ HiVT model loaded")

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading converted MF dataset...")
dataset = ArgoverseV2Dataset(
    root=MF_DATA_ROOT,
    split='val',
    local_radius=model.hparams.local_radius
)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)
print(f"✅ {len(dataset)} scenarios loaded")

# ── Run inference — focal agent only ─────────────────────────────────────────
print("\nRunning HiVT inference (focal agent only)...")
rows = []

with torch.no_grad():
    for batch_idx, data in enumerate(loader):
        try:
            data = data.to(DEVICE)
            y_hat, pi = model(data)

            probs = torch.softmax(pi, dim=-1)  # [N, 6]

            seq_id = data.seq_id if hasattr(data, 'seq_id') else f"scenario_{batch_idx}"
            if isinstance(seq_id, list):
                seq_id = seq_id[0]

            parts  = seq_id.rsplit('_w', 1)
            log_id = parts[0] if len(parts) == 2 else seq_id

            # ── Load raw parquet ──────────────────────────────────────────────
            scenario_parquet = None
            raw_dir = os.path.join(
                MF_DATA_ROOT, "dataset", "train", "scenarios", seq_id
            )
            parquet_files = list(Path(raw_dir).glob("*.parquet")) if os.path.exists(raw_dir) else []
            if parquet_files:
                scenario_parquet = pd.read_parquet(parquet_files[0])

            # ── Load map ──────────────────────────────────────────────────────
            static_map = load_static_map(raw_dir)

            # ── Focal track id ────────────────────────────────────────────────
            focal_track_id = "UNKNOWN"
            if scenario_parquet is not None:
                focal_ids = scenario_parquet['focal_track_id'].unique()
                if len(focal_ids) > 0:
                    focal_track_id = focal_ids[0]

            # ── Focal agent trajectories ──────────────────────────────────────
            focal_index = data.agent_index.item() if hasattr(data, 'agent_index') else 0
            focal_trajs = y_hat[:, focal_index, :, :2].cpu().numpy()  # [6, 60, 2]
            focal_probs = probs[focal_index].cpu().numpy()             # [6]

            best_mode = focal_probs.argmax()
            best_traj = focal_trajs[best_mode]
            best_prob = focal_probs[best_mode]

            # ── Predicted intention (agent-centric, no map) ───────────────────
            predicted_intention = infer_intention_from_trajectory(best_traj)

            # ── GT intention (city frame, with map) ───────────────────────────
            actual_intention = get_ground_truth_intention(
                scenario_parquet, focal_track_id, static_map
            )
            correct = (predicted_intention == actual_intention) \
                if actual_intention != "UNKNOWN" else None

            # ── All 6 mode intentions ─────────────────────────────────────────
            mode_intentions = [
                infer_intention_from_trajectory(focal_trajs[m]) for m in range(6)
            ]

            # ── Focal category ────────────────────────────────────────────────
            focal_category = 3
            if scenario_parquet is not None and focal_track_id in scenario_parquet['track_id'].values:
                focal_category = int(scenario_parquet[
                    scenario_parquet['track_id'] == focal_track_id
                ]['object_category'].iloc[0])

            rows.append({
                'scenario_id'        : seq_id,
                'log_id'             : log_id,
                'track_id'           : focal_track_id,
                'object_category'    : focal_category,
                'is_focal_agent'     : True,
                'best_mode'          : int(best_mode),
                'predicted_intention': predicted_intention,
                'actual_intention'   : actual_intention,
                'correct'            : correct,
                'traj_confidence'    : float(best_prob),
                'mode_0_prob'        : float(focal_probs[0]),
                'mode_1_prob'        : float(focal_probs[1]),
                'mode_2_prob'        : float(focal_probs[2]),
                'mode_3_prob'        : float(focal_probs[3]),
                'mode_4_prob'        : float(focal_probs[4]),
                'mode_5_prob'        : float(focal_probs[5]),
                'mode_0_intention'   : mode_intentions[0],
                'mode_1_intention'   : mode_intentions[1],
                'mode_2_intention'   : mode_intentions[2],
                'mode_3_intention'   : mode_intentions[3],
                'mode_4_intention'   : mode_intentions[4],
                'mode_5_intention'   : mode_intentions[5],
            })

        except Exception as e:
            print(f"❌ Error on scenario {batch_idx}: {e}")
            import traceback; traceback.print_exc()
            continue

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame(rows)

# ── Helper: print summary ─────────────────────────────────────────────────────
def print_summary(subset):
    known     = subset[subset['actual_intention'] != 'UNKNOWN']
    n_total   = len(known)
    n_correct = known['correct'].sum()
    avg_conf  = known['traj_confidence'].mean() if n_total > 0 else 0.0

    print(f"    Total focal agents : {len(subset)}")
    print(f"    With ground truth  : {n_total}")
    print(f"    Correct intention  : {n_correct} / {n_total} ({100*n_correct/max(n_total,1):.1f}%)")
    print(f"    Avg traj confidence: {avg_conf:.4f}")
    print(f"    Per-intention accuracy:")
    for intent in sorted(known['actual_intention'].unique()):
        sub    = known[known['actual_intention'] == intent]
        acc    = sub['correct'].mean() * 100
        avg_ic = sub['traj_confidence'].mean()
        print(f"      {intent:<22}: {acc:.1f}%  ({sub['correct'].sum()}/{len(sub)})  "
              f"avg conf: {avg_ic:.4f}")

# ── Print per log ─────────────────────────────────────────────────────────────
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

for log_id in df['log_id'].unique():
    print("\n" + "="*80)
    print(f"LOG: {log_id}")
    print("="*80)
    log_df = df[df['log_id'] == log_id]

    for scenario_id in log_df['scenario_id'].unique():
        print(f"\n  Scenario: {scenario_id}")
        print(f"  {'track_id':<40} {'predicted':<22} {'actual':<22} {'correct':<8} {'conf':<8} {'all modes'}")
        print(f"  {'-'*150}")
        scen_df = log_df[log_df['scenario_id'] == scenario_id]
        for _, row in scen_df.iterrows():
            mode_str = " | ".join([
                f"m{i}:{row[f'mode_{i}_intention'][:4]}({row[f'mode_{i}_prob']:.2f})"
                for i in range(6)
            ])
            correct_str = str(row['correct']) if row['correct'] is not None else "N/A"
            print(f"  {str(row['track_id']):<40} "
                  f"{row['predicted_intention']:<22} {row['actual_intention']:<22} "
                  f"{correct_str:<8} {row['traj_confidence']:.4f}   {mode_str}")

    print(f"\n  Summary for {log_id}:")
    print_summary(log_df)

# ── Grand total ───────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("GRAND TOTAL — ALL SCENARIOS (FOCAL AGENT ONLY)")
print("="*80)
print_summary(df)

# ── Save CSV ──────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ CSV saved to: {OUTPUT_CSV}")