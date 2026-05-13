script = '''
import sys, yaml, subprocess
from pathlib import Path

sys.path.insert(0, "/content/MultiTask-ViT-AV2")

REPO_DIR    = "/content/MultiTask-ViT-AV2"
BASE_CONFIG = f"{REPO_DIR}/configs/v2_mlp.yaml"
LOG_DIR     = "/content/ablation_logs"
Path(LOG_DIR).mkdir(exist_ok=True)

LAMBDA_VALUES = [0.01, 0.1, 0.5, 1.0]

def make_config(lam):
    with open(BASE_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["loss"]["traj_lambda"] = lam
    cfg["training"]["num_epochs"] = 3
    cfg["training"]["batch_size"] = 1
    cfg["checkpoints"]["filename"] = f"ablation_lambda_{lam}.pth"
    out = f"{LOG_DIR}/config_lambda_{lam}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return out

def run(config_path, lam):
    log_path = f"{LOG_DIR}/log_lambda_{lam}.txt"
    print(f"\\n{'='*50}")
    print(f"  Running λ = {lam}")
    print(f"{'='*50}\\n")
    with open(log_path, "w") as log_file:
        p = subprocess.Popen(
            ["python", f"{REPO_DIR}/training/train.py",
             "--config", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in p.stdout:
            print(line, end="")
            log_file.write(line)
        p.wait()
    print(f"\\n✅ λ={lam} done. Log saved to {log_path}")
    return p.returncode

print("🚀 Starting λ ablation — 4 runs × 3 epochs")
for lam in LAMBDA_VALUES:
    cfg_path = make_config(lam)
    run(cfg_path, lam)
print("\\n🏁 All 4 ablation runs complete.")
print(f"   Logs in: {LOG_DIR}")
'''

with open("/content/MultiTask-ViT-AV2/training/run_ablation.py", "w") as f:
    f.write(script)

print("Script saved.")