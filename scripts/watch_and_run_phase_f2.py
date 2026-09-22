"""
AI Search Framework - Phase F2: Automated Drive Watcher & Evaluation Trigger
Monitors Google Drive (G:\\My Drive\\ai_search_phase_f\\) for training checkpoints and exported GGUF.
Once export is complete:
1. Automatically deploys model to Ollama via scripts/deploy_phi35_gguf.py.
2. Automatically executes double-blind pairwise evaluation via scripts/run_phase_f_retrieval_eval.py.
3. Logs all audit checks to results/phase_f/.
"""

import os
import sys
import time
import subprocess

DRIVE_DIR = r"G:\My Drive\ai_search_phase_f"
CHECKPOINT_DIR = os.path.join(DRIVE_DIR, "checkpoints")
EXPORT_DIR = os.path.join(DRIVE_DIR, "exported_gguf")
TARGET_GGUF = os.path.join(EXPORT_DIR, "phi3.5_ft_retrieval_q4_k_m.gguf")

def main():
    print("=" * 80)
    print("PHASE F2 BACKGROUND WATCHER ACTIVE")
    print(f"Monitoring Target: {TARGET_GGUF}")
    print(f"Monitoring Checkpoints: {CHECKPOINT_DIR}")
    print("=" * 80)

    seen_checkpoints = set()
    last_log_time = time.time()

    while True:
        # 1. Check for incremental checkpoints
        if os.path.exists(CHECKPOINT_DIR):
            try:
                current_ckpts = set(os.listdir(CHECKPOINT_DIR))
                new_ckpts = current_ckpts - seen_checkpoints
                for ckpt in sorted(new_ckpts):
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TRAINING PROGRESS: Detected new checkpoint: {ckpt}")
                    seen_checkpoints.add(ckpt)
            except Exception as e:
                pass

        # 2. Check for exported GGUF
        if os.path.exists(TARGET_GGUF):
            size1 = os.path.getsize(TARGET_GGUF)
            # Ensure file is non-empty and has finished syncing (size > 1GB and stable for 10s)
            if size1 > 1024 * 1024 * 1024:  # > 1GB
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Detected exported GGUF ({size1 / (1024**3):.2f} GB). Verifying sync completion...")
                time.sleep(10)
                size2 = os.path.getsize(TARGET_GGUF)
                if size1 == size2:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sync confirmed! File size stable at {size2 / (1024**3):.2f} GB.")
                    break
                else:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] File still actively writing/syncing ({size2 / (1024**3):.2f} GB). Waiting...")
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GGUF file detected but size is only {size1 / (1024**2):.1f} MB. Sync in progress...")

        # Periodic heartbeat every 60s
        if time.time() - last_log_time > 60:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Watcher heartbeat: waiting for Colab training / export... (Checkpoints seen: {len(seen_checkpoints)})")
            last_log_time = time.time()

        time.sleep(10)

    print("\n" + "=" * 80)
    print("STEP F2.1: DEPLOYING MODEL TO OLLAMA")
    print("=" * 80)
    cmd_deploy = [sys.executable, "scripts/deploy_phi35_gguf.py", TARGET_GGUF]
    res_deploy = subprocess.run(cmd_deploy, text=True)
    if res_deploy.returncode != 0:
        print(f"Deployment failed with returncode {res_deploy.returncode}. Aborting.")
        sys.exit(res_deploy.returncode)

    print("\n" + "=" * 80)
    print("STEP F2.2: EXECUTING DOUBLE-BLIND PAIRWISE EVALUATION & AUTONOMOUS AUDIT LOOP")
    print("=" * 80)
    cmd_eval = [sys.executable, "scripts/run_phase_f_retrieval_eval.py"]
    res_eval = subprocess.run(cmd_eval, text=True)
    if res_eval.returncode != 0:
        print(f"Evaluation failed with returncode {res_eval.returncode}. Aborting.")
        sys.exit(res_eval.returncode)

    print("\n" + "=" * 80)
    print("PHASE F2 EXECUTION AND EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

