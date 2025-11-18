import os
import subprocess
import json
import shutil
import re
import time

# ===================================
# CONFIGURATION
# ===================================
TRAINER_SCRIPT = "trainer_no_validation.py"
BACKTESTER_SCRIPT = "backtester.py"
CONFIG_FILE = "config.py"
BEST_RESULT_JSON = "best_model.json"
MODELS_DIR = "rl_models"
ITERATIONS = 5
TICKER = "EURUSDc"

# ===================================
# UTILITY FUNCTIONS
# ===================================

def run_command(cmd):
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def update_learning_rate_in_config(new_lr):
    """Updates learning_rate in config_optimized.py dynamically."""
    with open(CONFIG_FILE, "r") as f:
        content = f.read()

    # Regex pattern to find: "learning_rate": 1e-4 or 0.0001 etc.
    updated_content = re.sub(
        r'"learning_rate":\s*[\deE\.\-]+',
        f'"learning_rate": {new_lr}',
        content
    )

    with open(CONFIG_FILE, "w") as f:
        f.write(updated_content)

    print(f"✅ Updated learning rate in {CONFIG_FILE} to {new_lr}")

def get_current_learning_rate():
    """Reads current learning_rate from config."""
    with open(CONFIG_FILE, "r") as f:
        content = f.read()
    match = re.search(r'"learning_rate":\s*([\deE\.\-]+)', content)
    if match:
        return float(match.group(1))
    raise ValueError("Could not find learning_rate in config file.")

def get_best_model_from_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Result file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    best_model = data.get("best_model")
    calmar_ratio = data.get("calmar_ratio", None)
    if not best_model:
        raise ValueError("JSON must contain key 'best_model'")
    return best_model, calmar_ratio

def cleanup_models(best_model_path):
    print("\n>>> Cleaning up other models...")
    for file in os.listdir(MODELS_DIR):
        full_path = os.path.join(MODELS_DIR, file)
        if file.endswith(".zip") and full_path != best_model_path:
            os.remove(full_path)
    print(">>> Cleanup complete.")

# ===================================
# MAIN TRAINING LOOP
# ===================================
lr = get_current_learning_rate()
print(f"🚀 Starting Auto-Training at learning rate {lr}")

for i in range(1, ITERATIONS + 1):
    print(f"\n==============================")
    print(f"🔁 Iteration {i} — Learning Rate: {lr}")
    print(f"==============================")

    # 1️⃣ Update learning rate in config
    update_learning_rate_in_config(lr)

    # 2️⃣ Run trainer
    run_command(f"python {TRAINER_SCRIPT}")

    # 3️⃣ Run backtester
    run_command(f"python {BACKTESTER_SCRIPT}")

    # 4️⃣ Load best model info
    best_model, best_score = get_best_model_from_json(BEST_RESULT_JSON)
    print(f"🏆 Best Model: {best_model} (Calmar Ratio: {best_score})")

    # 5️⃣ Rename best model
    best_model_path = os.path.join(MODELS_DIR, best_model)
    final_model_path = os.path.join(MODELS_DIR, f"rl_agent_{TICKER}_final.zip")

    if os.path.exists(best_model_path):
        if best_model_path != final_model_path:
            shutil.move(best_model_path, final_model_path)
            print(f"✅ Moved best model to: {final_model_path}")
        else:
            print("Source and destination are the same. No move needed.")
    else:
        print(f"Error: File does not exist -> {best_model_path}")

    # 6️⃣ Delete other models
    cleanup_models(final_model_path)

    # 7️⃣ Lower learning rate by one decimal (e.g. 1e-4 → 1e-5)
    lr = lr / 10.0
    time.sleep(3)

print("\n🎯 All training cycles completed successfully!")
