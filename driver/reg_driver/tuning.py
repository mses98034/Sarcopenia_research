
import optuna
import subprocess
import os
import re
import json
import configparser
from datetime import datetime
from optuna.pruners import MedianPruner

# --- Configuration ---
N_TRIALS = 30
TEMPLATE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'reg_configuration.txt')
WORKING_DIRECTORY = os.path.dirname(__file__)
TUNING_OUTPUT_DIR = os.path.join(WORKING_DIRECTORY, 'tuning_results')
os.makedirs(TUNING_OUTPUT_DIR, exist_ok=True)

def parse_pearson_from_output(output):
    """Parses the mean Pearson correlation from the stdout of train.py."""
    match = re.search(r"Pearson r\s+-\s+Mean:\s+([\d\.]+)", output)
    if match:
        return float(match.group(1))
    else:
        print("\n⚠️  Warning: Could not parse final Pearson correlation from training output. Returning 0.0")
        return 0.0

def objective(trial):
    """Defines a single experiment (trial) for Optuna to run."""
    # --- 1. Suggest Hyperparameters ---
    params = {
        'backbone': trial.suggest_categorical('backbone', ['resnet18', 'resnet34']),
        'model': trial.suggest_categorical('model', ['ResNetFusionTextNetRegression', 'ResNetFusionAttentionNetRegression']),
        # 'learning_algorithm': trial.suggest_categorical('learning_algorithm', ['adam', 'adamw']),
        'scheduler_type': trial.suggest_categorical('scheduler_type', ['plateau', 'cosine']),
        'loss_function': trial.suggest_categorical('loss_function', ['mse', 'ccc']),
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
        'contrastive_learning': trial.suggest_categorical('contrastive_learning', [True, False]),
    }
    if params['contrastive_learning']:
        params['contrastive_beta'] = trial.suggest_float('contrastive_beta', 0.001, 0.1, log=True)
        params['contrastive_temperature'] = trial.suggest_float('contrastive_temperature', 0.1, 1.0)
    else:
        params['contrastive_beta'] = 0.0
        params['contrastive_temperature'] = 0.5

    # --- 2. Create a Temporary Config File for this Trial ---
    config = configparser.ConfigParser()
    config.read(TEMPLATE_CONFIG_PATH)
    for section, key, value in [
        ('Network', 'backbone', params['backbone']),
        ('Network', 'model', params['model']),
        # ('Optimizer', 'learning_algorithm', params['learning_algorithm']),
        ('Optimizer', 'learning_rate', str(params['learning_rate'])),
        ('Optimizer', 'weight_decay', str(params['weight_decay'])),
        ('Scheduler', 'scheduler_type', params['scheduler_type']),
        ('Loss', 'loss_function', params['loss_function']),
        ('Contrastive', 'contrastive_learning', str(params['contrastive_learning'])),
        ('Contrastive', 'contrastive_beta', str(params['contrastive_beta'])),
        ('Contrastive', 'contrastive_temperature', str(params['contrastive_temperature']))
    ]:
        config.set(section, key, value)

    trial_config_filename = f"trial_{trial.number}_config.txt"
    trial_config_path = os.path.join(TUNING_OUTPUT_DIR, trial_config_filename)
    with open(trial_config_path, 'w') as f:
        config.write(f)

    # --- 3. Run the Training Script and Monitor for Pruning ---
    command = ['python', 'train.py', '--config', trial_config_path, '--gpu', '0', '--seed', '666']
    print(f"\n\n--- Starting Trial {trial.number} ---")
    print(f"Params: {trial.params}")

    full_output = []
    try:
        process = subprocess.Popen(command, cwd=WORKING_DIRECTORY, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line, end='')
            full_output.append(line)

            if line.strip().startswith("[TUNING_REPORT]"):
                try:
                    json_str = line.strip().replace("[TUNING_REPORT] ", "")
                    report_data = json.loads(json_str)
                    fold = report_data.get("fold")
                    pearson = report_data.get("pearson")
                    if fold is not None and pearson is not None:
                        trial.report(pearson, fold)
                        if trial.should_prune():
                            process.terminate()
                            print(f"--- [Tuning Supervisor]: Trial pruned at Fold {fold} due to poor performance. ---")
                            raise optuna.exceptions.TrialPruned()
                except Exception as e:
                    print(f"--- [Tuning Supervisor]: Warning - could not parse report line: {e} ---")
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args, stdout="".join(full_output))

    except (subprocess.CalledProcessError, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            print(f"\n🛑 Trial {trial.number} interrupted by user. Pruning trial.")
        else:
            print(f"\n❌ Trial {trial.number} failed with exit code {e.returncode}.")
        if 'process' in locals() and process.poll() is None:
            process.terminate()
        raise optuna.exceptions.TrialPruned()

    # --- 4. Parse Final Result and Return to Optuna ---
    output_str = "".join(full_output)
    final_pearson_score = parse_pearson_from_output(output_str)
    print(f"--- Trial {trial.number} Finished --- Final Pearson Score: {final_pearson_score:.4f} ---")
    return final_pearson_score

if __name__ == '__main__':
    # Configure a pruner to stop unpromising trials early.
    # n_startup_trials: Don't prune the first 5 trials.
    # n_warmup_steps: Don't prune within a trial before step 2 (i.e., after Fold 2 results).
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)
    study = optuna.create_study(direction='maximize', pruner=pruner)

    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("\n🛑 Tuning stopped manually.")

    # --- Print the Results ---
    print("\n\n==================================================")
    print("            TUNING COMPLETE                     ")
    print("==================================================")
    print(f"Number of finished trials: {len(study.trials)}")
    print("\n🏆 Best trial:")
    trial = study.best_trial
    print(f"  Value (Pearson r): {trial.value:.4f}")
    print("\n  Best Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
