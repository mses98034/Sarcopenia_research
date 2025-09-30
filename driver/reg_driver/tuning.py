import optuna
import subprocess
import os
import re
import configparser
from datetime import datetime

# --- Configuration ---
# The number of experiments Optuna will run.
N_TRIALS = 100

# The base configuration file to use as a template.
TEMPLATE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'reg_configuration.txt')

# Directory to run the training script from.
WORKING_DIRECTORY = os.path.dirname(__file__)

# Directory to store temporary configs and trial logs.
TUNING_OUTPUT_DIR = os.path.join(WORKING_DIRECTORY, 'tuning_results')
os.makedirs(TUNING_OUTPUT_DIR, exist_ok=True)


def parse_pearson_from_output(output):
    """Parses the mean Pearson correlation from the stdout of train.py."""
    # Regex to find the line 'Pearson r - Mean: 0.1234  0.0456'
    match = re.search(r"Pearson r\s+-\s+Mean:\s+([\d\.]+)", output)
    if match:
        return float(match.group(1))
    else:
        print("\n⚠️  Warning: Could not parse Pearson correlation from training output. Returning 0.0")
        print("==================== STDOUT ====================")
        print(output)
        print("==============================================")
        return 0.0

def objective(trial):
    """Defines a single experiment (trial) for Optuna to run."""
    # --- 1. Suggest Hyperparameters ---
    params = {
        'backbone': trial.suggest_categorical('backbone', ['resnet18', 'resnet34']),
        'model': trial.suggest_categorical('model', ['ResNetFusionTextNetRegression', 'ResNetFusionAttentionNetRegression']),
        # 'learning_algorithm': trial.suggest_categorical('learning_algorithm', ['adam', 'adamw']),
        # 'scheduler_type': trial.suggest_categorical('scheduler_type', ['plateau', 'cosine']),
        'loss_function': trial.suggest_categorical('loss_function', ['mse', 'huber', 'pearson', 'ccc']),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
    }

    # --- 2. Create a Temporary Config File for this Trial ---
    config = configparser.ConfigParser()
    config.read(TEMPLATE_CONFIG_PATH)

    config.set('Network', 'backbone', params['backbone'])
    config.set('Network', 'model', params['model'])
    # config.set('Optimizer', 'learning_algorithm', params['learning_algorithm'])
    config.set('Optimizer', 'learning_rate', str(params['learning_rate']))
    config.set('Optimizer', 'weight_decay', str(params['weight_decay']))
    # config.set('Scheduler', 'scheduler_type', params['scheduler_type'])
    config.set('Loss', 'loss_function', params['loss_function'])

    trial_config_filename = f"trial_{trial.number}_config.txt"
    trial_config_path = os.path.join(TUNING_OUTPUT_DIR, trial_config_filename)
    with open(trial_config_path, 'w') as f:
        config.write(f)

    # --- 3. Run the Training Script and Stream Output in Real-time ---
    command = [
        'python',
        'train.py',
        '--config', trial_config_path,
        '--gpu', '0',
    ]

    print(f"\n\n--- Starting Trial {trial.number} ---")
    print(f"Params: {trial.params}")
    print(f"Command: {' '.join(command)}")

    full_output = []
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            command,
            cwd=WORKING_DIRECTORY,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout
            text=True,
            # encoding='utf-8'
        )

        # Read and print output line by line
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line, end='') # Print to console in real-time
            full_output.append(line)
        
        process.wait()

        if process.returncode != 0:
            print(f"\n❌ Trial {trial.number} failed with exit code {process.returncode}.")
            raise optuna.exceptions.TrialPruned() # Prune trial if script fails

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Trial {trial.number} failed with an error.")
        full_output_str = "".join(full_output)
        print(full_output_str)
        raise optuna.exceptions.TrialPruned()
    except KeyboardInterrupt:
        print(f"\n🛑 Trial {trial.number} interrupted by user. Pruning trial.")
        process.terminate() # Ensure the subprocess is killed
        raise optuna.exceptions.TrialPruned()

    # --- 4. Parse the Result and Return to Optuna ---
    output_str = "".join(full_output)
    pearson_score = parse_pearson_from_output(output_str)
    
    print(f"--- Trial {trial.number} Finished --- Pearson Score: {pearson_score:.4f} ---")

    return pearson_score


if __name__ == '__main__':
    # --- Create and Run the Study ---
    # We want to maximize the Pearson correlation.
    study = optuna.create_study(direction='maximize')

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

    print("\nTo visualize the results, you can use optuna-dashboard:")
    print(f"  pip install optuna-dashboard")
    print(f"  optuna-dashboard sqlite:///optuna.db")
