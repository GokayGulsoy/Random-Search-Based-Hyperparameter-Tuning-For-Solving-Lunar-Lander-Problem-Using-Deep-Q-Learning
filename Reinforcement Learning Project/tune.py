import random
import warnings
from main import train_dqn

# since pygame library installed by Gymnasium uses an
# older method (pkg_resources) to find files which Python 
# now considers outdated it gives a warning, it can be 
# suppressed with the following code
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

# defining search space
param_grid = {
    "lr": [5e-3, 1e-3, 5e-4, 1e-4],
    "gamma": [0.99, 0.95, 0.85, 0.80],
    "batch_size": [32, 64, 128],
    "hidden_size": [32, 64, 128]
}

def get_random_params():
    return {k: random.choice(v) for k, v in param_grid.items()}

def run_tuning_session(n_trials=20):
    best_score = -float("inf")
    best_params = None
    
    print(f"--- Starting Random Search ({n_trials} trials) ---")
    
    # running parameter grid with random search
    for i in range(1, n_trials + 1): 
        # picking random parameters 
        params = get_random_params()
        print(f"\nTrial {i}/{n_trials} testing: {params}")
        final_avg_score = train_dqn(
            n_episodes=800,
            lr=params["lr"],
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            hidden_size=params["hidden_size"],
            filename=f"tuning_trial{i}",
            model_filename=f"tuning_trial_{i}.pth"
        )
        
        print(f"--> Score: {final_avg_score:.2f}")
        # track the winner
        if final_avg_score > best_score:
            best_score = final_avg_score
            best_params = params
            print(f"*** NEW BEST FOUND! ***")
        
    print("\n===========================")
    print("Tuning Complete.")
    print(f"Best Score: {best_score}")
    print(f"Best Parameters: {best_params}")
    print("\n===========================")
        
# running training with random-search 
# based hyperparameter tuning        
if __name__ == "__main__":
    run_tuning_session()        
               