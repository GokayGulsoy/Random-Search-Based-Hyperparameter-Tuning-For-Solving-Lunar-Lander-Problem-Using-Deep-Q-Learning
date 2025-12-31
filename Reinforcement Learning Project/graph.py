import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_results(filename="scores.npy"):
    # plotting reward
    input_file = f"scores/{filename}.npy"
    output_image = f"solved_model_plots/plot_{filename}.png"
    
    # loading data
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run main.py or tune.py first.")
        return

    scores = np.load(input_file)
    # calculate moving average (for the smooth file)
    window_size = 100
    if len(scores) < window_size:
        print(f"Warning: Not enough data for moving average (Need 100, got {len(scores)})")
        moving_avg = scores
    else:    
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode="valid")
    
    plt.figure(figsize=(10, 5))
    # raw scores 
    plt.plot(scores, label="Episode Score", color="lightblue", alpha=0.5)
    # moving average 
    plt.plot(np.arange(len(scores)-len(moving_avg)+1, len(scores)+1), moving_avg, 
             label="100-Episode Moving Average", color="blue", linewidth=2)
    
    # formatting
    plt.axhline(y=200, color="r", linestyle="--", label="Solved Threshold (200)")
    plt.ylabel("Reward")
    plt.xlabel("Episode #")
    plt.title(f"DQN Training Progress: LunarLander-v3: {filename}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # save & show
    plt.savefig(output_image, dpi=300)
    print(f"Reward Plot saved as {output_image}")
    plt.show()

    # plotting loss
    input_file = f"scores/{filename}_loss.npy"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    losses = np.load(input_file)
    # calculating moving average
    if len(losses) >= window_size:
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode="valid")
    else:
        moving_avg = losses
        
    plt.figure(figsize=(10, 5))
    # raw losses
    plt.plot(losses, label="Raw Loss", color="lightcoral", alpha=0.4)
    
    # smoothed loss
    plt.plot(np.arange(len(losses)-len(moving_avg)+1, len(losses)+1), moving_avg,
             label="100-Episode Moving Average", color="darkred", linewidth=2)
    
    plt.ylabel("Loss (MSE)")
    plt.xlabel("Episode #")
    plt.title(f"Training Loss: {filename}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_image = f"solved_model_plots/plot_loss_{filename}.png"
    plt.savefig(output_image, dpi=300)
    print(f"Loss plot saved as {output_image}")
    plt.show()

# plot graph    
if __name__ == "__main__":
    # file name to be plotted can be provided as a 
    # parameter to plot_training_results function
    plot_training_results("tuning_trial1")
