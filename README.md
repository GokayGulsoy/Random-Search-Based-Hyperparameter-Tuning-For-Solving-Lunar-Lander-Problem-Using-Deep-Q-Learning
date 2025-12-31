# Random-Search-Based-Hyperparameter-Tuning-For-Solving-Lunar-Lander-Problem-Using-Deep-Q-Learning

This project implements a Random Search-based hyperparameter-tuning strategy to solve the "LunarLander-v3" environment from Gynasium. It selects best 15 configurations yielded +240 rewards and performs comparative analysis on them.

<p align="center">
  <img src="lunar_lander_trained.gif" alt="GIF representing LunarLander landing">
</p> 

## Table of Contents 

- [Deep Q-Network (DQN) for LunarLander-v3](#deep-Q-network-for-solving-lunarlander-v3)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
  - [LunarLander-v3 Environment](#lunarlander-v3-environment) 
  - [Usage](#usage)
    - [Running Tuning Trials](#training-the-model)
    - [Running Inference on Model](#running-inference)
    - [Plotting Graphs for Selected Run](#plotting-graphs)
  - [Hyperparameters Used In Random Search](#hyperparameters-used-in-random-search)
  - [Hyperparameters Not Used In Random Search](#hyperparameters-not-used-in-random-search)
  - [Training Details](#training-details)
  - [Results](#results)
  - [Research Paper](url_to_reserach_parper)


## Introduction

The goal of this project is to train an agent using DQN algorithm to land the spacecraft safely and as fuel-efficiently as possible on a designated landing pad in the "LunarLander-v3" environment. The environment is a classical problem based on Box2D physics, where the agent must optimize its thrusters to achieve stable landing.

## Installation

### Prerequisites

Ensure you have Python installed. Project is tested with Python version 3.11.6, you can set up the required dependencies using:

```bash
# Clone the repository
git clone https://github.com/GokayGulsoy/Random-Search-Based-Hyperparameter-Tuning-For-Solving-Lunar-Lander-Problem-Using-Deep-Q-Learning.git
cd lunar-lander-project

# Create a virtual environment 
python -m venv venv
source venv/bin/activate # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## LunarLander-v3 Environment

The LunarLander-v3 environment is a reinforcement learning task where an agent controls a lander to safely land on a designated pad. The agent receives rewards based on its landing accuracy and penalties for fuel usage and crashes.

### Action Space

- **Discrete (4 actions)**
  - `0`: Do nothing
  - `1`: Fire left orientation engine
  - `2`: Fire main engine
  - `3`: Fire right orientation engine

 ### Observation Space
 - **Box(8, float32)**: Contains position, velocity, angle, angular velocity, and landing leg contact status.

### Rewards & Penalties

- Reward increases for smooth and centered landing.
- Penalties for excessive tilt, high speed, unnecessary thruster use.
- +10 points per leg in contact in contact with the ground, -100 for crahsing, +100 for a successful landing.

### Episode Termination

- The lander crashed.
- The lander gets outside of the viewport.
- The lander stops moving. 

## Usage

### Running Tuning Trials

To run random search-based hyperparameter-tuning, execute the command:

```bash
python tune.py
```

The training progress can be monitored using Tensorboard with command:

```bash
tensorboard --logdir=runs
```

Running tune.py creates model files for the runs which surpasses score of 240 and saves those models under the models directory. Scores for each run are recorded under the scores directory regardless of the score achieved for plotting and analysis.

## Running inference on Model

To test the trained model, set the model path for the chosen model that exceeds 240 reward under the model directory and hidden_size shold match the hidden_layer selected for the specific run which is logged to termial after running `tune.py` for code line `record_agent_solution(model_path="models/tuning_trial_3.pth", hidden_size=128)` located at the bottom of the `inference.py`. This logs the inference score to terminal and generate a landing simulation video under the videos directory.

## Plotting Graphs for Selected Runs

To visualize the reward and loss as graphs, choose the `.npy` file created under the scores directory after running `tune.py`. Then, provide the `plot_training_results("tuning_trial_{trial_number}")` code line located at the bottom of the `graph.py` with the name of the `{tuning_trial_num}.npy` score file excluding `.npy` extension, that will generate graphs showing how reward and loss changed over the episodes for that run under the `solved_model_plots` directory.  

## Hyperparameters Used In Random Search

- **Learning Rate:** \[0.005, 0.001, 0.0005, 0.0001\]
- **Discount Factor (Gamma):** \[0.99, 0.95, 0.85, 0.80\]
- **Batch Size:** \[32, 64, 128\]
- **Hidden Size:** \[32, 64, 128\]

## Hyperparameters Not Used In Random Search
- **Episode:** 800
- **Replay Buffer Size:** 100,000
- **Target Network Update:** Soft Updates, TAU = 0.005
- **Exploration Strategy:** ε-greedy (ε decays from 1.0 to 0.01 with a 0.995 decay factor per episode)
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error

## Training Details

- **Episode:** 800
- **Convergence Reward:** 200+ (environment considered solved)
- **Graphics Processing Unit:** GPU RTX 2060 (6GB VRAM)
- **Processor:** Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
- **RAM:** 16GB

## Experimetal Results

15 experiments are done in total by selecting best 15 configurations found by random search-based hyperparameter-tuning. Results of each experiment with 20 trials are provided under root project directory with naming convention `Experimental_Results_{experiment_number}.txt`. `Experimental_Results` directory under project root directory contains the experimental results which contains `model` subdirectory listing solved model files for each experiment. If directory for specific run is empty it means that no solved model was found for that experiment. Each `scores_run_{experiment_number}` directory contains the scores for rewards and losses for 20 tuning trials in given the specific experiment number. The `solved_model_plots` directory contains the `{experiment_number_in_text}_run` subdirectories each contais the plots for reward and loss graphs for runs that achieved 240+ reward. The `videos` directory contains the `run_{experiment_number}` folder that contains the .mp4 files for each solved model that simulates lunar lander landing.    

## Research Paper

Paper for project can be found here: [Research Paper](research_paper_link)
