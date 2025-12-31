# Random-Search-Based-Hyperparameter-Tuning-For-Solving-Lunar-Lander-Problem-Using-Deep-Q-Learning

This project implements a Random Search-based hyperparameter-tuning strategy to solve the "LunarLander-v3" environment from Gynasium. It selects best 15 configurations yielded +240 rewards and performs comparative analysis on them.

<p align="center>
  <img src="lunar_lander_trained.gif" alt="GIF representing LunarLander landing"
</p> 

## Table of Contents 

- [Deep Q-Network (DQN) for LunarLander-v3](#deep-Q-network-for-solving-lunarlander-v3)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
  - [LunarLander-v3 Environment](#lunarlander-v3-environment) 
  - [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Running Inference on Model](#running-inference)
  - [Hyperparameters](#hyperparameters)
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
git clone https://github.com/wtcherr/lunar-lander-dqn.git
cd lunar-lander-project
```









