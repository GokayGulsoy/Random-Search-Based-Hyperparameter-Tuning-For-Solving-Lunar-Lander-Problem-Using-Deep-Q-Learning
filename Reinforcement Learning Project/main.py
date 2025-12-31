import gymnasium as gym
import torch
import numpy as np
import os
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from agent import Agent

def train_dqn(n_episodes=1000, lr=1e-4, gamma=0.99, batch_size=64, hidden_size=64, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, filename="scores", model_filename="solved_model.pth"):
    writer = SummaryWriter(log_dir="runs/LunarLander")
    env = gym.make("LunarLander-v3")
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size=state_size, action_size=action_size, seed=0, gamma=gamma, lr=lr, hidden_size=hidden_size, batch_size=batch_size)
    
    scores = []
    losses = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    # create directory for saving models
    os.makedirs("models", exist_ok=True)
    print(f"Training on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    for i_episode in range(1, n_episodes+1):
        state, info = env.reset()
        score = 0
        total_loss = 0
        avg_loss = 0
        learn_steps = 0
        
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            loss_val = agent.step(state, action, reward, next_state, done)            
            
            if loss_val is not None:
                total_loss += loss_val
                learn_steps += 1
            
            state = next_state
            score += reward
            if done:
                break 
            
        # logging to tensorboard
        eps = max(eps_end, eps_decay*eps)
        
        writer.add_scalar("Reward/Episode", score, i_episode)
        avg100 = np.mean(scores_window) if len(scores_window) > 0 else 0.0
        writer.add_scalar("Reward/Average100", avg100, i_episode)
        writer.add_scalar("Epsilon", eps, i_episode)

        if learn_steps > 0:
            avg_loss = total_loss / learn_steps
            writer.add_scalar("Loss/Average", avg_loss, i_episode)
            
        scores_window.append(score)
        scores.append(score)
        losses.append(avg_loss)
            
        current_avg = np.mean(scores_window) if len(scores_window) > 0 else 0.0
        print(f"\rEpisode {i_episode}\tAvg Score: {current_avg:.2f}", end="")
        
        if current_avg >= 240.0:
            print(f"\nEnvironment solved in {i_episode-100} episodes!")
            torch.save(agent.qnetwork_policy.state_dict(), f"models/{model_filename}")    


    env.close()
    writer.close()
    
    # save scores for graphing
    np.save(f"scores/{filename}.npy", np.array(scores))
    np.save(f"scores/{filename}_loss.npy", np.array(losses))
    print(f"\nTraining finished. Saved scores and losses for {filename}.")
    
    final_score = np.mean(scores_window) if len(scores_window) > 0 else 0.0
    return final_score


# run training
if __name__ == "__main__":
    train_dqn()

