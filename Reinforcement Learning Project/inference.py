import torch
import warnings

# suppressing warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", message="pkg_resource is deprecated")

from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from agent import Agent

def record_agent_solution(model_path="models/solved_model.pth", video_folder="videos", hidden_size=64):
    """
    Loads a trained model and records one episode of it playing
    """
    # setting up environment
    # render_mode="rgb_array" is required for video recording
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # adding video wrapper
    # this automatically captures and saves an MP4
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda x: True,
        name_prefix="smart_lander"
    )
    
    # loading the agent and weights 
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size, seed=0, hidden_size=hidden_size)
    
    # checking if model exists
    try:
        agent.qnetwork_policy.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: No model found at {model_path}. Run main.py or tune.py first!")
    
    agent.qnetwork_policy.eval() # set to evaluation mode
    # running the episode
    state, info = env.reset()
    done = False
    total_reward = 0
    
    print("Recording video...")
    while not done:
        # eps=0.0 means 'Pure Exploitation' (always choose the best action)
        action = agent.act(state, eps=0.0)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
    env.close()
    print(f"Simulation finished. Total Reward: {total_reward:.2f}")
    print(f"Video saved in folder: {video_folder}")
    

# running inference
if __name__ == "__main__":
    # NOTE: update the parameters for the model that will be used 
    # for inference and video recording (use corresponding .pth model file
    # from models folder and that model's hidden size hyperparameter)
    record_agent_solution(model_path="models/tuning_trial_3.pth", hidden_size=128)    
    