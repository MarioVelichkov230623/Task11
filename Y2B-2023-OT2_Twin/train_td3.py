import subprocess
subprocess.run(["pip", "install", "--upgrade", "typing_extensions"])
import argparse
import gym
import json
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from wandb.integration.sb3 import WandbCallback
import wandb
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from ot2_gym_wrapper import OT2Env  # Import your custom environment
from stable_baselines3.common.callbacks import BaseCallback
from clearml import Task

# Use the appropriate project name and task name (if you are in the first group in Dean's mentor group, use the project name 'Mentor Group D/Group 1')
# It can also be helpful to include the hyperparameters in the task name
task = Task.init(project_name='Mentor Group J/Group 1', task_name='Mario_TD3_Training_1')
#copy these lines exactly as they are
#setting the base docker image

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

SEED = 42
np.random.seed(SEED)
# Parse command-line arguments
parser = argparse.ArgumentParser(description="TD3 Training Script")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the TD3 model")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--n_steps", type=int, default=2000000, help="Number of training steps")
parser.add_argument("--buffer_size", type=int, default=1000000, help="Replay buffer size")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--tau", type=float, default=0.005, help="Target smoothing coefficient")
parser.add_argument("--policy_delay", type=int, default=2, help="Policy update delay")
parser.add_argument("--action_noise_sigma", type=float, default=0.2, help="Sigma for action noise")
args = parser.parse_args()

# Load Wandb API key from a configuration file
with open("wandbkey.json", "r") as file:
    config = json.load(file)

wandb_key = config.get("WANDB_API_KEY")

if wandb_key is None:
    raise ValueError("Wandb API key not found in config.json.")

# Initialize Wandb project
wandb.login(key=wandb_key)
wandb.init(
    project="OT2_TD3_Local_Test",
    sync_tensorboard=False,  # Sync with TensorBoard
    config=vars(args)  # Log all hyperparameters to wandb
)

# Initialize the environment
env = OT2Env()
env.seed(SEED)
# Define action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.action_noise_sigma * np.ones(n_actions))

set_random_seed(SEED)
# Initialize the TD3 model
model = TD3(
    "MlpPolicy",              # Policy type
    env,                      # The Gym environment
    action_noise=action_noise, 
    learning_rate=args.learning_rate,       # Learning rate
    buffer_size=args.buffer_size,      # Replay buffer size
    batch_size=args.batch_size,           # Batch size
    tau=args.tau,                # Target smoothing coefficient
    gamma=args.gamma,               # Discount factor
    train_freq=args.n_steps,          # Training frequency
    policy_delay=args.policy_delay,           # Delayed policy updates
    verbose=1,                # Verbosity level
    tensorboard_log="./td3_tensorboard_logs/"
)

# Custom Callback for Additional Logging
class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Access the wrapped environment
        original_env = self.training_env.envs[0].unwrapped  # Access the unwrapped OT2Env

        # Log custom metrics every step
        obs = self.locals["new_obs"]
        wandb.log({
            "reward": self.locals["rewards"].item(),
            "distance_to_target": np.linalg.norm(obs - original_env.target_position),
        })
        return True


# Wandb callback for automatic logging
wandb_callback = WandbCallback(
    model_save_path="./models",
    gradient_save_freq=100,
    verbose=2
)

# Combine callbacks
custom_callback = CustomWandbCallback()

# Train the model

# Train the model for 300,000 timesteps, saving every 100,000 timesteps
total_timesteps = 300_000
save_interval = 100_000

# Calculate the number of increments
increments = total_timesteps // save_interval

for i in range(increments):
    # Train for 100,000 timesteps
    model.learn(
        total_timesteps=save_interval,
        callback=[wandb_callback, custom_callback],  # Log metrics and custom data
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name=f"td3_model_lr{args.learning_rate}_bs{args.batch_size}_run{wandb.run.id}"
    )
    
    # Save the model at the end of this increment
    model.save(f"models/td3_model_lr{args.learning_rate}_bs{args.batch_size}_run{wandb.run.id}_step{(i + 1) * save_interval}.zip")



# Close the environment
env.close()

# Finish Wandb logging
wandb.finish()
