import gym
from gym import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    """
    Custom Gymnasium environment for the OT-2 pipette control using PyBullet.
    """
    def __init__(self, reward_fn=None, done_conditions=None):
        super(OT2Env, self).__init__()

        # Initialize simulation
        self.sim = Simulation(num_agents=1)

        # Define action space (continuous velocities in x, y, z)
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32)

        # Define observation space (position of the pipette tip in x, y, z)
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]),
                                            high=np.array([1.0, 1.0, 1.0]),
                                            dtype=np.float32)

        # Target position (can be set dynamically for each episode)
        self.target_position = np.array([0.0, 0.0, 0.0])

        # Reward function and done conditions (default or custom)
        self.reward_fn = reward_fn if reward_fn else self.default_reward_fn
        self.done_conditions = done_conditions if done_conditions else self.default_done_conditions

    def reset(self):
        """Reset the environment to its initial state."""
        self.sim.reset()

        # Randomize the target position within the working envelope
        self.target_position = np.random.uniform(-0.8, 0.8, size=(3,))
        
        # Print the target position for debugging
        print(f"Target Position: {self.target_position}")

        # Get the pipette's initial position
        self.current_position = self.sim.get_pipette_position(self.sim.robotIds[0])

        return np.array(self.current_position, dtype=np.float32)





    def step(self, action):
        """Apply an action and advance the simulation."""
        # Clip the action to the defined action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply the action (velocities)
        self.sim.run([list(action) + [0]], num_steps=10)

        # Get the new pipette position
        self.current_position = self.sim.get_pipette_position(self.sim.robotIds[0])

        # Calculate reward
        reward = self.reward_fn(self.current_position, self.target_position)

        # Check if the episode is done
        done = self.done_conditions(self.current_position, self.target_position)

        # Additional info (optional, e.g., logging data)
        info = {}

        return np.array(self.current_position, dtype=np.float32), reward, done, info

    def render(self, mode="human"):
        """Render the environment (e.g., using PyBullet's GUI)."""
        pass  # Rendering is handled by PyBullet if needed

    def close(self):
        """Close the simulation and clean up resources."""
        self.sim.close()

    def default_reward_fn(self, current_position, target_position):
        """Default reward function: negative distance to the target."""
        distance = np.linalg.norm(current_position - target_position)
        return -distance  # Reward is higher when closer to the target

    def default_done_conditions(self, current_position, target_position):
        """Default done condition: close to target or out of bounds."""
        distance = np.linalg.norm(current_position - target_position)
        if distance < 0.05:  # Consider "done" if within 5 cm of the target
            return True
        if not self.observation_space.contains(current_position):
            return True  # Done if out of bounds
        return False

# Example usage of the environment
if __name__ == "__main__":
    env = OT2Env()
    obs = env.reset()
    print("Initial Observation:", obs)

    for _ in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
        if done:
            break

    env.close()
