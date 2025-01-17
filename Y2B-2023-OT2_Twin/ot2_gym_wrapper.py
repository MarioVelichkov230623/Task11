import gym
from gym import spaces
import numpy as np
from sim_class import Simulation
import pybullet as p

class OT2Env(gym.Env):
    """
    Custom Gymnasium environment for the OT-2 pipette control using PyBullet.
    """
    def __init__(self, reward_threshold_1=0.05, reward_threshold_2=0.02, reward_bonus_1=1.0, reward_bonus_2=2.0, reward_fn=None, done_conditions=None):
        super(OT2Env, self).__init__()

        # Initialize other attributes
        self.previous_position = None
        self.last_action = None

        self.reward_threshold_1 = reward_threshold_1
        self.reward_threshold_2 = reward_threshold_2
        self.reward_bonus_1 = reward_bonus_1
        self.reward_bonus_2 = reward_bonus_2

        # Initialize simulation
        self.sim = Simulation(num_agents=1)

        # Define action space (continuous velocities in x, y, z)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)


        # Define observation space (position of the pipette tip in x, y, z)
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]),
                                            high=np.array([1.0, 1.0, 1.0]),
                                            dtype=np.float32)

        # Target position (can be set dynamically for each episode)
        self.target_position = np.array([0.0, 0.0, 0.0])

        # Reward function and done conditions (default or custom)
        self.reward_fn = reward_fn if reward_fn else self.default_reward_fn
        self.done_conditions = done_conditions if done_conditions else self.default_done_conditions


    def update_target_marker(self):
        # Create a new marker for the updated target position
        print(f"Creating target marker at: {self.target_position}")

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.01,
            rgbaColor=[1, 0, 0, 1]
        )
        new_marker_id = p.createMultiBody(
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.target_position
        )

        # Add the new marker to the list of markers
        self.target_marker_ids.append(new_marker_id)




    def reset(self):
        """Reset the environment to its initial state."""
        # Remove all previous markers
        if hasattr(self, 'target_marker_ids') and self.target_marker_ids:
            for marker_id in self.target_marker_ids:
                try:
                    p.removeBody(marker_id)
                    print(f"Removed marker ID: {marker_id}")
                except Exception as e:
                    print(f"Error removing marker {marker_id}: {e}")
            self.target_marker_ids = []
        else:
            self.target_marker_ids = []

        # Reset the simulation
        self.sim.reset()

        # Set target position within the working envelope bounds
        x_min, x_max = -0.187, 0.253
        y_min, y_max = -0.171, 0.22
        z_min, z_max = 0.169, 0.29

        # Randomize target position within the bounds
        self.target_position = [
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(z_min, z_max),
        ]
        print(f"Target Position: {self.target_position}")

        # Update the target marker for visualization
        self.update_target_marker()

        # Initialize position and action tracking
        self.current_position = self.sim.get_pipette_position(self.sim.robotIds[0])
        self.previous_position = self.current_position  # Set previous position
        self.last_action = [0, 0, 0]  # Initialize last action as zero vector

        return np.array(self.current_position, dtype=np.float32)





    def step(self, action):
        """Apply an action and advance the simulation."""
        # Clip the action to the defined action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Update previous position
        self.previous_position = self.current_position

        # Apply the action (velocities)
        self.sim.run([list(action) + [0]], num_steps=10)

        # Get the new pipette position
        self.current_position = self.sim.get_pipette_position(self.sim.robotIds[0])

        # Save the last action
        self.last_action = action

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




    # THIS IS PROBABLY THE ONE TO WORK WITH AND REFINE
    def default_reward_fn(self, current_position, target_position):
        """
        Simple reward function for guiding the agent toward a target position.

        Args:
            current_position (array): Current position of the pipette tip.
            target_position (array): Target position.

        Returns:
            float: Computed reward.
        """
        # Convert positions to numpy arrays
        current_position = np.array(current_position)
        target_position = np.array(target_position)
        previous_position = np.array(self.previous_position)

        # Calculate distances
        distance_to_target = np.linalg.norm(current_position - target_position)
        previous_distance_to_target = np.linalg.norm(previous_position - target_position)

        # Initialize reward
        reward = 0

        # 1. Award for getting closer to the target
        if distance_to_target < previous_distance_to_target:
            reward += 1.0  # Reward for progress

        # 2. Penalize for getting farther from the target
        elif distance_to_target > previous_distance_to_target:
            reward -= 1.5  # Penalty for moving away

        # 3. Encourage exploration
        # Penalize small movements that don't change the position significantly
        movement = np.linalg.norm(current_position - previous_position)
        if movement < 0.01:  # Threshold for small movements
            reward -= 0.2  # Penalize stagnation or insufficient exploration

        # 4. Discourage getting stuck at a local minimum
        # Add a time penalty for staying near the same position
        time_penalty_threshold = 0.05  # Threshold for "near the same spot"
        if np.linalg.norm(current_position - previous_position) < time_penalty_threshold:
            reward -= 0.01  # Penalize remaining in the same general area

        # Bonus for reaching the target
        target_threshold = 0.01  # Define "close enough" to the target
        if distance_to_target < target_threshold:
            reward += 15.0  # Large bonus for precision

        return reward






    # def default_done_conditions(self, current_position, target_position):
    #     current_position = np.array(current_position)
    #     target_position = np.array(target_position)
    #     distance = np.linalg.norm(current_position - target_position)

    #     # Increase threshold to allow episodes to terminate when "close enough"
    #     threshold = 0.05  # 5 cm
    #     return distance < threshold

    def default_done_conditions(self, current_position, target_position):
        current_position = np.array(current_position)
        target_position = np.array(target_position)

        # Check if within precision threshold
        distance = np.linalg.norm(current_position - target_position)
        precision_threshold = 0.05  # 1mm
        return distance < precision_threshold




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
