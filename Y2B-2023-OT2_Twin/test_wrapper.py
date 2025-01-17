from ot2_gym_wrapper import OT2Env

# Initialize the environment
env = OT2Env()

# Reset the environment to start a new episode
observation = env.reset()
print(f"Initial Observation: {observation}")

# Run the environment for a few steps
for step in range(50):
    # Sample a random action
    action = env.action_space.sample()
    print(f"Step {step + 1}: Action Taken: {action}")

    # Take a step in the environment
    observation, reward, done, info = env.step(action)
    print(f"Observation: {observation}, Reward: {reward}, Done: {done}, Info: {info}")

    # Terminate if the episode is done
    if done:
        print("Episode finished!")
        break

# Close the environment
env.close()
