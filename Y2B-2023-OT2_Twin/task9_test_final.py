import time
import pybullet as p
from sim_class import Simulation

# Initialize the simulation
sim = Simulation(num_agents=1)

# Define velocities for the 8 corners of the working envelope
corners = [
    [0.5, 0.5, 0.5, 0],  # Top-Front-Right
    [0.5, 0.5, -0.5, 0],  # Bottom-Front-Right
    [0.5, -0.5, 0.5, 0],  # Top-Back-Right
    [0.5, -0.5, -0.5, 0],  # Bottom-Back-Right
    [-0.5, 0.5, 0.5, 0],  # Top-Front-Left
    [-0.5, 0.5, -0.5, 0],  # Bottom-Front-Left
    [-0.5, -0.5, 0.5, 0],  # Top-Back-Left
    [-0.5, -0.5, -0.5, 0]  # Bottom-Back-Left
]

# Record pipette positions
positions = []

for i, velocities in enumerate(corners):
    print(f"Moving to corner {i+1}: {velocities[:3]}")
    
    # Move toward the corner
    sim.run([velocities], num_steps=400)

    # Stop the robot
    sim.run([[0, 0, 0, 0]], num_steps=50)

    # Record the current position
    current_position = sim.get_pipette_position(sim.robotIds[0])
    current_position = [round(coord, 3) for coord in current_position]
    positions.append(current_position)
    print(f"Corner {i+1} position: {current_position}")

    # Add a marker at the current corner
    p.addUserDebugLine(
        [current_position[0], current_position[1], 0],  # Start at base
        current_position,  # End at corner
        [0, 1, 0],  # Green color
        2  # Line width
    )

    time.sleep(1.5)  # Pause for visualization

# End the simulation
sim.close()

# Print all recorded positions
print("Recorded positions of the full working envelope:")
for i, pos in enumerate(positions):
    print(f"Corner {i+1}: {pos}")
