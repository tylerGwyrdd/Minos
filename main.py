import six_DoF_simulator as simulator
import numpy as np
import matplotlib.pyplot as plt
import math

# ODE solver, simplest one, error prone
def forward_euler(state, derivatives, dt):
    dx = [dt * der for der in derivatives]
    new_state = [s + a for s, a in zip(state, dx)]
    return new_state

# ------------------------------------------
#  ----------- Simulating  ----------------
# ------------------------------------------

# ----------- system definitions ----------------

# system definitions
params = {
    "rigging_angle_Tau": 1.3,  # rigging angle: angle between body fixed frame and parafoil chord
    "glide_angle": 0,  # angle between earth horizontal and velocity of airflow over parafoil
    "Rp": 0.26,  # distance between parafoil and the center of mass for the system
    "Rc": 0.1,  # distance between CoM and payload weight location
    "S": 3.14,  # surface area of parafoil
    "I_zz": 0.0123,  # moment of inertia about z axis
    "m": 5,  # mass of system
    "flap_deflection": 0,  # deflection of flap on parafoil
    "b": 0.5,  # mean chord length
}

# ----------- inital state definitions ----------------
# inertial frame positions
p_n = 0 # north position (to the left)
p_u = 1000 # height position
p_inital = np.array([p_n,p_u]).reshape(2,1) # inertial frame position

# Veloscities in body fixed frame. 
U = 0.5 # veloscity forward
W = 0.5 # velocity downwards direction
v_inital = np.array([U,W]).reshape(2,1)

# pitch in body fixed frame
theta = math.radians(30) # pitch angle
q = 0.0 # angular velocity


# inital state
inital_state = [p_inital, v_inital, theta, q]

# -------------- simulation ----------------

data = []

# Time step
dt = 0.1
t = 0

# Run simulation
sim = simulator.ParafoilSimulation(params, inital_state)

state = inital_state
for i in range(20):
    data.append([t,sim.get_state()])
    der = sim.derivatives()
    sim.add_row_to_table(t)
    new_state = sim.update_state(forward_euler, dt)
    t += dt
    state = new_state
    
# might not be working because consideration for negitive angle of attacks are not considered. 


sim.print_table()
position = [lst[1][0] for lst in data]
x_pos = [lst[0] for lst in position]
y_pos = [lst[1] for lst in position]
plt.plot(x_pos, y_pos)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Parafoil simulation')
# Show the plot
plt.show()

