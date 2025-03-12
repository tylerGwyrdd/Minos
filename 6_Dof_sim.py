# 6 dof model
import numpy as np
import math
import matplotlib.pyplot as plt

# ----------- environment definitions -----------

p_density =  1.293 # (kg/m3) density of air at sea level

class ParafoilSimulation_6Dof:
    def __init__(self,params,state):
        self.set_state(state)
        self.set_system_params(params)
        self.table_rows = []
        # calculate the derivatives
        self.derivatives()
    
    def set_state(self, state):
        self.p = state[0] # position in inertial frame
        self.v = state[1] # velocity in body fixed frame
        self.eulers = state[2] # euler angles of body in inertal frame
        self.omega = state[3] # angular velocities in body fixed frame
        self.flap_l,self.flap_r = state[4] # flap deflection angle
        
        # calculate the flap deflection angles
        self.delta_a = self.flap_l - self.flap_r # asymmetric flap deflection
        self.delta_s = 0.5*(self.flap_l + self.flap_r) # symmetric flap deflection
        
        # calculate the local airspeed velocity
        self.Va = np.linalg.norm(self.v)

        # we also need to calculate the AoA
        self.angle_of_attack = np.arctan(self.v[1].item()/self.v[0].item())

        # also dont forget the sideslip angle...
        self.sideslip_angle = np.arcsin(-self.v[2].item()/np.linalg.norm(self.v))
        # update the transformations:
        self.update_kinematic_transforations()
        
    def get_state(self):
        return [self.p, self.v, self.theta, self.q]

    def update_kinematic_transforations(self):
        psi = self.eulers[0]
        theta = self.eulers[1]
        phi = self.eulers[2]

        self.CDM =  np.array([
        [np.cos(theta) * np.cos(psi), np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
        [np.cos(theta) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
        [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]
        ])

        # angular veloscities to euler angular rates
        # Ensure theta is not 90° or 270° to avoid division by zero
        if np.isclose(np.cos(theta), 0):
            raise ValueError("Theta cannot be ±90° (or π/2 radians) due to singularity.")

        self.angular_T = (1 / np.cos(theta)) * np.array([
            [0, np.sin(phi), np.cos(phi)],
            [0, np.cos(phi) * np.cos(theta), -np.sin(phi) * np.cos(theta)],
            [np.cos(theta), np.sin(phi), np.cos(phi) * np.sin(theta)]
        ])
    
    def areo_to_body(self,vector, inverse = False):
        

        return

    def body_to_inertial(self,vector, inverse = False):
        R_bi = self.CDM
        if inverse:
            R_bi = R_bi.T
        rotated_vector = np.dot(R_bi,vector)
        return rotated_vector
    
    def set_system_params(self,system_params):
        # areodynamic parameters
        self.glide_angle = math.radians(system_params["glide_angle"])
        self.flap_deflection = math.radians(system_params["flap_deflection"])
        self.S = system_params["S"] # surface area of parafoil
        self.c = system_params["c"] # mean chord length
        self.AR = system_params["AR"] # aspect ratio
        self.t = system_params["t"] # thickness of the parafoil
        self.b = system_params["b"] # wingspan of the parafoil

        # system parameters
        self.m = system_params["m"] # mass of entire system
        self.Rp = system_params["Rp"] # distance between parafoil and the center of mass for the system
        self.I_zz = system_params["I_zz"] # moment of inertia about the z axis
        
    def calculate_areo_force_coeff(self):
        # for lifting
        Clo = 0.1 # lift coefficient at zero lift
        Cla = 2 # lift coefficient changing due to angle of incidance

        Cl = Clo + Cla * self.angle_of_attack

        # for drag
        Cdo = 0.1 # drag coefficient at zero lift
        Cda = 2 # drag coefficient changing due to angle of incidance

        Cd = Cdo + Cda * self.angle_of_attack

        # for side force
        CyB = 0.1
        Cy = CyB * self.sideslip_angle

        return [Cl, Cd, Cy]
    
    def calculate_areo_moment_coeff(self):
        # for moment
        Cmo = 0.08 
        Cm_pitch = -0.2 * self.angle_of_attack
        Cm = Cmo + Cm_pitch
    
    def calculate_aero_forces(self):
        C_L, C_D, C_Y = self.calculate_areo_force_coeff()

        #lift force
        F_l = 0.5 * p_density * self.Va**2 * self.S * C_L

        #drag force
        F_d = 0.5 * p_density * self.Va**2 * self.S * C_D

        #side force
        F_y = 0.5 * p_density * self.Va**2 * self.S * C_Y

        F_aero = np.array([[-F_d],[F_y],-[F_l]])

        #rotate forces to the body frame
        F_areo = self.areo_to_body(F_aero)
        return F_areo

    def calculate_aero_moments(self):
        C_m,C_l,C_n = self.calculate_areo_moment_coeff()

        # pitching moment
        M = 0.5 * p_density * self.Va**2 * self.S * self.b * C_m

        # rolling moment
        L = 0.5 * p_density * self.Va**2 * self.S * self.b * C_l

        # yawing moment
        N = 0.5 * p_density * self.Va**2 * self.S * self.b * C_n

        M_aero = np.array([[L],[M],[N]])

        return M_aero
    
    def calculate_derivitives(self,areo_force, moments):
        # calculate the aero forces
        F_aero = self.calculate_aero_forces()

        # calculate the gravity force
        F_g = np.array([[0],[0],[-self.m * 9.81]])

        # calculate the total force
        F_total = F_aero + F_g

        
        # calculate the aero moments
        M_aero = self.calculate_aero_moments()

        # calculate the payload moment
        M_p = np.array([[0],[0],[0]])

        # calculate the total moment
        M_total = M_aero + M_p

        # calculate the acceleration
        a = F_total / self.m

        # calculate the angular acceleration
        alpha = np.dot(self.angular_T,M_total) / self.I

        return [a,alpha]

    def derivatives(self):
        a,alpha = self.calculate_derivitives()

        # calculate the velocity


