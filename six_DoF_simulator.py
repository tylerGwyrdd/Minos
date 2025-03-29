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

    def body_to_inertial(self,vector, inverse = False):
        R_bi = self.CDM
        if inverse:
            R_bi = R_bi.T
        rotated_vector = np.dot(R_bi,vector)
        return rotated_vector
    
    def body_to_wind(self,vector, inverse = False):
        R_bw = np.array([np.cos(self.angle_of_attack)*np.cos(self.sideslip_angle), -np.sin(self.sideslip_angle), np.cos(self.sideslip_angle)*np.sin(self.angle_of_attack)],
                        [np.sin(self.angle_of_attack)*np.cos(self.sideslip_angle), np.cos(self.angle_of_attack), np.sin(self.angle_of_attack)*np.sin(self.sideslip_angle)],
                        [-np.sin(self.angle_of_attack), 0, np.cos(self.angle_of_attack)])
        if inverse:
            R_bw = R_bw.T
        rotated_vector = np.dot(R_bw,vector)
        return rotated_vector

    def set_system_params(self,system_params):
        
        # areodynamic parameters
        self.S = system_params["S"] # surface area of parafoil
        self.c = system_params["c"] # mean chord length
        self.AR = system_params["AR"] # aspect ratio
        self.t = system_params["t"] # thickness of the parafoil
        self.b = system_params["b"] # wingspan of the parafoil
        self.rigging_angle = system_params["rigging_angle"] # rigging angle: angle between body fixed frame and parafoil chord
        
        # system parameters
        self.parafoil_mass = system_params["parafoil_mass"] # mass of parafoil
        self.payload_mass = system_params["payload_mass"] # mass of payload
        self.Rlc = system_params["Rlc"] # distance between payload CoM and connection point to parafoil
        self.Rpc = system_params["Rpc"] # distance between parafoil CoM and connection point to parafoil
        
        # calculate system parameters
        self.m = self.parafoil_mass + self.payload_mass # mass of entire system
        
        self.Rp = system_params["Rp"] # distance between parafoil and the center of mass for the system
        
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
        # for rolling
        Clo = 0.08 # coefficient at zero lift
        Cl_alpha = -0.2 #  coefficient due to angle of incidance
        Cl = Clo + Cl_alpha * self.angle_of_attack

        # for pitching
        Cmo = 0.08 # coefficient at zero lift
        Cm_alpha = -0.2 # coefficient due to angle of incidance
        Cm_q = -0.2 # coefficient due to pitch rate
        Cm = Cmo + Cm_alpha * self.angle_of_attack + Cm_q * self.c/(2*self.Va) * self.omega[1]

        # for yawing
        CnB = 0.08 # coefficient due to sideslip angle
        Cn_asym = -0.2 # coefficient due asymmetric flap deflection
        Cn_p = -0.2 # coefficient due to roll rate
        Cn_r = -0.2 # coefficient due to yaw rate
        Cn = CnB * self.sideslip_angle + Cn_asym * self.delta_a
        Cn += Cn_p * self.c/(2*self.Va) * self.omega[0] + Cn_r * self.c/(2*self.Va) * self.omega[2]
        
        return [Cl, Cm, Cn]
    
    def calculate_aero_forces(self):
        C_L, C_D, C_Y = self.calculate_areo_force_coeff()

        #lift force
        F_l = 0.5 * p_density * self.Va**2 * self.S * C_L

        #drag force
        F_d = 0.5 * p_density * self.Va**2 * self.S * C_D

        #side force
        F_y = 0.5 * p_density * self.Va**2 * self.S * C_Y

        F_aero_A = np.array([[-F_d],[F_y],-[F_l]])

        #rotate forces to the body frame
        F_areo = self.areo_to_body(F_aero_A)

        return F_areo

    def calculate_aero_moments(self):
        C_m,C_l,C_n = self.calculate_areo_moment_coeff()

        # rolling moment
        L = 0.5 * p_density * self.Va**2 * self.S * self.b * C_l

        # pitching moment
        M = 0.5 * p_density * self.Va**2 * self.S * self.b * C_m

        # yawing moment
        N = 0.5 * p_density * self.Va**2 * self.S * self.b * C_n

        M_aero_A = np.array([L,M,N])

        # rotate moments to the body frame
        M_aero = self.areo_to_body(M_aero_A)

        return M_aero
    
    def calculate_derivitives(self,areo_force, moments):
        # calculate the aero forces
        F_aero = self.calculate_aero_forces()

        # calculate the gravity force
        F_g = np.array([[0],[0],[-self.m * 9.81]])

        # calculate the total force
        F_total = F_aero + F_g

        # calculate the acceleration
        acc = F_total / self.m


        # calculate the aero moments
        M_aero = self.calculate_aero_moments()

        # calculate the payload moment
        M_p = np.array([[0],[0],[0]])

        # calculate the areo force moment
        M_f_areo = np.cross([self.Rp,0,0],F_aero)

        # calculate the total moment
        M_total = M_aero + M_p + np.dot(self.angular_T,M_aero) + M_f_areo

        # calculate the angular acceleration
        angular_acc = np.dot(self.angular_T,M_total) / self.I

        return [acc,angular_acc]

    def calculate_moments_of_inertia(self):
        """
        Computes the moments of inertia of system in the body frame.
        
        Parameters:
            R: 3x3 numpy array
                Rotation matrix from parafoil frame to body frame.
        
        Returns:
            I_rotated: 3x3 numpy array
                Inertia tensor in the rotated frame.
        """
        # Principle moments of inertia of the parafoil
        Ix_parafoil = 1.0
        Iy_parafoil = 2.0
        Iz_parafoil = 3.0
        # Rotation matrix from parafoil frame to body frame
        R = np.array([
            [np.cos(self.rigging_angle), 0, np.sin(self.rigging_angle)],
            [0, 1, 0],
            [-np.sin(self.rigging_angle), 0, np.cos(self.rigging_angle)]
        ])

        # Principle inertia tensor
        I_parafoil = np.diag([Ix_parafoil, Iy_parafoil, Iz_parafoil])

        # Rotate into the new frame
        I_rotated_parafoil = R @ I_parafoil @ R.T

        # ___________ payload moment of inertia _____________

        # define the inertia tensor of the payload, atm assume to be cylindrical
        Ix_payload = 0.1
        Iy_payload = 0.1
        Iz_payload = 0.1

        # principle inertia tensor of the payload
        I_payload = np.diag([Ix_payload, Iy_payload, Iz_payload])

        # ___________ Inertias at coM  _____________
        # calculate the inertia tensor at the center of mass
        # CoM of the system assumed to be at coM of Payload
        # use parallel axis theorem to represent parafoil tensor at coM of system

        # distance between parafoil and payload
        d = np.array([0, 0, self.Rp])
        d_outer = np.outer(d, d)
        I_parafoil_at_com = I_rotated_parafoil + self.parafoil_mass * ((np.dot(d, d) * np.eye(3)) - d_outer)

        # calculate the inertia tensor at the center of mass
        self.I = I_parafoil_at_com + I_payload

    def derivatives(self):
        acc,angular_acc = self.calculate_derivitives()

        # calculate the velocity
        v_dot = np.dot(self.CDM,acc)

        # calculate the angular velocity
        omega_dot = angular_acc


