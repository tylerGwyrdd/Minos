# 6 dof model
import numpy as np
import math
import matplotlib.pyplot as plt

# ----------- environment definitions -----------

p_density =  1.293 # (kg/m3) density of air at sea level

class ParafoilSimulation_6Dof:
    def __init__(self,params,state, inputs):
        self.set_system_params(params)
        self.set_inputs(inputs)
        self.set_state(state)
        
        # calculate the derivatives
        self.calculate_derivitives()

    def set_inputs(self,inputs):
        """
        Set the inputs for the simulation.
        inputs:
            [flap deflection angles, wind vector]
        """
        self.flap_l,self.flap_r = inputs[0]
        self.w = inputs[1] # wind vector in inertial frame

        # calculate the flap deflection angles
        self.delta_a = self.flap_l - self.flap_r # asymmetric flap deflection
        self.delta_s = 0.5*(self.flap_l + self.flap_r) # symmetric flap deflection

    def set_state(self, state):
        self.p = state[0] # position in inertial frame
        self.vb = state[1] # velocity in body fixed frame
        self.eulers = state[2] # euler angles IN RADIANS of body in inertal frame
        self.angular_vel = state[3] # angular velocities in body fixed frame

        # update the transformations:
        self.update_kinematic_transforations()
        self.update_wind_transformations()

    def get_state(self):
        return [self.p, self.vb, self.eulers, self.angular_vel]

    def update_kinematic_transforations(self):
        """
        Update the kinematic transformations based on the current state.
        """
        psi, theta, phi = self.eulers

        self.CDM =  np.array([
        [np.cos(theta) * np.cos(psi), np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
        [np.cos(theta) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
        [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]
        ])

        if np.isclose(np.cos(theta), 0):
            raise ValueError("Theta cannot be ±90° (or π/2 radians) due to singularity.")

        self.angular_vel_skew = np.array([
            [0, -self.angular_vel[2], self.angular_vel[1]],
            [self.angular_vel[2], 0, -self.angular_vel[0]],
            [-self.angular_vel[1], self.angular_vel[0], 0]
        ])

        # calculate the transformation matrix from angular velocity to euler rates
        self.T_angularVel_to_EulerRates = np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ])

    def update_wind_transformations(self):
        # calculate local airspeed in body fixed frame
        self.va = self.vb - self.body_to_inertial(self.w, True) # local airspeed in body fixed frame
        self.va_mag = np.linalg.norm(self.va) # magnitude of the local airspeed in body fixed frame

        # we also need to calculate the AoA and sideslip angle (radians)
        self.angle_of_attack = np.arctan(self.va[2]/self.va[0]) 
        self.sideslip_angle = np.arcsin(self.va[2]/self.va_mag)

        # calculate the rotation matrix from body to wind frame
        self.R_bw = np.array([
            [np.cos(self.angle_of_attack) * np.cos(self.sideslip_angle), -np.sin(self.sideslip_angle), np.cos(self.sideslip_angle) * np.sin(self.angle_of_attack)],
            [np.sin(self.angle_of_attack) * np.cos(self.sideslip_angle), np.cos(self.angle_of_attack), np.sin(self.angle_of_attack) * np.sin(self.sideslip_angle)],
            [-np.sin(self.angle_of_attack), 0, np.cos(self.angle_of_attack)]
        ])

    def body_to_inertial(self,vector, inverse = False):
        R_bi = self.CDM.T if inverse else self.CDM
        return np.dot(R_bi, vector)
    
    def body_to_wind(self,vector, inverse = False):
        R_bw = self.R_bw.T if inverse else self.R_bw
        return np.dot(R_bw, vector)

    def check_set_param(self, dict, var, param_name):
        """
        Check if the parameter is set in the dictionary and if its type matches the expected type.
        
        If not, raise an error.
        """
        param = dict.get(param_name)
        if param == None:
            print(f"Parameter '{param_name}' is not set. Using default value {var}.")
            return False
        elif type(var) != type(param):
            raise TypeError(f"Parameter '{param_name}' should be of type {type(var)}. Got {type(param)} instead.")
        var = param
        return True

    def set_system_params(self,system_params):
        """
        Set the system parameters for the simulation. Default values follow Snowflake PAD model.
        
        see: Yakimenko, Oleg A.. (2015). <i>Precision Aerial Delivery Systems - Modeling, Dynamics, and Control
        """

        # areodynamic parameters, default values follow Snowflake PAD model.
        self.S = 1.0
        self.check_set_param(system_params, self.S, "S") # surface area of parafoil
        self.c = 0.75
        self.check_set_param(system_params, self.c, "c") # mean chord length
        #self.AR = 0.0
        #self.check_set_param(system_params, self.AR, "AR") # aspect ratio
        self.t = 0.075
        self.check_set_param(system_params, self.t, "t") # thickness of the parafoil
        self.b = 1.35
        self.check_set_param(system_params, self.b, "b") # wingspan of the parafoil
        self.rigging_angle = -12.0
        self.check_set_param(system_params, self.rigging_angle, "rigging_angle")

        # system parameters
        self.m = 2.4 # mass of the system
        self.check_set_param(system_params,self.m, "m") # mass of the system
        self.Rp = np.array([0.046,0,-1.11]) # distance between parafoil and the center of mass for the system
        self.check_set_param(system_params,self.Rp, "Rp")
        
        self.I = np.array([[0.42,0,0.03],[0,0.4,0],[0.03,0,0.053]]) # moment of inertia of the system
        # if we have to do our own calculations, we need more info
        self.check_set_param(system_params,self.I, "I") # moment of inertia of the system
        """"
        self.parafoil_mass = 0.0
        self.check_set_param(system_params,self.parafoil_mass, "parafoil_mass") # mass of parafoil
        self.payload_mass = 0.0
        self.check_set_param(system_params,self.payload_mass, "payload_mass") # mass of payload
        self.Rlc = 0.0
        self.check_set_param(system_params,self.Rlc, "Rlc") # distance between payload CoM and connection point to parafoil
        self.Rpc = 0.0
        self.check_set_param(system_params,self.Rpc, "Rpc") # distance between parafoil CoM and connection point to parafoil
        
        self.m = self.parafoil_mass + self.payload_mass # mass of entire system

        """
        # ________________ areodynamic parameters ____________________
        # for drag
        self.CDo = 0.25
        self.check_set_param(system_params,self.CDo, "Cdo")
        self.CDa = 0.12
        self.check_set_param(system_params,self.CDa, "Cda") # drag coefficient

        # for lift
        self.CLo = 0.091
        self.check_set_param(system_params,self.CLo, "Cl") # lift coefficient
        self.CLa = 0.90
        self.check_set_param(system_params,self.CLa, "Cla") # lift coefficient changing due to angle of incidance
        
        # for side force
        self.CYB = -0.23
        self.check_set_param(system_params,self.CYB, "CyB") # side force coefficient

        # for rolling
        self.clB = -0.036 # coefficient due to sideslip angle
        self.check_set_param(system_params,self.clB, "clB")
        self.Clp = -0.84
        self.check_set_param(system_params,self.Clp, "Clp")
        self.Clr = -0.082
        self.check_set_param(system_params,self.Clr, "Clr")
        self.Cl_asym = -0.0035 # coefficient due to asymmetric flap deflection
        self.check_set_param(system_params,self.Cl_asym, "Cl_asym")

        # for pitching
        self.Cmo = 0.35 # coefficient at zero lift
        self.check_set_param(system_params,self.Cmo, "Cmo")
        self.Cma = -0.72 # coefficient due to angle of incidance
        self.check_set_param(system_params,self.Cma, "Cma")
        self.Cmq = -1.49
        self.check_set_param(system_params,self.Cmq, "Cmq") # coefficient due to pitch rate

        # for yawing
        self.CnB = -0.0015 # coefficient due to sideslip angle
        self.check_set_param(system_params,self.CnB, "CnB")
        self.Cn_p = -0.082 # coefficient due to roll rate
        self.check_set_param(system_params,self.Cn_p, "Cn_p")
        self.Cn_r = -0.27
        self.check_set_param(system_params,self.Cn_r, "Cn_r") # coefficient due to yaw rate
        self.Cn_asym = 0.0015
        self.check_set_param(system_params,self.Cn_asym, "Cn_asym") # coefficient due to asymmetric flap deflection           
            
    def calculate_areo_force_coeff(self):
        # for lifting
        self.CL = self.CLo + self.CLa * self.angle_of_attack

        # for drag
        self.CD = self.CDo + self.CDa * self.angle_of_attack

        # for side force
        self.CY = self.CYB * self.sideslip_angle

        return [self.CL, self.CD, self.CY]
    
    def calculate_areo_moment_coeff(self):

        # for rolling
        self.Cl = self.Clp * self.c/(2*self.va_mag) * self.angular_vel[0] + \
                    self.Clr * self.c/(2*self.va_mag) * self.angular_vel[2] + self.Cl_asym * self.delta_a

        # for pitching
        self.Cm = self.Cmo + self.Cma * self.angle_of_attack + \
                    self.Cmq * self.c/(2*self.va_mag) * self.angular_vel[1]

        # for yawing
        self.Cn = self.CnB * self.sideslip_angle + self.Cn_asym * self.delta_a + \
                    self.Cn_p * self.c/(2*self.va_mag) * self.angular_vel[0] + \
                    self.Cn_r * self.c/(2*self.va_mag) * self.angular_vel[2]

        return [self.Cl, self.Cm, self.Cn]
    
    def calculate_aero_forces(self):
        coeffs = self.calculate_areo_force_coeff()
        F_aero_A = 0.5 * p_density * self.va_mag**2 * self.S * np.array(coeffs)

        #rotate forces to the body frame
        F_areo = self.body_to_wind(F_aero_A, True)
        return F_areo

    def calculate_aero_moments(self):

        C_m, C_l, C_n = self.calculate_areo_moment_coeff()

        L = 0.5 * p_density * self.va_mag**2 * self.S * self.b * C_l
        M = 0.5 * p_density * self.va_mag**2 * self.S * self.c * C_m
        N = 0.5 * p_density * self.va_mag**2 * self.S * self.b * C_n

        M_aero_A = np.array([L,M,N])

        # rotate moments to the body frame
        M_aero = self.body_to_wind(M_aero_A,True)

        return M_aero
    
    def calculate_derivitives(self):        
        # calculate the aero forces
        F_aero = self.calculate_aero_forces()

        # calculate the gravity force
        theta, phi = self.eulers[1], self.eulers[2]
        F_g = self.m * 9.81 * np.array([
            -np.sin(theta),
            np.cos(theta) * np.sin(phi),
            np.cos(theta) * np.cos(phi)
        ])
        
        # calculate acceleration
        F_total = F_aero + F_g - self.m * np.dot(self.angular_vel_skew,self.vb)
        self.acc = F_total / self.m
        #print("     F_aero: ", F_aero)
        #print("     F_rot: ", np.dot(self.angular_vel_skew,self.vb))

        # calculate the aerodynamic moments
        M_aero = self.calculate_aero_moments()
        #print("     M_aero: ", M_aero)
        # calculate the moments due to aerodynamic forces
        M_f_areo = np.cross(self.Rp,F_aero)
        #print("     M_f_aero: ", M_f_areo)
        # calculate the anglular acceleration
        M_total = M_aero - np.dot(self.angular_vel_skew, np.dot(self.I,self.angular_vel)) + M_f_areo
        I_inv = np.linalg.inv(self.I)
        self.angular_acc = np.dot(I_inv,M_total)

        return [self.acc,self.angular_acc]

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

    def get_solver_derivities(self,state):
        print("state: ", state)
        
        old_state = self.get_state()
        self.set_state(state)
        # Calculate the derivatives
        self.calculate_derivitives() 
        # get eular rates
        euler_rates = np.dot(self.T_angularVel_to_EulerRates, self.angular_vel)
        # deritives of the variables in correct frames
        derivatives = [self.body_to_inertial(self.vb),self.acc,euler_rates,self.angular_acc]
        # Reset the simulation state to the original
        print("derivatives: ", derivatives)
        self.set_state(old_state)
        return derivatives
