# 6 dof model
import numpy as np
import math
import matplotlib.pyplot as plt

# ----------- environment definitions -----------

p_density =  1.293 # (kg/m3) density of air at sea level

class ParafoilSimulation_6Dof:
    """
    A class to simulate the dynamics of a parafoil system in 6 degrees of freedom.
    The simulation includes the effects of aerodynamic forces, moments, and gravity.
    doesnt include the effects of the payload, apparent mass"""
    def __init__(self,params,state, inputs):
        self.set_system_params(params)
        self.set_inputs(inputs)
        self.set_state(state)
        self.error = False
        # calculate the derivatives
        self.calculate_derivatives()

    def set_inputs(self,inputs):
        """
        Set the inputs for the simulation.
        inputs:
            [flap deflection angles, wind vector]
        """
        self.set_actual_flaps(inputs[0])
        self.w = inputs[1] # wind vector in inertial frame

    def set_actual_flaps(self, input):
        """
        set both actual and desired flaps"""
        self.flap_l,self.flap_r = input
        self.flap_l_desired,self.flap_r_desired = input
        self.update_flaps(1) # doesnt matter, they be the same
    
    def set_desired_flaps(self, input):
        self.flap_l_desired,self.flap_r_desired = input
        
    def update_flaps(self,dt):
        """updates"""
        max_rate = dt/self.flap_time_constant
        self.flap_l += np.clip(self.flap_l_desired - self.flap_l, -max_rate, max_rate) 
        self.flap_r += np.clip(self.flap_r_desired - self.flap_r, -max_rate, max_rate) 
        # calculate the flap deflection angles
        self.delta_a = self.flap_r - self.flap_l  # asymmetric flap deflection
        self.delta_s = 0.5*(self.flap_l + self.flap_r) # symmetric flap deflection
    
    def set_state(self, state):
        self.p = state[0] # position in inertial frame
        self.vb = self.safe_clamp_vector(state[1]) # velocity in body fixed frame
        self.eulers = state[2] # euler angles IN RADIANS of body in inertal frame
        self.angular_vel = self.safe_clamp_vector(state[3]) # angular velocities in body fixed frame
        # update the transformations:
        self.update_kinematic_transforations()
        self.update_wind_transformations()

    def set_coefficients(self, coefficients=None):
        """
        Set the aerodynamic coefficients for the simulation.
        Accepts either a list (in predefined order) or a dictionary (by name).
        """
        if coefficients is None:
            return

        # Handle dictionary input
        if isinstance(coefficients, dict):
            for name in [
                "CDo", "CDa", "CD_sym", "CLo", "CLa", "CL_sym",
                "CYB", "ClB", "Clp", "Clr", "Cl_asym", "Cmo",
                "Cma", "Cmq", "CnB", "Cn_p", "Cn_r", "Cn_asym"
            ]:
                if name in coefficients:
                    setattr(self, name, coefficients[name])
            return

        # Handle list input
        if isinstance(coefficients, list) or isinstance(coefficients, tuple):
            if len(coefficients) < 18:
                raise ValueError("Coefficient list must have at least 18 values.")
            (
                self.CDo, self.CDa, self.CD_sym,
                self.CLo, self.CLa, self.CL_sym,
                self.CYB, self.ClB, self.Clp, self.Clr, self.Cl_asym,
                self.Cmo, self.Cma, self.Cmq,
                self.CnB, self.Cn_p, self.Cn_r, self.Cn_asym
            ) = coefficients
            return

    def check_set_param(self, dict, param_name):
        """
        Check if the parameter is set in the dictionary and if its type matches the expected type.
        If so, sets the instance variable self.<attr_name> to the value from the dictionary.
        """
        param = dict.get(param_name)
        if param is None:
            return False
        elif not hasattr(self, param_name):
            raise AttributeError(f"Attribute '{param_name}' does not exist on the instance.")
        elif type(getattr(self, param_name)) != type(param):
            raise TypeError(f"Parameter '{param_name}' should be of type {type(getattr(self, param_name))}. Got {type(param)} instead.")
        
        setattr(self, param_name, param)

        # print(f"Set {param_name} to {param}")
        return True

    def set_system_params(self,system_params):
        """
        Set the system parameters for the simulation. Default values follow Snowflake PAD model.
        
        see: Yakimenko, Oleg A.. (2015). <i>Precision Aerial Delivery Systems - Modeling, Dynamics, and Control
        """
        # ------------------- simulation params --------------
        self.dt = 0.1
        self.check_set_param(system_params, "dt")

        # -------------------- parafoil params

        # aerodynamic parameters, default values follow Snowflake PAD model.
        self.S = 1.0
        self.check_set_param(system_params, "S") # surface area of parafoil
        self.c = 0.75
        self.check_set_param(system_params, "c") # mean chord length
        #self.AR = 0.0
        #self.check_set_param(system_params, self.AR, "AR") # aspect ratio
        self.t = 0.075
        self.check_set_param(system_params, "t") # thickness of the parafoil
        self.b = 1.35
        self.check_set_param(system_params, "b") # wingspan of the parafoil
        self.rigging_angle = np.radians(-12.0)
        self.check_set_param(system_params, "rigging_angle")

        # system parameters
        self.m = 2.4 # mass of the system
        self.check_set_param(system_params, "m") # mass of the system
        self.Rp = np.array([0.0,0,-1.11]) # distance between parafoil and the center of mass for the system
        
        
        self.check_set_param(system_params, "Rp")
        
        self.I = np.array([[0.42,0,0.03],[0,0.4,0],[0.03,0,0.053]]) # moment of inertia of the system
        # if we have to do our own calculations, we need more info
        self.check_set_param(system_params, "I") # moment of inertia of the system
        

        self.initial_pos = [0,0,0] # initial position of the system in inertial frame
        self.check_set_param(system_params, "initial_pos")

        self.flap_time_constant = 1 # number of seconds to go from 0 flap to max flap
        self.check_set_param(system_params, "flap_time_constant")
        
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

        # ________________ aerodynamic parameters ____________________
        # for drag
        self.CD = 0
        self.CDo = 0.25
        self.check_set_param(system_params, "CDo")
        self.CDa = 0.12
        self.check_set_param(system_params, "CDa") # drag coefficient
        self.CD_sym = 0.2
        self.check_set_param(system_params, "CD_sym")

        # for lift
        self.CL = 0
        self.CLo = 0.091
        self.check_set_param(system_params, "CL") # lift coefficient
        self.CLa = 0.90
        self.check_set_param(system_params, "CLa") # lift coefficient changing due to angle of incidance
        self.CL_sym = 0.2
        self.check_set_param(system_params, "CL_sym") # lift coefficient changing due to angle of incidance
        
        # for side force
        self.CYB = -0.23
        self.check_set_param(system_params, "CYB") # side force coefficient

        # for rolling
        self.Cl = 0
        self.ClB = -0.036 # coefficient due to sideslip angle
        self.check_set_param(system_params, "clB")
        self.Clp = -0.84
        self.check_set_param(system_params,"Clp")
        self.Clr = -0.082
        self.check_set_param(system_params,"Clr")
        self.Cl_asym = -0.0035 # coefficient due to asymmetric flap deflection
        self.check_set_param(system_params, "Cl_asym")

        # for pitching
        self.Cm = 0
        self.Cmo = 0.35 # coefficient at zero lift
        self.check_set_param(system_params, "Cmo")
        self.Cma = -0.72 # coefficient due to angle of incidance
        self.check_set_param(system_params, "Cma")
        self.Cmq = -1.49
        self.check_set_param(system_params, "Cmq") # coefficient due to pitch rate

        # for yawing
        self.Cn = 0
        self.CnB = -0.0015 # coefficient due to sideslip angle
        self.check_set_param(system_params, "CnB")
        self.Cn_p = -0.082 # coefficient due to roll rate
        self.check_set_param(system_params, "Cn_p")
        self.Cn_r = -0.27
        self.check_set_param(system_params, "Cn_r") # coefficient due to yaw rate
        self.Cn_asym = 0.0115
        self.check_set_param(system_params, "Cn_asym") # coefficient due to asymmetric flap deflection           
            
    def get_state(self):
        return [self.p, self.vb, self.eulers, self.angular_vel]
    
    def get_inertial_position(self):
        ned_to_world = np.array([1, 1, -1])  # Flip Z-axis
        inertial_pos = self.initial_pos + ned_to_world * self.p  # Element-wise multiplication
        return inertial_pos
    
    def get_inertial_state(self):
        return[self.get_inertial_position(), self.body_to_inertial(self.vb), self.eulers, self.get_euler_rates()]
    
    def get_CDM(self, euler_angles = None):
        """
        get the rotation matrix from body to inertial frame"""
        if euler_angles is None:
            phi, theta, psi = self.eulers
        else:
            phi, theta, psi = euler_angles
        return np.array([
        [np.cos(theta) * np.cos(psi), np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)],
        [np.cos(theta) * np.sin(psi), np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi), np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)],
        [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)]
        ])
    
    def get_euler_rates(self, angular_vel = None):
        """
        get the euler rates from the angular velocity vector.
        """
        if angular_vel is None:
            angular_vel = self.angular_vel
        return np.dot(self.T_angularVel_to_EulerRates, angular_vel)
    
    def get_angular_vel_skew(self, angular_vel = None):
        """
        get the skew symmetric matrix of the angular velocity vector.
        """
        if angular_vel is None:
            p, q, r = self.angular_vel
        else:
            p, q, r = angular_vel
        return np.array([
            [0, -r, q],
            [r, 0, -p],
            [-q, p, 0]
        ])
    
    def get_angular_vel_to_EulerRates_matrix(self, euler_angles = None):
        """
        get the transformation matrix from angular velocity to euler rates.
        """
        if euler_angles is None:
            phi, theta, psi = self.eulers
        else:
            phi, theta, psi = euler_angles
        if abs(theta - np.pi/2) < 0.1:
            self.error = True
            # print("theta is close to 90.")
            theta = np.pi/2 * 0.999
        return np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ])
    
    def update_kinematic_transforations(self, euler_angles = None, angular_vel = None):
        """
        Update the kinematic transformations based on the current state.
        """
        if euler_angles is None:
            euler_angles = self.eulers
        if angular_vel is None:
            angular_vel = self.angular_vel
        self.CDM = self.get_CDM(euler_angles)
        self.angular_vel_skew = self.get_angular_vel_skew(angular_vel)
        self.T_angularVel_to_EulerRates = self.get_angular_vel_to_EulerRates_matrix(euler_angles)

    def update_wind_transformations(self):
        epsilon = 1e-8  # small value to prevent division by zero
        # calculate local airspeed in body fixed frame
        self.va = self.vb - self.body_to_inertial(self.w, True) # local airspeed in body fixed frame
        if np.all(np.isfinite(self.va)):
            self.va_mag = np.linalg.norm(self.va) # magnitude of the local airspeed in body fixed frame
        else:
            self.error = True
            #print(f"Warning: Invalid airspeed vector: va={self.va}, vb = {self.vb}")
            self.va = np.array([epsilon,epsilon,epsilon])
            self.va_mag = epsilon 
            


        # Angle of attack: arctangent of vertical to forward airspeed
        self.angle_of_attack = np.arctan2(self.va[2], self.va[0] if abs(self.va[0]) > epsilon else epsilon * np.sign(self.va[0]))

        # Sideslip angle: arctangent of lateral to horizontal (forward + vertical) airspeed
        denom = np.sqrt(self.va[0]**2 + self.va[2]**2)
        denom_safe = denom if denom > epsilon else epsilon
        self.sideslip_angle = np.arctan2(self.va[1], denom_safe)

        # calculate the rotation matrix wind to body
        self.R_wb = np.array([
            [np.cos(self.angle_of_attack) * np.cos(self.sideslip_angle), -np.sin(self.sideslip_angle), np.cos(self.sideslip_angle) * np.sin(self.angle_of_attack)],
            [np.sin(self.angle_of_attack) * np.cos(self.sideslip_angle), np.cos(self.angle_of_attack), np.sin(self.angle_of_attack) * np.sin(self.sideslip_angle)],
            [-np.sin(self.angle_of_attack), 0, np.cos(self.angle_of_attack)]
        ])
    
    def body_to_inertial(self,vector, inverse = False):
        R_bi = self.CDM.T if inverse else self.CDM
        return np.dot(R_bi, vector)
    
    def body_to_wind(self,vector, inverse = False):
        R_bw = self.R_wb if inverse else self.R_wb.T
        return np.dot(R_bw, vector)

    def calculate_aero_force_coeff(self):
        # for lifting
        self.CL = self.CLo + self.CLa * (self.angle_of_attack + self.rigging_angle) + self.CL_sym*self.delta_s

        # for drag
        self.CD = self.CDo + self.CDa * (self.angle_of_attack + self.rigging_angle)

        # for side force
        self.CY = - self.CYB * self.sideslip_angle

        return [self.CD, self.CY, self.CL]
    
    def calculate_aero_moment_coeff(self):

        # for rolling
        self.Cl = self.ClB * self.sideslip_angle + self.Cl_asym * self.delta_a + \
                    self.Clp * self.c/(2*self.va_mag) * self.angular_vel[0]+ \
                    self.Clr * self.c/(2*self.va_mag) * self.angular_vel[2] 

        # for pitching
        self.Cm = self.Cmo + self.Cma * (self.angle_of_attack + self.rigging_angle) + \
                    self.Cmq * self.c/(2*self.va_mag) * self.angular_vel[1]

        # for yawing
        self.Cn = self.CnB * self.sideslip_angle + self.Cn_asym * self.delta_a + \
                    self.Cn_p * self.c/(2*self.va_mag) * self.angular_vel[0] + \
                    self.Cn_r * self.c/(2*self.va_mag) * self.angular_vel[2]

        return [self.Cl, self.Cm, self.Cn]
    
    def calculate_aero_forces(self):
        self.calculate_aero_force_coeff()
        # calculate the components
        Fa_x = 0.5 * p_density * self.va_mag**2 * self.S * self.CD
        Fa_y = 0.5 * p_density * self.va_mag**2 * self.S * self.CY
        Fa_z = 0.5 * p_density * self.va_mag**2 * self.S * self.CL

        F_aero_A = np.array([Fa_x,Fa_y,Fa_z])
        F_aero_A = self.safe_clamp_vector(F_aero_A)
        # rotate forces to the body frame and negify
        self.F_aero = - self.body_to_wind(F_aero_A, False)
        return self.F_aero

    def calculate_aero_moments(self):
        self.calculate_aero_moment_coeff()
        
        L = 0.5 * p_density * self.va_mag**2 * self.S * self.b * self.Cl
        M = 0.5 * p_density * self.va_mag**2 * self.S * self.c * self.Cm
        N = 0.5 * p_density * self.va_mag**2 * self.S * self.b * self.Cn
        # since 
        M_aero_A = np.array([L,M,N])
        M_aero_A = self.safe_clamp_vector(M_aero_A)
        # rotate moments to the body frame
        # self.M_aero = self.body_to_wind(M_aero_A,True)
        self.M_aero = M_aero_A
        return self.M_aero
    
    def calculate_derivatives(self):        
        # calculate the aero forces
        F_aero = self.calculate_aero_forces()

        # calculate the gravity force
        # theta, phi = self.eulers[1], self.eulers[2]
        self.F_g = self.body_to_inertial([0,0,self.m*9.81],True)

        # calculate acceleration
        self.F_fictious = 0 - self.m * np.dot(self.angular_vel_skew, self.vb)
        F_total = F_aero + self.F_g + self.F_fictious
        self.acc = F_total / self.m


        # calculate the aerodynamic moments
        self.calculate_aero_moments()
        #print("     M_aero: ", M_aero)
        # calculate the moments due to aerodynamic forces
        self.M_f_aero = np.cross(self.Rp,F_aero)
        #print("     M_f_aero: ", M_f_aero)
        # calculate the anglular acceleration
        self.M_fictious = - np.dot(self.angular_vel_skew, np.dot(self.I, self.angular_vel))
        self.M_total = self.M_aero + self.M_fictious
        I_inv = np.linalg.inv(self.I)
        self.angular_acc = np.dot(I_inv,self.M_total)
        return [self.acc,self.angular_acc]

    def calculate_apparent_mass_matrices(self):
            # Correlation factors for flat parafoil
        
        AR = self.b/self.c
        R = np.sqrt((self.b/2)**2 + np.linalg.norm(self.Rp)**2) # line length
        k_A  = 0.848
        k_B  = 0.34  # use avg or upper bound of 1.24 if needed
        k_C  = AR / (1 + AR)

        # 3D corrected factors
        k_A_star  = 0.84 * AR / (1 + AR)
        k_B_star  = 1.161 * AR / (1 + AR)
        k_C_star  = 0.848

        # Apparent mass (flat parafoil)
        m_x_flat = self.rho * k_A  * (np.pi / 4) * self.t**2 * self.b
        m_y_flat = self.rho * k_B  * (np.pi / 4) * self.t**2 * self.c
        m_z_flat = self.rho * k_C  * (np.pi / 4) * self.c**2 * self.b

        # Apparent moments of inertia (flat parafoil)
        I_x_flat = self.rho * k_A  * (np.pi / 48) * self.c**2 * self.b**3
        I_y_flat = self.rho * k_B  * (np.pi / 48) * self.c**4 * self.b
        I_z_flat = self.rho * k_C  * (np.pi / 48) * self.t**2 * self.b**3

                # Geometry
        a_bar = (R - R * np.cos(e0)) / (2 * R * np.sin(e0))  # Mean curvature
        a1 = self.c / 2
        a2 = self.b / 2

        # Apparent Masses
        m_x = m_x_flat * (1 + (8/3) * a_bar**2)
        m_y = (1 / a1**2) * (R**2 * m_y_flat + I_x_flat)
        m_z = m_z_flat * np.sqrt(1 + 2 * a_bar**2 * (1 - self.t**2))

        # Apparent Moments of Inertia
        I_x = (a1**2 / a1**2) * R**2 * m_y_flat + (a2**2 / a1**2) * I_x_flat
        I_y = I_y_flat * (1 + (np.pi / 6) * (1 + AR) * AR * a_bar**2 * self.t**2)
        I_z = (1 + 8 * a_bar**2) * I_z_flat

        I_am = np.dot(np.identity(3),[m_x,m_y,m_z])
        I_ai = np.dot(np.identity(3),[I_x,I_y,I_z])

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
    
    def get_solver_derivatives(self,state):
        old_state = self.get_state()
        self.set_state(state)
        # Calculate the derivatives
        self.calculate_derivatives() 
        # get eular rates
        euler_rates = np.dot(self.T_angularVel_to_EulerRates, self.angular_vel)
        # deritives for the vars: [position, velocity, euler angles, angular velocity]
        derivatives = [self.body_to_inertial(self.vb),self.acc,euler_rates,self.angular_acc]
        # Reset the simulation state to the original
        self.set_state(old_state)
        return derivatives

    def safe_clamp_vector(self, vec, max_abs=1e3):
        """
        Clamp a 3D vector to avoid NaN, inf, and excessively large values.

        Parameters:
            vec (np.ndarray): Input 3D vector.
            max_abs (float): Maximum allowed absolute value for each component.

        Returns:
            np.ndarray: A safe, clamped 3D vector.
        """
        safe_vec = np.zeros(3)

        for i in range(3):
            val = vec[i]
            if not np.isfinite(val):
                self.error = True
                safe_vec[i] = 0.0
            elif abs(val) > max_abs:
                self.error = True
                safe_vec[i] = np.clip(val, -max_abs, max_abs)
            else:
                safe_vec[i] = val

        return safe_vec