# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import numpy as np
import math
from random import uniform
from scipy.optimize import fsolve

def cstr_system(t, y, u):
    """
    Differential equations for a continuous stirred-tank reactor model
    
    t : Time [seconds]
    y : Controlled variables
        C_A : Concentration of reactant A [mol/l]
        C_B : Concentration of reactant B [mol/l]
        T_R : Temperature inside the reactor [Celsius]
        T_K : Temperature of cooling jacker [Celsius]
    u : Manipulated variables
        F : Flow [l/h]
        Q_dot : Heat flow [kW]
    """
    # Process parameters
    K0_ab = 1.287e12 # K0 [h^-1]
    K0_bc = 1.287e12 # K0 [h^-1]
    K0_ad = 9.043e9 # K0 [l/mol.h]
    R_gas = 8.3144621e-3 # Universal gas constant
    E_A_ab = 9758.3*1.00 #* R_gas# [kJ/mol]
    E_A_bc = 9758.3*1.00 #* R_gas# [kJ/mol]
    E_A_ad = 8560.0*1.0 #* R_gas# [kJ/mol]
    H_R_ab = 4.2 # [kJ/mol A]
    H_R_bc = -11.0 # [kJ/mol B] Exothermic
    H_R_ad = -41.85 # [kj/mol A] Exothermic
    Rou = 0.9342 # Density [kg/l]
    Cp = 3.01 # Specific Heat capacity [kJ/Kg.K]
    Cp_k = 2.0 # Coolant heat capacity [kJ/kg.k]
    A_R = 0.215 # Area of reactor wall [m^2]
    V_R = 10.01 #0.01 # Volume of reactor [l]
    m_k = 5.0 # Coolant mass [kg]
    T_in = 130.0 # Temp of inflow [Celsius]
    K_w = 4032.0 # [kJ/h.m^2.K]
    C_A0 = (5.7+4.5)/2.0*1.0 # Concentration of A in input Upper bound 5.7 lower bound 4.5 [mol/l]
    # Process variables
    F, Q_dot = u
    C_a, C_b, T_R, T_K = y
    T_dif = T_R-T_K
    # Rate constants
    K_1 = K0_ab * math.exp((-E_A_ab)/((T_R+273.15)))
    K_2 =  K0_bc * math.exp((-E_A_bc)/((T_R+273.15)))
    K_3 = K0_ad * math.exp((-E_A_ad)/((T_R+273.15)))
    # Differential equations
    dC_adt = F*(C_A0 - C_a) -K_1*C_a - K_3*(C_a**2)
    dC_bdt = -F*C_b + K_1*C_a - K_2*C_b
    dT_Rdt = ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R))
    dT_Kdt = (Q_dot + K_w*A_R*(T_dif))/(m_k*Cp_k)
    
    return dC_adt, dC_bdt, dT_Rdt, dT_Kdt

def steady_state_cstr(manipulated_vars):
    """
    A wrapper to support a steady-state simulation of the CSTR system
    for the manipulated variables
    """
    Q_dot = np.float64(manipulated_vars["Q_dot"]) # Heat flow [kW]
    F = np.float64(manipulated_vars["F"]) # Feed flow rate [l/h]
    u = np.array([F, Q_dot])
    y0 = np.zeros((4,))
    def steady_state(y):
        return cstr_system(0, y, u)
    # Solve for steady-state
    y0 = fsolve(steady_state, y0)
    # Add noise to ''measurements"
    C_a = y0[0] + uniform(-0.025, 0.025)  # Concentration of reactant A [mol/l]
    C_b = y0[1] + uniform(-0.025, 0.025)  # Concentration of reactant B [mol/l]
    T_R = y0[2] + uniform(-0.5, 0.5)  # Temperature inside the reactor [Celsius]
    T_K = y0[3] + uniform(-0.5, 0.5)  # Temperature in the cooling jacket [Celsius]
    measurement = {
        "F": F.round(2), "Q_dot": Q_dot.round(1),
        "C_a": C_a.round(4), "C_b": C_b.round(4),
        "T_R": T_R.round(1), "T_K": T_K.round(1)}
    
    return measurement