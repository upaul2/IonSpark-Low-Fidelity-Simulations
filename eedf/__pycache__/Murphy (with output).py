# Created by Ujjwal (debugged & fixed)

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, e, pi, m_e, hbar, k
from scipy.special import ellipk, ellipe


def schottky_const():
    return np.sqrt(e**3 / (4 * pi * epsilon_0))


def y_MG(F, phi):
    phi_J = phi * e
    return schottky_const() * np.sqrt(F) / phi_J


def v_MG(y):
    y = np.clip(y, 1e-12, 1 - 1e-12)  # avoid singularities

    k = np.sqrt((1 - y) / (1 + y))

    K = ellipk(k**2)
    E = ellipe(k**2)

    return np.sqrt(1 + y) * (E - y * K)


def t_MG(y):
    y = np.clip(y, 1e-12, 1 - 1e-12)

    k = np.sqrt((1 - y) / (1 + y))

    K = ellipk(k**2)
    E = ellipe(k**2)

    return (1 / np.sqrt(1 + y)) * ((1 + y) * E - y * K)


def d_MG(y, F, phi):
    phi_J = phi * e
    t = t_MG(y)
    return (2 * np.sqrt(2 * m_e) / (e * hbar)) * np.sqrt(phi_J) * t / F


def theta_T_MG(y, F, phi, T):
    d = d_MG(y, F, phi)
    x = d * pi * k * T

    # avoid division by zero
    return np.where(x == 0, 1, x / np.sin(x))


def J_murphy_good(F, phi, T):
    # constants
    a = e**3 / (15 * (pi**2) * hbar)
    b = 4 * np.sqrt(2 * m_e) / (3 * e * hbar)

    y = y_MG(F, phi)
    t = t_MG(y)
    v = v_MG(y)
    theta_T = theta_T_MG(y, F, phi, T)

    phi_J = phi * e

    return (a / (t**2)) * (F**2 / phi_J) * \
           np.exp(-b * v * (phi_J**1.5) / F) * theta_T


# -------- TEST PLOT --------
"""
phi = 4.5  # eV
T = 300

F = np.linspace(1e9, 5e9, 200)

J = J_murphy_good(F, phi, T)

plt.plot(F, J)
plt.xlabel("Electric field (V/m)")
plt.ylabel("Current density (A/m^2)")
plt.title("Murphy-Good emission")
plt.yscale("log")
plt.show()
"""

def estimate_electron_emission(
    voltage,
    tip_radius,
    tip_length,
    tube_radius,
    gap_length,
    phi=4.4,
    T=400,
    A_emit=1e-12
):
    """
    Estimate field-emitted current and volumetric electron source.

    Parameters:
    ----------
    voltage : Applied voltage [V]
    tip_radius : Needle tip radius [m]
    tip_length : Needle exposed length [m]
    tube_radius : Plasma/quartz radius [m]
    gap_length : Plasma gap/sheath length [m]
    phi : Work function [eV]
    T : Temperature [K]
    A_emit : Effective emitting area [m^2]

    Returns:
    -------
    dict containing:
        F_local : local electric field [V/m]
        J_emit  : emission current density [A/m^2]
        I_emit  : total emission current [A]
        Ndot    : emitted electrons/sec
        S_e     : volumetric source [m^-3 s^-1]
    """

    ############################################
    # FIELD ENHANCEMENT
    ############################################

    k_geo = 1.680 * (tip_radius/tube_radius + 0.468)**(-1.066)

    gamma = (2*tip_length/tip_radius) / (
        np.log((4*tip_length)/tip_radius) - 2
    )

    F_local = gamma * voltage / (k_geo * tip_radius)

    ############################################
    # MURPHY-GOOD CURRENT DENSITY
    ############################################

    J_emit = J_murphy_good(F_local, phi, T)

    ############################################
    # TOTAL CURRENT
    ############################################

    I_emit = J_emit * A_emit

    ############################################
    # ELECTRONS / SECOND
    ############################################

    Ndot = I_emit / e

    ############################################
    # SOURCE TERM
    ############################################

    V_plasma = pi * tube_radius**2 * gap_length

    S_e = Ndot / V_plasma

    return {
        "F_local": F_local,
        "J_emit": J_emit,
        "I_emit": I_emit,
        "Ndot": Ndot,
        "S_e": S_e
    }

results = estimate_electron_emission(
    voltage=400,
    tip_radius=1e-3,
    tip_length=10e-2,
    tube_radius= 1e-2,
    gap_length=4e-2
)

for key,val in results.items():
    print(f"{key}: {val:.3e}")