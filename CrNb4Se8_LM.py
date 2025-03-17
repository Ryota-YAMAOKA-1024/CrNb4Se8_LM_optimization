import math
import cmath
import argparse
import os
import subprocess


import numpy as np
# from scipy.optimize import curve_fit  # Not used in this version


##================== Physical Constants & Material Properties ==========================
# Fundamental constants
neutron_mass = 1.673 * 10 ** (-27)  # kg
Planck_constant = 6.626 * 10 ** (-34)  # J·s

# Neutron scattering lengths (in femtometers, fm)
Cr_scattering_length = 3.635
Nb_scattering_length = 7.054
Se_scattering_length = 7.970

# Atomic occupancies
Cr_occupancy = 0.75
Nb_occupancy = 1.0
Se_occupancy = 1.0

# Isotropic temperature factors (B_iso)
Cr_B_iso = 0.0
Nb_B_iso = 0.0
Se_B_iso = 0.0

# Other physical parameters
coefficient_gamma = 1.913
radius_of_electron = 2.818  # fm
Cr3_magnetic_moment = 2.0898  # µ_B (Bohr magneton)

# Magnetic form factor coefficients for Cr³⁺
Cr3_magnetic_form_factor_coefficients = [-0.3094, 0.0274, 0.3680, 17.0355, 0.6559, 6.5236, 0.2856]

# Lattice parameters (angstroms)
a_value = 6.904
b_value = 6.904
c_value = 12.57

# Neutron energy (meV)
neutron_energy_value = 34.05

# Measurement angle ranges (degrees)
max_2theta_value = 90
min_2theta_value = 2
max_omega_value = 135
min_omega_value = -60

# Lists for scattering properties
scattering_length_list_for_Cr1_4NbSe2 = [Cr_scattering_length, Nb_scattering_length, Se_scattering_length]
occupancy_list_for_Cr1_4NbSe2 = [Cr_occupancy, Nb_occupancy, Se_occupancy]
B_iso_list_for_Cr1_4NbSe2 = [Cr_B_iso, Nb_B_iso, Se_B_iso]

##================== Atomic Positions in the Unit Cell ==========================
def generate_atomic_coordinates(eps_Nb2, eps_Se1, eps_Se2_x, eps_Se2_z):
    """
    Generate atomic coordinates in the unit cell considering atomic displacements.

    Parameters:
        eps_Nb2: Displacement parameter for Nb (6h)
        eps_Se1: Displacement parameter for Se (4f)
        eps_Se2_x: x-direction displacement for Se (12k)
        eps_Se2_z: z-direction displacement for Se (12k)

    Returns:
        List of atomic coordinates categorized by element.
    """
    # Chromium (Cr) at Wyckoff position 2a
    Cr1_atomic_coordinates = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1/2]
    ]

    # Niobium (Nb) at Wyckoff position 2b
    Nb1_atomic_coordinates = [
        [0.0, 0.0, 1/4],
        [0.0, 0.0, 3/4]
    ]

    # Niobium (Nb) at Wyckoff position 6h (with displacement eps_Nb2)
    x_Nb2 = 1/2 + eps_Nb2
    Nb2_atomic_coordinates = [
        [x_Nb2, 2*x_Nb2, 1/4],
        [2*x_Nb2, x_Nb2, 1/4],
        [-x_Nb2, -2*x_Nb2, 1/4],
        [-2*x_Nb2, -x_Nb2, 1/4],
        [x_Nb2, -x_Nb2, 3/4],
        [-x_Nb2, x_Nb2, 3/4]
    ]

    # Selenium (Se) at Wyckoff position 4f (with displacement eps_Se1)
    z_Se1 = 5/8 + eps_Se1
    Se1_atomic_coordinates = [
        [1/3, 2/3, z_Se1],
        [2/3, 1/3, z_Se1 + 1/2],
        [2/3, 1/3, -z_Se1],
        [1/3, 2/3, -z_Se1 + 1/2]
    ]

    # Selenium (Se) at Wyckoff position 12k (with displacements eps_Se2_x and eps_Se2_z)
    x_Se2 = 1/6 + eps_Se2_x
    z_Se2 = 1/8 + eps_Se2_z
    Se2_atomic_coordinates = [
        [x_Se2, 2 * x_Se2, z_Se2],
        [-2 * x_Se2, -x_Se2, z_Se2],
        [x_Se2, -x_Se2, z_Se2],
        [-x_Se2, -2 * x_Se2, z_Se2 + 1/2],
        [2 * x_Se2, x_Se2, z_Se2 + 1/2],
        [-x_Se2, x_Se2, z_Se2 + 1/2],
        [2 * x_Se2, x_Se2, -z_Se2],
        [-x_Se2, -2 * x_Se2, -z_Se2],
        [-x_Se2, x_Se2, -z_Se2],
        [-2 * x_Se2, -x_Se2, -z_Se2 + 1/2],
        [x_Se2, 2 * x_Se2, -z_Se2 + 1/2],
        [x_Se2, -x_Se2, -z_Se2 + 1/2]
    ]

    # Grouping atomic coordinates by element
    Cr_atomic_coordinates = [Cr1_atomic_coordinates]
    Nb_atomic_coordinates = [Nb1_atomic_coordinates, Nb2_atomic_coordinates]
    Se_atomic_coordinates = [Se1_atomic_coordinates, Se2_atomic_coordinates]

    # Return structured atomic coordinates
    atomic_coordinates_list_for_Cr1_4NbSe2_temp = [Cr_atomic_coordinates, Nb_atomic_coordinates, Se_atomic_coordinates]

    return atomic_coordinates_list_for_Cr1_4NbSe2_temp

# Default values for atomic displacement parameters
eps_Nb2_default = 0.0079
eps_Se1_default = -0.0026
eps_Se2_x_default = 0.0017
eps_Se2_z_default = -0.0079
atomic_coordinates_list_for_Cr1_4NbSe2 = generate_atomic_coordinates(eps_Nb2_default, eps_Se1_default, eps_Se2_x_default, eps_Se2_z_default)

'''
eps_Nb2_optimized_all = 0.01390695137030588
eps_Se1_optimized_all = -0.012604450576591137
eps_Se2_x_optimized_all = 0.0018460657160098675
eps_Se2_z_optimized_all = -0.006644638374803201
atomic_coordinates_list_for_Cr1_4NbSe2 = generate_atomic_coordinates(eps_Nb2_optimized_all, eps_Se1_optimized_all, eps_Se2_x_optimized_all, eps_Se2_z_optimized_all)
'''


##================== Basic Functions =============================================
def Q_vector_length(H, K, L, a, b, c, alpha=90, beta=90, gamma=120):
    """
    Calculate the length of the Q-vector in reciprocal space, 
    considering a general triclinic lattice.

    Parameters:
        H, K, L: Miller indices (integers)
        a, b, c: Lattice constants (floats)
        alpha, beta, gamma: Lattice angles in degrees (default: 90° for cubic)

    Returns:
        Length of the Q-vector (float)
    """
    # Convert degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Calculate unit cell volume
    V = a * b * c * np.sqrt(
        1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 +
        2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )

    # Compute reciprocal lattice constants
    a_star = (b * c * np.sin(alpha_rad)) / V
    b_star = (a * c * np.sin(beta_rad)) / V
    c_star = (a * b * np.sin(gamma_rad)) / V

    # Compute reciprocal space angles using metric tensor relations
    cos_alpha_star = (np.cos(beta_rad) * np.cos(gamma_rad) - np.cos(alpha_rad)) / (np.sin(beta_rad) * np.sin(gamma_rad))
    cos_beta_star = (np.cos(alpha_rad) * np.cos(gamma_rad) - np.cos(beta_rad)) / (np.sin(alpha_rad) * np.sin(gamma_rad))
    cos_gamma_star = (np.cos(alpha_rad) * np.cos(beta_rad) - np.cos(gamma_rad)) / (np.sin(alpha_rad) * np.sin(beta_rad))

    # Compute reciprocal space metric tensor g* (in reciprocal lattice)
    g_star = np.array([
        [a_star**2, a_star * b_star * cos_gamma_star, a_star * c_star * cos_beta_star],
        [b_star * a_star * cos_gamma_star, b_star**2, b_star * c_star * cos_alpha_star],
        [c_star * a_star * cos_beta_star, c_star * b_star * cos_alpha_star, c_star**2]
    ])

    # Miller indices vector
    HKL = np.array([H, K, L])

    # Compute Q-vector length using metric tensor
    Q_squared = np.dot(HKL, np.dot(g_star, HKL))
    Q_length = 2 * np.pi * np.sqrt(Q_squared)

    return Q_length


def Debye_Waller_factor_from_B_iso(B_iso, H, K, L, a, b, c, alpha=90, beta=90, gamma=120):
    """
    Calculate the Debye-Waller factor from the isotropic temperature factor (B_iso)
    and the Q-vector in reciprocal space.

    Parameters:
        B_iso: Isotropic temperature factor (float)
        H, K, L: Miller indices (integers)
        a, b, c: Crystal lattice constants (floats)
        alpha, beta, gamma: Lattice angles in degrees (default: 90°)

    Returns:
        Debye-Waller factor (float)
    """
    # Calculate the Q-vector length
    Q_value = Q_vector_length(H, K, L, a, b, c, alpha, beta, gamma)

    # Calculate the Debye-Waller factor
    DWF = np.exp(-B_iso * (Q_value / (4 * np.pi))**2)

    return abs(DWF)


# Calculate the measurable Q range for given neutron energy and scattering angle
def allowed_Q(neutron_energy, theta_2):
    neutron_k = (neutron_energy / 2.072)**0.5
    theta_value = math.radians(theta_2 / 2)
    return 2 * neutron_k * math.sin(theta_value)


def Lorenz_factor(neutron_energy, H, K, L, a, b, c, alpha=90, beta=90, gamma=120):
    """
    Calculate the Lorenz factor using the neutron energy and the Q-vector in reciprocal space.

    Parameters:
        neutron_energy: Neutron energy (in meV)
        H, K, L: Miller indices (integers)
        a, b, c: Crystal lattice constants (floats)
        alpha, beta, gamma: Lattice angles in degrees (default: 90°)

    Returns:
        Lorenz factor (float)
    """
    # Calculate the neutron wavelength (in angstroms)
    neutron_lambda = math.sqrt(81.81 / neutron_energy)

    # Calculate the neutron wavenumber k (in angstrom^-1)
    neutron_k = math.sqrt(neutron_energy / 2.072)

    # Calculate the Q-vector length
    Q_value = Q_vector_length(H, K, L, a, b, c, alpha, beta, gamma)

    # Calculate the diffraction angle θ
    theta = math.asin(Q_value / (2 * neutron_k))

    # Calculate the Lorenz factor
    Lorenz = neutron_lambda**3 / math.sin(2 * theta)

    return Lorenz


##================== Nuclear Structure Factor =============================================
def nuclear_structure_factor(scattering_length_list, atomic_coordinates_list, occupancy_list, B_iso_list, H, K, L, a, b, c, alpha=90, beta=90, gamma=120):
    """
    Calculate the nuclear structure factor.

    Parameters:
        scattering_length_list: List of neutron scattering lengths
        atomic_coordinates_list: List of atomic coordinates
        occupancy_list: List of atomic occupancies
        B_iso_list: List of isotropic temperature factors
        H, K, L: Miller indices
        a, b, c: Lattice constants
        alpha, beta, gamma: Lattice angles in degrees

    Returns:
        Absolute value of the nuclear structure factor
    """
    NSF = 0
    for atom in range(len(scattering_length_list)):
        scattering_length = scattering_length_list[atom]
        atomic_coordinates = atomic_coordinates_list[atom]
        occupancy = occupancy_list[atom]
        Debye_Waller_factor = Debye_Waller_factor_from_B_iso(B_iso_list[atom], H, K, L, a, b ,c)

        for i in range(len(atomic_coordinates)):
            value_from_atom_i = 0
            for j in range(len(atomic_coordinates[i])):
                value_from_atom_i += cmath.exp(2 * cmath.pi * 1j * (H * atomic_coordinates[i][j][0] + K * atomic_coordinates[i][j][1] + L * atomic_coordinates[i][j][2]))
            NSF += scattering_length * Debye_Waller_factor * value_from_atom_i * occupancy

    return abs(NSF)


##================== Functions for Optimization =============================================
def s_value(Fcal_list, Fobs_list):
    """
    Calculate the scale factor 's' and its error 's_err' using the least squares method.

    Parameters:
        Fcal_list (list of floats): List of calculated structure factors.
        Fobs_list (list of tuples): List of observed structure factors and their errors.
                                    Each element is (F_obs, F_obs_err).

    Returns:
        s (float): Scale factor that minimizes the least squares difference.
        s_err (float): Estimated standard error of the scale factor.
    """

    # Initialize sums for the least squares calculation
    A = 0  # Denominator sum
    B = 0  # Numerator sum

    for i in range(len(Fcal_list)):
        F_obs, F_obs_err = Fobs_list[i]  # Observed value and its error
        F_cal = Fcal_list[i]  # Calculated structure factor

        # Compute summations for least squares estimation
        A += 2 * (F_obs ** 2) / (F_obs_err ** 2)  # Weighted sum of observed values squared
        B += 2 * (F_obs * F_cal) / (F_obs_err ** 2)  # Weighted sum of observed * calculated

    # Compute the scale factor s
    s = B / A

    # Compute the standard error of s
    s_err = np.sqrt(2 / A)  # Standard deviation formula from least squares

    return s, s_err


def ChiSquare(Fcal_list, Fobs_list, s):
    """
    Calculate the chi-square statistic.

    Parameters:
        Fcal_list: List of calculated structure factors
        Fobs_list: List of observed structure factors
        s: Scale factor

    Returns:
        Chi-square value
    """
    Xsq = 0.0
    for i in range(len(F_cal_list)):
        Xsq += (Fcal_list[i] - (s * Fobs_list[i][0])) ** 2 / (s * F_obs_list[i][1]) ** 2

    return Xsq


##================== Functions for Evaluation =============================================
def R_factor(F_cal_list, F_obs_list):
    """
    Calculate the R-factor to evaluate the model fit.

    Parameters:
        F_cal_list: List of calculated structure factors
        F_obs_list: List of observed structure factors

    Returns:
        R-factor (float)
    """
    numerator = 0
    denominator = 0

    for i in range(len(F_cal_list)):
        numerator += abs(F_cal_list[i] - F_obs_list[i])
        denominator += abs(F_obs_list[i])

    return numerator / denominator


##================== Main Function ================================================
HKL_list = []  # List to store Miller indices (H, K, L)
F_obs_list = []  # List to store observed structure factors and their errors
F_cal_list = []  # List to store calculated structure factors


# Using argparse for Command-Line Argument Parsing
parser = argparse.ArgumentParser(description='CrNb4Se8をLM法で構造最適化して結果をファイルに出力します。')
parser.add_argument('input_file', nargs='?', default='Fobs_Nuc.txt',
                    help='入力ファイル名 (デフォルト: Fobs_Nuc.txt)')
args = parser.parse_args()

# Read observed structure factors from file
input_file = args.input_file
FH1 = open(input_file, 'r')

for line in FH1:
    if line.find('#') != 0:  # Ignore comment lines
        values = line.split()
        HKL_list.append([int(values[0]), int(values[1]), int(values[2])])
        F_obs_list.append([float(values[3]), float(values[4])])

FH1.close()

# print(*HKL_list, sep='\n')  # Debugging: Print the loaded HKL list

##================== Before Optimization ========================================
# Compute calculated structure factors before optimization
for i in range(len(HKL_list)):
    H, K, L = HKL_list[i][0], HKL_list[i][1], HKL_list[i][2]
    F_cal_list.append(nuclear_structure_factor(
        scattering_length_list_for_Cr1_4NbSe2, 
        atomic_coordinates_list_for_Cr1_4NbSe2, 
        occupancy_list_for_Cr1_4NbSe2, 
        B_iso_list_for_Cr1_4NbSe2, 
        H, K, L, a_value, b_value, c_value
    ))

# Open file to write calculated and observed structure factors
FH2 = open('Fcal_Fobs_Nuc.txt', 'w')

# Compute scale factor and initial R-factor
s, s_err = s_value(F_cal_list, F_obs_list)
R_before_optimized = R_factor(F_cal_list, s * np.array(F_obs_list)[:, 0])

# Write initial R-factor and parameters
FH2.write('# R_before_optimized = {0}\n'.format(R_before_optimized))
FH2.write('# Scale factor (s) = {0}\n'.format(s))
FH2.write('# Scale factor error (s_err) = {0}\n'.format(s_err))
FH2.write('# Initial B_iso = {0}\n'.format(Cr_B_iso))

# Write initial atomic displacement parameters
FH2.write('# Initial eps_Nb2 = {0}\n'.format(eps_Nb2_default))
FH2.write('# Initial eps_Se1 = {0}\n'.format(eps_Se1_default))
FH2.write('# Initial eps_Se2_x = {0}\n'.format(eps_Se2_x_default))
FH2.write('# Initial eps_Se2_z = {0}\n'.format(eps_Se2_z_default))

# Write initial chromium occupancy 
FH2.write('# Initial Cr_occupancy = {0}\n'.format(Cr_occupancy))

# Uncomment the following lines to use optimized values (if available)
'''
FH2.write('# eps_Nb2 (optimized) = {0}\n'.format(eps_Nb2_optimized_all))
FH2.write('# eps_Se1 (optimized) = {0}\n'.format(eps_Se1_optimized_all))
FH2.write('# eps_Se2_x (optimized) = {0}\n'.format(eps_Se2_x_optimized_all))
FH2.write('# eps_Se2_z (optimized) = {0}\n'.format(eps_Se2_z_optimized_all))
'''

# Write structure factors data
FH2.write('# H  K  L  F_calculated  F_observed  F_observed_error\n')
for i in range(len(HKL_list)):
    FH2.write('{0} {1} {2} {3} {4} {5}\n'.format(
        HKL_list[i][0], HKL_list[i][1], HKL_list[i][2], 
        F_cal_list[i], s * F_obs_list[i][0], s * F_obs_list[i][1]
    ))

FH2.close()



##===================== Optimization ===========================================
# Model function (non-linear)
def model(points, params):
    """
    Compute the calculated structure factors based on model parameters.

    Parameters:
        points: List of HKL values (Miller indices)
        params: List of model parameters 
                [scale factor (s), B_iso, eps_Nb2, eps_Se1, eps_Se2_x, eps_Se2_z, Cr_occupancy]

    Returns:
        Array of calculated structure factors normalized by scale factor s
    """
    s, B_isoall, eps_Nb2, eps_Se1, eps_Se2_x, eps_Se2_z, Cr_occupancy = params
    B_iso_list_for_Cr1_4NbSe2_for_model = [B_isoall, B_isoall, B_isoall]
    occupancy_list_for_Cr1_4NbSe2_for_model = [Cr_occupancy, 1, 1]
    atomic_coordinates_list_for_Cr1_4NbSe2_for_model = generate_atomic_coordinates(eps_Nb2, eps_Se1, eps_Se2_x, eps_Se2_z)

    F_cal = np.array([
        nuclear_structure_factor(
            scattering_length_list_for_Cr1_4NbSe2, 
            atomic_coordinates_list_for_Cr1_4NbSe2_for_model, 
            occupancy_list_for_Cr1_4NbSe2_for_model, 
            B_iso_list_for_Cr1_4NbSe2_for_model, 
            points[i][0], points[i][1], points[i][2], 
            a_value, b_value, c_value
        ) for i in range(len(points))
    ])

    return F_cal / s

# Residual function (without considering experimental errors)
def residuals(params, points, F_obs):
    """
    Compute residuals for least squares fitting.

    Parameters:
        params: List of model parameters
        points: List of HKL values
        F_obs: Array of observed structure factors

    Returns:
        Residuals
    """
    return model(points, params) - F_obs

# Jacobian matrix computation (without considering experimental errors)
def jacobian(params, points, delta=1e-6):
    """
    Compute the Jacobian matrix for Levenberg-Marquardt optimization.

    Parameters:
        params: List of model parameters
        points: List of HKL values
        delta: Small perturbation for numerical differentiation

    Returns:
        Jacobian matrix
    """
    J = np.zeros((len(points), len(params)))
    for i in range(len(params)):
        params_up = params.copy()
        params_down = params.copy()
        params_up[i] += delta
        params_down[i] -= delta
        J[:, i] = (residuals(params_up, points, F_obs) - residuals(params_down, points, F_obs)) / (2 * delta)
    return J

# Levenberg-Marquardt method (considering experimental errors with weighting matrix W)
def levenberg_marquardt_with_errors(points, F_obs, F_obs_err, params_init, max_iter=100, tol=1e-6, lambda_init=0.01, lambda_factor=10):
    """
    Perform non-linear least squares optimization using the Levenberg-Marquardt algorithm,
    considering experimental errors via a weighting matrix W.

    Parameters:
        points: List of HKL values
        F_obs: Array of observed structure factors
        F_obs_err: Array of uncertainties in F_obs
        params_init: Initial guess for parameters
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        lambda_init: Initial damping factor for LM algorithm
        lambda_factor: Factor to adjust damping parameter

    Returns:
        Optimized parameters and their estimated errors
    """
    params = np.array(params_init, dtype=float)
    lambda_ = lambda_init

    for iteration in range(max_iter):
        res = residuals(params, points, F_obs)
        J = jacobian(params, points)
        W = np.diag(1 / F_obs_err**2)  # Weighting matrix W
        H = J.T @ W @ J  # J^T W J for Hessian approximation
        g = J.T @ W @ res  # Weighted gradient vector

        # Update parameters with regularization
        delta_params = np.linalg.solve(H + lambda_ * np.eye(len(params)), -g)

        # New parameter candidates
        params_new = params + delta_params
        res_new = residuals(params_new, points, F_obs)

        # Check improvement in residuals
        if np.linalg.norm(res_new) < np.linalg.norm(res):
            params = params_new  # Accept new parameters
            lambda_ /= lambda_factor  # Reduce lambda (move towards Gauss-Newton)
        else:
            lambda_ *= lambda_factor  # Increase lambda (move towards gradient descent)

        # Convergence check
        if np.linalg.norm(delta_params) < tol:
            break

    # Compute covariance matrix (inverse of J^T W J)
    try:
        cov_matrix = np.linalg.inv(H)
        param_errors = np.sqrt(np.diag(cov_matrix))  # Standard deviation (errors)
    except np.linalg.LinAlgError:
        cov_matrix = None
        param_errors = None

    return params, param_errors


# Extract observed data from input file
F_obs = np.array(F_obs_list)[:, 0]
F_obs_err = np.array(F_obs_list)[:, 1]  # Measurement uncertainties

# Initial parameter values
params_init = [7.177, 0, 0.0079, -0.0026, 0.0017, -0.0079, 0.75]

# Perform optimization
opt_params, param_errors = levenberg_marquardt_with_errors(HKL_list, F_obs, F_obs_err, params_init)

# Display results
print("Optimized Parameters:", opt_params)

if param_errors is not None:
    print("Parameter Errors:", param_errors)
else:
    print("Failed to compute parameter errors (possibly a singular matrix)")

# Compute differences from initial values
param_differences = opt_params - params_init
print("Differences from Initial Values:", param_differences)


##===================== Evaluation ==========================================
# Compute the R-factor before optimization
print('R_before_optimized = {0}'.format(R_before_optimized))

# Compute the R-factor after optimization
R_optimized = R_factor(model(HKL_list, opt_params) * opt_params[0], F_obs * opt_params[0])
print('R_optimized = {0}'.format(R_optimized))

# Display optimized chromium occupancy and its error
print('Cr_occupancy = {0}'.format(opt_params[6]))
print('Cr_occupancy_err = {0}'.format(param_errors[6]))


##===================== Output Results ======================================
# Open output file to store optimization results
FH3 = open('result_for_optimization.txt', 'w')

# Write R-factors before and after optimization
FH3.write('# R_before_optimized = {0}\n'.format(R_before_optimized))
FH3.write('# R_optimized = {0}\n'.format(R_optimized))

# Write optimized parameters
FH3.write('# Scale factor (s) = {0}\n'.format(opt_params[0]))
FH3.write('# B_iso = {0}\n'.format(opt_params[1]))
FH3.write('# eps_Nb2 = {0}\n'.format(opt_params[2]))
FH3.write('# eps_Se1 = {0}\n'.format(opt_params[3]))
FH3.write('# eps_Se2_x = {0}\n'.format(opt_params[4]))
FH3.write('# eps_Se2_z = {0}\n'.format(opt_params[5]))
FH3.write('# Cr_occupancy = {0}\n'.format(opt_params[6]))

# Write estimated errors for each parameter
FH3.write('# s_err = {0}\n'.format(param_errors[0]))
FH3.write('# B_iso_err = {0}\n'.format(param_errors[1]))
FH3.write('# eps_Nb2_err = {0}\n'.format(param_errors[2]))
FH3.write('# eps_Se1_err = {0}\n'.format(param_errors[3]))
FH3.write('# eps_Se2_x_err = {0}\n'.format(param_errors[4]))
FH3.write('# eps_Se2_z_err = {0}\n'.format(param_errors[5]))
FH3.write('# Cr_occupancy_err = {0}\n'.format(param_errors[6]))

# Write structure factors after optimization
FH3.write('# H  K  L  F_calculated  F_observed  F_observed_error\n')
for i in range(len(HKL_list)):
    FH3.write('{0} {1} {2} {3} {4} {5}\n'.format(
        HKL_list[i][0], HKL_list[i][1], HKL_list[i][2], 
        model(HKL_list, opt_params)[i] * opt_params[0], 
        F_obs[i] * opt_params[0], 
        F_obs_err[i] * opt_params[0]
    ))




##===================== Gnuplot Output ======================================
# Define file paths
output_file = os.path.join('.', "Fobs_vs_Fcal_for_optimization.png")  # Output image file from Gnuplot
script_filename = os.path.join('.', "PONTA_plot_template_for_optimization.gp")  # Gnuplot script file
FH3 = os.path.join('.', "result_for_optimization.txt")  # Data file used by Gnuplot

# Create Gnuplot script
gnuplot_script = f"""\
datafile1='{FH3}'

set title "nuclear scattering at HHL+H0L (optimized)" font "Arial,18"

# for png
set term png
set out '{output_file}'

set style line 1 lt 1 lc "#ff0000" lw 1 pt 7 ps 1.5
set style line 2 lt 1 lc "#0000FF" lw 1 pt 7 ps 1.5
set style line 11 lt 1 lc "#000000" lw 1 pt 6 ps 1.5

set size ratio 1  

set xlabel '|F_{{cal}}|' font "Arial,18"
set ylabel '|F_{{obs}}|' font "Arial,18"

set xrange[0:200]
set yrange[0:200]

plot datafile1 u 4:5:6 with yer pt 7 ps 1.5 lc "red" notitle ,\
x notitle
"""

# Write the Gnuplot script to a file (save in optimized)
with open(script_filename, "w") as f:
    f.write(gnuplot_script)

# Execute Gnuplot
subprocess.run(["gnuplot", script_filename])

# Output results
print(f"Gnuplot script saved at: {script_filename}")
print(f"Plot output saved at: {output_file}")
print(f"Data file used: {FH3}")