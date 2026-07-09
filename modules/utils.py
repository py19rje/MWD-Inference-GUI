#  MWD-Inference-GUI
#  --------------------------------------------------------------------------------------------------------

#  Authors:
#      Robert J. Elliott, py19rje@leeds.ac.uk
#      Daniel J. Read, d.j.read@leeds.ac.uk
#      Luisa Cutillo, l.cutillo@leeds.ac.uk
#      Johan Mattsson, k.j.l.mattsson@leeds.ac.uk

#  GitHub:
#      https://github.com/py19rje/MWD-Inference-GUI

#  --------------------------------------------------------------------------------------------------------

#  Copyright (2026): Robert J. Elliott, University of Leeds

#  This file is part of the software MWD-Inference-GUI

#  MWD-Inference-GUI is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  MWD-Inference-GUI is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this software.  If not, see <http://www.gnu.org/licenses/>.

#  This work forms part of the research programme of DPI, project \#861.

#  --------------------------------------------------------------------------------------------------------

import numpy as np
import math

def flory_schulz(m, Mn):
    return m/(Mn**2) * np.exp(-m/Mn)

def lognormal(x, mean, sigma):
    return (1/np.sqrt(2*math.pi*sigma**2)) * np.exp(-(np.log(x) - mean)**2 / (2 * sigma**2))
  
def sum_of_lognormals_Z(x, *weights):
    if isinstance(weights[0], (list, np.ndarray)):  
        weights = weights[0]
    result = np.zeros(len(x))

    for i in range(num_params):
        result = result + (weights[i] * lognormal(x, means_Z[i], sigma_poly))
    return result

M_e_PE = 820

num_params = 34

means = np.linspace(np.log(0.1*M_e_PE), np.log(10000*M_e_PE), 28) 
known_means = np.linspace(np.log(10*M_e_PE), np.log(1000*M_e_PE), 7) 
means_ratio = np.exp(means[1])/np.exp(means[0])
known_means_ratio = np.exp(known_means[1])/np.exp(known_means[0])
sigma_poly = 0.55 * (means_ratio/known_means_ratio)

means_Z = np.linspace(np.log(8e-3), np.log(8e+3), num_params)

m = np.logspace(1.5,7,num=400,base=10)
x = np.log(m)

PREDICTION_COLORS = [
    'orange', 'green', 'red', 'purple', 'brown', 'magenta', 'cyan', 'olive', 'navy', 'teal'
]

pred_alpha = 0.6

def G_prime_fit(w_alpha, g_values, tau_values):
    sum_over_i = 0.0
    w_squared = w_alpha*w_alpha
    numerator = g_values * tau_values * tau_values
    denom = 1 + (w_squared * tau_values * tau_values)
    to_sum = numerator/denom
    sum_over_i = np.sum(to_sum)
    G_prime_for_alpha = w_squared * sum_over_i
    return G_prime_for_alpha

def G_dub_prime_fit(w_alpha, g_values, tau_values):
    sum_over_j = 0.0
    w_squared = w_alpha*w_alpha
    numerator = g_values * tau_values
    denom = 1 + (w_squared * tau_values * tau_values)
    to_sum = numerator/denom
    sum_over_j = np.sum(to_sum)
    G_dub_prime_for_alpha = w_alpha * sum_over_j
    return G_dub_prime_for_alpha

def G_concat_fit(w_values, *log_g_values, tau_values, G_concat_inp_LN, lam):
    log_g_second_diffs = np.array([])
    for i in range(len(log_g_values)-2):
        g_i_second_diff = log_g_values[i+2] + log_g_values[i] - 2*log_g_values[i+1]
        log_g_second_diffs = np.append(log_g_second_diffs, g_i_second_diff)
    g_values = np.exp(log_g_values)
    G_prime_fit_values = np.array([])
    G_dub_prime_fit_values = np.array([])
    for alpha in w_values:
        G_prime_max = G_prime_fit(alpha, g_values, tau_values)
        G_prime_fit_values = np.append(G_prime_fit_values, G_prime_max)
        G_dub_prime_max = G_dub_prime_fit(alpha, g_values, tau_values)
        G_dub_prime_fit_values = np.append(G_dub_prime_fit_values, G_dub_prime_max)
    G_concat = np.concatenate((G_prime_fit_values, G_dub_prime_fit_values), axis=0) 
    G_log_concat_diff = (np.log(G_concat) - G_concat_inp_LN) 
    G_concat_with_diffs = np.concatenate((G_log_concat_diff, lam*log_g_second_diffs), axis = 0)
    return G_concat_with_diffs

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + ' E' + a.split('E')[1]
