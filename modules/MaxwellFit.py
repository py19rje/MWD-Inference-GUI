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
import scipy.optimize as opt
from modules.utils import G_concat_fit, G_prime_fit, G_dub_prime_fit

def build_maxwell_fit_inputs(w_values, Gp_data, Gpp_data, tau_H, tau_L, modes_per_decade):
    G_concat_inp_nat = np.concatenate((Gp_data, Gpp_data))
    G_concat_inp_ln = np.log(G_concat_inp_nat)

    longtau = -tau_L
    smalltau = -tau_H
    numoftau = 1 + (longtau - smalltau) * modes_per_decade
    tau_values = np.logspace(longtau, smalltau, int(numoftau), base=10)

    return {
        "w_values": w_values,
        "G_concat_inp_nat": G_concat_inp_nat,
        "G_concat_inp_ln": G_concat_inp_ln,
        "tau_values": tau_values,
        "numoftau": int(numoftau),
    }
    
def make_initial_guess(G_concat_inp_ln, numoftau, numomega):
    if numoftau < numomega:
        G_split = np.array_split(G_concat_inp_ln[0:int(numomega)], numoftau)
        G_dub_split = np.array_split(G_concat_inp_ln[int(numomega):], numoftau)
        log_prime_means = np.array([np.mean(subarray) for subarray in G_split])
        log_dub_means = np.array([np.mean(subarray) for subarray in G_dub_split])
        return (log_prime_means + log_dub_means) / 2 - 2

    desired_size = numoftau
    interpolated_indices = np.linspace(0, G_concat_inp_ln.shape[0] / 2 - 1, desired_size)
    interpolated_prime = np.interp(
        interpolated_indices,
        np.arange(G_concat_inp_ln.shape[0] / 2),
        G_concat_inp_ln[0:int(numomega)]
    )
    interpolated_dub = np.interp(
        interpolated_indices,
        np.arange(G_concat_inp_ln.shape[0] / 2),
        G_concat_inp_ln[int(numomega):]
    )
    return (interpolated_prime + interpolated_dub) / 2 - 2

def fit_maxwell_spectrum(
    w_values,
    Gp_data,
    Gpp_data,
    tau_H,
    tau_L,
    modes_per_decade,
    univ_space,
    fit_func,
):
    fit_inputs = build_maxwell_fit_inputs(
        w_values,
        Gp_data,
        Gpp_data,
        tau_H,
        tau_L,
        modes_per_decade,
    )

    tau_values = fit_inputs["tau_values"]
    G_concat_inp_ln = fit_inputs["G_concat_inp_ln"]
    numoftau = fit_inputs["numoftau"]
    numomega = len(w_values)

    if univ_space:
        bounds = ([-100] * numoftau, [50] * numoftau)
    else:
        bounds = ([-40] * numoftau, [50] * numoftau)

    initial_guess = make_initial_guess(G_concat_inp_ln, numoftau, numomega)

    p_w = len(w_values) / (max(np.log10(w_values)) - min(np.log10(w_values)))
    p_w_base = 95 / 14
    lam = 1 * (p_w / p_w_base) ** 0.5

    G_len_zeros = np.zeros(2 * len(w_values) + len(tau_values) - 2)

    optimiseresult, _ = opt.curve_fit(
        lambda w, *p: fit_func(w, *p, tau_values=tau_values, G_concat_inp_LN=G_concat_inp_ln, lam=lam),
        w_values,
        G_len_zeros,
        p0=initial_guess,
        bounds=bounds,
        method="trf",
    )

    pred_G_prime = np.array([])
    pred_G_dub_prime = np.array([])
    for alpha in w_values:
        predicted_Gprime = G_prime_fit(alpha, np.exp(optimiseresult), tau_values)
        pred_G_prime = np.append(pred_G_prime, predicted_Gprime)
        predicted_Gdubprime = G_dub_prime_fit(alpha, np.exp(optimiseresult), tau_values)
        pred_G_dub_prime = np.append(pred_G_dub_prime, predicted_Gdubprime)

    return {
        "optimiseresult": optimiseresult,
        "tau_values": tau_values,
        "pred_G_prime": pred_G_prime,
        "pred_G_dub_prime": pred_G_dub_prime,
        "mode_omega": 1 / tau_values,
    }