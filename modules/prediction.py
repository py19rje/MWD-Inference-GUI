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
import torch
from modules.utils import lognormal, sum_of_lognormals_Z

def build_prediction_tensor(optimiseresult):
    X_val = np.log10(np.exp(np.stack((optimiseresult, optimiseresult))))
    X_val = np.concatenate((X_val, X_val), axis=1)
    X_val = np.reshape(X_val, (2, 1, 2, -1))
    return torch.tensor(X_val, dtype=torch.float32)

def make_prediction_curve(prediction_np, class_to_use, M_e, z, x):
    if class_to_use == 0:  # Polydisperse
        curve = sum_of_lognormals_Z(z, prediction_np[0])
        curve = curve / np.trapz(curve, x=np.log(z))
        Mn = 1 / np.trapz(curve * np.exp(-x), x=x)
        Mw = np.trapz(curve * np.exp(x), x=x)
        PDI = Mw / Mn
        stats = [Mn, Mw, PDI]
    elif class_to_use == 1:  # Monodisperse
        PDI = 1.03
        sigma_mono = np.sqrt(np.log(PDI))
        Z_pred = 10**float(prediction_np[0])
        mean_Z_pred = np.log(Z_pred) - (sigma_mono**2) / 2
        curve = lognormal(z, mean_Z_pred, sigma_mono)
        curve = curve / np.trapz(curve, x=np.log(z))
        Mw = Z_pred * M_e
        Mn = Mw / PDI
        stats = [Mn, Mw, PDI]
    else:  # Bidisperse
        PDI = 1.03
        sigma_bin = np.sqrt(np.log(PDI))
        ZS_pred = 10**float(prediction_np[0][0])
        ZL_pred = 10**float(prediction_np[0][1])
        phiL_pred = float(prediction_np[0][2])
        phiS_pred = 1 - phiL_pred
        mean_ZS_pred = np.log(ZS_pred) - (sigma_bin**2) / 2
        mean_ZL_pred = np.log(ZL_pred) - (sigma_bin**2) / 2
        curve = phiL_pred * lognormal(z, mean_ZL_pred, sigma_bin) + phiS_pred * lognormal(z, mean_ZS_pred, sigma_bin)
        curve = curve / np.trapz(curve, x=np.log(z))
        stats = [ZS_pred, ZL_pred, phiL_pred, phiS_pred, PDI]
    return curve, stats

