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
from modules.utils import flory_schulz, lognormal

def build_lognormal_gpc(mw, pdi, mass_grid=None):
    if mass_grid is None:
        from modules.utils import m
        mass_grid = m

    if mw <= 0:
        raise ValueError("Mw must be positive.")
    if pdi <= 1:
        raise ValueError("PDI must be greater than 1.")

    sigma = np.sqrt(np.log(pdi))
    mean = np.log(mw) - (sigma**2) / 2

    y_data = lognormal(mass_grid, mean, sigma)
    y_data = y_data / np.trapz(y_data, x=np.log(mass_grid))

    return y_data

def build_flory_gpc(mw, mass_grid=None):
    if mass_grid is None:
        from modules.utils import m
        mass_grid = m

    if mw <= 0:
        raise ValueError("Mw must be positive.")

    mn = mw / 2
    y_data = flory_schulz(mass_grid, mn)
    y_data = y_data / np.trapz(y_data, x=np.log(mass_grid))

    return y_data