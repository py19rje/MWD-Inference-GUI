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

import os
import re
import numpy as np
from modules.utils import lognormal

def parse_mwd_file(file_path, class_to_use, m):
    if class_to_use == 0: # Polydisperse
        datafile = np.loadtxt(file_path, delimiter='\t', skiprows=1)
        data = datafile[datafile[:, 0].argsort()]
        if data[0,0] < 5:
            m_data = 10**data[:, 0]
        else:
            m_data = data[:,0]
        x_data = np.log(m_data)
        y_data = data[:, 1]
        y_data = y_data / np.trapz(y_data, x=x_data)
        
        m_data = m_data
        y_data_GPC = y_data
        
                            
        GPC_Mn = 1/np.trapz(y_data*np.exp(-x_data), x=x_data)
        GPC_Mw = np.trapz(y_data*np.exp(x_data), x=x_data)
        GPC_PDI = GPC_Mw/GPC_Mn
        
        stats = [GPC_Mn, GPC_Mw, GPC_PDI]
        return m_data, y_data_GPC, stats
    elif class_to_use == 1: # Monodisperse
        data = np.loadtxt(file_path, delimiter='\t')
        Mw_data = data[0]
        PDI_data = data[1]
        sigma_data = np.sqrt(np.log(PDI_data))
        mean_data = np.log(Mw_data)-(sigma_data**2)/2
        y_data = lognormal(m, mean_data, sigma_data)
        y_data = y_data / np.trapz(y_data, x = np.log(m))
                        
        GPC_Mw = Mw_data
        GPC_Mn = Mw_data / PDI_data
        GPC_PDI = PDI_data        
        
        stats = [GPC_Mn, GPC_Mw, GPC_PDI]
        return y_data, stats
    elif class_to_use == 2: # Bidisperse
        data = np.loadtxt(file_path, delimiter='\t')
        
        phi_values_data = data[:,0]
        Mw_values_data = data[:,1]
        PDI_values_data = data[:,2]

        phiL_data,phiS_data = phi_values_data[0],phi_values_data [1]
        MwL_data,MwS_data = Mw_values_data[0],Mw_values_data[1]
        PDIL_data,PDIS_data = PDI_values_data[0],PDI_values_data[1]
        
        sigmaL_data = np.sqrt(np.log(PDIL_data))
        sigmaS_data = np.sqrt(np.log(PDIS_data))

        mean_L_data = np.log(MwL_data)-(sigmaL_data**2)/2
        mean_S_data = np.log(MwS_data)-(sigmaS_data**2)/2

        y_data = phiL_data * lognormal(m, mean_L_data, sigmaL_data) + phiS_data * lognormal(m, mean_S_data, sigmaS_data)
        y_data = y_data / np.trapz(y_data, x = np.log(m))
        
        y_data_GPC = y_data
        
        GPC_MwS = MwS_data
        GPC_MwL = MwL_data
        
        GPC_PDIS = PDIS_data
        GPC_PDIL = PDIL_data
        
        stats = [GPC_MwS, GPC_MwL, phiL_data, phiS_data, GPC_PDIS, GPC_PDIL]
        return y_data_GPC, stats
