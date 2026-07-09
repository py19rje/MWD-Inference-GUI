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

def write_prediction_file(file_path, mass, curve, class_to_use, stats=None):
    curve = np.where(curve < 1e-10, 0, curve)

    with open(file_path, "w") as f:
        if class_to_use in (0, 1):
            if stats is not None:
                f.write(f"Mn = {stats[0]}\n")
                f.write(f"Mw = {stats[1]}\n")
                f.write(f"PDI = {stats[2]}\n")
        elif class_to_use == 2:
            if stats is not None:
                f.write(f"{stats[0]}\n")
                f.write(f"{stats[1]}\n")
                f.write(f"{stats[2]}\n")

        np.savetxt(f, np.column_stack((mass, curve)), delimiter="\t", fmt="%.6e")