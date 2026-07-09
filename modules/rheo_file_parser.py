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

def parse_rheo_file(path):
    if not os.path.isfile(path):
        return {"error": f"File not found: {path}"}
    unsorted_w = []
    unsorted_Gp = []
    unsorted_Gpp = []
    negative_w_indices = []
    negative_Gp_indices = []
    negative_Gpp_indices = []
    try:
        with open(path, "r") as fh:
            lines = fh.readlines()
    except Exception as e:
        return {"error": str(e)}

    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[,\s]+", line)
        if len(parts) < 3:
            # skip or collect as warning
            continue
        try:
            v1 = float(parts[0])
            v2 = float(parts[1])
            v3 = float(parts[2])
        except ValueError:
            # collect as warning and skip
            continue
        if v1 <= 0:
            negative_w_indices.append(i)
        if v2 <= 0:
            negative_Gp_indices.append(i)
        if v3 <= 0:
            negative_Gpp_indices.append(i)
        unsorted_w.append(v1)
        unsorted_Gp.append(v2)
        unsorted_Gpp.append(v3)

    filtered_w = []
    filtered_Gp = []
    filtered_Gpp = []
    for w,vp,vpp in zip(unsorted_w, unsorted_Gp, unsorted_Gpp):
        if w > 0 and vp > 0 and vpp > 0:
            filtered_w.append(w)
            filtered_Gp.append(vp)
            filtered_Gpp.append(vpp)

    if len(filtered_w) == 0:
        return {"error": "No valid (positive) data rows found."}

    arr_w = np.array(filtered_w)
    arr_Gp = np.array(filtered_Gp)
    arr_Gpp = np.array(filtered_Gpp)
    sort_idx = np.argsort(arr_w)
    arr_w = arr_w[sort_idx]
    arr_Gp = arr_Gp[sort_idx]
    arr_Gpp = arr_Gpp[sort_idx]

    warnings = []
    if negative_w_indices:
        warnings.append(f"Negative frequency values at line(s): {', '.join(map(str, negative_w_indices))}")
    if negative_Gp_indices:
        warnings.append(f"Negative G' at line(s): {', '.join(map(str, negative_Gp_indices))}")
    if negative_Gpp_indices:
        warnings.append(f"Negative G'' at line(s): {', '.join(map(str, negative_Gpp_indices))}")

    return {
        "w": arr_w,
        "Gp": arr_Gp,
        "Gpp": arr_Gpp,
        "warnings": warnings,
        "error": None
    }