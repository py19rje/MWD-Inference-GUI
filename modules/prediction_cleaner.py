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
from modules.utils import sum_of_lognormals_Z
from modules.utils import format_e
from decimal import Decimal

def clean_prediction_curve(
    raw_prediction,
    method,
    *,
    min_x=None,
    max_x=None,
    dump_below=None,
    threshold=0.0,
    z=None,
    M_e=None,
    means_Z=None,
    sigma_poly=None,
    ):
    
    prediction_to_clean = np.array(raw_prediction, dtype=float, copy=True)

    sum_cleaned = 0.0

    if min_x is not None:
        for n, value in enumerate(prediction_to_clean):
            mass = np.exp(means_Z[n] + sigma_poly**2 / 2) * M_e
            if mass < min_x:
                sum_cleaned += value
                prediction_to_clean[n] = 0.0

    if max_x is not None:
        for n, value in enumerate(prediction_to_clean):
            mass = np.exp(means_Z[n] + sigma_poly**2 / 2) * M_e
            if mass > max_x:
                prediction_to_clean[n] = 0.0

    if method == "Min/Max M Range with low tail redistribution" and sum_cleaned > 0:
        values_where_dump = np.zeros(len(prediction_to_clean))
        for n in range(len(prediction_to_clean)):
            mass = np.exp(means_Z[n] + sigma_poly**2 / 2) * M_e
            if min_x is not None and mass > min_x and mass < dump_below:
                values_where_dump[n] = prediction_to_clean[n]

        total_dump = np.sum(values_where_dump)
        fractions_to_dump = values_where_dump / total_dump if total_dump > 0 else values_where_dump
        to_dump = fractions_to_dump * sum_cleaned
        prediction_to_clean += to_dump

    if threshold != 0 and threshold < np.max(raw_prediction):
        prediction_to_clean[1.5 * prediction_to_clean < threshold] = 0.0

    cleaned_curve = sum_of_lognormals_Z(z, prediction_to_clean)
    cleaned_curve = cleaned_curve / np.trapz(cleaned_curve, x=np.log(z*M_e))
    return prediction_to_clean, cleaned_curve


def clean_predictions(predictions, method, params, *, z, m, M_e, means_Z, sigma_poly):
    cleaned = {}
    model_stats_dict = {}
    mn_list, mw_list, pdi_list = [], [], []
    for label, pred in predictions.items():
        raw, curve = clean_prediction_curve(
            pred["raw"],
            method,
            min_x=params.get("min_x"),
            max_x=params.get("max_x"),
            dump_below=params.get("dump_below"),
            threshold=params.get("threshold", 0.0),
            z=z,
            M_e=M_e,
            means_Z=means_Z,
            sigma_poly=sigma_poly,
        )
        cleaned[label] = {"raw": raw, "curve": curve}
        Mn = 1 / np.trapz(curve * np.exp(-np.log(m)), x=np.log(m))
        Mw = np.trapz(curve * m, x=np.log(m))
        PDI = Mw / Mn
        mn_list.append(Mn)
        mw_list.append(Mw)
        pdi_list.append(PDI)
        model_stats_dict[label] = [f"{format_e(Decimal(Mn))} (g/mol)", f"{format_e(Decimal(Mw))} (g/mol)", f"{PDI:.2f}"]
    if len(mn_list) > 1:
        mean_Mn = format_e(Decimal(np.mean(mn_list)))
        mean_Mw = format_e(Decimal(np.mean(mw_list)))
        mean_PDI = f"{np.mean(pdi_list):.2f}"
        model_stats_dict["Mean"] = [f"{mean_Mn} (g/mol)", f"{mean_Mw} (g/mol)", mean_PDI]

    return cleaned, model_stats_dict