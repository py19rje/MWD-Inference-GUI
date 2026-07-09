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
import torch
from NN_models.PytorchPoly_model import PolyModel
from NN_models.PytorchMono_model import MonoModel
from NN_models.PytorchBinary_model import BinaryModel

CLASS_TO_MODEL = {
    0: PolyModel,
    1: MonoModel,
    2: BinaryModel,
}
CLASS_LABEL = {
    0: "Polydisperse",
    1: "Monodisperse",
    2: "Bidisperse",
}

def find_model_files(class_to_use, directory="NN_models"):
    class_filter = CLASS_LABEL[class_to_use]
    return [f for f in os.listdir(directory)
            if class_filter in f and f.endswith(".pth")]

def reduce_model_name(name):
    if "_" in name:
        prefix, suffix = name.split("_", 1)
        i = ''.join(c for c in suffix if c.isdigit())
        prefix_short = {"Polydisperse":"Poly","Monodisperse":"Mono","Bidisperse":"Bi"}.get(prefix, prefix)
        return f"{prefix_short}_{i}" if i else prefix_short
    return name

def load_models(selected_display_names, class_to_use, directory="NN_models", device=None):
    ModelClass = CLASS_TO_MODEL[class_to_use]
    device = device or torch.device('cpu')
    models = []
    loaded_names = []
    reduced_names = []
    errors = []

    for display_name in selected_display_names:
        file_path = os.path.join(directory, f"{display_name}.pth")
        try:
            model = ModelClass()
            model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
            model.eval()
            models.append(model)
            loaded_names.append(f"{display_name}.pth")
            reduced_names.append(reduce_model_name(display_name))
        except Exception as exc:
            errors.append((display_name, str(exc)))

    return models, loaded_names, reduced_names, errors