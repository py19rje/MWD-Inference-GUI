import os
import json

current_dir = os.getcwd()
params_file_path = f"{current_dir}/parameters.json"

def load_parameters(file_path=params_file_path):
    with open(file_path, "r") as file:
        return json.load(file)

