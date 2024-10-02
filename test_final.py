import os
import re
import subprocess
import exr
import numpy as np

import multiprocessing as mp
import parmap

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity

os.chdir("opencl")

PREFIX = "#define INPUT_DATA_PATH "
scenes = [
    # ("BistroExteriorDoF", 101),
    # ("BistroExterior2", 301),
    # ("BistroExteriorDynamic", 301),
    # ("EmeraldSquare2", 301),
    # ("staircase", 301),
    # ("musicroom22", 301),
    ("staircase_dynamic9", 301),
]
directory_string = "/home/hchoi/nas/dataset_nrd4"
dst_dir = "/home/hchoi/nas/bmfr"

# Based on relL2/SSIM losses and ~101 frames
bmfr_best_params = {
    # "BistroExteriorDoF": (0.25, 0.1),
    # "BistroExterior2": (0.25, 0.1),
    # "BistroExteriorDynamic": (0.01, 0.0001),
    # "classroom": (0.0016, 0.25),
    # "EmeraldSquare2": (0.25, 0.0001),
    # "staircase": (0.01, 0.0001),
    # "terrazzo-kitchen": (0.0016, 0.25),
    # "musicroom22": (0.01, 0.0001),
    "staircase_dynamic9": (0.01, 0.0001),
}

if __name__ == "__main__":
    for scene in scenes:
        position_s = bmfr_best_params[scene[0]][0]
        normal_s = bmfr_best_params[scene[0]][1]

        frame_count = scene[1]

        new_directory = os.path.join(directory_string, scene[0])

        # Change scene
        new_code = f"{PREFIX}{new_directory}"
        print(new_code)
        with open("./bmfr.cpp", "r") as f:
            content = f.read()
            new_content = re.sub(
                f".*{PREFIX}.*$",
                new_code,
                content,
                flags=re.MULTILINE,
            )
        with open("./bmfr.cpp", "w") as f:
            f.write(new_content)

        # Change FRAME_COUNT in bmfr.cpp
        with open("bmfr.cpp", "r") as file:
            content = file.read()
        pattern = r"(#define FRAME_COUNT\s+)(\d+)"
        replacement = r"\g<1>{}".format(frame_count)
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        with open("bmfr.cpp", "w") as file:
            file.write(new_content)

        # Print the values of the current iteration
        print(f"Scene {scene[0]} - Position: {position_s}, Normal: {normal_s}")

        # Change position/normal threshold for reprojection
        with open(f"{new_directory}/camera_matrices.h", "r") as f:
            content = f.read()
            new_content = re.sub(
                r"^.*position_limit_squared\s*=\s*[0-9.]*",
                f"const float position_limit_squared = {position_s}",
                content,
                flags=re.MULTILINE,
            )
            new_content = re.sub(
                r"^.*normal_limit_squared\s*=\s*[0-9.]*",
                f"const float normal_limit_squared = {normal_s}",
                new_content,
                flags=re.MULTILINE,
            )
        with open(f"{new_directory}/camera_matrices.h", "w") as f:
            f.write(new_content)

        # Touch to trigger re-compile
        os.utime("bmfr.cpp", None)

        # no output
        subprocess.run(["make"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["./bmfr"])

        # Move all output exr files in outputs to a new directory
        if not os.path.exists(f"outputs/{scene[0]}"):
            os.makedirs(f"outputs/{scene[0]}")

        for i in range(frame_count):
            if os.path.exists(f"outputs/bmfr_{i:04d}.exr"):
                os.rename(f"outputs/bmfr_{i:04d}.exr",
                            f"outputs/{scene[0]}/bmfr_{i:04d}.exr")

        # Make sure the directory exists in /home/hchoi/nas
        if not os.path.exists(f"{dst_dir}/{scene[0]}"):
            os.makedirs(f"{dst_dir}/{scene[0]}")

        # Move all files through rsync and remove original files in asynchronous process
        subprocess.Popen(["rsync", 
                            "-avh", "--info=progress2", "--remove-source-files", 
                            "--no-o", "--no-g", "--no-perms", 
                            f"outputs/{scene[0]}/", f"{dst_dir}/{scene[0]}/"])
