import os
import re
import subprocess
import exr
import numpy as np

import multiprocessing as mp
import parmap

import matplotlib.pyplot as plt
from calculate_loss import *

os.chdir("opencl")

PREFIX = "#define INPUT_DATA_PATH "
scenes = [
    # ("BistroExterior2", 101), # Done
    # ("BistroExteriorDynamic", 101),
    # ("EmeraldSquare2", 101),
    # ("classroom", 101),
    ("musicroom21", 101),
]
TEST_COUNT = scenes[0][1]
dataset_dir = "/home/hchoi/nas/dataset_newscene/"
dst_dir = "/home/hchoi/nas/bmfr"
losses_dir = "/home/hchoi/nas/bmfr/losses"

positions = [0.01, 0.03, 0.04, 0.1, 0.5, 1.0, 2.0]
normals = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0]

for scene in scenes:
    print(f"Processing {scene[0]}...")
    # # Load reference exr file
    # def load_ref(i):
    #     ref = exr.read_all(f"/media/hchoi/extra/{scene}/ref_modul_{i:04d}.exr")['default']
    #     return ref

    # with mp.Pool(processes=20) as pool:
    #     refs = parmap.map(load_ref, range(101), pm_pbar=True)

    frame_count = scene[1]
    input_directory = os.path.join(dataset_dir, scene[0])

    # Change scene
    new_code = f"{PREFIX}{input_directory}"
    with open("./bmfr.cpp", "r") as f:
        content = f.read()
        new_content = re.sub(
            f"{PREFIX}.*$",
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
    replacement = r"\g<1>{}".format(TEST_COUNT)
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    with open("bmfr.cpp", "w") as file:
        file.write(new_content)

    # Iterate over the two arrays using nested for loops
    tuples = []
    for position in positions:
        for normal in normals:
            position_s = round(position ** 2, 6)
            normal_s = round(normal ** 2, 6)
            # Print the values of the current iteration
            print(f"Position: {position_s}, Normal: {normal_s}: ", end=" ", flush=True)

            # Check if the destination directory already exist
            dirname = f"{scene[0]}/{position_s:.6f}_{normal_s:.6f}"
            dst_path = f"{dst_dir}/outputs/{dirname}"
            if os.path.exists(dst_path):
                print("Already exists. Skipping...")
                continue

            # Change position/normal threshold for reprojection
            with open(f"{input_directory}/camera_matrices.h", "r") as f:
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
            with open(f"{input_directory}/camera_matrices.h", "w") as f:
                f.write(new_content)

            # Touch to trigger re-compile
            os.utime("bmfr.cpp", None)

            # no output
            subprocess.run(["make"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["./bmfr"])

            # Move all output exr files in outputs to a new directory
            if not os.path.exists(f"outputs/{dirname}"):
                os.makedirs(f"outputs/{dirname}")

            for i in range(TEST_COUNT):
                if os.path.exists(f"outputs/bmfr_{i:04d}.exr"):
                    os.rename(f"outputs/bmfr_{i:04d}.exr",
                              f"outputs/{dirname}/bmfr_{i:04d}.exr")
            print("Done.")

            # Make sure the directory exists in /home/hchoi/nas
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            # Move all files through rsync and remove original files in synchronous process
            subprocess.run(["rsync", 
                              "-avh", "--info=progress2", "--remove-source-files", 
                              "--no-o", "--no-g", "--no-perms", 
                              f"outputs/{dirname}/", dst_path], check=True)

    # Calculate errors
    # loss_fns = [relmse_loss, ssim_loss, psnr_loss, FLIP]
    loss_fns = [relmse_loss, ssim_loss]
    for scene, max_frames in scenes:
        print(f"Calculating errors for {scene}...")
        for loss_fn in loss_fns:
            print(f'{loss_fn.__name__}:')
            losses = find_best_params_for_loss(dataset_dir, f'{dst_dir}/outputs', losses_dir, scene, TEST_COUNT, positions, normals, loss_fn=loss_fn)
            avg_loss = np.mean(losses['avg'])
            min_loss = np.min(losses['avg'])
            max_loss = np.max(losses['avg'])
            std_loss = np.std(losses['avg'])
            best_params = losses['best_params']
            best_loss = losses['best_loss']
            print(f'Scene {scene}:')
            print(f'\tBest position_normal: {best_params}')
            print(f'\tBest loss: {best_loss:.6f}')
            print(f'\tavg loss: {avg_loss:.6f}')
            print(f'\tmin loss: {min_loss:.6f}')
            print(f'\tmax loss: {max_loss:.6f}')
            print(f'\tstd loss: {std_loss:.6f}')
        print(f'')
