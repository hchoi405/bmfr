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
    ("data_Arcade", 103),
    ("data_BistroExterior", 103),
    ("data_BistroExterior2", 103),
    ("data_Classroom", 603),
    ("data_Dining-room", 103),
    ("data_Dining-room-dynamic", 140),
    ("data_Staircase", 453)
]
directory_string = "/media/hchoi/extra"

# Define an empty array to hold the values
positions = [0.01, 0.03, 0.04, 0.1, 0.5, 1.0, 2.0]
normals = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0]


def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def rmse_loss(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

# Definition from ANF


def smape_loss(y_true, y_pred):
    numerator = np.abs(y_true - y_pred).sum(axis=2, keepdims=True)
    denominator = np.abs(y_true).sum(axis=2, keepdims=True) + \
        np.abs(y_pred).sum(axis=2, keepdims=True)
    return np.reciprocal(3) * np.mean(numerator / (denominator + 1e-2))


def relmse_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    true_mean = np.mean(y_true, axis=2, keepdims=True)
    return np.average(np.square(y_pred - y_true) / (np.square(true_mean) + 1e-2))


def tone_mapping(y):
    y = np.clip(y, 0, None)  # non-negative
    y = np.power(y / (y + 1), 1 / 2.2)  # gamma correction
    return y

# Definition from ANF


def psnr_loss(y_true, y_pred):
    # HDR -> LDR
    y_true = tone_mapping(y_true)
    y_pred = tone_mapping(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    return -10 * np.log10(mse)


def ssim_loss(y_true, y_pred):
    val = structural_similarity(
        y_true, y_pred, channel_axis=2, data_range=y_pred.max() - y_pred.min())
    return val
    # HDR -> LDR
    y_true = tone_mapping(y_true)
    y_pred = tone_mapping(y_pred)

    k1 = 0.01
    k2 = 0.03
    L = 255
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    window = np.ones((11, 11)) / 121

    print('window', window.shape)
    mu1 = np.convolve(y_true, window, mode='valid')
    mu2 = np.convolve(y_pred, window, mode='valid')
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = np.convolve(y_true ** 2, window, mode='valid') - mu1_sq
    sigma2_sq = np.convolve(y_pred ** 2, window, mode='valid') - mu2_sq
    sigma12 = np.convolve(y_true * y_pred, window, mode='valid') - mu1_mu2

    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
        ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return np.mean(ssim)


for scene in scenes:
    # # Load reference exr file
    # def load_ref(i):
    #     ref = exr.read_all(f"/media/hchoi/extra/{scene}/ref_modul_{i:04d}.exr")['default']
    #     return ref

    # with mp.Pool(processes=20) as pool:
    #     refs = parmap.map(load_ref, range(101), pm_pbar=True)

    frame_count = scene[1]

    new_directory = f"{directory_string}/{scene[0]}"

    # Change scene
    new_code = f"{PREFIX}{new_directory}"
    with open("./bmfr.cpp", "r") as f:
        content = f.read()
        new_content = re.sub(
            f".*{directory_string}.*$",
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

    # Iterate over the two arrays using nested for loops
    tuples = []
    for position in positions:
        for normal in normals:
            position_s = round(position ** 2, 6)
            normal_s = round(normal ** 2, 6)
            # Print the values of the current iteration
            print(f"Position: {position_s}, Normal: {normal_s}")

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
            subprocess.run(["./bmfr"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Move all output exr files in outputs to a new directory
            dirname = f"{scene[0]}/{position_s:.6f}_{normal_s:.6f}"
            if not os.path.exists(f"outputs/{dirname}"):
                os.makedirs(f"outputs/{dirname}")

            for i in range(frame_count):
                if os.path.exists(f"outputs/bmfr_{i:04d}.exr"):
                    os.rename(f"outputs/bmfr_{i:04d}.exr",
                              f"outputs/{dirname}/bmfr_{i:04d}.exr")

            # Make sure the directory exists in the NAS
            subprocess.run(["ssh", "nas", f"mkdir -p bmfr/outputs/{dirname}"])

            # Move all files through rsync and remove original files in asynchronous process
            subprocess.Popen(["rsync", "-av", "--remove-source-files", f"outputs/{dirname}/", f"nas:bmfr/outputs/{dirname}/"])

            continue
