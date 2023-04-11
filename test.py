import os
import re
import subprocess
import exr
import numpy as np

os.chdir("opencl")

PREFIX = "#define INPUT_DATA_PATH "
scenes = ["bistro2"]
directory_string = "/media/hchoi/extra"

# Define an empty array to hold the values
positions = [0.01, 0.03, 0.04, 0.1, 0.5, 1.0, 2.0]
normals = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
scale = 6

for scene in scenes:
    # Load reference exr file
    ref = exr.read_all(f"/media/hchoi/extra/sequence/{scene}/{scene}_ref/ref_0100.exr")['default']

    print("scene:", scene)
    scene_string = f"{scene}_anim"
    new_directory = f"{directory_string}/{scene_string}"
    # camera_string="/media/hchoi/extra/${scene}_anim/camera_matrices.h"
    print(os.getcwd())

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

    # Iterate over the two arrays using nested for loops
    rel_mses = []
    for position in positions:
        for normal in normals:
            position_s = round(position ** 2, scale)
            normal_s = round(normal ** 2, scale)
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

            # Move file according to the filename
            os.rename(
                "outputs/bmfr_0100.exr",
                f"outputs/bmfr_{position_s}_{normal_s}.exr",
            )

            # Read the output exr file and compare it to the reference exr file
            out = exr.read_all(f"outputs/bmfr_{position_s}_{normal_s}.exr".format(position_s))['default']
            out = np.clip(out, 0, None)
            ref_mean = np.mean(ref, axis=2, keepdims=True)
            rel_mse = np.average(np.square(out - ref) / (np.square(ref_mean) + 1e-2))
            rel_mses.append((position_s, normal_s, rel_mse))
            print(f"\tRelMSE: {rel_mse:.6f}")
    
    min_relmse = min(rel_mses, key=lambda x: x[2])
    print(min_relmse)
