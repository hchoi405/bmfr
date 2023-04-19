import os
import numpy as np
import exr
import parmap
from functools import partial
from skimage.metrics import structural_similarity
import glob

scenes = [
    ("data_Arcade", 103),
    ("data_BistroExterior", 103),
    ("data_BistroExterior2", 103),
    ("data_Classroom", 603),
    ("data_Dining-room", 103),
    ("data_Dining-room-dynamic", 140),
    ("data_Staircase", 453)
]
positions = [0.01, 0.03, 0.04, 0.1, 0.5, 1.0, 2.0]
normals = [0.01, 0.1, 0.2, 0.5, 1.0, 2.0]

def mse_loss(img1, img2):
    return np.mean((img1 - img2) ** 2)

def relmse_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    true_mean = np.mean(y_true, axis=2, keepdims=True)
    return np.average(np.square(y_pred - y_true) / (np.square(true_mean) + 1e-2))

# Definition from ANF
def smape_loss(y_true, y_pred):
    numerator = np.abs(y_true - y_pred).sum(axis=2, keepdims=True)
    denominator = np.abs(y_true).sum(axis=2, keepdims=True) + \
        np.abs(y_pred).sum(axis=2, keepdims=True)
    return np.mean(numerator / (denominator + 1e-2)) / 3

def tone_mapping(y):
    y = np.clip(y, 0, None)  # non-negative
    y = np.power(y / (y + 1), 1 / 2.2)  # gamma correction
    return y

def psnr_loss(y_true, y_pred):
    # HDR -> LDR
    y_true = tone_mapping(y_true)
    y_pred = tone_mapping(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    return -10 * np.log10(mse)

def ssim_loss(y_true, y_pred):
    val = structural_similarity(y_true, y_pred, channel_axis=2, data_range=y_pred.max() - y_pred.min())
    return val

def process_frame(scene, max_frames, position, normal, frame, loss_fn):
    frame_path = f'/home/hchoi/nas/bmfr/outputs/{scene}/{position:.6f}_{normal:.6f}/bmfr_{frame:04d}.exr'
    ref_path = f'/media/hchoi/extra/{scene}/ref_modul_{frame:04d}.exr'
    
    frame_img = exr.read_all(os.path.expanduser(frame_path))['default']
    ref_img = exr.read_all(ref_path)['default']
    
    return loss_fn(ref_img, frame_img)

def find_best_params_for_scene(scene, max_frames, loss_fn):
    best_params = None
    best_loss = float('inf')
    best_losses = {}
    # Min, max, mean and std
    min_loss = float('inf')
    max_loss = float('-inf')
    mean_loss = 0
    std_loss = 0
    avg_losses = []

    for position in positions:
        for normal in normals:
            dir = f'losses/{scene}/{loss_fn.__name__}'
            os.makedirs(dir, exist_ok=True)
            loss_path = f'{dir}/{position**2:.6f}_{normal**2:.6f}_*.txt'
            # Check if whether the loss file exists
            if len(glob.glob(loss_path)) > 0:
                # If exists, load the loss
                with open(glob.glob(loss_path)[0], 'r') as f:
                    losses = [float(line) for line in f.readlines()]
                avg_loss = np.mean(losses)
                avg_losses.append(avg_loss)
                
            else:
                # Or calculate the loss
                process_frame_partial = partial(process_frame, scene, max_frames, position**2, normal**2, loss_fn=loss_fn)
                losses = parmap.map(process_frame_partial, range(0, max_frames), pm_processes=20, pm_pbar=True)
                avg_loss = np.mean(losses)
                avg_losses.append(avg_loss)
                # Save losses to file
                with open(f'{dir}/{position**2:.6f}_{normal**2:.6f}_{avg_loss:.6f}.txt', 'w') as f:
                    for loss in losses:
                        f.write(f'{loss}\n')

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = (position**2, normal**2)
                best_losses = {frame: loss for frame, loss in enumerate(losses) if frame % 50 == 0}
    
    min_loss = np.min(avg_losses)
    max_loss = np.max(avg_losses)
    mean_loss = np.mean(avg_losses)
    std_loss = np.std(avg_losses)

    return {
        "params": best_params,
        "loss": best_loss,
        "stats": {"min": min_loss, "max": max_loss, "mean": mean_loss, "std": std_loss},
        "losses_by_frame": best_losses
    }


if __name__ == "__main__":
    for scene, max_frames in scenes:
        loss_fns = [mse_loss, relmse_loss, ssim_loss, psnr_loss, smape_loss]
        for loss_fn in loss_fns:
            res = find_best_params_for_scene(scene, max_frames, loss_fn=loss_fn)
            print(f"[{scene}, {loss_fn.__name__}] loss: Position {res['params'][0]:.6f}, Normal {res['params'][1]:.6f}, Loss {res['loss']:.6f}")
            print(f'\tMin: {res["stats"]["min"]:.6f}, Max: {res["stats"]["max"]:.6f}, Mean: {res["stats"]["mean"]:.6f}, Std: {res["stats"]["std"]:.6f}')
        print(f'')

       
