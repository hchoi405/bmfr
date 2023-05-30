import os
import numpy as np
import exr
import parmap
from functools import partial
from skimage.metrics import structural_similarity
import glob
import itertools

scenes = [
    ("data_Arcade", 103),
    # ("data_BistroExterior", 103),
    ("data_BistroExterior2", 103),
    ("data_Classroom", 103),
    ("data_Dining-room", 103),
    # ("data_Dining-room-dynamic", 103),
    ("data_Staircase", 103)
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
    # HDR -> LDR
    y_true = tone_mapping(y_true)
    y_pred = tone_mapping(y_pred)

    val = structural_similarity(y_true, y_pred, channel_axis=2, data_range=y_pred.max() - y_pred.min())
    return val

img_cache = {}
def process_frame(scene, max_frames, position, normal, frame, loss_fn):
    frame_path = f'/home/hchoi/nas/bmfr/outputs/{scene}/{position:.6f}_{normal:.6f}/bmfr_{frame:04d}.exr'
    ref_path = f'/media/hchoi/extra/{scene}/ref_modul_{frame:04d}.exr'

    if frame_path not in img_cache:
        img_cache[frame_path] = exr.read_all(frame_path)['default']
    if ref_path not in img_cache:
        img_cache[ref_path] = exr.read_all(ref_path)['default']
    frame_img = img_cache[frame_path]
    ref_img = img_cache[ref_path]
    
    return loss_fn(ref_img, frame_img)

def find_best_params_for_loss(loss_fn):
    scene_losses = {}
    for scene, max_frames in scenes:
        scene_losses[scene] = {}
        param_losses = []

        # Iterate position and normal combinations by permute
        for position, normal in itertools.product(positions, normals):
            # print(f'{scene} {pos**2:.6f} {norm**2:.6f}')
            position = position ** 2
            normal = normal ** 2
            key = f'{position:.6f}_{normal:.6f}'

            regenerate = False
            dir = f'losses/{scene}/{loss_fn.__name__}'
            os.makedirs(dir, exist_ok=True)
            loss_path = f'{dir}/{key}.txt'
            # print(loss_path)
            if os.path.exists(loss_path):
                with open(loss_path, 'r') as f:
                    losses = [float(line) for line in f.readlines()]
                if len(losses) < max_frames:
                    print(f'Loss file is not complete. {scene} {position:.6f} {normal:.6f}')
                    regenerate = True
                elif len(losses) > max_frames:
                    losses = losses[:max_frames]
                avg_loss = np.mean(losses)
                param_losses.append({key: avg_loss})
            else:
                regenerate = True
            
            if regenerate:
                print(f'Regenerating... {scene} {position:.6f} {normal:.6f}')
                process_frame_partial = partial(process_frame, scene, max_frames, position, normal, loss_fn=loss_fn)
                losses = parmap.map(process_frame_partial, range(0, max_frames), pm_processes=20, pm_pbar=True)
                avg_loss = np.mean(losses)
                param_losses.append({key: avg_loss})
                # Save losses to file
                with open(f'{dir}/{position:.6f}_{normal:.6f}.txt', 'w') as f:
                    for loss in losses:
                        f.write(f'{loss}\n')
        params = [list(param.keys())[0] for param in param_losses]
        avg_losses = [list(val.values())[0] for val in param_losses]
        # print(losses)
        if loss_fn == ssim_loss or loss_fn == psnr_loss:
            best_idx = np.argmax(avg_losses)
        else:
            best_idx = np.argmin(avg_losses)
        scene_losses[scene]['avg'] = avg_losses
        scene_losses[scene]['param_losses'] = param_losses
        scene_losses[scene]['best_params'] = params[best_idx]
        scene_losses[scene]['best_loss'] = avg_losses[best_idx]
    return scene_losses

if __name__ == "__main__":
    loss_fns = [mse_loss, relmse_loss, ssim_loss, psnr_loss, smape_loss]
    for loss_fn in loss_fns:
        print(f'{loss_fn.__name__}:')
        scene_losses = find_best_params_for_loss(loss_fn=loss_fn)
        for scene, max_frames in scenes:
            avg_loss = np.mean(scene_losses[scene]['avg'])
            min_loss = np.min(scene_losses[scene]['avg'])
            max_loss = np.max(scene_losses[scene]['avg'])
            std_loss = np.std(scene_losses[scene]['avg'])
            best_params = scene_losses[scene]['best_params']
            best_loss = scene_losses[scene]['best_loss']
            print(f'Scene {scene}:')
            print(f'\tBest position_normal: {best_params}')
            print(f'\tBest loss: {best_loss:.6f}')
            print(f'\tavg loss: {avg_loss:.6f}')
            print(f'\tmin loss: {min_loss:.6f}')
            print(f'\tmax loss: {max_loss:.6f}')
            print(f'\tstd loss: {std_loss:.6f}')
        print(f'')

       
