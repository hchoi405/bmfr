import os
import numpy as np
import exr
import parmap
from functools import partial
from skimage.metrics import structural_similarity
import glob
import itertools
import subprocess
import shutil
import uuid

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

def FLIP(y_pred, y_true):
    y_pred = np.maximum(y_pred, 0.0)
    y_true = np.maximum(y_true, 0.0)

    # Copy to temporary directory
    tmp_dir = './tmp'
    os.makedirs('./tmp', exist_ok=True)
    pred_path = os.path.join(tmp_dir, f'y_pred_{str(uuid.uuid4())}.exr')
    true_path = os.path.join(tmp_dir, f'y_true_{str(uuid.uuid4())}.exr')
    exr.write(pred_path, y_pred, compression=exr.ZIP_COMPRESSION)
    exr.write(true_path, y_true, compression=exr.ZIP_COMPRESSION)

    # Run flip
    try:
        result = subprocess.run(['python', './flip.py', '--no-exposure-map', '--no-error-map', '-r', true_path, '-t', pred_path], stdout=subprocess.PIPE)
        ret_str = result.stdout.decode("utf-8")
        # print(ret_str)

        # Parse the Mean: value
        strs = ret_str.split('\n')
        for i in range(len(strs)):
            if 'Mean:' in strs[i]:
                mean_str = strs[i]
                mean_err = float(mean_str.split(':')[-1])
                break
    except Exception as e:
        print(e)
        exit(-1)

    # Remove files
    os.remove(pred_path)
    os.remove(true_path)

    # flipErrorMap, meanFLIPError, parameters = flip.evaluate(y_true, y_pred, "HDR")
    # return flipErrorMap

    return mean_err

img_cache = {}
def process_frame(dataset_dir, output_dir, scene, max_frames, position, normal, frame, loss_fn):
    frame_path = os.path.join(output_dir, scene, f'{position:.6f}_{normal:.6f}', f'bmfr_{frame:04d}.exr')
    ref_path = os.path.join(dataset_dir, scene, f'ref_{frame:04d}.exr')

    if frame_path not in img_cache:
        img_cache[frame_path] = exr.read_all(frame_path)['default']
    if ref_path not in img_cache:
        img_cache[ref_path] = exr.read_all(ref_path)['default']
    frame_img = img_cache[frame_path]
    ref_img = img_cache[ref_path]
    
    return loss_fn(ref_img, frame_img)

def find_best_params_for_loss(dataset_dir, output_dir, losses_dir, scene, max_frames, positions, normals, loss_fn):
    scene_losses = {}
    param_losses = []

    # Iterate position and normal combinations by permute
    for position, normal in itertools.product(positions, normals):
        # print(f'{scene} {pos**2:.6f} {norm**2:.6f}')
        position = position ** 2
        normal = normal ** 2
        key = f'{position:.6f}_{normal:.6f}'

        regenerate = False
        dir = os.path.join(losses_dir, scene, loss_fn.__name__)
        os.makedirs(dir, exist_ok=True)
        loss_path = os.path.join(dir, f'{key}.txt')
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
            process_frame_partial = partial(process_frame, dataset_dir, output_dir, scene, max_frames, position, normal, loss_fn=loss_fn)
            num_cores = loss_fn.__name__ == 'FLIP' and 4 or 20
            losses = parmap.map(process_frame_partial, range(0, max_frames), pm_processes=num_cores, pm_pbar=True)
            avg_loss = np.mean(losses)
            param_losses.append({key: avg_loss})
            # Save losses to file
            with open(os.path.join(dir, f'{position:.6f}_{normal:.6f}.txt'), 'w') as f:
                for loss in losses:
                    f.write(f'{loss}\n')

    params = [list(param.keys())[0] for param in param_losses]
    avg_losses = [list(val.values())[0] for val in param_losses]
    # print(losses)
    if loss_fn == ssim_loss or loss_fn == psnr_loss:
        best_idx = np.argmax(avg_losses)
    else:
        best_idx = np.argmin(avg_losses)
    scene_losses['avg'] = avg_losses
    scene_losses['param_losses'] = param_losses
    scene_losses['best_params'] = params[best_idx]
    scene_losses['best_loss'] = avg_losses[best_idx]
    return scene_losses
