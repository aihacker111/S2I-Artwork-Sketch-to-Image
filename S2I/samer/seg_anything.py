import os
from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation

np.random.seed(200)
_palette = ((np.random.random((3 * 255)) * 0.7 + 0.3) * 255).astype(np.uint8).tolist()
_palette = [0, 0, 0] + _palette


def save_prediction(predict_mask, output_dir, file_name):
    save_mask = Image.fromarray(predict_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir, file_name))


def colorize_mask(predict_mask):
    save_mask = Image.fromarray(predict_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)


def draw_mask(img, mask, alpha=0.5, id_cnt=False):
    img_mask = img
    if id_cnt:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        for ids in obj_ids:
            # Overlay color on  binary mask
            if ids <= 255:
                color = _palette[ids * 3:ids * 3 + 3]
            else:
                color = [0, 0, 0]
            foreground = img * (1 - alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == ids)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            cnt = binary_dilation(binary_mask, iterations=1) ^ binary_mask
            img_mask[cnt, :] = 0
    else:
        binary_mask = (mask != 0)
        cnt = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[cnt, :] = 0

    return img_mask.astype(img.dtype)
