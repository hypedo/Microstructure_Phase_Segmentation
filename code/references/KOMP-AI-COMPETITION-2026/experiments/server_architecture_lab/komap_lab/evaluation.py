import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import PALETTE, CLASS_NAMES, patch_origins
from .metrics import confusion_matrix, class_iou, summarize


@torch.inference_mode()
def predict_image(model, image, context, target, stride, device, batch_size=4, d4=False):
    height, width = image.shape
    origins = patch_origins(height, width, target, stride)
    margin = (context - target) // 2
    padded = np.pad(image, margin, mode='reflect')
    total = torch.zeros(4, height, width)
    coverage = torch.zeros(height, width)
    for start in range(0, len(origins), batch_size):
        locations = origins[start:start + batch_size]
        tiles = torch.stack([torch.from_numpy(padded[y:y+context, x:x+context].copy()).float()[None] / 255
                             for y, x in locations]).to(device)
        probabilities = torch.zeros(len(locations), 4, target, target)
        transforms = [(k, flip) for k in range(4) for flip in (False, True)] if d4 else [(0, False)]
        for k, flip in transforms:
            transformed = tiles.rot90(k, (-2, -1))
            if flip:
                transformed = transformed.flip(-1)
            predicted = model(transformed).softmax(1)
            if flip:
                predicted = predicted.flip(-1)
            predicted = predicted.rot90(-k, (-2, -1))
            predicted = predicted[..., margin:margin + target, margin:margin + target]
            probabilities += predicted.cpu() / len(transforms)
        for (y, x), prob in zip(locations, probabilities):
            total[:, y:y+target, x:x+target] += prob
            coverage[y:y+target, x:x+target] += 1
    if (coverage == 0).any():
        raise RuntimeError('Uncovered pixels in tiled evaluation')
    return total / coverage[None]


def thin_skeleton(mask):
    """Zhang-Suen thinning, no extra image-processing dependency."""
    image = np.pad(mask.astype(bool), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p2, p3 = image[:-2, 1:-1], image[:-2, 2:]
            p4, p5 = image[1:-1, 2:], image[2:, 2:]
            p6, p7 = image[2:, 1:-1], image[2:, :-2]
            p8, p9 = image[1:-1, :-2], image[:-2, :-2]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            count = sum(p.astype(np.uint8) for p in neighbors)
            transitions = sum((~neighbors[i] & neighbors[(i+1) % 8]).astype(np.uint8) for i in range(8))
            triple_a = (p2 & p4 & p6) if phase == 0 else (p2 & p4 & p8)
            triple_b = (p4 & p6 & p8) if phase == 0 else (p2 & p6 & p8)
            remove = image[1:-1, 1:-1] & (count >= 2) & (count <= 6) & (transitions == 1) & ~triple_a & ~triple_b
            if remove.any():
                image[1:-1, 1:-1][remove] = False
                changed = True
        if not changed:
            return image[1:-1, 1:-1]


def evaluate(model, dataset, config, device, d4=False, export=None, diagnostic=False):
    model.eval()
    start = time.perf_counter()
    matrices, thin_hits, thin_count = [], 0, 0
    if export is not None:
        Path(export).mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(dataset.images):
        probabilities = predict_image(model, image, config['train']['patch'], config['train']['target'],
                                      config['train']['stride'], device,
                                      config['evaluation']['batch_size'], d4)
        pred = probabilities.argmax(0).numpy().astype(np.uint8)
        if dataset.labels:
            target = dataset.labels[i]
            matrices.append(confusion_matrix(target, pred))
            if diagnostic:
                skeleton = thin_skeleton(target == 2)
                thin_hits += int(((pred == 2) & skeleton).sum())
                thin_count += int(skeleton.sum())
        if export is not None:
            Image.fromarray(PALETTE[pred]).save(Path(export) / dataset.paths[i].name.replace('_image', '_mask'))
    elapsed = time.perf_counter() - start
    result = {'tta': 'd4' if d4 else 'single', 'filenames': [p.name for p in dataset.paths],
              'class_names': CLASS_NAMES, 'seconds': elapsed, 'seconds_per_image': elapsed / len(dataset.images)}
    if matrices:
        result.update(summarize(matrices, 'one'))
        summed = np.sum(matrices, 0)
        result.update(confusion_matrix=summed.tolist(), confusion_matrix_rows='target', confusion_matrix_columns='prediction',
                      per_image_confusion_matrices=[cm.tolist() for cm in matrices],
                      per_image_class_iou=[class_iou(cm, 'one').tolist() for cm in matrices],
                      global_class_iou=class_iou(summed, 'one').tolist())
        tp, predicted, actual = int(summed[2, 2]), int(summed[:, 2].sum()), int(summed[2].sum())
        result['eutectic'] = {'precision': tp / predicted if predicted else None, 'recall': tp / actual if actual else None}
        if diagnostic:
            result['eutectic'].update(thin_skeleton_recall=thin_hits / thin_count if thin_count else None,
                                     thin_skeleton_pixels=thin_count, thinning='Zhang-Suen; exact pixel match, no tolerance')
    return result
