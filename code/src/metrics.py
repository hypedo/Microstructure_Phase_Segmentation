"""KoMaP: equal-weight image means of four phase IoUs, never pooled IoU."""
import numpy as np

CLASS_NAMES = ['Al', 'Primary_Si', 'Al3Ni', 'Eutectic_Si']
PALETTE = np.array([[153,127,76], [76,178,76], [25,76,153], [204,204,204]], dtype=np.uint8)

def encode_mask(rgb):
    rgb = np.asarray(rgb)[..., :3]
    out = np.full(rgb.shape[:2], 255, np.uint8)
    for i, color in enumerate(PALETTE):
        out[np.all(rgb == color, axis=-1)] = i
    if np.any(out == 255):
        unknown = np.unique(rgb[out == 255], axis=0)
        raise ValueError(f'Unknown mask colors (no silent nearest mapping): {unknown[:20]}')
    return out

def confusion(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError(f'Shape mismatch {target.shape} != {prediction.shape}')
    if not np.isin(target, np.arange(4)).all() or not np.isin(prediction, np.arange(4)).all():
        raise ValueError('Labels must be in 0..3')
    return np.bincount((4*target.astype(np.int64)+prediction).ravel(), minlength=16).reshape(4,4)

def score_confusion(cm):
    intersection = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - intersection
    present = union > 0
    iou = np.divide(intersection, union, out=np.zeros(4, dtype=float), where=present)
    # Official website does not specify union==0. Report all conventions.
    return {
        'class_iou': [float(iou[i]) if present[i] else None for i in range(4)],
        'miou_absent_zero': float(iou.mean()),
        'miou_absent_one': float(np.where(present, iou, 1).mean()),
        'miou_present_only': float(iou[present].mean()) if present.any() else None,
        'target_counts': cm.sum(1).astype(int).tolist(),
        'confusion': cm.astype(int).tolist(),
    }

def summarize(rows):
    if not rows:
        raise ValueError('No images evaluated')
    result = {'n_images': len(rows), 'per_image': rows, 'class_names': CLASS_NAMES}
    for key in ('miou_absent_zero', 'miou_absent_one', 'miou_present_only'):
        result[key] = float(np.mean([r[key] for r in rows]))
    result['mean_class_iou'] = np.nanmean(np.array([[np.nan if v is None else v for v in r['class_iou']] for r in rows]),0).tolist()
    result['mean_class_iou_definition'] = 'Legacy conditional class means: exclude images with union=0 for that class'
    result['mean_class_iou_absent_zero'] = np.mean([[0. if v is None else v for v in r['class_iou']] for r in rows],0).tolist()
    result['mean_class_iou_absent_one'] = np.mean([[1. if v is None else v for v in r['class_iou']] for r in rows],0).tolist()
    assert abs(np.mean(result['mean_class_iou_absent_zero'])-result['miou_absent_zero'])<1e-12
    result['pooled_diagnostic_only'] = score_confusion(np.sum([r['confusion'] for r in rows], axis=0))
    return result

def self_test():
    truth = np.array([[0,1],[2,3]], dtype=np.uint8)
    assert score_confusion(confusion(truth,truth))['miou_absent_zero'] == 1
    assert score_confusion(confusion(truth,(truth+1)%4))['miou_absent_zero'] == 0
    assert np.array_equal(encode_mask(PALETTE[truth]),truth)
    empty = score_confusion(confusion(np.zeros((2,2),np.uint8),np.zeros((2,2),np.uint8)))
    assert empty['miou_absent_zero'] == .25 and empty['miou_absent_one'] == 1
    cm1 = np.diag([1,1,1,1]); cm2 = np.array([[100,0,0,0],[0,0,0,100],[0,0,100,0],[0,100,0,0]])
    result = summarize([score_confusion(cm1),score_confusion(cm2)])
    assert abs(result['miou_absent_zero']-.75) < 1e-12
    assert abs(result['pooled_diagnostic_only']['miou_absent_zero']-.75) > .1
    print('Metric tests passed: palette, perfect/wrong labels, absent classes, image vs pooled averaging.')

if __name__ == '__main__':
    self_test()
