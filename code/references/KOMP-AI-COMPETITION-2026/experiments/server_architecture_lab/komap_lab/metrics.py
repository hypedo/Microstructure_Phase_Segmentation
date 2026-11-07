import numpy as np


def confusion_matrix(target, pred, num_classes=4):
    target, pred = np.asarray(target), np.asarray(pred)
    if target.shape != pred.shape:
        raise ValueError('Prediction and target shapes differ')
    if target.min() < 0 or target.max() >= num_classes or pred.min() < 0 or pred.max() >= num_classes:
        raise ValueError('Invalid class ID')
    values = target.astype(np.int64).ravel() * num_classes + pred.astype(np.int64).ravel()
    return np.bincount(values, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def class_iou(cm, absent='one'):
    intersection = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - intersection
    fill = {'one': 1.0, 'zero': 0.0, 'ignore': float('nan')}[absent]
    result = np.full(len(cm), fill, dtype=np.float64)
    np.divide(intersection, union, out=result, where=union > 0)
    return result


def summarize(matrices, absent='one'):
    per_image_class = np.array([class_iou(cm, absent) for cm in matrices])
    per_image = np.nanmean(per_image_class, axis=1)
    counts = np.isfinite(per_image_class).sum(axis=0)
    class_means = np.divide(np.nansum(per_image_class, axis=0), counts,
                            out=np.zeros(per_image_class.shape[1]), where=counts > 0)
    return {'miou': float(per_image.mean()),
            'class_iou': [float(v) if n else None for v, n in zip(class_means, counts)],
            'per_image_miou': per_image.tolist(),
            'absent_policy': absent,
            'miou_absent_one': float(np.mean([class_iou(cm, 'one').mean() for cm in matrices])),
            'miou_absent_zero': float(np.mean([class_iou(cm, 'zero').mean() for cm in matrices])),
            'miou_absent_ignore': float(np.mean([np.nanmean(class_iou(cm, 'ignore')) for cm in matrices]))}
