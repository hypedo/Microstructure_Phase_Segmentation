# 설계 근거: 학습 목적과 평가 집계 단위의 정렬.
# Lovasz-Softmax (https://arxiv.org/abs/1705.08790)는 IoU 대리 목적 및 이미지/데이터셋
# 집계 차이의 관련 연구이다. 아래 손실은 Lovasz를 복제한 구현이 아니라 GT-present
# soft-IoU와 CE를 결합한 본 실험의 대리식이다. argmax 공식 점수와 수치적으로 동일하지 않다.
# 0.8/0.2는 실험 선택이며 공식 규정이 아니다. 세 추가 학습 실험은 시작 step 0이 선택되어
# 최종 제출에 새 미세조정 가중치가 채택되지 않았다는 점도 함께 해석해야 한다.

"""Operator clarification supplied by user, 2026-10-09.

Confusion rows are truth, columns prediction. Average only GT-present classes
WITHIN each image, then average images equally. Never mask out predicted pixels
belonging to an absent class: these remain false negatives of true classes.
Historical src/metrics.py and the Oct04 release are deliberately immutable.
"""
import numpy as np


def per_image_score(cm):
    cm = np.asarray(cm)
    if cm.shape[-2:] != (4, 4) or np.any(cm < 0):
        raise ValueError('Expected nonnegative (..., 4, 4) confusion matrices')
    truth = cm.sum(-1)
    present = truth > 0
    if np.any(present.sum(-1) == 0):
        raise ValueError('Empty ground truth image')
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    union = truth + cm.sum(-2) - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp, dtype=np.float64), where=union > 0)
    return (iou * present).sum(-1) / present.sum(-1)


def soft_present_iou_loss(logits, target):
    """Differentiable image-macro surrogate; logits B,C,H,W, target B,H,W.

    Presence derives from this evaluated image, never from predictions. Full
    image training aligns presence and weighting with the official objective;
    patch training is only a patch-level surrogate and is labelled as such.
    """
    import torch
    p = logits.float().softmax(1)
    y = torch.nn.functional.one_hot(target.long(), 4).movedim(-1, 1).float()
    intersection = (p * y).sum((-2, -1))
    truth = y.sum((-2, -1))
    present = truth > 0
    union = truth + p.sum((-2, -1)) - intersection
    # GT-present classes have union >= 1; clamp guards excluded empty classes.
    iou = intersection / union.clamp_min(1e-7)
    return (1 - (iou * present).sum(1) / present.sum(1)).mean()


def aligned_loss(logits, target, iou_weight=0.8):
    """Explicit candidate: 0.8 image-macro present soft-IoU + 0.2 image CE.

    Coefficients are declared experimental choices, not organizer requirements.
    CE preserves pixel supervision for every class, including absent predictions.
    """
    import torch.nn.functional as F
    ce = F.cross_entropy(logits.float(), target.long(), reduction='none').flatten(1).mean(1).mean()
    return iou_weight * soft_present_iou_loss(logits, target) + (1-iou_weight) * ce


def self_test():
    # GT has only Al; an erroneous Primary prediction must reduce Al IoU.
    cm = np.zeros((4, 4), np.int64); cm[0, 0] = 3; cm[0, 1] = 1
    assert per_image_score(cm) == .75  # union-present would wrongly yield .375
    a = np.diag([1, 0, 0, 0]); b = np.zeros((4, 4), int); b[0, 1] = 100
    assert per_image_score(np.stack([a, b])).mean() == .5
    assert per_image_score(a + b) != .5
    assert per_image_score(np.diag([1, 2, 3, 4])) == 1
    import torch
    target = torch.tensor([[[0, 0], [0, 0]], [[0, 1], [2, 3]]])
    pred = torch.tensor([[[0, 0], [0, 1]], [[0, 1], [2, 3]]])
    logits = (torch.nn.functional.one_hot(pred, 4).movedim(-1, 1).float()*80-40).requires_grad_()
    loss = soft_present_iou_loss(logits, target)
    assert abs(float(loss.detach())-.125)<1e-7
    aligned_loss(logits, target).backward()
    assert torch.isfinite(logits.grad).all()
    print('Official GT-present score, FP-as-FN, image-macro, hard-limit and gradient tests passed.')


if __name__ == '__main__':
    self_test()
