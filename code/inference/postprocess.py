# Phasera 조성 조건부 보정 — 결과서 3-2의 실제 구현.
# 예측 상 조성을 이용한 영상별 log-offset으로 다중 상 경쟁의 결정 경계를 조절한다.
# Menon et al. (ICLR 2021)의 logit adjustment와 수식 형태는 관련되지만, 그 논문의
# 전역 class-frequency 보정이나 이론적 보장을 재현한 것은 아니다.
# 출처: https://arxiv.org/abs/2007.07314
# m_c는 화학 조성/정답 면적분율의 실측치가 아니라 예측 확률의 공간 평균이다.
# Ni와 공정 Si의 상대 예측 질량, 초정 Si의 규모를 낮은 차원 특징으로 사용한다.
# 새로운 영상에도 같은 규칙을 적용할 수 있으나, 외부 시편 일반화는 별도로 검증해야 한다.

"""Inference-only phase-composition calibration. Never reads ground truth."""
import numpy as np


def validate_calibration(spec):
    if set(spec)!={'kind','parameters'} or spec['kind']!='phase_composition':
        raise ValueError('Unsupported conditional calibration schema')
    b=np.asarray(spec['parameters'],np.float64)
    if b.shape!=(9,) or not np.isfinite(b).all() or np.any(np.abs(b.reshape(3,3))>np.array([.5,.7,.7])+1e-12):
        raise ValueError('Invalid bounded calibration parameters')
    return b.reshape(3,3)


def conditional_offsets(prob,spec):
    beta=validate_calibration(spec)
    # 클래스 순서는 [Al, Primary Si, Al3Ni, Eutectic Si]이다.
    # r은 Ni 대 Eutectic Si의 균형, h는 Primary Si의 작은 질량 변화도 읽는 로그 특징이다.
    # 0.05, 분모 4, clip 범위는 이 제출에 고정된 설정이며 새로운 입력에서 재추정하지 않는다.
    mass=prob.mean((1,2),dtype=np.float64)
    ratio=2*mass[2]/max(mass[2]+mass[3],1e-8)-1
    primary=np.clip((np.log(max(mass[1],1e-6))-np.log(.05))/4,-1,1)
    # Al offset을 0으로 고정해 모든 class에 동일 상수를 더하는 비식별성을 없앤다.
    # 나머지 세 상에 각각 [상수, r, h]의 세 계수만 사용하므로 총 9개 계수다.
    return np.r_[0.,beta@np.array([1.,ratio,primary])]


def apply_postprocess(prob,settings):
    if prob.ndim!=3 or prob.shape[0]!=4 or not np.isfinite(prob).all() or np.any(prob<0):
        raise ValueError('Expected finite nonnegative 4xHxW probabilities')
    bias=settings.get('log_bias')
    if 'conditional_calibration' in settings:
        if bias is not None:raise ValueError('Ambiguous calibration: fixed and conditional biases')
        bias=conditional_offsets(prob,settings['conditional_calibration'])
    # 한 상의 offset을 올리면 다른 상과의 상대 경쟁 경계가 바뀐다.
    # 예측에만 나타난 상을 GT 기반으로 지우지 않는다. GT 존재 여부는 평가기에만 있다.
    scores=prob if bias is None else np.log(np.maximum(prob,1e-30).astype(np.float64))+np.asarray(bias,np.float64)[:,None,None]
    pred=scores.argmax(0).astype(np.uint8)
    threshold=settings.get('primary_component_max_pixels',0)
    gate=settings.get('primary_presence_min_fraction',0.)
    if threshold or gate:
        from scipy import ndimage
        other=scores.copy();other[1]=-np.inf;replacement=other.argmax(0).astype(np.uint8)
        if threshold:
            # 초정 Si는 큰 각형 입자로 관찰되므로 작은 고립 성분을 별도로 검토했다.
            # 8-연결 면적 128 px 이하의 성분을 선택적으로 정제한다. 영상 테두리 성분은 실제
            # 큰 입자의 잘린 조각일 수 있어 보호한다. Eutectic Si는 본래 가늘어 이 규칙을 적용하지 않는다.
            # 128 px는 이 데이터에서 선택한 경험값이며 물리적 입자 크기나 보편적인 임계값이 아니다.
            # 제거 픽셀은 Al로 일괄 치환하지 않고 나머지 세 상 중 보정 점수가 가장 높은 상으로 재할당한다.
            labels,n=ndimage.label(pred==1,structure=np.ones((3,3),bool));sizes=np.bincount(labels.ravel(),minlength=n+1)
            protect=np.zeros(n+1,bool);protect[0]=True;protect[np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])]=True
            remove=((sizes<=threshold)&~protect)[labels];pred[remove]=replacement[remove]
        if (pred==1).mean()<gate:
            remove=pred==1;pred[remove]=replacement[remove]
    return pred
