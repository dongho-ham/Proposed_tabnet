# Proposed TabNet

Self-supervised TabNet 기반 사출성형 품질 분류 모델 (Class Imbalance 문제 해결)

## 프로젝트 개요

**기간:** 2023.07 ~ 2023.12 (6개월)  
**역할:** 학술논문 작성 
**기술:** PyTorch, Pandas

## 문제 정의

**데이터셋:**
- 사출성형 품질 데이터 (34 features, 11,280 rows)
- Class imbalance: 정상 10,253개 / 불량 1,027개

**기존 모델 한계:**
- Attentive Transformer의 Sequential Attention 증가만으로는 성능 개선 제한
- 낮은 Inductive Bias로 인한 소량 데이터/국소적 패턴 학습 시 일반화 성능 저하
- 특정 Feature 과의존 현상

## 해결 방법

### 아키텍처 개선: 1D CNN-CBAM Module
```
<img width="1383" height="524" alt="image" src="https://github.com/user-attachments/assets/5b061e34-7539-4f6a-96ff-b26170ec55f6" />

```

**핵심 구성:**
1. **1D CNN**: 지역적 특징(Local feature) 강제 추출 → Inductive Bias 부여
2. **CBAM (Convolutional Block Attention Module)**:
   - Channel Attention: 중요 변수 필터링
   - Spatial Attention: 중요 시점 필터링
3. **3×3 소형 필터**: 연산 효율성과 성능 간 Trade-off 최적화

## 실험 결과

### 성능 비교 (Recall 중심)

<img width="1129" height="353" alt="image" src="https://github.com/user-attachments/assets/51b7e949-7e00-4d57-a644-c95cf8a701d6" />

<img width="1244" height="374" alt="스크린샷 2026-01-06 122215" src="https://github.com/user-attachments/assets/83833efb-fb7c-40a6-92b0-58491a61669b" />


### 핵심 개선사항

**불량 탐지 성능 (SSL TabNet 대비):**
- Recall: **+0.18** (0.78 → 0.96)
- F1-score: **+0.11** (0.87 → 0.98)

**Feature Importance 개선:**
- 특정 Feature 과의존 현상 해소
- 전역적·지역적 패턴 균형 학습

## 기술적 기여

1. **강건한 표현 학습**: Self-supervised Learning + 1D CNN 계층적 구조
2. **Inductive Bias 강화**: 국소 패턴 추출 능력 향상
3. **일반화 성능 향상**: 미학습 데이터 Recall 0.18 개선

## 실무 임팩트

- **미검출 리스크 최소화**: Recall +18%p로 사후 손실 비용 감소
- **스마트 팩토리 ROI 극대화**: 실시간 품질 관리 신뢰도 향상
- **확장 가능성**: Tabular 데이터 기반 제조 품질 예측 전반 적용 가능

## 실행 방법
```bash
# 의존성 설치
pip install torch pandas

# 학습
python train.py --model proposed_tabnet --data injection_molding.csv

# 평가
python evaluate.py --checkpoint best_model.pth
```

## 참고

- 논문 제출 예정 (2026년 상반기)
- Original TabNet Paper: [Arik & Pfister, 2020]

