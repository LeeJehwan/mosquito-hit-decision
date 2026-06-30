# `train.py`와 `infer.py` 발표용 설명 (Ensemble 반영판)

> 이 문서는 기존 [`train_infer_explanation.md`](train_infer_explanation.md)를 기반으로, 방법론 비교 시뮬레이션에서
> **1위가 된 Ensemble 모델**과 그 모델이 쓰는 **물리 기반(advanced) 특징**을 반영해 업데이트한 버전이다.
> 변경된 부분은 §6(특징), §8(모델), **§8.5(앙상블 방법론과 1위가 된 이유)**, §11~12(학습·추론 흐름), §16(슬라이드)에 집중돼 있다.
> 전체 실험·상세 분석은 [`experiments/RESULT.md`](experiments/RESULT.md) 참고.

## 1. 한 장 요약

이 프로젝트는 모기의 과거 400ms 3D 비행 궤적을 보고, 80ms 뒤 예측 조준점으로 발사했을 때 목표 반경 안에 들어갈지 판단하는 이진 분류 모델입니다.

- 입력: 샘플별 11개 시점의 `timestep_ms, x, y, z`
- 정답: 반경별 명중 여부 `hit_r001` ~ `hit_r005`
- 모델 출력: 명중 확률 `hit_probability`
- 최종 의사결정: 확률이 threshold 이상이면 발사 `fire_decision=1`, 아니면 보류 `fire_decision=0`
- 핵심 평가: 발사해서 맞추면 `+1`, 발사해서 빗나가면 `-2`, 발사하지 않으면 `0`

지원 모델은 세 가지입니다. **LightGBM**(기본), **Logistic Regression**(선형 baseline), 그리고 방법론 비교에서 가장
높은 점수를 낸 **Ensemble**(`--model-type ensemble`, 1위). Ensemble은 자동으로 물리 기반 advanced 특징을 사용합니다.

## 2. 전체 파이프라인

```text
원본 open/train/*.csv + open/train_labels.csv
        |
        | prepare_dataset.py
        v
prepared dataset
  dataset/train/*.csv
  dataset/test/*.csv
  dataset/metadata.json
        |
        | train.py  (--model-type {lightgbm | logistic | ensemble})
        v
학습 산출물
  hit_lgbm.pkl / hit_logistic.pkl / hit_ensemble.pkl
  feature_columns.json      # lightgbm·logistic=114개, ensemble=131개
  decision_threshold.json
  valid_metrics.json
        |
        | infer.py  (학습 때와 같은 --model-type)
        v
추론 산출물
  hit_predictions.csv
  test_metrics.json
```

`train.py`는 `dataset/train/`만 읽습니다. 그 안에서 다시 학습용/검증용으로 나누고, 검증용 데이터로 threshold를 고른 뒤 최종 모델은 `dataset/train/` 전체로 다시 학습합니다.

`infer.py`는 `dataset/test/`만 읽습니다. 저장된 모델, feature 목록, threshold를 불러와 테스트 데이터의 명중 확률과 발사 여부를 계산하고, 테스트 라벨은 마지막 평가에만 사용합니다.

> **모델별 특징 세트가 다르다.** `lightgbm`·`logistic`은 baseline 114개 특징을, `ensemble`은 거기에 물리 기반
> 특징 17개를 더한 131개를 사용합니다(§6). `train.py`/`infer.py`는 `--model-type`에 따라 자동으로 같은 특징을 생성합니다.

## 3. Dataset 구성

prepared dataset은 `prepare_dataset.py`가 원본 데이터를 고정 분할해서 만듭니다.

```text
dataset/train/*.csv     # 8,000개 샘플
dataset/test/*.csv      # 2,000개 샘플
dataset/metadata.json   # schema, split count, id hash
```

각 CSV 파일 하나가 하나의 비행 궤적 샘플입니다. 한 샘플에는 11개 시점이 들어 있고, 시간은 `-400ms`부터 `0ms`까지 40ms 간격입니다.

```csv
timestep_ms,x,y,z,hit_r001,hit_r002,hit_r003,hit_r004,hit_r005
-400,2.490842,0.377812,-0.327984,1,1,1,1,1
...
0,2.99692,0.483173,-0.182941,1,1,1,1,1
```

라벨 컬럼은 한 샘플의 11개 행에 모두 같은 값으로 반복됩니다. 모델은 행 단위가 아니라 샘플 단위로 학습하기 때문입니다.

## 4. Train 데이터는 어떻게 읽는가

`train.py`의 입력 경로는 `--train-dir` 또는 `--dataset-dir/train`, metadata는 `--metadata-path` 또는 `--dataset-dir/metadata.json`로 결정됩니다. 실제 로딩은 `src.dataset.load_prepared_trajectories()`가 담당하며, schema/timestep/개수/id hash를 검증한 뒤 다음을 만듭니다.

```python
trajectories = { "TRAIN_00001": DataFrame[timestep_ms, x, y, z], ... }
labels        = DataFrame[id, hit_r001, ..., hit_r005]
```

## 5. Test 데이터는 어떻게 읽는가

`infer.py`도 같은 방식으로 `dataset/test/`를 읽되 `split="test"`로 검증합니다. 테스트 라벨은 읽기는 하지만 예측 생성에는 쓰지 않고 `test_metrics.json` 계산에만 사용합니다.

## 6. Feature는 어떻게 뽑는가

특징 생성에는 두 단계가 있습니다.

### (A) baseline 특징 114개 — `src.features.build_feature_frame()`

모든 모델이 공통으로 쓰는 궤적 기술 특징입니다.

```text
33 position            # 11 timestep × (x,y,z)
+30 velocity           # 위치 차분 / 시간
+27 acceleration       # 속도 차분 / 시간
+ 3 recent_velocity    # 마지막 구간 속도
+ 3 mean_velocity
+ 3 recent_acceleration
+ 4 speed summary               (min/max/mean/std)
+ 4 acceleration_norm summary
+ 4 turn_angle summary
+ 3 path shape         # path_length, straight_distance, tortuosity
=114 features
```

### (B) advanced 물리 특징 17개 — `src.features_advanced.py` (Ensemble 전용)

이 문제의 본질은 *“고정된 선형 조준 공식이 80ms 뒤를 믿을 만한가”* 를 판단하는 것입니다. 그래서 **그 외삽이 빗나갈 위험을 직접 정량화**한 특징을 추가합니다. `--model-type ensemble`이면 baseline 114 + advanced 17 = **131개**를 사용합니다.

| 그룹 | 아이디어 | 특징 |
|---|---|---|
| **외삽 백테스트(7, 핵심)** | 라벨을 만든 등속 외삽을 **과거 구간에서 똑같이 재현**해 실제 오차를 측정. 예) `-120→-80ms` 속도로 `0ms`를 예측 → 실제 `0ms`와의 거리. 미래 외삽 오차의 직접 대리지표(반경과 같은 단위) | `bt_err_last`, `bt_err_last3_mean`, `bt_err_mean/max/min/std`, `bt_err_trend` |
| **등속 vs 등가속 간극(2)** | 가속도가 크면 직선 조준이 휜다. `0.5·a·0.08²` | `aim_curvature_gap`, `recent_accel_norm` |
| **저크 3차 미분(3)** | 가속도 자체가 변하면 보정도 못 믿음 | `recent_jerk_norm`, `jerk_norm_mean/max` |
| **속도 일관성(4)** | 단구간 속도와 장구간 속도가 어긋나면 등속 가정 붕괴 | `recent_vel_mismatch`, `speed_recent`, `speed_recent_ratio`, `speed_recent_change` |
| **최근 회전(1)** | 마지막 순간 방향이 꺾이면 외삽 위험 | `recent_turn_angle` |

### 전체 궤적 feature를 쓰는 이유

실제 발사 좌표는 `-40ms`와 `0ms` 두 점만으로 `aim = p(0) + recent_velocity·0.08`로 **고정** 계산됩니다. 즉 모델은 “어디를 조준할지”를 새로 정하지 않습니다. 모델이 하는 일은 “그 고정 발사 좌표를 믿고 쏴도 되는 상황인지”를 판단하는 것이고, baseline 특징은 그 신뢰도의 간접 근거, advanced 특징은 **직접 근거**(외삽이 실제로 얼마나 빗나갔나)입니다.

## 7. Feature 생성 샘플 (advanced 포함)

`dataset/train/TRAIN_00001.csv`에서 일부를 뽑으면 다음과 같습니다.

```csv
feature,value
position_00_x,2.490842
velocity_00_x,1.261275
recent_velocity_x,1.309075
speed_mean,1.346022
tortuosity,1.002824
bt_err_last,0.0068          # ← advanced: 가장 최근 외삽 오차 (반경 스케일)
bt_err_mean,0.0080          # ← advanced
aim_curvature_gap,0.0059    # ← advanced: 등속 vs 등가속 조준 간극
recent_accel_norm,1.8374    # ← advanced
```

## 8. 모델은 무엇을 보고 학습하는가

모델 입력 `X`는 특징(lightgbm·logistic 114개 / ensemble 131개)이고, 정답 `y`는 사용자가 지정한 반경의 라벨 하나입니다(`--radius 0.05`이면 `hit_r005`). 모델은 `predict_proba()`로 hit 확률을 냅니다.

```text
hit_probability = P(hit = 1 | trajectory features)
```

지원 모델은 세 가지입니다.

| `--model-type` | 모델 | 특징 세트 | 설명 |
|---|---|---|---|
| `lightgbm` | `LGBMClassifier` | baseline 114 | 기본값. leaf-wise 트리, 비선형·상호작용 |
| `logistic` | `StandardScaler + LogisticRegression` | baseline 114 | 선형 baseline. 스케일 차이를 표준화로 처리 |
| `ensemble` | **`SoftVotingEnsemble`** = LGBM + HistGradientBoosting + MLP | **baseline 114 + advanced 17 = 131** | **시뮬레이션 1위.** 세 모델의 확률 평균(soft voting) |

### 구현 위치

```text
src/model.py          SoftVotingEnsemble, create_ensemble()
src/config.py         EnsembleConfig, MODEL_TYPES = (lightgbm, logistic, ensemble)
src/features_advanced.py  advanced 17개 특징
산출물                hit_ensemble.pkl
```

`SoftVotingEnsemble`은 세 멤버 모델을 각각 학습한 뒤, 추론 시 `predict_proba`의 양성 확률을 **평균**합니다(hard voting이 아니라 확률 평균이라 신뢰도를 보존). MLP 멤버만 `StandardScaler`를 포함합니다(트리 모델은 스케일 불필요).

## 8.5 앙상블 방법론과 1위가 된 이유 (핵심 추가 분석)

5개 반경(0.01~0.05) 전체 평균순위에서 Ensemble이 **1.6위로 1위**, baseline LightGBM은 4.6위입니다. baseline LightGBM → Ensemble 개선은 **두 가지 독립 원인**으로 분해됩니다.

| 반경 | ① LGBM baseline | LGBM **+adv** | Ensemble **+adv** | 특징 효과 | 앙상블 효과 | 총 개선 |
|---|---|---|---|---|---|---|
| 0.05 | 0.881 | 0.876 | 0.885 | −0.005 | +0.009 | +0.004 |
| 0.03 | 0.773 | 0.776 | 0.785 | +0.003 | +0.009 | +0.012 |
| 0.01 | 0.211 | 0.232 | 0.245 | **+0.021** | +0.013 | **+0.034** |

1. **물리 특징(adv) 효과** — 외삽 위험을 직접 잰 특징은 **반경이 작을수록**(어려울수록) 크게 기여합니다(r=0.01 +0.021).
2. **앙상블 효과 — 오류 비상관 → 분산 감소** — LGBM(leaf-wise 트리)·HistGBM(level-wise 트리)·MLP(매끄러운 신경망)는 **서로 다르게 틀리므로**, 확률을 평균하면 오류가 상쇄되어 **AUROC(랭킹 품질)가 향상**됩니다.

| AUROC | r=0.05 | r=0.03 | r=0.02 | r=0.01 |
|---|---|---|---|---|
| LGBM baseline | 0.851 | 0.833 | 0.808 | 0.810 |
| Ensemble + adv | 0.866 | 0.859 | 0.829 | 0.833 |

3. **비용지표와의 연결** — 점수는 `발사 ⇔ P ≥ threshold` 결정에서 나오고, 빗나간 발사는 −2입니다. AUROC가 높다 = hit/miss를 경계에서 잘 분리한다 = **+1을 더 담으면서 −2를 더 거른다.** −2가 치명적인 작은 반경(r=0.01은 fire-all이 음수 −0.263)일수록 이 효과가 점수로 직결됩니다.
4. **확률 평활화 → threshold 견고성** — r=0.05에서 단일 `LGBM+adv`는 AUROC가 더 높은데도(0.875 vs 0.866) 점수는 낮습니다(0.876 vs 0.885). OOF threshold가 0.96으로 과적합해 너무 신중해진 탓(recall 0.967). 앙상블은 확률이 매끄러워 threshold(0.815)가 일반화되고 recall 0.983으로 **+1 기회를 더 담아** 역전합니다.

> 한 줄 요약: *“앙상블의 승리 = (1) 외삽 위험을 잰 물리 특징 + (2) 다르게 틀리는 모델들의 확률 평균이 만든 랭킹 향상·threshold 견고성. 문제가 어려울수록 효과가 커진다.”*

## 9. 맞췄다/못 맞췄다는 어떻게 정해지는가

prepared dataset의 라벨은 `prepare_dataset.py`에서 만들어집니다.

```text
recent_velocity = (position_0ms - position_-40ms) / 0.04
aim_position    = position_0ms + recent_velocity * 0.08
error           = distance(aim_position, actual_future_position)
hit             = 1 if error <= radius else 0
```

## 10. Threshold는 왜 필요한가

모델은 0/1이 아니라 확률을 냅니다. 발사 결정으로 바꾸려면 threshold가 필요합니다.

```text
fire_decision = 1 if hit_probability >= threshold else 0
```

`--threshold`를 직접 주지 않으면 검증 데이터에서 `hit_score`가 가장 큰 threshold를 탐색합니다(동점이면 더 큰 값). **이 메커니즘은 세 모델 모두 동일**합니다.

> 참고: 비용 구조상 기대보상 `3·P − 2 > 0 ⇔ P > 2/3`이므로, 확률이 잘 보정되면 이론 최적 threshold는 0.667 근처입니다. (실험에서 보정 모델이 2위였던 근거.)

## 11. Train 단계별 산출 데이터

1. prepared train 로딩 → `trajectories`, `labels`, `metadata`
2. **feature frame 생성** — `--model-type ensemble`이면 `build_feature_frame_advanced()`(131개), 그 외엔 `build_feature_frame()`(114개)
3. feature와 label merge
4. train/validation 층화 분할 (기본 6,400 / 1,600)
5. validation model 학습 → 검증 데이터로 threshold 선택
6. **final model을 train 전체(8,000)로 재학습** — ensemble이면 세 멤버 모두 재학습
7. 저장: `hit_*.pkl`, `feature_columns.json`, `decision_threshold.json`(`model_type` 기록), `valid_metrics.json`

## 12. Infer 단계별 산출 데이터

1. prepared test 로딩
2. **학습 때와 같은 특징 생성** — ensemble이면 advanced 131개. 이후 `feature_columns.json` 기준으로 컬럼 정렬
3. artifact 검증 — 선택한 `--model-type`과 threshold artifact의 `model_type` 일치, threshold 범위, 학습/추론 반경·source hash 일치
4. 확률 예측(`predict_proba`) → threshold로 발사 결정
5. 조준점·확률·발사 여부·실제 라벨 저장(`hit_predictions.csv`), 지표 저장(`test_metrics.json`)

## 13. 평가 지표는 어떻게 계산하는가

`src.metrics.evaluate_predictions(y_true, probabilities, threshold)`가 확률을 발사 결정으로 바꾼 뒤 일반 classification metric과 대회식 hit score를 함께 계산합니다.

## 14. 각 평가 지표의 의미

| 지표 | 의미 |
|---|---|
| `accuracy/precision/recall/f1` | 분류 기본 지표 (precision=쏜 것 중 명중률) |
| `auroc` | threshold와 무관한 확률 랭킹 품질 = 명중/빗나감 판별력 |
| `hit_score` / `mean_hit_score` | 발사 보상 총합/평균 — 실제 의사결정 목표 |
| `shots_fired` | 발사 결정 수 |

### hit score 보상 규칙

```text
fire_decision = 0                 ->  0점
fire_decision = 1 and actual = 1  -> +1점
fire_decision = 1 and actual = 0  -> -2점
```

빗나간 발사가 `-2`로 크게 벌점이므로, “맞을 가능성이 충분히 높을 때만 쏘는” 모델·threshold가 좋습니다.

## 15. 평가 결과 예시 (radius=0.05, test 2,000)

| 모델 | accuracy | precision | recall | auroc | mean_hit_score |
|---|---|---|---|---|---|
| LightGBM (baseline) | 0.951 | 0.967 | 0.982 | 0.851 | 0.881 |
| **Ensemble + adv (1위)** | ~0.95 | 0.971 | 0.983 | 0.866 | **0.885** |

> 실행 방법에 따른 차이: `experiments/run_methodology_comparison.py`는 5-fold OOF로 threshold(0.815)를 골라 0.885를 재현합니다.
> `train.py`는 기존 설계대로 단일 검증 split에서 threshold(예: 0.89)를 골라 ≈0.872가 나옵니다 — **모델 자체는 동일**합니다.

## 16. 발표 슬라이드 구성안

1. **문제 정의** — 고정 조준 공식의 신뢰도 판단(좌표 예측이 아님), 빗나간 발사 −2
2. **데이터 구조** — 11 timestep 3D 궤적, train 8,000 / test 2,000, 반경별 라벨
3. **라벨 생성 원리** — 등속 외삽 조준점 vs 실제 미래 위치 거리 ≤ radius
4. **Feature Engineering** — baseline 114개(위치/속도/가속도/회전/경로형태)
5. **물리 기반 advanced 특징(신규)** — 외삽 백테스트로 “외삽이 빗나갈 위험”을 직접 정량화(17개)
6. **모델 3종** — LightGBM / Logistic / **Ensemble(LGBM+HistGBM+MLP 소프트보팅, 1위)**
7. **학습·추론 흐름** — `--model-type`에 따라 특징·모델·threshold 자동 일치
8. **평가 지표** — classification + decision(hit_score), AUROC의 역할
9. **결과: 앙상블이 1위인 이유** — ① 물리 특징 효과 ② 오류 비상관(AUROC↑) ③ threshold 견고성, 반경↓일수록 격차↑
10. **핵심 메시지** — 좌표 예측 → 발사 의사결정 재정의, 물리 특징 + 앙상블 + threshold 분리가 점수를 끌어올림

## 17. 우승 모델 사용법

```bash
# 학습
uv run python train.py --dataset-dir dataset --radius 0.05 \
  --model-type ensemble --output runs/ensemble_01

# 추론 (학습 때와 같은 --model-type, --output)
uv run python infer.py --dataset-dir dataset --radius 0.05 \
  --model-type ensemble --output runs/ensemble_01
```

`--model-type ensemble`이면 advanced 특징 생성과 소프트보팅 앙상블이 자동 적용되고, 산출물은 `hit_ensemble.pkl`로 저장됩니다.
방법론 비교 실험과 상세 분석은 [`experiments/RESULT.md`](experiments/RESULT.md)를 참고하세요.
