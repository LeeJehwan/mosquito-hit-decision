# `train.py`와 `infer.py` 발표용 설명

## 1. 한 장 요약

이 프로젝트는 모기의 과거 400ms 3D 비행 궤적을 보고, 80ms 뒤 예측 조준점으로 발사했을 때 목표 반경 안에 들어갈지 판단하는 이진 분류 모델입니다.

- 입력: 샘플별 11개 시점의 `timestep_ms, x, y, z`
- 정답: 반경별 명중 여부 `hit_r001` ~ `hit_r005`
- 모델 출력: 명중 확률 `hit_probability`
- 최종 의사결정: 확률이 threshold 이상이면 발사 `fire_decision=1`, 아니면 보류 `fire_decision=0`
- 핵심 평가: 발사해서 맞추면 `+1`, 발사해서 빗나가면 `-2`, 발사하지 않으면 `0`

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
        | train.py
        v
학습 산출물
  hit_lgbm.pkl 또는 hit_logistic.pkl
  feature_columns.json
  decision_threshold.json
  valid_metrics.json
        |
        | infer.py
        v
추론 산출물
  hit_predictions.csv
  test_metrics.json
```

`train.py`는 `dataset/train/`만 읽습니다. 그 안에서 다시 학습용/검증용으로 나누고, 검증용 데이터로 threshold를 고른 뒤 최종 모델은 `dataset/train/` 전체로 다시 학습합니다.

`infer.py`는 `dataset/test/`만 읽습니다. 저장된 모델, feature 목록, threshold를 불러와 테스트 데이터의 명중 확률과 발사 여부를 계산하고, 테스트 라벨은 마지막 평가에만 사용합니다.

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
-360,2.541293,0.388259,-0.31267,1,1,1,1,1
-320,2.592528,0.397126,-0.298637,1,1,1,1,1
...
-40,2.944557,0.47663,-0.193358,1,1,1,1,1
0,2.99692,0.483173,-0.182941,1,1,1,1,1
```

### 컬럼 의미

| 컬럼 | 의미 |
|---|---|
| `timestep_ms` | 현재 시점 `0ms` 기준 과거 시간 |
| `x`, `y`, `z` | 해당 시점의 3차원 위치 |
| `hit_r001` | 반경 `0.01` 안에 들어가면 1 |
| `hit_r002` | 반경 `0.02` 안에 들어가면 1 |
| `hit_r003` | 반경 `0.03` 안에 들어가면 1 |
| `hit_r004` | 반경 `0.04` 안에 들어가면 1 |
| `hit_r005` | 반경 `0.05` 안에 들어가면 1 |

라벨 컬럼은 한 샘플의 11개 행에 모두 같은 값으로 반복됩니다. 모델은 행 단위가 아니라 샘플 단위로 학습하기 때문입니다.

## 4. Train 데이터는 어떻게 읽는가

`train.py`의 입력 경로는 다음 순서로 결정됩니다.

- `--train-dir`가 있으면 그 경로 사용
- 없으면 `--dataset-dir/train` 사용
- metadata는 `--metadata-path`가 있으면 그 경로 사용
- 없으면 `--dataset-dir/metadata.json` 사용

실제 로딩은 `src.dataset.load_prepared_trajectories()`가 담당합니다.

1. `metadata.json`을 읽고 schema version과 컬럼 구성을 검증합니다.
2. `dataset/train/*.csv` 파일 개수가 metadata의 `counts.train`과 같은지 확인합니다.
3. 각 CSV를 읽고 필수 컬럼과 11개 timestep 여부를 확인합니다.
4. 궤적 데이터는 `timestep_ms, x, y, z`만 분리합니다.
5. 라벨은 각 파일 첫 행에서 `hit_r001` ~ `hit_r005`를 가져옵니다.
6. 전체 id hash가 metadata와 일치하는지 검증합니다.

로딩 후 메모리에는 다음 두 데이터가 만들어집니다.

```python
trajectories = {
    "TRAIN_00001": DataFrame[timestep_ms, x, y, z],
    "TRAIN_00002": DataFrame[timestep_ms, x, y, z],
    ...
}
```

```text
labels
id           hit_r001  hit_r002  hit_r003  hit_r004  hit_r005
TRAIN_00001        1         1         1         1         1
TRAIN_00002        0         0         0         0         1
...
```

## 5. Test 데이터는 어떻게 읽는가

`infer.py`의 테스트 입력 경로도 같은 방식입니다.

- `--test-dir`가 있으면 그 경로 사용
- 없으면 `--dataset-dir/test` 사용
- metadata는 `--metadata-path`가 있으면 그 경로 사용
- 없으면 `--dataset-dir/metadata.json` 사용

`infer.py`도 `load_prepared_trajectories()`를 호출하지만 `split="test"`로 읽습니다. 따라서 파일 개수는 metadata의 `counts.test`, id hash는 `split_id_sha256.test`와 비교됩니다.

중요한 점은 테스트 라벨을 읽기는 하지만, 모델 예측을 만드는 데에는 쓰지 않는다는 것입니다. 라벨은 `test_metrics.json`을 만들 때만 사용됩니다.

## 6. Feature는 어떻게 뽑는가

feature 생성은 `src.features.build_feature_frame()`에서 수행됩니다. 각 샘플의 11개 위치 데이터를 하나의 feature row로 바꿉니다.

### 원본 위치 feature

11개 timestep의 `x, y, z`를 그대로 펼칩니다.

```text
position_00_x, position_00_y, position_00_z
position_01_x, position_01_y, position_01_z
...
position_10_x, position_10_y, position_10_z
```

위치 feature 수는 `11 * 3 = 33`개입니다.

### 속도 feature

연속된 두 위치 차이를 시간 차이로 나눕니다.

```text
velocity = (position[t+1] - position[t]) / elapsed_seconds
```

11개 위치에서 구간은 10개이므로 속도 feature는 `10 * 3 = 30`개입니다.

```text
velocity_00_x, velocity_00_y, velocity_00_z
...
velocity_09_x, velocity_09_y, velocity_09_z
```

### 가속도 feature

연속된 두 속도 차이를 평균 시간 간격으로 나눕니다.

```text
acceleration = (velocity[t+1] - velocity[t]) / average_elapsed_seconds
```

속도 10개에서 가속도 구간은 9개이므로 가속도 feature는 `9 * 3 = 27`개입니다.

```text
acceleration_00_x, acceleration_00_y, acceleration_00_z
...
acceleration_08_x, acceleration_08_y, acceleration_08_z
```

### 최근 속도와 평균 속도

마지막 구간의 속도와 전체 평균 속도를 추가합니다.

```text
recent_velocity_x, recent_velocity_y, recent_velocity_z
mean_velocity_x, mean_velocity_y, mean_velocity_z
```

### 최근 가속도

마지막 가속도 벡터를 추가합니다.

```text
recent_acceleration_x, recent_acceleration_y, recent_acceleration_z
```

### 요약 통계 feature

속력, 가속도 크기, 회전각에 대해 `min`, `max`, `mean`, `std`를 계산합니다.

```text
speed_min, speed_max, speed_mean, speed_std
acceleration_norm_min, acceleration_norm_max, acceleration_norm_mean, acceleration_norm_std
turn_angle_min, turn_angle_max, turn_angle_mean, turn_angle_std
```

### 궤적 형태 feature

```text
path_length        # 모든 구간 이동 거리의 합
straight_distance  # 시작점과 끝점 사이 직선 거리
tortuosity         # path_length / straight_distance
```

`tortuosity`는 경로가 얼마나 구불구불한지 나타냅니다. 1에 가까우면 거의 직선, 클수록 우회가 많습니다.

### feature 개수

현재 저장된 `outputs/radius_005/feature_columns.json` 기준 feature는 총 114개입니다.

```text
33 position
+30 velocity
+27 acceleration
+ 3 recent_velocity
+ 3 mean_velocity
+ 3 recent_acceleration
+ 4 speed summary
+ 4 acceleration_norm summary
+ 4 turn_angle summary
+ 3 path shape
=114 features
```

### 전체 궤적 feature를 쓰는 이유

실제 발사 좌표 자체는 `-40ms`와 `0ms` 두 점만 사용해 계산합니다.

```text
aim_position = position_0ms + recent_velocity * 0.08
```

따라서 이 프로젝트의 모델은 “어디를 조준할지”를 새로 정하는 모델이 아닙니다. 조준점은 위의 고정 공식으로 이미 정해져 있습니다.

모델이 하는 일은 “그 고정 발사 좌표를 믿고 쏴도 되는 상황인지”를 판단하는 것입니다. 즉 전체 400ms 궤적 feature는 발사 좌표 계산용이 아니라, 고정 조준 공식의 신뢰도를 판단하기 위한 정보입니다.

예를 들어 마지막 `-40ms -> 0ms` 속도가 비슷한 두 샘플이 있더라도, 과거 궤적은 다를 수 있습니다.

```text
샘플 A: 400ms 동안 거의 직선으로 안정적으로 이동
샘플 B: 중간에 크게 휘고 최근에도 가속/회전 중
```

두 샘플의 고정 조준 좌표는 비슷하게 계산될 수 있지만, 실제 명중 가능성은 다를 수 있습니다. 샘플 A는 최근 속도 extrapolation이 비교적 믿을 만하고, 샘플 B는 80ms 뒤 위치가 현재 속도만으로 잘 설명되지 않을 가능성이 큽니다.

이 차이를 보기 위해 모델은 전체 위치, 속도, 가속도, 회전각, 경로 굴곡도 feature를 사용합니다.

```text
고정 조준 공식: 어디로 쏠지 계산
ML 모델: 그 조준이 맞을 만큼 신뢰 가능한지 판단
```

## 7. Feature 생성 샘플

`dataset/train/TRAIN_00001.csv`에서 일부 feature를 뽑으면 다음과 같습니다.

```csv
feature,value
position_00_x,2.490842
position_00_y,0.377812
position_00_z,-0.327984
velocity_00_x,1.261275
velocity_00_y,0.261175
velocity_00_z,0.382850
recent_velocity_x,1.309075
recent_velocity_y,0.163575
recent_velocity_z,0.260425
speed_mean,1.346022
speed_std,0.001378
acceleration_norm_mean,2.299905
turn_angle_mean,0.068344
path_length,0.538409
straight_distance,0.536892
tortuosity,1.002824
```

이렇게 만들어진 feature row는 `labels`와 `id` 기준으로 합쳐져 학습 dataset이 됩니다.

```text
dataset
id           position_00_x  ...  tortuosity  hit_r001  hit_r002  ...  hit_r005
TRAIN_00001       2.490842  ...    1.002824        1         1   ...        1
TRAIN_00002       ...       ...    ...             0         0   ...        1
```

## 8. 모델은 무엇을 보고 학습하는가

모델 입력 `X`는 114개 feature입니다.

```python
x_train = fit_dataset.loc[:, feature_columns]
```

모델 정답 `y`는 사용자가 지정한 반경에 해당하는 라벨 하나입니다.

```python
label_column = radius_to_label(args.radius)
y_train = fit_dataset[label_column]
```

예를 들어 `--radius 0.05`이면 `hit_r005`를 예측합니다. `--radius 0.02`이면 `hit_r002`를 예측합니다.

즉 모델은 “현재까지의 궤적 모양, 속도, 가속도, 회전 특성”을 보고 “고정 조준 공식으로 계산한 발사 좌표가 80ms 뒤 반경 안에 들어갈 가능성”을 학습합니다. 모델이 발사 좌표를 직접 바꾸지는 않고, 그 좌표로 발사할지 말지를 결정하는 신뢰도 점수를 만듭니다.

지원 모델은 두 가지입니다.

| `--model-type` | 모델 | 특징 |
|---|---|---|
| `lightgbm` | `LGBMClassifier` | 기본값, 비선형 패턴과 feature interaction 처리 |
| `logistic` | `StandardScaler + LogisticRegression` | 선형 기준선 모델 |

모델은 `predict_proba()`로 class 1, 즉 hit 확률을 냅니다.

```text
hit_probability = P(hit = 1 | trajectory features)
```

## 9. 맞췄다/못 맞췄다는 어떻게 정해지는가

prepared dataset의 라벨은 `prepare_dataset.py`에서 만들어집니다.

1. 현재 시점 `0ms` 위치를 가져옵니다.
2. 최근 속도는 `-40ms`에서 `0ms`까지의 위치 변화로 계산합니다.
3. 80ms 뒤 조준점을 계산합니다.
4. 실제 80ms 뒤 위치와 조준점의 유클리드 거리를 `error`로 계산합니다.
5. `error <= radius`이면 hit, 아니면 miss입니다.

수식으로 쓰면 다음과 같습니다.

```text
recent_velocity = (position_0ms - position_-40ms) / 0.04
aim_position = position_0ms + recent_velocity * 0.08
error = distance(aim_position, actual_future_position)
hit = 1 if error <= radius else 0
```

예를 들어 테스트 샘플 `TRAIN_00004`의 조준점은 다음과 같습니다.

```csv
id,aim_x,aim_y,aim_z
TRAIN_00004,3.563683,0.165195,1.041963
```

이 샘플의 `hit_r005=1`이면 반경 0.05 안에 들어갔다는 뜻입니다.

## 10. Threshold는 왜 필요한가

모델은 곧바로 0/1을 내지 않고 확률을 냅니다.

```text
hit_probability = 0.99985
```

이 확률을 실제 발사 결정으로 바꾸려면 threshold가 필요합니다.

```text
fire_decision = 1 if hit_probability >= threshold else 0
```

`train.py`에서 `--threshold`를 직접 지정하지 않으면, 검증 데이터에서 threshold 후보를 탐색합니다.

- 기본 후보 범위: `0.0` ~ `1.0`
- 기본 간격: `0.01`
- 선택 기준: `hit_score`가 가장 큰 threshold
- 동점이면 threshold가 더 큰 값을 선택합니다.

예시로 `outputs/radius_005/decision_threshold.json`에는 다음 값이 저장되어 있습니다.

```json
{
  "threshold": 0.89,
  "radius": 0.05,
  "validation_hit_score": 1405,
  "validation_mean_hit_score": 0.878125,
  "source": "validation_search"
}
```

## 11. Train 단계별 산출 데이터

### 1단계: prepared train 읽기

```text
trajectories: dict[id -> 11-row DataFrame]
labels: DataFrame[id, hit_r001, hit_r002, hit_r003, hit_r004, hit_r005]
metadata: schema and hash information
```

### 2단계: feature frame 생성

```text
feature_frame
id           position_00_x  position_00_y  ...  path_length  tortuosity
TRAIN_00001       2.490842       0.377812  ...     0.538409    1.002824
...
```

### 3단계: feature와 label merge

```text
dataset = feature_frame.merge(labels, on="id")
```

### 4단계: train/validation 분할

`split_train_validation()`은 라벨 조합을 기준으로 층화 분할합니다. 즉 특정 반경에서는 맞고 특정 반경에서는 틀리는 샘플 비율이 학습/검증에 최대한 비슷하게 유지됩니다.

기본값은 다음과 같습니다.

```text
dataset/train 전체: 8,000개
fit_dataset: 6,400개
valid_dataset: 1,600개
```

### 5단계: validation model 학습

검증용 모델은 `fit_dataset`만 보고 학습합니다.

```text
x_train = 114개 feature
y_train = hit_r005 같은 선택 반경 라벨
```

검증 데이터에 대해 확률을 예측하고 threshold를 선택합니다.

### 6단계: final model 학습

threshold를 고른 뒤 최종 제출/추론에 사용할 모델은 `dataset/train` 8,000개 전체로 다시 학습합니다.

### 7단계: 저장 파일

```text
hit_lgbm.pkl 또는 hit_logistic.pkl   # 최종 학습 모델
feature_columns.json                 # 추론 시 사용할 feature 순서
decision_threshold.json              # 선택된 threshold와 반경 정보
valid_metrics.json                   # validation 평가 결과
```

## 12. Infer 단계별 산출 데이터

### 1단계: prepared test 읽기

```text
trajectories: dict[id -> 11-row DataFrame]
labels: DataFrame[id, hit_r001, ..., hit_r005]
metadata: schema and test hash information
```

### 2단계: feature 생성과 schema 정렬

테스트 데이터에서도 동일한 방식으로 114개 feature를 만듭니다. 이후 `feature_columns.json`을 기준으로 컬럼 존재 여부와 순서를 맞춥니다.

이 단계가 중요한 이유는 학습 때와 추론 때 feature 순서가 달라지면 모델 입력 의미가 바뀌기 때문입니다.

### 3단계: artifact 검증

`infer.py`는 추론 전에 다음을 검증합니다.

- 선택한 `--model-type`과 threshold artifact의 `model_type`이 같은가
- 오래된 artifact처럼 `model_type`이 없으면 `lightgbm`으로 간주합니다.
- 저장된 threshold가 0~1 사이인가
- 학습 반경과 추론 반경이 같은가
- 학습 source hash와 테스트 metadata source hash가 같은가

### 4단계: 확률 예측과 발사 결정

```text
probabilities = model.predict_proba(features)[:, 1]
fire_decision = probabilities >= threshold
```

### 5단계: 조준점과 결과 저장

`build_aim_frame()`이 각 샘플의 조준점을 다시 계산하고, 예측 확률과 발사 여부를 붙입니다.

```csv
id,aim_x,aim_y,aim_z,hit_probability,fire_decision,hit_r005
TRAIN_00004,3.563683,0.165195,1.041963,0.9998515721,1,1
TRAIN_00005,2.191300,-0.079555,-0.008270,0.9996585885,1,1
TRAIN_00007,4.286490,1.259345,1.016388,0.9996306775,1,1
```

## 13. 평가 지표는 어떻게 계산하는가

평가는 `src.metrics.evaluate_predictions()`에서 수행됩니다.

입력은 세 가지입니다.

```text
y_true        # 실제 hit 라벨, 0 또는 1
probabilities # 모델이 예측한 hit 확률
threshold     # 발사 여부를 정하는 기준
```

먼저 확률을 발사 결정으로 바꿉니다.

```text
fire_decision = 1 if probability >= threshold else 0
```

그 다음 일반 classification metric과 대회식 hit score를 함께 계산합니다.

## 14. 각 평가 지표의 의미

| 지표 | 의미 | 해석 |
|---|---|---|
| `accuracy` | 전체 샘플 중 예측 결정이 실제 hit/miss와 일치한 비율 | 직관적인 정답률 |
| `precision` | 발사한 샘플 중 실제로 맞은 비율 | 발사 결정의 신중함 |
| `recall` | 실제 맞출 수 있었던 샘플 중 발사한 비율 | 기회를 놓치지 않는 정도 |
| `f1` | precision과 recall의 조화평균 | 신중함과 적극성의 균형 |
| `auroc` | threshold와 무관하게 hit 샘플의 확률을 miss보다 높게 주는 능력 | 확률 ranking 품질 |
| `hit_score` | 발사 보상 총합 | 실제 의사결정 목표에 가장 가까운 점수 |
| `mean_hit_score` | 샘플당 평균 발사 보상 | 데이터 크기가 달라도 비교 가능한 score |
| `shots_fired` | 발사 결정한 샘플 수 | 모델이 얼마나 공격적으로 발사했는지 |
| `samples` | 평가 샘플 수 | 평가 대상 개수 |

### hit score 보상 규칙

```text
fire_decision = 0                 ->  0점
fire_decision = 1 and actual = 1  -> +1점
fire_decision = 1 and actual = 0  -> -2점
```

이 규칙 때문에 단순히 많이 쏘는 모델이 항상 좋지는 않습니다. 빗나간 발사는 `-2`로 강하게 벌점을 받으므로, 모델은 “맞을 가능성이 충분히 높은 경우만 쏘는” threshold를 찾아야 합니다.

## 15. 평가 결과 예시

`outputs/radius_005/valid_metrics.json` 예시입니다.

```json
{
  "threshold": 0.89,
  "accuracy": 0.956875,
  "precision": 0.9638324873096447,
  "recall": 0.992161985630307,
  "f1": 0.9777920823945928,
  "auroc": 0.8710135461335303,
  "hit_score": 1405,
  "mean_hit_score": 0.878125,
  "shots_fired": 1576,
  "samples": 1600,
  "radius": 0.05,
  "train_samples": 6400,
  "validation_samples": 1600,
  "dataset_samples": 8000
}
```

`outputs/radius_005/test_metrics.json` 예시입니다.

```json
{
  "threshold": 0.89,
  "accuracy": 0.951,
  "precision": 0.9670781893004116,
  "recall": 0.9822361546499477,
  "f1": 0.9745982374287195,
  "auroc": 0.8510668027508446,
  "hit_score": 1752,
  "mean_hit_score": 0.876,
  "shots_fired": 1944,
  "samples": 2000,
  "radius": 0.05
}
```

이 결과는 threshold 0.89에서 2,000개 테스트 샘플 중 1,944개에 발사했고, 총 보상은 1,752점, 샘플당 평균 보상은 0.876점이라는 뜻입니다.

## 16. 발표 슬라이드 구성안

### Slide 1. 문제 정의

- 과거 400ms 모기 궤적을 기반으로 80ms 뒤 명중 가능성 판단
- 발사 좌표는 `-40ms`와 `0ms` 기반 고정 공식으로 계산
- 모델의 역할은 좌표 자체를 새로 예측하는 것이 아니라 “이 고정 조준점을 믿고 발사할지 말지” 결정하는 것
- 빗나간 발사에는 큰 벌점이 있으므로 확률과 threshold가 중요

### Slide 2. 데이터 구조

- 샘플 하나는 11개 timestep의 3D 위치
- train 8,000개, test 2,000개
- 각 샘플에는 반경별 hit 라벨이 붙어 있음
- `metadata.json`으로 split과 schema 무결성 검증

### Slide 3. 라벨 생성 원리

- 현재 위치와 최근 속도로 80ms 뒤 조준점 계산
- 실제 미래 위치와 조준점 거리 계산
- 거리 <= radius이면 hit
- 반경별로 `hit_r001` ~ `hit_r005` 생성

### Slide 4. Feature Engineering

- 11개 위치를 펼친 position feature
- 위치 차이로 velocity 계산
- 속도 차이로 acceleration 계산
- speed, acceleration norm, turn angle 통계
- path length, straight distance, tortuosity로 궤적 형태 반영
- 전체 궤적 feature는 발사 좌표 계산이 아니라 고정 조준 공식의 신뢰도 판단에 사용

### Slide 5. 학습 흐름

- `dataset/train` 로딩
- feature frame 생성
- 라벨 merge
- train/validation 층화 분할
- validation model로 threshold 탐색
- train 전체로 final model 재학습
- model, feature schema, threshold, metrics 저장

### Slide 6. 모델이 배우는 것

- 입력: 114개 궤적 feature
- 정답: 선택 반경의 hit 라벨
- 출력: hit 확률
- 의미: 고정 조준 좌표가 실제로 맞을 가능성
- LightGBM은 비선형 궤적 패턴을 학습
- Logistic Regression은 선형 baseline

### Slide 7. 추론 흐름

- `dataset/test` 로딩
- 학습 때와 동일한 feature 생성
- `feature_columns.json`으로 컬럼 정렬
- 모델 확률 예측
- threshold로 발사 여부 결정
- 조준점, 확률, 발사 여부, 실제 라벨 저장

### Slide 8. 평가 지표

- classification metric: accuracy, precision, recall, f1, auroc
- decision metric: hit_score, mean_hit_score, shots_fired
- 이 프로젝트에서는 hit_score가 실제 의사결정 목적에 가장 직접적

### Slide 9. 결과 해석 예시

- threshold 0.89
- test accuracy 0.951
- test precision 0.967
- test recall 0.982
- test hit_score 1752
- shots_fired 1944 / 2000

### Slide 10. 핵심 메시지

- 좌표 예측 문제가 아니라 발사 의사결정 문제로 재정의
- 3D 궤적의 위치, 속도, 가속도, 회전, 경로 형태를 feature로 사용
- threshold 최적화가 최종 점수에 큰 영향을 줌
- 검증용 threshold 선택과 테스트 평가를 분리해 과적합 위험을 줄임
