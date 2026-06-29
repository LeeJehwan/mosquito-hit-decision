# Mosquito Hit Decision

과거 400ms의 3D 궤적으로 80ms 뒤의 명중 가능성과 발사 여부를 예측하는 이진 분류 baseline이다. LightGBM과 Logistic Regression을 지원한다.

## Setup

```bash
uv sync
```

## Prepare Fixed Dataset

`open/`은 원본으로 유지한다. 아래 명령을 한 번 실행해 원본과 동일한 궤적 CSV 형식에 반경별 명중 라벨만 추가한 고정 데이터셋을 만든다.

```bash
uv run python prepare_dataset.py --source-dir open --dataset-dir dataset --seed 42
```

생성 결과는 다음과 같다.

```text
dataset/train/*.csv     # 8,000 sample files
dataset/test/*.csv      # 2,000 sample files
dataset/metadata.json   # schema and split hashes
```

각 샘플 CSV의 열은 다음과 같다. hit 라벨은 샘플의 11개 행에 동일하게 기록된다.

```text
timestep_ms,x,y,z,hit_r001,hit_r002,hit_r003,hit_r004,hit_r005
```

동일한 원본과 seed를 사용하면 항상 같은 분할이 생성된다. 특징 생성 로직을 변경한 경우에는 데이터셋을 다시 생성해야 한다.

## Train

기본 모델은 기존과 동일한 LightGBM이다.

```bash
uv run python train.py --dataset-dir dataset --radius 0.05
```

Logistic Regression은 다음과 같이 선택한다. 위치·속도·가속도 특징의 스케일 차이를 처리하기 위해 `StandardScaler`를 포함한 pipeline으로 학습한다.

```bash
uv run python train.py --dataset-dir dataset --radius 0.05 \
  --model-type logistic --output runs/logistic_01
```

학습은 `dataset/train/`만 읽는다. 8,000개를 기본적으로 6,400개 학습/1,600개 검증으로 층화 분할하고, 검증 데이터에서 threshold를 선택한 뒤 최종 모델은 8,000개 전체로 학습한다. 특징은 각 방법론이 고정 궤적 데이터에서 생성한다. 지원 반경은 `0.01`, `0.02`, `0.03`, `0.04`, `0.05`이다.

학습 산출물과 추론 결과를 실험별 단일 디렉터리에서 관리하려면 `--output`을 사용한다.

```bash
uv run python train.py --dataset-dir dataset --radius 0.05 --output runs/experiment_01
```

학습과 추론은 기본적으로 `tqdm` 진행률을 표시한다. 로그가 필요 없는 환경에서는 `--no-progress`를 사용한다.

```bash
uv run python train.py --dataset-dir dataset --output runs/experiment_01 --no-progress
```

위 명령은 다음 파일을 `runs/experiment_01/`에 저장한다.

```text
hit_lgbm.pkl             # LightGBM; Logistic은 hit_logistic.pkl
feature_columns.json
decision_threshold.json
valid_metrics.json
```

주요 설정은 CLI에서 변경할 수 있다. 전체 옵션은 다음 명령으로 확인한다.

```bash
uv run python train.py --help
```

`--threshold`를 생략하면 검증 데이터의 평균 hit score가 최대가 되는 값을 탐색한다. 명시하면 탐색을 건너뛴다.

```bash
uv run python train.py --radius 0.03 --threshold 0.8
```

## Infer

```bash
uv run python infer.py --dataset-dir dataset --radius 0.05
```

Logistic Regression 실험은 학습 때와 같은 `--model-type`과 `--output`을 지정한다.

```bash
uv run python infer.py --dataset-dir dataset --radius 0.05 \
  --model-type logistic --output runs/logistic_01
```

추론은 `dataset/test/`만 읽으며, test 라벨은 threshold 탐색이나 모델 선택에 사용하지 않고 최종 지표 계산에만 사용한다.

학습 때 사용한 `--output`을 그대로 지정하면 해당 디렉터리에서 모델 artifacts를 읽고 추론 결과도 함께 저장한다.

```bash
uv run python infer.py --dataset-dir dataset --radius 0.05 --output runs/experiment_01
```

추론 결과와 공통 평가 지표는 각각 `runs/experiment_01/hit_predictions.csv`, `runs/experiment_01/test_metrics.json`에 저장된다. `--model-path`, `--feature-path`, `--threshold-path`, `--output-path`, `--metrics-output`을 지정하면 개별 경로가 `--output`보다 우선한다.

LightGBM과 Logistic Regression의 threshold와 metrics가 서로 덮어쓰여지지 않도록 모델별로 별도 `--output` 디렉터리를 사용한다. 추론 시 선택한 모델 유형은 threshold artifact에 기록된 유형과 일치해야 한다.

저장된 threshold를 일시적으로 덮어쓸 수 있다.

```bash
uv run python infer.py --dataset-dir dataset --radius 0.05 --threshold 0.7
```

결과는 기본적으로 `outputs/hit_predictions.csv`와 `outputs/test_metrics.json`에 저장된다. 추론 반경은 학습 artifact에 기록된 반경과 일치해야 한다.

## Test

```bash
uv run pytest
```

## Compare Experiments

여러 학습 결과 JSON을 표와 그래프로 비교하려면 다음 노트북을 실행한다.

```bash
uv run jupyter lab notebooks/compare_experiments.ipynb
```

노트북의 `RESULT_FILES`에 `train.py --metrics-output`으로 생성한 JSON 경로나 glob 패턴을 입력한다.
