# AI프로그래밍 코딩테스트 코드 백과사전

> 1~14강 교안 PDF와 [수업 GitHub labs](https://github.com/dearyonghoon/course-ai-programming-2026/tree/main/labs) 노트북을 바탕으로 정리했습니다.
> **사용처(언제) → 목적(왜) → 코드(복붙)** 순서입니다.
> 모든 코드 블록은 TensorFlow 2.21 / Keras 3 / scikit-learn 1.9 환경에서 수업 데이터로 실행해 확인했습니다.
> (MNIST/CIFAR 예제는 같은 형태의 대체 데이터로 확인, `1-4`와 `1-6`의 파일 이름은 예시라서 바꿔 써야 합니다.)
> 개정판에서 새로 추가한 코드 중 **NumPy·pandas 부분은 실행해 확인**했고, **Keras 부분(14-I, 17장의 L11 헬퍼 함수 등)은 문법 검사만** 했습니다. 처음 쓸 때 한 번 실행해 보세요.

## 사용법

- `Ctrl+F`로 **키워드**(예: `분할`, `결측`, `혼동행렬`, `학습곡선`, `momentum`)를 검색합니다.
- GitHub에서 코드 블록 오른쪽 위의 **복사 버튼**을 누르면 바로 복사됩니다.
- 변수 이름은 문서 전체에서 통일했습니다: `df`, `X`, `y`, `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`, `model`, `history`
- 처음부터 끝까지 한 번에 돌아가는 완성 코드는 [14. 기출 유형별 완성 템플릿](#14-기출-유형별-완성-템플릿)에 있습니다.
- 수업 GitHub 노트북(L02~L14)의 **최종 실습·도전 문제 정답과 꼭 기억할 것**은 [17장](#17-수업-github-실습-복습-l02l14-최종-실습도전-문제-정답)에 있습니다. 교수님이 실습 코드 복습을 강조하셨으니 시험 전에 한 번 훑어보세요.
- 시험 직전에는 [16. 시험 전 체크리스트](#16-시험-전-체크리스트)를 한 번 보세요.

**Colab 단축키**

| 동작 | 단축키 |
|---|---|
| 셀 실행 후 다음 셀로 | `Shift + Enter` |
| 아래에 코드 셀 추가 | `Ctrl + M`, `B` |
| 셀 삭제 | `Ctrl + M`, `D` |
| 선택 줄 주석 처리/해제 | `Ctrl + /` |
| 실행 중단 | `Ctrl + M`, `I` |
| 함수 도움말 | 함수 이름 뒤에 `?` 실행 (예: `train_test_split?`) |

## 목차

- **[0. 기본 import 세트](#0-기본-import-세트)**
- **[1. 데이터 불러오기](#1-데이터-불러오기)**
  - [1-1. Google Drive 마운트 후 CSV 읽기](#1-1-google-drive-마운트-후-csv-읽기)
  - [1-2. 헤더가 없는 CSV](#1-2-헤더가-없는-csv)
  - [1-3. 인터넷 URL에서 바로 읽기 (Drive 없이)](#1-3-인터넷-url에서-바로-읽기-drive-없이)
  - [1-4. 내 컴퓨터에서 파일 업로드](#1-4-내-컴퓨터에서-파일-업로드)
  - [1-5. Keras 내장 데이터셋](#1-5-keras-내장-데이터셋)
  - [1-6. 엑셀, 구분자가 다른 파일](#1-6-엑셀-구분자가-다른-파일)
- **[2. 데이터 확인 (EDA)](#2-데이터-확인-eda)**
  - [2-1. 상관관계와 히트맵](#2-1-상관관계와-히트맵)
  - [2-2. 클래스별 분포 비교 히스토그램 (6강 Pima)](#2-2-클래스별-분포-비교-히스토그램-6강-pima)
  - [2-3. 산점도 행렬 (6강 Iris)](#2-3-산점도-행렬-6강-iris)
- **[3. DataFrame 조작 (슬라이싱, 필터링, 라벨링)](#3-dataframe-조작-슬라이싱-필터링-라벨링)**
  - [3-1. 열(컬럼) 선택](#3-1-열컬럼-선택)
  - [3-2. 행 선택 (슬라이싱)](#3-2-행-선택-슬라이싱)
  - [3-3. 조건 필터링](#3-3-조건-필터링)
  - [3-4. 라벨링 (새 컬럼 만들기)](#3-4-라벨링-새-컬럼-만들기)
  - [3-5. 정렬, 그룹 통계, 기타](#3-5-정렬-그룹-통계-기타)
- **[4. 결측치 처리](#4-결측치-처리)**
  - [4-1. 결측치 확인](#4-1-결측치-확인)
  - [4-2. 결측치 제거 (dropna)](#4-2-결측치-제거-dropna)
  - [4-3. 0을 결측으로 보고 처리 (Pima)](#4-3-0을-결측으로-보고-처리-pima)
  - [4-4. 결측치 채우기 (fillna)](#4-4-결측치-채우기-fillna)
  - [4-5. 누수 없이 채우기 (11강 Ames 방식)](#4-5-누수-없이-채우기-11강-ames-방식)
- **[5. 라벨/범주형 인코딩](#5-라벨범주형-인코딩)**
  - [5-1. 라벨 형태 ↔ 손실 함수 짝 (가장 중요)](#5-1-라벨-형태--손실-함수-짝-가장-중요)
  - [5-2. 라벨 원핫 인코딩](#5-2-라벨-원핫-인코딩)
  - [5-3. 문자열 라벨 → 정수 (LabelEncoder)](#5-3-문자열-라벨--정수-labelencoder)
  - [5-4. 입력 특성의 범주형 원핫 (pd.get_dummies)](#5-4-입력-특성의-범주형-원핫-pdget_dummies)
  - [5-5. 입력 특성의 범주형 원핫 (OneHotEncoder, 11강 정석)](#5-5-입력-특성의-범주형-원핫-onehotencoder-11강-정석)
- **[6. 데이터 분할 (train / val / test)](#6-데이터-분할-train--val--test)**
  - [6-1. train : test = 8 : 2](#6-1-train--test--8--2)
  - [6-2. train : val : test = 6 : 2 : 2 (두 번 분할)](#6-2-train--val--test--6--2--2-두-번-분할)
  - [6-3. validation_split vs validation_data](#6-3-validation_split-vs-validation_data)
  - [6-4. 시계열 분할 (섞으면 안 됨, 11강 ETTh1)](#6-4-시계열-분할-섞으면-안-됨-11강-etth1)
  - [6-5. 슬라이딩 윈도우 만들기 (과거 N개 → 미래 M개)](#6-5-슬라이딩-윈도우-만들기-과거-n개--미래-m개)
- **[7. 스케일링 (표준화, 정규화)](#7-스케일링-표준화-정규화)**
  - [7-1. 입력 표준화 (StandardScaler)](#7-1-입력-표준화-standardscaler)
  - [7-2. 0~1 정규화 (MinMaxScaler)](#7-2-01-정규화-minmaxscaler)
  - [7-3. 목표값(y) 표준화와 원래 단위 복원 (회귀)](#7-3-목표값y-표준화와-원래-단위-복원-회귀)
  - [7-4. 로그 변환 목표값 (11강)](#7-4-로그-변환-목표값-11강)
  - [7-5. 이미지 정규화](#7-5-이미지-정규화)
  - [7-6. 수치형(표준화) + 범주형(원핫) 합치기 (11강 Ames)](#7-6-수치형표준화--범주형원핫-합치기-11강-ames)
  - [7-7. 특성 선택 (Top-K, 11강)](#7-7-특성-선택-top-k-11강)
- **[8. 모델 구성 (출력층·손실 짝 포함)](#8-모델-구성-출력층손실-짝-포함)**
  - [8-1. Sequential 작성 방식 두 가지](#8-1-sequential-작성-방식-두-가지)
  - [8-2. 이진 분류 (6강 Pima, Wine)](#8-2-이진-분류-6강-pima-wine)
  - [8-3. 다중 분류, 다중 라벨 (6강 Iris)](#8-3-다중-분류-다중-라벨-6강-iris)
  - [8-4. 회귀 (출력 1개, k개)](#8-4-회귀-출력-1개-k개)
  - [8-5. 층 옵션 모음 (초기화, BN, Dropout, L2, 활성화)](#8-5-층-옵션-모음-초기화-bn-dropout-l2-활성화)
  - [8-6. CNN](#8-6-cnn)
  - [8-7. 시계열 MLP (11강 ETTh1)](#8-7-시계열-mlp-11강-etth1)
  - [8-8. 모델을 함수로 만들기 (여러 설정 비교할 때)](#8-8-모델을-함수로-만들기-여러-설정-비교할-때)
- **[9. 컴파일: 손실, 옵티마이저, 사용자 정의 손실](#9-컴파일-손실-옵티마이저-사용자-정의-손실)**
  - [9-1. compile 기본형](#9-1-compile-기본형)
  - [9-2. 옵티마이저](#9-2-옵티마이저)
  - [9-3. 사용자 정의 손실 (Custom Loss)](#9-3-사용자-정의-손실-custom-loss)
  - [9-4. Keras 내장 손실로 값 직접 계산 (7강)](#9-4-keras-내장-손실로-값-직접-계산-7강)
- **[10. 콜백과 학습 (fit)](#10-콜백과-학습-fit)**
  - [10-1. EarlyStopping (조기 종료)](#10-1-earlystopping-조기-종료)
  - [10-2. ModelCheckpoint (최고 모델 저장) + 불러오기](#10-2-modelcheckpoint-최고-모델-저장--불러오기)
  - [10-3. ReduceLROnPlateau, TensorBoard](#10-3-reducelronplateau-tensorboard)
  - [10-4. 과적합 억제 기법 모음](#10-4-과적합-억제-기법-모음)
  - [10-5. 데이터 증강 (10강)](#10-5-데이터-증강-10강)
  - [10-6. fit 기본형과 옵션](#10-6-fit-기본형과-옵션)
  - [10-7. 모델, 스케일러 저장/불러오기](#10-7-모델-스케일러-저장불러오기)
- **[11. 평가, 예측, 혼동행렬](#11-평가-예측-혼동행렬)**
  - [11-1. evaluate (테스트 성능)](#11-1-evaluate-테스트-성능)
  - [11-2. predict (예측값 만들기)](#11-2-predict-예측값-만들기)
  - [11-3. 혼동행렬과 분류 지표](#11-3-혼동행렬과-분류-지표)
  - [11-4. 회귀 지표](#11-4-회귀-지표)
  - [11-5. 틀린 이미지 확인 (14강)](#11-5-틀린-이미지-확인-14강)
  - [11-6. 여러 모델 결과를 표로 비교](#11-6-여러-모델-결과를-표로-비교)
- **[12. 시각화 (학습곡선, 예측 비교)](#12-시각화-학습곡선-예측-비교)**
  - [12-1. 학습곡선 (Learning Curve)](#12-1-학습곡선-learning-curve)
  - [12-2. 예측 vs 실제 산점도 (회귀)](#12-2-예측-vs-실제-산점도-회귀)
  - [12-3. 예측값과 실제값을 선으로 (시계열, 샘플 순서)](#12-3-예측값과-실제값을-선으로-시계열-샘플-순서)
  - [12-4. 데이터, 함수 그래프](#12-4-데이터-함수-그래프)
- **[13. NumPy 직접 구현과 계산 문제](#13-numpy-직접-구현과-계산-문제)**
  - [13-1. numpy 랜덤 + 조건 게임](#13-1-numpy-랜덤--조건-게임)
  - [13-2. 활성화/손실 함수 직접 계산](#13-2-활성화손실-함수-직접-계산)
  - [13-3. Affine 계산과 Keras 가중치 확인](#13-3-affine-계산과-keras-가중치-확인)
  - [13-4. 가중치 업데이트 (GD, momentum, Adam)](#13-4-가중치-업데이트-gd-momentum-adam)
  - [13-5. 선형/로지스틱 회귀 직접 구현 (3강, 4강)](#13-5-선형로지스틱-회귀-직접-구현-3강-4강)
  - [13-6. 역전파 직접 계산 (5강)](#13-6-역전파-직접-계산-5강)
  - [13-7. BN, LN, Dropout 직접 계산 (9강, 10강)](#13-7-bn-ln-dropout-직접-계산-9강-10강)
  - [13-8. 파라미터 수, CNN 크기 계산기](#13-8-파라미터-수-cnn-크기-계산기)
  - [13-9. 초기화 표준편차 (8강)](#13-9-초기화-표준편차-8강)
  - [13-10. 확률, 통계 계산 (2강)](#13-10-확률-통계-계산-2강)
  - [13-11. 합성곱·풀링 직접 구현 (12강)](#13-11-합성곱풀링-직접-구현-12강)
- **[14. 기출 유형별 완성 템플릿](#14-기출-유형별-완성-템플릿)**
  - [14-A. 범용 회귀 (CSV → 결측 → 범주형 → 스케일링 → MLP)](#14-a-범용-회귀-csv--결측--범주형--스케일링--mlp)
  - [14-B. 다중 출력 회귀 (2D ~ 6D 출력)](#14-b-다중-출력-회귀-2d--6d-출력)
  - [14-C. 이진 분류 (Pima) + 혼동행렬](#14-c-이진-분류-pima--혼동행렬)
  - [14-D. 다중 분류 (Iris) + 혼동행렬](#14-d-다중-분류-iris--혼동행렬)
  - [14-E. CNN (MNIST 분류)](#14-e-cnn-mnist-분류)
  - [14-F. 시계열 예측 (ETTh1: 과거 96 → 미래 24)](#14-f-시계열-예측-etth1-과거-96--미래-24)
  - [14-G. numpy 랜덤 게임 시뮬레이션](#14-g-numpy-랜덤-게임-시뮬레이션)
  - [14-H. 조기 종료 + 체크포인트 + 학습곡선 (6강 Wine 전체)](#14-h-조기-종료--체크포인트--학습곡선-6강-wine-전체)
  - [14-I. Fashion-MNIST CNN 종합 (기출 항목 전부)](#14-i-fashion-mnist-cnn-종합-기출-항목-전부)
- **[15. 자주 나는 오류와 해결](#15-자주-나는-오류와-해결)**
- **[16. 시험 전 체크리스트](#16-시험-전-체크리스트)**
- **[17. 수업 GitHub 실습 복습 (L02~L14): 최종 실습·도전 문제 정답](#17-수업-github-실습-복습-l02l14-최종-실습도전-문제-정답)**
  - [17-0. 꼭 기억할 것 한눈에 보기](#17-0-꼭-기억할-것-한눈에-보기)
  - [17-1. L02 AI를 위한 수학](#17-1-l02-ai를-위한-수학)
  - [17-2. L03 선형 회귀](#17-2-l03-선형-회귀)
  - [17-3. L04 로지스틱 회귀와 분류](#17-3-l04-로지스틱-회귀와-분류)
  - [17-4. L05 신경망 (Forward, Backpropagation)](#17-4-l05-신경망-forward-backpropagation)
  - [17-5. L06 분류 실습 (Pima, Iris, Wine) 직접 해보기](#17-5-l06-분류-실습-pima-iris-wine-직접-해보기)
  - [17-6. L07 손실 함수](#17-6-l07-손실-함수)
  - [17-7. L08 신경망 학습 (Optimizer, LR, Initialization)](#17-7-l08-신경망-학습-optimizer-lr-initialization)
  - [17-8. L09 학습 안정화 (Activation, Scaling, BN, LN)](#17-8-l09-학습-안정화-activation-scaling-bn-ln)
  - [17-9. L10 정규화 (Regularization)](#17-9-l10-정규화-regularization)
  - [17-10. L11 회귀 실습 (Ames, ETTh1, 기구학) 체크리스트와 직접 해보기](#17-10-l11-회귀-실습-ames-etth1-기구학-체크리스트와-직접-해보기)
  - [17-11. L14 CNN (Fashion-MNIST 직접 해보기, TensorBoard)](#17-11-l14-cnn-fashion-mnist-직접-해보기-tensorboard)

### 기출 유형 → 바로가기

| 기출 유형 | 바로가기 |
|---|---|
| numpy 랜덤 + 조건 게임 | [13-1](#13-1-numpy-랜덤--조건-게임), [14-G](#14-g-numpy-랜덤-게임-시뮬레이션) |
| 다른 데이터셋으로 회귀 | [14-A](#14-a-범용-회귀-csv--결측--범주형--스케일링--mlp) |
| 결측치 제거 | [4](#4-결측치-처리) |
| DataFrame 라벨링, 슬라이싱 | [3](#3-dataframe-조작-슬라이싱-필터링-라벨링) |
| 사용자 정의 손실 | [9-3](#9-3-사용자-정의-손실-custom-loss) |
| 학습곡선 시각화 | [12-1](#12-1-학습곡선-learning-curve) |
| 2차원 출력 회귀 | [14-B](#14-b-다중-출력-회귀-2d--6d-출력) |
| Affine 계산 | [13-3](#13-3-affine-계산과-keras-가중치-확인) |
| w 업데이트, momentum | [13-4](#13-4-가중치-업데이트-gd-momentum-adam) |
| train/val 분할 | [6](#6-데이터-분할-train--val--test) |
| 과적합 억제 기법 적용 | [10-4](#10-4-과적합-억제-기법-모음) |
| weight decay | [9-2](#9-2-옵티마이저), [10-4](#10-4-과적합-억제-기법-모음) |
| 혼동행렬 출력 | [11-3](#11-3-혼동행렬과-분류-지표) |
| CCE, BCE 계산 | [13-2](#13-2-활성화손실-함수-직접-계산) |
| CNN | [8-6](#8-6-cnn), [14-E](#14-e-cnn-mnist-분류) |
| 수업 실습 최종 실습·도전 문제 | [17](#17-수업-github-실습-복습-l02l14-최종-실습도전-문제-정답) |

---

## 0. 기본 import 세트

**사용처:** 시험 시작하자마자 첫 셀에 붙여 넣기
**목적:** 필요한 라이브러리를 한 번에 불러와 `NameError`를 예방
**검색 키워드:** import, 라이브러리, 첫 셀, NameError, seed, 시드, 재현성, set_random_seed

```python
# ── 데이터 처리 · 그래프 ─────────────────────────────────────────
import numpy as np                 # 배열 계산 (np.array, np.mean, np.where ...)
import pandas as pd                # 표 데이터 (DataFrame: read_csv, iloc, fillna ...)
import matplotlib.pyplot as plt    # 그래프 (plt.plot, plt.hist, plt.show ...)
import seaborn as sns              # 히트맵(heatmap), 산점도 행렬(pairplot)
import tensorflow as tf            # Keras 포함 (tf.keras ...)

# ── scikit-learn: 분할 · 전처리 · 평가 지표 ─────────────────────────
from sklearn.model_selection import train_test_split          # train/val/test 분할
from sklearn.preprocessing import (
    StandardScaler,     # 평균 0, 표준편차 1 표준화
    MinMaxScaler,       # 0~1 정규화
    OneHotEncoder,      # 범주형 입력 → 0/1 열 (분할 후 train 기준)
    LabelEncoder,       # 문자열 라벨 → 0, 1, 2 정수
)
from sklearn.metrics import (
    mean_absolute_error,      # 회귀 MAE
    mean_squared_error,       # 회귀 MSE (RMSE = np.sqrt(MSE))
    r2_score,                 # 회귀 결정계수 R²
    accuracy_score,           # 분류 정확도
    confusion_matrix,         # 혼동행렬 (행=실제, 열=예측)
    ConfusionMatrixDisplay,   # 혼동행렬 그림
    classification_report,    # precision/recall/f1 한 번에
    precision_score,
    recall_score,
    f1_score,
)

# ── Keras: 모델 · 층 · 옵티마이저 · 콜백 ───────────────────────────
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Activation,
    Flatten, Conv2D, MaxPooling2D, ReLU, LeakyReLU,
)
from tensorflow.keras.optimizers import SGD, Adam, AdamW, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2           # 가중치 L2 규제 (kernel_regularizer=l2(0.001))
from tensorflow.keras.utils import to_categorical      # 정수 라벨 → 원핫

# ── 재현성 · 출력 형식 ─────────────────────────────────────────────
tf.keras.utils.set_random_seed(42)              # numpy, python random, tf 시드를 한 번에 고정
np.set_printoptions(precision=4, suppress=True) # 소수 4자리, 지수표기(1e-05) 대신 일반 숫자로
print(tf.__version__)                           # 버전 확인 (Keras 3이면 2.16 이상)
```

> **팁:** Colab 기본 폰트에는 한글이 없어서 그래프 제목을 한글로 쓰면 네모 칸(□)으로 깨집니다. 그래프 라벨은 영어로 쓰세요.

[⬆ 목차로 돌아가기](#목차)

---

## 1. 데이터 불러오기

### 1-1. Google Drive 마운트 후 CSV 읽기

**사용처:** 데이터 파일이 Drive에 있을 때 (수업 기본 방식)
**목적:** Colab에서 Drive 경로에 접근
**검색 키워드:** 드라이브, Drive, mount, 마운트, read_csv, CSV 읽기, 경로, path

```python
from google.colab import drive
drive.mount('/content/drive')      # 실행하면 권한 허용 창 → 허용하면 /content/drive 아래에 Drive가 붙음

# 경로: 왼쪽 파일 탭에서 파일 우클릭 → '경로 복사'로 붙여 넣으면 오타가 없음
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data/pima-indians-diabetes3.csv')
df.head()                          # 앞 5행 확인 (셀 마지막 줄이면 print 없이도 표로 출력됨)
```

[⬆ 목차로 돌아가기](#목차)

### 1-2. 헤더가 없는 CSV

**사용처:** 첫 줄부터 데이터인 파일 (예: `wine.csv`)
**목적:** 첫 줄이 컬럼 이름으로 잡혀서 데이터가 1행 사라지는 것을 방지
**검색 키워드:** header=None, 헤더 없음, 컬럼 이름, names, wine, 열 번호

```python
# header=None → 첫 줄도 데이터로 읽고, 열 이름은 0, 1, 2, ... 정수로 붙음
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data/wine.csv', header=None)
# 이때 열 선택은 df[12] 또는 df.iloc[:, 12]  (문자열 '12' 아님)

# 컬럼 이름을 직접 붙이고 싶을 때 (names 개수 = 열 개수)
# df = pd.read_csv('wine.csv', header=None, names=['f0','f1',...,'label'])
```

[⬆ 목차로 돌아가기](#목차)

### 1-3. 인터넷 URL에서 바로 읽기 (Drive 없이)

**사용처:** Drive 마운트가 귀찮거나 오류가 날 때. 수업 GitHub의 `data` 폴더에 데이터가 모두 있습니다.
**목적:** 인증 없이 바로 데이터 로드 (시험에서 인터넷 사용 허용)
**검색 키워드:** URL, 인터넷, GitHub raw, 데이터셋 목록, pima, iris, wine, ames, ETTh1, IRB2400

```python
# raw.githubusercontent.com 주소여야 CSV 원본이 읽힘 (github.com/.../blob/... 주소는 HTML이라 오류)
BASE = 'https://raw.githubusercontent.com/dearyonghoon/course-ai-programming-2026/main/data/'

df_pima = pd.read_csv(BASE + 'pima-indians-diabetes3.csv')   # 768 x 9, 라벨 'diabetes' (0/1)
df_iris = pd.read_csv(BASE + 'iris3.csv')                    # 150 x 5, 라벨 'species' (문자열 3종)
df_wine = pd.read_csv(BASE + 'wine.csv', header=None)        # 6497 x 13, 라벨 12번 열 (1=red, 0=white)
df_ames = pd.read_csv(BASE + 'ames_train.csv')               # 1460 x 81, 목표 'SalePrice'
df_ett  = pd.read_csv(BASE + 'ETTh1.csv')                    # 17420 x 8, 목표 'OT' (시간 순서 유지!)
df_irb  = pd.read_csv(BASE + 'datasetIRB2400.csv')           # 300000 x 18 (기구학, 크니까 로딩 수 초)
```

[⬆ 목차로 돌아가기](#목차)

### 1-4. 내 컴퓨터에서 파일 업로드

**사용처:** 시험장에서 새 CSV 파일을 받았을 때
**목적:** 로컬 파일을 Colab 세션에 업로드
**검색 키워드:** 업로드, upload, files.upload, 내 컴퓨터, 로컬 파일

```python
from google.colab import files
uploaded = files.upload()               # 파일 선택 창이 뜸 → 고르면 /content/ 에 저장됨
print(list(uploaded.keys()))            # 업로드된 파일 이름 확인 (같은 이름이 있으면 '파일 (1).csv'처럼 바뀜)
df = pd.read_csv('업로드한파일이름.csv')   # 업로드한 파일 이름 그대로

# 이름을 몰라도 되는 방법: 방금 올린 첫 파일을 바로 읽기
# df = pd.read_csv(next(iter(uploaded)))
```

> 세션이 끊기면(런타임 초기화) 업로드한 파일이 사라집니다. 다시 업로드하세요.

[⬆ 목차로 돌아가기](#목차)

### 1-5. Keras 내장 데이터셋

**사용처:** MNIST, Fashion-MNIST, CIFAR-10 문제
**목적:** 다운로드 없이 바로 이미지 데이터 사용
**검색 키워드:** mnist, fashion_mnist, cifar10, load_data, 이미지 데이터, 내장 데이터셋

```python
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

# 반환: ((훈련 이미지, 훈련 라벨), (테스트 이미지, 테스트 라벨))  → 괄호 모양 그대로 받기
(X_train, y_train), (X_test, y_test) = mnist.load_data()           # (60000,28,28), 라벨 정수 0~9
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()  # 동일한 형태 (옷 10종)
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()        # (50000,32,32,3), 라벨 (N,1)
#   └ CIFAR 라벨은 (N,1) 2차원 → y_train = y_train.ravel() 로 (N,)로 펴서 쓰면 편함
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train.dtype, X_train.min(), X_train.max())   # uint8, 0, 255 → 7-5처럼 /255 정규화 필요
```

**Fashion-MNIST 클래스 이름 (그래프 제목용):**

```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   # 라벨 0~9 순서
```

[⬆ 목차로 돌아가기](#목차)

### 1-6. 엑셀, 구분자가 다른 파일

**검색 키워드:** 엑셀, excel, xlsx, 탭, tab, sep, 구분자, 인코딩, encoding, cp949, 한글 깨짐

```python
df = pd.read_excel('data.xlsx')                 # 엑셀 (sheet_name='시트이름'으로 시트 지정 가능)
df = pd.read_csv('data.txt', sep='\t')          # 탭 구분 (세미콜론이면 sep=';', 공백이면 sep=r'\s+')
df = pd.read_csv('data.csv', encoding='cp949')  # 한글 깨질 때 (또는 'euc-kr', 'utf-8-sig')
```

[⬆ 목차로 돌아가기](#목차)

---

## 2. 데이터 확인 (EDA)

**사용처:** 데이터를 받자마자 구조를 파악할 때
**목적:** 행/열 수, 자료형, 결측치, 라벨 분포 확인
**검색 키워드:** EDA, shape, head, info, describe, dtypes, value_counts, 클래스 불균형, nunique, 데이터 확인

```python
print(df.shape)                 # (행, 열)  예: Pima (768, 9)
df.head()                       # 앞 5행 (df.tail()은 뒤 5행, df.head(10)은 앞 10행)
df.info()                       # 컬럼별 자료형, 결측 아닌 개수 (Non-Null Count < 행 수면 결측 있음)
df.describe()                   # 수치형 컬럼 통계 (평균, 표준편차, 최소, 최대, 사분위)
                                #  └ min이 0인 bmi/plasma 등은 '측정 누락'일 수 있음 (4-3 참고)
df.columns                      # 컬럼 이름 목록
df.dtypes                       # 자료형 (object = 문자열 → 원핫 필요)
df.isnull().sum()               # 컬럼별 결측치 개수
df['diabetes'].value_counts()   # 라벨 분포 (클래스 불균형 확인)
df['diabetes'].value_counts(normalize=True)   # 비율로 (합 = 1)
df.nunique()                    # 컬럼별 고유값 개수 (적으면 범주형일 가능성)
```

> 셀에서 여러 줄을 쓰면 **마지막 줄만** 화면에 출력됩니다. 다 보고 싶으면 `print()`로 감싸거나 셀을 나누세요.

[⬆ 목차로 돌아가기](#목차)

### 2-1. 상관관계와 히트맵

**사용처:** 어떤 특성이 목표와 관련이 큰지 볼 때 (6강 Pima)
**목적:** 특성 선택, 데이터 이해
**검색 키워드:** 상관관계, 상관계수, corr, heatmap, 히트맵, numeric_only, 특성 선택

```python
# 상관계수: -1 ~ 1 (1에 가까우면 같이 증가, -1이면 반대로, 0이면 선형 관계 없음)
corr = df.corr(numeric_only=True)          # 문자열 컬럼이 있으면 numeric_only=True 필수

# 목표 열('diabetes')과의 상관 순위: 큰 값일수록 목표와 관련이 큼
print(corr['diabetes'].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,          # 칸 안에 숫자 표시
    fmt='.2f',           # 소수 둘째 자리
    cmap='coolwarm',     # 음수 파랑, 양수 빨강
    vmin=-1, vmax=1,     # 색 범위를 상관계수 범위로 고정
    linewidths=0.5,      # 칸 사이 선
)
plt.show()
```

[⬆ 목차로 돌아가기](#목차)

### 2-2. 클래스별 분포 비교 히스토그램 (6강 Pima)

**검색 키워드:** 히스토그램, hist, 분포 비교, 클래스별, negative, positive, histtype

```python
# 라벨(0/1)로 행을 나눔 → 같은 특성이 두 그룹에서 어떻게 다른지 비교
negative = df[df['diabetes'] == 0]     # 당뇨 아님
positive = df[df['diabetes'] == 1]     # 당뇨

plt.hist(
    [negative['plasma'], positive['plasma']],   # 리스트로 두 그룹을 함께 넣음
    bins=30,                                    # 막대(구간) 개수
    histtype='step',                            # 선으로만 그려서 겹쳐도 보이게
    linewidth=2,
    label=['negative', 'positive'],
    # density=True,                             # 개수 대신 비율(면적=1)로 비교하고 싶을 때
)
plt.xlabel('plasma')
plt.legend()
plt.show()
```

[⬆ 목차로 돌아가기](#목차)

### 2-3. 산점도 행렬 (6강 Iris)

**검색 키워드:** pairplot, 산점도 행렬, scatter matrix, hue, iris

```python
# 모든 수치 열 쌍의 산점도 + 대각선은 분포. hue=라벨 열 → 클래스별 색
sns.pairplot(df_iris, hue='species')
plt.show()
```

[⬆ 목차로 돌아가기](#목차)

---

## 3. DataFrame 조작 (슬라이싱, 필터링, 라벨링)

### 3-1. 열(컬럼) 선택

**사용처:** 입력 X와 라벨 y 분리
**목적:** 원하는 열만 꺼내기
**검색 키워드:** 열 선택, 컬럼 선택, iloc, X y 분리, drop columns, values, to_numpy, 입력 라벨 분리

```python
# 위치(번호)로: iloc[행, 열]  →  끝 번호는 포함 안 됨
X = df.iloc[:, 0:8]        # 모든 행(:), 0~7번 열 → DataFrame (768, 8)
y = df.iloc[:, 8]          # 8번 열 하나 → Series (768,)
X = df.iloc[:, :-1]        # 마지막 열 빼고 전부
y = df.iloc[:, -1]         # 마지막 열 (-1 = 뒤에서 첫 번째)

# 이름으로
X = df[['plasma', 'bmi', 'age']]           # 여러 열 → 대괄호 2개 (DataFrame)
y = df['diabetes']                          # 한 열 → Series
X = df.drop(columns=['diabetes'])           # 특정 열 제외한 나머지 (원본 df는 그대로)
X_ames = df_ames.drop(columns=['Id', 'SalePrice'])   # 여러 열 제외 (Ames 예)

# numpy 배열로 변환 (Keras, 수식 계산용)
X_np = X.values          # 또는 X.to_numpy()
```

> `df['a','b']`(대괄호 1개)는 오류입니다. 여러 열은 반드시 `df[['a','b']]`.

[⬆ 목차로 돌아가기](#목차)

### 3-2. 행 선택 (슬라이싱)

**검색 키워드:** 행 선택, 슬라이싱, iloc, loc, sample, 무작위 행, 범위

```python
df.iloc[0]            # 첫 행 (Series로 나옴)
df.iloc[10:20]        # 10~19행 (20 미포함)
df.iloc[:100, :3]     # 앞 100행, 앞 3열
df.loc[0:5, 'plasma':'bmi']   # loc은 이름(인덱스 라벨) 기준이고 끝을 '포함'함 (0~5행 = 6개 행)
df.loc[df['age'] > 50, ['plasma', 'bmi']]   # 조건 + 열 이름 (행 조건, 열 목록)
df.sample(5, random_state=42)  # 무작위 5행 (frac=0.1이면 10%)
```

> **주의:** `iloc[a:b]`는 b **미포함**, `loc[a:b]`는 b **포함**입니다.

[⬆ 목차로 돌아가기](#목차)

### 3-3. 조건 필터링

**사용처:** 특정 조건을 만족하는 행만 뽑을 때
**목적:** 부분 데이터 추출, 이상치 제거
**검색 키워드:** 조건 필터, 필터링, filter, &, |, ~, isin, str.contains, query, 이상치 제거, 조건 개수

```python
df[df['age'] >= 50]                                   # 단일 조건
df[(df['age'] >= 50) & (df['bmi'] > 30)]              # AND: & + 괄호 필수 (and 쓰면 오류)
df[(df['age'] < 25) | (df['diabetes'] == 1)]          # OR: |  (or 쓰면 오류)
df[~(df['bmi'] == 0)]                                 # NOT: ~  (= df[df['bmi'] != 0])
df_iris[df_iris['species'].isin(['Iris-setosa', 'Iris-virginica'])]   # 여러 값 중 하나
df_iris[df_iris['species'].str.contains('setosa')]    # 문자열 포함
df.query('age >= 50 and bmi > 30')                    # 문자열 조건식 (여기선 and/or 사용 가능)

# 조건 만족 개수, 비율  (True=1, False=0 으로 계산됨)
(df['age'] >= 50).sum()       # 개수
(df['age'] >= 50).mean()      # 비율 (0~1)
```

**이상치 제거 (IQR 방식, 필요할 때):**

```python
q1, q3 = df['bmi'].quantile(0.25), df['bmi'].quantile(0.75)   # 1사분위, 3사분위
iqr = q3 - q1
df_no_out = df[(df['bmi'] >= q1 - 1.5 * iqr) & (df['bmi'] <= q3 + 1.5 * iqr)]
print(df.shape, '→', df_no_out.shape)
```

[⬆ 목차로 돌아가기](#목차)

### 3-4. 라벨링 (새 컬럼 만들기)

**사용처:** "~이면 1, 아니면 0으로 라벨을 만드시오", "구간별 등급을 매기시오"
**목적:** 조건으로 새 라벨, 파생 변수 생성
**검색 키워드:** 라벨링, 새 컬럼, 파생 변수, np.where, np.select, pd.cut, qcut, map, apply, lambda, 등급, 구간

```python
# (1) 이진 라벨: np.where(조건, 참일때값, 거짓일때값)
df['old'] = np.where(df['age'] >= 50, 1, 0)

# (2) 조건이 여러 개: np.select  → 위에서부터 처음 맞는 조건의 값을 씀
conds   = [df['bmi'] < 18.5, df['bmi'] < 25, df['bmi'] < 30]
choices = ['under', 'normal', 'over']
df['bmi_class'] = np.select(conds, choices, default='obese')   # 어느 조건도 아니면 default

# (3) 구간 나누기: pd.cut (bins 경계, 기본은 (a, b] 오른쪽 포함 / right=False면 [a, b))
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 30, 50, 100],                  # (0,30], (30,50], (50,100]
    labels=['young', 'middle', 'old'],      # 구간 개수 = bins 개수 - 1
)
df['age_q'] = pd.qcut(df['age'], q=4, labels=False)    # 같은 개수로 4등분 (0~3)

# (4) 값 바꾸기: map (딕셔너리)  → 딕셔너리에 없는 값은 NaN이 됨
df_iris['label'] = df_iris['species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2,
})

# (5) 함수 적용: apply
def grade(row):                         # row = 한 행 (Series), row['열이름']으로 접근
    if row['plasma'] >= 140 and row['bmi'] >= 30:
        return 'high'
    elif row['plasma'] >= 140:
        return 'mid'
    return 'low'
df['risk'] = df.apply(grade, axis=1)            # axis=1: 행 단위 (빼먹으면 열 단위로 돌아 오류)
df['bmi_round'] = df['bmi'].apply(lambda v: round(v))   # 한 열에만 적용할 때는 Series.apply

# (6) 조건에 맞는 행만 값 바꾸기: loc[행조건, 열이름] = 값
df.loc[df['age'] >= 60, 'old'] = 2

# (7) 열 끼리 계산 (원소별로 계산됨)
df['ratio'] = df['plasma'] / (df['bmi'] + 1e-8)   # 0으로 나누기 방지용 아주 작은 수
```

[⬆ 목차로 돌아가기](#목차)

### 3-5. 정렬, 그룹 통계, 기타

**검색 키워드:** 정렬, sort_values, groupby, 그룹별 평균, agg, crosstab, 교차표, rename, astype, reset_index, concat, 중복 제거, drop index

```python
df.sort_values('age', ascending=False)                 # 내림차순 정렬 (기본 오름차순)
df.sort_values(['diabetes', 'age'])                    # 여러 기준 (앞 기준 먼저)
df.groupby('diabetes')['plasma'].mean()                # 그룹별 평균 → 그룹 개수만큼의 Series
df.groupby('diabetes')[['plasma', 'bmi']].agg(['mean', 'std'])   # 그룹별 여러 통계 (수치 열만 지정)
pd.crosstab(df['age_group'], df['diabetes'])           # 교차표 (행 범주 × 열 범주 개수)
df.rename(columns={'plasma': 'glucose'})               # 컬럼 이름 변경 (df = 로 다시 대입해야 반영)
df['MSSubClass'] = df['MSSubClass'].astype(str)        # 자료형 변경 (숫자 코드 → 범주)
df = df.reset_index(drop=True)                         # 행 삭제 후 인덱스 재정렬 (0,1,2,...)
pd.concat([df1, df2], axis=0)                          # 위아래로 붙이기 (axis=1이면 옆으로)
df.drop_duplicates()                                   # 중복 행 제거
df.drop(index=[0, 1, 2])                               # 특정 행 삭제 (인덱스 번호)
```

[⬆ 목차로 돌아가기](#목차)

---

## 4. 결측치 처리

### 4-1. 결측치 확인

**사용처:** 모든 문제의 전처리 첫 단계
**목적:** 결측치가 있으면 학습 중 loss가 `nan`이 됨
**검색 키워드:** 결측, 결측치 확인, NaN, null, isnull, isna, missing, 결측 있는 열만, 결측 비율

```python
df.isnull().sum()                                        # 컬럼별 개수 (isna()와 같음)
df.isnull().sum().sum()                                  # 전체 개수 (숫자 하나)
df.isnull().mean().sort_values(ascending=False)          # 컬럼별 결측 비율 (0~1, 큰 순서)

# 결측 있는 컬럼만 (열별 '개수'를 먼저 구하고 0보다 큰 것만 남김)
na = df.isnull().sum()
print(na[na > 0])                      # 한 줄로: df.isnull().sum()[df.isnull().sum() > 0]

df[df.isnull().any(axis=1)]                              # 결측이 하나라도 있는 '행' (열별 개수 아님!)
```

> "결측치가 있는 **열**과 개수"를 물으면 `na[na > 0]`, "결측치가 있는 **행**"을 물으면 `df[df.isnull().any(axis=1)]`.

[⬆ 목차로 돌아가기](#목차)

### 4-2. 결측치 제거 (dropna)

**사용처:** "결측치를 제거하시오"
**목적:** 결측이 있는 행/열 삭제
**검색 키워드:** 결측 제거, dropna, subset, thresh, how='all', 열 삭제, 결측 비율 높은 열

```python
df_clean = df.dropna()                          # 결측이 하나라도 있는 행 삭제
df_clean = df.dropna(subset=['bmi', 'age'])     # 특정 열에 결측 있는 행만 삭제
df_clean = df.dropna(how='all')                 # 모든 값이 결측인 행만 삭제
df_clean = df.dropna(axis=1)                    # 결측 있는 '열' 삭제
df_clean = df.dropna(thresh=5)                  # 결측 아닌 값이 5개 이상인 행만 남김
df_clean = df_clean.reset_index(drop=True)      # 인덱스 재정렬 (iloc/loc 헷갈림 방지)

# 결측 비율이 높은 열 삭제 (예: 50% 이상)
cols_to_drop = df.columns[df.isnull().mean() > 0.5]   # 조건에 맞는 열 이름 목록
df_clean = df.drop(columns=cols_to_drop)

print(df.shape, '→', df_clean.shape)            # 삭제 전후 비교 출력
```

> **주의:** `dropna()`는 원본을 바꾸지 않습니다. `df = df.dropna()`처럼 다시 대입해야 합니다.
> X와 y를 나눈 뒤 X에서만 행을 지우면 y와 행 개수가 안 맞습니다. **나누기 전에 df에서** 지우세요.

[⬆ 목차로 돌아가기](#목차)

### 4-3. 0을 결측으로 보고 처리 (Pima)

**사용처:** 혈당, 혈압, BMI 등이 0인 것은 측정 누락이라는 문제 조건이 있을 때
**목적:** 의미 없는 0을 NaN으로 바꾼 뒤 제거하거나 채움
**검색 키워드:** 0을 결측으로, replace 0 nan, Pima 0값, 측정 누락, zero

```python
zero_cols = ['plasma', 'pressure', 'thickness', 'insulin', 'bmi']   # 0이 나올 수 없는 열
df[zero_cols] = df[zero_cols].replace(0, np.nan)   # 0 → NaN (라벨 열 'diabetes'는 넣으면 안 됨!)
print(df.isnull().sum())                           # 열별로 몇 개가 0이었는지 확인

df_drop = df.dropna()                     # 방법 1: 제거
# df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())   # 방법 2: 중앙값으로 채우기
```

[⬆ 목차로 돌아가기](#목차)

### 4-4. 결측치 채우기 (fillna)

**검색 키워드:** 결측 채우기, fillna, 평균, 중앙값, median, 최빈값, mode, ffill, 앞 값으로, 대체, impute

```python
df['col'] = df['col'].fillna(0)                            # 상수
df['col'] = df['col'].fillna(df['col'].mean())             # 평균
df['col'] = df['col'].fillna(df['col'].median())           # 중앙값 (이상치에 강함)
df['cat'] = df['cat'].fillna(df['cat'].mode()[0])          # 최빈값 (범주형), [0] 필수 (mode()는 Series)
df['cat'] = df['cat'].fillna('None')                       # '없음' 범주
df = df.fillna(df.median(numeric_only=True))               # 수치형 전체를 각 열 중앙값으로
df['ts'] = df['ts'].ffill()                                # 시계열: 앞 값으로 채우기 (bfill은 뒤 값)

# 그룹별 중앙값으로 채우기 (예: 같은 클래스끼리의 중앙값)
# transform은 원래 행 개수 그대로 값을 돌려줘서 fillna에 바로 넣을 수 있음
df['bmi'] = df['bmi'].fillna(df.groupby('diabetes')['bmi'].transform('median'))
```

> `df['col'].fillna(..., inplace=True)`는 pandas 버전에 따라 경고/미반영이 생깁니다. **`df['col'] = df['col'].fillna(...)`** 형태로 쓰세요.

[⬆ 목차로 돌아가기](#목차)

### 4-5. 누수 없이 채우기 (11강 Ames 방식)

**사용처:** 분할 후 채워야 할 때 (정석)
**목적:** 채움 값(중앙값, 최빈값)을 **train에서만** 계산해서 val/test 정보가 새지 않게
**검색 키워드:** 누수, leakage, Ames 결측, absence, 'None' 채우기, train 통계로 채우기, LotFrontage, Electrical

```python
# 1) 먼저 분할 (6장 참고) → X_train, X_val, X_test
#    .copy(): 분할 결과가 원본의 '뷰'일 때 생기는 SettingWithCopyWarning 방지
X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()

# 2) '시설 없음' 의미의 결측: 범주형은 'None', 수치형은 0
#    (예: PoolQC가 비어 있다 = 수영장이 없다 → 진짜 누락이 아님)
absence_cat_cols = [
    'PoolQC', 'Alley', 'Fence', 'FireplaceQu', 'MiscFeature', 'MasVnrType',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
]
absence_num_cols = ['GarageYrBlt', 'MasVnrArea']
for data in [X_train, X_val, X_test]:          # 세 데이터에 같은 규칙 적용
    data[absence_cat_cols] = data[absence_cat_cols].fillna('None')
    data[absence_num_cols] = data[absence_num_cols].fillna(0)

# 3) 진짜 누락: train 통계로 채움 (val/test의 값으로 계산하면 누수)
lot_frontage_median = X_train['LotFrontage'].median()   # 수치형 → 중앙값
electrical_mode = X_train['Electrical'].mode()[0]       # 범주형 → 최빈값 ([0]으로 값 하나)
for data in [X_train, X_val, X_test]:
    data['LotFrontage'] = data['LotFrontage'].fillna(lot_frontage_median)
    data['Electrical'] = data['Electrical'].fillna(electrical_mode)

print(
    X_train.isnull().sum().sum(),
    X_val.isnull().sum().sum(),
    X_test.isnull().sum().sum(),
)  # 모두 0
```

**범용 버전 (컬럼 이름을 모를 때):**

```python
num_cols = X_train.select_dtypes(include='number').columns   # 수치형 열 이름
cat_cols = X_train.select_dtypes(include='object').columns   # 문자열(범주형) 열 이름

num_fill = X_train[num_cols].median()           # train 중앙값 (열별 Series)
cat_fill = X_train[cat_cols].mode().iloc[0]     # train 최빈값 (mode()는 표 → 첫 행 = 열별 최빈값)
for data in [X_train, X_val, X_test]:
    data[num_cols] = data[num_cols].fillna(num_fill)   # Series를 넣으면 열 이름끼리 맞춰서 채움
    data[cat_cols] = data[cat_cols].fillna(cat_fill)
```

[⬆ 목차로 돌아가기](#목차)

---

## 5. 라벨/범주형 인코딩

### 5-1. 라벨 형태 ↔ 손실 함수 짝 (가장 중요)

**검색 키워드:** 출력층, 손실 짝, loss 선택, sigmoid, softmax, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy, mse, 활성화 함수 선택

| 문제 | y 형태 | 출력층 | loss |
|---|---|---|---|
| 이진 분류 | 0/1 정수 (N,) | `Dense(1, activation='sigmoid')` | `binary_crossentropy` |
| 다중 분류 (원핫) | (N, C) 원핫 | `Dense(C, activation='softmax')` | `categorical_crossentropy` |
| 다중 분류 (정수) | 0~C-1 정수 (N,) | `Dense(C, activation='softmax')` | `sparse_categorical_crossentropy` |
| 다중 라벨 | (N, C) multi-hot | `Dense(C, activation='sigmoid')` | `binary_crossentropy` |
| 회귀 1개 | 실수 (N,) | `Dense(1)` (활성화 없음) | `mse` (또는 `mae`) |
| 회귀 k개 | 실수 (N, k) | `Dense(k)` | `mse` |

> 원핫 라벨인데 `sparse_...`를 쓰거나, 정수 라벨인데 `categorical_...`을 쓰면 shape 오류가 납니다.
> 이진 분류를 `Dense(2, softmax)` + 원핫(또는 sparse)으로 풀어도 됩니다 (6강 Pima softmax 버전).

[⬆ 목차로 돌아가기](#목차)

### 5-2. 라벨 원핫 인코딩

**사용처:** 다중 분류 + `categorical_crossentropy`
**목적:** 클래스 번호를 (N, C) 벡터로
**검색 키워드:** 원핫, one-hot, get_dummies, to_categorical, 라벨 원핫, np.eye

```python
# 문자열 라벨 → 원핫 (6강 Iris 방식). 컬럼 순서는 알파벳순
y_onehot = pd.get_dummies(df_iris['species'], dtype=int)     # dtype=int 안 쓰면 True/False
print(y_onehot.columns.tolist())                             # 열 순서 = 클래스 번호 순서

# 정수 라벨 → 원핫 (14강 MNIST 방식)
y_train_oh = to_categorical(y_train, 10)       # (N,) → (N, 10), 두 번째 인자 = 클래스 개수
y_test_oh  = to_categorical(y_test, 10)

# numpy만으로: 단위행렬에서 정수 번호 행을 꺼냄 (to_categorical과 같은 결과)
# y_train_oh = np.eye(10)[y_train]

# 원핫 → 정수 되돌리기 (혼동행렬 등에 필요)
# y_int = np.argmax(y_test_oh, axis=1)
```

[⬆ 목차로 돌아가기](#목차)

### 5-3. 문자열 라벨 → 정수 (LabelEncoder)

**사용처:** `sparse_categorical_crossentropy`를 쓰거나 혼동행렬을 그릴 때
**목적:** 'Iris-setosa' → 0 같은 정수 변환
**검색 키워드:** LabelEncoder, 문자열 라벨 정수, fit_transform, classes_, inverse_transform, 라벨 인코딩

```python
le = LabelEncoder()
y_int = le.fit_transform(df_iris['species'])    # 0, 1, 2 (알파벳순으로 번호 부여)
print(le.classes_)                              # 번호 순서대로 클래스 이름
# 예측 번호 → 이름: le.inverse_transform(pred_int)
```

[⬆ 목차로 돌아가기](#목차)

### 5-4. 입력 특성의 범주형 원핫 (pd.get_dummies)

**사용처:** 빠르게 끝내야 할 때. 분할 **전**에 전체 데이터에 적용
**목적:** 문자열 컬럼을 0/1 컬럼으로 펼침
**검색 키워드:** get_dummies, 범주형 입력, 문자열 열, 원핫, drop_first, columns=, 더미 변수

```python
X = pd.get_dummies(X, dtype=int)                           # 문자열(object) 열 전부 자동 원핫
# 지정한 열만:      X = pd.get_dummies(X, columns=['MSZoning', 'Street'], dtype=int)
# 첫 범주 열 제거:  X = pd.get_dummies(X, drop_first=True, dtype=int)   # k개 범주 → k-1개 열
print(X.shape)                                             # 열 개수가 늘어난 것 확인
```

> 숫자로 된 범주 열(예: `MSSubClass`)은 자동으로 원핫되지 않습니다. 먼저 `X['MSSubClass'] = X['MSSubClass'].astype(str)`
> 결과에 bool 열이 섞이면 Keras 입력에서 오류가 날 수 있으니 `dtype=int`를 쓰고, 넣을 때 `X.values.astype('float32')`.

[⬆ 목차로 돌아가기](#목차)

### 5-5. 입력 특성의 범주형 원핫 (OneHotEncoder, 11강 정석)

**사용처:** 분할 **후** train 기준으로 인코딩 (누수 방지)
**목적:** val/test에 처음 보는 범주가 나와도 오류 없이 처리
**검색 키워드:** OneHotEncoder, handle_unknown, sparse_output, 범주형 입력, 분할 후 인코딩, 처음 보는 범주

```python
cat_cols = X_train.select_dtypes(include='object').columns   # 문자열 열 이름
encoder = OneHotEncoder(
    handle_unknown='ignore',   # val/test에만 있는 새 범주 → 오류 대신 전부 0으로
    sparse_output=False,       # 결과를 일반 numpy 배열로 (True면 희소행렬)
)
# sklearn 1.2 미만이면 sparse=False 로 바꾸세요
X_train_cat = encoder.fit_transform(X_train[cat_cols])   # train으로 fit (범주 목록 학습)
X_val_cat   = encoder.transform(X_val[cat_cols])         # val/test는 transform만
X_test_cat  = encoder.transform(X_test[cat_cols])
print(X_train_cat.shape)                                 # (train 행 수, 원핫 열 수)
# 만들어진 열 이름: encoder.get_feature_names_out(cat_cols)
```

[⬆ 목차로 돌아가기](#목차)

---

## 6. 데이터 분할 (train / val / test)

### 6-1. train : test = 8 : 2

**사용처:** "학습/테스트 데이터를 8:2로 나누시오"
**목적:** 학습에 쓰지 않은 데이터로 일반화 성능 평가
**검색 키워드:** 분할, train_test_split, 8:2, test_size, stratify, random_state, shuffle, 학습 테스트 나누기

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # test 비율 (정수를 주면 개수, 예: test_size=10000)
    shuffle=True,         # 기본값 True (섞은 뒤 분할)
    random_state=42,      # 재현성 (같은 숫자면 매번 같은 분할)
    stratify=y            # 분류 문제: 클래스 비율 유지 (회귀에서는 빼기)
)
print(X_train.shape, X_test.shape)
```

> **반환 순서:** `X_train, X_test, y_train, y_test` (X 두 개 먼저, y 두 개 나중)

[⬆ 목차로 돌아가기](#목차)

### 6-2. train : val : test = 6 : 2 : 2 (두 번 분할)

**사용처:** "검증 데이터를 따로 두시오", 11강 Ames
**목적:** val로 모델/하이퍼파라미터 선택, test는 마지막 평가에 한 번만 사용
**검색 키워드:** 검증 분리, validation, 6:2:2, 두 번 분할, train val test, 64:16:20, 70:15:15

```python
# 1차: 전체의 20%를 test로
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)
# 2차: 남은 80% 중 25% = 전체의 20%를 val로
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25,          # 0.2 / 0.8
    random_state=42,
)

print(len(X_train), len(X_val), len(X_test))
# 실제 비율 확인 (:.0% → 백분율, 소수점 없이)
print(f'{len(X_train)/len(X):.0%} : {len(X_val)/len(X):.0%} : {len(X_test)/len(X):.0%}')
```

**두 번째 test_size 계산법:** `val비율 / (1 - test비율)`

| 원하는 비율 | 1차 test_size | 2차 test_size |
|---|---|---|
| 60 : 20 : 20 | 0.2 | 0.2 / 0.8 = **0.25** |
| 64 : 16 : 20 (11강 기구학) | 0.2 | **0.2** (남은 80%의 20% = 16%) |
| 70 : 15 : 15 | 0.15 | 0.15 / 0.85 ≈ **0.1765** |
| 80 : 10 : 10 | 0.1 | 0.1 / 0.9 ≈ **0.1111** |

분류 문제라면 두 번 모두 `stratify=y`, `stratify=y_train_val`을 넣습니다.

[⬆ 목차로 돌아가기](#목차)

### 6-3. validation_split vs validation_data

**사용처:** fit에서 검증 데이터를 지정할 때
**검색 키워드:** validation_split, validation_data, 검증 데이터 지정, fit 검증

```python
# (A) train에서 자동으로 떼어냄: 6강 Wine, 14강 CNN 방식
#     X_train의 '마지막' 25%를 검증용으로 사용 (섞지 않음)
history = model.fit(
    X_train, y_train,
    validation_split=0.25,
    epochs=50,
    batch_size=32,
)

# (B) 미리 나눈 val을 직접 줌: 11강 방식
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),   # 튜플 (X, y)로
    epochs=50,
    batch_size=32,
)
```

> **주의:** `validation_split`은 섞지 않고 **데이터의 마지막 25%**를 떼어냅니다. 데이터가 라벨 순으로 정렬돼 있으면 검증 셋이 한쪽 클래스만 갖게 되므로, `train_test_split`(기본 shuffle=True)을 먼저 거치거나 (B)를 쓰세요.
> 8:2 분할 뒤 `validation_split=0.25`이면 최종 비율은 **6:2:2**입니다.

[⬆ 목차로 돌아가기](#목차)

### 6-4. 시계열 분할 (섞으면 안 됨, 11강 ETTh1)

**사용처:** 시간 순서가 있는 데이터
**목적:** 미래 정보가 학습에 섞이는 것(누수)을 방지
**검색 키워드:** 시계열 분할, 시간 순서, shuffle=False, 70:15:15, ETTh1, 앞에서부터 자르기

```python
n = len(df)
train_end = int(n * 0.70)          # 앞 70% 끝 위치
val_end = int(n * 0.85)            # 그다음 15% 끝 위치
train_df = df.iloc[:train_end]     # 과거
val_df   = df.iloc[train_end:val_end]
test_df  = df.iloc[val_end:]       # 가장 최근 (미래)

# 또는 train_test_split에서 shuffle=False (앞 80% train, 뒤 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
```

> 시계열은 **원본을 먼저 시간순으로 분할한 뒤** 구간마다 윈도우(6-5)를 만듭니다. 윈도우를 먼저 만들고 섞어서 나누면 같은 시점이 train/test에 겹쳐 성능이 부풀려집니다.

[⬆ 목차로 돌아가기](#목차)

### 6-5. 슬라이딩 윈도우 만들기 (과거 N개 → 미래 M개)

**사용처:** 시계열 예측 (ETTh1: 과거 96시간 → 미래 24시간)
**검색 키워드:** 윈도우, window, sliding, 슬라이딩, make_windows, 과거 N개 미래 M개, 시계열 입력 만들기, newaxis

```python
INPUT_LEN, PRED_LEN = 96, 24                       # 과거 96개 보고 → 다음 24개 예측
series = df['OT'].to_numpy(dtype=np.float32)       # 1차원 배열 (시간 순서)

def make_windows(series, input_len, pred_len):
    X, Y = [], []
    # 만들 수 있는 윈도우 개수 = 전체 길이 - 입력 길이 - 예측 길이 + 1
    for i in range(len(series) - input_len - pred_len + 1):
        X.append(series[i : i + input_len])                          # 입력: i ~ i+95
        Y.append(series[i + input_len : i + input_len + pred_len])   # 정답: 바로 뒤 24개
    return np.array(X), np.array(Y)

X_win, Y_win = make_windows(series, INPUT_LEN, PRED_LEN)
print(X_win.shape, Y_win.shape)     # (N, 96), (N, 24)
X_win = X_win[..., np.newaxis]      # (N, 96, 1)이 필요하면 (Conv1D/LSTM 입력). MLP는 (N, 96) 그대로
```

[⬆ 목차로 돌아가기](#목차)

---

## 7. 스케일링 (표준화, 정규화)

### 7-1. 입력 표준화 (StandardScaler)

**사용처:** 특성마다 값의 범위가 다를 때 (거의 항상)
**목적:** 평균 0, 표준편차 1로 맞춰 학습 안정화. **fit은 train만**
**검색 키워드:** 표준화, StandardScaler, fit_transform, transform, 스케일링, scaling, 정규화, 평균 0 표준편차 1, 누수

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # train: 평균/표준편차 계산(fit) + 변환(transform)
X_val   = scaler.transform(X_val)         # val, test: train 통계로 변환만
X_test  = scaler.transform(X_test)
print(scaler.mean_, scaler.scale_)        # train 평균, 표준편차 (열마다 하나씩)
# 공식: z = (x - mean) / std
```

> **절대 금지:** `scaler.fit_transform(X_test)` → test 정보가 새는 데이터 누수
> 결과는 numpy 배열이 됩니다 (DataFrame 열 이름이 사라짐).

[⬆ 목차로 돌아가기](#목차)

### 7-2. 0~1 정규화 (MinMaxScaler)

**검색 키워드:** MinMaxScaler, 0~1 정규화, min-max, 최소 최대 스케일링

```python
mm = MinMaxScaler()                     # 공식: (x - min) / (max - min)  → train 기준 0~1
X_train = mm.fit_transform(X_train)     # train의 min, max로 fit
X_test  = mm.transform(X_test)          # test는 0~1을 살짝 벗어날 수 있음 (정상)
```

[⬆ 목차로 돌아가기](#목차)

### 7-3. 목표값(y) 표준화와 원래 단위 복원 (회귀)

**사용처:** 집값처럼 y 값이 큰 회귀 (11강 Ames, 기구학)
**목적:** 학습 안정화. 평가는 원래 단위로 되돌려서
**검색 키워드:** y 표준화, 타깃 스케일링, target scaling, inverse_transform, 원래 단위 복원, reshape(-1, 1), y_scaler

```python
y_scaler = StandardScaler()
# 스케일러는 2차원 입력만 받음 → y가 1차원 (N,)이면 reshape(-1, 1)로 (N, 1) 만든 뒤
# .ravel()로 다시 (N,)로 펴서 저장
y_train_s = y_scaler.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()
y_val_s   = y_scaler.transform(np.asarray(y_val).reshape(-1, 1)).ravel()
y_test_s  = y_scaler.transform(np.asarray(y_test).reshape(-1, 1)).ravel()

# 학습 후 예측값 → 원래 단위
y_pred_s = model.predict(X_test, verbose=0)               # (N, 1) 표준화된 값
y_pred = y_scaler.inverse_transform(y_pred_s).ravel()     # (N,) 원래 단위 (예: 달러)
print('MAE (원래 단위):', mean_absolute_error(y_test, y_pred))

# 표준화 공간의 MAE를 원래 단위로: 표준편차만 곱하면 됨 (평균 이동은 오차에서 상쇄)
# mae_original = mae_scaled * y_scaler.scale_[0]
```

**y가 여러 열인 경우 (기구학, 2D 출력):** `reshape` 없이 바로 넣습니다.

```python
scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train)      # (N, k) → 열마다 따로 표준화
y_val_s   = scaler_y.transform(y_val)
y_pred = scaler_y.inverse_transform(model.predict(X_test_s, verbose=0))   # (N, k) 원래 단위
```

[⬆ 목차로 돌아가기](#목차)

### 7-4. 로그 변환 목표값 (11강)

**사용처:** y 분포가 오른쪽으로 긴 꼬리를 가질 때 (가격 등)
**검색 키워드:** 로그 변환, log1p, expm1, 왜도, skew, 오른쪽 꼬리, 가격 로그

```python
print(pd.Series(y_train).skew())                       # 왜도 > 1 정도면 꼬리가 긴 분포
y_train_log = np.log1p(y_train)                        # log(1 + y)  (y=0이어도 안전)
y_val_log   = np.log1p(y_val)
# 학습 후: 예측도 로그 단위로 나오므로 반드시 되돌린 뒤 평가
y_pred = np.expm1(model.predict(X_test, verbose=0).ravel())   # exp(x) - 1 로 복원
```

[⬆ 목차로 돌아가기](#목차)

### 7-5. 이미지 정규화

**검색 키워드:** 이미지 정규화, /255, 0~1, reshape, 채널 축, (N, 28, 28, 1), float32, flatten 입력

```python
X_train = X_train.astype('float32') / 255.0       # 0~255 → 0~1 (정수로 나누면 안 되니 float 변환 먼저)
X_test  = X_test.astype('float32') / 255.0
X_train_fc = X_train.reshape(-1, 28 * 28)         # MLP용: (N, 784)  (-1 = 개수는 자동 계산)
X_train_cnn = X_train.reshape(-1, 28, 28, 1)      # CNN용: (N, 28, 28, 1) 채널 축 추가
# CIFAR-10은 이미 (N, 32, 32, 3)이라 reshape 없이 /255만
print(X_train.min(), X_train.max())               # 0.0 1.0 확인
```

[⬆ 목차로 돌아가기](#목차)

### 7-6. 수치형(표준화) + 범주형(원핫) 합치기 (11강 Ames)

**검색 키워드:** 수치형 범주형 합치기, concatenate, select_dtypes, input_dim, Ames 전처리, 열 합치기

```python
num_cols = X_train.select_dtypes(include='number').columns   # 수치형 열
cat_cols = X_train.select_dtypes(include='object').columns   # 문자열 열

# 수치형: train 기준 표준화
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_val_num   = scaler.transform(X_val[num_cols])
X_test_num  = scaler.transform(X_test[num_cols])

# 범주형: train 기준 원핫
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = encoder.fit_transform(X_train[cat_cols])
X_val_cat   = encoder.transform(X_val[cat_cols])
X_test_cat  = encoder.transform(X_test[cat_cols])

# 옆으로(axis=1) 붙여서 신경망 입력 하나로 (행 개수는 같아야 함)
X_train_input = np.concatenate([X_train_num, X_train_cat], axis=1)
X_val_input   = np.concatenate([X_val_num, X_val_cat], axis=1)
X_test_input  = np.concatenate([X_test_num, X_test_cat], axis=1)
input_dim = X_train_input.shape[1]        # Input(shape=(input_dim,))에 넣을 값
print(input_dim)
```

[⬆ 목차로 돌아가기](#목차)

### 7-7. 특성 선택 (Top-K, 11강)

**검색 키워드:** 특성 선택, feature selection, SelectKBest, f_regression, f_classif, Top-K, 상위 k개

```python
from sklearn.feature_selection import SelectKBest, f_regression   # 분류는 f_classif

# 각 특성과 y의 선형 관계 점수(F값)를 계산해 상위 k개 열만 남김
selector = SelectKBest(score_func=f_regression, k=100)
X_train_k = selector.fit_transform(X_train_input, y_train_s)      # train으로만 fit
X_val_k   = selector.transform(X_val_input)                       # 같은 열만 골라냄
X_test_k  = selector.transform(X_test_input)
# 선택된 열 번호: selector.get_support(indices=True)
```

[⬆ 목차로 돌아가기](#목차)

---
## 8. 모델 구성 (출력층·손실 짝 포함)

### 8-1. Sequential 작성 방식 두 가지

**검색 키워드:** Sequential, add, 리스트 방식, 모델 만들기, Input, Dense, summary, count_params, 파라미터 수, model build

```python
# 입력 특성 수(열 개수)를 데이터에서 직접 읽어옴
# → 숫자(8, 13 등)를 하드코딩하면 데이터가 바뀔 때 shape 오류가 남
input_dim = X_train.shape[1]     # X_train: (샘플 수, 특성 수) → [1]이 특성 수

# (A) add 방식 (6강): 빈 모델을 만들고 층을 한 줄씩 추가
model = Sequential()
model.add(Input(shape=(input_dim,)))                       # 입력 모양 (튜플이라 쉼표 필수: (8,))
model.add(Dense(12, activation='relu', name='Dense_1'))    # 은닉층 1: 뉴런 12개, name은 summary 표시용
model.add(Dense(8, activation='relu', name='Dense_2'))     # 은닉층 2: 뉴런 8개
model.add(Dense(1, activation='sigmoid', name='Dense_3'))  # 출력층: 이진 분류 → 확률 1개 (0~1)

# (B) 리스트 방식 (11강, 14강): 층 목록을 한 번에 넘김 (결과는 (A)와 같음)
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.summary()              # 층별 출력 shape, 파라미터 수 표 출력
print(model.count_params())  # 전체 파라미터 수 (숫자 하나)
```

**파라미터 수 공식:** Dense = (입력 + 1) × 출력. 6강 Pima: (8+1)×12 + (12+1)×8 + (8+1)×1 = 108 + 104 + 9 = **221**

[⬆ 목차로 돌아가기](#목차)

### 8-2. 이진 분류 (6강 Pima, Wine)

**검색 키워드:** 이진 분류, binary classification, sigmoid, binary_crossentropy, Pima, Wine, 당뇨, 레드 화이트 와인

```python
model = Sequential([
    Input(shape=(X_train.shape[1],)),   # 특성 수 (Pima 8, Wine 12)
    Dense(30, activation='relu'),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid'),     # 출력 1개 + sigmoid = "1일 확률" (0~1)
])

# 이진 분류 짝: sigmoid 1개 ↔ binary_crossentropy, 라벨 y는 0/1 정수 (N,)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],               # history에 'accuracy', 'val_accuracy'로 기록됨
)
```

**softmax로 이진 분류 (4강):** 라벨을 `to_categorical(y, 2)`로 바꾸고 `Dense(2, activation='softmax')` + `categorical_crossentropy`

[⬆ 목차로 돌아가기](#목차)

### 8-3. 다중 분류, 다중 라벨 (6강 Iris)

**검색 키워드:** 다중 분류, multiclass, softmax, categorical_crossentropy, sparse_categorical_crossentropy, 다중 라벨, multi-label, Iris, 붓꽃

```python
# ── (1) 원핫 라벨 (get_dummies, to_categorical) → categorical_crossentropy ──
model = Sequential([
    Input(shape=(4,)),                  # Iris 특성 4개
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax'),     # 출력 수 = 클래스 수, softmax → 3개 확률 합이 1
])
model.compile(
    loss='categorical_crossentropy',    # y가 (N, 3) 원핫일 때
    optimizer='adam',
    metrics=['accuracy'],
)

# ── (2) 정수 라벨 (LabelEncoder, MNIST 원본 라벨 0~9) → sparse_ ──
#      모델은 그대로 두고 loss만 바꿈. y가 (N,) 정수일 때
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)
# 실수 주의: 정수 라벨 + categorical_crossentropy → shape 오류
#            원핫 라벨 + sparse_categorical_crossentropy → shape 오류

# ── (3) 다중 라벨 (한 샘플이 여러 클래스에 동시에 속함) ──
#      출력 sigmoid(클래스마다 독립 확률) + binary_crossentropy, y는 (N, C) multi-hot
model_ml = Sequential([
    Input(shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(5, activation='sigmoid'),     # 클래스 5개, 각각 0~1 (합이 1일 필요 없음)
])
model_ml.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 예측: (model_ml.predict(X) >= 0.5).astype(int) → 클래스마다 0/1
```

[⬆ 목차로 돌아가기](#목차)

### 8-4. 회귀 (출력 1개, k개)

**검색 키워드:** 회귀, regression, 출력 1개, 다중 출력, multi-output, 2D 출력, 6D 출력, 기구학, Ames, linear, mse, mae, he_normal

```python
# ── 출력 1개 (11강 Ames 기본 모델) ──
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation='relu', kernel_initializer='he_normal'),   # He 초기화: ReLU에 권장
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dense(1),                            # 회귀: 활성화 없음 (= linear, 값 범위 제한 없음)
])
model.compile(
    loss='mse',                          # 회귀 손실: 평균제곱오차
    optimizer=Adam(learning_rate=2e-4),  # 11강 학습률 2e-4
    metrics=['mae'],                     # 해석하기 쉬운 평균절대오차를 같이 기록
)

# ── 출력 k개 (2D 출력 회귀, 기구학 6D 출력) ──
n_out = y_train.shape[1]                 # y가 (N, k) 2차원이어야 함 → 열 수 k
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(n_out),                        # 출력 뉴런 수 = y 열 수, 활성화 없음
])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])   # mse는 k개 출력의 평균
```

[⬆ 목차로 돌아가기](#목차)

### 8-5. 층 옵션 모음 (초기화, BN, Dropout, L2, 활성화)

**검색 키워드:** kernel_initializer, he_normal, glorot, Xavier, BatchNormalization, BN, LayerNormalization, Dropout, l2, kernel_regularizer, LeakyReLU, ELU, swish, tanh, use_bias

```python
# 아래는 "층 하나"의 문법 모음입니다 (Sequential([...]) 리스트 안에 넣어 사용)

# He 초기화 (ReLU 계열에 권장)
# Xavier = 'glorot_uniform' (Dense 기본값), LeCun = 'lecun_normal'
Dense(64, activation='relu', kernel_initializer='he_normal')

# 배치 정규화 순서: Dense(편향 없음) → BN → 활성화 (11강, 9강)
#   use_bias=False: BN이 평균을 빼고 beta를 더하므로 Dense의 bias는 의미가 없음
Dense(64, use_bias=False, kernel_initializer='he_normal')
BatchNormalization()
Activation('relu')

# 층 정규화 (9강): 배치가 아니라 '샘플 하나' 안에서 정규화 (배치 크기 영향 없음)
from tensorflow.keras.layers import LayerNormalization
LayerNormalization()

# 드롭아웃: 학습 때만 뉴런 30%를 무작위로 끔. 은닉층 뒤에 (출력층 뒤에는 넣지 않음)
Dropout(0.3)

# L2 정규화 (10강): 손실에 0.001 × Σw² 를 더해 큰 가중치에 벌점
Dense(64, activation='relu', kernel_regularizer=l2(0.001))

# 활성화 함수를 '층'으로 쓰기 (9강): Dense 뒤에 따로 붙임
from tensorflow.keras.layers import ELU
Dense(32); ReLU()                           # 리스트 안에서는 Dense(32), ReLU() 두 원소
Dense(16); LeakyReLU(negative_slope=0.2)    # 음수 구간 기울기 0.2 (옛 버전 인자명: alpha)
Dense(12); ELU(alpha=1.0)
Dense(8, activation='swish')                # 문자열로 지정 가능: 'relu', 'tanh', 'swish', 'gelu' ...
Dense(8, activation='tanh')
```

**BN + Dropout + He를 모두 넣은 MLP 예시:**

```python
model = Sequential([
    Input(shape=(input_dim,)),
    # 블록 1: Dense(편향 X) → BN → ReLU → Dropout
    Dense(64, use_bias=False, kernel_initializer='he_normal'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),
    # 블록 2
    Dense(32, use_bias=False, kernel_initializer='he_normal'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),
    Dense(1),                            # 회귀 출력 (분류면 sigmoid/softmax로 바꿈)
])
```

[⬆ 목차로 돌아가기](#목차)

### 8-6. CNN

**사용처:** 이미지 분류 (14강 MNIST, Fashion-MNIST, CIFAR-10)
**목적:** 합성곱으로 공간 특징 추출 → Flatten → Dense 분류
**검색 키워드:** CNN, Conv2D, 합성곱, MaxPooling2D, 풀링, Flatten, padding, same, valid, 이미지 분류, MNIST, Fashion-MNIST, CIFAR-10, 출력 크기 공식

```python
# 입력: (N, 28, 28, 1) = (샘플, 높이, 너비, 채널), 값 0~1, 라벨 원핫 (to_categorical)
model = Sequential([
    Input(shape=(28, 28, 1)),            # 흑백 채널 1 (컬러면 3)
    Conv2D(32, kernel_size=(3, 3)),      # 필터 32개, 3×3. padding 기본 'valid': 28 → 26
    BatchNormalization(),
    ReLU(),
    Conv2D(64, kernel_size=(3, 3)),      # 26 → 24, 채널 64
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2)),      # 2×2 영역 최댓값: 24 → 12 (파라미터 0)
    Flatten(),                           # (12, 12, 64) → 12*12*64 = 9216 길이 벡터
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax'),     # 클래스 10개
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',     # 라벨이 정수면 'sparse_categorical_crossentropy'
    metrics=['accuracy'],
)
model.summary()
```

**padding='same' 3블록 버전 (14강 CNN2):**

```python
model = Sequential([
    Input(shape=(28, 28, 1)),
    # padding='same': 가장자리를 0으로 채워 크기 유지 (28 → 28)
    # use_bias=False: 뒤에 BN이 오므로 bias 불필요
    Conv2D(32, (3, 3), padding='same', use_bias=False), BatchNormalization(), ReLU(),
    Conv2D(64, (3, 3), padding='same', use_bias=False), BatchNormalization(), ReLU(),
    MaxPooling2D((2, 2)),                # 28 → 14
    Conv2D(128, (3, 3), padding='same', use_bias=False), BatchNormalization(), ReLU(),
    MaxPooling2D((2, 2)),                # 14 → 7
    Flatten(),                           # 7*7*128 = 6272
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax'),
])
```

**CIFAR-10 (컬러, 14강 TensorBoard 노트북):** 입력만 `Input(shape=(32, 32, 3))`로 바꾸고 블록마다 `Dropout(0.25)`

**간단 CNN (10강 증강 예제):**

```python
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),   # 활성화를 Conv2D 안에 바로 지정 (BN 없을 때)
    MaxPooling2D((2, 2)),                    # 26 → 13
    Flatten(),                               # 13*13*32 = 5408
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # 정수 라벨 (to_categorical 안 했을 때)
    metrics=['accuracy'],
)
```

**CNN 크기 공식**

| 항목 | 공식 |
|---|---|
| 출력 크기 | (N + 2P − F) / S + 1 |
| `padding='valid'` | P = 0 → 3×3이면 크기 2 감소 |
| `padding='same'` | 출력 크기 = 입력 크기 (stride 1) |
| Conv2D 파라미터 | (F × F × C_in + 1) × C_out (`use_bias=False`면 +1 없음) |
| BN 파라미터 | 4 × C (학습 2C: γ, β / 비학습 2C: 이동평균, 이동분산) |
| MaxPooling, Flatten | 파라미터 0 |

[⬆ 목차로 돌아가기](#목차)

### 8-7. 시계열 MLP (11강 ETTh1)

**검색 키워드:** 시계열, time series, ETTh1, MLP, 윈도우, 과거 96 미래 24, multi-step, Direct, Flatten

```python
model = Sequential([
    Input(shape=(96, 1)),        # 과거 96개 값 (시점 96, 특성 1) → X shape (N, 96, 1)
    Flatten(),                   # (96, 1) → 96 길이 벡터 (MLP는 1차원 입력)
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dense(24),                   # 미래 24개를 한 번에 예측 (Direct 방식), 활성화 없음
])
model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
# X가 (N, 96) 2차원이면 Input(shape=(96,))로 쓰고 Flatten은 빼도 됨
```

[⬆ 목차로 돌아가기](#목차)

### 8-8. 모델을 함수로 만들기 (여러 설정 비교할 때)

**사용처:** "Dropout 0.1, 0.2, 0.3을 비교하시오" 같은 문제
**목적:** 매번 새 모델을 만들어 공정하게 비교
**검색 키워드:** build_mlp, 모델 함수, 하이퍼파라미터 비교, dropout 비교, 반복 실험, results 표, set_random_seed

```python
def build_mlp(input_dim, hidden=(64, 32), dropout=0.0, use_bn=False, wd=0.0, lr=2e-4):
    """회귀용 MLP를 새로 만들어 compile까지 해서 돌려줌.
    hidden : 은닉층 뉴런 수 튜플 (64, 32) → 은닉층 2개
    dropout: 0이면 Dropout 층 없음
    use_bn : True면 Dense(편향X) → BN → ReLU 순서
    wd     : 0보다 크면 AdamW(weight decay), 아니면 Adam
    """
    layers = [Input(shape=(input_dim,))]                # 층을 리스트에 모았다가 한 번에 Sequential
    for units in hidden:
        if use_bn:
            layers += [
                Dense(units, use_bias=False, kernel_initializer='he_normal'),
                BatchNormalization(),
                Activation('relu'),
            ]
        else:
            layers.append(Dense(units, activation='relu', kernel_initializer='he_normal'))
        if dropout > 0:
            layers.append(Dropout(dropout))             # 각 은닉층 뒤에 Dropout
    layers.append(Dense(1))                             # 회귀 출력 1개
    model = Sequential(layers)

    # weight decay 여부에 따라 옵티마이저 선택
    opt = AdamW(learning_rate=lr, weight_decay=wd) if wd > 0 else Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    return model

results = []
for rate in [0.1, 0.2, 0.3]:
    tf.keras.utils.set_random_seed(42)       # 공정 비교: 매번 같은 시드 (초기 가중치 동일)
    m = build_mlp(X_train.shape[1], dropout=rate)   # 반드시 매번 '새 모델' (재사용하면 이어서 학습됨)
    h = m.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_mae', patience=20, restore_best_weights=True)],
        verbose=0,
    )
    results.append({
        'dropout': rate,
        'best_val_mae': min(h.history['val_mae']),                  # 가장 낮은 검증 MAE
        'best_epoch': int(np.argmin(h.history['val_mae'])) + 1,     # 그때 epoch (1부터 셈)
    })

print(pd.DataFrame(results).sort_values('best_val_mae'))   # 검증 MAE 낮은 순 = 좋은 순
```

[⬆ 목차로 돌아가기](#목차)

---

## 9. 컴파일: 손실, 옵티마이저, 사용자 정의 손실

### 9-1. compile 기본형

**검색 키워드:** compile, loss, optimizer, metrics, 손실 함수 지정, 옵티마이저 지정

```python
model.compile(
    loss='binary_crossentropy',  # 5-1 표 참고: mse, mae, categorical_crossentropy ...
    optimizer='adam',            # 문자열이면 기본 학습률(0.001). 바꾸려면 Adam(learning_rate=1e-3) 객체
    metrics=['accuracy'],        # 회귀는 ['mae']. 기록만 하고 학습(가중치 갱신)에는 영향 없음
)
```

[⬆ 목차로 돌아가기](#목차)

### 9-2. 옵티마이저

**검색 키워드:** optimizer, SGD, momentum, 모멘텀, nesterov, Adagrad, RMSprop, Adam, AdamW, weight decay, 학습률, learning_rate, CosineDecay, ExponentialDecay, 학습률 스케줄

```python
# 각 줄은 옵티마이저 '객체'를 만드는 문법 → compile(optimizer=여기) 에 넣어 사용
SGD(learning_rate=0.01)                                  # 기본 경사하강: w ← w − lr·g
SGD(learning_rate=0.01, momentum=0.9)                    # 모멘텀: 이전 이동 방향을 0.9만큼 유지
SGD(learning_rate=0.01, momentum=0.9, nesterov=True)     # 네스테로프 모멘텀
SGD(learning_rate=0.01, weight_decay=1e-4)               # SGD + weight decay

from tensorflow.keras.optimizers import Adagrad          # 기본 import 세트에 없으면 따로 import
Adagrad(learning_rate=0.01)                              # 기울기 제곱 누적으로 학습률 자동 감소
RMSprop(learning_rate=0.001)                             # 기울기 제곱의 지수이동평균
Adam(learning_rate=0.001)                                # 모멘텀 + RMSprop, 기본값 lr=0.001
AdamW(learning_rate=1e-3, weight_decay=1e-4)             # Adam + 분리된 weight decay

# 사용 예
model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4), loss='mse')
```

**Weight decay 방법 정리 (기출)**

| 방법 | 코드 | 메모 |
|---|---|---|
| AdamW | `optimizer=AdamW(learning_rate=2e-4, weight_decay=1e-1)` | 11강 Ames 방식, 손실과 분리된 감쇠 |
| L2 정규화 | `Dense(64, kernel_regularizer=l2(0.001))` | 손실에 λ‖W‖² 추가, 10강 |
| SGD 감쇠 | `SGD(learning_rate=0.01, weight_decay=1e-4)` | SGD에서는 L2와 사실상 같음 |

**학습률 스케줄 (8강):**

```python
from tensorflow.keras.optimizers.schedules import CosineDecay, ExponentialDecay

# CosineDecay: lr_t = lr0 × 0.5 × (1 + cos(π·t/T))  (alpha=0이면 마지막에 0까지 감소)
#   decay_steps는 'epoch'가 아니라 '배치(step)' 수 → steps_per_epoch × epochs 로 계산
lr_schedule = CosineDecay(initial_learning_rate=1e-3, decay_steps=1000, alpha=0.0)

# ExponentialDecay: 1000 step마다 학습률 × 0.9
# lr_schedule = ExponentialDecay(1e-3, decay_steps=1000, decay_rate=0.9)

# 스케줄 객체를 learning_rate 자리에 그대로 넣음 (AdamW도 가능)
model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mse')
```

[⬆ 목차로 돌아가기](#목차)

### 9-3. 사용자 정의 손실 (Custom Loss)

**사용처:** "MSE를 직접 정의해서 학습하시오", "두 손실을 가중합하시오"
**목적:** 함수(`y_true`, `y_pred`) → 스칼라 손실을 만들어 `compile(loss=함수이름)`
**검색 키워드:** 사용자 정의 손실, custom loss, def loss, y_true y_pred, tf.reduce_mean, tf.square, tf.abs, tf.where, Huber, RMSE, BCE 직접, 가중 MSE, combined_loss, 클로저, closure, custom_objects

```python
import tensorflow as tf

# ─── 규칙 4가지 ───
# 1) 인자 순서는 반드시 (y_true, y_pred)
# 2) 안에서는 tf. 연산만 (np. 쓰면 미분이 끊겨 오류)
# 3) 스칼라(숫자 하나) 또는 샘플별 값을 돌려줌 → Keras가 평균
# 4) compile(loss=함수이름)  ← 괄호 없이!

# (1) MSE 직접 정의 (3강 squared_2norm_loss)
def custom_mse(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)            # int 라벨일 때 자료형 오류 방지 (int ↔ float)
    return tf.reduce_mean(tf.square(y_true - y_pred)) # 평균((y − ŷ)²)

# (2) MAE: 평균(|y − ŷ|)
def custom_mae(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# (3) RMSE: √MSE (원래 단위로 해석 가능)
def custom_rmse(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# (4) MSE + MAE 가중합 (가중치 0.5 : 0.5)
def mse_mae_loss(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.5 * mse + 0.5 * mae

# (5) Huber 손실: |오차| ≤ delta면 0.5·오차² (제곱), 크면 delta·(|오차| − 0.5·delta) (직선)
#     → 이상치(큰 오차)의 영향을 MSE보다 덜 받음
def custom_huber(y_true, y_pred, delta=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    err = y_true - y_pred
    small = tf.abs(err) <= delta                      # True/False 마스크
    return tf.reduce_mean(
        tf.where(
            small,                                    # 조건
            0.5 * tf.square(err),                     # True일 때 (작은 오차)
            delta * (tf.abs(err) - 0.5 * delta),      # False일 때 (큰 오차)
        )
    )

# (6) 출력별 가중치 MSE (2D 출력 회귀: 첫 번째 출력을 2배 중요하게)
def weighted_mse(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    w = tf.constant([2.0, 1.0])                       # 길이 = 출력 개수 (y가 (N, 2)일 때)
    return tf.reduce_mean(tf.square(y_true - y_pred) * w)   # (N, 2) × (2,) 브로드캐스팅

# (7) BCE 직접 정의 (이진 분류): −평균[y·log(p) + (1−y)·log(1−p)]
def custom_bce(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)   # log(0) = −inf 방지
    return -tf.reduce_mean(
        y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
    )

# (8) 내장 손실 조합 (7강 combined_loss): 손실 객체를 만들어 두고 가중합
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
mse_fn, bce_fn = MeanSquaredError(), BinaryCrossentropy()
def combined_loss(y_true, y_pred):
    return 0.5 * mse_fn(y_true, y_pred) + 0.5 * bce_fn(y_true, y_pred)

# (9) 하이퍼파라미터를 받는 손실 (클로저): 바깥 함수가 '손실 함수'를 만들어 돌려줌
def make_weighted_loss(alpha):
    def loss(y_true, y_pred):                         # 실제 Keras가 호출하는 함수
        y_true = tf.cast(y_true, y_pred.dtype)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        return alpha * mse + (1 - alpha) * mae        # alpha는 바깥 함수의 값을 기억
    return loss

# 사용법: 함수 이름을 그대로 (괄호 없이)
model.compile(optimizer='adam', loss=custom_mse, metrics=['mae'])
# 클로저만 괄호 사용 (바깥 함수를 실행해서 안쪽 함수를 받아옴):
# model.compile(optimizer='adam', loss=make_weighted_loss(0.7))

# 회귀에서 y가 (N,)이면 y_pred (N,1)과 브로드캐스팅되어 (N,N)이 됨 → 손실 이상
# → y_train = y_train.reshape(-1, 1) 로 맞춰 두면 안전
```

**사용자 정의 지표 (metric):**

```python
# 손실과 같은 형식의 함수를 metrics 리스트에 넣으면 '기록용' 지표가 됨 (학습에는 영향 없음)
def rmse_metric(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

model.compile(optimizer='adam', loss='mse', metrics=['mae', rmse_metric])
# history.history 키: 'rmse_metric', 'val_rmse_metric' (함수 이름이 그대로 키가 됨)
```

**사용자 정의 손실을 쓴 모델 불러오기:**

```python
# 저장된 파일에는 함수 '이름'만 있으므로 불러올 때 실제 함수를 알려줘야 함
model = load_model('model.keras', custom_objects={'custom_mse': custom_mse})
# 예측만 할 거면: load_model('model.keras', compile=False)  (손실 함수 필요 없음)
```

[⬆ 목차로 돌아가기](#목차)

### 9-4. Keras 내장 손실로 값 직접 계산 (7강)

**검색 키워드:** losses, MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy, 손실 값 계산, 손실 직접 확인

```python
from tensorflow.keras import losses

# 손실 객체(y_true, y_pred) 형태로 호출하면 손실 값(텐서)이 나옴 → float()로 숫자 변환
# MSE: ((1−1)² + (2−2.5)² + (3−2)²) / 3 = (0 + 0.25 + 1) / 3
print(float(losses.MeanSquaredError()([[1., 2., 3.]], [[1., 2.5, 2.]])))           # 0.4167

# BCE: 정답 1, 예측 0.8 → −log(0.8)
print(float(losses.BinaryCrossentropy()([[1.]], [[0.8]])))                         # 0.2231

# CCE: 정답 클래스(1번)의 예측 확률 0.8 → −log(0.8)
print(float(losses.CategoricalCrossentropy()([[0., 1., 0.]], [[0.1, 0.8, 0.1]])))  # 0.2231
```

[⬆ 목차로 돌아가기](#목차)

---

## 10. 콜백과 학습 (fit)

### 10-1. EarlyStopping (조기 종료)

**사용처:** epoch를 크게 주고 검증 성능이 멈추면 자동 종료
**목적:** 과적합 방지, 시간 절약
**검색 키워드:** EarlyStopping, 조기 종료, early stopping, patience, monitor, restore_best_weights, val_loss, callbacks

```python
early_stopping = EarlyStopping(
    monitor='val_loss',            # 감시 대상: 'val_loss', 'val_mae', 'val_accuracy'
    patience=20,                   # 개선이 없어도 기다리는 epoch 수
    mode='min',                    # loss, mae는 'min', accuracy는 'max' (기본 'auto')
    restore_best_weights=True,     # 종료 후 최고 시점의 가중치로 복원 (꼭 넣기)
                                   #  없으면 '마지막' epoch 가중치가 남음
    verbose=1,                     # 멈출 때 "Epoch 00xx: early stopping" 출력
)
# 사용: model.fit(..., callbacks=[early_stopping])  ← 리스트로 넣기
```

> `monitor` 이름은 `compile(metrics=...)`와 맞아야 합니다. `metrics=['mae']`이면 `'val_mae'`, `metrics=['accuracy']`이면 `'val_accuracy'`

[⬆ 목차로 돌아가기](#목차)

### 10-2. ModelCheckpoint (최고 모델 저장) + 불러오기

**검색 키워드:** ModelCheckpoint, 체크포인트, checkpoint, 모델 저장, save_best_only, load_model, best_model, 최고 모델 불러오기

```python
checkpoint = ModelCheckpoint(
    filepath='best_model.keras',   # 확장자 .keras 필수 (Drive에 저장하려면 Drive 경로)
    monitor='val_loss',            # 이 값이 좋아질 때만 저장
    save_best_only=True,           # 성능이 좋아질 때만 덮어쓰기 (False면 매 epoch 저장)
    verbose=1,                     # 저장할 때마다 메시지
)
# epoch 번호와 성능을 파일 이름에 넣기 (6강): '{epoch:02d}-{val_accuracy:.4f}.keras'
#   → 예: 49-0.9362.keras (이때 compile에 metrics=['accuracy'] 필요)

history = model.fit(
    X_train, y_train,
    validation_split=0.25,         # train의 마지막 25%를 검증으로
    epochs=2000,                   # EarlyStopping이 있으니 크게 줘도 됨
    batch_size=500,
    callbacks=[early_stopping, checkpoint],
    verbose=0,
)

best_model = load_model('best_model.keras')            # 검증 성능이 가장 좋았던 모델
print(best_model.evaluate(X_test, y_test, verbose=0))  # [loss, accuracy]
```

[⬆ 목차로 돌아가기](#목차)

### 10-3. ReduceLROnPlateau, TensorBoard

**검색 키워드:** ReduceLROnPlateau, 학습률 감소, factor, cooldown, min_lr, TensorBoard, 텐서보드, log_dir, histogram_freq

```python
# 검증 성능이 멈추면 학습률을 줄여서 더 세밀하게 학습
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # 11강은 'val_mae'
    factor=0.5,            # 새 학습률 = 학습률 × 0.5
    patience=3,            # 3 epoch 개선 없으면 감소 (11강은 15)
    min_lr=1e-5,           # 이 아래로는 안 내려감
    verbose=1,
)
# 11강 설정:
# ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=15,
#                   cooldown=5, min_lr=1e-5, mode='min')
#   cooldown=5: 학습률을 줄인 뒤 5 epoch 동안은 다시 줄이지 않음

from tensorflow.keras.callbacks import TensorBoard
import datetime
# 실행할 때마다 다른 폴더 (날짜-시간)에 로그 저장 → 실험끼리 비교 가능
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)   # 1 epoch마다 가중치 분포 기록
# 학습 후 셀에서:  %load_ext tensorboard   →   %tensorboard --logdir logs/fit

history = model.fit(
    X_train, y_train,
    validation_split=0.25,
    epochs=30,
    batch_size=128,
    callbacks=[early_stopping, checkpoint, reduce_lr],   # TensorBoard도 쓰면 tensorboard_cb 추가
    verbose=1,
)
```

[⬆ 목차로 돌아가기](#목차)

### 10-4. 과적합 억제 기법 모음

**사용처:** "과적합 억제 기법을 1개 이상 적용하시오" (기출)
**목적:** train 성능만 좋고 val 성능이 나빠지는 현상 완화. 아래 중 여러 개를 함께 써도 됩니다.
**검색 키워드:** 과적합, overfitting, 과적합 억제, 규제, regularization, Dropout, L2, weight decay, AdamW, early stopping, 모델 축소, 데이터 증강

| 기법 | 코드 위치 | 코드 |
|---|---|---|
| Dropout | 모델 은닉층 뒤 | `Dropout(0.3)` |
| L2 정규화 | Dense 인자 | `kernel_regularizer=l2(0.001)` |
| Weight decay | 옵티마이저 | `AdamW(learning_rate=1e-3, weight_decay=1e-4)` |
| Early stopping | fit 콜백 | `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)` |
| Batch normalization | Dense 뒤 | `BatchNormalization()` |
| 모델 축소 | 은닉 유닛 수 줄이기 | `Dense(32)` → `Dense(16)` |
| 데이터 증강 | 이미지 | `ImageDataGenerator(...)` (10-5) |
| 특성 선택 | 입력 | `SelectKBest` (7-7) |

**한 번에 적용한 예시 (복붙용):**

```python
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(
        64, activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-3),                 # (2) L2: 손실에 0.001 × Σw² 추가
    ),
    Dropout(0.3),                                    # (1) Dropout: 학습 때 30% 뉴런 끔
    Dense(
        32, activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-3),                 # (2) L2
    ),
    Dropout(0.3),
    Dense(1, activation='sigmoid'),                  # 이진 분류 출력
])
model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),   # (3) weight decay
    loss='binary_crossentropy',
    metrics=['accuracy'],
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,                       # (4) early stopping
)
history = model.fit(
    X_train, y_train,
    validation_split=0.25,
    epochs=500,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0,
)
# 실제로 학습한 epoch 수 = 기록 길이 (500보다 작으면 조기 종료된 것)
print('stopped at epoch', len(history.history['loss']))
```

[⬆ 목차로 돌아가기](#목차)

### 10-5. 데이터 증강 (10강)

**검색 키워드:** 데이터 증강, augmentation, ImageDataGenerator, rotation_range, width_shift_range, zoom_range, horizontal_flip, datagen.flow, 실시간 증강, RandomFlip, RandomRotation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 입력은 (N, 28, 28, 1) 4차원이어야 함 (채널 축 필수)
datagen = ImageDataGenerator(
    rotation_range=10,          # ±10도 회전
    width_shift_range=0.1,      # 좌우 ±10% 이동
    height_shift_range=0.1,     # 상하 ±10% 이동
    zoom_range=0.1,             # ±10% 확대/축소
    horizontal_flip=False,      # 숫자는 좌우 반전하면 의미가 바뀜 → False (옷은 True 가능)
)

# datagen.flow(...): 배치마다 '새로 변형한' 이미지를 만들어 줌 (실시간 증강)
# 검증/테스트 데이터는 증강하지 않음 → validation_data는 원본 그대로
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=3,
    validation_data=(X_test, y_test),
)

# 모델 안에 넣는 증강 층 방식 (학습 때만 동작, predict/evaluate 때는 통과):
# from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
# Sequential([Input((28, 28, 1)), RandomFlip('horizontal'), RandomRotation(0.05),
#             RandomZoom(0.1), Conv2D(...), ...])
```

[⬆ 목차로 돌아가기](#목차)

### 10-6. fit 기본형과 옵션

**검색 키워드:** fit, epochs, batch_size, validation_data, validation_split, shuffle, verbose, history, best epoch

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),   # 미리 나눈 검증 데이터 (또는 validation_split=0.25)
    epochs=100,                       # 전체 데이터를 100번 반복
    batch_size=32,                    # epoch당 업데이트 수 = ceil(N / batch_size)
    shuffle=True,                     # epoch마다 순서 섞기 (기본 True)
    callbacks=[early_stopping],
    verbose=1,                        # 0: 출력 없음, 1: 진행바, 2: epoch마다 한 줄
)

# history.history: epoch별 기록이 담긴 딕셔너리 (리스트 길이 = 학습한 epoch 수)
print(history.history.keys())         # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

best_epoch = int(np.argmin(history.history['val_loss'])) + 1   # argmin은 0부터 → +1
print('best epoch:', best_epoch, 'best val_loss:', min(history.history['val_loss']))
```

[⬆ 목차로 돌아가기](#목차)

### 10-7. 모델, 스케일러 저장/불러오기

**검색 키워드:** model.save, load_model, 모델 저장, 모델 불러오기, joblib, 스케일러 저장, pkl

```python
model.save('my_model.keras')           # 구조 + 가중치 + 옵티마이저 상태 모두 저장
model = load_model('my_model.keras')   # 바로 predict / evaluate / 추가 학습 가능

import joblib                          # 스케일러 저장 (11강 기구학)
joblib.dump(scaler, 'scaler.pkl')      # 학습 때 쓴 평균/표준편차를 파일로
scaler = joblib.load('scaler.pkl')     # 새 데이터 예측 시 같은 스케일러로 transform
```

[⬆ 목차로 돌아가기](#목차)

---

## 11. 평가, 예측, 혼동행렬

### 11-1. evaluate (테스트 성능)

**검색 키워드:** evaluate, 테스트 성능, test loss, test accuracy, score, 평가

```python
# compile의 loss + metrics 순서대로 리스트를 돌려줌: [loss, metric1, ...]
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0], 'Test accuracy:', score[1])

# 언패킹 버전 (metrics가 1개일 때 값 2개)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')
```

[⬆ 목차로 돌아가기](#목차)

### 11-2. predict (예측값 만들기)

**검색 키워드:** predict, 예측, 확률, argmax, 임계값, threshold, ravel, 새 샘플 예측

**이진 분류:** 확률 → 0/1 (임계값 0.5)

```python
y_prob = model.predict(X_test, verbose=0).ravel()   # (N, 1) → (N,) 1차원으로 펴기
y_pred = (y_prob >= 0.5).astype(int)                # 0.5 이상이면 1, 아니면 0
y_true = np.asarray(y_test)                         # Series → numpy 배열
print('accuracy:', np.mean(y_pred == y_true))       # 맞힌 비율
```

**다중 분류:** 가장 큰 확률의 인덱스

```python
y_prob = model.predict(X_test, verbose=0)           # (N, C): 샘플마다 클래스 C개 확률
y_pred = np.argmax(y_prob, axis=1)                  # axis=1: 각 행에서 가장 큰 값의 위치 → (N,)
y_true = np.argmax(np.asarray(y_test), axis=1)      # 원핫 라벨 → 정수
# y_true = np.asarray(y_test)                       # 정수 라벨이면 그대로
print('accuracy:', np.mean(y_pred == y_true))
```

**회귀**

```python
y_pred = model.predict(X_test, verbose=0).ravel()   # 출력 1개: (N, 1) → (N,)
# y_pred = model.predict(X_test, verbose=0)         # 출력 k개: (N, k) 그대로
# y를 표준화했다면 y_scaler.inverse_transform(...)으로 원래 단위로 (7-3)
```

**새 샘플 1개 예측:** 반드시 2차원 + 학습 때와 같은 스케일러

```python
new = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])   # (1, 특성 수): 대괄호 2겹
new_s = scaler.transform(new)                            # fit_transform 아님! (학습 통계 재사용)
print(model.predict(new_s, verbose=0))                   # [[확률]]
```

[⬆ 목차로 돌아가기](#목차)

### 11-3. 혼동행렬과 분류 지표

**사용처:** "혼동행렬을 출력하시오" (기출)
**목적:** 클래스별로 맞고 틀린 개수를 확인
**검색 키워드:** 혼동행렬, confusion matrix, confusion_matrix, ConfusionMatrixDisplay, TP TN FP FN, precision, recall, F1, 정밀도, 재현율, classification_report, 헷갈린 쌍, 다중 분류 혼동행렬, 직접 구현

```python
# y_true, y_pred: 둘 다 정수 라벨 1차원 배열 (11-2에서 만든 것)
cm = confusion_matrix(y_true, y_pred)        # 행 = 실제, 열 = 예측
print(cm)
# 이진 분류 배치:  [[TN, FP],
#                  [FN, TP]]
tn, fp, fn, tp = cm.ravel()                  # 2×2를 펴면 TN, FP, FN, TP 순서
print(f'TN={tn}, FP={fp}, FN={fn}, TP={tp}')

# 그림으로
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
disp.plot(cmap='Blues')                      # 칸 안에 개수 표시
plt.title('Confusion Matrix')
plt.show()

# 한 줄 버전
# ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues'); plt.show()

# seaborn 히트맵 버전
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,                              # 칸 안에 숫자
    fmt='d',                                 # 정수 형식
    cmap='Blues',
    xticklabels=['pred 0', 'pred 1'],
    yticklabels=['true 0', 'true 1'],
)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()

# 지표
print('accuracy :', accuracy_score(y_true, y_pred))      # (TP + TN) / 전체
print('precision:', precision_score(y_true, y_pred))     # TP / (TP + FP): 양성이라 한 것 중 진짜
print('recall   :', recall_score(y_true, y_pred))        # TP / (TP + FN): 진짜 양성 중 찾은 비율
print('f1       :', f1_score(y_true, y_pred))            # 2PR / (P + R)
print(classification_report(y_true, y_pred, digits=4))   # 클래스별 지표 표
```

**다중 분류:** 코드는 같습니다. 이름표만 클래스 수만큼 넣습니다.

```python
class_names = ['setosa', 'versicolor', 'virginica']      # 또는 le.classes_, y_onehot.columns
cm = confusion_matrix(y_true, y_pred)                    # (C, C)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap='Blues')
plt.xticks(rotation=45)                                  # 이름이 길면 (Fashion-MNIST) 기울이기
plt.show()
print(classification_report(y_true, y_pred, target_names=class_names))

# 다중 분류 precision/recall은 average 지정 필수 (안 하면 오류)
# precision_score(y_true, y_pred, average='macro')   # 클래스별 값의 단순 평균
# recall_score(y_true, y_pred, average='weighted')   # 클래스 개수로 가중 평균
```

**NumPy로 직접 만들기 (sklearn 없이) + 가장 많이 헷갈린 쌍:**

```python
import numpy as np

def my_confusion_matrix(y_true, y_pred, n_classes):
    """행 = 실제, 열 = 예측. (실제 t, 예측 p) 칸에 1씩 더함"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def top_confusions(cm, names, k=3):
    """대각선(정답)을 뺀 칸 중 개수가 큰 k개 = 가장 많이 헷갈린 (실제 → 예측) 쌍"""
    off = cm.copy()
    np.fill_diagonal(off, 0)                     # 맞힌 칸은 0으로
    idx = np.argsort(off.ravel())[::-1][:k]      # 펴서 큰 순서로 k개 위치
    for i in idx:
        t, p = divmod(i, cm.shape[1])            # 1차원 위치 → (행, 열)
        print(f'실제 {names[t]} → 예측 {names[p]}: {off[t, p]}개')

# 예시 (3클래스). 실제 모델이면 y_true, y_pred를 11-2에서 만든 것으로
names = ['setosa', 'versicolor', 'virginica']
y_true_ex = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
y_pred_ex = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 1])
cm = my_confusion_matrix(y_true_ex, y_pred_ex, 3)
print(cm)
# [[2 1 0]
#  [0 2 2]
#  [0 1 2]]
print('accuracy        :', np.trace(cm) / cm.sum())          # 대각선 합 / 전체 = 0.6
print('클래스별 recall :', cm.diagonal() / cm.sum(axis=1))   # 행 합 = 실제 개수
print('클래스별 precision:', cm.diagonal() / cm.sum(axis=0)) # 열 합 = 예측 개수
top_confusions(cm, names, k=2)
# 실제 versicolor → 예측 virginica: 2개
# 실제 setosa → 예측 versicolor: 1개   (1개짜리 칸이 여럿이면 어느 것이 나올지는 순서에 따라 다름)
```

**pandas로 혼동행렬:**

```python
# 행 = Actual, 열 = Predicted 이름이 붙은 표
print(pd.crosstab(pd.Series(y_true, name='Actual'), pd.Series(y_pred, name='Predicted')))
```

**임계값을 바꿔 가며 비교 (4강):**

```python
# 임계값을 낮추면 양성을 더 많이 잡음 → recall ↑, precision ↓
for th in [0.3, 0.5, 0.7]:
    pred = (y_prob >= th).astype(int)                     # y_prob: 11-2 이진 분류 확률
    print(
        f'th={th}: '
        f'precision={precision_score(y_true, pred, zero_division=0):.3f}, '  # 양성 예측 0개면 0
        f'recall={recall_score(y_true, pred):.3f}, '
        f'f1={f1_score(y_true, pred):.3f}'
    )
```

[⬆ 목차로 돌아가기](#목차)

### 11-4. 회귀 지표

**검색 키워드:** 회귀 지표, MAE, MSE, RMSE, R2, r2_score, mean_absolute_error, 출력별 MAE, Wrapped MAE, 각도 오차

```python
mae  = mean_absolute_error(y_test, y_pred)   # 평균 |y − ŷ|: 원래 단위 그대로 해석
mse  = mean_squared_error(y_test, y_pred)    # 평균 (y − ŷ)²: 단위가 제곱
rmse = np.sqrt(mse)                          # √MSE: 원래 단위
r2   = r2_score(y_test, y_pred)              # 1에 가까울수록 좋음 (0이면 평균 예측 수준)
print(f'MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}')
```

**출력이 여러 개일 때 출력별 MAE (11강 기구학):** `y_test`, `y_pred`가 (N, 6)

```python
labels = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
units  = ['mm', 'mm', 'mm', 'rad', 'rad', 'rad']
for i, (label, unit) in enumerate(zip(labels, units)):
    # [:, i]: i번째 출력(열)만 꺼내서 비교
    mae_i = mean_absolute_error(y_test[:, i], y_pred[:, i])
    print(f'MAE for {label:>5}: {mae_i:.3f} {unit}')   # :>5 → 5칸 오른쪽 정렬

# 각도 출력이면 −π와 +π를 같은 방향으로 보는 Wrapped MAE (11강 역기구학)
# arctan2(sin, cos)로 차이를 −π ~ π 범위로 되돌림 (359°와 1°의 차이 = 2°)
angle_diff = np.arctan2(np.sin(y_pred - y_test), np.cos(y_pred - y_test))
print(np.mean(np.abs(angle_diff), axis=0))           # axis=0: 출력(열)별 평균
```

[⬆ 목차로 돌아가기](#목차)

### 11-5. 틀린 이미지 확인 (14강)

**검색 키워드:** 틀린 이미지, 오분류, misclassified, wrong, np.where, imshow, subplots 2x5, 이미지 시각화

```python
y_prob = model.predict(X_test, verbose=0)            # (N, 10)
y_pred = np.argmax(y_prob, axis=1)                   # 예측 클래스
y_true = np.argmax(y_test_oh, axis=1)                # 정수 라벨이면 y_test 그대로
wrong = np.where(y_pred != y_true)[0]                # 틀린 샘플의 인덱스 배열
print('wrong:', len(wrong))

fig, axes = plt.subplots(2, 5, figsize=(10, 4))      # 2행 5열 = 10칸
for ax, idx in zip(axes.ravel(), wrong[:10]):        # ravel: 2차원 축 배열 → 1차원
    ax.imshow(X_test[idx].squeeze(), cmap='gray')    # (28,28,1) → (28,28)
    ax.set_title(f'T:{y_true[idx]} P:{y_pred[idx]}') # T: 정답, P: 예측
    ax.axis('off')                                   # 눈금 숨기기
plt.tight_layout(); plt.show()
```

[⬆ 목차로 돌아가기](#목차)

### 11-6. 여러 모델 결과를 표로 비교

**검색 키워드:** 결과 비교, 표, DataFrame, results, 모델 비교, sort_values

```python
results = []                                          # 딕셔너리를 한 줄씩 모음
# 값은 예시 (실제로는 evaluate/계산한 값을 넣기)
results.append({'model': 'Baseline', 'test_mae': 20503, 'params': model.count_params()})
results.append({'model': 'Dropout 0.1', 'test_mae': 17782, 'params': model.count_params()})

results_df = (
    pd.DataFrame(results)                             # 리스트 of 딕셔너리 → 표
      .sort_values('test_mae')                        # MAE 낮은(좋은) 순
      .reset_index(drop=True)                         # 0, 1, 2 ... 다시 번호
)
print(results_df)
```

[⬆ 목차로 돌아가기](#목차)

---

## 12. 시각화 (학습곡선, 예측 비교)

### 12-1. 학습곡선 (Learning Curve)

**사용처:** "모델 성능(학습 과정)을 시각화하시오" (기출)
**목적:** train/val 곡선으로 과적합 여부 판단 (train은 계속 내려가는데 val이 오르면 과적합)
**검색 키워드:** 학습곡선, 학습 과정 시각화, 과적합 확인, loss 그래프, accuracy 그래프, history, learning curve, val_loss, plot

```python
# 가장 기본: loss
# history = model.fit(...) 의 반환값. history.history는 딕셔너리
#   키 예시: 'loss', 'val_loss', 'accuracy', 'val_accuracy', 'mae', 'val_mae'
#   (val_ 로 시작하는 키는 fit에 validation_split 또는 validation_data를 줬을 때만 생김)
plt.plot(history.history['loss'], label='Train loss')          # epoch별 train loss (리스트)
plt.plot(history.history['val_loss'], label='Validation loss') # epoch별 검증 loss
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()                  # label= 로 붙인 이름을 범례로 표시
plt.grid(alpha=0.3)           # 연한 격자
plt.show()
# 해석: train↓ val↓ → 정상 학습 / train↓ val↑ → 과적합 / 둘 다 높음 → 과소적합
```

**loss와 accuracy를 나란히 (분류):**

```python
hist = history.history                         # 딕셔너리를 짧은 이름으로
fig, axes = plt.subplots(1, 2, figsize=(12, 4))   # 1행 2열 그림, axes[0], axes[1]

# 왼쪽: loss
axes[0].plot(hist['loss'], label='Train')
axes[0].plot(hist['val_loss'], label='Validation')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 오른쪽: accuracy
# 주의: compile에서 metrics=['accuracy']를 줘야 'accuracy' 키가 생김
axes[1].plot(hist['accuracy'], label='Train')
axes[1].plot(hist['val_accuracy'], label='Validation')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()     # 그래프끼리 글자 겹침 방지
plt.show()
```

**회귀 (loss + MAE), 최고 epoch 표시:**

```python
hist = history.history
best = int(np.argmin(hist['val_loss']))     # val_loss가 가장 작은 epoch 번호 (0부터 시작)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, key in zip(axes, ['loss', 'mae']):  # 왼쪽 loss, 오른쪽 mae (compile metrics=['mae'] 필요)
    ax.plot(hist[key], label=f'Train {key}')
    ax.plot(hist['val_' + key], label=f'Validation {key}')
    # 최고 epoch 위치에 세로 점선. 사람이 읽는 번호는 +1 (1번째 epoch부터 셈)
    ax.axvline(best, linestyle='--', color='gray', label=f'Best epoch {best + 1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(key)
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**DataFrame으로 (6강 Wine 방식, 점으로 찍기):**

```python
# 열: loss, accuracy, val_loss, val_accuracy ... / 행: epoch
hist_df = pd.DataFrame(history.history)
print(hist_df.tail())                     # 마지막 5 epoch 값 확인
x_len = np.arange(len(hist_df))           # x축: 0, 1, 2, ..., epoch수-1

# 'o' = 점 모양, c = 색, markersize = 점 크기
plt.plot(x_len, hist_df['val_loss'], 'o', c='red', markersize=2, label='Validation loss')
plt.plot(x_len, hist_df['loss'], 'o', c='blue', markersize=2, label='Train loss')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 한 줄 버전: DataFrame.plot이 열마다 선을 그려줌
hist_df[['loss', 'val_loss']].plot(title='Loss')
plt.show()
```

**표준화한 y의 MAE를 원래 단위로 그리기 (11강 Ames):**

```python
# y를 StandardScaler로 표준화해서 학습했다면 history의 mae도 '표준화 단위'
# 표준화: y_s = (y − μ)/σ  →  오차도 σ로 나눠진 상태  →  원래 단위 MAE = mae_scaled × σ
# y_scaler.scale_[0] = σ (train y의 표준편차). 평균 μ는 오차에서 빠지므로 곱하기만 하면 됨
train_mae = np.array(history.history['mae']) * y_scaler.scale_[0]
val_mae   = np.array(history.history['val_mae']) * y_scaler.scale_[0]
plt.plot(train_mae, label='Training MAE')
plt.plot(val_mae, label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE ($)')
plt.legend()
plt.grid()
plt.show()
```

**여러 모델의 val 곡선 비교 (9강 BN vs no BN):**

```python
# history_a, history_b: 서로 다른 모델을 fit한 결과 (같은 데이터, 같은 epoch 권장)
plt.plot(history_a.history['val_loss'], label='Model A')
plt.plot(history_b.history['val_loss'], label='Model B')
plt.xlabel('Epoch')
plt.ylabel('Validation loss')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# EarlyStopping을 쓰면 모델마다 곡선 길이(epoch 수)가 다를 수 있음 → 정상
```

[⬆ 목차로 돌아가기](#목차)

### 12-2. 예측 vs 실제 산점도 (회귀)

**검색 키워드:** 예측 vs 실제, 산점도, y=x 대각선, 회귀 결과 시각화, scatter, actual vs predicted, parity plot

```python
# y_pred = model.predict(X_test).ravel()  처럼 1차원으로 맞춰두면 편함
plt.figure(figsize=(5, 5))                     # 정사각형 그림 (대각선이 45도로 보이게)
plt.scatter(y_test, y_pred, alpha=0.5)         # x축 실제값, y축 예측값, alpha=투명도
# 대각선 y = x 를 그릴 범위: 두 배열 전체의 최솟값 ~ 최댓값
lo = min(np.min(y_test), np.min(y_pred))
hi = max(np.max(y_test), np.max(y_pred))
plt.plot([lo, hi], [lo, hi], 'r--', label='y = x (perfect)')   # 빨간 점선. 점이 선에 붙을수록 정확
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**출력이 여러 개일 때 (2D 출력 회귀):**

```python
# y_test, y_pred 모양: (N, 출력개수). i번째 출력은 [:, i] 로 꺼냄
names = ['out1', 'out2']                     # 출력 이름 (출력 개수만큼)
fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
# 출력이 1개면 axes가 배열이 아니라 축 1개 → np.atleast_1d로 항상 배열처럼 다룸
for i, ax in enumerate(np.atleast_1d(axes)):
    ax.scatter(y_test[:, i], y_pred[:, i], s=5, alpha=0.5)   # s = 점 크기
    lo, hi = y_test[:, i].min(), y_test[:, i].max()
    ax.plot([lo, hi], [lo, hi], 'r--')                        # y = x 기준선
    # 제목에 출력별 MAE 표시 (sklearn mean_absolute_error = mean(|y − ŷ|))
    mae_i = mean_absolute_error(y_test[:, i], y_pred[:, i])
    ax.set_title(f'{names[i]}  MAE={mae_i:.3f}')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
plt.tight_layout()
plt.show()
```

[⬆ 목차로 돌아가기](#목차)

### 12-3. 예측값과 실제값을 선으로 (시계열, 샘플 순서)

**검색 키워드:** 시계열 예측 그래프, 실제값 예측값 선 그래프, 순서대로 비교, time series plot, true vs predicted line

```python
n_plot = 300                                  # 앞 300개만 그림 (전부 그리면 너무 빽빽함)
plt.figure(figsize=(10, 4))                   # 가로로 긴 그림
# np.asarray: Series/DataFrame이어도 numpy 배열로 바꿔서 [:n_plot] 슬라이싱이 위치 기준으로 되게
plt.plot(np.asarray(y_test)[:n_plot], label='True')
plt.plot(np.asarray(y_pred)[:n_plot], label='Predicted')
plt.xlabel('Sample index')                    # 시계열이면 시간 순서
plt.legend()
plt.grid(True)
plt.show()
# 다중 출력(N, 24)이면 한 샘플의 24시간을 그리려면: plt.plot(y_test[0]); plt.plot(y_pred[0])
```

[⬆ 목차로 돌아가기](#목차)

### 12-4. 데이터, 함수 그래프

**검색 키워드:** 함수 그래프, 활성화 함수 그리기, 히스토그램, 막대그래프, 이미지 출력, sigmoid tanh relu plot, hist, bar, imshow

```python
x = np.linspace(-5, 5, 300)                   # -5 ~ 5 를 300등분한 점 (끝 5 포함)
plt.plot(x, 1 / (1 + np.exp(-x)), label='sigmoid')   # σ(x) = 1/(1+e^-x), 범위 (0, 1)
plt.plot(x, np.tanh(x), label='tanh')                # tanh, 범위 (-1, 1)
plt.plot(x, np.maximum(0, x), label='ReLU')          # max(0, x)
plt.axhline(0, lw=0.8)                        # y=0 가로선 (lw = 선 굵기)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.hist(df['age'], bins=30)                  # 히스토그램: 값 분포, bins = 막대 개수
plt.show()

plt.bar(['A', 'B', 'C'], [0.91, 0.88, 0.93])  # 막대그래프: (이름 리스트, 값 리스트)
plt.show()

plt.imshow(X_train[0], cmap='gray')           # 이미지 1장 (28,28). (28,28,1)이면 .squeeze()
plt.axis('off')                               # 축 눈금 숨김
plt.show()
```

[⬆ 목차로 돌아가기](#목차)

---

## 13. NumPy 직접 구현과 계산 문제

### 13-1. numpy 랜덤 + 조건 게임

**사용처:** "주사위/동전/가위바위보를 N번 시행하고 조건에 따라 점수를 계산하시오" (기출)
**목적:** 난수 생성 + 조건문 + 집계
**검색 키워드:** 주사위, 동전, 가위바위보, 난수, 시뮬레이션, 랜덤 게임, 조건 점수, random, randint, choice, seed, while, 연속 횟수

**난수 함수 요약**

| 함수 | 결과 | 주의 |
|---|---|---|
| `np.random.seed(42)` | 시드 고정 | 같은 결과 재현 |
| `np.random.randint(1, 7, size=10)` | 1~6 정수 10개 | **끝값 7은 미포함** |
| `np.random.rand(3)` | [0, 1) 실수 3개 | |
| `np.random.uniform(-1, 1, size=5)` | [-1, 1) 균등분포 | |
| `np.random.normal(0, 1, size=5)` | 평균 0, 표준편차 1 정규분포 | `randn(5)`와 같은 분포 |
| `np.random.choice(['가위','바위','보'])` | 하나 무작위 선택 | |
| `np.random.choice(10, size=3, replace=False)` | 0~9 중 중복 없이 3개 | |
| `np.random.choice([0, 1], size=10, p=[0.3, 0.7])` | 확률 지정 | p의 합 = 1 |
| `np.random.shuffle(arr)` | 원본 배열을 섞음 | 반환값 None |
| `np.random.permutation(10)` | 0~9를 섞은 새 배열 | |
| `rng = np.random.default_rng(42)` → `rng.integers(1, 7, size=10)` | 새 방식 | 끝값 미포함 |

```python
np.random.seed(42)          # 시드 고정 → 실행할 때마다 같은 주사위 결과

# (1) 주사위 2개를 10번 던져 합이 7 이상이면 승리
wins = 0                    # 승리 횟수 누적 변수 (반복문 밖에서 0으로 시작)
for game in range(10):      # game = 0, 1, ..., 9 (10판)
    d1 = np.random.randint(1, 7)    # 1~6 (7은 미포함!)
    d2 = np.random.randint(1, 7)
    total = d1 + d2
    if total >= 7:          # 조건: 합이 7 이상
        result = 'win'
        wins += 1           # wins = wins + 1
    else:
        result = 'lose'
    # f-string: {} 안의 변수가 값으로 바뀜. game은 0부터라 +1 해서 1판부터 출력
    print(f'game {game + 1}: {d1} + {d2} = {total} → {result}')
print('wins:', wins, 'win rate:', wins / 10)   # 승률 = 승리 수 / 전체 판 수
```

```python
# (2) 같은 문제를 벡터화 (반복문 없이, 대량 시뮬레이션)
np.random.seed(42)
N = 100000
dice = np.random.randint(1, 7, size=(N, 2))     # (N, 2): 한 행 = 한 판의 주사위 2개
total = dice.sum(axis=1)                        # axis=1 → 행마다(가로로) 합 → (N,)
win = total >= 7                                # True/False 배열 (N,)
# True=1, False=0 이라 평균 = 비율. 이론값 21/36 = 0.5833
print('win rate:', win.mean())
# np.where(조건, 참일 때, 거짓일 때): 배열 전체에 if-else
print('result:', np.where(win, 'win', 'lose')[:10])   # 앞 10판만 출력
```

```python
# (3) 점수 규칙이 여러 개: 합 12면 +3, 7 이상이면 +1, 아니면 -1
np.random.seed(0)
score = 0
for _ in range(20):                  # _ : 반복 번호를 안 쓸 때 관례적인 변수 이름
    s = np.random.randint(1, 7) + np.random.randint(1, 7)
    if s == 12:                      # 가장 특별한 조건을 먼저 검사 (12도 '7 이상'이라 순서 중요)
        score += 3
    elif s >= 7:
        score += 1
    else:
        score -= 1
print('final score:', score)

# 벡터화: np.select(조건 리스트, 값 리스트, default=나머지 값)
#   앞의 조건부터 검사해서 처음 True인 조건의 값을 씀 (if-elif와 같은 순서 규칙)
#   주의: 위 반복문과 난수를 이어서 뽑으므로 결과 값은 위와 다를 수 있음
s = np.random.randint(1, 7, size=20) + np.random.randint(1, 7, size=20)   # (20,)
points = np.select([s == 12, s >= 7], [3, 1], default=-1)
print(points, points.sum())          # 판별 점수 배열, 총점
```

```python
# (4) 가위바위보 N판
np.random.seed(1)
moves = ['rock', 'paper', 'scissors']
beats = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}   # key가 value를 이김
record = {'win': 0, 'lose': 0, 'draw': 0}                           # 결과 집계용 딕셔너리
for _ in range(10):
    me, com = np.random.choice(moves), np.random.choice(moves)      # 각자 하나씩 무작위
    if me == com:                    # 같은 손 → 무승부
        record['draw'] += 1
    elif beats[me] == com:           # 내가 이기는 상대가 com이면 승리
        record['win'] += 1
    else:                            # 나머지는 패배
        record['lose'] += 1
print(record)                        # 세 값의 합 = 10
```

```python
# (5) 목표에 도달할 때까지 반복 (while): 누적합이 30 이상이 될 때까지 몇 번 던지나
np.random.seed(3)
total, count = 0, 0
while total < 30:                    # 조건이 True인 동안 계속 반복 (몇 번일지 모를 때 while)
    roll = np.random.randint(1, 7)
    total += roll                    # 누적합
    count += 1                       # 던진 횟수
print(f'{count}번 만에 {total} 도달')

# (6) 동전 100번: 앞면(1) 비율, 연속 앞면 최대 길이
np.random.seed(7)
coin = np.random.randint(0, 2, size=100)    # 0 또는 1 (2는 미포함)
print('head ratio:', coin.mean())           # 1의 비율 = 평균
max_run = run = 0                           # run: 현재 연속 길이, max_run: 지금까지 최대
for c in coin:
    run = run + 1 if c == 1 else 0          # 앞면이면 연속 +1, 뒷면이면 0으로 리셋
    max_run = max(max_run, run)             # 최대값 갱신
print('longest head streak:', max_run)

# (7) 결과를 DataFrame으로 정리
np.random.seed(42)
games = pd.DataFrame({'d1': np.random.randint(1, 7, 10),      # 열 이름: 값 배열 (10판)
                      'd2': np.random.randint(1, 7, 10)})
games['sum'] = games['d1'] + games['d2']                       # 열끼리 더하기 → 새 열
games['result'] = np.where(games['sum'] >= 7, 'win', 'lose')   # 조건 라벨링
print(games)
print(games['result'].value_counts())                          # win/lose 개수
```

[⬆ 목차로 돌아가기](#목차)

### 13-2. 활성화/손실 함수 직접 계산

**검색 키워드:** 시그모이드, 소프트맥스, 렐루, 활성화 함수 구현, MSE, MAE, BCE, CCE, 크로스엔트로피, 엔트로피, KL 발산, 코사인 유사도, sigmoid, softmax, relu, leaky relu, cross entropy, entropy, KL divergence, cosine similarity

```python
def sigmoid(z):
    # σ(z) = 1 / (1 + e^(−z)),  출력 범위 (0, 1). 이진 분류 확률
    return 1 / (1 + np.exp(-z))

def relu(z):
    # ReLU(z) = max(0, z). 음수는 0, 양수는 그대로 (np.maximum은 원소별 최대)
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.1):
    # z ≥ 0 이면 z, 음수면 α·z (음수 쪽 기울기가 0이 아니라 α)
    return np.where(z >= 0, z, alpha * z)

def softmax(z):
    # softmax(z)_i = e^(z_i) / Σ_j e^(z_j),  합 = 1 인 확률 벡터
    z = z - np.max(z)                 # 오버플로 방지: 모든 값에서 최댓값을 빼도 결과는 같음
    e = np.exp(z)
    return e / e.sum()

def softmax_batch(Z):                 # Z 모양 (N, C): 샘플 N개, 클래스 C개
    # 행(샘플)마다 softmax. keepdims=True → (N,1) 모양 유지해서 (N,C)와 브로드캐스팅
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)   # 각 행의 합 = 1

print(sigmoid(0), sigmoid(2))                       # 0.5, 0.8808
print(softmax(np.array([1.0, 2.0, 3.0])))           # [0.0900 0.2447 0.6652]
# sigmoid 미분: σ'(z) = σ(z)(1 − σ(z)), 최댓값 0.25 (z=0일 때). z=2면 0.105
print(sigmoid(2) * (1 - sigmoid(2)))
```

```python
# MSE = mean((y − ŷ)²),  MAE = mean(|y − ŷ|)
y_true = np.array([1., 2., 3.])
y_pred = np.array([1., 2.5, 2.])
print('MSE:', np.mean((y_true - y_pred) ** 2))      # (0 + 0.25 + 1)/3 = 0.4167
print('MAE:', np.mean(np.abs(y_true - y_pred)))     # (0 + 0.5 + 1)/3 = 0.5

# BCE: −[y ln p + (1−y) ln(1−p)] 의 평균  (y: 0/1 정답, p: 양성 확률)
def bce(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)      # p가 0이나 1이면 log(0) = −inf → 아주 작은 값으로 자름
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
# y=1,p=0.8 → −ln0.8 = 0.2231 / y=0,p=0.2 → −ln0.8 = 0.2231 → 평균 0.2231
print('BCE:', bce(np.array([1, 0]), np.array([0.8, 0.2])))   # 0.2231

# CCE: −Σ y ln ŷ  (원핫이면 정답 클래스의 −ln 확률만 남음)
y_onehot = np.array([0, 1, 0, 0])               # 정답 = 1번 클래스
y_hat = np.array([0.02, 0.85, 0.10, 0.03])      # 모델 확률 (softmax 출력)
print('CCE:', -np.sum(y_onehot * np.log(y_hat)))    # 0.1625 = -ln(0.85)

# 배치 CCE (샘플 평균): 행마다 −Σ y ln p 를 구한 뒤 평균
Y = np.array([[0, 0, 1], [1, 0, 0]])            # 정답 원핫 (N=2, C=3)
P = np.array([[0.2, 0.7, 0.1], [0.7, 0.2, 0.1]])
# 샘플1: −ln 0.1 = 2.3026 / 샘플2: −ln 0.7 = 0.3567
print('batch CCE:', -np.mean(np.sum(Y * np.log(P), axis=1)))   # (2.3026 + 0.3567)/2 = 1.3296

# 엔트로피, KL, 코사인 유사도
p = np.array([0.7, 0.2, 0.1])
q = np.array([0.5, 0.3, 0.2])
# 엔트로피 H(p) = −Σ p log p  (log2 → 단위 bit, ln → 단위 nat)
print('entropy(bits):', -np.sum(p * np.log2(p)))    # 1.1568
# KL(p‖q) = Σ p ln(p/q) ≥ 0,  p=q일 때만 0. 대칭 아님: KL(p‖q) ≠ KL(q‖p)
print('KL(p||q):', np.sum(p * np.log(p / q)))       # 0.0851
# 코사인 유사도 = a·b / (‖a‖‖b‖), 방향이 같으면 1, 수직이면 0, 반대면 −1
a, b = np.array([1., 1.]), np.array([2., 2.])
print('cosine:', a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))   # 1.0 (같은 방향)
```

[⬆ 목차로 돌아가기](#목차)

### 13-3. Affine 계산과 Keras 가중치 확인

**사용처:** "z = Wx + b를 계산하시오", "층을 통과한 출력을 구하시오"
**검색 키워드:** 아핀 변환, Wx+b, 순전파, 행렬곱, 원소곱, 가중치 꺼내기, get_weights, set_weights, affine, forward pass, kernel, bias, 전치

```python
# 교안 표기: W (출력, 입력) 행렬, x 열벡터 → z = W @ x + b
x = np.array([1.0, 2.0])                  # 입력 2개
W1 = np.array([[0.5, 0.2],                # 1번 뉴런의 가중치 [w11, w12]
               [-0.3, 0.8]])              # 2번 뉴런의 가중치 → W1 모양 (출력2, 입력2)
b1 = np.array([0.1, -0.1])
# z1[0] = 0.5·1 + 0.2·2 + 0.1 = 1.0 / z1[1] = −0.3·1 + 0.8·2 − 0.1 = 1.2
z1 = W1 @ x + b1          # [1.0, 1.2]
h = relu(z1)              # 둘 다 양수라 그대로 [1.0, 1.2]
W2 = np.array([[0.7, -0.4]])              # 출력 1개, 입력 2개 → (1, 2)
b2 = np.array([0.2])
# ŷ = 0.7·1.0 − 0.4·1.2 + 0.2 = 0.42
y_hat = W2 @ h + b2       # [0.42]
print(z1, h, y_hat)

# 행렬곱(@) vs 원소곱(*)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)              # [[19 22] [43 50]]  (행·열 내적: 1·5+2·7=19 ...)
print(A * B)              # [[ 5 12] [21 32]]  (같은 위치끼리 곱, Hadamard)
```

```python
# 배치 계산: X (N, 입력) @ W (입력, 출력) + b  ← Keras Dense가 쓰는 배치
X = np.array([[1.0, 2.0], [3.0, 4.0]])          # 샘플 2개 (한 행 = 한 샘플)
W = np.array([[0.5, -0.3], [0.2, 0.8]])         # (입력 2, 출력 2)  ← Keras kernel 모양
b = np.array([0.1, -0.1])                       # (출력 2,)
# 결과 (2, 2):
#   [[0.5+0.4+0.1, −0.3+1.6−0.1],
#    [1.5+0.8+0.1, −0.9+3.2−0.1]] = [[1.0, 1.2], [2.4, 2.2]]
print(X @ W + b)                                 # 브로드캐스팅으로 b가 각 행에 더해짐
```

```python
# Keras 층의 가중치 꺼내기 / 넣기 → 직접 계산과 비교
m = Sequential([Input(shape=(2,)), Dense(2, activation='relu'), Dense(1)])
# get_weights() → [kernel, bias] 리스트. kernel shape = (입력, 출력) = (2, 2), bias = (2,)
kernel, bias = m.layers[0].get_weights()
print(kernel.shape, bias.shape)                 # (2, 2) (2,)

# set_weights([kernel, bias]): 모양이 정확히 같아야 함
m.layers[0].set_weights([np.array([[0.5, -0.3], [0.2, 0.8]]), np.array([0.1, -0.1])])
m.layers[1].set_weights([np.array([[0.7], [-0.4]]), np.array([0.2])])   # (2,1), (1,)
x_in = np.array([[1.0, 2.0]])                   # 배치 모양 (1, 2): 샘플 1개라도 2차원
# 직접 계산: relu(x @ W1 + b1) @ W2 + b2
W1_k, b1_k = m.layers[0].get_weights()
W2_k, b2_k = m.layers[1].get_weights()
manual = relu(x_in @ W1_k + b1_k) @ W2_k + b2_k
print('manual:', manual, ' keras:', m.predict(x_in, verbose=0))   # 둘이 같아야 함
```

> **주의:** 교안 수식은 `W @ x`에서 W가 (출력, 입력)이고, Keras kernel은 (입력, 출력)이라 `x @ W`입니다. 서로 전치(`.T`) 관계입니다.

[⬆ 목차로 돌아가기](#목차)

### 13-4. 가중치 업데이트 (GD, momentum, Adam)

**사용처:** "w를 한 번 업데이트한 값을 구하시오", "momentum을 적용해 2 step 계산하시오" (기출)
**검색 키워드:** 가중치 업데이트, 경사하강, 학습률, 모멘텀, 아담, 가중치 감쇠, L2, w 업데이트, gradient descent, momentum, Adam, weight decay, AdamW, bias correction, 편향 보정

```python
# (1) 경사하강 한 스텝: w ← w − α·∂L/∂w   (α = 학습률 lr)
w, lr, grad = 5.0, 0.1, 4.0
w = w - lr * grad                              # 5 − 0.1×4
print(w)                                       # 4.6

# 예: L = (w·x + b − y)²  (샘플 1개)의 기울기
x, y, w, b, lr = 2.0, 5.0, 1.0, 1.0, 0.1
y_hat = w * x + b                              # 1·2 + 1 = 3
# 연쇄 법칙: ∂L/∂w = 2(ŷ − y)·∂ŷ/∂w = 2(ŷ − y)·x
grad_w = 2 * (y_hat - y) * x                   # 2·(3−5)·2 = −8
grad_b = 2 * (y_hat - y)                       # ∂ŷ/∂b = 1 → 2·(3−5) = −4
# 동시 업데이트 (w를 먼저 바꾸고 그 w로 b 기울기를 다시 계산하면 안 됨)
w, b = w - lr * grad_w, b - lr * grad_b        # 1 + 0.8, 1 + 0.4
print(w, b)                                    # 1.8, 1.4
# 손실이 ½(ŷ−y)² 이면 기울기에서 2가 빠짐 (5강)
```

```python
# (2) Momentum (8강 교안 형태: 지수이동평균)
#   m ← β·m + (1−β)·g,   w ← w − α·m
def momentum_slide(w, grads, lr=0.1, beta=0.9):
    m = 0.0                                   # 처음 속도(모멘텀)는 0
    for t, g in enumerate(grads, 1):          # t = 1, 2, ... (enumerate 시작값 1)
        m = beta * m + (1 - beta) * g         # 이전 방향 β만큼 + 이번 기울기 (1−β)만큼
        w = w - lr * m                        # 기울기 대신 m으로 이동
        print(f'step {t}: g={g}, m={m:.4f}, w={w:.4f}')
    return w

# step1: m = 0.9·0 + 0.1·4 = 0.4,  w = 5 − 0.1·0.4 = 4.96
# step2: m = 0.9·0.4 + 0.1·2 = 0.56, w = 4.96 − 0.056 = 4.904
momentum_slide(5.0, [4.0, 2.0])      # step1: m=0.4, w=4.96 / step2: m=0.56, w=4.904
```

```python
# (3) Momentum (8강 노트북 / Keras 형태): v ← μ·v + g, w ← w − α·v
#     Keras SGD(momentum=μ) 내부: u ← μ·u − α·g, w ← w + u
#     (학습률이 일정하면 노트북 형태와 같은 결과)
def momentum_keras(w, grads, lr=0.1, mu=0.9):
    v = 0.0
    for t, g in enumerate(grads, 1):
        v = mu * v + g                        # (1−β)를 곱하지 않음 → 이동량이 더 큼
        w = w - lr * v
        print(f'step {t}: g={g}, v={v:.4f}, w={w:.4f}')
    return w

# step1: v = 4, w = 5 − 0.4 = 4.6 / step2: v = 0.9·4 + 2 = 5.6, w = 4.6 − 0.56 = 4.04
momentum_keras(5.0, [4.0, 2.0])      # step1: v=4.0, w=4.6 / step2: v=5.6, w=4.04
```

> 교안 형태는 (1−β)가 곱해져서 이동량이 작습니다. 문제에 주어진 **수식 형태를 그대로** 따라 계산하세요.

```python
# (4) Adam 한 스텝씩 (8강)
#   m ← β1·m + (1−β1)·g        (1차 모멘트: 기울기 평균)
#   v ← β2·v + (1−β2)·g²       (2차 모멘트: 기울기 제곱 평균)
#   m̂ = m/(1−β1^t), v̂ = v/(1−β2^t)   (편향 보정: 초반에 m, v가 0 쪽으로 치우친 것 보정)
#   w ← w − α·m̂/(√v̂ + ε)
def adam_steps(w, grads, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    m = v = 0.0
    for t, g in enumerate(grads, 1):          # t는 1부터 (β^t 계산에 사용)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        print(f'step {t}: m={m:.4f}, v={v:.6f}, m_hat={m_hat:.4f}, '
              f'v_hat={v_hat:.4f}, w={w:.6f}')
    return w

# step1: m=0.4, v=0.016, m̂=4, v̂=16 → 이동 0.1·4/4 = 0.1 → w = 4.9
adam_steps(5.0, [4.0, 2.0], lr=0.1)  # 첫 스텝 이동량 ≈ lr (m_hat/√v_hat = g/|g| = 1)
```

```python
# (5) Weight decay와 L2 (10강)
w, lr, lam, grad_data = 5.0, 0.1, 0.01, 4.0   # lam = λ (감쇠 계수)
# Weight decay: w ← (1 − αλ)·w − α·∇L_data
#   = 0.999·5 − 0.4 = 4.595
w_wd = (1 - lr * lam) * w - lr * grad_data
# L2 정규화 (L_total = L_data + λ/2·‖w‖²): 기울기 = ∇L_data + λw
#   w ← w − α(∇L_data + λw) = 5 − 0.1·(4 + 0.05) = 4.595  → SGD에서는 위와 같은 값
w_l2 = w - lr * (grad_data + lam * w)
print(w_wd, w_l2)                    # 4.595, 4.595
# AdamW: w ← (1 − αλ)·w − α·m_hat / (√v_hat + ε)   (감쇠를 Adam 이동과 분리 = decoupled)
# 참고: Keras l2(λ)는 손실에 λ·Σw² (½ 없음)를 더하므로 기울기는 2λw
```

[⬆ 목차로 돌아가기](#목차)

### 13-5. 선형/로지스틱 회귀 직접 구현 (3강, 4강)

**검색 키워드:** 선형회귀 직접 구현, 최소제곱법, 정규방정식, 경사하강법, 로지스틱 회귀 직접 구현, 기울기 절편, least squares, normal equation, gradient descent, logistic regression, polyfit

```python
x = np.array([1, 3, 5, 7, 9], dtype=float)        # 공부 시간
y = np.array([75, 77, 85, 83, 90], dtype=float)   # 점수

# (1) 최소제곱법 공식
#   a = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²,   b = ȳ − a·x̄
mx, my = x.mean(), y.mean()                # x̄ = 5, ȳ = 82
a = np.sum((x - mx) * (y - my)) / np.sum((x - mx) ** 2)   # 72 / 40 = 1.8
b = my - a * mx                            # 82 − 1.8·5 = 73
print(f'y = {a:.4f}x + {b:.4f}')           # y = 1.8000x + 73.0000

# (2) 정규방정식 β = (XᵀX)⁻¹Xᵀy
X = np.column_stack((x, np.ones(len(x))))  # [x, 1] 열을 붙인 (5, 2) 설계행렬. 1열 = 절편용
beta = np.linalg.inv(X.T @ X) @ X.T @ y    # inv = 역행렬
print(beta)                                # [1.8, 73.]  (기울기, 절편 순서 = X 열 순서)
# 같은 결과: np.linalg.lstsq(X, y, rcond=None)[0],  np.polyfit(x, y, 1)

# (3) 경사하강법: L = (1/n)Σ(y − (ax+b))²
#   ∂L/∂a = −(2/n)Σ x·(y − ŷ),  ∂L/∂b = −(2/n)Σ (y − ŷ)
a, b, lr, n = 0.0, 0.0, 0.003, len(x)
for epoch in range(8001):
    y_pred = a * x + b
    error = y - y_pred                     # 실제 − 예측
    a_diff = -(2 / n) * np.sum(x * error)
    b_diff = -(2 / n) * np.sum(error)
    a -= lr * a_diff                       # a = a − lr·∂L/∂a
    b -= lr * b_diff
    if epoch % 2000 == 0:                  # 2000 epoch마다 진행 상황 출력
        print(f'epoch {epoch}: a={a:.4f}, b={b:.4f}, MSE={np.mean(error ** 2):.4f}')
# 끝까지 돌면 a ≈ 1.8, b ≈ 73 에 가까워짐 (lr이 작아서 b가 천천히 수렴)
```

```python
# (4) 로지스틱 회귀 경사하강 (BCE의 기울기 = (p − y)·x)
#   p = σ(a·x + b),  L = BCE 평균
#   ∂L/∂a = mean((p − y)·x),  ∂L/∂b = mean(p − y)   ← sigmoid+BCE 조합이라 식이 단순해짐
hours = np.array([1, 3, 5, 7, 9, 11, 13], dtype=float)
outcome = np.array([0, 0, 0, 1, 1, 1, 1], dtype=float)   # 0 불합격, 1 합격
a, b, lr = 0.0, 0.0, 0.05
for epoch in range(3000):
    p = sigmoid(a * hours + b)             # 합격 확률 (7,)
    error = p - outcome                    # 예측확률 − 정답 (부호 주의: 선형회귀와 반대 방향 정의)
    a -= lr * np.mean(error * hours)
    b -= lr * np.mean(error)
print(a, b, 'P(pass | 8h)=', sigmoid(a * 8 + b))   # 8시간 공부 시 합격 확률
# 분류 결정: 확률 ≥ 0.5 → 1 (합격)
```

[⬆ 목차로 돌아가기](#목차)

### 13-6. 역전파 직접 계산 (5강)

**검색 키워드:** 역전파, 연쇄 법칙, 미분, 기울기 계산, 2층 신경망, 자동미분, 수치미분, backpropagation, chain rule, gradient, GradientTape, numerical gradient

```python
# 2층 스칼라 네트워크: z1 = w1·x + b1, h = σ(z1), ŷ = w2·h + b2, L = ½(ŷ − y)²
x, y = 1.5, 2.0
w1, b1, w2, b2 = 0.8, 0.1, 1.2, -0.2
# 순전파 (forward)
z1 = w1 * x + b1          # 0.8·1.5 + 0.1 = 1.3
h = sigmoid(z1)           # σ(1.3) ≈ 0.7858
y_hat = w2 * h + b2       # 1.2·0.7858 − 0.2 ≈ 0.7429
L = 0.5 * (y_hat - y) ** 2

# 역전파 (backward): 출력 쪽부터 연쇄 법칙으로 곱해 내려감
dL_dyhat = y_hat - y              # ∂L/∂ŷ = ŷ − y  (½ 덕분에 2가 사라짐)
dL_dw2 = dL_dyhat * h             # ŷ = w2·h + b2 → ∂ŷ/∂w2 = h
dL_db2 = dL_dyhat                 # ∂ŷ/∂b2 = 1
dL_dh = dL_dyhat * w2             # ∂ŷ/∂h = w2 → 아래층으로 전달
dL_dz1 = dL_dh * h * (1 - h)      # sigmoid 미분 σ'(z1) = h(1−h)
dL_dw1 = dL_dz1 * x               # z1 = w1·x + b1 → ∂z1/∂w1 = x
dL_db1 = dL_dz1                   # ∂z1/∂b1 = 1
print(L, dL_dw1, dL_db1, dL_dw2, dL_db2)
# 업데이트 예: w1 = w1 − lr·dL_dw1 (나머지도 같은 방식)

# TensorFlow 자동미분으로 검증 (TF 필요)
w1_t, b1_t, w2_t, b2_t = [tf.Variable(v) for v in (0.8, 0.1, 1.2, -0.2)]
with tf.GradientTape() as tape:                 # 이 블록 안의 계산을 기록
    y_hat_t = w2_t * tf.sigmoid(w1_t * x + b1_t) + b2_t
    loss_t = 0.5 * (y_hat_t - y) ** 2
# 기록된 계산으로 각 변수에 대한 기울기를 자동 계산 → 위 직접 계산 값과 같아야 함
print([float(g) for g in tape.gradient(loss_t, [w1_t, b1_t, w2_t, b2_t])])

# 수치 미분으로 검증: (L(w+ε) − L(w−ε)) / 2ε
#   예) eps = 1e-5
#       Lp = 0.5 * (w2 * sigmoid((w1 + eps) * x + b1) + b2 - y) ** 2
#       Lm = 0.5 * (w2 * sigmoid((w1 - eps) * x + b1) + b2 - y) ** 2
#       print((Lp - Lm) / (2 * eps))   # ≈ dL_dw1
```

[⬆ 목차로 돌아가기](#목차)

### 13-7. BN, LN, Dropout 직접 계산 (9강, 10강)

**검색 키워드:** 배치 정규화, 레이어 정규화, 드롭아웃 직접 구현, 정규화 계산, gamma beta, batch normalization, layer normalization, dropout, inverted dropout, keep prob

```python
Z = np.array([[1., 10., 100.],     # 행 = 샘플(배치 4개), 열 = 특성(3개)
              [2., 12., 80.],
              [3., 14., 120.],
              [4., 16., 100.]])
eps = 1e-5                          # 분모가 0이 되는 것 방지
# BN: 특성(열)별로, 배치 방향 axis=0
#   ẑ = (z − μ_열) / √(σ²_열 + ε)    ← 열마다 평균 0, 분산 1
Z_bn = (Z - Z.mean(axis=0)) / np.sqrt(Z.var(axis=0) + eps)
gamma, beta = np.array([1., 2., 0.5]), np.array([0., 1., -1.])   # 학습되는 스케일, 이동 (특성마다)
S = gamma * Z_bn + beta                     # 스케일, 이동: s = γ·ẑ + β
# LN: 샘플(행)별로 axis=1   (keepdims=True로 (4,1) 모양 유지 → 브로드캐스팅)
Z_ln = (Z - Z.mean(axis=1, keepdims=True)) / np.sqrt(Z.var(axis=1, keepdims=True) + eps)
print(Z_bn.mean(axis=0), Z_bn.std(axis=0))  # ≈ 0, ≈ 1  (열마다)
# 참고: np.var는 n으로 나눔 (BN과 동일). pandas .var()는 n−1로 나눔

# Inverted Dropout: 학습 때 살아남은 값을 1/keep_prob 배
#   → 평균(기댓값)이 유지되어 추론 때는 아무것도 안 곱해도 됨
rng = np.random.default_rng(0)
h = np.array([0.5, 1.0, 2.0, 0.2, 1.5, 0.8])
keep_prob = 1 - 0.5                          # Dropout(0.5) → 50% 유지
mask = (rng.random(len(h)) < keep_prob).astype(float)   # 1 = 살림, 0 = 끔
print(h * mask / keep_prob)                 # 꺼진 곳은 0, 살아남은 곳은 2배. 추론 때는 h 그대로
```

[⬆ 목차로 돌아가기](#목차)

### 13-8. 파라미터 수, CNN 크기 계산기

**검색 키워드:** 파라미터 수 계산, 출력 크기 계산, 합성곱 출력 크기, Dense 파라미터, Conv2D 파라미터, param count, output size, stride, padding, summary

```python
def dense_params(n_in, n_out, bias=True):
    # Dense: 가중치 n_in × n_out + 편향 n_out
    return n_in * n_out + (n_out if bias else 0)

def conv_out(n, f, p=0, s=1):
    # 출력 크기 = ⌊(n + 2p − f)/s⌋ + 1   (n 입력 크기, f 필터 크기, p 패딩, s 스트라이드)
    # 'same' 패딩(s=1)이면 p = (f−1)/2 → 출력 = 입력.  'valid' = p 0
    return (n + 2 * p - f) // s + 1

def conv_params(f, c_in, c_out, bias=True):
    # Conv2D: 필터 하나 = f×f×c_in 가중치, 필터가 c_out개 + 편향 c_out
    return f * f * c_in * c_out + (c_out if bias else 0)

# 8→12→8→1: (8·12+12) + (12·8+8) + (8·1+1) = 108 + 104 + 9
print(dense_params(8, 12) + dense_params(12, 8) + dense_params(8, 1))   # 221 (6강 Pima)
# 4→12→8→3: 60 + 104 + 27
print(dense_params(4, 12) + dense_params(12, 8) + dense_params(8, 3))   # 191 (6강 Iris)
print(conv_out(28, 3), conv_out(28, 3, p=1), conv_out(28, 2, s=2))      # 26, 28, 14
# 3·3·1·32+32 = 320,  3·3·32·64+64 = 18496
print(conv_params(3, 1, 32), conv_params(3, 32, 64))                    # 320, 18496
# 참고: MaxPooling, Flatten, Dropout은 파라미터 0. BatchNormalization은 채널당 4개
#      (γ, β 학습 + 이동평균·이동분산 학습 안 함 → summary의 Non-trainable)
```

[⬆ 목차로 돌아가기](#목차)

### 13-9. 초기화 표준편차 (8강)

**검색 키워드:** 가중치 초기화, He 초기화, Xavier, Glorot, LeCun, 표준편차, fan_in, fan_out, weight initialization

```python
fan_in, fan_out = 128, 128             # 층의 입력 개수, 출력 개수
std_lecun  = np.sqrt(1 / fan_in)                   # LeCun: √(1/fan_in)  (SELU 등)
# Xavier(Glorot) normal: √(2/(fan_in+fan_out))  (sigmoid, tanh). 8강 노트북은 sqrt(1/fan_in)
std_xavier = np.sqrt(2 / (fan_in + fan_out))
std_he     = np.sqrt(2 / fan_in)                   # He: √(2/fan_in)  (ReLU 계열)
# 평균 0, 표준편차 std_he 정규분포로 (128, 128) 가중치 생성 → 실제 표준편차 확인
W = np.random.default_rng(0).normal(0, std_he, size=(fan_in, fan_out))
print(std_lecun, std_xavier, std_he, W.std())      # 0.0884 0.0884 0.125 ≈0.125
# Keras: Dense(64, kernel_initializer='he_normal')  / 'glorot_uniform'(기본값) / 'lecun_normal'
```

[⬆ 목차로 돌아가기](#목차)

### 13-10. 확률, 통계 계산 (2강)

**검색 키워드:** 베이즈 정리, 조건부 확률, 기댓값, 분산, 공분산, 노름, 행렬식, 역행렬, 고윳값, Bayes, expectation, variance, covariance, norm, determinant, inverse, eigenvalue

```python
# 베이즈 정리: P(D|+) = P(+|D)P(D) / P(+)
#   P(+) = P(+|D)P(D) + P(+|not D)P(not D)   (전체 확률의 법칙)
P_D, P_pos_D, P_pos_notD = 0.01, 0.99, 0.05     # 유병률, 민감도, 위양성률
P_pos = P_pos_D * P_D + P_pos_notD * (1 - P_D)  # 0.0099 + 0.0495 = 0.0594
print('P(D|+) =', P_pos_D * P_D / P_pos)           # 0.1667

# 기댓값, 분산, 공분산, 노름
xv = np.array([1, 2, 3, 4, 5, 6])                  # 주사위 눈
p = np.ones(6) / 6                                 # 각 확률 1/6
print('E[X] =', np.sum(xv * p))                    # E[X] = Σ x·p(x) = 3.5
# np.var/np.std: n으로 나눔 (모분산). np.cov: n−1로 나눔 (표본 공분산 행렬 2×2)
print(np.var(xv), np.std(xv), np.cov(xv, 2 * xv))
v = np.array([3., 4., -2.])
# L1 = Σ|v| = 9,  L2 = √(9+16+4) = 5.385,  L∞ = max|v| = 4
print(np.linalg.norm(v, 1), np.linalg.norm(v), np.linalg.norm(v, np.inf))   # 9, 5.385, 4
M = np.array([[1., 2.], [3., 4.]])
# 프로베니우스 노름 √Σm² = √30 = 5.477,  det = 1·4 − 2·3 = −2,  역행렬 = [[−2, 1], [1.5, −0.5]]
print(np.linalg.norm(M, 'fro'), np.linalg.det(M), np.linalg.inv(M))
vals, vecs = np.linalg.eig(np.array([[2., 1.], [1., 2.]]))   # 고윳값 3, 1 (vecs 열 = 고유벡터)
print(vals)
```

[⬆ 목차로 돌아가기](#목차)

### 13-11. 합성곱·풀링 직접 구현 (12강)

**사용처:** "합성곱 결과를 직접 계산하시오", "stride/padding에 따른 출력을 구하시오", "max pooling 결과는?"
**목적:** Conv2D, MaxPooling2D가 내부에서 하는 계산을 NumPy로 확인
**검색 키워드:** 합성곱 직접 구현, 컨볼루션, 필터, 커널, 스트라이드, 패딩, 제로 패딩, 맥스풀링, 평균풀링, 전역 평균 풀링, 1x1 합성곱, 다채널, convolution, conv2d, kernel, stride, zero padding, max pooling, average pooling, global average pooling, 1x1 conv

```python
import numpy as np

# (1) 2D 합성곱 (딥러닝의 conv = 필터를 뒤집지 않는 cross-correlation)
#   출력[i, j] = Σ (입력 조각 ⊙ 필터) + b    (⊙ = 원소별 곱, Hadamard)
#   출력 크기 = ⌊(n + 2p − f)/s⌋ + 1
def conv2d(x, w, b=0.0, stride=1, pad=0):
    x = np.pad(x, pad)                          # 상하좌우에 0을 pad칸씩 붙임 (zero padding)
    kh, kw = w.shape                            # 필터 높이, 너비
    oh = (x.shape[0] - kh) // stride + 1        # 출력 높이 (이미 패딩한 x 기준이라 2p 불필요)
    ow = (x.shape[1] - kw) // stride + 1        # 출력 너비
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            r, c = i * stride, j * stride       # 현재 조각의 왼쪽 위 위치 (stride만큼 점프)
            patch = x[r:r + kh, c:c + kw]       # 필터 크기만큼 잘라냄
            out[i, j] = np.sum(patch * w) + b   # 원소별 곱 → 전부 더함 → 편향
    return out

img = np.zeros((6, 6))
img[:, 3:] = 9                                  # 왼쪽 3열 0, 오른쪽 3열 9 → 세로 경계
w_edge = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])                 # 세로 경계 검출 필터 (왼쪽 − 오른쪽)
print(conv2d(img, w_edge))                      # (4, 4): 경계 열에서 −27 (= (0 − 9)×3)
print(conv2d(img, w_edge, stride=2))            # (2, 2): 한 칸씩 건너뜀
print(conv2d(img, w_edge, pad=1).shape)         # (6, 6): pad=1이면 3×3 필터에서 크기 유지
```

```python
# (2) 풀링: 학습 파라미터 없음. 영역마다 대표값 하나
def pool2d(x, size=2, stride=2, mode='max'):
    oh = (x.shape[0] - size) // stride + 1
    ow = (x.shape[1] - size) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            r, c = i * stride, j * stride
            region = x[r:r + size, c:c + size]
            out[i, j] = region.max() if mode == 'max' else region.mean()   # 최댓값 or 평균
    return out

a = np.arange(1, 17).reshape(4, 4)              # [[1..4],[5..8],[9..12],[13..16]]
print(pool2d(a, mode='max'))                    # [[ 6.  8.] [14. 16.]]
print(pool2d(a, mode='avg'))                    # [[ 3.5  5.5] [11.5 13.5]]
print(a.mean())                                 # 전역 평균 풀링(GAP): 채널 하나 → 숫자 하나 8.5
```

```python
# (3) 다채널 입력: 필터도 입력 채널 수만큼 깊이를 가짐 (k, k, C)
#   출력[i, j] = Σ_채널 Σ_조각 (x ⊙ w) + b   → 필터 1개당 출력 채널 1개
def conv2d_multi(x, w, b=0.0):
    k = w.shape[0]
    oh, ow = x.shape[0] - k + 1, x.shape[1] - k + 1     # stride 1, padding 없음
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(x[i:i + k, j:j + k, :] * w) + b   # (k,k,C) ⊙ (k,k,C) → 합
    return out

rng = np.random.default_rng(0)
x3 = rng.integers(0, 5, size=(5, 5, 3)).astype(float)   # 5×5 RGB 같은 3채널 입력
w3 = rng.integers(-1, 2, size=(3, 3, 3)).astype(float)  # 3×3×3 필터 1개
print(conv2d_multi(x3, w3).shape)               # (3, 3)
# 검증: 채널별 2D 합성곱 결과를 더한 것과 같음
check = sum(conv2d(x3[:, :, c], w3[:, :, c]) for c in range(3))
print(np.allclose(conv2d_multi(x3, w3), check)) # True

# (4) 1×1 합성곱 = 픽셀마다 채널 벡터에 (C_in → C_out) 행렬을 곱하는 것 (공간 크기 그대로)
x = rng.integers(0, 5, size=(4, 4, 3)).astype(float)    # (H, W, C_in=3)
W = np.array([[1.0, 0.0],
              [0.5, 1.0],
              [0.0, -1.0]])                     # (C_in=3, C_out=2)
b = np.array([0.0, 1.0])
y = x @ W + b                                   # 마지막 축끼리 곱: (4,4,3) @ (3,2) → (4,4,2)
print(y.shape)                                  # (4, 4, 2)
print('1x1 conv 파라미터 수:', W.size + b.size) # (C_in + 1) × C_out = 8
```

[⬆ 목차로 돌아가기](#목차)

---

## 14. 기출 유형별 완성 템플릿

> 각 템플릿은 **[0. 기본 import 세트](#0-기본-import-세트)를 실행한 뒤** 셀 하나에 붙여 넣으면 처음부터 끝까지 돌아갑니다.
> 데이터는 수업 GitHub URL에서 바로 읽습니다. Drive를 쓰려면 주석 처리된 줄로 바꾸세요.
> 문제에서 주는 조건(분할 비율, 층 구조, epoch, patience 등)으로 숫자만 바꾸면 됩니다.

[⬆ 목차로 돌아가기](#목차)

### 14-A. 범용 회귀 (CSV → 결측 → 범주형 → 스케일링 → MLP)

**검색 키워드:** 회귀 템플릿, regression template, 다른 데이터셋 회귀, 결측 채우기, 원핫, OneHotEncoder, StandardScaler, y 표준화, inverse_transform, MAE, RMSE, R2, 예측 vs 실제

**사용처:** "주어진 데이터셋으로 회귀 모델을 학습하고 평가하시오" (기출: 다른 데이터셋으로 회귀)
**바꿀 곳:** `DATA`, `TARGET`, `DROP_COLS`

```python
# [0] 재현성: numpy, random, tf 시드를 한 번에 고정 (실행할 때마다 결과가 같아짐)
tf.keras.utils.set_random_seed(42)

BASE = 'https://raw.githubusercontent.com/dearyonghoon/course-ai-programming-2026/main/data/'
DATA = BASE + 'ames_train.csv'
# Drive를 쓸 때: DATA = '/content/drive/MyDrive/Colab Notebooks/data/ames_train.csv'
TARGET = 'SalePrice'                # 예측할 열 (y)
DROP_COLS = ['Id']                  # 학습에 쓰지 않을 열 (ID, 날짜 등). 없으면 []

# [1] 로드, X/y 분리
df = pd.read_csv(DATA)
print(df.shape)                                       # (행 수, 열 수)
X = df.drop(columns=DROP_COLS + [TARGET]).copy()      # 입력: 목표와 불필요한 열 제외
y = df[TARGET].copy()                                 # 목표: Series (N,)

# [2] 6:2:2 분할
#   1차: 전체의 20%를 test로 → 남은 80%(X_tv)
#   2차: 남은 80% 중 25%를 val로 → 전체의 0.8 × 0.25 = 20%
#   (두 번째 test_size는 '남은 데이터 기준'이라는 점 주의)
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25,
                                                  random_state=42)
# .copy(): 뒤에서 값을 채울 때 SettingWithCopyWarning 방지
X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
print(len(X_train), len(X_val), len(X_test))          # 대략 6 : 2 : 2

# [3] 결측치: train 통계로 채우기 (수치형 중앙값, 범주형 최빈값)
#     val/test 값으로 통계를 내면 정보 누수 → 반드시 X_train에서만 계산
num_cols = X_train.select_dtypes(include='number').columns   # 숫자 열 이름 목록
cat_cols = X_train.select_dtypes(exclude='number').columns   # 문자열(범주) 열 이름 목록
num_fill = X_train[num_cols].median()                        # 열별 중앙값 (Series)
# mode()는 DataFrame을 돌려주므로 .iloc[0]으로 첫 행(= 열별 최빈값)만 꺼냄
cat_fill = X_train[cat_cols].mode().iloc[0] if len(cat_cols) > 0 else None
for data in [X_train, X_val, X_test]:                 # 세 데이터에 같은 값으로 채움
    data[num_cols] = data[num_cols].fillna(num_fill)
    if len(cat_cols) > 0:
        data[cat_cols] = data[cat_cols].fillna(cat_fill)
print('missing:',
      X_train.isnull().sum().sum(),                   # 세 값 모두 0이어야 정상
      X_val.isnull().sum().sum(),
      X_test.isnull().sum().sum())

# [4] 수치형 표준화 + 범주형 원핫 (train으로만 fit, val/test는 transform만)
scaler = StandardScaler()
parts = {'train': [scaler.fit_transform(X_train[num_cols])],   # 평균/표준편차 학습 + 변환
         'val':   [scaler.transform(X_val[num_cols])],         # train 기준으로 변환만
         'test':  [scaler.transform(X_test[num_cols])]}
if len(cat_cols) > 0:
    # handle_unknown='ignore': train에 없던 범주가 val/test에 나와도 오류 없이 전부 0
    # sparse_output=False: numpy 배열로 받기 (sklearn 1.2 미만은 sparse=False)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # astype(str): 숫자/문자 섞인 열도 문자열로 통일해야 오류가 안 남
    parts['train'].append(encoder.fit_transform(X_train[cat_cols].astype(str)))
    parts['val'].append(encoder.transform(X_val[cat_cols].astype(str)))
    parts['test'].append(encoder.transform(X_test[cat_cols].astype(str)))
# 수치형 블록과 원핫 블록을 옆으로(axis=1) 붙여 최종 입력 행렬 생성, Keras용 float32
X_train_in = np.concatenate(parts['train'], axis=1).astype('float32')
X_val_in   = np.concatenate(parts['val'], axis=1).astype('float32')
X_test_in  = np.concatenate(parts['test'], axis=1).astype('float32')
print('input dim:', X_train_in.shape[1])              # Input(shape=...)에 들어갈 특성 수

# [5] y 표준화 (가격처럼 값이 크면 loss/gradient가 너무 커짐)
#     스케일러는 2차원 입력만 받음 → reshape(-1, 1) 후 ravel()로 다시 (N,)
y_scaler = StandardScaler()
y_train_s = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
y_val_s   = y_scaler.transform(y_val.to_numpy().reshape(-1, 1)).ravel()

# [6] 모델 (과적합 억제: Dropout + EarlyStopping)
model = Sequential([
    Input(shape=(X_train_in.shape[1],)),              # 특성 수를 직접 쓰지 말고 shape에서 읽기
    Dense(64, activation='relu', kernel_initializer='he_normal'),  # ReLU엔 He 초기화
    Dropout(0.1),                                     # 학습 때만 10% 뉴런을 끔
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.1),
    Dense(1),                                         # 회귀 출력: 1개, 활성화 없음(linear)
])
model.compile(loss='mse', optimizer=Adam(learning_rate=2e-4), metrics=['mae'])
# monitor 이름은 metrics 이름 앞에 'val_'를 붙인 것 ('val_mae'), 작을수록 좋으니 mode='min'
early_stopping = EarlyStopping(monitor='val_mae', patience=20, mode='min',
                               restore_best_weights=True)   # 가장 좋았던 가중치로 복원
history = model.fit(X_train_in, y_train_s,
                    validation_data=(X_val_in, y_val_s),    # 미리 나눈 val 사용
                    epochs=300, batch_size=32,
                    callbacks=[early_stopping], verbose=0)
print('epochs run:', len(history.history['loss']),          # 실제로 돈 epoch 수
      ' best epoch:', int(np.argmin(history.history['val_mae'])) + 1)  # 0부터라 +1

# [7] 학습곡선 (원래 단위 MAE)
#     표준화 공간의 MAE × y 표준편차(scale_[0]) = 원래 단위 MAE
plt.plot(np.array(history.history['mae']) * y_scaler.scale_[0], label='Train MAE')
plt.plot(np.array(history.history['val_mae']) * y_scaler.scale_[0],
         label='Validation MAE')
plt.xlabel('Epoch'); plt.ylabel('MAE (original unit)')
plt.legend(); plt.grid(alpha=0.3); plt.show()

# [8] 테스트 평가 (원래 단위로 복원)
#     predict 결과는 (N, 1) → inverse_transform → ravel()로 (N,)
#     ravel()을 안 하면 y_test(N,)와 (N,1)이 브로드캐스팅되어 지표가 틀어짐
y_pred = y_scaler.inverse_transform(model.predict(X_test_in, verbose=0)).ravel()
print(f'Test MAE : {mean_absolute_error(y_test, y_pred):,.2f}')
print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}')   # RMSE = √MSE
print(f'Test R2  : {r2_score(y_test, y_pred):.4f}')                     # 1에 가까울수록 좋음

# 예측 vs 실제 산점도: 빨간 점선(y = x) 위에 모일수록 정확
plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
lo, hi = y_test.min(), y_test.max()
plt.plot([lo, hi], [lo, hi], 'r--')
plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.grid(alpha=0.3); plt.show()
```

> **11강 Ames 방식 그대로 하려면:** 3) 결측 처리를 [4-5](#4-5-누수-없이-채우기-11강-ames-방식)로 바꾸고, 분할 전에 `X['MSSubClass'] = X['MSSubClass'].astype(str)`를 넣습니다.

[⬆ 목차로 돌아가기](#목차)

### 14-B. 다중 출력 회귀 (2D ~ 6D 출력)

**검색 키워드:** 다중 출력 회귀, 2차원 출력, multi-output regression, 기구학, kinematics, IRB2400, scaler_y, 출력별 MAE

**사용처:** "두 개의 값을 동시에 예측하는 회귀 모델" (기출: 2차원 출력 회귀), 11강 기구학
**바꿀 곳:** `INPUT_COLS`, `TARGET_COLS`

```python
# [0] 시드 고정 + 데이터 로드
tf.keras.utils.set_random_seed(42)
BASE = 'https://raw.githubusercontent.com/dearyonghoon/course-ai-programming-2026/main/data/'
df = pd.read_csv(BASE + 'datasetIRB2400.csv')       # 30만 행이라 로드에 시간이 조금 걸림
# df = df.sample(50000, random_state=42)        # 시간이 부족하면 일부만 사용

# [1] 입력/출력 열 지정
INPUT_COLS  = ['q1_in', 'q2_in', 'q3_in', 'q4_in', 'q5_in', 'q6_in']   # 관절 각도 6개
TARGET_COLS = ['x', 'y']                        # 2D 출력. 6D면 ['x','y','z','yaw','pitch','roll']
# 역기구학: INPUT_COLS = ['x','y','z','yaw','pitch','roll'], TARGET_COLS = ['q1_out', ..., 'q6_out']

X = df[INPUT_COLS].values                       # (N, 6) numpy 배열
y = df[TARGET_COLS].values                      # (N, 2) → y도 2차원이라 reshape 필요 없음

# [2] 64 : 16 : 20 분할 (test 20% → 남은 80%의 20% = 전체 16%가 val)
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.2,
                                                  random_state=42)

# [3] X, y 각각 표준화 (스케일러 2개! train으로만 fit)
scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_val_s   = scaler_X.transform(X_val)
X_test_s  = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train)     # 열(출력)마다 따로 평균/표준편차
y_val_s   = scaler_y.transform(y_val)
y_test_s  = scaler_y.transform(y_test)

# [4] 모델: 출력층 노드 수 = 목표 열 수, 활성화 없음
model = Sequential([
    Input(shape=(len(INPUT_COLS),)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(TARGET_COLS)),                    # 출력 수 = 목표 열 수, 활성화 없음
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])   # mse는 출력별 평균까지 냄
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
history = model.fit(X_train_s, y_train_s,
                    validation_data=(X_val_s, y_val_s),
                    epochs=50, batch_size=512,           # 데이터가 많아 배치를 크게
                    callbacks=[early_stopping], verbose=1)

# [5] 학습곡선
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(True); plt.show()

# [6] 평가: 표준화 공간 (X도 반드시 스케일링한 것)
test_mse, test_mae = model.evaluate(X_test_s, y_test_s, verbose=0)   # [loss, mae] 순서
print(f'Test MSE (std): {test_mse:.4f}, Test MAE (std): {test_mae:.4f}')

# [7] 원래 단위 복원 + 출력별 MAE
y_pred = scaler_y.inverse_transform(
    model.predict(X_test_s, batch_size=512, verbose=0))   # (N, 2) 원래 단위
for i, name in enumerate(TARGET_COLS):                    # i번째 열 = i번째 출력
    print(f'MAE for {name:>5}: {mean_absolute_error(y_test[:, i], y_pred[:, i]):.3f}')

# [8] 출력별 예측 vs 실제 (점이 많아 앞 2000개만)
fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(5 * len(TARGET_COLS), 4))
for i, ax in enumerate(np.atleast_1d(axes)):              # 출력 1개여도 반복 가능하게
    ax.scatter(y_test[:2000, i], y_pred[:2000, i], s=3, alpha=0.4)
    lo, hi = y_test[:, i].min(), y_test[:, i].max()
    ax.plot([lo, hi], [lo, hi], 'r--'); ax.set_title(TARGET_COLS[i])   # y = x 기준선
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
plt.tight_layout(); plt.show()
```

> **11강 노트북 버그 주의:** 수업 노트북(L11_04)은 평가 셀에서 스케일링 안 한 `X_test`를 넣고, 그래프 셀에서 정의되지 않은 `y_true`를 씁니다. 위 템플릿처럼 `X_test_s`, `y_test`를 쓰세요.

**임의의 두 수치 열로 2D 회귀 (데이터 이름만 바꿔 쓰는 버전):**

```python
# 예: ETTh1에서 나머지 5개 변수로 OT와 HUFL 두 값을 동시에 예측
df = pd.read_csv(BASE + 'ETTh1.csv')
TARGET_COLS = ['OT', 'HUFL']                          # 동시에 맞힐 두 열
X = df.drop(columns=['date'] + TARGET_COLS).values    # 날짜(문자열)와 목표 열 제외 → (N, 5)
y = df[TARGET_COLS].values                            # (N, 2)
# 이후는 위 템플릿의 분할 → 스케일링 → 모델 → 평가 그대로
# (모델 Input은 X.shape[1], 출력층은 Dense(len(TARGET_COLS)))
```

[⬆ 목차로 돌아가기](#목차)

### 14-C. 이진 분류 (Pima) + 혼동행렬

**검색 키워드:** 이진 분류, binary classification, Pima, 당뇨, sigmoid, binary_crossentropy, 혼동행렬, confusion matrix, TN FP FN TP, classification_report, stratify, 임계값 0.5

```python
# [0] 시드 + 데이터 로드
tf.keras.utils.set_random_seed(42)
BASE = 'https://raw.githubusercontent.com/dearyonghoon/course-ai-programming-2026/main/data/'
df = pd.read_csv(BASE + 'pima-indians-diabetes3.csv')   # 768 x 9
X = df.iloc[:, 0:8]                          # 0~7번 열: 입력 8개 (끝 번호 8은 미포함)
y = df.iloc[:, 8]                            # 8번 열 'diabetes': 0/1 라벨
print(y.value_counts())                      # 0이 약 500, 1이 약 268 → 불균형

# [1] 8:2 분할, stratify=y: train/test의 0/1 비율을 원본과 같게 유지
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# [2] 표준화 (train으로만 fit) → 결과는 numpy 배열
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# [3] 모델: 이진 분류 출력 = Dense(1, sigmoid) + binary_crossentropy
model = Sequential([
    Input(shape=(8,)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid'),          # 출력 = 양성(1)일 확률 0~1
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    validation_split=0.25,    # train의 뒤쪽 25%를 자동으로 검증용
                    epochs=200, batch_size=10,
                    callbacks=[early_stopping], verbose=0)
print('epochs run:', len(history.history['loss']))

# [4] 테스트 평가: compile에 metrics를 1개 줬으니 [loss, accuracy] 2개가 나옴
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

# [5] 학습곡선
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3); plt.show()

# [6] 혼동행렬
y_prob = model.predict(X_test, verbose=0).ravel()   # (N,1) → (N,) 확률
y_pred = (y_prob >= 0.5).astype(int)                # 임계값 0.5로 0/1 결정
cm = confusion_matrix(y_test, y_pred)               # 행 = 실제, 열 = 예측
print(cm)
tn, fp, fn, tp = cm.ravel()                         # 2x2를 펴면 [TN, FP, FN, TP] 순서
print(f'TN={tn}, FP={fp}, FN={fn}, TP={tp}')
ConfusionMatrixDisplay(cm, display_labels=['negative', 'positive']).plot(cmap='Blues')
plt.show()
# precision, recall, f1을 클래스별로 한 번에 출력
print(classification_report(y_test, y_pred, digits=4))
```

[⬆ 목차로 돌아가기](#목차)

### 14-D. 다중 분류 (Iris) + 혼동행렬

**검색 키워드:** 다중 분류, multiclass, Iris, softmax, categorical_crossentropy, LabelEncoder, to_categorical, 원핫, argmax, 혼동행렬, confusion matrix, target_names

```python
# [0] 시드 + 데이터 로드
tf.keras.utils.set_random_seed(42)
BASE = 'https://raw.githubusercontent.com/dearyonghoon/course-ai-programming-2026/main/data/'
df = pd.read_csv(BASE + 'iris3.csv')             # 150 x 5, 4번 열 = 품종 문자열
X = df.iloc[:, 0:4].values                       # (150, 4) numpy

# [1] 라벨: 문자열 → 정수 → 원핫
le = LabelEncoder()
y_int = le.fit_transform(df.iloc[:, 4])          # 0, 1, 2 (알파벳 순서), le.classes_에 이름 저장
y = to_categorical(y_int, 3)                     # 원핫 (N, 3)

# [2] 8:2 분할: stratify에는 원핫이 아니라 정수 라벨(y_int)을 넣는 게 안전
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_int)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# [3] 모델: 출력 노드 = 클래스 수(3), softmax + categorical_crossentropy (원핫 라벨)
model = Sequential([
    Input(shape=(4,)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax'),              # 세 확률의 합 = 1
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
    X_train, y_train,
    validation_split=0.25, epochs=200, batch_size=10,
    callbacks=[EarlyStopping(monitor='val_loss', patience=20,
                             restore_best_weights=True)],
    verbose=0)
print(model.evaluate(X_test, y_test, verbose=0))   # [loss, accuracy]

# [4] 혼동행렬: 확률 (N,3) → argmax로 클래스 번호, 원핫 정답도 argmax로 되돌림
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)            # 3x3, 대각선 = 맞힌 개수
print(cm)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(cmap='Blues',
                                                            xticks_rotation=45)
plt.show()
print(classification_report(y_true, y_pred, target_names=le.classes_))
```

> 정수 라벨 버전: `y = y_int`, 출력 그대로, `loss='sparse_categorical_crossentropy'`, `y_true = y_test`

[⬆ 목차로 돌아가기](#목차)

### 14-E. CNN (MNIST 분류)

**검색 키워드:** CNN, 합성곱, Conv2D, MaxPooling2D, MNIST, Fashion-MNIST, CIFAR-10, 이미지 분류, reshape 28 28 1, /255, BatchNormalization, ModelCheckpoint, ReduceLROnPlateau, 혼동행렬

```python
# [0] 시드 + 데이터 로드
tf.keras.utils.set_random_seed(42)
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()   # (60000,28,28), 라벨 0~9 정수

# [1] 전처리: float32 → 채널 축 추가 (N,28,28,1) → 0~1 정규화
X_train = X_train.astype('float32').reshape(-1, 28, 28, 1) / 255.0
X_test  = X_test.astype('float32').reshape(-1, 28, 28, 1) / 255.0
y_train_oh = to_categorical(y_train, 10)       # (N,) → (N, 10) 원핫
y_test_oh  = to_categorical(y_test, 10)

# [2] 모델: Conv(bias 없음) → BN → ReLU 블록 2개 → Pool → Flatten → Dense
model = Sequential([
    Input(shape=(28, 28, 1)),
    # padding='same': 출력 크기 유지(28x28). BN이 평균을 빼므로 Conv bias는 꺼도 됨
    Conv2D(32, (3, 3), padding='same', use_bias=False), BatchNormalization(), ReLU(),
    Conv2D(64, (3, 3), padding='same', use_bias=False), BatchNormalization(), ReLU(),
    MaxPooling2D((2, 2)),                      # 28x28 → 14x14
    Flatten(),                                 # 14*14*64 = 12544 길이 벡터로 펴기
    Dense(128, activation='relu'),
    Dropout(0.3),                              # 과적합 억제
    Dense(10, activation='softmax'),           # 10 클래스 확률
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()                                # 층별 출력 shape, 파라미터 수 확인

# [3] 콜백: 조기 종료 + 최고 모델 저장 + 정체 시 학습률 절반
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('mnist_cnn_best.keras', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
]
history = model.fit(X_train, y_train_oh,
                    validation_split=0.25,     # train의 25%를 검증용으로
                    epochs=10, batch_size=128,
                    callbacks=callbacks, verbose=1)

# [4] 테스트 평가
test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

# [5] 학습곡선: loss와 accuracy를 나란히
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, key in zip(axes, ['loss', 'accuracy']):
    ax.plot(history.history[key], label='Train')
    ax.plot(history.history['val_' + key], label='Validation')   # 'val_loss', 'val_accuracy'
    ax.set_title(key); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# [6] 혼동행렬: 정답은 원래 정수 라벨 y_test 사용 (원핫 아님)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)          # 10x10
ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(cmap='Blues')
plt.show()
```

> **Fashion-MNIST:** `mnist` → `fashion_mnist` 로만 바꾸면 됩니다. (기출 항목을 전부 넣은 버전은 [14-I](#14-i-fashion-mnist-cnn-종합-기출-항목-전부))
> **CIFAR-10:** `cifar10.load_data()`, reshape 없이 `/255.0`, `Input(shape=(32, 32, 3))`, 라벨은 `to_categorical(y_train, 10)` (y가 (N,1)이어도 됨). 혼동행렬에는 `y_test.ravel()`
> **시간이 부족하면:** `X_train[:10000]`처럼 일부만 학습

[⬆ 목차로 돌아가기](#목차)

### 14-F. 시계열 예측 (ETTh1: 과거 96 → 미래 24)

**검색 키워드:** 시계열, time series, ETTh1, OT, 슬라이딩 윈도우, window, 과거 96 미래 24, Last Value, Seasonal Naive, 기준선, baseline, 잔차 예측, residual, 시간순 분할, shuffle=False

```python
# [0] 시드 + 데이터 로드
tf.keras.utils.set_random_seed(42)
BASE = 'https://raw.githubusercontent.com/dearyonghoon/course-ai-programming-2026/main/data/'
df = pd.read_csv(BASE + 'ETTh1.csv')                   # 17420 x 8, 1시간 간격
ot = df[['OT']].to_numpy(dtype=np.float32)             # (N, 1) 2차원 (스케일러용)
INPUT_LEN, PRED_LEN = 96, 24                           # 과거 96시간 → 미래 24시간

# [1] 시간순 70 / 15 / 15 (섞으면 미래 정보 누수), 스케일러는 train 구간만으로 fit
n = len(ot); train_end, val_end = int(n * 0.70), int(n * 0.85)
sc = StandardScaler().fit(ot[:train_end])
ot_s = sc.transform(ot).astype(np.float32)            # 전체를 train 기준으로 변환

# [2] 슬라이딩 윈도우: X[i] = 과거 96개 (96,1), Y[i] = 그다음 24개 (24,)
def windows(series, start, end):
    X, Y = [], []
    # 마지막 윈도우도 목표 24개를 다 가질 수 있는 위치까지
    for i in range(start, end - INPUT_LEN - PRED_LEN + 1):
        X.append(series[i:i + INPUT_LEN])                              # (96, 1)
        Y.append(series[i + INPUT_LEN:i + INPUT_LEN + PRED_LEN, 0])    # (24,)
    return np.array(X), np.array(Y)

# 목표 구간이 각 분할 안에 들어가도록 입력은 앞 구간과 겹치게 시작
# (val의 첫 목표가 train_end부터 시작하려면 입력은 train_end - 96부터 필요)
X_train, y_train = windows(ot_s, 0, train_end)
X_val, y_val     = windows(ot_s, train_end - INPUT_LEN, val_end)
X_test, y_test   = windows(ot_s, val_end - INPUT_LEN, n)
print(X_train.shape, y_train.shape, X_val.shape, X_test.shape)   # (N, 96, 1), (N, 24)

# [3] MLP: (96,1)을 Flatten으로 96개 벡터로 펴서 Dense, 출력 24개를 한 번에 (Direct)
model = Sequential([
    Input(shape=(INPUT_LEN, 1)), Flatten(),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dense(PRED_LEN),                                   # 미래 24개, 활성화 없음
])
model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=150, batch_size=64,
    shuffle=False,                                     # 수업 방식: 순서 유지
    callbacks=[EarlyStopping(monitor='val_loss', patience=15,
                             restore_best_weights=True)],
    verbose=0)

# [4] 표준화 값 → °C로 되돌리는 함수 (어떤 shape이든 펴서 변환 후 원래 shape로)
def inv(a):                                     # 표준화 → °C
    return sc.inverse_transform(a.reshape(-1, 1)).reshape(a.shape)

# [5] 모델 vs 기준선 비교 (모델이 기준선보다 나빠야 할 이유가 없음)
pred = model.predict(X_test, verbose=0)                           # (N, 24)
last_value = np.repeat(X_test[:, -1, :], PRED_LEN, axis=1)       # 기준선: 마지막 값 유지
seasonal = X_test[:, -PRED_LEN:, 0]                               # 기준선: 24시간 전 값
for name, p in [('MLP', pred), ('Last Value', last_value), ('Seasonal Naive', seasonal)]:
    mse_c = mean_squared_error(inv(y_test).ravel(), inv(p).ravel())    # °C² 단위
    mae_c = mean_absolute_error(inv(y_test).ravel(), inv(p).ravel())   # °C 단위
    print(f'{name:15s} MSE(°C²)={mse_c:.4f}  MAE(°C)={mae_c:.4f}')

# 잔차 예측 (11강): 목표를 y − 마지막값으로 바꾸고, 예측 시 마지막값을 더함
# r_train = y_train - np.repeat(X_train[:, -1, :], PRED_LEN, axis=1)
# 출력층: Dense(PRED_LEN, kernel_initializer='zeros')  → 학습 시작 시 Last Value와 같은 예측
# y_hat = np.repeat(X_test[:, -1, :], PRED_LEN, axis=1) + model.predict(X_test)

# [6] 예측 예시 그림: 과거 96시간(음수 시간) + 실제/예측 24시간
plt.figure(figsize=(10, 4))
plt.plot(range(-INPUT_LEN, 0), inv(X_test[0, :, 0]), label='History')
plt.plot(range(PRED_LEN), inv(y_test[0]), marker='o', label='Actual')
plt.plot(range(PRED_LEN), inv(pred[0]), marker='o', label='Predicted')
plt.axvline(0, ls='--'); plt.legend(); plt.grid(True); plt.show()
```

[⬆ 목차로 돌아가기](#목차)

### 14-G. numpy 랜덤 게임 시뮬레이션

**검색 키워드:** numpy 랜덤, 난수, random, 주사위, dice, 게임 시뮬레이션, np.random.seed, randint, 조건 분기, 확률 추정, 기대값, np.select, 벡터화

**사용처:** "난수로 게임을 N번 시뮬레이션하고 조건에 따른 결과를 집계하시오" (기출 1번 유형)

```python
def play_game(n_games=10, seed=42, verbose=True):
    """주사위 2개: 합이 7 이상이면 승리(+1), 같은 눈(더블)이면 보너스(+2), 그 외 패배(-1)"""
    np.random.seed(seed)                     # 같은 seed → 같은 결과 (재현성)
    rows = []                                # 게임마다 딕셔너리 1개씩 모아서 DataFrame으로
    for g in range(1, n_games + 1):          # 1번 게임부터 n_games번 게임까지
        d1, d2 = np.random.randint(1, 7), np.random.randint(1, 7)     # 1~6 (7은 미포함)
        s = d1 + d2
        # 조건 순서가 중요: 더블을 먼저 검사해야 (6,6)처럼 합 7 이상인 더블도 보너스
        if d1 == d2:
            result, point = 'double', 2
        elif s >= 7:
            result, point = 'win', 1
        else:
            result, point = 'lose', -1
        rows.append({'game': g, 'd1': d1, 'd2': d2, 'sum': s,
                     'result': result, 'point': point})
        if verbose:
            # :2d = 2칸 정수, :6s = 6칸 문자열, :+d = 부호 붙인 정수
            print(f'Game {g:2d}: {d1} + {d2} = {s:2d} → {result:6s} ({point:+d})')
    log = pd.DataFrame(rows)                 # 한 행 = 한 게임
    print('\n총점:', log['point'].sum())
    print(log['result'].value_counts())      # 결과별 횟수
    return log

log = play_game(10)

# 대량 시뮬레이션으로 확률 추정 (반복문 없이 배열 한 번에 = 벡터화)
np.random.seed(0)
N = 100000
d = np.random.randint(1, 7, size=(N, 2))     # (N, 2): 각 행 = 주사위 2개
s = d.sum(axis=1)                            # 행별 합 (N,)
# np.select: 조건 목록을 앞에서부터 검사, 처음 참인 조건의 값을 사용, 아무것도 아니면 default
point = np.select([d[:, 0] == d[:, 1], s >= 7], [2, 1], default=-1)
print('double 확률:', np.mean(d[:, 0] == d[:, 1]), '(이론 1/6 = 0.1667)')   # True 비율
print('게임당 기대 점수:', point.mean())
```

[⬆ 목차로 돌아가기](#목차)

### 14-H. 조기 종료 + 체크포인트 + 학습곡선 (6강 Wine 전체)

**검색 키워드:** 조기 종료, EarlyStopping, ModelCheckpoint, save_best_only, load_model, 최고 모델, Wine, 와인, 학습곡선, history.history, hist_df, validation_split

```python
# [0] 시드 + 데이터 로드 (헤더 없는 CSV → header=None)
tf.keras.utils.set_random_seed(42)
BASE = 'https://raw.githubusercontent.com/dearyonghoon/course-ai-programming-2026/main/data/'
df = pd.read_csv(BASE + 'wine.csv', header=None)    # 6497 x 13
X = df.iloc[:, 0:12]                                # 0~11번 열: 화학 성분 12개
y = df.iloc[:, 12]                                  # 12번 열: 1=red, 0=white

# [1] 8:2 분할 (stratify로 red/white 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

# [2] 모델 (6강 구조 그대로: 30-12-8-1)
model = Sequential([
    Input(shape=(12,)),
    Dense(30, activation='relu'),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# [3] 콜백 2개
#   EarlyStopping: val_loss가 20 epoch 동안 안 좋아지면 멈춤
#   (6강은 restore_best_weights 없이 → 대신 체크포인트로 최고 모델을 따로 저장)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
modelpath = 'best_wine_model.keras'
# Drive: '/content/drive/MyDrive/Colab Notebooks/models/wine/best_wine_model.keras'
#   ModelCheckpoint: val_loss가 최저일 때만 파일 덮어쓰기 (확장자는 .keras 필수)
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0,
                               save_best_only=True)

# [4] 학습: epochs=2000으로 크게 주고 조기 종료에 맡김
history = model.fit(X_train, y_train,
                    epochs=2000, batch_size=500,
                    validation_split=0.25,          # train의 25%를 검증용 → 전체 6:2:2
                    verbose=0,
                    callbacks=[early_stopping_callback, checkpointer])
print('stopped at epoch', len(history.history['loss']))

# [5] 저장된 최고 모델을 불러와 테스트 평가
best_model = load_model(modelpath)
score = best_model.evaluate(X_test, y_test, verbose=0)   # [loss, accuracy]
print('Test loss:', score[0], 'Test accuracy:', score[1])

# [6] 학습곡선 (6강 방식: 점 그래프)
hist_df = pd.DataFrame(history.history)      # 열: loss, accuracy, val_loss, val_accuracy
x_len = np.arange(len(hist_df))              # epoch 번호 0, 1, 2, ...
plt.plot(x_len, hist_df['val_loss'], 'o', c='red', markersize=2, label='Validation loss')
plt.plot(x_len, hist_df['loss'], 'o', c='blue', markersize=2, label='Train loss')
plt.legend(loc='upper right'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.show()
```

[⬆ 목차로 돌아가기](#목차)

### 14-I. Fashion-MNIST CNN 종합 (기출 항목 전부)

**검색 키워드:** Fashion-MNIST, 패션, 기출 CNN, 0~1 정규화, 검증 분리, 앞 10장, 2x5 그리드, subplots(2, 5), 실시간 증강, data augmentation, RandomFlip, RandomRotation, RandomZoom, ImageDataGenerator, Conv2D, weight decay, AdamW, 과적합 억제, Dropout, EarlyStopping, confusion matrix, 혼동행렬, 가장 헷갈린 클래스

**사용처:** 기출 CNN 종합 문제 (0~1 정규화 · 검증 분리 · 앞 10장 2×5 시각화 · 실시간 증강 2가지 이상 · Conv2D CNN · weight decay(AdamW) · 과적합 억제 · confusion matrix)
**바꿀 곳:** 문제에서 주는 검증 비율, epoch, 증강 종류. CSV로 주면 `[1]`의 주석 처리된 CSV 버전 사용

```python
# [0] 시드 고정
tf.keras.utils.set_random_seed(42)
from tensorflow.keras import layers               # 증강 층(RandomFlip 등)을 쓰기 위해

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   # 라벨 0~9의 이름

# [1] 데이터 로드
from tensorflow.keras.datasets import fashion_mnist
(x_full, y_full), (x_test, y_test) = fashion_mnist.load_data()   # (60000,28,28), (10000,28,28)
# CSV로 주어진 경우 (첫 열 label, 나머지 784열 픽셀):
# df = pd.read_csv('fashion-mnist_train.csv')
# y_full = df['label'].values
# x_full = df.drop(columns='label').values.reshape(-1, 28, 28)

# [2] 0~1 정규화 + CNN 입력 모양 (N, 28, 28, 1)
x_full = x_full.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
print(x_full.shape, x_full.min(), x_full.max())   # (60000, 28, 28, 1) 0.0 1.0

# [3] 훈련에서 검증 데이터 분리 (10%, 클래스 비율 유지)
x_train, x_val, y_train, y_val = train_test_split(
    x_full, y_full, test_size=0.1, stratify=y_full, random_state=42)
print(x_train.shape, x_val.shape)                 # (54000, 28, 28, 1) (6000, 28, 28, 1)
print(np.bincount(y_train))                       # 클래스마다 5400개씩이면 stratify 성공

# [4] 훈련 데이터 앞 10장을 2×5 그리드로 시각화
fig, axes = plt.subplots(2, 5, figsize=(12, 5))   # 2행 5열 = 10칸
for i, ax in enumerate(axes.ravel()):             # 2차원 axes를 1차원으로 펴서 0~9번 칸
    ax.imshow(x_train[i].squeeze(), cmap='gray')  # (28,28,1) → (28,28)로 그려야 함
    ax.set_title(class_names[y_train[i]])         # 라벨 번호 → 이름
    ax.axis('off')                                # 눈금 숨기기
plt.tight_layout(); plt.show()

# [5] 실시간 데이터 증강 (3가지): 모델 안에 넣으면 fit 때만 동작,
#     evaluate/predict 때는 그대로 통과 → 검증·테스트에는 증강이 걸리지 않음
augment = Sequential([
    layers.RandomFlip('horizontal'),              # 좌우 반전 (옷은 좌우가 바뀌어도 같은 클래스)
    layers.RandomRotation(0.05),                  # ±0.05 × 360° ≈ ±18° 회전
    layers.RandomZoom(0.1),                       # ±10% 확대/축소
], name='augment')

# [6] Conv2D 기반 CNN + 과적합 억제 (BatchNorm, Dropout)
model = Sequential([
    Input(shape=(28, 28, 1)),
    augment,                                      # 증강 층이 맨 앞
    Conv2D(32, (3, 3), padding='same', use_bias=False), BatchNormalization(), ReLU(),
    Conv2D(32, (3, 3), padding='same', use_bias=False), BatchNormalization(), ReLU(),
    MaxPooling2D((2, 2)),                         # 28 → 14
    Dropout(0.25),                                # 과적합 억제 1
    Conv2D(64, (3, 3), padding='same', use_bias=False), BatchNormalization(), ReLU(),
    MaxPooling2D((2, 2)),                         # 14 → 7
    Dropout(0.25),
    Flatten(),                                    # 7*7*64 = 3136
    Dense(128, activation='relu'),
    Dropout(0.4),                                 # 과적합 억제 2
    Dense(10, activation='softmax'),              # 10 클래스
])
model.summary()

# [7] weight decay: AdamW (가중치 감쇠를 업데이트 단계에서 따로 적용)
#     라벨이 0~9 정수이므로 sparse_categorical_crossentropy (원핫이면 categorical)
model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# [8] 과적합 억제 3: EarlyStopping + 최고 모델 저장
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('fashion_best.keras', monitor='val_loss', save_best_only=True),
]
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),   # [3]에서 나눈 검증 데이터
                    epochs=30, batch_size=128,
                    callbacks=callbacks, verbose=1)

# [9] 학습곡선 (train vs validation)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, key in zip(axes, ['loss', 'accuracy']):
    ax.plot(history.history[key], label='Train')
    ax.plot(history.history['val_' + key], label='Validation')
    ax.set_title(key); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# [10] 테스트 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

# [11] confusion matrix (10x10): 행 = 실제, 열 = 예측
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)   # 확률 10개 → 가장 큰 번호
cm = confusion_matrix(y_test, y_pred)
print(cm)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap='Blues',
                                                            xticks_rotation=45)
plt.show()

# 클래스별 recall(= 대각선 / 행 합)과 가장 많이 헷갈린 쌍 3개
recall = cm.diagonal() / cm.sum(axis=1)
for name, r in zip(class_names, recall):
    print(f'{name:>12}: recall {r:.3f}')
off = cm.copy()
np.fill_diagonal(off, 0)                          # 맞힌 칸(대각선)은 제외
for idx in np.argsort(off.ravel())[::-1][:3]:     # 큰 값 순서로 3개
    t, p = divmod(idx, 10)                        # 1차원 위치 → (실제 행, 예측 열)
    print(f'실제 {class_names[t]} → 예측 {class_names[p]}: {off[t, p]}개')
```

> **증강을 `ImageDataGenerator`로 하라고 하면 (10강 방식):** 모델에서 `augment` 줄을 빼고
> `datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)` →
> `model.fit(datagen.flow(x_train, y_train, batch_size=128), validation_data=(x_val, y_val), epochs=30, callbacks=callbacks)`. 검증 데이터는 증강하지 않습니다.
> **혼동행렬을 sklearn 없이:** `cm = np.zeros((10, 10), dtype=int)` → `for t, p in zip(y_test, y_pred): cm[t, p] += 1`

[⬆ 목차로 돌아가기](#목차)

---

## 15. 자주 나는 오류와 해결

| 증상 / 오류 메시지 | 원인 | 해결 |
|---|---|---|
| `could not convert string to float` | 문자열 열이 X에 남아 있음 | `pd.get_dummies(X, dtype=int)` 또는 `X.select_dtypes('number')` |
| `Unsupported object type float/bool` (Tensor 변환 실패) | X에 object/bool 열이 섞임 | `X = X.astype('float32')`, get_dummies에 `dtype=int` |
| loss가 `nan` | 결측치가 남았거나 값 범위가 너무 큼 | `X.isnull().sum()` 확인, 스케일링, 학습률 낮추기 |
| `Shapes (None, 3) and (None, 1) are incompatible` | 라벨 형태와 출력/손실 불일치 | [5-1 표](#5-1-라벨-형태--손실-함수-짝-가장-중요)대로 맞추기 |
| `Received a label value of 3 which is outside the valid range of [0, 3)` | sparse CE에서 라벨이 1부터 시작하거나 출력 수 부족 | `y - 1` 또는 `LabelEncoder`, 출력 수 = 클래스 수 |
| `expected axis -1 of input shape to have value 8` | Input 크기와 X 열 수 불일치 | `Input(shape=(X_train.shape[1],))` |
| CNN에서 `expected ndim=4, found ndim=3` | 채널 축 없음 | `X.reshape(-1, 28, 28, 1)` |
| `Expected 2D array, got 1D array` (스케일러) | y가 1차원 | `y.values.reshape(-1, 1)` |
| `unexpected keyword argument 'sparse_output'` | sklearn 1.2 미만 | `OneHotEncoder(sparse=False)` |
| `KeyError: 'val_accuracy'` | 검증 데이터가 없거나 metrics 이름이 다름 | `validation_split` 지정, `print(history.history.keys())` |
| `The filepath provided must end in .keras` | 체크포인트 확장자 | `'best.keras'` (가중치만이면 `.weights.h5` + `save_weights_only=True`) |
| `could not convert string to float` in `df.corr()` | 문자열 열 포함 | `df.corr(numeric_only=True)` |
| 정확도가 0.65 근처에서 멈춤 (Pima) | 스케일링 안 함, 전부 한 클래스로 예측 | `StandardScaler` 적용, epoch/구조 확인 |
| MAE가 이상하게 큼 | `y_test (N,)` − `y_pred (N,1)` 브로드캐스팅으로 (N,N) 계산 | `y_pred.ravel()` |
| 셀을 다시 실행하면 이어서 학습됨 | 같은 model 객체를 재사용 | 모델 생성 셀부터 다시 실행, `tf.keras.backend.clear_session()` |
| `SettingWithCopyWarning` | 슬라이스한 DataFrame 수정 | `.copy()` 붙이기 |
| 그래프 한글이 □□로 표시 | Colab에 한글 폰트 없음 | 라벨을 영어로 |
| `FileNotFoundError` (Drive) | 경로 오타, 마운트 안 함 | `drive.mount` 다시, [1-3 URL 방식](#1-3-인터넷-url에서-바로-읽기-drive-없이) 사용 |
| `load_model`에서 `Unknown loss function` | 사용자 정의 손실 사용 | `load_model(path, custom_objects={'함수이름': 함수})` |
| 학습이 너무 오래 걸림 | 데이터가 큼 | `df.sample(50000, random_state=42)`, `batch_size` 키우기, `verbose=0` |
| `imshow`에서 `Invalid shape (28, 28, 1)` | 채널 축이 붙은 이미지 | `plt.imshow(x[i].squeeze(), cmap='gray')` |
| `'AxesSubplot' object is not subscriptable` / 2×5 그리드 반복 오류 | `axes`가 2차원 배열 | `for i, ax in enumerate(axes.ravel()):` |
| 사용자 정의 손실에서 `Graph execution error` / 미분 안 됨 | 함수 안에 `np.` 연산 사용 | `tf.reduce_mean`, `tf.square` 등 `tf.` 연산만 사용 |
| `compile(loss=my_loss())`에서 `missing 2 required positional arguments` | 손실 함수에 괄호를 붙여 호출 | `loss=my_loss` (괄호 없이 함수 이름) |
| sparse CE인데 혼동행렬이 이상함 / `argmax` 오류 | 원핫 정답에 argmax 안 함 (또는 정수 정답에 argmax 함) | 원핫이면 `np.argmax(y_test, axis=1)`, 정수면 그대로 |

[⬆ 목차로 돌아가기](#목차)

---

## 16. 시험 전 체크리스트

**시작할 때**
- [ ] Colab 새 노트북, **런타임 유형** 확인 (CNN이면 GPU가 빠름)
- [ ] AI 기능 끄기: **도구 → 설정 → AI 지원** → '생성형 AI 기능 숨기기' 체크, '인라인 완성 표시'(회색 글씨 자동완성) 체크 해제
- [ ] [0. 기본 import 세트](#0-기본-import-세트) 실행
- [ ] 데이터 로드 후 `df.shape`, `df.head()`, `df.isnull().sum()`, `df.dtypes` 확인

**전처리**
- [ ] 분할 비율: 두 번째 `test_size`는 **남은 데이터 기준**
- [ ] 분류면 `stratify`, 시계열이면 `shuffle=False`
- [ ] 결측치 처리 후 `isnull().sum().sum() == 0` 확인
- [ ] 문자열 열 인코딩 완료
- [ ] 스케일러 `fit_transform`은 **train만**, val/test는 `transform`

**모델**
- [ ] `Input(shape=(X_train.shape[1],))`
- [ ] 출력층 + 손실 짝 ([5-1 표](#5-1-라벨-형태--손실-함수-짝-가장-중요))
- [ ] 회귀 출력층은 활성화 없음, 출력 수 = y 열 수
- [ ] EarlyStopping `monitor` 이름이 metrics와 일치, `restore_best_weights=True`

**평가와 제출**
- [ ] 테스트 평가는 마지막에 `model.evaluate(X_test, y_test)`
- [ ] y를 스케일링했다면 `inverse_transform` 후 원래 단위로 지표 계산
- [ ] 분류: 혼동행렬, 회귀: MAE/RMSE + 예측 vs 실제 그림
- [ ] 학습곡선 (train vs validation) 그리기
- [ ] 문제에서 요구한 **출력(print)** 을 모두 했는지 확인
- [ ] 노트북 저장 (`Ctrl+S`)

**헷갈리기 쉬운 포인트 (최종 정리)**

*공통*
- 스케일러·selector는 **train으로만 fit**, val/test는 `transform`만. `fit_transform` = `fit` + `transform` 한 번에. (SelectKBest는 `fit_transform(X, y)`로 y도 넣음)
- `predict` 결과는 **스케일된 값** → `sy.inverse_transform(pred)` 후 원래 단위로 MAE 계산
- 샘플 1개 예측은 `X[0:1]` (모양 `(1, 6)`, 2차원 유지). `X[0]`은 `(6,)`라 에러
- 열(출력)별 MAE: `np.mean(np.abs(pred - y), axis=0)` → `axis=0`은 세로(샘플) 방향 평균 → 열마다 1개
- `def run(...)` 같은 함수는 같은 코드를 여러 번 돌리려고 묶은 것뿐. 시험에서는 위에서 아래로 펼쳐 써도 됨
  - `run(Q_IN, POSE, batch_size=256)` = 빈칸 `in_cols=Q_IN`, `out_cols=POSE`, `batch_size=256`, 나머지는 기본값으로 채워 한 번 실행

*분할*
- 랜덤 64:16:20 = `train_test_split` 두 번 (`test_size=0.2` → 남은 80%에서 다시 `0.2` = 16%)
- `stratify=y` → 클래스 비율 유지. Fashion-MNIST 60,000개를 9:1 → 54,000 / 6,000, 10클래스라 **클래스당 5,400 / 600** (`np.bincount(y)`로 확인)
- 시계열은 섞지 않음: `train_end = int(n*0.70)`, `val_end = int(n*0.85)` → 앞 70% train, 70~85% val, 나머지 test

*CNN*
- `Conv2D(32, 3)` = 3×3 필터 32개. 필터 숫자가 학습되는 가중치, 출력 채널 = 32
- 출력 크기 = **(N − F + 2P) / S + 1**. `padding='same'`이면 P = (F−1)/2 (3×3 → P=1) → stride 1에서 크기 유지. 크기를 반으로 줄이는 건 `MaxPooling2D(2)`
- Conv 두 줄 쌓기: 3×3 두 번 = 5×5 범위를 보면서 파라미터는 적고 ReLU는 2번
- BN: 층 출력을 평균 0, 분산 1로 정리 → 학습 안정·가속. BN에 자체 이동값이 있어 Conv/Dense는 `use_bias=False`
- Dense 앞에 `Flatten()` 필수. 출력 `Dense(10, 'softmax')`, 정수 라벨 → `sparse_categorical_crossentropy`, 원핫 → `categorical_crossentropy`
- 증강: 수업 방식은 `ImageDataGenerator(...)` + `model.fit(datagen.flow(X, y, batch_size=...))`. 증강 레이어(RandomFlip 등)도 같은 역할

*시각화*
- `plt.plot(history.history['loss'])` → 에폭마다 1개씩 저장된 리스트라 x축은 자동으로 0, 1, 2, … (에폭)
- 2×5 그리드: `fig, axes = plt.subplots(2, 5)` → `for i, ax in enumerate(axes.ravel()):` → `ax.imshow(X[i].squeeze(), cmap='gray')`
- 멈춘 에폭 `len(h.history['loss'])`, 최고 에폭 `np.argmin(h.history['val_loss']) + 1`

*시계열 (ETTh1)*
- 흐름: 순서대로 분할 → train으로 스케일 → 윈도우 생성 → 모델
- 윈도우: 과거 96시간 → 다음 24시간. 개수 = 길이 − 96 − 24 + 1 (17,420 → 17,301)
- 기준선(마지막 값 복사) 1.46°C > 그냥 MLP 1.61 (짐). **Residual**: (정답 − 마지막 값)을 학습, 예측 때 마지막 값을 더함 → 1.36 최고

*기구학*
- FK: q1~q6 → x, y, z, yaw, pitch, roll / IK: 반대. 코드는 **X, y 열만 바꾸면 끝**
- IK가 어려운 이유: 같은 위치를 만드는 관절각 조합이 여러 개
- 모델 `Input(6) → Dense relu ×3 → Dense(6)` (활성화 없음, `mse`)
- Wrapped MAE: +π와 −π는 같은 방향 → `d = np.arctan2(np.sin(p - t), np.cos(p - t))`로 −π~π로 바꾼 뒤 `np.mean(np.abs(d), axis=0)`

*Ames 회귀*
- 결측 처리 → 범주형 OneHot, 수치형 StandardScaler → y도 스케일 → MLP → `inverse_transform` → 달러 MAE
- 결과: Dropout 0.1 최고, BN·log 변환은 오히려 나빠짐
- `SelectKBest(f_regression, k)`: y와의 관계 점수로 상위 k개 열만 (분류는 `f_classif`). 열을 하나씩 따로 평가해서 interaction을 놓칠 수 있음 → 성능이 떨어질 수 있음


[⬆ 목차로 돌아가기](#목차)

---

## 17. 수업 GitHub 실습 복습 (L02~L14): 최종 실습·도전 문제 정답

> [수업 GitHub labs](https://github.com/dearyonghoon/course-ai-programming-2026/tree/main/labs) 노트북 끝에 있는 **최종 실습 / 도전 / 꼭 기억할 것**을 정답 코드와 함께 정리했습니다.
> L06, L11, L14 노트북에는 같은 역할의 **직접 해보기 / 결과 해석 체크리스트**가 있어서 함께 넣었습니다.
> 각 정답 코드 블록은 [0. 기본 import 세트](#0-기본-import-세트)만 실행했다면 **단독으로 돌아가도록** 필요한 데이터와 함수를 블록 안에 다시 정의했습니다.
> NumPy만 쓰는 블록의 주석에 적힌 출력값은 직접 실행해서 확인한 값입니다. Keras(TensorFlow) 블록은 실행 환경에 따라 숫자가 조금씩 달라질 수 있습니다.

[⬆ 목차로 돌아가기](#목차)

### 17-0. 꼭 기억할 것 한눈에 보기
**검색 키워드:** 꼭 기억할 것, 요약, 핵심 정리, summary, key takeaways, L02~L14

| 강의 | 꼭 기억할 것 (노트북 원문 요약) |
|---|---|
| L02 수학 | 공식 암기보다 **shape와 연산의 의미**. 핵심 구조는 `y = Wx + b` (입력 → affine 변환 → 다음 표현) |
| L03 선형 회귀 | **입력 → 모델 → 예측 → 손실 계산 → 파라미터 업데이트** (이후 신경망도 같은 구조) |
| L04 분류 | **입력 → logit(z = ax + b) → sigmoid → 확률**, 다중 분류는 **logits → softmax → 확률분포**, 학습은 cross entropy를 줄이는 방향 |
| L05 신경망 | ① Forward는 값을 계산 ② Backward는 loss의 gradient를 뒤에서 앞으로 전달 ③ Backprop = chain rule 반복. 변수와 gradient는 같은 shape (`W ↔ dL/dW`) |
| L06 실습 | Train → 파라미터 업데이트, Validation → 과적합 확인·best epoch·early stopping, Test → 최종 평가 |
| L07 손실 | ① Loss는 예측과 정답의 차이를 스칼라 하나로 요약 ② task와 출력 의미에 따라 loss 선택 ③ 많은 loss는 NLL로 해석 ④ 여러 목적은 composite loss |
| L08 학습 | ① Optimizer는 **어느 방향으로** ② Learning rate는 **얼마나 크게** ③ Initialization은 **시작 scale**을 결정 |
| L09 안정화 | activation은 gradient 흐름에 영향, input scaling은 기본 전처리(Train `fit_transform`, Test `transform`), BN은 배치 방향(열), LN은 샘플 내부(행) 정규화, 학습 때는 배치 통계·추론 때는 이동 통계 |
| L10 정규화 | 정규화는 일반화를 위해 복잡도 제어, L1은 희소성·L2는 축소, L1/L2에는 스케일링 중요, Adam에서는 L2와 decoupled weight decay 구분, Dropout은 학습 때만, 증강은 라벨을 보존해야 함 |
| L11 회귀 실습 | 한 번에 하나씩 바꿔서 원인 확인, 강한 baseline 먼저 확인, target 표현(잔차)만 바꿔도 성능이 달라짐, scaler는 train에서만 fit |
| L14 CNN | FC는 이미지를 1차원 벡터로, CNN은 공간 구조를 유지, 깊은 conv는 계층적 특징, learning curve + error analysis 함께 보기 |

[⬆ 목차로 돌아가기](#목차)

### 17-1. L02 AI를 위한 수학
**검색 키워드:** 행렬, transpose, 전치, dot, 내적, outer, 외적, 행렬곱, 원소곱, Hadamard, broadcasting, 브로드캐스팅, norm, 노름, cosine similarity, 코사인 유사도, eig, 고유값, 조건부확률, Bayes, 베이즈, 기댓값, 분산, softmax

**최종 실습 (기본·비교·확률) 정답**

```python
np.set_printoptions(precision=4, suppress=True)   # 소수 4자리, 지수표기(1e-05) 끄기

# ── 1. (3, 4) matrix를 만들고 transpose한 뒤 shape 확인 ──
A = np.arange(12).reshape(3, 4)          # 0~11을 3행 4열로 배치
print('1.', A.shape, '→', A.T.shape)     # (3, 4) → (4, 3)   .T = 행과 열을 바꿈

# ── 2. dot product와 outer product의 결과 shape 비교 ──
u = np.array([1.0, 3.0, 2.0])
v = np.array([1.0, 0.0, 2.0])
# u @ v  = 1*1 + 3*0 + 2*2 = 5.0  → 스칼라 (shape = ())
# np.outer(u, v)[i, j] = u[i] * v[j] → (3, 3) 행렬
print('2. dot:', u @ v, np.shape(u @ v),
      '/ outer shape:', np.outer(u, v).shape)        # 5.0 () / (3, 3)

# ── 3. A @ B (행렬곱) vs A * B (원소곱, Hadamard) ──
A2 = np.array([[1, 2], [3, 4]])
B2 = np.array([[5, 6], [7, 8]])
# 행렬곱: (i행) · (j열) 내적 → [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
# 원소곱: 같은 위치끼리 곱함   → [[1*5, 2*6], [3*7, 4*8]]
print('3. A@B=\n', A2 @ B2)          # [[19 22] [43 50]]
print('   A*B=\n', A2 * B2)          # [[ 5 12] [21 32]]

# ── 4. broadcasting: (4, 3) matrix의 모든 행에 같은 bias 더하기 ──
X = np.arange(12).reshape(4, 3)
bias = np.array([10, 20, 30])            # shape (3,)
print('4.\n', X + bias)                  # bias (3,)가 4개 행에 자동 복사되어 더해짐 → (4, 3)
# 첫 행: [0, 1, 2] + [10, 20, 30] = [10, 21, 32]

# ── 5. 1-norm, 2-norm, infinity-norm ──
x = np.array([3.0, 4.0, -2.0])
# 1-norm  = |3| + |4| + |-2|      = 9
# 2-norm  = sqrt(9 + 16 + 4)      = sqrt(29) ≈ 5.3852
# inf-norm = max(|3|, |4|, |-2|)  = 4
print('5.', np.linalg.norm(x, 1),
      np.linalg.norm(x, 2),
      np.linalg.norm(x, np.inf))         # 9.0 5.3852 4.0

# ── 6. cosine similarity가 1, 0, -1이 되는 세 쌍 ──
def cos_sim(a, b):
    # cos θ = (a·b) / (‖a‖ ‖b‖)   방향이 같으면 1, 수직이면 0, 반대면 -1
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1.0, 2.0])
print('6.', cos_sim(a, 3 * a),                    # 같은 방향 (크기만 3배)  → 1.0
      cos_sim(a, np.array([-2.0, 1.0])),         # 수직 (1*-2 + 2*1 = 0) → 0.0
      cos_sim(a, -a))                            # 반대 방향             → -1.0

# ── 7. eig로 eigenpair를 구하고 W @ x = λx 검증 (모든 쌍) ──
W = np.array([[2.0, 1.0], [1.0, 2.0]])
vals, vecs = np.linalg.eig(W)            # vals: 고유값 (2,), vecs: 고유벡터를 '열'로 담은 (2, 2)
for i in range(len(vals)):
    # i번째 고유벡터는 vecs[:, i] (행 vecs[i]가 아님에 주의!)
    print('7.', vals[i],
          np.allclose(W @ vecs[:, i], vals[i] * vecs[:, i]))   # 3.0 True / 1.0 True
# numpy 버전에 따라 고유값이 (3+0j)처럼 복소수로 표시될 수 있음
# → 대칭행렬이면 np.linalg.eigh(W)를 쓰면 실수로 나옴 (오름차순: 1.0, 3.0)

# ── 8. 사건을 직접 정의하고 조건부확률 (주사위: A = 짝수, B = 3보다 큼) ──
outcomes = np.arange(1, 7)               # [1 2 3 4 5 6]
A_ev = outcomes % 2 == 0                 # 짝수 → [F T F T F T]
B_ev = outcomes > 3                      # 4,5,6 → [F F F T T T]
# P(A|B) = P(A ∩ B) / P(B),  np.mean(불리언) = True 비율 = 확률
print('8. P(A|B) =', np.mean(A_ev & B_ev) / np.mean(B_ev))   # (2/6)/(3/6) = 0.6667

# ── 9. Bayes: prior 1% → 10%일 때 posterior 변화 ──
def posterior(prior, sens=0.99, false_pos=0.05):
    # sens = P(양성 | 병), false_pos = P(양성 | 정상)
    p_pos = sens * prior + false_pos * (1 - prior)   # 전확률: P(양성)
    return sens * prior / p_pos                       # Bayes: P(병 | 양성)

print('9.', posterior(0.01), '→', posterior(0.10))    # 0.1667 → 0.6875

# ── 10. 확률분포를 만들고 기댓값과 분산 ──
xs = np.array([0, 1, 2])                 # 값
ps = np.array([0.2, 0.5, 0.3])           # 확률 (합 = 1)
E = np.sum(xs * ps)                      # E[X] = Σ x·p = 0 + 0.5 + 0.6 = 1.1
Var = np.sum((xs - E) ** 2 * ps)         # Var = Σ (x − E)²·p  (= E[X²] − (E[X])² = 1.7 − 1.21)
print('10. E =', E, 'Var =', Var)        # 1.1, 0.49
```

**도전 정답: 노름, 코사인 유사도, softmax 직접 구현**

```python
def my_l1_norm(x):
    # L1 노름 = 절댓값의 합
    return np.sum(np.abs(x))

def my_l2_norm(x):
    # L2 노름 = 제곱합의 제곱근 (유클리드 길이)
    return np.sqrt(np.sum(x ** 2))

def my_cosine_similarity(x, y):
    # 내적 / (두 벡터 길이의 곱)
    return np.sum(x * y) / (my_l2_norm(x) * my_l2_norm(y))

def my_softmax(x):
    z = x - np.max(x)      # 최댓값을 빼도 결과는 같음 (exp(z-c)/Σexp(z-c) = exp(z)/Σexp(z))
                           # → exp(1000) 같은 오버플로 방지
    e = np.exp(z)
    return e / np.sum(e)   # 합이 1인 확률분포

x = np.array([3.0, 4.0, -2.0])
y = np.array([1.0, 0.0, 2.0])

# 직접 만든 함수 vs numpy 내장 결과 비교 → 모두 True
print(np.isclose(my_l1_norm(x), np.linalg.norm(x, 1)))
print(np.isclose(my_l2_norm(x), np.linalg.norm(x)))
print(np.isclose(my_cosine_similarity(x, y),
                 x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))

logits = np.array([1.0, 2.0, 3.0, 4.0])
print(my_softmax(logits), my_softmax(logits).sum())   # [0.0321 0.0871 0.2369 0.6439] 1.0
print(np.allclose(my_softmax(logits),
                  tf.nn.softmax(logits).numpy()))     # TensorFlow 결과와 같음 → True
```

**노트북 함수 직접 구현: `cosine_similarity` (L02 노트북의 함수 이름 그대로)**

```python
def cosine_similarity(x, y):
    """두 벡터의 코사인 유사도 cos θ = (x · y) / (‖x‖₂ ‖y‖₂)
    - x, y: 같은 길이의 1차원 배열
    - 반환: -1 ~ 1 사이 실수 (1: 같은 방향, 0: 수직, -1: 반대 방향)
    - 크기(길이)는 무시하고 '방향'만 비교함 → [1, 1]과 [2, 2]는 1
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dot = x @ y                                   # 내적 = Σ x_i * y_i
    norm_x = np.sqrt(np.sum(x ** 2))              # ‖x‖₂
    norm_y = np.sqrt(np.sum(y ** 2))              # ‖y‖₂
    return dot / (norm_x * norm_y)

x = np.array([1.0, 1.0])
print('same direction :', cosine_similarity(x, [2.0, 2.0]))     # 1.0
print('orthogonal     :', cosine_similarity(x, [1.0, -1.0]))    # 0.0
print('opposite       :', cosine_similarity(x, [-1.0, -1.0]))   # -1.0
# 주의: 영벡터(길이 0)가 들어오면 0으로 나누게 됨 → nan
```

**꼭 기억할 것:** AI에서 수학은 공식 암기보다 **배열의 shape와 연산이 무엇을 의미하는지** 이해하는 것이 중요합니다. 핵심 구조는 **y = Wx + b**, 즉 **입력 → 선형/affine 변환 → 다음 표현**입니다.

[⬆ 목차로 돌아가기](#목차)

### 17-2. L03 선형 회귀
**검색 키워드:** 선형 회귀, linear regression, 경사하강법, gradient descent, 학습률, learning rate, 최소제곱법, LSM, least squares, 정규방정식, Keras 선형 회귀, 사용자 정의 손실, custom loss, MAE, MSE, squared_2norm_loss, abs_loss

**기본 1. 학습률을 바꾸어 수렴 속도 비교**

```python
x = np.array([1, 3, 5, 7, 9], dtype=float)        # 공부 시간
y = np.array([75, 77, 85, 83, 90], dtype=float)   # 점수

def train_linear_regression(lr, epochs=1000):
    """y_pred = a*x + b 를 MSE 경사하강법으로 학습
    - MSE = mean((y - y_pred)²)
    - ∂MSE/∂a = -(2/n) Σ x·error,  ∂MSE/∂b = -(2/n) Σ error   (error = y - y_pred)
    - 반환: 최종 a, b, epoch별 MSE 기록
    """
    a = b = 0.0                 # 기울기, 절편 0에서 시작
    n = len(x)
    history = []
    for _ in range(epochs):
        error = y - (a * x + b)                       # 오차 = 정답 - 예측 (shape (5,))
        a -= lr * (-(2 / n) * np.sum(x * error))      # a ← a − lr·∂L/∂a
        b -= lr * (-(2 / n) * np.sum(error))          # b ← b − lr·∂L/∂b
        loss = np.mean(error ** 2)                    # 이번 step 전의 MSE
        history.append(loss)
        if not np.isfinite(loss):                     # inf/nan → 발산했으므로 중단
            break
    return a, b, np.array(history)

for lr in [0.0003, 0.003, 0.03]:
    a, b, hist = train_linear_regression(lr)
    print(f'lr={lr:<7} a={a:10.4f} b={b:10.4f} final MSE={hist[-1]:.4f}')
    plt.plot(hist, label=f'lr={lr}')

plt.yscale('log')            # 손실 차이가 커서 로그 축으로 봐야 비교됨
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.show()
# 결과(1000 epoch):
#   lr=0.0003 → MSE 925 (매우 느림)
#   lr=0.003  → MSE 75 (더 빠르지만 아직 수렴 전, 8000 epoch면 a≈1.8)
#   lr=0.03   → 발산 (값이 폭발해서 inf/nan)
```

**기본 2~3. Keras 단순/다중 선형 회귀로 예측값 바꿔 보기**

```python
tf.random.set_seed(0)                              # 가중치 초기값 고정 (결과 재현)
x_tf = np.array([1, 3, 5, 7, 9], dtype=np.float32).reshape(-1, 1)   # (5,) → (5, 1): 입력은 2차원
y_tf = np.array([75, 77, 85, 83, 90], dtype=np.float32)

# 입력 1개 → 출력 1개, 활성화 없음(linear) = y = w·x + b
model = Sequential([Input(shape=(1,)), Dense(1, activation='linear')])
model.compile(optimizer='sgd', loss='mse')         # sgd 기본 lr = 0.01
model.fit(x_tf, y_tf, epochs=2000, verbose=0)      # verbose=0: 진행 로그 끄기

for hours in [2, 7, 10]:                           # 2. hours 값을 바꿔 예측
    # model(입력, training=False)는 predict와 같은 결과 (작은 입력에 빠름)
    # 입력은 반드시 (샘플 수, 특성 수) = (1, 1) 모양
    x_new = np.array([[hours]], dtype=np.float32)
    pred = model(x_new, training=False).numpy().squeeze()
    print(f'{hours}시간 → 예측 점수 {pred:.2f}')

# 다중 선형 회귀: 입력 2개 (공부 시간, 과외 횟수) → y = w1·x1 + w2·x2 + b
x_multi = np.array([[1, 0], [3, 1], [5, 5], [7, 2], [9, 4]], dtype=np.float32)   # (5, 2)
model_multi = Sequential([Input(shape=(2,)), Dense(1, activation='linear')])
model_multi.compile(optimizer='sgd', loss='mse')
model_multi.fit(x_multi, y_tf, epochs=2000, verbose=0)

for hours, past_exams in [(6, 4), (8, 1), (2, 5)]:      # 3. 두 입력을 모두 바꿔 예측
    pred = model_multi(np.array([[hours, past_exams]], dtype=np.float32),
                       training=False).numpy().squeeze()    # 입력 (1, 2)
    print(f'hours={hours}, past_exams={past_exams} → {pred:.2f}')
```

**비교 4. 최소제곱법 vs 경사하강법 vs Keras 파라미터**

```python
x = np.array([1, 3, 5, 7, 9], dtype=float)
y = np.array([75, 77, 85, 83, 90], dtype=float)

# (1) 최소제곱법 (정확한 해, 반복 없음)
#     a = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²,  b = ȳ − a·x̄
mx, my = x.mean(), y.mean()
a_lsm = np.sum((x - mx) * (y - my)) / np.sum((x - mx) ** 2)
b_lsm = my - a_lsm * mx

# (2) 직접 구현한 경사하강법 (lr 0.003, 8001 epoch: 교안 3강 설정)
a_gd = b_gd = 0.0
for _ in range(8001):
    error = y - (a_gd * x + b_gd)
    a_gd -= 0.003 * (-(2 / len(x)) * np.sum(x * error))
    b_gd -= 0.003 * (-(2 / len(x)) * np.sum(error))

# (3) Keras (Dense(1) 하나 = 선형 회귀)
tf.random.set_seed(0)
m = Sequential([Input(shape=(1,)), Dense(1)])
m.compile(optimizer='sgd', loss='mse')
m.fit(x.reshape(-1, 1).astype('float32'), y.astype('float32'), epochs=2000, verbose=0)
w_k, b_k = m.layers[0].get_weights()     # w_k: (1, 1) 가중치 행렬, b_k: (1,) 편향

print(pd.DataFrame({
    'method': ['LSM', 'GD (manual)', 'Keras'],
    'a (slope)': [a_lsm, a_gd, float(w_k.squeeze())],
    'b (intercept)': [b_lsm, b_gd, float(b_k.squeeze())],
}))
# LSM 정확한 해: a = 1.8, b = 73.0
# 세 방법 모두 a ≈ 1.8, b ≈ 73 근처로 모임 (LSM이 정확한 해, 나머지는 반복으로 근사)
```

**도전 5. 사용자 정의 손실을 MAE로 바꾸고 MSE와 비교**

```python
x_multi = np.array([[1, 0], [3, 1], [5, 5], [7, 2], [9, 4]], dtype=np.float32)
y_multi = np.array([75, 77, 85, 83, 90], dtype=np.float32)

# 사용자 정의 손실 규칙: (y_true, y_pred) 순서, tf 연산만 사용, 스칼라(또는 샘플별 값) 반환
def squared_2norm_loss(y_true, y_pred):            # 노트북의 MSE 손실
    return tf.reduce_mean(tf.square(y_true - y_pred))   # mean((y − ŷ)²)

def abs_loss(y_true, y_pred):                      # 정답: MAE = mean(|y − ŷ|)
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# (이름, 손실 함수, optimizer) 세 가지 설정 비교
for name, loss_fn, opt in [('MSE + sgd', squared_2norm_loss, 'sgd'),
                           ('MAE + sgd', abs_loss, 'sgd'),
                           ('MAE + Adam(0.1)', abs_loss, Adam(learning_rate=0.1))]:
    tf.random.set_seed(0)                          # 같은 초기값에서 시작해야 공정한 비교
    m = Sequential([Input(shape=(2,)), Dense(1, activation='linear')])
    m.compile(optimizer=opt, loss=loss_fn)         # compile(loss=함수이름) ← 괄호 붙이지 않음
    m.fit(x_multi, y_multi, epochs=2000, verbose=0)

    pred = m(x_multi, training=False).numpy().squeeze()   # (5, 1) → (5,)
    w, b = m.layers[0].get_weights()
    print(f'{name:16s}: w={np.round(w.squeeze(), 3)}, b={b.squeeze():.3f}, '
          f'pred={np.round(pred, 2)}, '
          f'MAE={np.mean(np.abs(y_multi - pred)):.3f}, '
          f'MSE={np.mean((y_multi - pred) ** 2):.3f}')
print('target:', y_multi)
```

실행 결과와 해석 (TF 2.21, seed 0):

| 설정 | b (절편) | MAE | MSE |
|---|---|---|---|
| MSE + sgd (노트북 그대로) | 72.80 | 0.808 | **0.723** |
| MAE + sgd (손실만 MAE로) | 8.55 | 23.37 | 1030.2 |
| MAE + Adam(lr=0.1) | 72.17 | **0.727** | 1.108 |

- **손실만 MAE로 바꾸면 같은 설정에서 학습이 실패합니다.** MAE의 기울기는 오차 크기와 상관없이 ±1/N로 일정해서, `sgd`(lr 0.01)로 2000 epoch를 돌려도 절편 b가 최대 약 20밖에 이동하지 못합니다(정답은 약 73). MSE는 오차가 클수록 기울기가 커져서 빨리 따라갑니다.
- 학습률을 키우거나 Adam처럼 step 크기를 조절하는 optimizer를 쓰면 MAE도 수렴합니다.
- 수렴한 두 모델은 **각자 최소화한 지표에서 이깁니다.** MAE로 학습하면 MAE가 더 작고, MSE로 학습하면 MSE가 더 작습니다. MSE는 큰 오차를 제곱해 더 강하게 줄이고, MAE는 모든 오차를 같은 비율로 다뤄 이상치에 덜 민감합니다.

**꼭 기억할 것:** 선형 회귀의 학습 과정은 이후 신경망 학습과 같은 기본 구조입니다. **입력 → 모델 → 예측 → 손실 계산 → 파라미터 업데이트**

[⬆ 목차로 돌아가기](#목차)

### 17-3. L04 로지스틱 회귀와 분류
**검색 키워드:** 로지스틱 회귀, logistic regression, sigmoid, 시그모이드, BCE, binary cross entropy, 이진 교차 엔트로피, softmax, CCE, cross entropy, entropy, 엔트로피, 혼동행렬, confusion matrix, TP TN FP FN, accuracy, precision, recall, F1, threshold, 임계값, 불균형, logistic_probability, confusion_counts, classification_metrics

**기본 1~4**

```python
def sigmoid(z):
    # σ(z) = 1 / (1 + e^(−z)) : 실수 → (0, 1) 확률
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))

def binary_cross_entropy(y_true, y_prob, eps=1e-7):
    # BCE = −mean( y·log p + (1−y)·log(1−p) )
    y_prob = np.clip(y_prob, eps, 1 - eps)          # p가 0이나 1이면 log(0) = −inf → 살짝 잘라냄
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

# ── 1. sigmoid(-5), sigmoid(0), sigmoid(5) ──
print('1.', sigmoid([-5, 0, 5]))                    # [0.0067 0.5 0.9933]  (σ(0) = 0.5)

# ── 2. a(기울기)는 곡선의 가파름, b는 위치(중심 x = -b/a)를 바꿈 ──
hours = np.array([1, 3, 5, 7, 9, 11, 13], dtype=float)
outcome = np.array([0, 0, 0, 1, 1, 1, 1], dtype=float)    # 불합격 0 / 합격 1
x_line = np.linspace(0, 14, 300)                           # 곡선을 그릴 x 300개
for a, b in [(1.0, -6.0), (3.0, -18.0), (0.5, -3.0), (1.0, -8.0)]:
    # a·x + b = 0 인 지점(x = −b/a)에서 확률이 0.5
    plt.plot(x_line, sigmoid(a * x_line + b),
             label=f'a={a}, b={b} (center={-b / a:.1f})')
plt.scatter(hours, outcome)
plt.axhline(0.5, ls='--')                           # 임계값 0.5 기준선
plt.legend(); plt.show()

# ── 3. 정답(1)과 반대 방향으로 확률을 줄수록 BCE 급증 ──
for p in [0.9, 0.5, 0.1, 0.01]:
    bce_val = binary_cross_entropy(np.array([1.0]), np.array([p]))   # y=1이면 BCE = −log p
    print(f'3. y=1, p={p}: BCE={bce_val:.4f}')     # 0.1054, 0.6931, 2.3026, 4.6052

# ── 4. 직접 구현한 logistic regression에서 learning rate 비교 ──
for lr in [0.005, 0.05, 0.5]:
    a = b = 0.0
    losses = []
    for epoch in range(3000):
        p = sigmoid(a * hours + b)                  # 순전파: 확률 (7,)
        losses.append(binary_cross_entropy(outcome, p))
        error = p - outcome          # BCE + sigmoid를 미분하면 ∂L/∂z = p − y
        a -= lr * np.mean(error * hours)            # ∂L/∂a = mean((p − y)·x)
        b -= lr * np.mean(error)                    # ∂L/∂b = mean(p − y)
    print(f'4. lr={lr}: a={a:.3f}, b={b:.3f}, final BCE={losses[-1]:.4f}')
    plt.plot(losses, label=f'lr={lr}')
plt.xlabel('Epoch'); plt.ylabel('BCE'); plt.legend(); plt.show()
# lr이 클수록 BCE가 빨리 줄어듦 (이 데이터는 완전히 분리 가능해서 a가 계속 커짐)
```

**노트북 함수 직접 구현: `logistic_probability` (L04 노트북의 함수 이름 그대로)**

```python
def logistic_probability(x, a, b):
    """로지스틱 회귀 모델의 합격 확률 P(y=1 | x) = σ(a·x + b)
    - x: 입력 (스칼라 또는 배열)
    - a: 기울기(weight), b: 절편(bias)
    - 1단계: logit z = a·x + b (실수 전체)  →  2단계: sigmoid로 (0, 1) 확률
    """
    z = a * np.asarray(x, dtype=float) + b          # linear score (logit)
    return 1.0 / (1.0 + np.exp(-z))                 # sigmoid

hours = np.array([1, 3, 5, 7, 9, 11, 13], dtype=float)
p = logistic_probability(hours, a=1.0, b=-6.0)     # 아직 학습하지 않은 예시 파라미터
print(np.round(p, 4))           # [0.0067 0.0474 0.2689 0.7311 0.9526 0.9933 0.9991]
print((p >= 0.5).astype(int))   # 임계값 0.5로 분류 → [0 0 0 1 1 1 1]  (중심 x = 6)
```

**TensorFlow/Keras 5~7**

```python
tf.random.set_seed(0)
hours = np.array([1, 3, 5, 7, 9, 11, 13], dtype=np.float32).reshape(-1, 1)   # (7, 1)
outcome = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.float32)                 # (7,)

# Dense(1, sigmoid) 하나 = 로지스틱 회귀 그 자체 (σ(w·x + b))
model = Sequential([Input(shape=(1,)), Dense(1, activation='sigmoid')])
model.compile(optimizer=SGD(learning_rate=0.05), loss='binary_crossentropy')
model.fit(hours, outcome, epochs=1500, verbose=0)

# ── 5. 학습된 weight와 bias ──
weight, bias = model.layers[0].get_weights()       # weight (1, 1), bias (1,)
print('5. weight =', weight.squeeze(), 'bias =', bias.squeeze())

# ── 6. 새로운 study time의 합격 확률 ──
for h in [4.0, 6.5, 8.0]:
    x_new = np.array([[h]], dtype=np.float32)           # (1, 1)
    prob = float(model(x_new, training=False).numpy().squeeze())
    print(f'6. {h}시간 → 합격 확률 {prob:.4f}, 예측 class {int(prob >= 0.5)}')

# ── 7. softmax 모델의 두 출력 확률 합 = 1 확인 ──
tf.random.set_seed(0)
y_cat = to_categorical(outcome.astype(int), num_classes=2)   # 0 → [1, 0], 1 → [0, 1]
model_softmax = Sequential([
    Input(shape=(1,)),
    Dense(10, activation='relu'),         # 은닉층 추가
    Dense(2, activation='softmax'),       # 클래스 2개 확률
])
model_softmax.compile(optimizer=SGD(learning_rate=0.01),
                      loss='categorical_crossentropy',    # 원핫 라벨 → categorical
                      metrics=['accuracy'])
model_softmax.fit(hours, y_cat, epochs=2000, verbose=0)

x_grid = np.linspace(0, 14, 50, dtype=np.float32).reshape(-1, 1)   # (50, 1)
probs = model_softmax(x_grid, training=False).numpy()              # (50, 2)
print('7. row sums all 1:', np.allclose(probs.sum(axis=1), 1.0))   # 각 행(샘플) 합 = 1 → True
```

**Cross Entropy 8~9**

```python
def cross_entropy(p, q, eps=1e-12):
    # H(p, q) = −Σ p·log q   (p: 정답 분포, q: 예측 분포), 자연로그 → 단위 nats
    return -np.sum(p * np.log(np.clip(q, eps, 1.0)))

def entropy(p, eps=1e-12):
    # H(p) = −Σ p·log₂ p   (밑 2 → 단위 bits), 불확실할수록 큼
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log2(p))

# ── 8. one-hot target과 두 prediction의 CCE 비교 ──
target = np.array([0, 1, 0], dtype=float)          # 정답 클래스 1
good = np.array([0.05, 0.90, 0.05])                # 정답 클래스에 0.9
bad = np.array([0.45, 0.10, 0.45])                 # 정답 클래스에 0.1
# 원핫이면 CCE = −log(정답 클래스 확률) → −log 0.9, −log 0.1
print('8. good CCE =', cross_entropy(target, good),
      '/ bad CCE =', cross_entropy(target, bad))   # 0.1054 / 2.3026

# ── 9. entropy가 높은 분포 vs 낮은 분포 ──
high = np.array([0.25, 0.25, 0.25, 0.25])          # 균등 → 가장 불확실
low = np.array([0.97, 0.01, 0.01, 0.01])           # 한쪽에 몰림 → 확실
print('9. high =', entropy(high), 'bits / low =', entropy(low), 'bits')   # 2.0 / 0.2419
```

**Performance Metrics 10~12**

```python
# ── 10. 임의의 y_true, y_pred에서 TP/TN/FP/FN 직접 계산 ──
rng = np.random.default_rng(0)
y_true = rng.integers(0, 2, 20)          # 0/1 정답 20개
y_pred = rng.integers(0, 2, 20)          # 0/1 예측 20개
tp = np.sum((y_true == 1) & (y_pred == 1))   # 양성을 양성으로 (맞힘)
tn = np.sum((y_true == 0) & (y_pred == 0))   # 음성을 음성으로 (맞힘)
fp = np.sum((y_true == 0) & (y_pred == 1))   # 음성인데 양성이라고 함 (오경보)
fn = np.sum((y_true == 1) & (y_pred == 0))   # 양성인데 놓침
print('10. TP, TN, FP, FN =', tp, tn, fp, fn)
# sklearn confusion_matrix는 [[TN, FP], [FN, TP]] 순서 (행 = 실제, 열 = 예측)
print('    sklearn [[TN, FP], [FN, TP]] =',
      confusion_matrix(y_true, y_pred).ravel())   # ravel 순서: tn, fp, fn, tp

# ── 11. threshold를 바꾸면 precision과 recall이 반대로 움직임 ──
probs = np.array([0.95, 0.80, 0.65, 0.55, 0.45, 0.30, 0.10])   # 모델이 낸 양성 확률
labels = np.array([1, 1, 0, 1, 0, 1, 0])
for th in [0.3, 0.5, 0.7]:
    pred = (probs >= th).astype(int)                # 확률 → 0/1
    print(f'11. th={th}: precision={precision_score(labels, pred):.3f}, '
          f'recall={recall_score(labels, pred):.3f}')
# th=0.3: 0.667 / 1.000,  th=0.5: 0.750 / 0.750,  th=0.7: 1.000 / 0.500
# threshold ↑ → positive 예측이 줄어 precision ↑, recall ↓

# ── 12. 불균형 데이터: 양성 5개, 음성 9995개에서 전부 음성으로 예측 ──
tp, fp, fn, tn = 0, 0, 5, 9995
print('12. accuracy =', (tp + tn) / (tp + tn + fp + fn),
      ', recall =', tp / (tp + fn))                 # 0.9995, 0.0
# 정확도는 99.95%지만 양성을 하나도 못 찾음
# → 불균형 데이터에서는 precision/recall/F1을 함께 봐야 함
```

**노트북 함수 직접 구현: `confusion_counts`, `classification_metrics` (L04 노트북의 함수 이름 그대로)**

```python
def confusion_counts(y_true, y_pred):
    """이진 분류의 TP, TN, FP, FN 개수를 직접 셈
    - y_true, y_pred: 0/1 배열 (리스트, Series도 OK → 정수 배열로 변환)
    - 반환 순서: (tp, tn, fp, fn)   ※ sklearn ravel() 순서(tn, fp, fn, tp)와 다름!
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))   # 실제 1, 예측 1
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))   # 실제 0, 예측 0
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))   # 실제 0, 예측 1 (False Positive)
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))   # 실제 1, 예측 0 (False Negative)
    return tp, tn, fp, fn

def classification_metrics(tp, tn, fp, fn):
    """개수로부터 accuracy, precision, recall, F1 계산
    accuracy  = (TP + TN) / 전체
    precision = TP / (TP + FP)   ← '양성이라고 예측한 것' 중 진짜 양성 비율
    recall    = TP / (TP + FN)   ← '진짜 양성' 중 찾아낸 비율 (= sensitivity, TPR)
    F1        = 2·P·R / (P + R)  ← precision과 recall의 조화평균
    분모가 0이면 정의할 수 없으므로 nan 반환
    """
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = np.nan
    return accuracy, precision, recall, f1

y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1])
y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 1])
tp, tn, fp, fn = confusion_counts(y_true, y_pred)
print('TP, TN, FP, FN =', tp, tn, fp, fn)            # 3 3 1 1

acc, prec, rec, f1 = classification_metrics(tp, tn, fp, fn)
print(f'Accuracy : {acc:.4f}')                       # 0.7500
print(f'Precision: {prec:.4f}')                      # 0.7500
print(f'Recall   : {rec:.4f}')                       # 0.7500
print(f'F1 Score : {f1:.4f}')                        # 0.7500

# 확률 → 임계값 → 지표까지 한 번에 (임계값만 바꿔 비교할 때)
probs = np.array([0.95, 0.80, 0.65, 0.55, 0.45, 0.30, 0.10])
labels = np.array([1, 1, 0, 1, 0, 1, 0])
for th in [0.3, 0.5, 0.7]:
    m = classification_metrics(*confusion_counts(labels, (probs >= th).astype(int)))
    print(f'th={th}: acc={m[0]:.3f} prec={m[1]:.3f} rec={m[2]:.3f} f1={m[3]:.3f}')
# th=0.3: acc=0.714 prec=0.667 rec=1.000 f1=0.800
# th=0.5: acc=0.714 prec=0.750 rec=0.750 f1=0.750
# th=0.7: acc=0.714 prec=1.000 rec=0.500 f1=0.667
```

**도전 정답: 분류 함수 직접 구현**

```python
def my_sigmoid(z):
    # 1 / (1 + e^(−z))
    return 1.0 / (1.0 + np.exp(-z))

def my_binary_cross_entropy(y_true, y_prob, eps=1e-7):
    # −mean(y·log p + (1−y)·log(1−p)), log(0) 방지를 위해 clip
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

def my_softmax(z):
    # 최댓값을 빼서 오버플로 방지 후 exp / Σexp
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

def my_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)          # 맞힌 것 / 전체

def my_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0   # 양성 예측 중 정답

def my_recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0   # 실제 양성 중 찾아냄

# 검증: 라이브러리 결과와 비교
yt = np.array([1, 1, 1, 0, 0, 0, 0, 1])
yp = np.array([1, 0, 1, 0, 1, 0, 0, 1])
tp = np.sum((yt == 1) & (yp == 1)); tn = np.sum((yt == 0) & (yp == 0))
fp = np.sum((yt == 0) & (yp == 1)); fn = np.sum((yt == 1) & (yp == 0))
print(my_accuracy(tp, tn, fp, fn), accuracy_score(yt, yp))       # 0.75 0.75
print(my_precision(tp, fp), precision_score(yt, yp))             # 0.75 0.75
print(my_recall(tp, fn), recall_score(yt, yp))                   # 0.75 0.75
print(my_sigmoid(0.0),                                            # 0.5
      np.allclose(my_softmax(np.array([1., 3., 2., .5])),
                  tf.nn.softmax([1., 3., 2., .5]).numpy()))      # True
print(my_binary_cross_entropy(np.array([1.0, 0.0]), np.array([0.8, 0.2])),
      float(tf.keras.losses.BinaryCrossentropy()([[1.0], [0.0]],
                                                 [[0.8], [0.2]])))   # 0.2231 0.2231
```

**꼭 기억할 것:** 이진 분류는 **입력 → linear score(logit, z = ax + b) → sigmoid → probability**, 다중 분류는 **입력 → logits → softmax → probability distribution**입니다. 학습은 정답과 예측 사이의 **cross entropy loss를 줄이는 방향**으로 파라미터를 업데이트하는 과정입니다.

[⬆ 목차로 돌아가기](#목차)

### 17-4. L05 신경망 (Forward, Backpropagation)
**검색 키워드:** 신경망, neural network, forward, 순전파, backward, 역전파, backpropagation, chain rule, 연쇄 법칙, gradient, 기울기, numerical gradient, 수치 미분, gradient checking, dying ReLU, simple_loss, sigmoid_neuron_loss, manual_train

**21. Backpropagation 체크리스트 (말로 답하기) 모범답안**

| 질문 | 답 |
|---|---|
| 1. Forward pass 방향 | 입력 → 출력(예측) → loss 방향으로 **값**을 계산 |
| 2. Backward pass 방향 | loss → 출력층 → 은닉층 → 입력 방향으로 **gradient**를 전달 |
| 3. `z`와 `h`의 차이 | `z = Wx + b`는 활성화 전 값(pre-activation), `h = σ(z)`는 활성화 함수를 통과한 값 |
| 4. gradient가 양수 | 그 파라미터를 키우면 loss가 **증가**한다는 뜻 |
| 5. gradient를 빼는 이유 | gradient는 loss가 가장 빠르게 증가하는 방향이므로, 반대 방향(−)으로 가야 loss가 감소 |
| 6. upstream × local | 뒤에서 온 gradient(∂L/∂출력)에 현재 연산의 국소 미분(∂출력/∂입력)을 곱해 ∂L/∂입력을 구하는 chain rule |
| 7. output gradient가 먼저 필요한 이유 | 은닉층 gradient = (출력층에서 내려온 upstream gradient) × (은닉층 local derivative)이기 때문 |
| 8. 자동미분이 하는 일 | forward 연산을 기록(`GradientTape`)해 두고 chain rule로 모든 파라미터의 gradient를 자동 계산 |

**노트북 함수 직접 구현: `simple_loss`, `sigmoid_neuron_loss` (L05 노트북의 함수 이름 그대로)**

```python
# ── (1) simple_loss: 활성화 없는 뉴런 하나 ŷ = w·x + b, L = ½(ŷ − y)² ──
x, y, b = 2.0, 5.0, 1.0                    # 입력, 정답, 편향 (고정)

def simple_loss(w):
    """w만 바꿔가며 loss를 계산 (수치 미분용으로 w의 함수 형태로 만듦)"""
    y_hat = w * x + b                      # 순전파
    return 0.5 * (y_hat - y) ** 2          # ½ 제곱오차 (½은 미분하면 2와 약분되도록)

w = 1.0
# 해석적(손으로 푼) 미분: dL/dw = (ŷ − y) · x = (3 − 5) · 2 = −4
y_hat = w * x + b
grad_analytic = (y_hat - y) * x
# 수치 미분 (중앙 차분): [L(w+ε) − L(w−ε)] / 2ε
eps = 1e-5
grad_numerical = (simple_loss(w + eps) - simple_loss(w - eps)) / (2 * eps)
print('loss(w) =', simple_loss(w))                     # 0.5 * (3 − 5)² = 2.0
print('analytic  gradient =', grad_analytic)           # -4.0
print('numerical gradient =', round(grad_numerical, 6))   # -4.0
# 업데이트 한 번: w ← w − lr·dL/dw  → loss가 줄어드는지 확인
lr = 0.1
print('after update: w =', w - lr * grad_analytic,
      'loss =', simple_loss(w - lr * grad_analytic))   # w = 1.4, loss = 0.72 (2.0에서 감소)

# ── (2) sigmoid_neuron_loss: 시그모이드 뉴런 ŷ = σ(w·x + b), L = ½(ŷ − y)² ──
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_neuron_loss(w, x=2.0, b=0.0, y=1.0):
    """시그모이드 뉴런 하나의 loss (기본값은 노트북 예제와 같은 상황: x=2, b=0, 정답 1)"""
    z = w * x + b                          # pre-activation
    y_hat = sigmoid(z)                     # activation
    return 0.5 * (y_hat - y) ** 2

w = 0.5
z = w * 2.0 + 0.0                          # z = 1.0
y_hat = sigmoid(z)                         # 0.7311
# chain rule: dL/dw = (ŷ − y) · σ'(z) · x,  σ'(z) = ŷ(1 − ŷ)
dL_dyhat = y_hat - 1.0                     # upstream
dyhat_dz = y_hat * (1.0 - y_hat)           # local (sigmoid 미분)
dL_dw = dL_dyhat * dyhat_dz * 2.0          # × dz/dw = x
grad_numerical = (sigmoid_neuron_loss(w + eps)
                  - sigmoid_neuron_loss(w - eps)) / (2 * eps)
print('Backprop gradient :', round(dL_dw, 8))                       # -0.10575419
print('Numerical gradient:', round(grad_numerical, 8))              # -0.10575419
print('Difference        :', abs(dL_dw - grad_numerical) < 1e-8)    # True → 역전파가 맞음
```

**22. 직접 해보기 1: learning rate 0.001, 0.1, 1.0 비교**

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def manual_train(lr, steps=100, act='sigmoid',
                 w1=0.8, b1=0.1, w2=1.2, b2=-0.2, x=1.5, y=2.0):
    """입력 1 → 은닉 뉴런 1 → 출력 1 인 아주 작은 신경망을 손으로 학습
    순전파:  z1 = w1·x + b1,  h = act(z1),  ŷ = w2·h + b2,  L = ½(ŷ − y)²
    역전파:  dL/dz2 = ŷ − y
             dL/dw2 = dL/dz2 · h,           dL/db2 = dL/dz2
             dL/dh  = dL/dz2 · w2
             dL/dz1 = dL/dh · act'(z1)      (sigmoid' = h(1−h), ReLU' = 1 if z1>0 else 0)
             dL/dw1 = dL/dz1 · x,           dL/db1 = dL/dz1
    """
    losses = []
    for _ in range(steps):
        # ---- forward ----
        z1 = w1 * x + b1
        h = sigmoid(z1) if act == 'sigmoid' else max(0.0, z1)
        y_hat = w2 * h + b2
        losses.append(0.5 * (y_hat - y) ** 2)
        # ---- backward (뒤에서 앞으로) ----
        dL_dz2 = y_hat - y                                   # 출력층 upstream
        dL_dw2, dL_db2 = dL_dz2 * h, dL_dz2
        dL_dh = dL_dz2 * w2                                  # 은닉층으로 전달
        local = h * (1 - h) if act == 'sigmoid' else float(z1 > 0)   # sigmoid' 또는 ReLU'
        dL_dz1 = dL_dh * local
        dL_dw1, dL_db1 = dL_dz1 * x, dL_dz1
        # ---- update (모든 gradient를 구한 뒤 한꺼번에) ----
        w1 -= lr * dL_dw1
        b1 -= lr * dL_db1
        w2 -= lr * dL_dw2
        b2 -= lr * dL_db2
    return np.array(losses), (w1, b1, w2, b2)

for lr in [0.001, 0.1, 1.0]:
    losses, _ = manual_train(lr)
    print(f'lr={lr}: first loss={losses[0]:.4f}, last loss={losses[-1]:.6f}')
    plt.plot(losses, label=f'lr={lr}')
plt.yscale('log'); plt.xlabel('Step'); plt.ylabel('Loss'); plt.legend(); plt.show()
# 결과: 0.001 → 100 step 후에도 loss 0.557 (거의 안 줄어듦)
#       0.1   → 거의 0으로 안정적 감소
#       1.0   → 이 작은 예제에서는 가장 빨리 수렴하지만,
#               데이터·모델이 바뀌면 진동하거나 발산할 수 있음
```

**23. 직접 해보기 2: sigmoid 대신 ReLU, z1이 음수가 되는 경우**

```python
# (위 블록의 manual_train을 먼저 실행)
# w1 = -0.8이면 z1 = -0.8 * 1.5 + 0.1 = -1.1 < 0 → ReLU 출력 h = 0, 기울기 = 0
losses, (w1, b1, w2, b2) = manual_train(lr=0.1, act='relu', w1=-0.8)
print('w1, b1 (안 바뀜):', w1, b1)            # -0.8, 0.1 그대로 → hidden이 "죽은" 상태
print('w2 (안 바뀜):', w2, ' b2:', b2)        # h = 0이라 dL/dw2 = 0, b2만 학습됨 (b2 → 2.0)
print('final loss:', losses[-1])              # b2만으로 맞추므로 결국 0에 가까워짐
# 결론: z1 < 0이면 ReLU의 local derivative가 0
#       → 그 아래(w1, b1)로 gradient가 전달되지 않음 (dying ReLU)
# sigmoid는 기울기가 작아도 0은 아님 → 이 현상이 activation 선택(Leaky ReLU 등)과 연결됨
```

**24. 도전: Numerical Gradient로 모든 파라미터 확인 (gradient checking)**

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

x, y = 1.5, 2.0
params = {'w1': 0.8, 'b1': 0.1, 'w2': 1.2, 'b2': -0.2}

def loss_fn(p):
    # 파라미터 딕셔너리를 받아 순전파 → loss
    h = sigmoid(p['w1'] * x + p['b1'])
    y_hat = p['w2'] * h + p['b2']
    return 0.5 * (y_hat - y) ** 2

# (1) Backpropagation (chain rule)
h = sigmoid(params['w1'] * x + params['b1'])
y_hat = params['w2'] * h + params['b2']
dL_dz2 = y_hat - y                                   # 출력층 gradient
dL_dz1 = dL_dz2 * params['w2'] * h * (1 - h)         # × w2 × sigmoid'(z1)
backprop = {'w1': dL_dz1 * x, 'b1': dL_dz1,
            'w2': dL_dz2 * h, 'b2': dL_dz2}

# (2) Numerical gradient: (L(θ+ε) − L(θ−ε)) / 2ε, 파라미터마다 하나씩
eps = 1e-5
for name in params:
    plus = dict(params);  plus[name] += eps          # dict(...)로 복사해야 원본이 안 바뀜
    minus = dict(params); minus[name] -= eps
    numerical = (loss_fn(plus) - loss_fn(minus)) / (2 * eps)
    # 상대 오차 = |차이| / (|두 값의 크기 합|)
    rel_err = (abs(numerical - backprop[name])
               / max(abs(numerical) + abs(backprop[name]), 1e-12))
    print(f'{name}: backprop={backprop[name]: .8f}, '
          f'numerical={numerical: .8f}, rel_error={rel_err:.2e}')
# 상대 오차가 1e-7 이하 수준이면 backprop 구현이 맞음
```

**꼭 기억할 세 문장**
1. **Forward pass는 값을 계산한다.**
2. **Backward pass는 loss에 대한 gradient를 뒤에서 앞으로 전달한다.**
3. **Backpropagation은 chain rule을 network에 반복 적용한 것이다.**

추가로 **기억할 것:** 각 변수와 그 gradient는 같은 shape를 가집니다 (`W ↔ dL/dW`, `b ↔ dL/db`, `z ↔ dL/dz`, `h ↔ dL/dh`). 핵심 공식은 **downstream gradient = upstream gradient × local derivative**, 업데이트는 **W ← W − η·∂L/∂W**

[⬆ 목차로 돌아가기](#목차)

### 17-5. L06 분류 실습 (Pima, Iris, Wine) 직접 해보기
**검색 키워드:** Pima, 당뇨, Iris, 붓꽃, Wine, 와인, 이진 분류, 다중 분류, epoch 비교, hidden 크기, 파라미터 수, count_params, precision recall metrics, EarlyStopping, patience, 조기 종료, best model, validation_split, run_pima, run_iris, run_wine

**Pima 1~3: epoch 비교, hidden 크기 변경, precision/recall 지표 추가**

```python
BASE = 'https://raw.githubusercontent.com/dearyonghoon/course-ai-programming-2026/main/data/'
df = pd.read_csv(BASE + 'pima-indians-diabetes3.csv')   # (768, 9), 마지막 열 'diabetes'
X = df.iloc[:, 0:8]          # 특성 8개 (0~7번 열)
y = df.iloc[:, 8]            # 라벨 (0/1)

def run_pima(epochs=100, first_units=12):
    """epoch 수와 첫 은닉층 크기를 바꿔가며 같은 모델을 학습하고 마지막 epoch 지표를 출력"""
    tf.keras.utils.set_random_seed(42)            # 매번 같은 초기값 → 공정한 비교
    model = Sequential([
        Input(shape=(8,)),
        Dense(first_units, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid'),           # 이진 분류 출력
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy',                                        # 3. 지표 추가
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    history = model.fit(X, y, epochs=epochs, batch_size=10, verbose=0)
    # history.history = {'loss': [...], 'accuracy': [...], 'precision': [...], 'recall': [...]}
    last = {k: round(v[-1], 4) for k, v in history.history.items()}   # 지표별 마지막 값
    print(f'epochs={epochs}, Dense({first_units}) → {last}, params={model.count_params()}')
    return history

h50 = run_pima(epochs=50)                     # 1. epochs 50 vs 200
h200 = run_pima(epochs=200)
h20 = run_pima(epochs=100, first_units=20)    # 2. Dense(12) → Dense(20): 파라미터 221 → 357
# 파라미터: (8+1)*12 + (12+1)*8 + (8+1)*1 = 108 + 104 + 9 = 221
#           (8+1)*20 + (20+1)*8 + (8+1)*1 = 180 + 168 + 9 = 357

plt.plot(h50.history['loss'], label='epochs=50')
plt.plot(h200.history['loss'], label='epochs=200')
plt.xlabel('Epoch'); plt.ylabel('Train loss'); plt.legend(); plt.show()
# 주의: 이 실습은 학습 데이터 전체로 학습/평가하므로 epoch를 늘리면 train 성능만 좋아질 수 있음
#       (과적합 여부는 validation으로 확인)
```

**Iris 1~3: neuron 수, epoch 비교, 파라미터 191개 계산**

```python
BASE = 'https://raw.githubusercontent.com/dearyonghoon/course-ai-programming-2026/main/data/'
df = pd.read_csv(BASE + 'iris3.csv')                 # (150, 5), 마지막 열 'species' (문자열)
X = df.iloc[:, 0:4]                                  # 꽃받침/꽃잎 길이·너비 4개
y = pd.get_dummies(df.iloc[:, 4], dtype=int)         # 문자열 라벨 → 원핫 (150, 3)

def run_iris(h1=12, h2=8, epochs=100):
    """은닉층 크기(h1, h2)와 epoch를 바꿔 학습하고 마지막 정확도·손실·파라미터 수 출력"""
    tf.keras.utils.set_random_seed(42)
    model = Sequential([
        Input(shape=(4,)),
        Dense(h1, activation='relu'),
        Dense(h2, activation='relu'),
        Dense(3, activation='softmax'),               # 3개 클래스 확률
    ])
    model.compile(loss='categorical_crossentropy',    # 원핫 라벨이므로 categorical
                  optimizer='adam', metrics=['accuracy'])
    h = model.fit(X, y, epochs=epochs, batch_size=10, verbose=0)
    print(f'hidden=({h1},{h2}), epochs={epochs}: acc={h.history["accuracy"][-1]:.4f}, '
          f'loss={h.history["loss"][-1]:.4f}, params={model.count_params()}')

run_iris(12, 8, 100)          # 기본: params 191
run_iris(32, 16, 100)         # 1. neuron 수 변경
run_iris(12, 8, 50)           # 2. epochs 50 vs 200
run_iris(12, 8, 200)

# 3. 191개인 이유: Dense 파라미터 = (입력 + 1) × 출력   (+1은 bias)
print((4 + 1) * 12, (12 + 1) * 8, (8 + 1) * 3,
      '합계 =', (4 + 1) * 12 + (12 + 1) * 8 + (8 + 1) * 3)   # 60 104 27 합계 = 191
```

**Wine 1~4: patience 비교, early stopping 켜기/끄기, best model을 쓰는 이유**

```python
BASE = 'https://raw.githubusercontent.com/dearyonghoon/course-ai-programming-2026/main/data/'
df = pd.read_csv(BASE + 'wine.csv', header=None)    # 헤더 없음 → header=None 필수 (6497, 13)
X = df.iloc[:, 0:12]                                 # 특성 12개
y = df.iloc[:, 12]                                   # 1 = red, 0 = white
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True,
    random_state=42, stratify=y)                     # stratify: 클래스 비율 유지

def run_wine(patience=None, epochs=2000):
    """patience=None이면 EarlyStopping 없이 끝까지 학습"""
    tf.keras.utils.set_random_seed(42)
    model = Sequential([
        Input(shape=(12,)),
        Dense(30, activation='relu'),
        Dense(12, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    callbacks = ([] if patience is None
                 else [EarlyStopping(monitor='val_loss', patience=patience)])
    h = model.fit(X_train, y_train, epochs=epochs, batch_size=500,
                  validation_split=0.25,             # train의 마지막 25%를 validation으로
                  callbacks=callbacks, verbose=0)
    best = int(np.argmin(h.history['val_loss'])) + 1  # val_loss 최소 epoch (1부터 셈)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'patience={patience}: stopped at {len(h.history["loss"])}, best epoch={best}, '
          f'test loss={test_loss:.4f}, acc={test_acc:.4f}')
    return h

for p in [5, 20, 50]:          # 1. patience가 클수록 더 오래 학습하고 늦게 멈춤
    run_wine(patience=p)
h_off = run_wine(patience=None)   # 2. early stopping 끔: 2000 epoch 끝까지 (CPU에서 몇 분 걸림)

# 3. train/val loss가 갈라지는 시점 관찰
plt.plot(h_off.history['loss'], label='Train loss')
plt.plot(h_off.history['val_loss'], label='Validation loss')
plt.axvline(int(np.argmin(h_off.history['val_loss'])), ls='--', color='gray',
            label='min val_loss')                    # val_loss 최소 지점 세로선
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.show()
```

제가 실행한 결과 (seed 42, CPU):

| 설정 | 멈춘 epoch | best epoch | Test loss | Test acc |
|---|---|---|---|---|
| patience=5 | 601 | 596 | 0.0585 | 0.9954 |
| patience=20 | 826 | 806 | 0.0479 | 0.9954 |
| patience=50 | 856 | 806 | 0.0493 | 0.9946 |
| early stopping 끔 | 2000 | 1879 | 0.0399 | 0.9954 |

- patience가 작으면 일시적인 정체에도 너무 일찍 멈출 수 있고(5 → 601), 크면 더 기다렸다가 멈춥니다.
- 이 데이터에서는 validation loss가 2000 epoch 가까이 아주 천천히 계속 내려가서, 끝까지 학습해도 크게 과적합되지 않았습니다. 대신 시간이 훨씬 오래 걸립니다. 실행마다 결과가 조금씩 다를 수 있습니다.

4번 답: 마지막 epoch 모델은 이미 train 데이터에 과적합되기 시작했을 수 있습니다. **validation loss가 가장 낮은 시점**의 모델이 처음 보는 데이터에서 가장 잘 일반화될 것으로 기대되므로, `ModelCheckpoint(save_best_only=True)`나 `EarlyStopping(restore_best_weights=True)`로 그 모델을 사용합니다. 참고로 `restore_best_weights`를 쓰지 않은 EarlyStopping은 멈춘 시점(최고점 + patience)의 가중치를 남깁니다.

**핵심 흐름 (L06 정리):** Train set → 파라미터 업데이트 / Validation set → 과적합 확인, best epoch 선택, early stopping / Test set → 최종 성능 평가

[⬆ 목차로 돌아가기](#목차)

### 17-6. L07 손실 함수

**검색 키워드:** 손실 함수, loss function, Bernoulli, likelihood, 우도, log-likelihood, NLL, MLE, MAE, MSE, BCE, CCE, cross entropy, KL divergence, cosine similarity, cosine loss, composite loss, 결합 손실, 가중 손실

**최종 실습 1~10**

```python
# ─────────────────────────────────────────────────────────────
# 1. Bernoulli 관측을 바꾸면 MLE θ̂ = 관측 평균(앞면 비율)
#    NLL(θ) = −Σ [x·log θ + (1−x)·log(1−θ)]   (x는 0/1 관측값 배열)
#    NLL이 가장 작은 θ = MLE(최대우도추정치) = x.mean()
# ─────────────────────────────────────────────────────────────
def bernoulli_nll(theta, x):
    return -np.sum(x * np.log(theta) + (1 - x) * np.log(1 - theta))

# θ 후보 981개 (0.01 ~ 0.99, 간격 0.001). 0과 1은 log(0) 때문에 제외
thetas = np.linspace(0.01, 0.99, 981)

for obs in [np.array([1, 1, 0, 1, 0, 1, 1, 0]),              # 1이 5개 / 8개 → 0.625
            np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0])]:       # 1이 2개 / 10개 → 0.2
    nll = np.array([bernoulli_nll(t, obs) for t in thetas])  # θ마다 NLL 계산
    print('1. MLE =', thetas[np.argmin(nll)],               # NLL 최소 위치의 θ
          ' mean =', obs.mean())                            # 0.625 / 0.2 (둘이 같음)

# ─────────────────────────────────────────────────────────────
# 2. likelihood 최대 = log-likelihood 최대 = NLL 최소 (같은 θ)
#    log는 단조증가 함수라서 최대가 되는 위치가 바뀌지 않음
# ─────────────────────────────────────────────────────────────
x = np.array([1, 1, 0, 1, 0, 1, 1, 0])
# likelihood L(θ) = Π θ^x (1−θ)^(1−x)   (곱이라서 샘플이 많으면 0에 가까워짐 → 보통 log 사용)
lik = np.array([np.prod(t ** x * (1 - t) ** (1 - x)) for t in thetas])
ll = np.log(lik)       # log-likelihood
nll = -ll              # negative log-likelihood (손실로 쓰는 형태)
print('2.',
      thetas[np.argmax(lik)],    # likelihood 최대
      thetas[np.argmax(ll)],     # log-likelihood 최대
      thetas[np.argmin(nll)])    # NLL 최소 → 모두 0.625

# ─────────────────────────────────────────────────────────────
# 3. 큰 오차 하나가 있을 때 MAE vs MSE
#    MAE = mean(|y − ŷ|),  MSE = mean((y − ŷ)²)
# ─────────────────────────────────────────────────────────────
y_true = np.array([1, 2, 3, 4, 5], dtype=float)
for name, y_pred in [('good', np.array([1, 2, 3, 4, 5.5])),       # 마지막만 0.5 오차
                     ('outlier', np.array([1, 2, 3, 4, 10.]))]:   # 마지막만 5.0 오차
    mae_v = np.mean(np.abs(y_true - y_pred))
    mse_v = np.mean((y_true - y_pred) ** 2)
    print(f'3. {name}: MAE={mae_v:.2f}, MSE={mse_v:.2f}')
# good: 0.10 / 0.05, outlier: 1.00 / 5.00
# → 오차가 10배 커지면 MAE는 10배, MSE는 100배 → MSE는 큰 오차를 제곱해 훨씬 크게 벌줌

# ─────────────────────────────────────────────────────────────
# 4. cosine-based loss = 1 − cos
#    cos(a, b) = (a·b) / (‖a‖·‖b‖)   범위 [−1, 1]
#    → 1 − cos 범위 [0, 2]: 같은 방향 0, 직교 1, 반대 방향 2 (길이는 무시, 방향만 비교)
# ─────────────────────────────────────────────────────────────
a = np.array([1.0, 1.0])
for name, b in [('same', np.array([2.0, 2.0])),          # 같은 방향 (길이만 2배)
                ('orthogonal', np.array([1.0, -1.0])),   # 직각
                ('opposite', np.array([-1.0, -1.0]))]:   # 정반대
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f'4. {name}: cos={cos:.1f}, 1-cos={1 - cos:.1f}')   # 1-cos: 0, 1, 2

# ─────────────────────────────────────────────────────────────
# 5. BCE: 틀린 class에 높은 확신을 줄수록 loss가 폭증 (정답 y=0인데 p(y=1)이 큼)
#    BCE = −[y·log p + (1−y)·log(1−p)]  → y=0이면 −log(1−p)
# ─────────────────────────────────────────────────────────────
for p in [0.6, 0.9, 0.99, 0.999]:
    print(f'5. y=0, p={p}: BCE={-np.log(1 - p):.3f}')      # 0.916, 2.303, 4.605, 6.908

# ─────────────────────────────────────────────────────────────
# 6~7. CCE: 정답 확률 0.9 → 0.5 → 0.1, 그리고 CCE = −log(정답 확률)
#    CCE = −Σ_k t_k·log p_k  (t는 원핫이라 정답 칸만 남음)
# ─────────────────────────────────────────────────────────────
target = np.array([0, 1, 0])                     # 정답 = 1번 클래스 (원핫)
for p_correct in [0.9, 0.5, 0.1]:
    rest = (1 - p_correct) / 2                   # 나머지 확률을 두 칸에 반씩
    pred = np.array([rest, p_correct, rest])     # 합 = 1인 확률 분포
    cce = -np.sum(target * np.log(pred))
    print(f'6. p={p_correct}: CCE={cce:.4f}, -log(p)={-np.log(p_correct):.4f}')
    # 0.1054, 0.6931, 2.3026 (두 값이 항상 같음 → 7번 답)

# ─────────────────────────────────────────────────────────────
# 8. 세 예측 분포의 KL divergence
#    KL(p‖q) = Σ p·log(p/q)  ≥ 0,  p == q 일 때만 0, 비대칭(KL(p‖q) ≠ KL(q‖p))
# ─────────────────────────────────────────────────────────────
def kl(p, q):
    return np.sum(p * np.log(p / q))

p = np.array([0.7, 0.2, 0.1])                    # 기준(정답) 분포
for q in [np.array([0.7, 0.2, 0.1]),             # 똑같음 → 0
          np.array([0.5, 0.3, 0.2]),             # 조금 다름 → 0.0851
          np.array([0.1, 0.2, 0.7])]:            # 많이 다름 → 1.1675
    print('8. KL(p||q) =', round(kl(p, q), 4))    # 0, 0.0851, 1.1675

# ─────────────────────────────────────────────────────────────
# 9. Keras 손실 객체 직접 호출 (객체를 만든 뒤 함수처럼 (y_true, y_pred) 넣기)
# ─────────────────────────────────────────────────────────────
from tensorflow.keras.losses import (
    MeanSquaredError,
    BinaryCrossentropy,
    CategoricalCrossentropy,
)
print('9. MSE:', float(MeanSquaredError()([[1., 2., 3.]], [[1., 2.5, 2.]])),        # 0.4167
      'BCE:', float(BinaryCrossentropy()([[1.]], [[0.8]])),                        # 0.2231 = −log 0.8
      'CCE:', float(CategoricalCrossentropy()([[0., 1., 0.]], [[0.1, 0.8, 0.1]])))  # 0.2231 = −log 0.8

# ─────────────────────────────────────────────────────────────
# 10. compile(loss='mse')의 'mse' = 샘플마다 mean((y − ŷ)², 마지막 축) → 배치 평균
#     (출력이 여러 개일 때: 먼저 출력 축 평균, 다음에 샘플 평균)
# ─────────────────────────────────────────────────────────────
yt = np.array([[1., 2.], [3., 4.]])      # 샘플 2개 × 출력 2개
yp = np.array([[1.5, 2.], [2., 5.]])
manual = np.mean(np.mean((yt - yp) ** 2, axis=-1))   # 샘플별 [0.125, 1.0] → 평균 0.5625
print('10.', manual, float(MeanSquaredError()(yt, yp)))   # 0.5625 0.5625
```

**도전 11~12: composite loss = 0.7·MSE + 0.3·BCE, 가중치 바꾸기**

```python
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

# 손실 객체는 한 번만 만들어 두고 재사용
mse_fn, bce_fn = MeanSquaredError(), BinaryCrossentropy()

# 가중치를 인자로 받는 "손실 함수를 만들어 주는 함수" (closure)
#   → compile(loss=make_composite_loss(0.7, 0.3)) 처럼 괄호를 붙여 '안쪽 함수'를 넘김
def make_composite_loss(w_mse=0.7, w_bce=0.3):
    def composite_loss(y_true, y_pred):          # Keras 규칙: (y_true, y_pred) 순서
        return w_mse * mse_fn(y_true, y_pred) + w_bce * bce_fn(y_true, y_pred)
    return composite_loss

y_true = tf.constant([[1.0], [0.0], [1.0]])
y_pred = tf.constant([[0.8], [0.3], [0.4]])
mse_v, bce_v = float(mse_fn(y_true, y_pred)), float(bce_fn(y_true, y_pred))
print(f'MSE={mse_v:.4f}, BCE={bce_v:.4f}')        # 손으로 계산: MSE=0.1633, BCE≈0.4987

for w in [(0.7, 0.3), (0.3, 0.7)]:
    total = float(make_composite_loss(*w)(y_true, y_pred))   # *w → (w_mse, w_bce)로 풀어서 전달
    print(f'{w}: total={total:.4f}  (직접: {w[0] * mse_v + w[1] * bce_v:.4f})')
    # 손 계산: (0.7, 0.3) → 약 0.2639, (0.3, 0.7) → 약 0.3981
# BCE 값이 MSE보다 크므로 BCE 가중치를 키우면(0.3/0.7) total loss가 커지고, 학습도 BCE 쪽 목표를 더 중시

# 모델 학습에 사용 (입력 3개의 합이 1.5보다 크면 1인 가짜 이진 분류 데이터)
tf.keras.utils.set_random_seed(0)
Xd = np.random.rand(200, 3).astype('float32')
yd = (Xd.sum(axis=1) > 1.5).astype('float32')
model = Sequential([
    Input(shape=(3,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid'),      # BCE를 쓰므로 출력은 0~1 확률
])
model.compile(optimizer='adam',
              loss=make_composite_loss(0.7, 0.3),   # 괄호 O: 바깥 함수를 실행해 안쪽 함수를 받음
              metrics=['accuracy'])
h = model.fit(Xd, yd, epochs=20, verbose=0)
print('final loss:', h.history['loss'][-1], 'acc:', h.history['accuracy'][-1])
```

**L07 실습 함수 직접 구현 (노트북 함수 이름 그대로)**

`bernoulli_likelihood`, `bernoulli_log_likelihood`, `mae`, `mse`, `cosine_similarity`, `cosine_loss`, `kl_divergence`를 NumPy만으로 만든 버전입니다. 시험에서 "직접 함수로 구현하시오"가 나오면 이 블록을 그대로 쓰면 됩니다.

```python
import numpy as np

# ── 1) Bernoulli likelihood: L(θ) = Π θ^x · (1−θ)^(1−x)
#    x: 0/1 관측 배열, theta: 1이 나올 확률 (0 < θ < 1)
def bernoulli_likelihood(theta, x):
    x = np.asarray(x, dtype=float)
    return np.prod(theta ** x * (1 - theta) ** (1 - x))

# ── 2) Bernoulli log-likelihood: log L(θ) = Σ [x·log θ + (1−x)·log(1−θ)]
#    곱 대신 합 → 샘플이 많아도 0으로 언더플로가 나지 않음. NLL = −(이 값)
def bernoulli_log_likelihood(theta, x):
    x = np.asarray(x, dtype=float)
    return np.sum(x * np.log(theta) + (1 - x) * np.log(1 - theta))

# ── 3) 회귀 손실: MAE = mean|y − ŷ|, MSE = mean (y − ŷ)²
def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))

def mse(y_true, y_pred):
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)

# ── 4) 코사인 유사도 cos = a·b / (‖a‖‖b‖),  코사인 손실 = 1 − cos
def cosine_similarity(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_loss(a, b):
    return 1.0 - cosine_similarity(a, b)

# ── 5) KL divergence: KL(p‖q) = Σ p · log(p / q)
#    eps로 잘라서(clip) log(0), 0으로 나누기 방지 → p에 0이 있어도 계산 가능
def kl_divergence(p, q, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    return np.sum(p * np.log(p / q))

# ── 사용 예
x = np.array([1, 1, 0, 1, 0, 1, 1, 0])
print('L(0.625)     =', round(bernoulli_likelihood(0.625, x), 6))
print('logL(0.625)  =', round(bernoulli_log_likelihood(0.625, x), 4))
print('log(L) 확인  =', round(np.log(bernoulli_likelihood(0.625, x)), 4))   # 위와 같음
print('MAE, MSE     =', mae([1, 2, 3], [1, 2.5, 2]), round(mse([1, 2, 3], [1, 2.5, 2]), 4))
print('cos, 1-cos   =', round(cosine_similarity([1, 0], [1, 1]), 4),       # 45도 → 0.7071
      round(cosine_loss([1, 0], [1, 1]), 4))
print('KL           =', round(kl_divergence([0.7, 0.2, 0.1], [0.5, 0.3, 0.2]), 4),
      round(kl_divergence([0.5, 0.5, 0.0], [0.5, 0.25, 0.25]), 4))           # p에 0이 있어도 OK
```

```text
L(0.625)     = 0.005029
logL(0.625)  = -5.2925
log(L) 확인  = -5.2925
MAE, MSE     = 0.5 0.4167
cos, 1-cos   = 0.7071 0.2929
KL           = 0.0851 0.3466
```

**꼭 기억할 것**
1. Loss는 model prediction과 target 사이의 mismatch를 **하나의 scalar**로 요약합니다.
2. 어떤 loss를 쓸지는 **task와 output의 의미**에 따라 달라집니다 (회귀 MSE/MAE, 이진 BCE, 다중 CCE, 분포 KL, 표현 유사도 cosine).
3. 많은 supervised loss는 **likelihood와 NLL** 관점으로 해석됩니다 (Likelihood 최대화 = Log-Likelihood 최대화 = NLL 최소화).
4. 여러 목적을 동시에 학습할 때는 **composite loss**를 씁니다.

[⬆ 목차로 돌아가기](#목차)

### 17-7. L08 신경망 학습 (Optimizer, LR, Initialization)

**검색 키워드:** optimizer, 옵티마이저, gradient descent, 경사하강법, full-batch GD, mini-batch SGD, 미니배치, momentum, 모멘텀, Adam, learning rate, 학습률, warm-up, cosine decay, 스케줄, initialization, 초기화, Xavier, Glorot, He, train_full_batch_gd, train_minibatch_sgd, train_momentum, initialize_weight

**공통 준비 (아래 L08 블록들보다 먼저 실행)**

```python
# 데이터 6개 (공부 시간 → 점수). 둘 다 표준화해서 w, b가 0 근처에서 시작해도 잘 수렴하게 함
x_raw = np.array([1, 3, 5, 7, 9, 11], dtype=float)
y_raw = np.array([75, 77, 85, 83, 90, 91], dtype=float)
x = (x_raw - x_raw.mean()) / x_raw.std()      # 평균 0, 표준편차 1
y = (y_raw - y_raw.mean()) / y_raw.std()

# 모델 ŷ = w·x + b 의 MSE와 기울기를 한 번에 반환
#   L = mean((ŷ − y)²)
#   ∂L/∂w = 2·mean((ŷ − y)·x),  ∂L/∂b = 2·mean(ŷ − y)
def mse_and_grad(w, b, xb, yb):
    error = (w * xb + b) - yb
    return np.mean(error ** 2), 2 * np.mean(error * xb), 2 * np.mean(error)

def train_sgd(lr=0.05, epochs=100, batch_size=2, momentum=0.0, seed=0):
    """batch_size=len(x)이면 full-batch GD, momentum>0이면 momentum SGD (v = μv + g, w −= lr·v)"""
    rng = np.random.default_rng(seed)
    w = b = v_w = v_b = 0.0          # 파라미터와 속도(velocity) 모두 0에서 시작
    losses = []
    updates = 0                      # 파라미터를 몇 번 바꿨는지 셈
    for _ in range(epochs):
        idx = rng.permutation(len(x))                 # 매 epoch마다 순서 섞기
        for s in range(0, len(x), batch_size):        # 미니배치 단위로 자름
            bi = idx[s:s + batch_size]
            _, gw, gb = mse_and_grad(w, b, x[bi], y[bi])
            v_w = momentum * v_w + gw                 # 이전 방향을 μ만큼 기억 (μ=0이면 그냥 SGD)
            v_b = momentum * v_b + gb
            w -= lr * v_w
            b -= lr * v_b
            updates += 1
        losses.append(mse_and_grad(w, b, x, y)[0])   # epoch 끝에 전체 데이터 손실 기록
    return np.array(losses), updates

def train_adam(lr=0.05, epochs=100, batch_size=2, b1=0.9, b2=0.999, eps=1e-8, seed=0):
    rng = np.random.default_rng(seed)
    w = b = 0.0
    m = np.zeros(2)      # 1차 모멘트 (기울기의 지수 이동 평균)  [w, b]
    v = np.zeros(2)      # 2차 모멘트 (기울기 제곱의 지수 이동 평균)
    t = 0                # 업데이트 횟수 (편향 보정에 사용)
    losses = []
    for _ in range(epochs):
        idx = rng.permutation(len(x))
        for s in range(0, len(x), batch_size):
            bi = idx[s:s + batch_size]
            _, gw, gb = mse_and_grad(w, b, x[bi], y[bi])
            g = np.array([gw, gb])
            t += 1
            m = b1 * m + (1 - b1) * g                 # m_t = β1·m + (1−β1)·g
            v = b2 * v + (1 - b2) * g ** 2            # v_t = β2·v + (1−β2)·g²
            m_hat = m / (1 - b1 ** t)                 # 편향 보정 (초반에 0 쪽으로 쏠림 방지)
            v_hat = v / (1 - b2 ** t)
            step = lr * m_hat / (np.sqrt(v_hat) + eps)   # 파라미터별로 크기가 조절된 보폭
            w -= step[0]
            b -= step[1]
        losses.append(mse_and_grad(w, b, x, y)[0])
    return np.array(losses)
```

**Optimizer 1~4**

```python
# 1. Full-batch GD vs mini-batch SGD의 update 횟수 (데이터 6개, 100 epoch)
_, n_gd = train_sgd(batch_size=6)       # 한 epoch에 1번 업데이트
_, n_sgd = train_sgd(batch_size=2)      # 한 epoch에 6/2 = 3번 업데이트
print('1. GD updates =', n_gd, '/ SGD(batch 2) updates =', n_sgd)    # 100 / 300

# 2. batch_size = 1, 2, 6 비교 (작을수록 update가 많고 곡선이 들쭉날쭉)
#    (확인값: 100 epoch 뒤에는 셋 다 손실 ≈ 0.0925 로 수렴)
for bs in [1, 2, 6]:
    plt.plot(train_sgd(batch_size=bs)[0], label=f'batch_size={bs}')
plt.yscale('log')                       # 손실이 작아지는 모습을 보려고 로그 축
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.title('2. Batch size')
plt.show()

# 3. momentum 0.0, 0.5, 0.9 (lr 0.02)
#    (확인값: 10 epoch 손실 0.167 / 0.097 / 0.142 → 0.5가 가장 빠름, 0.9는 관성 때문에 오버슈트)
for mu in [0.0, 0.5, 0.9]:
    plt.plot(train_sgd(lr=0.02, momentum=mu)[0], label=f'momentum={mu}')
plt.yscale('log'); plt.xlabel('Epoch'); plt.legend(); plt.title('3. Momentum')
plt.show()

# 4. Adam learning rate 비교 (너무 작으면 느리고, 너무 크면 최저점 근처에서 흔들림)
for lr in [0.001, 0.01, 0.05, 0.2]:
    plt.plot(train_adam(lr=lr), label=f'Adam lr={lr}')
plt.yscale('log'); plt.xlabel('Epoch'); plt.legend(); plt.title('4. Adam LR')
plt.show()
```

**Learning Rate 5~7**

```python
# 5. 너무 작은 / 적절한 / 너무 큰 LR (full-batch GD)
def train_with_lr(lr, epochs=40):
    w = b = 0.0
    losses = []
    for _ in range(epochs):
        loss, gw, gb = mse_and_grad(w, b, x, y)
        losses.append(loss)
        w -= lr * gw
        b -= lr * gb
        if not np.isfinite(loss) or loss > 1e6:   # 발산하면 중단 (inf, nan 방지)
            break
    return np.array(losses)

for lr in [0.001, 0.05, 1.2]:
    plt.plot(train_with_lr(lr), label=f'lr={lr}')
plt.yscale('log'); plt.legend(); plt.title('5. LR')
plt.show()
# 확인값(40 epoch): lr=0.001 → 0.869 (거의 안 내려감), lr=0.05 → 0.093 (수렴),
#                   lr=1.2 → 22 epoch 만에 1e6 초과로 발산해서 중단

# 6~7. warm-up step 수 변경, constant vs cosine decay
#   warm-up 구간 (s < warmup): lr = base_lr · s / warmup      (0에서 직선으로 증가)
#   그 뒤 cosine 구간         : lr = base_lr · 0.5 · (1 + cos(π · 진행률))  (base_lr → 0)
def warmup_cosine(total=100, warmup=10, base_lr=1e-3):
    lr = np.zeros(total)
    for s in range(total):
        if s < warmup:
            lr[s] = base_lr * s / warmup
        else:
            progress = (s - warmup) / (total - warmup - 1)   # 0 → 1
            lr[s] = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    return lr

steps = np.arange(100)
plt.plot(steps, np.full(100, 1e-3), label='Constant')
plt.plot(steps, 1e-3 * 0.5 * (1 + np.cos(np.pi * steps / 99)), label='Cosine (no warm-up)')
for wu in [5, 10, 30]:
    plt.plot(steps, warmup_cosine(warmup=wu), label=f'Warm-up {wu} + Cosine')
plt.xlabel('Step'); plt.ylabel('LR'); plt.legend(); plt.title('6~7. LR schedules')
plt.show()
# Keras에서는:
#   schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=1000)
#   Adam(learning_rate=schedule)      # decay_steps는 epoch가 아니라 '배치 step' 수
```

**Initialization 8~11**

```python
# depth층짜리 (width × width) 신경망에 정규분포 입력을 통과시키며 층별 출력 표준편차 기록
#   method: 가중치 표준편차 선택
#     'small'  → 0.01           (너무 작음: 신호가 층마다 줄어듦)
#     'large'  → 0.10           (실습용 '큰' 값. width=128이면 Xavier 0.088과 비슷해 포화는 약함)
#     'xavier' → sqrt(1/width)  (tanh에 적합, Glorot)
#     'he'     → sqrt(2/width)  (ReLU에 적합. ReLU가 절반을 0으로 만드는 것을 2배로 보정)
def forward_activation_stats(method, activation, depth=6, width=128, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 1, size=(n, width))     # 입력 (샘플 n개 × 특성 width개)
    stds = []
    acts = []
    std = {'small': 0.01,
           'large': 0.10,
           'xavier': np.sqrt(1 / width),
           'he': np.sqrt(2 / width)}[method]
    for _ in range(depth):
        z = h @ rng.normal(0, std, size=(width, width))         # Affine (bias 생략)
        h = np.tanh(z) if activation == 'tanh' else np.maximum(0, z)
        stds.append(h.std())
        acts.append(h)
    return np.array(stds), acts

layers = np.arange(1, 7)          # x축: 1~6층
s_small, _ = forward_activation_stats('small', 'tanh')
print('8. small+tanh std:', np.round(s_small, 5))
# 출력: [1.1115e-01 1.2750e-02 1.4400e-03 1.6000e-04 2.0000e-05 0.0000e+00]
# = 0.111 → 0.0128 → 0.0014 → 0.00016 → 0.00002 → 0 → 층마다 약 1/9씩 작아짐 → 신호 소멸 (vanishing)

_, acts_large = forward_activation_stats('large', 'tanh')
plt.hist(acts_large[-1].ravel(), bins=50)
plt.title('9. Large init + tanh: layer 6')
plt.show()
# 주의(확인값): 6층 출력 std ≈ 0.39로 ±1에 몰리지 않음 (0.10이 Xavier 0.088과 비슷해서).
#   포화(saturation)를 확실히 보려면 std를 1.0으로 올리면 됨 → 6층에서 |h| > 0.99인 값이 약 81%
#   → ±1 근처에 몰린 tanh는 기울기 1 − tanh² ≈ 0 → gradient 소실

for method, act in [('xavier', 'tanh'), ('xavier', 'relu'), ('he', 'relu')]:     # 10~11
    s, _ = forward_activation_stats(method, act)
    print(f'{method}+{act}:', np.round(s, 4))
    plt.plot(layers, s, marker='o', label=f'{method} + {act}')
plt.plot(layers, s_small, marker='o', label='small + tanh')
plt.xlabel('Layer'); plt.ylabel('Activation std'); plt.legend()
plt.show()
# 확인값:
#   xavier+tanh: [0.6249 0.4884 0.4086 0.3549 0.3099 0.2835]  (천천히 줄어듦, small보다 훨씬 안정)
#   xavier+relu: [0.5805 0.4101 0.2962 0.2283 0.1682 0.1076]  (층마다 줄어듦: ReLU가 절반을 0으로)
#   he+relu    : [0.821  0.8202 0.8376 0.9132 0.9514 0.8605]  (거의 일정 → He가 ReLU에 맞는 이유)
```

**L08 실습 함수 직접 구현 (노트북 함수 이름 그대로)**

`train_full_batch_gd`, `train_minibatch_sgd`, `train_momentum`, `initialize_weight`입니다. 위의 `train_sgd`를 노트북처럼 셋으로 나눈 버전이고, 최종 `w`, `b`와 손실 기록, 업데이트 경로까지 돌려줍니다. 위 **공통 준비** 블록(`x`, `y`, `mse_and_grad`)을 먼저 실행하세요.

```python
# ── 1) Full-batch GD: 매 epoch마다 '전체 데이터'로 기울기 1번 계산 → 업데이트 1번
def train_full_batch_gd(x, y, lr=0.05, epochs=100):
    w, b = 0.0, 0.0
    losses = []          # epoch별 손실 (업데이트 '전' 손실)
    path = []            # 업데이트 후 (w, b) 기록 → 길이 = 업데이트 횟수
    for epoch in range(epochs):
        loss, grad_w, grad_b = mse_and_grad(w, b, x, y)
        w -= lr * grad_w                 # w ← w − η·∂L/∂w
        b -= lr * grad_b
        losses.append(loss)
        path.append((w, b))
    return w, b, np.array(losses), np.array(path)

# ── 2) Mini-batch SGD: epoch마다 섞고 batch_size개씩 잘라 여러 번 업데이트
def train_minibatch_sgd(x, y, lr=0.05, epochs=100, batch_size=2, seed=0):
    rng = np.random.default_rng(seed)
    w, b = 0.0, 0.0
    epoch_losses = []
    path = []
    n = len(x)
    for epoch in range(epochs):
        idx = rng.permutation(n)                       # 0..n-1을 무작위 순서로
        for start in range(0, n, batch_size):
            bi = idx[start:start + batch_size]         # 이번 미니배치의 인덱스
            _, grad_w, grad_b = mse_and_grad(w, b, x[bi], y[bi])
            w -= lr * grad_w
            b -= lr * grad_b
            path.append((w, b))
        epoch_losses.append(mse_and_grad(w, b, x, y)[0])   # epoch 끝: 전체 데이터 손실
    return w, b, np.array(epoch_losses), np.array(path)

# ── 3) Momentum SGD: v ← μ·v + g,  w ← w − η·v   (지난 방향을 μ만큼 이어감)
def train_momentum(x, y, lr=0.05, momentum=0.9, epochs=100, batch_size=2, seed=0):
    rng = np.random.default_rng(seed)
    w, b = 0.0, 0.0
    v_w, v_b = 0.0, 0.0                  # 속도(velocity)는 0에서 시작
    epoch_losses = []
    n = len(x)
    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            bi = idx[start:start + batch_size]
            _, grad_w, grad_b = mse_and_grad(w, b, x[bi], y[bi])
            v_w = momentum * v_w + grad_w
            v_b = momentum * v_b + grad_b
            w -= lr * v_w
            b -= lr * v_b
        epoch_losses.append(mse_and_grad(w, b, x, y)[0])
    return w, b, np.array(epoch_losses)

# ── 4) 가중치 초기화: (fan_in, fan_out) 모양의 정규분포 가중치
#    fan_in = 입력 뉴런 수, fan_out = 출력 뉴런 수
def initialize_weight(fan_in, fan_out, method, rng):
    if method == 'small':
        std = 0.01
    elif method == 'large':
        std = 0.10                       # 실습용 '큰' 값
    elif method == 'xavier':
        std = np.sqrt(1.0 / fan_in)      # Xavier(Glorot) normal: tanh/sigmoid에 적합
    elif method == 'he':
        std = np.sqrt(2.0 / fan_in)      # He normal: ReLU에 적합
    else:
        raise ValueError('Unknown initialization method: ' + method)
    return rng.normal(0.0, std, size=(fan_in, fan_out))

# ── 사용 예
w_gd, b_gd, loss_gd, path_gd = train_full_batch_gd(x, y, lr=0.05, epochs=100)
print('GD      : w=%.4f b=%.4f updates=%d final loss=%.4f'
      % (w_gd, b_gd, len(path_gd), loss_gd[-1]))
w_s, b_s, loss_s, path_s = train_minibatch_sgd(x, y, lr=0.05, epochs=100, batch_size=2)
print('SGD     : w=%.4f b=%.4f updates=%d final loss=%.4f'
      % (w_s, b_s, len(path_s), loss_s[-1]))
w_m, b_m, loss_m = train_momentum(x, y, lr=0.02, momentum=0.9)
print('Momentum: w=%.4f b=%.4f final loss=%.4f' % (w_m, b_m, loss_m[-1]))
# 표준화된 x, y에서 최소제곱 해: w = mean(x·y) (= 상관계수), b = 0
print('정답(최소제곱) w =', round(np.mean(x * y), 4))

rng = np.random.default_rng(0)
for m in ['small', 'large', 'xavier', 'he']:
    W = initialize_weight(128, 64, m, rng)
    print(f'{m:7s} shape={W.shape} std={W.std():.4f}')
```

```text
GD      : w=0.9526 b=0.0000 updates=100 final loss=0.0925
SGD     : w=0.9506 b=0.0005 updates=300 final loss=0.0925
Momentum: w=0.9592 b=0.0018 final loss=0.0926
정답(최소제곱) w = 0.9526
small   shape=(128, 64) std=0.0100
large   shape=(128, 64) std=0.0991
xavier  shape=(128, 64) std=0.0882
he      shape=(128, 64) std=0.1252
```

**생각해 보기 12~13 모범답안**
- **12.** Optimizer는 gradient를 이용해 **어떤 규칙으로 움직일지**(모멘텀 누적, 파라미터별 적응 step 등)를 바꾸고, learning rate는 그 규칙으로 정한 방향으로 **한 번에 얼마나 크게** 움직일지를 바꿉니다.
- **13.** 초기화가 나쁘면 첫 forward부터 activation이 0으로 사라지거나(너무 작음) 포화되어(너무 큼) gradient가 거의 0이 됩니다. 기울기가 없으면 어떤 optimizer도 의미 있는 업데이트를 할 수 없고, 0 초기화는 뉴런들이 대칭이라 같은 값만 학습합니다.

**꼭 기억할 것**
1. **Optimizer**는 gradient를 이용해 parameter를 **어떻게** 움직일지 결정합니다.
2. **Learning rate**는 한 번에 **얼마나 크게** 움직일지 결정합니다 (작으면 느림, 크면 불안정·발산).
3. **Initialization**은 학습을 시작할 때 activation과 gradient의 **scale**을 결정합니다 (Xavier + tanh, He + ReLU가 비교적 안정).

[⬆ 목차로 돌아가기](#목차)

### 17-8. L09 학습 안정화 (Activation, Scaling, BN, LN)

**검색 키워드:** 활성화 함수, activation, sigmoid, tanh, ReLU, Leaky ReLU, 도함수, 기울기, sigmoid_grad, vanishing gradient, 기울기 소실, zero-centered, input scaling, 스케일링, StandardScaler, MinMaxScaler, fit_transform, transform, 데이터 누수, BatchNorm, 배치 정규화, LayerNorm, 층 정규화, gamma, beta

**Activation 1~4**

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 1. sigmoid 미분 σ'(z) = σ(z)(1 − σ(z)) 의 최댓값과 위치
xs = np.linspace(-10, 10, 2001)               # 간격 0.01 → 0이 정확히 포함됨
grad = sigmoid(xs) * (1 - sigmoid(xs))
print('1. sigmoid 미분 최대 위치 x =', xs[np.argmax(grad)],
      ', 최댓값 =', grad.max())              # x = 0, 0.25
# → 여러 층을 지나면 0.25를 계속 곱해 기울기가 작아짐 (vanishing gradient)

# 2. 출력 평균: sigmoid는 항상 양수(0~1) → 평균 ≈ 0.5 (zero-centered 아님)
#              tanh는 −1~1 대칭 → 평균 ≈ 0
rng = np.random.default_rng(0)
z = rng.normal(0, 2, 10000)
print('2. mean sigmoid =', sigmoid(z).mean(),
      '/ mean tanh =', np.tanh(z).mean())     # ≈0.502 / ≈0.004

# 3. ReLU 미분: z > 0 이면 1, 아니면 0
zz = np.array([-3.0, -1.0, -0.2, 0.0, 0.5, 2.0])
print('3. ReLU grad =', (zz > 0).astype(float))     # [0 0 0 0 1 1] → 음수 영역 기울기 0 (dying ReLU)

# 4. Leaky ReLU 미분: z ≥ 0 이면 1, 음수면 alpha (기울기가 0이 되지 않음)
for alpha in [0.01, 0.1, 0.3]:
    print(f'4. LeakyReLU alpha={alpha} grad =',
          np.where(zz >= 0, 1.0, alpha))      # 음수 영역 기울기 = alpha
```

**L09 실습 함수 직접 구현: `tanh`, `sigmoid_grad` (+ `tanh_grad`)**

```python
np.set_printoptions(precision=4, suppress=True)   # 지수 표기 없이 소수 4자리

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# tanh(z) = (e^z − e^−z) / (e^z + e^−z),  출력 범위 (−1, 1), 0을 중심으로 대칭
def tanh(z):
    return np.tanh(z)

# numpy 함수 없이 공식 그대로 (시험에서 '직접 구현'을 요구할 때)
def tanh_manual(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

# σ'(z) = σ(z)·(1 − σ(z)),  최대 0.25 (z = 0)
def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1.0 - s)

# tanh'(z) = 1 − tanh²(z),  최대 1 (z = 0) → sigmoid보다 기울기가 4배 큼
def tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2

z = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
print('tanh        :', np.round(tanh(z), 4))
print('tanh 직접   :', np.round(tanh_manual(z), 4))
print('sigmoid_grad:', np.round(sigmoid_grad(z), 4))
print('tanh_grad   :', np.round(tanh_grad(z), 4))
print('10층 sigmoid 최대 기울기 곱 0.25**10 =', 0.25 ** 10)
```

```text
tanh        : [-0.9999 -0.7616  0.      0.7616  0.9999]
tanh 직접   : [-0.9999 -0.7616  0.      0.7616  0.9999]
sigmoid_grad: [0.0066 0.1966 0.25   0.1966 0.0066]
tanh_grad   : [0.0002 0.42   1.     0.42   0.0002]
10층 sigmoid 최대 기울기 곱 0.25**10 = 9.5367431640625e-07
```

**Input Scaling 5~8**

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 두 열의 단위(크기)가 100배 차이 나는 데이터
X = np.array([[10.0, 1000.0],
              [20.0, 1500.0],
              [30.0, 2000.0],
              [40.0, 2500.0],
              [50.0, 3000.0]])

# MinMax: (x − min) / (max − min) → 열마다 0~1
print('5. MinMax:\n', MinMaxScaler().fit_transform(X))
# [[0 0] [0.25 0.25] [0.5 0.5] [0.75 0.75] [1 1]]

# Standard: (x − 평균) / 표준편차 → 열마다 평균 0, 표준편차 1
X_std = StandardScaler().fit_transform(X)
print('5. Standard:\n', X_std)                 # [−1.414, −0.707, 0, 0.707, 1.414] (두 열 같음)
print('6. mean =', X_std.mean(axis=0),
      'std =', X_std.std(axis=0))             # [0, 0], [1, 1]

# 7. test에 fit_transform을 하면 test 자신의 평균/표준편차로 변환됨 (잘못된 방법)
X_train = np.array([[1.0, 100.0], [2.0, 120.0], [3.0, 140.0], [4.0, 160.0]])
X_test = np.array([[5.0, 180.0], [6.0, 200.0]])
sc = StandardScaler().fit(X_train)            # 평균·표준편차는 train에서만 계산
print('7. 올바름 transform:\n', sc.transform(X_test))
# [[2.236 2.236] [3.130 3.130]] → train 기준으로 '큰 값'이라는 정보가 유지됨
print('7. 잘못됨 fit_transform:\n', StandardScaler().fit_transform(X_test))
# [[-1 -1] [1 1]] → test만의 평균으로 다시 맞춰서 의미가 달라짐
# 이유: 학습한 모델은 train 기준 스케일을 가정함 + test 정보를 미리 쓰는 데이터 누수
```

```python
# 8. MNIST에서 MinMaxScaler(÷255) 대신 StandardScaler
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32')     # (60000, 28, 28) → (60000, 784)
x_test = x_test.reshape(-1, 784).astype('float32')

scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)    # 784개 픽셀 열마다 평균·표준편차 계산
x_test_s = scaler.transform(x_test)          # test는 transform만
print('8. mean ≈', x_train_s.mean().round(4), ', std ≈', x_train_s.std().round(4))
# 테두리 픽셀처럼 항상 0인 열은 표준편차가 0 → sklearn이 scale=1로 처리해 0으로 유지됨
# (그래서 전체 std는 정확히 1이 아니라 1보다 조금 작게 나옴)
```

**Batch Normalization 9~12, Layer Normalization 13~15**

```python
# 샘플 4개(행) × 특성 3개(열). 열마다 크기가 다름
Z = np.array([[1., 10., 100.],
              [2., 12., 80.],
              [3., 14., 120.],
              [4., 16., 100.]])
eps = 1e-5      # 분산이 0일 때 0으로 나누는 것 방지

# 9. 각 column(특성)을 직접 표준화 = BN 1단계 (배치 방향 axis=0)
#    ẑ = (z − μ_batch) / sqrt(σ²_batch + ε)
Z_bn = (Z - Z.mean(axis=0)) / np.sqrt(Z.var(axis=0) + eps)
print('9.', Z_bn.mean(axis=0).round(6), Z_bn.std(axis=0).round(4))      # [0 0 0] [1 1 1]

# 10. BN 2단계: y = γ·ẑ + β  → 출력 평균 = β, 표준편차 ≈ γ (학습으로 다시 조정 가능)
for gamma, beta in [(np.array([1., 2., .5]), np.array([0., 1., -1.])),
                    (np.array([3., 3., 3.]), np.array([5., 5., 5.]))]:
    S = gamma * Z_bn + beta
    print('10. mean =', S.mean(axis=0).round(4), 'std =', S.std(axis=0).round(4))
    # 첫 번째: mean [0 1 −1], std [1 2 0.5] / 두 번째: mean [5 5 5], std [3 3 3]

# 13~14. LN은 샘플(행) 방향 axis=1, keepdims=True로 (4,1) 모양을 유지해야 브로드캐스팅됨
Z_ln = (Z - Z.mean(axis=1, keepdims=True)) / np.sqrt(Z.var(axis=1, keepdims=True) + eps)
print('14. Z_bn.mean(axis=0) =', Z_bn.mean(axis=0).round(6),
      '/ Z_ln.mean(axis=1) =', Z_ln.mean(axis=1).round(6))
# 둘 다 0. 단, BN은 '열 평균'이 0, LN은 '행 평균'이 0
# 13. BN: 같은 뉴런(열)을 배치 안의 여러 샘플로 정규화 / LN: 한 샘플(행) 안의 여러 뉴런으로 정규화
```

```python
# 11~12, 15. BN을 활성화 이전에 둔 모델, BN 유무 learning curve 비교, LayerNormalization으로 교체
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import LayerNormalization

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0    # 0~1 정규화
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# norm=None이면 일반 MLP, norm=BatchNormalization 또는 LayerNormalization이면 그 층을 끼움
def build(norm=None):
    layers = [Input(shape=(784,))]
    for units in [256, 128]:
        if norm is None:
            layers.append(Dense(units, activation='relu'))
        else:
            # 11. Dense(활성화 없이) → Norm → 활성화 순서
            layers += [Dense(units), norm(), Activation('relu')]
    layers.append(Dense(10, activation='softmax'))    # 10클래스 확률
    m = Sequential(layers)
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # 라벨이 정수 0~9
              metrics=['accuracy'])
    return m

histories = {}
for name, norm in [('No BN', None),
                   ('BN', BatchNormalization),
                   ('LN', LayerNormalization)]:
    tf.keras.utils.set_random_seed(42)                  # 세 모델 공정 비교를 위해 같은 시드
    histories[name] = build(norm).fit(
        x_train, y_train,
        epochs=3, batch_size=32,
        validation_data=(x_test, y_test),
        verbose=0,
    )
    print(name, 'val_acc =', [round(v, 4) for v in histories[name].history['val_accuracy']])

for name, h in histories.items():                       # 12. learning curve 비교
    plt.plot(h.history['val_loss'], marker='o', label=name)
plt.xlabel('Epoch'); plt.ylabel('Validation loss'); plt.legend()
plt.show()
```

**꼭 기억할 것**
1. Activation function은 **gradient flow**에 직접 영향을 줍니다 (sigmoid 미분 최대 0.25 → vanishing, ReLU 음수 영역 0, Leaky ReLU는 α 유지).
2. Input scaling은 optimization을 쉽게 만드는 기본 전처리입니다. **Train → `fit_transform()`, Test → `transform()`**
3. BatchNorm은 **mini-batch 방향(열)**으로 뉴런별 activation을 정규화합니다. 학습 때는 배치 통계, 추론 때는 이동(moving) 통계를 씁니다.
4. LayerNorm은 **샘플 내부 feature 방향(행)**으로 정규화합니다.
5. Normalization의 목적은 activation과 gradient의 **scale을 안정적으로 유지**하는 것입니다.

[⬆ 목차로 돌아가기](#목차)

### 17-9. L10 정규화 (Regularization)

**검색 키워드:** 정규화, regularization, 과적합, overfitting, 과소적합, capacity, 다항 회귀, PolynomialFeatures, L1, L2, Ridge, Lasso, 희소성, sparsity, weight decay, AdamW, kernel_regularizer, Dropout, 데이터 증강, augmentation, ImageDataGenerator, 시계열 증강, jitter, time shift, scaling, masking, objective, fit_scalar_l2

**Overfitting / Capacity 1~2**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
f = lambda t: 0.5 * t ** 2 - t + 2                     # 실제 함수 (2차)
x_tr = np.linspace(-3, 3, 18)                          # train 18개
y_tr = f(x_tr) + rng.normal(0, 1.2, len(x_tr))         # 잡음 추가
x_te = rng.uniform(-3, 3, 200)                         # 새 데이터 200개 (test)
y_te = f(x_te) + rng.normal(0, 1.2, 200)
x_grid = np.linspace(-3.2, 3.2, 400)                   # 곡선 그리기용

for degree in [1, 2, 5, 14]:                                                     # 1. degree 비교
    # PolynomialFeatures(d): x → [1, x, x², …, x^d] 로 특성 확장 후 선형 회귀
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_tr.reshape(-1, 1), y_tr)               # sklearn 입력은 2차원 (N, 1)
    tr_mse = np.mean((model.predict(x_tr.reshape(-1, 1)) - y_tr) ** 2)
    te_mse = np.mean((model.predict(x_te.reshape(-1, 1)) - y_te) ** 2)
    print(f'degree={degree:2d}: train MSE={tr_mse:.3f}, test MSE={te_mse:.3f}')
    plt.plot(x_grid, model.predict(x_grid.reshape(-1, 1)), label=f'degree {degree}')
plt.scatter(x_tr, y_tr, color='k'); plt.ylim(-5, 15); plt.legend()
plt.show()
# 결과 (train / test MSE): degree 1 = 3.14 / 3.96 (과소적합), degree 2 = 0.77 / 1.84 (test 최소, 실제 함수가 2차),
#                          degree 5 = 0.57 / 2.05, degree 14 = 1.43 / 2.37
# 2. degree가 커지면 train 점들 사이에서 곡선이 크게 흔들림(그림) → noise까지 따라가 test 오차가 커짐 = 과적합
#    (degree 14는 x^14 값이 너무 커서 수치 계산이 불안정해 train MSE가 오히려 올라갈 수 있음)
```

**L1 / L2 3~6**

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# 특성 8개 중 3개만 실제로 y에 영향 (나머지 5개는 쓸모없는 특성)
X, y_reg = make_regression(n_samples=200, n_features=8, n_informative=3,
                           noise=15.0, random_state=0)
X_s = StandardScaler().fit_transform(X)                # L1/L2 전에는 스케일링 필수 (6번 참고)

for alpha in [0.01, 1.0, 10.0, 100.0]:                                             # 3~4
    ridge = Ridge(alpha=alpha).fit(X_s, y_reg)                  # L2: 손실 + α·Σw²
    lasso = Lasso(alpha=alpha, max_iter=20000).fit(X_s, y_reg)  # L1: 손실 + α·Σ|w|
    print(f'alpha={alpha:>6}: Ridge={np.round(ridge.coef_, 1)}')
    print(f'              Lasso={np.round(lasso.coef_, 1)}'
          f'  → 정확히 0인 계수 {np.sum(lasso.coef_ == 0)}개')
# Ridge는 전체를 조금씩 축소, Lasso는 alpha가 커질수록 계수를 정확히 0으로 만듦(희소성)
# 확인값: Lasso의 0인 계수 개수 alpha 0.01→0개, 1→3개, 10→5개(쓸모없는 5개가 정확히 제거), 100→8개(전부 0)

# 5. L2 scalar 예제: λ가 커질수록 기울기 w가 0으로 축소
#    목적함수 J(w, b) = mean((w·xc + b − y)²) + λ·w²   (b에는 penalty 없음)
x = np.array([1, 3, 5, 7], dtype=float)
y = np.array([75, 77, 85, 83], dtype=float)
xc = x - x.mean()                                      # 중심화 → b는 y 평균(80)으로 고정됨
for lam in [0.0, 0.1, 1.0, 10.0]:
    w, b = 0.0, y.mean()
    for _ in range(5000):
        err = (w * xc + b) - y
        w -= 0.01 * (2 * np.mean(err * xc) + 2 * lam * w)   # ∂J/∂w = 2·mean(err·xc) + 2λw
        b -= 0.01 * (2 * np.mean(err))                      # ∂J/∂b = 2·mean(err)
    print(f'5. lambda={lam:>4}: w={w:.4f}, b={b:.4f}')
    # w: 1.60 → 1.57 → 1.33 → 0.53, b = 80
# 해석: 이 목적함수의 해는 w = mean(xc·y) / (mean(xc²) + λ) = 8 / (5 + λ) → λ가 커질수록 기울기가 0으로 축소

# 6. 스케일링 전/후 Ridge 계수 (두 특성이 비슷한 영향을 주는 데이터)
rng = np.random.default_rng(1)
x1 = rng.normal(0, 1, 200)                             # 단위가 작은 특성 (std 1)
x2 = rng.normal(0, 1000, 200)                          # 단위가 큰 특성 (std 1000)
y6 = 3.0 * x1 + 0.003 * x2 + rng.normal(0, 0.5, 200)   # 3·1 ≈ 0.003·1000 → 영향력 비슷
X6 = np.column_stack([x1, x2])
print('6. raw   :', Ridge(alpha=10.0).fit(X6, y6).coef_)                               # [2.889, 0.0029]
print('6. scaled:', Ridge(alpha=10.0).fit(StandardScaler().fit_transform(X6), y6).coef_)   # [2.697, 2.466]
# raw: 단위 때문에 계수 크기가 1000배 차이 → penalty(λw²)가 특성마다 전혀 다른 강도로 걸리고 계수 비교도 불가능
# scaled: 두 특성의 실제 영향이 비슷하다는 것이 계수에 그대로 드러나고 같은 기준으로 정규화됨 → L1/L2 전에 스케일링
```

**L10 실습 함수 직접 구현: `objective`, `fit_scalar_l2`**

노트북은 `x_centered`, `y`를 전역 변수로 씁니다. 여기서는 기본값으로 넣어 두었고, 다른 데이터면 `xc=`, `yv=`로 바꿔 넣으면 됩니다.

```python
x = np.array([1, 3, 5, 7], dtype=float)
y = np.array([75, 77, 85, 83], dtype=float)
x_centered = x - x.mean()            # [-3, -1, 1, 3]

# J(w, b) = mean((w·xc + b − y)²)  +  λ·w²
#           ───── data loss ─────    ─ L2 penalty ─
def objective(w, b, lam, xc=x_centered, yv=y):
    y_hat = w * xc + b
    data_loss = np.mean((y_hat - yv) ** 2)
    l2_penalty = lam * (w ** 2)
    return data_loss + l2_penalty

# 경사하강법으로 J를 최소화 (b는 y 평균에서 시작)
def fit_scalar_l2(lam, lr=0.01, steps=5000, xc=x_centered, yv=y):
    w, b = 0.0, yv.mean()
    for _ in range(steps):
        error = (w * xc + b) - yv
        grad_w = 2 * np.mean(error * xc) + 2 * lam * w     # penalty의 미분 2λw가 추가됨
        grad_b = 2 * np.mean(error)                        # b는 penalty 없음
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

for lam in [0.0, 0.1, 1.0, 10.0]:
    w, b = fit_scalar_l2(lam)
    print(f'lambda={lam:>4}: w={w:.4f}, b={b:.2f}, objective={objective(w, b, lam):.4f}')
# 검산: w = 8 / (5 + λ)
```

```text
lambda= 0.0: w=1.6000, b=80.00, objective=4.2000
lambda= 0.1: w=1.5686, b=80.00, objective=4.4510
lambda= 1.0: w=1.3333, b=80.00, objective=6.3333
lambda=10.0: w=0.5333, b=80.00, objective=12.7333
```

**Weight Decay 7~8**

```python
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2

# 7. AdamW: weight decay를 손실이 아니라 업데이트 단계에서 따로 적용 (decoupled)
#    w ← w − lr·(Adam 보폭) − lr·weight_decay·w
optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)

# 8. Adam + 층별 L2: 손실에 λ·Σw² 를 더함 (층마다 다른 λ)
model_l2 = Sequential([
    Input(shape=(784,)),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),   # λ = 0.001
    Dense(256, activation='relu', kernel_regularizer=l2(0.005)),   # λ = 0.005
    Dense(10, activation='softmax'),
])
model_l2.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
model_l2.summary()
print(model_l2.losses)    # 각 층의 L2 penalty 값 (학습 loss에 더해짐)
```

**Dropout 9~10**

```python
from tensorflow.keras.layers import Dropout

tf.random.set_seed(0)
sample = tf.ones((1, 10))            # 1이 10개인 입력

for rate in [0.1, 0.5, 0.8]:                                                       # 9
    out = Dropout(rate)(sample, training=True).numpy()     # training=True: 학습 모드로 강제
    print(f'rate={rate}: {np.round(out, 2)}'
          f'  (살아남은 값 = 1/(1-{rate}) = {1 / (1 - rate):.2f})')
print('training=False:', Dropout(0.5)(sample, training=False).numpy())             # 10. 항상 원본 그대로
# 10. training=True: 무작위로 끄고 남은 값을 1/(1-rate)배 (inverted dropout → 기대값 유지)
#     training=False(추론): 아무것도 끄지 않음 → 같은 입력이면 항상 같은 출력
#     model.fit은 자동으로 training=True, evaluate/predict는 자동으로 False
```

**Data Augmentation 11~14**

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), _ = mnist.load_data()
# 첫 이미지 1장만 (1, 28, 28, 1) 모양 + 0~1 정규화
sample_x = x_train[:1].reshape(-1, 28, 28, 1).astype('float32') / 255.0

settings = {
    'rotation 10': dict(rotation_range=10),         # 11. ±10도 회전
    'rotation 45': dict(rotation_range=45),         # 11. ±45도 (너무 크면 숫자 의미가 바뀔 수 있음)
    'zoom 0.3': dict(zoom_range=0.3),               # 12. 0.7~1.3배 확대/축소
    'shift 0.2': dict(width_shift_range=0.2),       # 12. 가로 ±20% 이동
}
fig, axes = plt.subplots(1, len(settings) + 1, figsize=(12, 3))
axes[0].imshow(sample_x[0, :, :, 0], cmap='gray')
axes[0].set_title('original')
for ax, (name, kw) in zip(axes[1:], settings.items()):
    # **kw: 딕셔너리를 키워드 인자로 풀어서 전달 (rotation_range=10 처럼)
    gen = ImageDataGenerator(**kw).flow(sample_x, batch_size=1, shuffle=False, seed=0)
    aug = next(gen)                                 # 증강된 배치 1개 꺼내기 → (1, 28, 28, 1)
    ax.imshow(aug[0, :, :, 0], cmap='gray')
    ax.set_title(name)
for ax in axes:
    ax.axis('off')
plt.show()
```

```python
# 13~14. 시계열 jitter의 sigma, random masking 길이 변경
t = np.linspace(0, 4 * np.pi, 120)
series = np.sin(t) + 0.2 * np.sin(4 * t)           # 원본 시계열 (120개)
rng = np.random.default_rng(0)

fig, axes = plt.subplots(1, 2, figsize=(12, 3))
for sigma in [0.05, 0.15, 0.5]:                    # 13. jitter = 원본 + 정규분포 잡음
    axes[0].plot(series + rng.normal(0, sigma, len(series)), label=f'sigma={sigma}')
axes[0].plot(series, 'k', lw=2, label='original')
axes[0].set_title('13. Jitter'); axes[0].legend()

for length in [10, 20, 50]:                        # 14. 45번부터 length개를 0으로 가림
    masked = series.copy()                         # copy 안 하면 원본까지 바뀜
    masked[45:45 + length] = 0.0
    axes[1].plot(masked, label=f'mask length={length}')
axes[1].set_title('14. Random masking'); axes[1].legend()
plt.show()
# sigma나 mask 길이가 너무 크면 원래 패턴이 망가져 "라벨을 보존하는 변환"이 아니게 됨
```

**L10 실습 함수 직접 구현: 시계열 증강 `time_shift`, `scale_series`, `jitter`, `random_mask`**

```python
# 시간 이동: 끝 값들이 앞으로 돌아오는 원형 이동 (np.roll)
def time_shift(x, shift=15):
    return np.roll(x, shift)

# 크기 변환: 전체에 같은 배율을 곱함
def scale_series(x, factor=1.5):
    return factor * x

# 잡음 추가: 원본 + N(0, sigma²). seed를 받아서 재현 가능
def jitter(x, sigma=0.15, seed=0):
    local_rng = np.random.default_rng(seed)
    return x + local_rng.normal(0, sigma, size=len(x))

# 구간 가리기: start부터 length개를 0으로 (원본 보호를 위해 copy)
def random_mask(x, start=45, length=20):
    out = x.copy()
    out[start:start + length] = 0.0
    return out

s = np.arange(10, dtype=float)
print('time_shift  :', time_shift(s, 3))
print('scale_series:', scale_series(s, 2.0))
print('jitter      :', np.round(jitter(s, sigma=0.1), 3))
print('random_mask :', random_mask(s, start=2, length=4))
print('원본 유지    :', s)
```

```text
time_shift  : [7. 8. 9. 0. 1. 2. 3. 4. 5. 6.]
scale_series: [ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18.]
jitter      : [0.013 0.987 2.064 3.01  3.946 5.036 6.13  7.095 7.93  8.873]
random_mask : [0. 1. 0. 0. 0. 0. 6. 7. 8. 9.]
원본 유지    : [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
```

**꼭 기억할 것**
1. Regularization은 일반화를 개선하기 위해 model의 **effective complexity**를 제어합니다.
2. **L1은 sparsity**(일부 계수 0), **L2는 weight shrinkage**와 연결됩니다.
3. L1/L2를 쓸 때는 **feature scaling**이 중요합니다.
4. Adam에서는 **L2 penalty와 decoupled weight decay(AdamW)**를 구분해야 합니다.
5. **Dropout은 training에서만** 활성화됩니다.
6. Data augmentation은 **label을 보존하는 transformation**이어야 합니다.
7. 정규화가 너무 약하면 high variance(과적합), 너무 강하면 high bias(과소적합)입니다.

[⬆ 목차로 돌아가기](#목차)

### 17-10. L11 회귀 실습 (Ames, ETTh1, 기구학) 체크리스트와 직접 해보기

**검색 키워드:** 회귀, regression, Ames, 집값, SalePrice, 결측치, handle_missing_values, preprocess_features, OneHotEncoder, inverse_target, build_mlp, make_callbacks, evaluate_model, run_experiment, run_log_target_experiment, log1p, expm1, ETTh1, 시계열, time series, 윈도우, create_all_windows, baseline, last value, seasonal naive, make_naive_predictions, residual, anchor, make_last_value_anchor, make_seasonal_anchor, evaluate_predictions, fit_for_validation, evaluate_saved_model, add_dense_block, reset_seed, reset_random_state, 기구학, kinematics, FK, IK, 가중 MSE, wrapped angle

**Ames 결과 해석 체크리스트 (교안 결과표 기준)**

| 모델 | Val MAE | Test MAE |
|---|---|---|
| Baseline MLP | 20,370 | 20,503 |
| BatchNorm | 31,624 | 31,743 |
| AdamW weight decay | 19,852 | 20,324 |
| Dropout 0.10 / 0.20 / 0.30 | 17,816 / 18,675 / 19,671 | 17,782 (0.10) |
| Dropout 0.10 + LR schedule | 17,816 | 17,782 |
| Log target + Dropout | (log 단위라 비교 안 함) | 26,047 |
| Top-100 features + Dropout | - | 19,872 |
| Small MLP (32→16) + Dropout | - | 19,751 |

| 질문 | 답 |
|---|---|
| 1. Baseline의 train/val MAE gap | 학습이 진행될수록 train MAE는 계속 내려가지만 val MAE는 정체 → **gap이 커짐** (과적합) |
| 2. BN이 도움이 됐나 | **아니오.** 가장 나쁨 (31,624). 작은 tabular 데이터 + 배치 32의 통계가 불안정해 오히려 방해 |
| 3. Weight decay만 적용 | Baseline보다 **약간 개선** (val 20,370 → 19,852) |
| 4. 가장 좋은 dropout rate | **0.10** (rate가 커질수록 val MAE 증가) |
| 5. LR scheduling 추가 효과 | **추가 개선 없음** (Dropout 0.10과 같은 결과) |
| 6. Log transform | 전체 달러 MAE는 **악화** (17,782 → 26,047). log 공간에서는 상대 오차를 줄이므로, `expm1`로 되돌리면 고가 주택의 작은 log 오차가 큰 달러 오차가 됨 |
| 7. Top-K가 full보다 나쁜 이유 | `f_regression`은 특성 하나씩의 **선형** 관계만 봄 → 비선형·조합 효과가 있는 특성(특히 원핫 열)을 버림 |
| 8. Small MLP | 파라미터는 줄었지만 성능이 나빠짐 (19,751 > 17,782) → 과적합 감소보다 **capacity 부족** 쪽 |

> 원칙: 여러 기법을 한꺼번에 쓰지 말고 **한 번에 하나씩** 바꿔서 원인을 확인합니다. 실험 코드는 [8-8](#8-8-모델을-함수로-만들기-여러-설정-비교할-때)의 `build_mlp` 반복 패턴을 쓰면 됩니다.

**L11 Ames 실습 함수 직접 구현 (노트북 함수 이름 그대로)**

노트북 L11_01의 실험 흐름을 함수로 다시 만든 것입니다: `reset_seed` → `handle_missing_values` → `preprocess_features` → y 표준화와 `inverse_target` → `build_mlp` → `make_callbacks` → `run_experiment`(내부에서 `evaluate_model`) → `plot_learning_and_prediction`, 그리고 `run_log_target_experiment`.
`handle_missing_values`, `preprocess_features`, `inverse_target`은 작은 가짜 데이터로 실행해 확인했습니다. Keras를 쓰는 함수는 문법 검사만 했습니다(여기에는 TensorFlow가 없음).

```python
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, AdamW

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 20

# ── 0) 재현성: python random, numpy, tensorflow 시드를 모두 고정하고 이전 모델 그래프를 지움
def reset_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.clear_session()

# ── 1) 결측치 처리 (Ames): '시설 없음' 의미 → 'None' / 0, 진짜 누락 → train 통계
ABSENCE_CAT_COLS = ['PoolQC', 'Alley', 'Fence', 'FireplaceQu', 'MiscFeature',
                    'MasVnrType', 'GarageType', 'GarageFinish', 'GarageQual',
                    'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinType2']          # 범주형 15개
ABSENCE_NUM_COLS = ['GarageYrBlt', 'MasVnrArea']              # 수치형 2개

def handle_missing_values(X_train, X_val, X_test):
    # copy: 원본 DataFrame을 건드리지 않음 (SettingWithCopyWarning도 방지)
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    for data in [X_train, X_val, X_test]:
        data[ABSENCE_CAT_COLS] = data[ABSENCE_CAT_COLS].fillna('None')
        data[ABSENCE_NUM_COLS] = data[ABSENCE_NUM_COLS].fillna(0)
    # 채움값은 반드시 train에서만 계산 (val/test 정보 누수 방지)
    lot_frontage_median = X_train['LotFrontage'].median()
    electrical_mode = X_train['Electrical'].mode()[0]       # mode()는 Series → [0]
    for data in [X_train, X_val, X_test]:
        data['LotFrontage'] = data['LotFrontage'].fillna(lot_frontage_median)
        data['Electrical'] = data['Electrical'].fillna(electrical_mode)
    return X_train, X_val, X_test

# ── 2) 특성 전처리: 범주형 → 원핫(train에서 fit), 수치형 → 표준화(train에서 fit) → 옆으로 붙이기
def preprocess_features(X_train, X_val, X_test):
    cat_cols = X_train.select_dtypes(include='object').columns   # 문자열 열
    num_cols = X_train.select_dtypes(include='number').columns   # 숫자 열
    # handle_unknown='ignore': val/test에만 있는 새 범주는 전부 0으로 (오류 방지)
    # sparse_output=False: 일반 numpy 배열로 (sklearn 1.2 미만이면 sparse=False)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler()

    X_train_cat = encoder.fit_transform(X_train[cat_cols])   # fit은 train만
    X_val_cat = encoder.transform(X_val[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_val_num = scaler.transform(X_val[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    return {
        'X_train': np.concatenate([X_train_num, X_train_cat], axis=1),   # axis=1: 열 방향으로 붙임
        'X_val': np.concatenate([X_val_num, X_val_cat], axis=1),
        'X_test': np.concatenate([X_test_num, X_test_cat], axis=1),
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'encoder': encoder,
        'scaler': scaler,
    }

# ── 3) 목표값 표준화와 되돌리기 (y_train, y_val, y_test는 Series라고 가정)
#      y_scaler = StandardScaler()
#      y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
#      y_val_scaled   = y_scaler.transform(y_val.to_numpy().reshape(-1, 1)).flatten()
#      y_test_array   = y_test.to_numpy()                    # test는 원래 단위(달러)로 평가
def inverse_target(y_scaled):
    # (N,) → (N, 1)로 바꿔 inverse_transform → 다시 (N,)
    return y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).flatten()

# ── 4) 설정만 바꿔 여러 MLP를 만드는 함수 (BN, Dropout, AdamW 선택)
def build_mlp(input_dim, hidden_units=(64, 32), dropout_rate=0.0, use_batch_norm=False,
              optimizer='adam', learning_rate=2e-4, weight_decay=0.0):
    layers = [Input(shape=(input_dim,))]
    for units in hidden_units:
        if use_batch_norm:
            # Dense(bias 없음) → BN → ReLU  (BN의 β가 bias 역할을 대신함)
            layers.append(Dense(units, use_bias=False, kernel_initializer='he_normal'))
            layers.append(BatchNormalization())
            layers.append(Activation('relu'))
        else:
            layers.append(Dense(units, activation='relu', kernel_initializer='he_normal'))
        if dropout_rate > 0:
            layers.append(Dropout(dropout_rate))
    layers.append(Dense(1))                    # 회귀 출력 1개, 활성화 없음
    model = Sequential(layers)
    if optimizer == 'adamw':
        opt = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        opt = Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    return model

# ── 5) 콜백 목록: (선택) ReduceLROnPlateau + 항상 EarlyStopping
def make_callbacks(use_lr_scheduler=False, patience=PATIENCE):
    callbacks = []
    if use_lr_scheduler:
        # val_mae가 15 epoch 동안 안 좋아지면 lr을 절반으로, 줄인 뒤 5 epoch는 대기(cooldown)
        callbacks.append(ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=15,
                                           cooldown=5, min_lr=1e-5, mode='min', verbose=1))
    callbacks.append(EarlyStopping(monitor='val_mae', patience=patience, mode='min',
                                   restore_best_weights=True, verbose=1))   # 가장 좋던 가중치로 복원
    return callbacks

# ── 6) test 평가: 예측(표준화 단위) → 원래 달러 단위로 되돌려 MAE, RMSE
def evaluate_model(model, X_test_input, y_test_array):
    y_pred_scaled = model.predict(X_test_input, verbose=0).flatten()   # (N, 1) → (N,)
    y_pred = inverse_target(y_pred_scaled)
    mse = mean_squared_error(y_test_array, y_pred)
    return {
        'test_mae': mean_absolute_error(y_test_array, y_pred),
        'test_rmse': np.sqrt(mse),
        'y_pred': y_pred,
    }

# ── 7) 실험 1회: 시드 리셋 → 모델 → 학습 → 기록(dict) 반환
#      **model_kwargs: build_mlp에 넘길 설정 (예: dropout_rate=0.1, optimizer='adamw', weight_decay=0.1)
def run_experiment(name, X_train_input, X_val_input, X_test_input,
                   y_train_scaled, y_val_scaled, y_test_array, seed=SEED, **model_kwargs):
    reset_seed(seed)
    use_lr_scheduler = model_kwargs.pop('use_lr_scheduler', False)   # build_mlp 인자가 아니라서 빼냄
    model = build_mlp(input_dim=X_train_input.shape[1], **model_kwargs)
    history = model.fit(
        X_train_input, y_train_scaled,
        validation_data=(X_val_input, y_val_scaled),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=make_callbacks(use_lr_scheduler),
        verbose=0,
    )
    best_epoch = int(np.argmin(history.history['val_mae']) + 1)       # 1부터 세기
    # 표준화 단위 MAE × y 표준편차 = 달러 단위 MAE (평균 이동은 MAE에 영향 없음)
    best_val_mae = min(history.history['val_mae']) * y_scaler.scale_[0]
    evaluation = evaluate_model(model, X_test_input, y_test_array)
    record = {
        'model': name,
        'best_epoch': best_epoch,
        'best_val_mae': best_val_mae,
        'test_mae': evaluation['test_mae'],
        'test_rmse': evaluation['test_rmse'],
        'parameters': model.count_params(),
    }
    return model, history, evaluation['y_pred'], record

# ── 8) 학습곡선(달러 MAE) + 예측 vs 실제 산점도
def plot_learning_and_prediction(history, y_pred, y_test_array, title):
    best_epoch = int(np.argmin(history.history['val_mae']) + 1)
    train_mae = np.array(history.history['mae']) * y_scaler.scale_[0]
    val_mae = np.array(history.history['val_mae']) * y_scaler.scale_[0]
    test_mae = mean_absolute_error(y_test_array, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(train_mae, label='Training MAE')
    axes[0].plot(val_mae, label='Validation MAE')
    axes[0].axvline(best_epoch - 1, linestyle='--', color='gray', label=f'Best epoch {best_epoch}')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MAE ($)')
    axes[0].set_title(f'{title}: Learning Curve'); axes[0].legend(); axes[0].grid(True)

    lo = min(y_test_array.min(), y_pred.min())
    hi = max(y_test_array.max(), y_pred.max())
    axes[1].scatter(y_test_array, y_pred, alpha=0.6)
    axes[1].plot([lo, hi], [lo, hi], '--', color='red')      # y = x 선 (완벽한 예측)
    axes[1].set_xlabel('Actual Sale Price'); axes[1].set_ylabel('Predicted Sale Price')
    axes[1].set_title(f'{title}\nTest MAE: ${test_mae:,.0f}'); axes[1].grid(True)
    plt.tight_layout()
    plt.show()

# ── 9) log 목표값 실험: y → log1p(y)로 학습, 예측은 expm1으로 되돌려 달러 MAE
def run_log_target_experiment(X_train_input, X_val_input, X_test_input,
                              y_train, y_val, y_test_array, dropout_rate=0.10, seed=SEED):
    reset_seed(seed)
    y_train_log = np.log1p(np.asarray(y_train))      # log(1 + y): y=0이어도 안전
    y_val_log = np.log1p(np.asarray(y_val))
    model = build_mlp(input_dim=X_train_input.shape[1], hidden_units=(64, 32),
                      dropout_rate=dropout_rate)
    history = model.fit(
        X_train_input, y_train_log,
        validation_data=(X_val_input, y_val_log),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=make_callbacks(False),
        verbose=0,
    )
    y_pred = np.expm1(model.predict(X_test_input, verbose=0).flatten())   # exp(ŷ) − 1
    record = {
        'model': 'Log target + Dropout',
        'best_epoch': int(np.argmin(history.history['val_mae']) + 1),
        'best_val_mae': np.nan,          # val MAE가 log 단위라 달러와 비교하지 않음
        'test_mae': mean_absolute_error(y_test_array, y_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test_array, y_pred)),
        'parameters': model.count_params(),
    }
    return model, history, y_pred, record

# ── 사용 순서 (Ames)
# df = pd.read_csv(BASE + 'ames_train.csv')
# X = df.drop(columns=['Id', 'SalePrice']); y = df['SalePrice']
# X['MSSubClass'] = X['MSSubClass'].astype(str)
# X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, random_state=42)
# X_train, X_val, X_test = handle_missing_values(X_train, X_val, X_test)
# feat = preprocess_features(X_train, X_val, X_test)
# y_scaler = StandardScaler()
# y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
# y_val_scaled = y_scaler.transform(y_val.to_numpy().reshape(-1, 1)).flatten()
# y_test_array = y_test.to_numpy()
# results = []
# for name, kw in [('Baseline', {}),
#                  ('Dropout 0.1', dict(dropout_rate=0.1)),
#                  ('AdamW', dict(optimizer='adamw', weight_decay=1e-1)),
#                  ('Dropout 0.1 + LR', dict(dropout_rate=0.1, use_lr_scheduler=True))]:
#     model, history, y_pred, record = run_experiment(
#         name, feat['X_train'], feat['X_val'], feat['X_test'],
#         y_train_scaled, y_val_scaled, y_test_array, **kw)
#     results.append(record)
# print(pd.DataFrame(results).sort_values('test_mae'))
# plot_learning_and_prediction(history, y_pred, y_test_array, name)
```

> pandas 3.x에서는 문자열 열이 `object`가 아니라 `str` 자료형일 수 있습니다. 경고가 뜨거나 범주형 열이 안 잡히면 `select_dtypes(include=['object', 'string'])`로 바꾸세요. Colab(pandas 2.x)에서는 위 코드 그대로 동작합니다.

```python
# 위 handle_missing_values / preprocess_features / inverse_target 동작 확인 (가짜 데이터, NumPy+pandas만)
rng = np.random.default_rng(0)
n = 12

def pick(vals):                      # vals 중 무작위로 n개 고르기 (None/np.nan = 결측)
    return pd.Series([vals[i] for i in rng.integers(0, len(vals), n)], dtype=object)

df = pd.DataFrame({c: pick(['Gd', 'TA', None]) for c in ABSENCE_CAT_COLS})
df['GarageYrBlt'] = pick([1990.0, 2005.0, np.nan]).astype(float)
df['MasVnrArea'] = pick([0.0, 120.0, np.nan]).astype(float)
df['LotFrontage'] = pick([60.0, 80.0, np.nan]).astype(float)
df['Electrical'] = pick(['SBrkr', 'FuseA', None])
df['GrLivArea'] = rng.integers(800, 2500, n).astype(float)
print('결측 전:', int(df.isna().sum().sum()))

X_train, X_val, X_test = df.iloc[:8], df.iloc[8:10], df.iloc[10:]
tr, va, te = handle_missing_values(X_train, X_val, X_test)
print('남은 결측:', tr.isna().sum().sum(), va.isna().sum().sum(), te.isna().sum().sum())
print('LotFrontage 채움값 (train 중앙값):', X_train['LotFrontage'].median())

feat = preprocess_features(tr, va, te)
print('num', len(feat['num_cols']), 'cat', len(feat['cat_cols']),
      'X_train', feat['X_train'].shape, 'X_val', feat['X_val'].shape)

y_train = pd.Series([200000., 150000, 300000, 180000, 220000, 160000, 250000, 175000])
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
print('scaled', np.round(y_train_scaled[:3], 4), '→ 복원', inverse_target(y_train_scaled[:3]))
```

```text
결측 전: 89
남은 결측: 0 0 0
LotFrontage 채움값 (train 중앙값): 70.0
num 4 cat 16 X_train (8, 49) X_val (2, 49)
scaled [-0.0926 -1.1504  2.0232] → 복원 [200000. 150000. 300000.]
```

**ETTh1 Direct 결과 해석 체크리스트**

교안 결과(°C² MSE): Last Value **3.97**, Basic MLP **4.14**, Last-Value Residual MLP **3.37**.
제가 seed 42로 한 번 돌린 결과(TF 2.21): Last Value 3.97, Seasonal Naive 5.17, Basic MLP(shuffle=False) 3.71, Basic MLP(shuffle=True) 3.58

| 질문 | 답 |
|---|---|
| 1. 더 강한 baseline | **Last Value** (Seasonal Naive보다 오차가 작음) |
| 2. Basic MLP가 두 baseline을 모두 이기나 | Seasonal은 이김. Last Value는 교안 결과에서는 **못 이김**(4.14 > 3.97), 제 1회 실행에서는 근소하게 이김 → seed에 따라 바뀌는 **근소한 차이** |
| 3. 예측이 지나치게 smooth한가 | 그렇다. MSE는 평균적인 예측을 선호해서 급변을 완만하게 예측함 |
| 4. shuffle=False vs True | train 구간 윈도우끼리만 섞으므로 **미래 정보 누수는 없음**. 섞으면 mini-batch가 다양해져 gradient가 안정적. 제 실행에서는 shuffle=True가 더 좋았음 (3.58 vs 3.71) |
| 5. last-value가 강하다는 의미 | OT가 **천천히 변하고 자기상관이 강한** 시계열 (바로 전 값이 좋은 예측) |
| 6. MLP가 급변을 예측하기 어려운 이유 | 입력이 과거 OT뿐이라 급변의 원인 정보가 없고, 급변은 드물어 MSE 학습에서 평균 쪽으로 끌려감 |

**ETTh1 Residual 결과 해석 체크리스트**

| 질문 | 답 |
|---|---|
| 1. Direct MLP가 Last Value를 이기나 | 교안 기준 **아니오** (4.14 vs 3.97) |
| 2. Last-Value Residual MLP | **예.** Direct보다 좋고 Last Value baseline도 이김 (3.37) |
| 3. Seasonal Residual이 유리한 경우 | 하루(24시간) 주기 패턴이 강하고 반복될 때 |
| 4. Residual target 평균 절댓값 | **더 작음** (제 계산, 표준화 단위: direct 0.78 vs residual 0.26) → 학습할 범위가 작아짐 |
| 5. Zero-init residual head 장점 | 학습 시작 시 출력이 0 → 예측 = anchor(Last Value)에서 출발. 강한 baseline에서 **보정량만 학습**해 안정적 |
| 6. 첫 step부터 크게 어긋나는 현상 | residual은 마지막 관측값에서 출발해 보정량만 더하므로 첫 미래 step 오차가 작게 유지됨 (제 실행: 첫 step MAE 0.43°C). Direct MLP는 첫 값부터 새로 만들어야 해서 어긋날 수 있음 |
| 7. MAE가 가장 큰 horizon | 먼 미래일수록 커짐. 제 실행에서 Residual MLP의 horizon별 MAE는 0.43°C(1시간 뒤) → 1.79°C(24시간 뒤)로 **마지막 step이 최대** |

**L11 ETTh1 실습 함수 직접 구현 (노트북 함수 이름 그대로)**

L11_02(Direct)와 L11_03(Residual)의 함수들입니다. 윈도우 만들기, baseline, anchor, 평가 함수는 NumPy와 sklearn만 써서 가짜 시계열로 실행해 확인했습니다. 아래쪽 Keras 함수(`reset_random_state`, `add_dense_block`, `build_mlp`, `fit_for_validation`, `evaluate_saved_model`)는 문법 검사만 했습니다.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

INPUT_LEN = 96       # 과거 96시간(4일)을 보고
PRED_LEN = 24        # 미래 24시간을 한 번에 예측 (Direct multi-output)

# ── 1) 모든 윈도우 만들기
#      X[i] = features[i : i+96]           → 모양 (윈도우 수, 96, 특성 수)
#      y[i] = target[i+96 : i+120, 0]      → 모양 (윈도우 수, 24)
#      target_starts / target_ends: 각 윈도우의 '정답 구간' 위치 → 분할 마스크에 사용
def create_all_windows(features, target, input_len, pred_len):
    X, y, target_starts, target_ends = [], [], [], []
    max_start = len(features) - input_len - pred_len + 1     # 만들 수 있는 윈도우 개수
    for input_start in range(max_start):
        target_start = input_start + input_len
        target_end = target_start + pred_len
        X.append(features[input_start:target_start])
        y.append(target[target_start:target_end, 0])
        target_starts.append(target_start)
        target_ends.append(target_end)
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            np.asarray(target_starts),
            np.asarray(target_ends))

# ── 가짜 시계열 (실제 시험에서는 df['OT'] 사용): 24시간 주기 + 잡음
rng = np.random.default_rng(0)
n = 1000
t = np.arange(n)
ot = (20 + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)).astype(np.float32)
X_raw = ot.reshape(-1, 1)          # (n, 1): 입력 특성 = OT 하나 (univariate)
y_raw = ot.reshape(-1, 1)          # (n, 1): 목표 = OT

# 시간 순서대로 70 / 15 / 15 (섞으면 미래 정보 누수)
train_end, val_end = int(n * 0.70), int(n * 0.85)
x_scaler = StandardScaler().fit(X_raw[:train_end])      # 스케일러는 train 구간만
y_scaler = StandardScaler().fit(y_raw[:train_end])
X_scaled = x_scaler.transform(X_raw).astype(np.float32)
y_scaled = y_scaler.transform(y_raw).astype(np.float32)

X_all, y_all, ts, te = create_all_windows(X_scaled, y_scaled, INPUT_LEN, PRED_LEN)
# 정답 구간이 각 split 안에 완전히 들어가는 윈도우만 사용 (경계에 걸친 윈도우는 버림)
train_mask = te <= train_end
val_mask = (ts >= train_end) & (te <= val_end)
test_mask = (ts >= val_end) & (te <= n)
X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]
X_test, y_test = X_all[test_mask], y_all[test_mask]
print('X_all', X_all.shape, 'y_all', y_all.shape)
print('train', X_train.shape, 'val', X_val.shape, 'test', X_test.shape)

# ── 2) 표준화 단위 → °C (모양 유지: (N, 24) → (N·24, 1) → 되돌린 뒤 (N, 24))
def inverse_target(values_scaled):
    shape = values_scaled.shape
    return y_scaler.inverse_transform(values_scaled.reshape(-1, 1)).reshape(shape)

# ── 3) 평가: 표준화 단위와 원래 단위(°C) 모두 MSE, MAE
def evaluate_predictions(y_true_scaled, y_pred_scaled):
    yt, yp = inverse_target(y_true_scaled), inverse_target(y_pred_scaled)
    return {
        'MSE (Normalized)': mean_squared_error(y_true_scaled.ravel(), y_pred_scaled.ravel()),
        'MAE (Normalized)': mean_absolute_error(y_true_scaled.ravel(), y_pred_scaled.ravel()),
        'MSE (°C²)': mean_squared_error(yt.ravel(), yp.ravel()),
        'MAE (°C)': mean_absolute_error(yt.ravel(), yp.ravel()),
    }

# ── 4) 학습 없는 baseline 두 개
#      Last Value: 마지막 관측값을 24번 반복
#      Seasonal Naive: 24시간 전 같은 시각 값 = 입력 윈도우의 마지막 24개
def make_naive_predictions(X):
    last_value = X[:, -1, 0]                                           # (N,)
    last_value_pred = np.repeat(last_value[:, None], PRED_LEN, axis=1)  # (N, 24)
    seasonal_pred = X[:, -PRED_LEN:, 0]                                 # (N, 24)
    return {'Last Value': last_value_pred, 'Seasonal Naive (24h)': seasonal_pred}

# ── 5) Residual 학습용 anchor (= baseline 예측). 목표를 y − anchor로 바꿔 '보정량'만 학습
def make_last_value_anchor(X):
    return np.repeat(X[:, -1, 0:1], PRED_LEN, axis=1)    # 0:1로 (N, 1) 모양 유지 → (N, 24)

def make_seasonal_anchor(X):
    return X[:, -PRED_LEN:, 0].copy()

for name, pred in make_naive_predictions(X_val).items():
    m = evaluate_predictions(y_val, pred)
    print(f'{name:22s}', {k: round(float(v), 4) for k, v in m.items()})

for name, fn in [('last', make_last_value_anchor), ('seasonal', make_seasonal_anchor)]:
    anchor = fn(X_train)
    resid = y_train - anchor                          # Residual 목표
    print(f'{name:8s} anchor {anchor.shape}  |direct| 평균 {np.abs(y_train).mean():.4f}'
          f'  |residual| 평균 {np.abs(resid).mean():.4f}'
          f'  복원 OK: {np.allclose(resid + anchor, y_train, atol=1e-6)}')
# 예측할 때: y_pred = model.predict(X) + anchor  (anchor를 다시 더해야 원래 값)
```

```text
X_all (881, 96, 1) y_all (881, 24)
train (581, 96, 1) val (127, 96, 1) test (127, 96, 1)
Last Value             {'MSE (Normalized)': 2.0492, 'MAE (Normalized)': 1.1631, 'MSE (°C²)': 25.7238, 'MAE (°C)': 4.121}
Seasonal Naive (24h)   {'MSE (Normalized)': 0.0353, 'MAE (Normalized)': 0.1512, 'MSE (°C²)': 0.4433, 'MAE (°C)': 0.5356}
last     anchor (581, 24)  |direct| 평균 0.9018  |residual| 평균 1.1538  복원 OK: True
seasonal anchor (581, 24)  |direct| 평균 0.9018  |residual| 평균 0.1575  복원 OK: True
```

> 가짜 데이터는 24시간 주기가 아주 강해서 Seasonal 쪽이 훨씬 좋게 나왔습니다. 실제 ETTh1 OT는 천천히 변해서 **Last Value가 더 강한 baseline**입니다(위 체크리스트). 어떤 anchor가 좋은지는 데이터마다 baseline 오차를 먼저 비교해서 정합니다.

```python
# ── Keras를 쓰는 ETTh1 함수들 (문법 검사만 함)
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping

BATCH_SIZE = 64
MAX_EPOCHS = 150
LEARNING_RATE = 1e-3

# 시드 고정 + (가능하면) GPU 연산까지 결정적으로
def reset_random_state(seed):
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass                       # 지원하지 않는 환경이면 그냥 넘어감

# layers 리스트에 Dense(+BN)+ReLU(+Dropout) 블록을 붙임 (리스트를 직접 수정, 반환값 없음)
def add_dense_block(layers, units, use_batch_norm=False, dropout_rate=0.0):
    layers.append(Dense(units, kernel_initializer='he_normal',
                        use_bias=not use_batch_norm))    # BN을 쓰면 bias 끔
    if use_batch_norm:
        layers.append(BatchNormalization())
    layers.append(Activation('relu'))
    if dropout_rate > 0:
        layers.append(Dropout(dropout_rate))

# config 딕셔너리로 설정: {'name': ..., 'use_batch_norm': ..., 'dropout_rate': ..., 'optimizer': 'adam'/'adamw',
#                          'weight_decay': ..., 'shuffle': ..., 'zero_init_head': ...}
def build_mlp(config):
    layers = [Input(shape=(INPUT_LEN, 1)), Flatten()]     # (96, 1) → 96개로 펼침
    add_dense_block(layers, 128, config.get('use_batch_norm', False), config.get('dropout_rate', 0.0))
    add_dense_block(layers, 64, config.get('use_batch_norm', False), config.get('dropout_rate', 0.0))
    if config.get('zero_init_head', False):
        # Residual용: 출력층 가중치를 0으로 시작 → 처음 예측 = anchor 그대로
        layers.append(Dense(PRED_LEN, kernel_initializer='zeros', bias_initializer='zeros'))
    else:
        layers.append(Dense(PRED_LEN))                      # 24시간 출력
    model = Sequential(layers)
    if config.get('optimizer', 'adam').lower() == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE,
                                              weight_decay=config.get('weight_decay', 1e-4))
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# 학습 + 기록 + (선택) 저장.  y_tr, y_va에 Residual 목표(y − anchor)를 넣으면 Residual 학습
def fit_for_validation(config, seed, X_tr, y_tr, X_va, y_va, save_model=True, model_dir='models'):
    reset_random_state(seed)
    model = build_mlp(config)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=config.get('shuffle', True),      # 윈도우는 이미 만들어져서 섞어도 누수 없음
        callbacks=[EarlyStopping(monitor='val_loss', patience=15,
                                 restore_best_weights=True, mode='min', verbose=1)],
        verbose=0,
    )
    best_index = int(np.argmin(history.history['val_loss']))
    record = {
        'Experiment': config['name'],
        'Seed': seed,
        'Parameters': model.count_params(),
        'Best Epoch': best_index + 1,
        'Best Val Loss': float(history.history['val_loss'][best_index]),
        'Best Val MAE (°C)': float(history.history['val_mae'][best_index] * y_scaler.scale_[0]),
    }
    model_path = None
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        safe_name = ''.join(ch if ch.isalnum() else '_' for ch in config['name'])   # 파일명에 못 쓰는 문자 → _
        model_path = os.path.join(model_dir, f'{safe_name}_seed{seed}.keras')
        model.save(model_path)
    return model, history, record, model_path

# 저장한 모델 불러와 test 예측 + 평가 (Residual이면 anchor를 더해서 평가)
def evaluate_saved_model(model_path, X_te, y_te, anchor=None):
    model = tf.keras.models.load_model(model_path)
    pred = model.predict(X_te, verbose=0)
    if anchor is not None:
        pred = pred + anchor
    return model, pred, evaluate_predictions(y_te, pred)

# ── 사용 예
# import os
# cfg = {'name': 'Basic MLP', 'shuffle': True}
# model, history, record, path = fit_for_validation(cfg, 42, X_train, y_train, X_val, y_val)
# _, pred, metrics = evaluate_saved_model(path, X_test, y_test)
#
# Residual (Last Value anchor):
# a_tr, a_va, a_te = (make_last_value_anchor(X) for X in (X_train, X_val, X_test))
# cfg = {'name': 'Last-Value Residual MLP', 'zero_init_head': True}
# model, history, record, path = fit_for_validation(cfg, 42, X_train, y_train - a_tr, X_val, y_val - a_va)
# _, pred, metrics = evaluate_saved_model(path, X_test, y_test, anchor=a_te)
```

**Forward / Inverse Kinematics 직접 해보기**

```python
import time

BASE = 'https://raw.githubusercontent.com/dearyonghoon/course-ai-programming-2026/main/data/'
df = pd.read_csv(BASE + 'datasetIRB2400.csv')
# df = df.sample(60000, random_state=42)     # 시간이 부족하면 일부만

POSE = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']                 # 끝단 위치(mm) 3개 + 자세(rad) 3개
Q_IN = ['q1_in', 'q2_in', 'q3_in', 'q4_in', 'q5_in', 'q6_in']   # 관절 각도 입력
Q_OUT = ['q1_out', 'q2_out', 'q3_out', 'q4_out', 'q5_out', 'q6_out']

# in_cols → out_cols 데이터를 64:16:20으로 나누고 X, y 모두 train 기준 표준화
def prepare(in_cols, out_cols):
    X, y = df[in_cols].values, df[out_cols].values
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.2, random_state=42)
    sx, sy = StandardScaler().fit(X_tr), StandardScaler().fit(y_tr)       # train에서만 fit
    return (sx.transform(X_tr), sx.transform(X_va), sx.transform(X_te),
            sy.transform(y_tr), sy.transform(y_va),
            y_te,          # test 정답은 원래 단위 그대로 (최종 평가용)
            sy)            # 예측을 되돌릴 때 쓰는 y 스케일러

# 모델 만들고 학습 → 원래 단위 MAE 출력.  FK: run(Q_IN, POSE) / IK: run(POSE, Q_OUT)
def run(in_cols, out_cols, hidden=(128, 128, 64), batch_size=512, loss='mse'):
    X_tr, X_va, X_te, y_tr, y_va, y_te, sy = prepare(in_cols, out_cols)
    tf.keras.utils.set_random_seed(42)
    model = Sequential(
        [Input(shape=(6,))]
        + [Dense(u, activation='relu') for u in hidden]     # 은닉층 개수/크기는 hidden으로
        + [Dense(6)]                                        # 출력 6개, 활성화 없음 (회귀)
    )
    model.compile(optimizer='adam', loss=loss, metrics=['mae'])
    t0 = time.time()
    h = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=50,
        batch_size=batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)],
        verbose=0,
    )
    y_pred = sy.inverse_transform(model.predict(X_te, batch_size=512, verbose=0))   # 원래 단위로
    mae = np.mean(np.abs(y_pred - y_te), axis=0)                      # 원래 단위 출력별 MAE (6개)
    print(f'hidden={hidden}, batch={batch_size}: {time.time() - t0:.1f}s, '
          f'stopped at epoch {len(h.history["loss"])} '
          f'(best {int(np.argmin(h.history["val_loss"])) + 1}), '
          f'best val_loss={min(h.history["val_loss"]):.4f}')
    print('   MAE (원래 단위):', dict(zip(out_cols, np.round(mae, 3))))
    print('   MAE / 표준편차 :', dict(zip(out_cols, np.round(mae / sy.scale_, 3))))   # 단위 없는 비교
    return y_pred, y_te

# FK 1. batch 256 vs 512 시간 비교 / 4. 종료 epoch 확인
run(Q_IN, POSE, batch_size=256)
run(Q_IN, POSE, batch_size=512)
# FK 2. hidden 128-128-64 → 256-128-64
run(Q_IN, POSE, hidden=(256, 128, 64))
# IK 1~3. 같은 함수로 역기구학 (입력 pose → 출력 joint), hidden을 키워 비교
y_pred_ik, y_te_ik = run(POSE, Q_OUT)
run(POSE, Q_OUT, hidden=(256, 256, 128))
```

```python
# FK 3, 6. 위치(mm)와 자세(rad) 오차 비교: 단위가 달라 raw MAE를 평균하면 mm 값이 지배함
#          → 출력별로 따로 보거나 'MAE / 표준편차'처럼 단위 없는 값으로 비교 (위 run 출력의 두 번째 줄)

# FK 5. position과 orientation에 다른 loss weight 주기 (표준화된 y 기준 가중 MSE)
#       loss = mean( w_k · (y_k − ŷ_k)² )   w = [위치 3개 가중치, 자세 3개 가중치]
def make_weighted_mse(weights):
    w = tf.constant(weights, dtype=tf.float32)          # (6,) → 마지막 축에 브로드캐스팅
    def weighted_mse(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)          # float64 정답이 들어와도 안전
        return tf.reduce_mean(tf.square(y_true - y_pred) * w)
    return weighted_mse

run(Q_IN, POSE, loss=make_weighted_mse([2.0, 2.0, 2.0, 1.0, 1.0, 1.0]))   # 위치를 2배 중시

# IK 6. 일반 MAE vs wrapped angular MAE
#   각도 차이를 (−π, π] 범위로 감기: atan2(sin d, cos d)
angle_diff = np.arctan2(np.sin(y_pred_ik - y_te_ik), np.cos(y_pred_ik - y_te_ik))
print('plain MAE  :', np.round(np.mean(np.abs(y_pred_ik - y_te_ik), axis=0), 4))
print('wrapped MAE:', np.round(np.mean(np.abs(angle_diff), axis=0), 4))
# 각도는 θ와 θ ± 2π가 같은 자세 → 예측 3.1, 정답 −3.1이면 plain 오차 6.2지만 실제 차이는 약 0.08 rad
```

제가 전체 데이터(300,000행)로 실행한 결과 (seed 42, CPU, 최대 50 epoch):

| 실험 | 시간 | best val_loss | 비고 |
|---|---|---|---|
| FK 128-128-64, batch 256 | 124.6s | 0.1062 | 업데이트가 2배라 같은 epoch에서 더 낮은 loss |
| FK 128-128-64, batch 512 | 93.1s | 0.1256 | epoch당 더 빠름 |
| FK 256-128-64, batch 512 | 105.8s | 0.1081 | 첫 층을 키우니 개선 |
| FK 가중 MSE (위치 ×2) | 91.9s | - | 위치 MAE 개선(x 58.9 → 52.0mm), 자세 MAE는 약간 악화(yaw 0.78 → 0.82) |
| IK 128-128-64 | 91.3s | 0.4572 | FK보다 훨씬 큼 |
| IK 256-256-128 | 166.7s | 0.2552 | 크게 개선 |

| 질문 | 답 |
|---|---|
| FK 1. batch 256 vs 512 | 512가 epoch당 빠름. 256은 업데이트 횟수가 많아 같은 epoch에서 loss가 더 낮음 |
| FK 2. 256-128-64 | 용량이 늘어 val loss 개선 (0.1256 → 0.1081) |
| FK 3. 위치와 자세 중 오차가 큰 쪽 | 원래 단위로는 비교 불가(mm vs rad). `MAE / 표준편차`로 보면 위치 약 0.09~0.11, 자세(yaw 0.43, roll 0.36, pitch 0.23)로 **자세 쪽 상대 오차가 더 큼** |
| FK 4. EarlyStopping 종료 epoch | 이 실행에서는 50 epoch 동안 val loss가 계속 좋아져 **멈추지 않고 50 epoch를 다 씀** (`len(history.history['loss'])`로 확인) |
| FK 5. 다른 loss weight | 표준화된 y에 출력별 가중치를 곱한 가중 MSE (위 코드). 가중치를 준 쪽 오차가 줄고 나머지는 늘어나는 trade-off |
| FK 6. raw MAE 평균이 해석하기 어려운 이유 | mm와 rad는 단위·크기가 달라 mm 값(수십)이 평균을 지배함 |
| IK 1. FK vs IK | IK가 훨씬 어려움 (best val_loss 0.457 vs 0.126) |
| IK 2. MAE가 가장 큰 joint | **q6** (2.98 rad), 다음은 q4. q6는 범위가 ±6.98 rad(±2π 넘음)라 같은 자세에 여러 각도가 대응 |
| IK 3. hidden을 키우면 | val loss 개선 (0.457 → 0.255) |
| IK 5. 여러 joint 해가 가능한 경우 | 같은 pose에 해가 여러 개면 MSE는 그 **평균**을 학습 → 평균값은 유효한 해가 아닐 수 있고 오차가 커짐 (mapping ambiguity) |
| IK 6. 일반 MAE vs wrapped MAE | 2π 차이 나는 각도를 같은 자세로 보면 오차가 줄어듦: q6 2.98 → 1.62 rad, q4 1.47 → 1.42 rad |
| 체크포인트 | FK: joint angles → pose, IK: pose → joint angles. **scaler는 train에서만 fit**, test는 최종 평가에만 |

[⬆ 목차로 돌아가기](#목차)

### 17-11. L14 CNN (Fashion-MNIST 직접 해보기, TensorBoard)

**검색 키워드:** CNN, 합성곱, Conv2D, MaxPooling2D, Fashion-MNIST, MNIST, Deeper CNN, BatchNormalization, 혼동행렬, confusion matrix, 자주 혼동되는 클래스, 오분류, TensorBoard, 학습 곡선, histogram_freq

**직접 해보기 1~5: Fashion-MNIST에 같은 전처리 + Deeper CNN, MNIST와 비교, 자주 혼동되는 클래스**

```python
from tensorflow.keras.datasets import mnist, fashion_mnist

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   # Fashion-MNIST 라벨 0~9

# 2. MNIST와 동일한 전처리
#    float32 변환 → (N, 28, 28) → (N, 28, 28, 1) 채널 축 추가 → ÷255로 0~1
#    라벨은 원핫 (N,) → (N, 10)  (categorical_crossentropy용), 원래 정수 y_test도 함께 반환
def load_and_preprocess(loader):
    (X_train, y_train), (X_test, y_test) = loader.load_data()
    X_train = X_train.astype('float32').reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1) / 255.0
    return X_train, to_categorical(y_train, 10), X_test, to_categorical(y_test, 10), y_test

# 3. 구조는 바꾸지 않음 (14강 CNN2)
#    Conv(bias 없음) → BN → ReLU 블록, 'same' 패딩이라 Pooling 전까지 28×28 유지
#    28 → (pool) 14 → (pool) 7 → Flatten: 7·7·128 = 6272
def build_deeper_cnn():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), padding='same', use_bias=False), BatchNormalization(), ReLU(),
        Conv2D(64, (3, 3), padding='same', use_bias=False), BatchNormalization(), ReLU(),
        MaxPooling2D((2, 2)),                                       # 28 → 14
        Conv2D(128, (3, 3), padding='same', use_bias=False), BatchNormalization(), ReLU(),
        MaxPooling2D((2, 2)),                                       # 14 → 7
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

results = {}
for name, loader in [('MNIST', mnist), ('Fashion-MNIST', fashion_mnist)]:     # 1. Fashion-MNIST 불러오기
    tf.keras.utils.set_random_seed(42)
    X_train, y_train_oh, X_test, y_test_oh, y_test = load_and_preprocess(loader)
    model = build_deeper_cnn()
    model.fit(
        X_train, y_train_oh,
        validation_split=0.25,             # train의 뒤 25%를 검증으로 자동 분리
        epochs=10, batch_size=128, verbose=0,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
        ],
    )
    results[name] = model.evaluate(X_test, y_test_oh, verbose=0)[1]   # [loss, accuracy]의 [1]
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)     # 확률 10개 → 가장 큰 클래스 번호
print('4. test accuracy:', results)              # Fashion-MNIST가 더 낮음 (옷 종류끼리 비슷해서)

# 5. Fashion-MNIST에서 자주 혼동되는 클래스 (마지막 루프가 Fashion-MNIST라서 y_test, y_pred가 그쪽 값)
cm = confusion_matrix(y_test, y_pred)            # 행 = 실제, 열 = 예측 (10×10)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap='Blues', xticks_rotation=45)
plt.show()

off = cm.copy()
np.fill_diagonal(off, 0)                         # 대각선(정답) 제외 → 틀린 칸만 남김
for idx in np.argsort(off.ravel())[::-1][:5]:    # 1차원으로 펴서 큰 값 순서 상위 5개
    t, p = divmod(idx, 10)                       # 1차원 위치 → (행=실제, 열=예측)
    print(f'true {class_names[t]:12s} → pred {class_names[p]:12s}: {off[t, p]}개')
# 보통 Shirt ↔ T-shirt/top, Pullover ↔ Coat, Shirt ↔ Coat 같은 비슷한 옷끼리 많이 혼동됨
```

**TensorBoard 확인할 내용 모범답안**

| 항목 | 답 |
|---|---|
| 1. train/val loss가 함께 감소하나 | Scalars 탭의 `epoch_loss`에서 train과 validation 곡선을 같이 봄 |
| 2. 과적합 시작 시점 | validation loss가 최저점을 찍고 다시 오르기 시작하는 epoch (EarlyStopping의 best epoch와 같은 지점) |
| 3. Histogram 변화 | Histograms 탭에서 층별 가중치 분포가 epoch마다 퍼지거나 이동하는 모습 확인 (`histogram_freq=1` 필요) |
| 4. 여러 run 비교 | `log_dir`에 시간을 붙이면 실행할 때마다 새 폴더(run)가 생겨 TensorBoard에서 겹쳐 비교 가능 |

```python
import datetime
from tensorflow.keras.callbacks import TensorBoard

# 실행할 때마다 다른 폴더 이름 (예: logs/fit/20261021-031500) → run끼리 겹쳐 비교 가능
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_cb = TensorBoard(log_dir=log_dir,
                             histogram_freq=1)     # 1 epoch마다 가중치 히스토그램 기록

# model.fit(..., callbacks=[tensorboard_cb, early_stopping, checkpoint])
# 학습 후 새 셀에서:
# %load_ext tensorboard
# %tensorboard --logdir logs/fit
```

**정리 (L14)**
- FC network는 이미지를 **1차원 vector**로 펼쳐서 처리합니다.
- CNN은 이미지의 **공간 구조를 유지**하며 활용합니다.
- 더 깊은 convolution layer는 더 풍부한 **계층적(hierarchical) feature**를 학습합니다.
- **Learning curve와 error analysis**(오분류 샘플, 혼동행렬)를 함께 보면 모델을 더 잘 이해할 수 있습니다.
- MNIST pipeline은 Fashion-MNIST에도 **그대로 재사용**할 수 있습니다.

[⬆ 목차로 돌아가기](#목차)

