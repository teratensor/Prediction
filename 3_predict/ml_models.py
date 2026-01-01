"""
3_predict/ml_models.py - LightGBM 기반 ord2~ord5 예측 모델

각 위치별로 별도 모델 학습
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: lightgbm not installed")

from common import load_winning_data
from feature_engineering import build_ordN_features

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def get_valid_range(position: int, ord1: int, ord6: int, known_ords: Dict[int, int] = None) -> Tuple[int, int]:
    """위치별 유효 범위 계산"""
    known_ords = known_ords or {}

    if position == 2:
        min_val = ord1 + 1
        max_val = known_ords.get(4, ord6 - 3) - 2
    elif position == 3:
        min_val = known_ords.get(2, ord1 + 1) + 1
        max_val = known_ords.get(4, ord6 - 2) - 1
    elif position == 4:
        min_val = ord1 + 3
        max_val = ord6 - 2
    elif position == 5:
        min_val = known_ords.get(4, ord1 + 4) + 1
        max_val = ord6 - 1
    else:
        return ord1, ord6

    return max(min_val, 1), min(max_val, 45)


def build_training_data(all_data: List[Dict], position: int) -> Tuple[np.ndarray, np.ndarray]:
    """ord 위치별 학습 데이터 생성"""
    X_list = []
    y_list = []

    for i in range(20, len(all_data)):
        row = all_data[i]
        target_round = row['round']

        ord1 = row['ord1']
        ord6 = row['ord6']
        actual_value = row[f'ord{position}']

        # known_ords 구성
        known_ords = {}
        if position == 2:
            known_ords[4] = row['ord4']
        elif position == 3:
            known_ords[2] = row['ord2']
            known_ords[4] = row['ord4']
        elif position == 5:
            known_ords[4] = row['ord4']

        past_data = [d for d in all_data if d['round'] < target_round]
        recent_10 = past_data[-10:]
        recent_20 = past_data[-20:]

        # 빈도 계산
        freq = Counter(d[f'ord{position}'] for d in recent_10)
        freq_20 = Counter(d[f'ord{position}'] for d in recent_20)

        # 위치별 비율 통계
        ratios = []
        for d in past_data[-50:]:
            d_span = d['ord6'] - d['ord1']
            if d_span > 0:
                ratio = (d[f'ord{position}'] - d['ord1']) / d_span
                ratios.append(ratio)
        avg_ratio = np.mean(ratios) if ratios else 0.5

        span = ord6 - ord1
        expected = ord1 + span * avg_ratio

        # 유효 범위
        min_val, max_val = get_valid_range(position, ord1, ord6, known_ords)

        # 이전 회차 정보
        prev = past_data[-1] if past_data else None
        prev_value = prev[f'ord{position}'] if prev else 0

        # 피처 생성
        features = [
            ord1,
            ord6,
            span,
            avg_ratio,
            expected,
            freq.get(actual_value, 0),  # 타겟의 빈도
            freq_20.get(actual_value, 0),
            min_val,
            max_val,
            max_val - min_val,  # 범위 크기
            prev_value,
            actual_value - prev_value if prev else 0,
            1 if actual_value in [prev[f'ord{i}'] for i in range(1, 7)] else 0 if prev else 0,  # 이월 여부
            actual_value % 10,  # 끝자리
            1 if actual_value in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43} else 0,  # 소수
            actual_value % 2,  # 홀짝
        ]

        # known_ords 추가
        for pos in [2, 3, 4]:
            if pos in known_ords:
                features.append(known_ords[pos])
            else:
                features.append(0)

        X_list.append(features)
        y_list.append(actual_value)

    return np.array(X_list), np.array(y_list)


def train_ordN_model(all_data: List[Dict], position: int, save: bool = True):
    """ord 위치별 LightGBM 모델 학습"""
    if not HAS_LGB:
        print("LightGBM이 설치되지 않았습니다.")
        return None

    print(f"\n=== ord{position} 모델 학습 ===")
    X, y = build_training_data(all_data, position)
    print(f"  데이터: X={X.shape}, y={y.shape}")
    print(f"  y 범위: {y.min()} ~ {y.max()}")

    # Train/Test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # LightGBM 모델 (회귀)
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(X_train, y_train)

    # 평가
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mae = np.mean(np.abs(train_pred - y_train))
    test_mae = np.mean(np.abs(test_pred - y_test))

    print(f"  학습 MAE: {train_mae:.2f}")
    print(f"  테스트 MAE: {test_mae:.2f}")

    # Top-K 적중률 계산
    def calc_topk_accuracy(pred, actual, k=10):
        hits = 0
        for p, a in zip(pred, actual):
            # 예측값 주변 K개
            candidates = list(range(max(1, int(p) - k//2), min(46, int(p) + k//2 + 1)))
            if a in candidates:
                hits += 1
        return hits / len(actual)

    top10_acc = calc_topk_accuracy(test_pred, y_test, k=10)
    print(f"  Top-10 적중률: {top10_acc:.2%}")

    if save:
        model_path = MODEL_DIR / f"ord{position}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  모델 저장: {model_path}")

    return model


class OrdNPredictor:
    """ord2~ord5 예측기"""

    def __init__(self, position: int):
        self.position = position
        self.model = self._load_model()

    def _load_model(self):
        model_path = MODEL_DIR / f"ord{self.position}_model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None

    def predict(
        self,
        target_round: int,
        all_data: List[Dict],
        ord1: int,
        ord6: int,
        known_ords: Dict[int, int] = None,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        상위 K개 후보 예측

        Returns:
            [(ord값, score), ...]
        """
        known_ords = known_ords or {}
        past_data = [d for d in all_data if d['round'] < target_round]
        recent_10 = past_data[-10:]
        recent_20 = past_data[-20:]

        # 유효 범위
        min_val, max_val = get_valid_range(self.position, ord1, ord6, known_ords)

        if min_val > max_val:
            return []

        # 빈도 계산
        freq = Counter(d[f'ord{self.position}'] for d in recent_10)
        freq_20 = Counter(d[f'ord{self.position}'] for d in recent_20)

        # 비율 통계
        ratios = []
        for d in past_data[-50:]:
            d_span = d['ord6'] - d['ord1']
            if d_span > 0:
                ratio = (d[f'ord{self.position}'] - d['ord1']) / d_span
                ratios.append(ratio)
        avg_ratio = np.mean(ratios) if ratios else 0.5

        span = ord6 - ord1
        expected = ord1 + span * avg_ratio

        prev = past_data[-1] if past_data else None
        prev_value = prev[f'ord{self.position}'] if prev else 0

        # 각 후보에 대해 점수 계산
        candidates = []

        for val in range(min_val, max_val + 1):
            if self.model is not None:
                features = [
                    ord1,
                    ord6,
                    span,
                    avg_ratio,
                    expected,
                    freq.get(val, 0),
                    freq_20.get(val, 0),
                    min_val,
                    max_val,
                    max_val - min_val,
                    prev_value,
                    val - prev_value if prev else 0,
                    1 if prev and val in [prev[f'ord{i}'] for i in range(1, 7)] else 0,
                    val % 10,
                    1 if val in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43} else 0,
                    val % 2,
                ]

                for pos in [2, 3, 4]:
                    if pos in known_ords:
                        features.append(known_ords[pos])
                    else:
                        features.append(0)

                pred = self.model.predict([features])[0]
                # 예측값과의 거리로 점수 계산
                score = 1.0 / (1.0 + abs(val - pred))
            else:
                # 모델 없으면 expected 기준
                score = 1.0 / (1.0 + abs(val - expected))

            # 빈도 보너스
            score += freq.get(val, 0) * 0.1

            candidates.append((val, score))

        # 점수 높은 순 정렬
        candidates.sort(key=lambda x: -x[1])

        return candidates[:top_k]


def train_all_models(all_data: List[Dict]):
    """모든 위치 모델 학습"""
    for position in [2, 3, 4, 5]:
        train_ordN_model(all_data, position)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='모델 학습')
    parser.add_argument('--position', type=int, help='특정 위치만 학습')
    args = parser.parse_args()

    data = load_winning_data()
    print(f"총 {len(data)}개 회차 로드")

    if args.train:
        if args.position:
            train_ordN_model(data, args.position)
        else:
            train_all_models(data)
    else:
        train_all_models(data)
