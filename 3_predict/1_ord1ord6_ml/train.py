"""
1_ord1ord6_ml/train.py - XGBoost 기반 (ord1, ord6) 쌍 예측 모델 학습

앙상블:
- XGBoost (70%) + 빈도 기반 (30%)
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import load_winning_data
from feature_engineering import build_ord1ord6_features, build_training_data_ord1ord6

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not installed")


MODEL_PATH = Path(__file__).parent / "model.pkl"


def train_xgboost_model(all_data: list, save: bool = True):
    """XGBoost 모델 학습"""
    if not HAS_XGB:
        print("XGBoost가 설치되지 않았습니다.")
        return None

    print("학습 데이터 생성 중...")
    X, y = build_training_data_ord1ord6(all_data)
    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"  정답 비율: {y.mean():.2%}")

    # Train/Test split (시간순)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n학습: {len(X_train)}, 테스트: {len(X_test)}")

    # XGBoost 모델
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # 평가
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\n학습 정확도: {train_score:.4f}")
    print(f"테스트 정확도: {test_score:.4f}")

    # 피처 중요도
    feature_names = [
        'span', 'ord1_freq_10', 'ord6_freq_10', 'ord1_freq_20', 'ord6_freq_20',
        'span_freq_10', 'ord1_range', 'ord6_range', 'ord1_is_prime', 'ord6_is_prime',
        'ord1_is_odd', 'ord6_is_odd', 'ord1_last_digit', 'ord6_last_digit',
        'ord1_diff_prev', 'ord6_diff_prev', 'span_normalized'
    ]

    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    print("\n피처 중요도 (상위 10개):")
    for i in sorted_idx[:10]:
        print(f"  {feature_names[i]}: {importance[i]:.4f}")

    if save:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n모델 저장: {MODEL_PATH}")

    return model


def load_model():
    """저장된 모델 로드"""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None


class Ord1Ord6Predictor:
    """(ord1, ord6) 쌍 예측기"""

    def __init__(self, model=None):
        self.model = model or load_model()
        self.xgb_weight = 0.7
        self.freq_weight = 0.3

    def predict_pairs(
        self,
        target_round: int,
        all_data: list,
        top_k: int = 100
    ) -> list:
        """
        상위 K개 (ord1, ord6) 쌍 예측

        Returns:
            [(ord1, ord6, score), ...]
        """
        past_data = [d for d in all_data if d['round'] < target_round]
        recent_10 = past_data[-10:]

        # 빈도 계산
        freq_ord1 = Counter(d['ord1'] for d in recent_10)
        freq_ord6 = Counter(d['ord6'] for d in recent_10)

        # 모든 유효한 쌍 생성
        pairs = []
        for ord1 in range(1, 41):
            for ord6 in range(ord1 + 5, 46):
                pairs.append((ord1, ord6))

        # 피처 생성 및 예측
        scored_pairs = []

        for ord1, ord6 in pairs:
            features = build_ord1ord6_features(target_round, all_data, ord1, ord6)

            feature_values = [
                features['span'],
                features['ord1_freq_10'],
                features['ord6_freq_10'],
                features['ord1_freq_20'],
                features['ord6_freq_20'],
                features['span_freq_10'],
                features['ord1_range'],
                features['ord6_range'],
                features['ord1_is_prime'],
                features['ord6_is_prime'],
                features['ord1_is_odd'],
                features['ord6_is_odd'],
                features['ord1_last_digit'],
                features['ord6_last_digit'],
                features['ord1_diff_prev'],
                features['ord6_diff_prev'],
                features['span_normalized'],
            ]

            # XGBoost 점수
            if self.model is not None:
                xgb_score = self.model.predict_proba([feature_values])[0][1]
            else:
                xgb_score = 0.5

            # 빈도 점수
            freq_score = (freq_ord1.get(ord1, 0) + freq_ord6.get(ord6, 0)) / 20.0

            # 앙상블 점수
            total_score = xgb_score * self.xgb_weight + freq_score * self.freq_weight

            scored_pairs.append((ord1, ord6, total_score))

        # 점수 높은 순 정렬
        scored_pairs.sort(key=lambda x: -x[2])

        return scored_pairs[:top_k]


def backtest_model(all_data: list, start_round: int = 1150, end_round: int = 1204):
    """모델 백테스트"""
    predictor = Ord1Ord6Predictor()

    hit_count = 0
    total = 0

    for target_round in range(start_round, end_round + 1):
        actual = next((d for d in all_data if d['round'] == target_round), None)
        if not actual:
            continue

        actual_pair = (actual['ord1'], actual['ord6'])
        predictions = predictor.predict_pairs(target_round, all_data, top_k=100)
        predicted_pairs = [(p[0], p[1]) for p in predictions]

        if actual_pair in predicted_pairs:
            hit_count += 1
            rank = predicted_pairs.index(actual_pair) + 1
            status = f"✓ {rank}위"
        else:
            status = "✗ 미포함"

        total += 1
        print(f"{target_round}회: ({actual['ord1']}, {actual['ord6']}) → {status}")

    print(f"\n적중률: {hit_count}/{total} = {hit_count/total*100:.1f}%")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='모델 학습')
    parser.add_argument('--backtest', action='store_true', help='백테스트')
    args = parser.parse_args()

    data = load_winning_data()
    print(f"총 {len(data)}개 회차 로드")

    if args.train:
        train_xgboost_model(data)
    elif args.backtest:
        backtest_model(data)
    else:
        # 기본: 학습 후 백테스트
        train_xgboost_model(data)
        print("\n" + "="*50)
        print("백테스트 시작")
        print("="*50)
        backtest_model(data)
