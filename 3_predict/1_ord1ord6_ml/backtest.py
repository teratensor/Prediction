"""
(ord1, ord6) 쌍 예측 ML 모델 백테스트

477개의 가능한 (ord1, ord6) 쌍 중 Top-30 적중률 측정
ord1: 1~10, ord6: 35~45 (실제 분포 기반)
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import sys
sys.path.append(str(Path(__file__).parent.parent / "9_ord2_ml"))
from features import load_winning_numbers, PRIMES, get_range_index

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 가능한 (ord1, ord6) 쌍 생성
def get_all_pairs() -> List[Tuple[int, int]]:
    """모든 가능한 (ord1, ord6) 쌍 반환"""
    pairs = []
    for ord1 in range(1, 41):  # ord1: 1~40
        for ord6 in range(ord1 + 5, 46):  # ord6: ord1+5 ~ 45
            pairs.append((ord1, ord6))
    return pairs

ALL_PAIRS = get_all_pairs()
print(f"총 (ord1, ord6) 쌍: {len(ALL_PAIRS)}개")


FEATURE_COLS = [
    # 기본 피처
    'ord1', 'ord6', 'span',
    # ord1 피처
    'ord1_range_idx', 'ord1_freq', 'ord1_is_prime', 'ord1_is_odd',
    # ord6 피처
    'ord6_range_idx', 'ord6_freq', 'ord6_is_prime', 'ord6_is_odd',
    # 쌍 피처
    'pair_freq', 'span_category', 'sum_ord1_ord6',
    'both_prime', 'both_odd', 'both_even',
    # 통계 피처
    'ord1_recent_freq', 'ord6_recent_freq',
    'span_freq', 'expected_sum_match',
    # 조건부 피처
    'ord6_given_ord1_freq', 'ord1_given_ord6_freq',
]


class Ord1Ord6FeatureExtractor:
    """(ord1, ord6) 쌍 피처 추출기"""

    def __init__(self, train_data: List[Dict]):
        self.train_data = train_data
        self._compute_statistics()

    def _compute_statistics(self):
        # ord1, ord6 개별 빈도
        self.ord1_freq = Counter()
        self.ord6_freq = Counter()
        for r in self.train_data:
            self.ord1_freq[r['ord1']] += 1
            self.ord6_freq[r['ord6']] += 1

        # (ord1, ord6) 쌍 빈도
        self.pair_freq = Counter()
        for r in self.train_data:
            self.pair_freq[(r['ord1'], r['ord6'])] += 1

        # span 빈도
        self.span_freq = Counter()
        for r in self.train_data:
            self.span_freq[r['ord6'] - r['ord1']] += 1

        # 최근 10회 빈도
        recent = self.train_data[-10:] if len(self.train_data) >= 10 else self.train_data
        self.ord1_recent = Counter(r['ord1'] for r in recent)
        self.ord6_recent = Counter(r['ord6'] for r in recent)

        # 조건부 빈도: P(ord6 | ord1)
        self.ord6_given_ord1 = defaultdict(Counter)
        self.ord1_given_ord6 = defaultdict(Counter)
        for r in self.train_data:
            self.ord6_given_ord1[r['ord1']][r['ord6']] += 1
            self.ord1_given_ord6[r['ord6']][r['ord1']] += 1

        # 평균 합계
        self.avg_sum = np.mean([sum(r[f'ord{i}'] for i in range(1, 7)) for r in self.train_data])

    def extract_features(self, ord1: int, ord6: int) -> Dict:
        features = {}

        # 기본
        features['ord1'] = ord1
        features['ord6'] = ord6
        features['span'] = ord6 - ord1

        # ord1 피처
        features['ord1_range_idx'] = get_range_index(ord1)
        total_ord1 = sum(self.ord1_freq.values())
        features['ord1_freq'] = self.ord1_freq.get(ord1, 0) / total_ord1 if total_ord1 > 0 else 0
        features['ord1_is_prime'] = 1 if ord1 in PRIMES else 0
        features['ord1_is_odd'] = ord1 % 2

        # ord6 피처
        features['ord6_range_idx'] = get_range_index(ord6)
        total_ord6 = sum(self.ord6_freq.values())
        features['ord6_freq'] = self.ord6_freq.get(ord6, 0) / total_ord6 if total_ord6 > 0 else 0
        features['ord6_is_prime'] = 1 if ord6 in PRIMES else 0
        features['ord6_is_odd'] = ord6 % 2

        # 쌍 피처
        total_pairs = sum(self.pair_freq.values())
        features['pair_freq'] = self.pair_freq.get((ord1, ord6), 0) / total_pairs if total_pairs > 0 else 0

        span = ord6 - ord1
        if span <= 30:
            features['span_category'] = 0
        elif span <= 35:
            features['span_category'] = 1
        elif span <= 40:
            features['span_category'] = 2
        else:
            features['span_category'] = 3

        features['sum_ord1_ord6'] = ord1 + ord6
        features['both_prime'] = 1 if (ord1 in PRIMES and ord6 in PRIMES) else 0
        features['both_odd'] = 1 if (ord1 % 2 == 1 and ord6 % 2 == 1) else 0
        features['both_even'] = 1 if (ord1 % 2 == 0 and ord6 % 2 == 0) else 0

        # 통계 피처
        features['ord1_recent_freq'] = self.ord1_recent.get(ord1, 0) / 10
        features['ord6_recent_freq'] = self.ord6_recent.get(ord6, 0) / 10

        total_span = sum(self.span_freq.values())
        features['span_freq'] = self.span_freq.get(span, 0) / total_span if total_span > 0 else 0

        # 예상 합계 적합도 (139 기준)
        expected_mid_sum = self.avg_sum - ord1 - ord6
        actual_mid_range = (ord6 - ord1 - 1) * 23  # 중간값들의 대략적 합
        features['expected_sum_match'] = 1 - abs(expected_mid_sum - actual_mid_range / 4) / 100

        # 조건부 빈도
        ord6_given = self.ord6_given_ord1[ord1]
        total_given = sum(ord6_given.values())
        features['ord6_given_ord1_freq'] = ord6_given.get(ord6, 0) / total_given if total_given > 0 else 0

        ord1_given = self.ord1_given_ord6[ord6]
        total_given = sum(ord1_given.values())
        features['ord1_given_ord6_freq'] = ord1_given.get(ord1, 0) / total_given if total_given > 0 else 0

        return features


def features_to_array(features: Dict) -> np.ndarray:
    return np.array([features.get(col, 0) for col in FEATURE_COLS])


def run_backtest(min_train_size: int = 50):
    print("=" * 60)
    print("(ord1, ord6) 쌍 예측 ML 모델 백테스트")
    print("=" * 60)

    data = load_winning_numbers()
    print(f"데이터: {len(data)}회차")

    results = []
    feature_importance_sum = defaultdict(float)
    feature_importance_count = 0

    for i in range(min_train_size, len(data)):
        train_data = data[:i]
        target = data[i]

        if (i - min_train_size) % 50 == 0:
            print(f"  진행: {i - min_train_size + 1}/{len(data) - min_train_size}")

        # 학습 데이터 준비
        X_train, y_train = [], []
        train_window = train_data[-100:] if len(train_data) > 100 else train_data

        for j, tr in enumerate(train_window):
            if j < 10:
                continue
            prev_data = train_window[:j]
            extractor = Ord1Ord6FeatureExtractor(prev_data)

            for pair in ALL_PAIRS:
                feat = extractor.extract_features(pair[0], pair[1])
                X_train.append(features_to_array(feat))
                y_train.append(1 if (pair[0] == tr['ord1'] and pair[1] == tr['ord6']) else 0)

        if len(X_train) < 10:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if HAS_XGBOOST:
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.08,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                random_state=42, use_label_encoder=False,
                eval_metric='logloss', verbosity=0
            )
            model.fit(X_train, y_train)

            for idx, imp in enumerate(model.feature_importances_):
                if idx < len(FEATURE_COLS):
                    feature_importance_sum[FEATURE_COLS[idx]] += imp
            feature_importance_count += 1

        # 예측
        extractor = Ord1Ord6FeatureExtractor(train_data)

        X_test = []
        for pair in ALL_PAIRS:
            feat = extractor.extract_features(pair[0], pair[1])
            X_test.append(features_to_array(feat))
        X_test = np.array(X_test)

        if HAS_XGBOOST:
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = np.array([X_test[i][11] for i in range(len(X_test))])  # pair_freq

        sorted_indices = np.argsort(-probs)
        sorted_pairs = [ALL_PAIRS[i] for i in sorted_indices]

        actual = (target['ord1'], target['ord6'])
        predicted = sorted_pairs[0]

        top10 = sorted_pairs[:10]
        top20 = sorted_pairs[:20]
        top30 = sorted_pairs[:30]
        top50 = sorted_pairs[:50]

        try:
            actual_rank = sorted_pairs.index(actual) + 1
        except ValueError:
            actual_rank = len(sorted_pairs) + 1

        results.append({
            'round': target['round'],
            'actual_ord1': actual[0],
            'actual_ord6': actual[1],
            'predicted_ord1': predicted[0],
            'predicted_ord6': predicted[1],
            'actual_rank': actual_rank,
            'exact_match': 1 if predicted == actual else 0,
            'in_top10': 1 if actual in top10 else 0,
            'in_top20': 1 if actual in top20 else 0,
            'in_top30': 1 if actual in top30 else 0,
            'in_top50': 1 if actual in top50 else 0,
        })

    # 결과 저장
    output_path = OUTPUT_DIR / "backtest_results.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[저장] {output_path}")

    if HAS_XGBOOST and feature_importance_count > 0:
        imp_path = OUTPUT_DIR / "feature_importance.csv"
        with open(imp_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['feature', 'importance'])
            avg_imp = {k: v/feature_importance_count for k, v in feature_importance_sum.items()}
            for feat, imp in sorted(avg_imp.items(), key=lambda x: -x[1]):
                writer.writerow([feat, f"{imp:.6f}"])
        print(f"[저장] {imp_path}")

    # 통계 출력
    n = len(results)
    print(f"\n[결과] {n}회차 (총 {len(ALL_PAIRS)}개 쌍 중)")
    print(f"  Top-1:  {sum(r['exact_match'] for r in results)} ({sum(r['exact_match'] for r in results)/n*100:.1f}%)")
    print(f"  Top-10: {sum(r['in_top10'] for r in results)} ({sum(r['in_top10'] for r in results)/n*100:.1f}%)")
    print(f"  Top-20: {sum(r['in_top20'] for r in results)} ({sum(r['in_top20'] for r in results)/n*100:.1f}%)")
    print(f"  Top-30: {sum(r['in_top30'] for r in results)} ({sum(r['in_top30'] for r in results)/n*100:.1f}%)")
    print(f"  Top-50: {sum(r['in_top50'] for r in results)} ({sum(r['in_top50'] for r in results)/n*100:.1f}%)")
    print(f"  평균 순위: {sum(r['actual_rank'] for r in results)/n:.1f}")

    return results


if __name__ == "__main__":
    run_backtest()
