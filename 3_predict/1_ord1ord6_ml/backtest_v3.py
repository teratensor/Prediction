"""
(ord1, ord6) 쌍 예측 ML 모델 백테스트 v3

개선: ord1, ord6 개별 모델로 예측 후 조합 확률 계산
P(ord1, ord6) = P(ord1) * P(ord6 | ord1)
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
from features import load_winning_numbers, PRIMES

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_all_pairs() -> List[Tuple[int, int]]:
    pairs = []
    for ord1 in range(1, 41):
        for ord6 in range(ord1 + 5, 46):
            pairs.append((ord1, ord6))
    return pairs

ALL_PAIRS = get_all_pairs()

# ord1 후보 (1~40)
ORD1_CANDIDATES = list(range(1, 41))
# ord6 후보 (6~45)
ORD6_CANDIDATES = list(range(6, 46))


class Ord1Predictor:
    """ord1 개별 예측기"""

    FEATURES = ['value', 'freq', 'recent_5', 'recent_10', 'gap',
                'is_prime', 'is_odd', 'trend', 'prev_ord1', 'prev_ord1_diff']

    def __init__(self, train_data: List[Dict]):
        self.train_data = train_data
        self.n = len(train_data)
        self._compute_stats()

    def _compute_stats(self):
        self.freq = Counter(r['ord1'] for r in self.train_data)

        recent_5 = self.train_data[-5:]
        recent_10 = self.train_data[-10:]
        self.recent_5 = Counter(r['ord1'] for r in recent_5)
        self.recent_10 = Counter(r['ord1'] for r in recent_10)

        self.last_seen = {}
        for i, r in enumerate(self.train_data):
            self.last_seen[r['ord1']] = i

        # 트렌드
        if self.n >= 20:
            r10 = Counter(r['ord1'] for r in self.train_data[-10:])
            p10 = Counter(r['ord1'] for r in self.train_data[-20:-10])
            self.trend = {v: r10.get(v, 0) - p10.get(v, 0) for v in range(1, 46)}
        else:
            self.trend = defaultdict(int)

        # 이전 ord1
        self.prev_ord1 = self.train_data[-1]['ord1'] if self.train_data else 0

    def extract_features(self, candidate: int) -> np.ndarray:
        return np.array([
            candidate,
            self.freq.get(candidate, 0) / self.n,
            self.recent_5.get(candidate, 0) / 5,
            self.recent_10.get(candidate, 0) / 10,
            (self.n - self.last_seen.get(candidate, 0)) / self.n,
            1 if candidate in PRIMES else 0,
            candidate % 2,
            self.trend.get(candidate, 0),
            self.prev_ord1,
            abs(candidate - self.prev_ord1),
        ])


class Ord6Predictor:
    """ord6 개별 예측기 (ord1 조건부)"""

    FEATURES = ['value', 'freq', 'recent_5', 'recent_10', 'gap',
                'is_prime', 'is_odd', 'trend', 'ord1', 'span',
                'span_freq', 'cond_freq']

    def __init__(self, train_data: List[Dict]):
        self.train_data = train_data
        self.n = len(train_data)
        self._compute_stats()

    def _compute_stats(self):
        self.freq = Counter(r['ord6'] for r in self.train_data)
        self.span_freq = Counter(r['ord6'] - r['ord1'] for r in self.train_data)

        recent_5 = self.train_data[-5:]
        recent_10 = self.train_data[-10:]
        self.recent_5 = Counter(r['ord6'] for r in recent_5)
        self.recent_10 = Counter(r['ord6'] for r in recent_10)

        self.last_seen = {}
        for i, r in enumerate(self.train_data):
            self.last_seen[r['ord6']] = i

        # 트렌드
        if self.n >= 20:
            r10 = Counter(r['ord6'] for r in self.train_data[-10:])
            p10 = Counter(r['ord6'] for r in self.train_data[-20:-10])
            self.trend = {v: r10.get(v, 0) - p10.get(v, 0) for v in range(1, 46)}
        else:
            self.trend = defaultdict(int)

        # 조건부 빈도 P(ord6 | ord1)
        self.cond_freq = defaultdict(Counter)
        for r in self.train_data:
            self.cond_freq[r['ord1']][r['ord6']] += 1

    def extract_features(self, ord1: int, candidate: int) -> np.ndarray:
        span = candidate - ord1
        cond = self.cond_freq[ord1]
        cond_total = sum(cond.values())

        return np.array([
            candidate,
            self.freq.get(candidate, 0) / self.n,
            self.recent_5.get(candidate, 0) / 5,
            self.recent_10.get(candidate, 0) / 10,
            (self.n - self.last_seen.get(candidate, 0)) / self.n,
            1 if candidate in PRIMES else 0,
            candidate % 2,
            self.trend.get(candidate, 0),
            ord1,
            span,
            self.span_freq.get(span, 0) / self.n,
            cond.get(candidate, 0) / cond_total if cond_total > 0 else 0,
        ])


def run_backtest_v3(min_train_size: int = 50):
    print("=" * 60)
    print("(ord1, ord6) 개별 예측 후 조합 - v3")
    print("=" * 60)
    print(f"총 쌍: {len(ALL_PAIRS)}개")

    data = load_winning_numbers()
    print(f"데이터: {len(data)}회차")

    results = []

    for i in range(min_train_size, len(data)):
        train_data = data[:i]
        target = data[i]

        if (i - min_train_size) % 50 == 0:
            print(f"  진행: {i - min_train_size + 1}/{len(data) - min_train_size}")

        train_window = train_data[-100:] if len(train_data) > 100 else train_data

        # ord1 모델 학습
        X1_train, y1_train = [], []
        for j, tr in enumerate(train_window):
            if j < 10:
                continue
            pred = Ord1Predictor(train_window[:j])
            for cand in ORD1_CANDIDATES:
                X1_train.append(pred.extract_features(cand))
                y1_train.append(1 if cand == tr['ord1'] else 0)

        # ord6 모델 학습
        X6_train, y6_train = [], []
        for j, tr in enumerate(train_window):
            if j < 10:
                continue
            pred = Ord6Predictor(train_window[:j])
            for cand in ORD6_CANDIDATES:
                if cand > tr['ord1'] + 4:
                    X6_train.append(pred.extract_features(tr['ord1'], cand))
                    y6_train.append(1 if cand == tr['ord6'] else 0)

        if len(X1_train) < 10 or len(X6_train) < 10:
            continue

        X1_train = np.array(X1_train)
        y1_train = np.array(y1_train)
        X6_train = np.array(X6_train)
        y6_train = np.array(y6_train)

        if HAS_XGBOOST:
            model1 = xgb.XGBClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'
            )
            model1.fit(X1_train, y1_train)

            model6 = xgb.XGBClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'
            )
            model6.fit(X6_train, y6_train)

        # 예측
        ord1_pred = Ord1Predictor(train_data)
        ord6_pred = Ord6Predictor(train_data)

        # ord1 확률
        X1_test = np.array([ord1_pred.extract_features(c) for c in ORD1_CANDIDATES])
        if HAS_XGBOOST:
            ord1_probs = model1.predict_proba(X1_test)[:, 1]
        else:
            ord1_probs = np.array([ord1_pred.freq.get(c, 0) for c in ORD1_CANDIDATES])
            ord1_probs = ord1_probs / ord1_probs.sum()

        # 각 쌍에 대해 P(ord1) * P(ord6 | ord1) 계산
        pair_probs = []
        for ord1, ord6 in ALL_PAIRS:
            ord1_idx = ord1 - 1  # 0-indexed
            p_ord1 = ord1_probs[ord1_idx]

            X6 = ord6_pred.extract_features(ord1, ord6).reshape(1, -1)
            if HAS_XGBOOST:
                p_ord6 = model6.predict_proba(X6)[0, 1]
            else:
                p_ord6 = ord6_pred.freq.get(ord6, 0) / len(train_data)

            pair_probs.append(p_ord1 * p_ord6)

        pair_probs = np.array(pair_probs)

        # 정렬
        sorted_indices = np.argsort(-pair_probs)
        sorted_pairs = [ALL_PAIRS[i] for i in sorted_indices]

        actual = (target['ord1'], target['ord6'])
        predicted = sorted_pairs[0]

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
            'in_top10': 1 if actual in sorted_pairs[:10] else 0,
            'in_top20': 1 if actual in sorted_pairs[:20] else 0,
            'in_top30': 1 if actual in sorted_pairs[:30] else 0,
            'in_top50': 1 if actual in sorted_pairs[:50] else 0,
            'in_top100': 1 if actual in sorted_pairs[:100] else 0,
        })

    # 결과 저장
    output_path = OUTPUT_DIR / "backtest_results_v3.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[저장] {output_path}")

    # 통계
    n = len(results)
    print(f"\n[결과] {n}회차 (총 {len(ALL_PAIRS)}개 쌍)")
    print(f"  Top-1:   {sum(r['exact_match'] for r in results):3d} ({sum(r['exact_match'] for r in results)/n*100:5.1f}%)")
    print(f"  Top-10:  {sum(r['in_top10'] for r in results):3d} ({sum(r['in_top10'] for r in results)/n*100:5.1f}%)")
    print(f"  Top-20:  {sum(r['in_top20'] for r in results):3d} ({sum(r['in_top20'] for r in results)/n*100:5.1f}%)")
    print(f"  Top-30:  {sum(r['in_top30'] for r in results):3d} ({sum(r['in_top30'] for r in results)/n*100:5.1f}%)")
    print(f"  Top-50:  {sum(r['in_top50'] for r in results):3d} ({sum(r['in_top50'] for r in results)/n*100:5.1f}%)")
    print(f"  Top-100: {sum(r['in_top100'] for r in results):3d} ({sum(r['in_top100'] for r in results)/n*100:5.1f}%)")
    print(f"  평균 순위: {sum(r['actual_rank'] for r in results)/n:.1f}")

    return results


if __name__ == "__main__":
    run_backtest_v3()
