"""
ord5 예측 ML 모델 백테스트

ord1, ord4, ord6이 주어졌을 때 ord5 예측
ord5 최빈 범위: 30-39 (54.6%)
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List

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

# ord5 최빈 범위: 30-39 (54.6%)
ORD5_OPTIMAL_RANGE = (30, 39)

FEATURE_COLS = [
    # 기본 피처
    'candidate', 'range_idx', 'range_prob', 'pos5_freq', 'is_hot', 'is_cold',
    'is_prime', 'overall_freq', 'in_recent_3',
    # 위치 피처
    'relative_position', 'distance_from_ord4', 'distance_to_ord6',
    # ord1/ord4/ord6 의존성
    'ord1_is_prime', 'ord4_is_prime', 'ord6_is_prime',
    'ord4_is_odd', 'ord6_is_odd',
    'same_parity_as_ord4', 'same_parity_as_ord6',
    # 범위 피처
    'is_in_optimal_range', 'span_ord4_ord6', 'position_in_remaining',
    # 조건부 피처
    'cond_prob_given_ord4_range', 'ord6_decade',
]


class Ord5FeatureExtractor:
    """ord5 피처 추출기"""

    def __init__(self, train_data: List[Dict]):
        self.train_data = train_data
        self._compute_statistics()

    def _compute_statistics(self):
        self.ball_freq = Counter()
        for r in self.train_data:
            for i in range(1, 7):
                self.ball_freq[r[f'ord{i}']] += 1

        self.pos_freq = {i: Counter() for i in range(1, 7)}
        for r in self.train_data:
            for i in range(1, 7):
                self.pos_freq[i][r[f'ord{i}']] += 1

        self.ord5_range_freq = Counter()
        for r in self.train_data:
            self.ord5_range_freq[get_range_index(r['ord5'])] += 1

        recent = self.train_data[-10:] if len(self.train_data) >= 10 else self.train_data
        freq = Counter()
        for r in recent:
            for i in range(1, 7):
                freq[r[f'ord{i}']] += 1
        sorted_nums = sorted(range(1, 46), key=lambda x: -freq.get(x, 0))
        self.hot_bits = set(sorted_nums[:10])
        self.cold_bits = set(sorted_nums[-10:])

        # ord4 범위별 ord5 분포
        self.ord5_by_ord4_range = defaultdict(Counter)
        for r in self.train_data:
            ord4_range = get_range_index(r['ord4'])
            self.ord5_by_ord4_range[ord4_range][r['ord5']] += 1

    def get_candidates(self, ord1: int, ord4: int, ord6: int) -> List[int]:
        """가능한 ord5 후보 반환"""
        return list(range(ord4 + 1, ord6))

    def extract_features(self, ord1: int, ord4: int, ord6: int, candidate: int) -> Dict:
        features = {'candidate': candidate}

        range_idx = get_range_index(candidate)
        features['range_idx'] = range_idx
        total = sum(self.ord5_range_freq.values())
        features['range_prob'] = self.ord5_range_freq.get(range_idx, 0) / total if total > 0 else 0

        total_pos5 = sum(self.pos_freq[5].values())
        features['pos5_freq'] = self.pos_freq[5].get(candidate, 0) / total_pos5 if total_pos5 > 0 else 0

        features['is_hot'] = 1 if candidate in self.hot_bits else 0
        features['is_cold'] = 1 if candidate in self.cold_bits else 0
        features['is_prime'] = 1 if candidate in PRIMES else 0

        total_freq = sum(self.ball_freq.values())
        features['overall_freq'] = self.ball_freq.get(candidate, 0) / total_freq if total_freq > 0 else 0

        recent_3 = self.train_data[-3:] if len(self.train_data) >= 3 else self.train_data
        recent_balls = set()
        for r in recent_3:
            for i in range(1, 7):
                recent_balls.add(r[f'ord{i}'])
        features['in_recent_3'] = 1 if candidate in recent_balls else 0

        # 위치 피처
        span = ord6 - ord1
        features['relative_position'] = (candidate - ord1) / span if span > 0 else 0
        features['distance_from_ord4'] = candidate - ord4
        features['distance_to_ord6'] = ord6 - candidate

        # 의존성
        features['ord1_is_prime'] = 1 if ord1 in PRIMES else 0
        features['ord4_is_prime'] = 1 if ord4 in PRIMES else 0
        features['ord6_is_prime'] = 1 if ord6 in PRIMES else 0
        features['ord4_is_odd'] = ord4 % 2
        features['ord6_is_odd'] = ord6 % 2
        features['same_parity_as_ord4'] = 1 if (ord4 % 2) == (candidate % 2) else 0
        features['same_parity_as_ord6'] = 1 if (ord6 % 2) == (candidate % 2) else 0

        # 범위 피처
        features['is_in_optimal_range'] = 1 if ORD5_OPTIMAL_RANGE[0] <= candidate <= ORD5_OPTIMAL_RANGE[1] else 0
        features['span_ord4_ord6'] = ord6 - ord4
        remaining_span = ord6 - ord4 - 1
        features['position_in_remaining'] = (candidate - ord4) / remaining_span if remaining_span > 0 else 0

        # 조건부 확률
        ord4_range = get_range_index(ord4)
        cond_dist = self.ord5_by_ord4_range[ord4_range]
        total_cond = sum(cond_dist.values())
        features['cond_prob_given_ord4_range'] = cond_dist.get(candidate, 0) / total_cond if total_cond > 0 else 0

        features['ord6_decade'] = ord6 // 10

        return features


def features_to_array(features: Dict) -> np.ndarray:
    return np.array([features.get(col, 0) for col in FEATURE_COLS])


def run_backtest(min_train_size: int = 50):
    print("=" * 60)
    print("ord5 예측 ML 모델 백테스트")
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

        X_train, y_train = [], []
        train_window = train_data[-100:] if len(train_data) > 100 else train_data

        for j, tr in enumerate(train_window):
            if j < 10:
                continue
            prev_data = train_window[:j]
            extractor = Ord5FeatureExtractor(prev_data)

            candidates = extractor.get_candidates(tr['ord1'], tr['ord4'], tr['ord6'])
            for cand in candidates:
                feat = extractor.extract_features(tr['ord1'], tr['ord4'], tr['ord6'], cand)
                X_train.append(features_to_array(feat))
                y_train.append(1 if cand == tr['ord5'] else 0)

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

        extractor = Ord5FeatureExtractor(train_data)
        candidates = extractor.get_candidates(target['ord1'], target['ord4'], target['ord6'])

        if not candidates:
            continue

        X_test = []
        for cand in candidates:
            feat = extractor.extract_features(target['ord1'], target['ord4'], target['ord6'], cand)
            X_test.append(features_to_array(feat))
        X_test = np.array(X_test)

        if HAS_XGBOOST:
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = np.array([X_test[i][3] for i in range(len(X_test))])

        sorted_indices = np.argsort(-probs)
        sorted_candidates = [candidates[i] for i in sorted_indices]

        actual = target['ord5']
        predicted = sorted_candidates[0]

        top3 = sorted_candidates[:3]
        top5 = sorted_candidates[:5]
        top10 = sorted_candidates[:10]

        try:
            actual_rank = sorted_candidates.index(actual) + 1
        except ValueError:
            actual_rank = len(sorted_candidates) + 1

        results.append({
            'round': target['round'],
            'ord1': target['ord1'],
            'ord4': target['ord4'],
            'ord6': target['ord6'],
            'actual_ord5': actual,
            'predicted_ord5': predicted,
            'actual_rank': actual_rank,
            'num_candidates': len(candidates),
            'exact_match': 1 if predicted == actual else 0,
            'in_top3': 1 if actual in top3 else 0,
            'in_top5': 1 if actual in top5 else 0,
            'in_top10': 1 if actual in top10 else 0,
            'error': abs(predicted - actual)
        })

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

    n = len(results)
    print(f"\n[결과] {n}회차")
    print(f"  Top-1: {sum(r['exact_match'] for r in results)} ({sum(r['exact_match'] for r in results)/n*100:.1f}%)")
    print(f"  Top-5: {sum(r['in_top5'] for r in results)} ({sum(r['in_top5'] for r in results)/n*100:.1f}%)")
    print(f"  Top-10: {sum(r['in_top10'] for r in results)} ({sum(r['in_top10'] for r in results)/n*100:.1f}%)")

    return results


if __name__ == "__main__":
    run_backtest()
