"""
ord4 예측 ML 모델 백테스트

ord1, ord6이 주어졌을 때 ord4 예측
현재 공식: ord4 = round(ord1 + (ord6 - ord1) × 0.60) ± 10
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

FEATURE_COLS = [
    # 기본 피처
    'candidate', 'range_idx', 'range_prob', 'pos4_freq', 'is_hot', 'is_cold',
    'is_prime', 'overall_freq', 'in_recent_3',
    # 위치 피처 (핵심!)
    'relative_position', 'distance_from_formula', 'formula_value',
    # ord1/ord6 의존성
    'ord1_is_prime', 'ord6_is_prime', 'ord1_is_odd', 'ord6_is_odd',
    'span', 'span_category',
    # 범위 피처
    'is_in_formula_range', 'deviation_from_center',
]


class Ord4FeatureExtractor:
    """ord4 피처 추출기"""

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

        self.ord4_range_freq = Counter()
        for r in self.train_data:
            self.ord4_range_freq[get_range_index(r['ord4'])] += 1

        recent = self.train_data[-10:] if len(self.train_data) >= 10 else self.train_data
        freq = Counter()
        for r in recent:
            for i in range(1, 7):
                freq[r[f'ord{i}']] += 1
        sorted_nums = sorted(range(1, 46), key=lambda x: -freq.get(x, 0))
        self.hot_bits = set(sorted_nums[:10])
        self.cold_bits = set(sorted_nums[-10:])

    def get_formula_value(self, ord1: int, ord6: int) -> int:
        """공식 기반 ord4 예측값"""
        return round(ord1 + (ord6 - ord1) * 0.60)

    def get_candidates(self, ord1: int, ord6: int, range_size: int = 10) -> List[int]:
        """공식 값 ± range_size 범위의 후보"""
        formula_val = self.get_formula_value(ord1, ord6)
        candidates = []
        for v in range(formula_val - range_size, formula_val + range_size + 1):
            if ord1 < v < ord6:
                candidates.append(v)
        return sorted(set(candidates))

    def extract_features(self, ord1: int, ord6: int, candidate: int) -> Dict:
        features = {'candidate': candidate}

        range_idx = get_range_index(candidate)
        features['range_idx'] = range_idx
        total = sum(self.ord4_range_freq.values())
        features['range_prob'] = self.ord4_range_freq.get(range_idx, 0) / total if total > 0 else 0

        total_pos4 = sum(self.pos_freq[4].values())
        features['pos4_freq'] = self.pos_freq[4].get(candidate, 0) / total_pos4 if total_pos4 > 0 else 0

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

        # 위치 피처 (가장 중요!)
        span = ord6 - ord1
        features['relative_position'] = (candidate - ord1) / span if span > 0 else 0

        formula_val = self.get_formula_value(ord1, ord6)
        features['formula_value'] = formula_val
        features['distance_from_formula'] = abs(candidate - formula_val)

        # ord1/ord6 의존성
        features['ord1_is_prime'] = 1 if ord1 in PRIMES else 0
        features['ord6_is_prime'] = 1 if ord6 in PRIMES else 0
        features['ord1_is_odd'] = ord1 % 2
        features['ord6_is_odd'] = ord6 % 2
        features['span'] = span

        if span <= 25:
            features['span_category'] = 0
        elif span <= 35:
            features['span_category'] = 1
        else:
            features['span_category'] = 2

        # 공식 범위 내 여부
        features['is_in_formula_range'] = 1 if abs(candidate - formula_val) <= 5 else 0
        features['deviation_from_center'] = abs(features['relative_position'] - 0.60)

        return features


def features_to_array(features: Dict) -> np.ndarray:
    return np.array([features.get(col, 0) for col in FEATURE_COLS])


def run_backtest(min_train_size: int = 50):
    print("=" * 60)
    print("ord4 예측 ML 모델 백테스트")
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
            extractor = Ord4FeatureExtractor(prev_data)

            candidates = extractor.get_candidates(tr['ord1'], tr['ord6'])
            for cand in candidates:
                feat = extractor.extract_features(tr['ord1'], tr['ord6'], cand)
                X_train.append(features_to_array(feat))
                y_train.append(1 if cand == tr['ord4'] else 0)

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

        extractor = Ord4FeatureExtractor(train_data)
        candidates = extractor.get_candidates(target['ord1'], target['ord6'])

        if not candidates:
            continue

        X_test = []
        for cand in candidates:
            feat = extractor.extract_features(target['ord1'], target['ord6'], cand)
            X_test.append(features_to_array(feat))
        X_test = np.array(X_test)

        if HAS_XGBOOST:
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = np.array([1.0 / (1 + X_test[i][11]) for i in range(len(X_test))])

        sorted_indices = np.argsort(-probs)
        sorted_candidates = [candidates[i] for i in sorted_indices]

        actual = target['ord4']
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
            'ord6': target['ord6'],
            'actual_ord4': actual,
            'predicted_ord4': predicted,
            'formula_ord4': extractor.get_formula_value(target['ord1'], target['ord6']),
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

    # 공식 정확도 비교
    formula_exact = sum(1 for r in results if r['formula_ord4'] == r['actual_ord4'])
    print(f"  공식 정확도: {formula_exact} ({formula_exact/n*100:.1f}%)")

    return results


if __name__ == "__main__":
    run_backtest()
