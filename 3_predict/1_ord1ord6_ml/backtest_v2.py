"""
(ord1, ord6) 쌍 예측 ML 모델 백테스트 v2

개선 사항:
1. 시계열 피처 강화 (최근 N회 트렌드)
2. 연속 출현/미출현 패턴
3. 앙상블: XGBoost + 빈도 기반 점수
4. ord1, ord6 개별 예측 후 결합
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


def get_all_pairs() -> List[Tuple[int, int]]:
    pairs = []
    for ord1 in range(1, 41):
        for ord6 in range(ord1 + 5, 46):
            pairs.append((ord1, ord6))
    return pairs

ALL_PAIRS = get_all_pairs()


FEATURE_COLS_V2 = [
    # 기본 피처
    'ord1', 'ord6', 'span',
    # ord1 피처
    'ord1_freq', 'ord1_is_prime', 'ord1_is_odd',
    'ord1_recent_5', 'ord1_recent_10', 'ord1_recent_20',
    'ord1_gap',  # 마지막 출현 후 경과 회차
    # ord6 피처
    'ord6_freq', 'ord6_is_prime', 'ord6_is_odd',
    'ord6_recent_5', 'ord6_recent_10', 'ord6_recent_20',
    'ord6_gap',
    # 쌍 피처
    'pair_freq', 'pair_recent_20',
    'span_freq', 'span_recent_10',
    # 조건부 피처
    'ord6_given_ord1_freq', 'ord1_given_ord6_freq',
    # 트렌드 피처
    'ord1_trend',  # 최근 증가/감소 추세
    'ord6_trend',
    'span_trend',
    # 조합 피처
    'sum_ord1_ord6',
    'both_prime', 'both_odd', 'parity_match',
]


class Ord1Ord6FeatureExtractorV2:
    """개선된 피처 추출기"""

    def __init__(self, train_data: List[Dict]):
        self.train_data = train_data
        self.n = len(train_data)
        self._compute_statistics()

    def _compute_statistics(self):
        # 기본 빈도
        self.ord1_freq = Counter(r['ord1'] for r in self.train_data)
        self.ord6_freq = Counter(r['ord6'] for r in self.train_data)
        self.pair_freq = Counter((r['ord1'], r['ord6']) for r in self.train_data)
        self.span_freq = Counter(r['ord6'] - r['ord1'] for r in self.train_data)

        # 최근 N회 빈도
        self.ord1_recent = {5: Counter(), 10: Counter(), 20: Counter()}
        self.ord6_recent = {5: Counter(), 10: Counter(), 20: Counter()}
        self.pair_recent_20 = Counter()
        self.span_recent_10 = Counter()

        for i, r in enumerate(self.train_data):
            if i >= self.n - 5:
                self.ord1_recent[5][r['ord1']] += 1
                self.ord6_recent[5][r['ord6']] += 1
            if i >= self.n - 10:
                self.ord1_recent[10][r['ord1']] += 1
                self.ord6_recent[10][r['ord6']] += 1
                self.span_recent_10[r['ord6'] - r['ord1']] += 1
            if i >= self.n - 20:
                self.ord1_recent[20][r['ord1']] += 1
                self.ord6_recent[20][r['ord6']] += 1
                self.pair_recent_20[(r['ord1'], r['ord6'])] += 1

        # 마지막 출현 위치 (gap 계산용)
        self.ord1_last = {}
        self.ord6_last = {}
        for i, r in enumerate(self.train_data):
            self.ord1_last[r['ord1']] = i
            self.ord6_last[r['ord6']] = i

        # 조건부 빈도
        self.ord6_given_ord1 = defaultdict(Counter)
        self.ord1_given_ord6 = defaultdict(Counter)
        for r in self.train_data:
            self.ord6_given_ord1[r['ord1']][r['ord6']] += 1
            self.ord1_given_ord6[r['ord6']][r['ord1']] += 1

        # 트렌드 계산 (최근 10회 vs 이전 10회)
        if self.n >= 20:
            recent_10 = self.train_data[-10:]
            prev_10 = self.train_data[-20:-10]

            self.ord1_trend_data = {}
            self.ord6_trend_data = {}
            self.span_trend_data = {}

            for v in range(1, 46):
                recent_cnt = sum(1 for r in recent_10 if r['ord1'] == v)
                prev_cnt = sum(1 for r in prev_10 if r['ord1'] == v)
                self.ord1_trend_data[v] = recent_cnt - prev_cnt

                recent_cnt = sum(1 for r in recent_10 if r['ord6'] == v)
                prev_cnt = sum(1 for r in prev_10 if r['ord6'] == v)
                self.ord6_trend_data[v] = recent_cnt - prev_cnt

            for span in range(5, 45):
                recent_cnt = sum(1 for r in recent_10 if r['ord6'] - r['ord1'] == span)
                prev_cnt = sum(1 for r in prev_10 if r['ord6'] - r['ord1'] == span)
                self.span_trend_data[span] = recent_cnt - prev_cnt
        else:
            self.ord1_trend_data = defaultdict(int)
            self.ord6_trend_data = defaultdict(int)
            self.span_trend_data = defaultdict(int)

    def extract_features(self, ord1: int, ord6: int) -> Dict:
        features = {}
        span = ord6 - ord1

        # 기본
        features['ord1'] = ord1
        features['ord6'] = ord6
        features['span'] = span

        # ord1 피처
        features['ord1_freq'] = self.ord1_freq.get(ord1, 0) / self.n
        features['ord1_is_prime'] = 1 if ord1 in PRIMES else 0
        features['ord1_is_odd'] = ord1 % 2
        features['ord1_recent_5'] = self.ord1_recent[5].get(ord1, 0) / 5
        features['ord1_recent_10'] = self.ord1_recent[10].get(ord1, 0) / 10
        features['ord1_recent_20'] = self.ord1_recent[20].get(ord1, 0) / 20
        features['ord1_gap'] = (self.n - self.ord1_last.get(ord1, 0)) / self.n

        # ord6 피처
        features['ord6_freq'] = self.ord6_freq.get(ord6, 0) / self.n
        features['ord6_is_prime'] = 1 if ord6 in PRIMES else 0
        features['ord6_is_odd'] = ord6 % 2
        features['ord6_recent_5'] = self.ord6_recent[5].get(ord6, 0) / 5
        features['ord6_recent_10'] = self.ord6_recent[10].get(ord6, 0) / 10
        features['ord6_recent_20'] = self.ord6_recent[20].get(ord6, 0) / 20
        features['ord6_gap'] = (self.n - self.ord6_last.get(ord6, 0)) / self.n

        # 쌍 피처
        features['pair_freq'] = self.pair_freq.get((ord1, ord6), 0) / self.n
        features['pair_recent_20'] = self.pair_recent_20.get((ord1, ord6), 0) / 20
        features['span_freq'] = self.span_freq.get(span, 0) / self.n
        features['span_recent_10'] = self.span_recent_10.get(span, 0) / 10

        # 조건부 빈도
        ord6_given = self.ord6_given_ord1[ord1]
        total = sum(ord6_given.values())
        features['ord6_given_ord1_freq'] = ord6_given.get(ord6, 0) / total if total > 0 else 0

        ord1_given = self.ord1_given_ord6[ord6]
        total = sum(ord1_given.values())
        features['ord1_given_ord6_freq'] = ord1_given.get(ord1, 0) / total if total > 0 else 0

        # 트렌드
        features['ord1_trend'] = self.ord1_trend_data.get(ord1, 0)
        features['ord6_trend'] = self.ord6_trend_data.get(ord6, 0)
        features['span_trend'] = self.span_trend_data.get(span, 0)

        # 조합 피처
        features['sum_ord1_ord6'] = ord1 + ord6
        features['both_prime'] = 1 if (ord1 in PRIMES and ord6 in PRIMES) else 0
        features['both_odd'] = 1 if (ord1 % 2 == 1 and ord6 % 2 == 1) else 0
        features['parity_match'] = 1 if (ord1 % 2 == ord6 % 2) else 0

        return features


def features_to_array(features: Dict) -> np.ndarray:
    return np.array([features.get(col, 0) for col in FEATURE_COLS_V2])


class FrequencyScorer:
    """빈도 기반 스코어러 (앙상블용)"""

    def __init__(self, train_data: List[Dict]):
        self.n = len(train_data)

        # 개별 빈도
        self.ord1_freq = Counter(r['ord1'] for r in train_data)
        self.ord6_freq = Counter(r['ord6'] for r in train_data)
        self.pair_freq = Counter((r['ord1'], r['ord6']) for r in train_data)
        self.span_freq = Counter(r['ord6'] - r['ord1'] for r in train_data)

        # 최근 빈도
        recent = train_data[-20:]
        self.ord1_recent = Counter(r['ord1'] for r in recent)
        self.ord6_recent = Counter(r['ord6'] for r in recent)

    def score(self, ord1: int, ord6: int) -> float:
        score = 0
        span = ord6 - ord1

        # ord1 빈도 (30점)
        score += (self.ord1_freq.get(ord1, 0) / self.n) * 30

        # ord6 빈도 (30점)
        score += (self.ord6_freq.get(ord6, 0) / self.n) * 30

        # span 빈도 (20점)
        score += (self.span_freq.get(span, 0) / self.n) * 20

        # 쌍 빈도 (10점)
        score += (self.pair_freq.get((ord1, ord6), 0) / self.n) * 100

        # 최근 출현 보너스 (10점)
        score += (self.ord1_recent.get(ord1, 0) / 20) * 5
        score += (self.ord6_recent.get(ord6, 0) / 20) * 5

        return score


def run_backtest_v2(min_train_size: int = 50, ensemble_weight: float = 0.3):
    print("=" * 60)
    print("(ord1, ord6) 쌍 예측 ML 모델 백테스트 v2")
    print("=" * 60)
    print(f"총 쌍: {len(ALL_PAIRS)}개")
    print(f"앙상블: XGBoost {1-ensemble_weight:.0%} + Frequency {ensemble_weight:.0%}")

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
            extractor = Ord1Ord6FeatureExtractorV2(prev_data)

            for pair in ALL_PAIRS:
                feat = extractor.extract_features(pair[0], pair[1])
                X_train.append(features_to_array(feat))
                y_train.append(1 if (pair[0] == tr['ord1'] and pair[1] == tr['ord6']) else 0)

        if len(X_train) < 10:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # XGBoost 학습
        if HAS_XGBOOST:
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.08,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, use_label_encoder=False,
                eval_metric='logloss', verbosity=0
            )
            model.fit(X_train, y_train)

            for idx, imp in enumerate(model.feature_importances_):
                if idx < len(FEATURE_COLS_V2):
                    feature_importance_sum[FEATURE_COLS_V2[idx]] += imp
            feature_importance_count += 1

        # 예측
        extractor = Ord1Ord6FeatureExtractorV2(train_data)
        freq_scorer = FrequencyScorer(train_data)

        X_test = []
        freq_scores = []
        for pair in ALL_PAIRS:
            feat = extractor.extract_features(pair[0], pair[1])
            X_test.append(features_to_array(feat))
            freq_scores.append(freq_scorer.score(pair[0], pair[1]))

        X_test = np.array(X_test)
        freq_scores = np.array(freq_scores)

        # 앙상블
        if HAS_XGBOOST:
            xgb_probs = model.predict_proba(X_test)[:, 1]
        else:
            xgb_probs = np.zeros(len(ALL_PAIRS))

        # 정규화
        if freq_scores.max() > freq_scores.min():
            freq_probs = (freq_scores - freq_scores.min()) / (freq_scores.max() - freq_scores.min())
        else:
            freq_probs = np.ones_like(freq_scores) / len(freq_scores)

        final_probs = (1 - ensemble_weight) * xgb_probs + ensemble_weight * freq_probs

        # 정렬
        sorted_indices = np.argsort(-final_probs)
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
            'in_top150': 1 if actual in sorted_pairs[:150] else 0,
            'in_top200': 1 if actual in sorted_pairs[:200] else 0,
        })

    # 결과 저장
    output_path = OUTPUT_DIR / "backtest_results_v2.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[저장] {output_path}")

    if HAS_XGBOOST and feature_importance_count > 0:
        imp_path = OUTPUT_DIR / "feature_importance_v2.csv"
        with open(imp_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['feature', 'importance'])
            avg_imp = {k: v/feature_importance_count for k, v in feature_importance_sum.items()}
            for feat, imp in sorted(avg_imp.items(), key=lambda x: -x[1]):
                writer.writerow([feat, f"{imp:.6f}"])
        print(f"[저장] {imp_path}")

    # 통계
    n = len(results)
    print(f"\n[결과] {n}회차 (총 {len(ALL_PAIRS)}개 쌍)")
    print(f"  Top-1:   {sum(r['exact_match'] for r in results):3d} ({sum(r['exact_match'] for r in results)/n*100:5.1f}%)")
    print(f"  Top-10:  {sum(r['in_top10'] for r in results):3d} ({sum(r['in_top10'] for r in results)/n*100:5.1f}%)")
    print(f"  Top-20:  {sum(r['in_top20'] for r in results):3d} ({sum(r['in_top20'] for r in results)/n*100:5.1f}%)")
    print(f"  Top-30:  {sum(r['in_top30'] for r in results):3d} ({sum(r['in_top30'] for r in results)/n*100:5.1f}%)")
    print(f"  Top-50:  {sum(r['in_top50'] for r in results):3d} ({sum(r['in_top50'] for r in results)/n*100:5.1f}%)")
    print(f"  Top-100: {sum(r['in_top100'] for r in results):3d} ({sum(r['in_top100'] for r in results)/n*100:5.1f}%)")
    print(f"  Top-150: {sum(r['in_top150'] for r in results):3d} ({sum(r['in_top150'] for r in results)/n*100:5.1f}%)")
    print(f"  Top-200: {sum(r['in_top200'] for r in results):3d} ({sum(r['in_top200'] for r in results)/n*100:5.1f}%)")
    print(f"  평균 순위: {sum(r['actual_rank'] for r in results)/n:.1f}")

    return results


if __name__ == "__main__":
    run_backtest_v2(ensemble_weight=0.3)
