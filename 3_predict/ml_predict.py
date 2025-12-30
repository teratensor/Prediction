"""
통합 ML 예측 스크립트

ord2, ord3, ord4, ord5 ML 모델을 사용하여 전체 조합 생성
"""

import csv
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
from itertools import product

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed")

# 경로 설정
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH.parent / "1_data" / "winning_numbers.csv"
RESULT_PATH = BASE_PATH.parent / "result"
RESULT_PATH.mkdir(exist_ok=True)

# 공통 상수
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
RANGES = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 45)]

def get_range_index(num: int) -> int:
    for i, (start, end) in enumerate(RANGES):
        if start <= num <= end:
            return i
    return 4

def load_winning_numbers() -> List[Dict]:
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'round': int(row['round']),
                'ord1': int(row['ord1']),
                'ord2': int(row['ord2']),
                'ord3': int(row['ord3']),
                'ord4': int(row['ord4']),
                'ord5': int(row['ord5']),
                'ord6': int(row['ord6']),
            })
    return sorted(results, key=lambda x: x['round'])


class MLPredictor:
    """통합 ML 예측기"""

    def __init__(self, train_data: List[Dict]):
        self.train_data = train_data
        self._compute_statistics()
        self._train_models()

    def _compute_statistics(self):
        """통계 계산"""
        self.ball_freq = Counter()
        self.pos_freq = {i: Counter() for i in range(1, 7)}

        for r in self.train_data:
            for i in range(1, 7):
                ball = r[f'ord{i}']
                self.ball_freq[ball] += 1
                self.pos_freq[i][ball] += 1

        # HOT/COLD
        recent = self.train_data[-10:]
        freq = Counter()
        for r in recent:
            for i in range(1, 7):
                freq[r[f'ord{i}']] += 1
        sorted_nums = sorted(range(1, 46), key=lambda x: -freq.get(x, 0))
        self.hot_bits = set(sorted_nums[:10])
        self.cold_bits = set(sorted_nums[-10:])

        # 범위별 빈도
        self.range_freq = {i: Counter() for i in range(2, 6)}
        for r in self.train_data:
            for i in range(2, 6):
                self.range_freq[i][get_range_index(r[f'ord{i}'])] += 1

    def _get_common_features(self, candidate: int, pos: int) -> Dict:
        """공통 피처 추출"""
        features = {}
        features['candidate'] = candidate
        features['range_idx'] = get_range_index(candidate)

        total_pos = sum(self.pos_freq[pos].values())
        features[f'pos{pos}_freq'] = self.pos_freq[pos].get(candidate, 0) / total_pos if total_pos > 0 else 0

        features['is_hot'] = 1 if candidate in self.hot_bits else 0
        features['is_cold'] = 1 if candidate in self.cold_bits else 0
        features['is_prime'] = 1 if candidate in PRIMES else 0

        total_freq = sum(self.ball_freq.values())
        features['overall_freq'] = self.ball_freq.get(candidate, 0) / total_freq if total_freq > 0 else 0

        recent_3 = self.train_data[-3:]
        recent_balls = set()
        for r in recent_3:
            for i in range(1, 7):
                recent_balls.add(r[f'ord{i}'])
        features['in_recent_3'] = 1 if candidate in recent_balls else 0

        return features

    def _train_models(self):
        """각 포지션별 모델 학습"""
        if not HAS_XGBOOST:
            self.models = {}
            return

        self.models = {}
        train_window = self.train_data[-100:] if len(self.train_data) > 100 else self.train_data

        # ord2 모델
        X2, y2 = [], []
        for j, tr in enumerate(train_window):
            if j < 10:
                continue
            for cand in range(tr['ord1'] + 1, tr['ord6']):
                feat = self._extract_ord2_features(tr['ord1'], tr['ord6'], cand)
                X2.append(list(feat.values()))
                y2.append(1 if cand == tr['ord2'] else 0)

        if len(X2) > 10:
            self.models['ord2'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.08,
                random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'
            )
            self.models['ord2'].fit(np.array(X2), np.array(y2))

        # ord4 모델 (ord1, ord6만 사용)
        X4, y4 = [], []
        for j, tr in enumerate(train_window):
            if j < 10:
                continue
            formula = round(tr['ord1'] + (tr['ord6'] - tr['ord1']) * 0.60)
            for cand in range(max(tr['ord1'] + 3, formula - 10), min(tr['ord6'] - 2, formula + 11)):
                feat = self._extract_ord4_features(tr['ord1'], tr['ord6'], cand)
                X4.append(list(feat.values()))
                y4.append(1 if cand == tr['ord4'] else 0)

        if len(X4) > 10:
            self.models['ord4'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.08,
                random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'
            )
            self.models['ord4'].fit(np.array(X4), np.array(y4))

        # ord3 모델 (ord1, ord2, ord6 사용)
        X3, y3 = [], []
        for j, tr in enumerate(train_window):
            if j < 10:
                continue
            for cand in range(tr['ord2'] + 1, tr['ord6']):
                feat = self._extract_ord3_features(tr['ord1'], tr['ord2'], tr['ord6'], cand)
                X3.append(list(feat.values()))
                y3.append(1 if cand == tr['ord3'] else 0)

        if len(X3) > 10:
            self.models['ord3'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.08,
                random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'
            )
            self.models['ord3'].fit(np.array(X3), np.array(y3))

        # ord5 모델 (ord1, ord4, ord6 사용)
        X5, y5 = [], []
        for j, tr in enumerate(train_window):
            if j < 10:
                continue
            for cand in range(tr['ord4'] + 1, tr['ord6']):
                feat = self._extract_ord5_features(tr['ord1'], tr['ord4'], tr['ord6'], cand)
                X5.append(list(feat.values()))
                y5.append(1 if cand == tr['ord5'] else 0)

        if len(X5) > 10:
            self.models['ord5'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.08,
                random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'
            )
            self.models['ord5'].fit(np.array(X5), np.array(y5))

    def _extract_ord2_features(self, ord1: int, ord6: int, candidate: int) -> Dict:
        feat = self._get_common_features(candidate, 2)
        feat['distance_from_ord1'] = candidate - ord1
        feat['is_in_optimal_range'] = 1 if 10 <= candidate <= 19 else 0
        feat['ord1_is_prime'] = 1 if ord1 in PRIMES else 0
        feat['ord6_is_prime'] = 1 if ord6 in PRIMES else 0
        span = ord6 - ord1
        feat['relative_position'] = (candidate - ord1) / span if span > 0 else 0
        return feat

    def _extract_ord3_features(self, ord1: int, ord2: int, ord6: int, candidate: int) -> Dict:
        feat = self._get_common_features(candidate, 3)
        feat['distance_from_ord2'] = candidate - ord2
        feat['is_in_optimal_range'] = 1 if 15 <= candidate <= 24 else 0
        feat['ord2_is_prime'] = 1 if ord2 in PRIMES else 0
        feat['same_parity_as_ord2'] = 1 if (ord2 % 2) == (candidate % 2) else 0
        span = ord6 - ord2
        feat['relative_position'] = (candidate - ord2) / span if span > 0 else 0
        return feat

    def _extract_ord4_features(self, ord1: int, ord6: int, candidate: int) -> Dict:
        feat = self._get_common_features(candidate, 4)
        formula = round(ord1 + (ord6 - ord1) * 0.60)
        feat['distance_from_formula'] = abs(candidate - formula)
        feat['is_in_formula_range'] = 1 if abs(candidate - formula) <= 5 else 0
        feat['ord1_is_prime'] = 1 if ord1 in PRIMES else 0
        feat['ord6_is_prime'] = 1 if ord6 in PRIMES else 0
        span = ord6 - ord1
        feat['relative_position'] = (candidate - ord1) / span if span > 0 else 0
        return feat

    def _extract_ord5_features(self, ord1: int, ord4: int, ord6: int, candidate: int) -> Dict:
        feat = self._get_common_features(candidate, 5)
        feat['distance_from_ord4'] = candidate - ord4
        feat['distance_to_ord6'] = ord6 - candidate
        feat['is_in_optimal_range'] = 1 if 30 <= candidate <= 39 else 0
        feat['ord4_is_prime'] = 1 if ord4 in PRIMES else 0
        feat['same_parity_as_ord4'] = 1 if (ord4 % 2) == (candidate % 2) else 0
        span = ord6 - ord4
        feat['relative_position'] = (candidate - ord4) / span if span > 0 else 0
        return feat

    def predict_top_k(self, ord1: int, ord6: int, k: int = 5) -> Dict:
        """각 포지션별 Top-K 예측"""
        results = {}

        # ord2 예측
        ord2_candidates = list(range(ord1 + 1, ord6))
        if 'ord2' in self.models and ord2_candidates:
            X = np.array([list(self._extract_ord2_features(ord1, ord6, c).values()) for c in ord2_candidates])
            probs = self.models['ord2'].predict_proba(X)[:, 1]
            sorted_idx = np.argsort(-probs)
            results['ord2'] = [(ord2_candidates[i], probs[i]) for i in sorted_idx[:k]]
        else:
            results['ord2'] = [(c, 1.0/len(ord2_candidates)) for c in ord2_candidates[:k]]

        # ord4 예측
        formula = round(ord1 + (ord6 - ord1) * 0.60)
        ord4_candidates = list(range(max(ord1 + 3, formula - 10), min(ord6 - 2, formula + 11)))
        if 'ord4' in self.models and ord4_candidates:
            X = np.array([list(self._extract_ord4_features(ord1, ord6, c).values()) for c in ord4_candidates])
            probs = self.models['ord4'].predict_proba(X)[:, 1]
            sorted_idx = np.argsort(-probs)
            results['ord4'] = [(ord4_candidates[i], probs[i]) for i in sorted_idx[:k]]
        else:
            results['ord4'] = [(c, 1.0/len(ord4_candidates)) for c in ord4_candidates[:k]]

        return results

    def predict_ord3(self, ord1: int, ord2: int, ord6: int, k: int = 5) -> List[Tuple[int, float]]:
        """ord3 예측"""
        candidates = list(range(ord2 + 1, ord6))
        if 'ord3' in self.models and candidates:
            X = np.array([list(self._extract_ord3_features(ord1, ord2, ord6, c).values()) for c in candidates])
            probs = self.models['ord3'].predict_proba(X)[:, 1]
            sorted_idx = np.argsort(-probs)
            return [(candidates[i], probs[i]) for i in sorted_idx[:k]]
        return [(c, 1.0/len(candidates)) for c in candidates[:k]]

    def predict_ord5(self, ord1: int, ord4: int, ord6: int, k: int = 5) -> List[Tuple[int, float]]:
        """ord5 예측"""
        candidates = list(range(ord4 + 1, ord6))
        if 'ord5' in self.models and candidates:
            X = np.array([list(self._extract_ord5_features(ord1, ord4, ord6, c).values()) for c in candidates])
            probs = self.models['ord5'].predict_proba(X)[:, 1]
            sorted_idx = np.argsort(-probs)
            return [(candidates[i], probs[i]) for i in sorted_idx[:k]]
        return [(c, 1.0/len(candidates)) for c in candidates[:k]]


def generate_combinations(predictor: MLPredictor, ord1: int, ord6: int,
                          top_k: int = 5) -> List[Tuple[int, ...]]:
    """전체 조합 생성"""

    # 1단계: ord2, ord4 예측
    top_preds = predictor.predict_top_k(ord1, ord6, k=top_k)
    ord2_list = [x[0] for x in top_preds['ord2']]
    ord4_list = [x[0] for x in top_preds['ord4']]

    combinations = []

    for ord2 in ord2_list:
        for ord4 in ord4_list:
            if ord4 <= ord2:
                continue

            # ord3 예측 (ord2 < ord3 < ord4)
            ord3_candidates = predictor.predict_ord3(ord1, ord2, ord6, k=top_k)
            ord3_list = [x[0] for x in ord3_candidates if ord2 < x[0] < ord4][:top_k]

            # ord5 예측 (ord4 < ord5 < ord6)
            ord5_candidates = predictor.predict_ord5(ord1, ord4, ord6, k=top_k)
            ord5_list = [x[0] for x in ord5_candidates][:top_k]

            for ord3 in ord3_list:
                for ord5 in ord5_list:
                    if ord1 < ord2 < ord3 < ord4 < ord5 < ord6:
                        combinations.append((ord1, ord2, ord3, ord4, ord5, ord6))

    return combinations


def predict_round(target_round: int, top_k: int = 5):
    """특정 회차 예측"""
    print("=" * 60)
    print(f"통합 ML 예측: {target_round}회차")
    print("=" * 60)

    data = load_winning_numbers()

    # 타겟 회차 이전 데이터로 학습
    train_data = [r for r in data if r['round'] < target_round]
    target = next((r for r in data if r['round'] == target_round), None)

    print(f"학습 데이터: {len(train_data)}회차")

    if target:
        actual = (target['ord1'], target['ord2'], target['ord3'],
                  target['ord4'], target['ord5'], target['ord6'])
        print(f"실제 당첨번호: {actual}")

    # 모델 학습
    print("\n모델 학습 중...")
    predictor = MLPredictor(train_data)

    # 모든 (ord1, ord6) 쌍에 대해 조합 생성
    print(f"\n조합 생성 중 (top_k={top_k})...")

    all_combinations = []
    ord1_range = range(1, 11)  # ord1: 1~10
    ord6_range = range(35, 46)  # ord6: 35~45

    for ord1 in ord1_range:
        for ord6 in ord6_range:
            if ord6 - ord1 < 5:
                continue
            combs = generate_combinations(predictor, ord1, ord6, top_k=top_k)
            all_combinations.extend(combs)

    print(f"생성된 조합 수: {len(all_combinations):,}개")

    # 결과 저장
    output_path = RESULT_PATH / "ml_result.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6'])
        for comb in all_combinations:
            writer.writerow(comb)

    print(f"\n[저장] {output_path}")

    # 실제 당첨번호와 비교
    if target:
        print("\n" + "=" * 60)
        print("당첨번호 비교")
        print("=" * 60)

        actual_set = set(actual)
        match_counts = Counter()
        best_match = 0
        best_comb = None

        for comb in all_combinations:
            match = len(set(comb) & actual_set)
            match_counts[match] += 1
            if match > best_match:
                best_match = match
                best_comb = comb

        print(f"\n실제 당첨: {actual}")
        print(f"최고 적중: {best_match}개 일치")
        if best_comb:
            print(f"최고 조합: {best_comb}")

        print(f"\n[매치 분포]")
        for i in range(7):
            cnt = match_counts.get(i, 0)
            pct = cnt / len(all_combinations) * 100 if all_combinations else 0
            print(f"  {i}개 일치: {cnt:,}개 ({pct:.2f}%)")

        # 6개 일치 여부
        has_6 = match_counts.get(6, 0) > 0
        print(f"\n🎯 6개 일치 조합: {'있음 ✓' if has_6 else '없음 ✗'}")

    return all_combinations


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target_round = int(sys.argv[1])
    else:
        target_round = 1204

    if len(sys.argv) > 2:
        top_k = int(sys.argv[2])
    else:
        top_k = 5

    predict_round(target_round, top_k)
