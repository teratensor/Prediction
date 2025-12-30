"""
ord2 예측 ML 모델 백테스트 v4

v3 대비 개선 사항:
1. ord1/ord6 자체 특성 피처 추가 (소수, 홀짝, 끝자리 등)
2. ord1-ord2, ord2-ord6 관계 피처
3. 조건부 확률 피처 (ord1이 소수일 때 ord2도 소수일 확률 등)
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

try:
    import xgb
    HAS_XGBOOST = True
except ImportError:
    try:
        import xgboost as xgb
        HAS_XGBOOST = True
    except ImportError:
        HAS_XGBOOST = False
        print("Warning: XGBoost not installed.")

from features import load_winning_numbers, FeatureExtractor, PRIMES, get_range_index

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# v4 피처 컬럼
V4_FEATURE_COLS = [
    # 기본 피처 (v1)
    'range_idx', 'range_prob', 'pos2_freq', 'is_hot', 'is_cold',
    'sum_contribution', 'expected_remaining_avg', 'segment',
    'is_prime', 'is_consecutive', 'consecutive_prob',
    'relative_position', 'overall_freq', 'in_recent_3',
    # v2/v3 피처
    'distance_from_ord1', 'is_in_optimal_range', 'ord1_relative', 'span_ratio',
    'is_in_10_19', 'candidate_normalized', 'ord1_size_category',
    'num_candidates_norm', 'historical_rank',
    # v4 신규: ord1 특성
    'ord1_is_prime',        # ord1이 소수인지
    'ord1_is_odd',          # ord1이 홀수인지
    'ord1_last_digit',      # ord1 끝자리 (0-9)
    'ord1_decade',          # ord1 십의 자리 (0-4)
    # v4 신규: ord6 특성
    'ord6_is_prime',        # ord6이 소수인지
    'ord6_is_odd',          # ord6이 홀수인지
    'ord6_last_digit',      # ord6 끝자리
    'ord6_decade',          # ord6 십의 자리
    # v4 신규: ord1-candidate 관계
    'same_parity_as_ord1',  # ord1과 홀짝 동일
    'same_decade_as_ord1',  # ord1과 십의 자리 동일
    'both_prime_with_ord1', # ord1, candidate 둘 다 소수
    # v4 신규: 조건부 확률
    'cond_prob_prime_given_ord1_prime',  # ord1이 소수일 때 ord2도 소수일 조건부 확률
    'cond_prob_same_parity',             # ord1과 홀짝 같을 조건부 확률
    # v4 신규: 스팬 기반
    'span_category',        # 스팬 크기 카테고리 (좁음/중간/넓음)
    'position_in_span',     # 스팬 내 위치 (앞/중간/뒤)
]


class CondProbCalculator:
    """조건부 확률 계산기"""

    def __init__(self, train_data: List[Dict]):
        # ord1이 소수일 때 ord2도 소수인 경우
        self.ord1_prime_ord2_prime = 0
        self.ord1_prime_total = 0

        # ord1과 ord2 홀짝 같은 경우
        self.same_parity_count = 0
        self.total_count = len(train_data)

        # ord1 소수별 ord2 분포
        self.ord2_when_ord1_prime = Counter()
        self.ord2_when_ord1_not_prime = Counter()

        for r in train_data:
            ord1, ord2 = r['ord1'], r['ord2']

            # 소수 조건부
            if ord1 in PRIMES:
                self.ord1_prime_total += 1
                if ord2 in PRIMES:
                    self.ord1_prime_ord2_prime += 1
                self.ord2_when_ord1_prime[ord2] += 1
            else:
                self.ord2_when_ord1_not_prime[ord2] += 1

            # 홀짝 조건부
            if ord1 % 2 == ord2 % 2:
                self.same_parity_count += 1

    def prob_ord2_prime_given_ord1_prime(self) -> float:
        if self.ord1_prime_total == 0:
            return 0.5
        return self.ord1_prime_ord2_prime / self.ord1_prime_total

    def prob_same_parity(self) -> float:
        if self.total_count == 0:
            return 0.5
        return self.same_parity_count / self.total_count

    def get_ord2_prob_given_ord1_prime(self, candidate: int, ord1_is_prime: bool) -> float:
        """ord1 소수 여부에 따른 ord2 조건부 확률"""
        if ord1_is_prime:
            total = sum(self.ord2_when_ord1_prime.values())
            if total == 0:
                return 0
            return self.ord2_when_ord1_prime.get(candidate, 0) / total
        else:
            total = sum(self.ord2_when_ord1_not_prime.values())
            if total == 0:
                return 0
            return self.ord2_when_ord1_not_prime.get(candidate, 0) / total


def extract_v4_features(base_features: Dict, ord1: int, ord6: int,
                        range_probs: Dict, ord2_freq_rank: Dict,
                        num_candidates: int, cond_calc: CondProbCalculator) -> Dict:
    """v4 피처 추출 (ord1/ord6 특성 포함)"""
    features = base_features.copy()
    candidate = features.get('candidate', 0)
    span = ord6 - ord1

    # v2/v3 피처
    features['distance_from_ord1'] = candidate - ord1
    features['is_in_optimal_range'] = 1 if 10 <= candidate <= 19 else 0
    features['ord1_relative'] = candidate / max(ord1, 1)
    features['span_ratio'] = (candidate - ord1) / span if span > 0 else 0
    features['is_in_10_19'] = 1 if 10 <= candidate <= 19 else 0
    features['candidate_normalized'] = candidate / 45.0

    if ord1 <= 5:
        features['ord1_size_category'] = 0
    elif ord1 <= 10:
        features['ord1_size_category'] = 1
    else:
        features['ord1_size_category'] = 2

    features['num_candidates_norm'] = num_candidates / 45.0
    features['historical_rank'] = ord2_freq_rank.get(candidate, 23) / 45.0

    # ========== v4 신규 피처 ==========

    # ord1 특성
    features['ord1_is_prime'] = 1 if ord1 in PRIMES else 0
    features['ord1_is_odd'] = ord1 % 2
    features['ord1_last_digit'] = ord1 % 10
    features['ord1_decade'] = ord1 // 10

    # ord6 특성
    features['ord6_is_prime'] = 1 if ord6 in PRIMES else 0
    features['ord6_is_odd'] = ord6 % 2
    features['ord6_last_digit'] = ord6 % 10
    features['ord6_decade'] = ord6 // 10

    # ord1-candidate 관계
    features['same_parity_as_ord1'] = 1 if (ord1 % 2) == (candidate % 2) else 0
    features['same_decade_as_ord1'] = 1 if (ord1 // 10) == (candidate // 10) else 0
    features['both_prime_with_ord1'] = 1 if (ord1 in PRIMES and candidate in PRIMES) else 0

    # 조건부 확률
    features['cond_prob_prime_given_ord1_prime'] = (
        cond_calc.prob_ord2_prime_given_ord1_prime() if ord1 in PRIMES else 0
    )
    features['cond_prob_same_parity'] = cond_calc.prob_same_parity()

    # 스팬 기반
    if span <= 25:
        features['span_category'] = 0  # 좁음
    elif span <= 35:
        features['span_category'] = 1  # 중간
    else:
        features['span_category'] = 2  # 넓음

    rel_pos = (candidate - ord1) / span if span > 0 else 0
    if rel_pos <= 0.33:
        features['position_in_span'] = 0  # 앞쪽
    elif rel_pos <= 0.66:
        features['position_in_span'] = 1  # 중간
    else:
        features['position_in_span'] = 2  # 뒤쪽

    return features


def features_to_array_v4(features: Dict) -> np.ndarray:
    """v4 피처를 numpy 배열로 변환"""
    return np.array([features.get(col, 0) for col in V4_FEATURE_COLS])


class RuleBasedScorerV4:
    """v4 규칙 기반 점수 모델"""

    def __init__(self, train_data: List[Dict]):
        self.ord2_freq = Counter()
        self.ord2_by_ord1_prime = defaultdict(Counter)  # ord1 소수 여부별
        self.ord2_by_ord1_parity = defaultdict(Counter)  # ord1 홀짝별

        for r in train_data:
            ord1, ord2 = r['ord1'], r['ord2']
            self.ord2_freq[ord2] += 1

            # ord1 소수 여부별 ord2 분포
            ord1_prime_key = 'prime' if ord1 in PRIMES else 'not_prime'
            self.ord2_by_ord1_prime[ord1_prime_key][ord2] += 1

            # ord1 홀짝별 ord2 분포
            ord1_parity_key = 'odd' if ord1 % 2 == 1 else 'even'
            self.ord2_by_ord1_parity[ord1_parity_key][ord2] += 1

    def score(self, candidate: int, ord1: int, ord6: int, num_candidates: int) -> float:
        score = 0

        # 1. 전체 빈도
        max_freq = max(self.ord2_freq.values()) if self.ord2_freq else 1
        score += (self.ord2_freq.get(candidate, 0) / max_freq) * 25

        # 2. 범위별 점수
        if 10 <= candidate <= 19:
            score += 25
        elif 1 <= candidate <= 9:
            score += 15
        elif 20 <= candidate <= 29:
            score += 10
        else:
            score += 5

        # 3. ord1 소수 여부 기반 조건부 점수
        ord1_prime_key = 'prime' if ord1 in PRIMES else 'not_prime'
        cond_freq = self.ord2_by_ord1_prime[ord1_prime_key]
        if cond_freq:
            max_cond = max(cond_freq.values())
            score += (cond_freq.get(candidate, 0) / max_cond) * 15

        # 4. ord1 홀짝 기반 조건부 점수
        ord1_parity_key = 'odd' if ord1 % 2 == 1 else 'even'
        parity_freq = self.ord2_by_ord1_parity[ord1_parity_key]
        if parity_freq:
            max_parity = max(parity_freq.values())
            score += (parity_freq.get(candidate, 0) / max_parity) * 10

        # 5. 소수 보너스
        if candidate in PRIMES:
            score += 8
            # ord1도 소수면 추가 보너스
            if ord1 in PRIMES:
                score += 5

        # 6. 홀짝 일치 보너스
        if ord1 % 2 == candidate % 2:
            score += 5

        return score


def run_backtest_v4(min_train_size: int = 50, ensemble_weight: float = 0.3):
    """v4 백테스트 실행"""
    print("=" * 60)
    print("ord2 예측 ML 모델 백테스트 v4 (ord1/ord6 특성 추가)")
    print("=" * 60)
    print(f"앙상블 가중치: XGBoost {1-ensemble_weight:.0%} + Rule {ensemble_weight:.0%}")
    print(f"신규 피처: ord1/ord6 소수, 홀짝, 조건부 확률 등")

    data = load_winning_numbers()
    print(f"데이터: {len(data)}회차 ({data[0]['round']}~{data[-1]['round']})")

    use_xgb = HAS_XGBOOST
    if not use_xgb:
        print("XGBoost 미설치 → Rule 기반만 사용")
        ensemble_weight = 1.0

    results = []
    feature_importance_sum = defaultdict(float)
    feature_importance_count = 0

    for i in range(min_train_size, len(data)):
        train_data = data[:i]
        target = data[i]

        if (i - min_train_size) % 50 == 0:
            print(f"  진행: {i - min_train_size + 1}/{len(data) - min_train_size} "
                  f"(회차 {target['round']})")

        # 통계 계산
        range_counts = Counter()
        ord2_freq = Counter()
        for r in train_data:
            range_counts[get_range_index(r['ord2'])] += 1
            ord2_freq[r['ord2']] += 1

        total_count = sum(range_counts.values())
        range_probs = {k: v/total_count for k, v in range_counts.items()}

        sorted_ord2 = sorted(ord2_freq.keys(), key=lambda x: -ord2_freq[x])
        ord2_freq_rank = {num: rank+1 for rank, num in enumerate(sorted_ord2)}

        # 조건부 확률 계산기
        cond_calc = CondProbCalculator(train_data)

        # 규칙 기반 스코어러
        rule_scorer = RuleBasedScorerV4(train_data)

        # 학습 데이터 준비
        X_train = []
        y_train = []

        train_window = train_data[-100:] if len(train_data) > 100 else train_data

        for j, tr in enumerate(train_window):
            if j < 10:
                continue
            prev_data = train_window[:j]
            extractor = FeatureExtractor(prev_data)
            prev_cond_calc = CondProbCalculator(prev_data)

            candidates = extractor.get_all_candidates(tr['ord1'], tr['ord6'])
            num_cand = len(candidates)

            for cand in candidates:
                v4_feat = extract_v4_features(
                    cand, tr['ord1'], tr['ord6'],
                    range_probs, ord2_freq_rank, num_cand, prev_cond_calc
                )
                X_train.append(features_to_array_v4(v4_feat))
                y_train.append(1 if cand['candidate'] == tr['ord2'] else 0)

        if len(X_train) < 10:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # XGBoost 학습
        if use_xgb:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.08,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            model.fit(X_train, y_train)

            for idx, importance in enumerate(model.feature_importances_):
                if idx < len(V4_FEATURE_COLS):
                    feature_importance_sum[V4_FEATURE_COLS[idx]] += importance
            feature_importance_count += 1

        # 예측
        extractor = FeatureExtractor(train_data)
        candidates = extractor.get_all_candidates(target['ord1'], target['ord6'])

        if not candidates:
            continue

        num_candidates = len(candidates)

        test_features_list = []
        rule_scores = []

        for c in candidates:
            v4_feat = extract_v4_features(
                c, target['ord1'], target['ord6'],
                range_probs, ord2_freq_rank, num_candidates, cond_calc
            )
            test_features_list.append(v4_feat)
            rule_score = rule_scorer.score(c['candidate'], target['ord1'], target['ord6'], num_candidates)
            rule_scores.append(rule_score)

        X_test = np.array([features_to_array_v4(f) for f in test_features_list])

        if use_xgb:
            xgb_probs = model.predict_proba(X_test)[:, 1]
        else:
            xgb_probs = np.zeros(len(candidates))

        rule_scores = np.array(rule_scores)
        if rule_scores.max() > rule_scores.min():
            rule_probs = (rule_scores - rule_scores.min()) / (rule_scores.max() - rule_scores.min())
        else:
            rule_probs = np.ones_like(rule_scores) / len(rule_scores)

        final_probs = (1 - ensemble_weight) * xgb_probs + ensemble_weight * rule_probs

        sorted_indices = np.argsort(-final_probs)
        sorted_candidates = [candidates[i]['candidate'] for i in sorted_indices]

        actual_ord2 = target['ord2']
        predicted_ord2 = sorted_candidates[0] if sorted_candidates else -1

        top3 = sorted_candidates[:3]
        top5 = sorted_candidates[:5]
        top10 = sorted_candidates[:10]

        in_top3 = 1 if actual_ord2 in top3 else 0
        in_top5 = 1 if actual_ord2 in top5 else 0
        in_top10 = 1 if actual_ord2 in top10 else 0
        exact_match = 1 if predicted_ord2 == actual_ord2 else 0

        try:
            actual_rank = sorted_candidates.index(actual_ord2) + 1
        except ValueError:
            actual_rank = len(sorted_candidates) + 1

        results.append({
            'round': target['round'],
            'ord1': target['ord1'],
            'ord6': target['ord6'],
            'actual_ord2': actual_ord2,
            'predicted_ord2': predicted_ord2,
            'actual_rank': actual_rank,
            'num_candidates': num_candidates,
            'top3': str(top3),
            'top5': str(top5),
            'exact_match': exact_match,
            'in_top3': in_top3,
            'in_top5': in_top5,
            'in_top10': in_top10,
            'error': abs(predicted_ord2 - actual_ord2)
        })

    # 결과 저장
    output_path = OUTPUT_DIR / "backtest_results_v4.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[저장] {output_path}")

    if use_xgb and feature_importance_count > 0:
        importance_path = OUTPUT_DIR / "feature_importance_v4.csv"
        with open(importance_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['feature', 'importance'])
            avg_importance = {k: v / feature_importance_count
                            for k, v in feature_importance_sum.items()}
            for feat, imp in sorted(avg_importance.items(), key=lambda x: -x[1]):
                writer.writerow([feat, f"{imp:.6f}"])
        print(f"[저장] {importance_path}")

    print_statistics_v4(results)
    return results


def print_statistics_v4(results: List[Dict]):
    """v4 통계 출력"""
    print("\n" + "=" * 60)
    print("백테스트 v4 결과 통계")
    print("=" * 60)

    n = len(results)
    exact = sum(r['exact_match'] for r in results)
    top3 = sum(r['in_top3'] for r in results)
    top5 = sum(r['in_top5'] for r in results)
    top10 = sum(r['in_top10'] for r in results)
    avg_error = sum(r['error'] for r in results) / n
    avg_rank = sum(r['actual_rank'] for r in results) / n

    print(f"\n테스트 회차: {n}회")

    print(f"\n[적중률]")
    print(f"  정확도 (Top-1): {exact:4d}회 ({exact/n*100:5.1f}%)")
    print(f"  Top-3 적중:     {top3:4d}회 ({top3/n*100:5.1f}%)")
    print(f"  Top-5 적중:     {top5:4d}회 ({top5/n*100:5.1f}%)")
    print(f"  Top-10 적중:    {top10:4d}회 ({top10/n*100:5.1f}%)")

    print(f"\n[오차]")
    print(f"  평균 오차: {avg_error:.2f}")
    print(f"  평균 순위: {avg_rank:.2f}")

    print(f"\n[범위별 적중률]")
    range_stats = defaultdict(lambda: {'total': 0, 'top5': 0, 'top10': 0})
    for r in results:
        range_idx = r['actual_ord2'] // 10
        range_stats[range_idx]['total'] += 1
        range_stats[range_idx]['top5'] += r['in_top5']
        range_stats[range_idx]['top10'] += r['in_top10']

    for range_idx in sorted(range_stats.keys()):
        stats = range_stats[range_idx]
        total = stats['total']
        top5_pct = stats['top5'] / total * 100 if total > 0 else 0
        top10_pct = stats['top10'] / total * 100 if total > 0 else 0
        range_name = f"{range_idx*10:02d}-{range_idx*10+9:02d}"
        print(f"  {range_name}: {total:3d}회, Top-5 {top5_pct:5.1f}%, Top-10 {top10_pct:5.1f}%")


if __name__ == "__main__":
    run_backtest_v4(min_train_size=50, ensemble_weight=0.3)
