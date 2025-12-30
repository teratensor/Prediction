"""
ord2 예측 ML 모델 백테스트 v3 (추가 개선)

v2 대비 개선 사항:
1. 10-19 범위 예측 강화 (v2에서 하락한 부분)
2. 후보 수별 적응적 가중치
3. ord1 크기별 보정
4. 앙상블: XGBoost + 규칙 기반 점수 결합
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
    print("Warning: XGBoost not installed.")

from features import load_winning_numbers, FeatureExtractor, PRIMES, get_range_index

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# v3 피처 컬럼 (v2 + 추가)
V3_FEATURE_COLS = [
    # 기본 피처
    'range_idx', 'range_prob', 'pos2_freq', 'is_hot', 'is_cold',
    'sum_contribution', 'expected_remaining_avg', 'segment',
    'is_prime', 'is_consecutive', 'consecutive_prob',
    'relative_position', 'overall_freq', 'in_recent_3',
    # v2 추가 피처
    'distance_from_ord1', 'is_in_optimal_range', 'ord1_relative', 'span_ratio',
    # v3 추가 피처
    'is_in_10_19',          # 10-19 범위 여부
    'candidate_normalized', # 정규화된 후보 값
    'ord1_size_category',   # ord1 크기 카테고리 (0=작음, 1=중간, 2=큼)
    'num_candidates_norm',  # 정규화된 후보 수
    'historical_rank',      # 과거 ord2 빈도 순위
]


def extract_v3_features(base_features: Dict, ord1: int, ord6: int,
                        range_probs: Dict, ord2_freq_rank: Dict,
                        num_candidates: int) -> Dict:
    """v3 피처 추출"""
    features = base_features.copy()
    candidate = features.get('candidate', 0)

    # v2 피처
    features['distance_from_ord1'] = candidate - ord1
    features['is_in_optimal_range'] = 1 if 10 <= candidate <= 19 else 0
    features['ord1_relative'] = candidate / max(ord1, 1)
    span = ord6 - ord1
    features['span_ratio'] = (candidate - ord1) / span if span > 0 else 0

    # v3 추가 피처
    features['is_in_10_19'] = 1 if 10 <= candidate <= 19 else 0
    features['candidate_normalized'] = candidate / 45.0

    # ord1 크기 카테고리
    if ord1 <= 5:
        features['ord1_size_category'] = 0  # 작음
    elif ord1 <= 10:
        features['ord1_size_category'] = 1  # 중간
    else:
        features['ord1_size_category'] = 2  # 큼

    # 후보 수 정규화
    features['num_candidates_norm'] = num_candidates / 45.0

    # 과거 ord2 빈도 순위 (1~45, 낮을수록 자주 나옴)
    features['historical_rank'] = ord2_freq_rank.get(candidate, 23) / 45.0

    return features


def features_to_array_v3(features: Dict) -> np.ndarray:
    """v3 피처를 numpy 배열로 변환"""
    return np.array([features.get(col, 0) for col in V3_FEATURE_COLS])


class RuleBasedScorer:
    """규칙 기반 점수 모델 (앙상블용)"""

    def __init__(self, train_data: List[Dict]):
        """학습 데이터 기반 통계 계산"""
        self.ord2_freq = Counter()
        self.ord2_by_ord1_range = defaultdict(Counter)

        for r in train_data:
            self.ord2_freq[r['ord2']] += 1
            # ord1 크기별 ord2 분포
            if r['ord1'] <= 5:
                self.ord2_by_ord1_range[0][r['ord2']] += 1
            elif r['ord1'] <= 10:
                self.ord2_by_ord1_range[1][r['ord2']] += 1
            else:
                self.ord2_by_ord1_range[2][r['ord2']] += 1

    def score(self, candidate: int, ord1: int, num_candidates: int) -> float:
        """규칙 기반 점수 계산"""
        score = 0

        # 1. 전체 빈도 점수
        max_freq = max(self.ord2_freq.values()) if self.ord2_freq else 1
        score += (self.ord2_freq.get(candidate, 0) / max_freq) * 30

        # 2. 범위별 점수 (10-19 강화)
        if 10 <= candidate <= 19:
            score += 25  # 최빈 범위 보너스
        elif 1 <= candidate <= 9:
            score += 15
        elif 20 <= candidate <= 29:
            score += 10
        else:
            score += 5

        # 3. ord1 기반 조건부 점수
        ord1_cat = 0 if ord1 <= 5 else (1 if ord1 <= 10 else 2)
        cat_freq = self.ord2_by_ord1_range[ord1_cat]
        if cat_freq:
            max_cat_freq = max(cat_freq.values())
            score += (cat_freq.get(candidate, 0) / max_cat_freq) * 20

        # 4. 소수 보너스
        if candidate in PRIMES:
            score += 8

        # 5. 후보 수가 적으면 상위에 더 집중
        if num_candidates <= 20:
            if 10 <= candidate <= 19:
                score += 10

        return score


def run_backtest_v3(min_train_size: int = 50, ensemble_weight: float = 0.3):
    """
    v3 백테스트 실행 (XGBoost + Rule 앙상블)

    Args:
        min_train_size: 최소 학습 데이터 크기
        ensemble_weight: 규칙 기반 점수 가중치 (0~1)
    """
    print("=" * 60)
    print("ord2 예측 ML 모델 백테스트 v3 (앙상블)")
    print("=" * 60)
    print(f"앙상블 가중치: XGBoost {1-ensemble_weight:.0%} + Rule {ensemble_weight:.0%}")

    data = load_winning_numbers()
    print(f"데이터: {len(data)}회차 ({data[0]['round']}~{data[-1]['round']})")

    use_xgb = HAS_XGBOOST
    if not use_xgb:
        print("XGBoost 미설치 → Rule 기반만 사용")
        ensemble_weight = 1.0

    results = []
    feature_importance_sum = defaultdict(float)
    feature_importance_count = 0

    # Rolling window 백테스트
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

        # ord2 빈도 순위
        sorted_ord2 = sorted(ord2_freq.keys(), key=lambda x: -ord2_freq[x])
        ord2_freq_rank = {num: rank+1 for rank, num in enumerate(sorted_ord2)}

        # 규칙 기반 스코어러
        rule_scorer = RuleBasedScorer(train_data)

        # 학습 데이터 준비
        X_train = []
        y_train = []

        train_window = train_data[-100:] if len(train_data) > 100 else train_data

        for j, tr in enumerate(train_window):
            if j < 10:
                continue
            prev_data = train_window[:j]
            extractor = FeatureExtractor(prev_data)

            candidates = extractor.get_all_candidates(tr['ord1'], tr['ord6'])
            num_cand = len(candidates)

            for cand in candidates:
                v3_feat = extract_v3_features(
                    cand, tr['ord1'], tr['ord6'],
                    range_probs, ord2_freq_rank, num_cand
                )
                X_train.append(features_to_array_v3(v3_feat))
                y_train.append(1 if cand['candidate'] == tr['ord2'] else 0)

        if len(X_train) < 10:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # XGBoost 모델 학습
        if use_xgb:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,      # L1 정규화
                reg_lambda=1.0,     # L2 정규화
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            model.fit(X_train, y_train)

            for idx, importance in enumerate(model.feature_importances_):
                if idx < len(V3_FEATURE_COLS):
                    feature_importance_sum[V3_FEATURE_COLS[idx]] += importance
            feature_importance_count += 1

        # 예측
        extractor = FeatureExtractor(train_data)
        candidates = extractor.get_all_candidates(target['ord1'], target['ord6'])

        if not candidates:
            continue

        num_candidates = len(candidates)

        # 피처 추출 및 점수 계산
        test_features_list = []
        rule_scores = []

        for c in candidates:
            v3_feat = extract_v3_features(
                c, target['ord1'], target['ord6'],
                range_probs, ord2_freq_rank, num_candidates
            )
            test_features_list.append(v3_feat)

            # 규칙 기반 점수
            rule_score = rule_scorer.score(c['candidate'], target['ord1'], num_candidates)
            rule_scores.append(rule_score)

        X_test = np.array([features_to_array_v3(f) for f in test_features_list])

        # 앙상블 점수 계산
        if use_xgb:
            xgb_probs = model.predict_proba(X_test)[:, 1]
        else:
            xgb_probs = np.zeros(len(candidates))

        # 규칙 점수 정규화
        rule_scores = np.array(rule_scores)
        if rule_scores.max() > rule_scores.min():
            rule_probs = (rule_scores - rule_scores.min()) / (rule_scores.max() - rule_scores.min())
        else:
            rule_probs = np.ones_like(rule_scores) / len(rule_scores)

        # 앙상블 결합
        final_probs = (1 - ensemble_weight) * xgb_probs + ensemble_weight * rule_probs

        # 정렬
        sorted_indices = np.argsort(-final_probs)
        sorted_candidates = [candidates[i]['candidate'] for i in sorted_indices]

        # 결과 저장
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
    output_path = OUTPUT_DIR / "backtest_results_v3.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[저장] {output_path}")

    if use_xgb and feature_importance_count > 0:
        importance_path = OUTPUT_DIR / "feature_importance_v3.csv"
        with open(importance_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['feature', 'importance'])
            avg_importance = {k: v / feature_importance_count
                            for k, v in feature_importance_sum.items()}
            for feat, imp in sorted(avg_importance.items(), key=lambda x: -x[1]):
                writer.writerow([feat, f"{imp:.6f}"])
        print(f"[저장] {importance_path}")

    print_statistics_v3(results)
    return results


def print_statistics_v3(results: List[Dict]):
    """v3 통계 출력"""
    print("\n" + "=" * 60)
    print("백테스트 v3 결과 통계")
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
    range_stats = defaultdict(lambda: {'total': 0, 'exact': 0, 'top5': 0, 'top10': 0})
    for r in results:
        range_idx = r['actual_ord2'] // 10
        range_stats[range_idx]['total'] += 1
        range_stats[range_idx]['exact'] += r['exact_match']
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
    # 앙상블 가중치 테스트
    run_backtest_v3(min_train_size=50, ensemble_weight=0.3)
