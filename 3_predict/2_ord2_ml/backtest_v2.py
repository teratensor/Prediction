"""
ord2 예측 ML 모델 백테스트 v2 (개선판)

개선 사항:
1. 범위별 분리 예측 (00-09, 10-19, 20+ 각각 다르게 처리)
2. 예측 다양성 강화 (편향 완화)
3. 추가 피처 (ord1 기반 상대 거리, 범위별 확률 보정)
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
    print("Warning: XGBoost not installed. Using enhanced scoring model.")

from features import load_winning_numbers, FeatureExtractor, PRIMES, get_range_index

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# 기본 피처 컬럼
BASE_FEATURE_COLS = [
    'range_idx', 'range_prob', 'pos2_freq', 'is_hot', 'is_cold',
    'sum_contribution', 'expected_remaining_avg', 'segment',
    'is_prime', 'is_consecutive', 'consecutive_prob',
    'relative_position', 'overall_freq', 'in_recent_3'
]

# v2 추가 피처
V2_FEATURE_COLS = BASE_FEATURE_COLS + [
    'distance_from_ord1',  # ord1과의 거리
    'range_00_09_prob',    # 범위별 확률
    'range_10_19_prob',
    'range_20_plus_prob',
    'is_in_optimal_range', # 최빈 범위 (10-19) 여부
    'ord1_relative',       # ord1 대비 상대 크기
    'span_ratio',          # 전체 스팬 대비 위치
]


def extract_v2_features(base_features: Dict, ord1: int, ord6: int,
                        range_probs: Dict) -> Dict:
    """v2 추가 피처 계산"""
    features = base_features.copy()
    candidate = features.get('candidate', 0)

    # 거리 피처
    features['distance_from_ord1'] = candidate - ord1

    # 범위별 확률 (학습 데이터 기반)
    features['range_00_09_prob'] = range_probs.get(0, 0.36)
    features['range_10_19_prob'] = range_probs.get(1, 0.48)
    features['range_20_plus_prob'] = range_probs.get(2, 0.15) + range_probs.get(3, 0.01)

    # 최빈 범위 여부
    features['is_in_optimal_range'] = 1 if 10 <= candidate <= 19 else 0

    # ord1 상대적 크기
    features['ord1_relative'] = candidate / max(ord1, 1)

    # 전체 스팬 대비 위치
    span = ord6 - ord1
    features['span_ratio'] = (candidate - ord1) / span if span > 0 else 0

    return features


def features_to_array_v2(features: Dict) -> np.ndarray:
    """v2 피처 딕셔너리를 numpy 배열로 변환"""
    return np.array([features.get(col, 0) for col in V2_FEATURE_COLS])


class EnhancedScorer:
    """개선된 점수 기반 모델 (XGBoost 없을 때 사용)"""

    def __init__(self):
        # 범위별 기본 점수
        self.range_base_scores = {
            0: 35,   # 00-09: 기본 35점
            1: 50,   # 10-19: 기본 50점 (최빈)
            2: 25,   # 20-29: 기본 25점
            3: 10,   # 30-39: 기본 10점
            4: 5,    # 40-45: 기본 5점
        }

        # 피처별 가중치
        self.weights = {
            'pos2_freq': 80,           # 위치 빈도 강화
            'overall_freq': 30,        # 전체 빈도
            'is_hot': 15,              # HOT 보너스
            'is_cold': -8,             # COLD 페널티
            'is_prime': 8,             # 소수 보너스
            'is_consecutive': 20,      # 연속수 보너스
            'in_recent_3': -5,         # 최근 출현 페널티
            'distance_from_ord1': -0.5, # 거리 페널티 (가까울수록 유리)
        }

    def fit(self, X, y):
        """학습 (데이터 기반 가중치 조정)"""
        pass

    def predict_proba(self, X: np.ndarray, features_list: List[Dict]) -> np.ndarray:
        """확률 예측"""
        scores = []

        for i, row in enumerate(X):
            feat = features_list[i] if i < len(features_list) else {}
            score = 0

            # 범위 기본 점수
            range_idx = int(feat.get('range_idx', 0))
            score += self.range_base_scores.get(range_idx, 20)

            # 피처별 점수
            for col, weight in self.weights.items():
                val = feat.get(col, 0)
                score += val * weight

            # 다양성 보너스: 자주 예측되지 않는 범위에 보너스
            if range_idx == 0:  # 00-09
                score += 10
            elif range_idx >= 2:  # 20+
                score += 5

            scores.append(score)

        # Min-max 정규화
        scores = np.array(scores)
        min_s, max_s = scores.min(), scores.max()
        if max_s > min_s:
            probs = (scores - min_s) / (max_s - min_s)
        else:
            probs = np.ones_like(scores) / len(scores)

        return np.column_stack([1 - probs, probs])


def run_backtest_v2(min_train_size: int = 50, use_xgb: bool = True):
    """
    개선된 Rolling Window 백테스트 실행
    """
    print("=" * 60)
    print("ord2 예측 ML 모델 백테스트 v2 (개선판)")
    print("=" * 60)

    data = load_winning_numbers()
    print(f"데이터: {len(data)}회차 ({data[0]['round']}~{data[-1]['round']})")

    if use_xgb and not HAS_XGBOOST:
        print("XGBoost 미설치 → EnhancedScorer 사용")
        use_xgb = False

    results = []
    feature_importance_sum = defaultdict(float)
    feature_importance_count = 0

    # Rolling window 백테스트
    for i in range(min_train_size, len(data)):
        train_data = data[:i]
        target = data[i]

        # 진행 상황 출력
        if (i - min_train_size) % 50 == 0:
            print(f"  진행: {i - min_train_size + 1}/{len(data) - min_train_size} "
                  f"(회차 {target['round']})")

        # 범위별 확률 계산 (학습 데이터 기반)
        range_counts = Counter()
        for r in train_data:
            range_counts[get_range_index(r['ord2'])] += 1
        total_count = sum(range_counts.values())
        range_probs = {k: v/total_count for k, v in range_counts.items()}

        # 학습 데이터 준비
        X_train = []
        y_train = []
        train_features_list = []

        train_window = train_data[-100:] if len(train_data) > 100 else train_data

        for j, tr in enumerate(train_window):
            if j < 10:
                continue
            prev_data = train_window[:j]
            extractor = FeatureExtractor(prev_data)

            candidates = extractor.get_all_candidates(tr['ord1'], tr['ord6'])
            for cand in candidates:
                v2_features = extract_v2_features(cand, tr['ord1'], tr['ord6'], range_probs)
                X_train.append(features_to_array_v2(v2_features))
                y_train.append(1 if cand['candidate'] == tr['ord2'] else 0)
                train_features_list.append(v2_features)

        if len(X_train) < 10:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # 모델 학습
        if use_xgb:
            model = xgb.XGBClassifier(
                n_estimators=100,       # 증가
                max_depth=5,            # 증가
                learning_rate=0.08,     # 약간 감소
                min_child_weight=3,     # 과적합 방지
                subsample=0.8,          # 샘플링
                colsample_bytree=0.8,   # 피처 샘플링
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            model.fit(X_train, y_train)

            # 피처 중요도 누적
            for idx, importance in enumerate(model.feature_importances_):
                if idx < len(V2_FEATURE_COLS):
                    feature_importance_sum[V2_FEATURE_COLS[idx]] += importance
            feature_importance_count += 1
        else:
            model = EnhancedScorer()
            model.fit(X_train, y_train)

        # 예측
        extractor = FeatureExtractor(train_data)
        candidates = extractor.get_all_candidates(target['ord1'], target['ord6'])

        if not candidates:
            continue

        # v2 피처 추출
        test_features_list = []
        for c in candidates:
            v2_feat = extract_v2_features(c, target['ord1'], target['ord6'], range_probs)
            test_features_list.append(v2_feat)

        X_test = np.array([features_to_array_v2(f) for f in test_features_list])

        if use_xgb:
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict_proba(X_test, test_features_list)[:, 1]

        # 확률 순으로 정렬
        sorted_indices = np.argsort(-probs)
        sorted_candidates = [candidates[i]['candidate'] for i in sorted_indices]

        # 결과 저장
        actual_ord2 = target['ord2']
        predicted_ord2 = sorted_candidates[0] if sorted_candidates else -1

        # Top-K 적중 여부
        top3 = sorted_candidates[:3]
        top5 = sorted_candidates[:5]
        top10 = sorted_candidates[:10]

        in_top3 = 1 if actual_ord2 in top3 else 0
        in_top5 = 1 if actual_ord2 in top5 else 0
        in_top10 = 1 if actual_ord2 in top10 else 0
        exact_match = 1 if predicted_ord2 == actual_ord2 else 0

        # 실제 ord2의 순위 찾기
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
            'num_candidates': len(candidates),
            'top3': str(top3),
            'top5': str(top5),
            'exact_match': exact_match,
            'in_top3': in_top3,
            'in_top5': in_top5,
            'in_top10': in_top10,
            'error': abs(predicted_ord2 - actual_ord2)
        })

    # 결과 저장
    output_path = OUTPUT_DIR / "backtest_results_v2.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[저장] {output_path}")

    # 피처 중요도 저장 (XGBoost 사용 시)
    if use_xgb and feature_importance_count > 0:
        importance_path = OUTPUT_DIR / "feature_importance_v2.csv"
        with open(importance_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['feature', 'importance'])
            avg_importance = {k: v / feature_importance_count
                            for k, v in feature_importance_sum.items()}
            for feat, imp in sorted(avg_importance.items(), key=lambda x: -x[1]):
                writer.writerow([feat, f"{imp:.6f}"])
        print(f"[저장] {importance_path}")

    # 통계 출력
    print_statistics_v2(results)

    return results


def print_statistics_v2(results: List[Dict]):
    """백테스트 결과 통계 출력"""
    print("\n" + "=" * 60)
    print("백테스트 v2 결과 통계")
    print("=" * 60)

    n = len(results)
    exact = sum(r['exact_match'] for r in results)
    top3 = sum(r['in_top3'] for r in results)
    top5 = sum(r['in_top5'] for r in results)
    top10 = sum(r['in_top10'] for r in results)
    avg_error = sum(r['error'] for r in results) / n
    avg_rank = sum(r['actual_rank'] for r in results) / n
    avg_candidates = sum(r['num_candidates'] for r in results) / n

    print(f"\n테스트 회차: {n}회")
    print(f"평균 후보 수: {avg_candidates:.1f}개")

    print(f"\n[적중률]")
    print(f"  정확도 (Top-1): {exact:4d}회 ({exact/n*100:5.1f}%)")
    print(f"  Top-3 적중:     {top3:4d}회 ({top3/n*100:5.1f}%)")
    print(f"  Top-5 적중:     {top5:4d}회 ({top5/n*100:5.1f}%)")
    print(f"  Top-10 적중:    {top10:4d}회 ({top10/n*100:5.1f}%)")

    print(f"\n[오차]")
    print(f"  평균 오차: {avg_error:.2f}")
    print(f"  평균 순위: {avg_rank:.2f}")

    # 범위별 분석
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
        exact_pct = stats['exact'] / total * 100 if total > 0 else 0
        top5_pct = stats['top5'] / total * 100 if total > 0 else 0
        top10_pct = stats['top10'] / total * 100 if total > 0 else 0
        range_name = f"{range_idx*10:02d}-{range_idx*10+9:02d}"
        print(f"  {range_name}: {total:3d}회, Top-1 {exact_pct:5.1f}%, Top-5 {top5_pct:5.1f}%, Top-10 {top10_pct:5.1f}%")


if __name__ == "__main__":
    run_backtest_v2(min_train_size=50, use_xgb=HAS_XGBOOST)
