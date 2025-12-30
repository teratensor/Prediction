"""
ord2 예측 ML 모델 백테스트

Rolling Window 방식으로 전체 데이터(827~1204) 백테스트
XGBoost 분류 모델 사용
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using simple scoring model.")

from features import load_winning_numbers, FeatureExtractor

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# 피처 컬럼 (모델 학습에 사용)
FEATURE_COLS = [
    'range_idx', 'range_prob', 'pos2_freq', 'is_hot', 'is_cold',
    'sum_contribution', 'expected_remaining_avg', 'segment',
    'is_prime', 'is_consecutive', 'consecutive_prob',
    'relative_position', 'overall_freq', 'in_recent_3'
]


def features_to_array(features: Dict) -> np.ndarray:
    """피처 딕셔너리를 numpy 배열로 변환"""
    return np.array([features.get(col, 0) for col in FEATURE_COLS])


class SimpleScorer:
    """XGBoost 없을 때 사용하는 간단한 점수 기반 모델"""

    def __init__(self):
        # 피처별 가중치 (경험적 설정)
        self.weights = {
            'range_prob': 30,
            'pos2_freq': 50,
            'is_hot': 10,
            'is_cold': -5,
            'is_prime': 5,
            'is_consecutive': 8,
            'overall_freq': 20,
            'in_recent_3': -3,
            'segment': -3,  # Rest7 페널티
        }

    def fit(self, X, y):
        """학습 (SimpleScorer는 학습 불필요)"""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """확률 예측 (점수를 0~1로 변환)"""
        scores = []
        for row in X:
            score = 0
            for i, col in enumerate(FEATURE_COLS):
                if col in self.weights:
                    score += row[i] * self.weights[col]
            scores.append(score)

        # Min-max 정규화
        scores = np.array(scores)
        min_s, max_s = scores.min(), scores.max()
        if max_s > min_s:
            probs = (scores - min_s) / (max_s - min_s)
        else:
            probs = np.ones_like(scores) / len(scores)

        return np.column_stack([1 - probs, probs])


def run_backtest(min_train_size: int = 50, use_xgb: bool = True):
    """
    Rolling Window 백테스트 실행

    Args:
        min_train_size: 최소 학습 데이터 크기
        use_xgb: XGBoost 사용 여부
    """
    print("=" * 60)
    print("ord2 예측 ML 모델 백테스트")
    print("=" * 60)

    data = load_winning_numbers()
    print(f"데이터: {len(data)}회차 ({data[0]['round']}~{data[-1]['round']})")

    if use_xgb and not HAS_XGBOOST:
        print("XGBoost 미설치 → SimpleScorer 사용")
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

        # 학습 데이터 준비
        X_train = []
        y_train = []

        # 최근 학습 데이터에서 샘플 추출 (메모리/속도 위해 최근 100회만)
        train_window = train_data[-100:] if len(train_data) > 100 else train_data

        for j, tr in enumerate(train_window):
            # 이전 데이터로 피처 추출기 생성
            if j < 10:
                continue  # 최소 10회차 필요
            prev_data = train_window[:j]
            extractor = FeatureExtractor(prev_data)

            candidates = extractor.get_all_candidates(tr['ord1'], tr['ord6'])
            for cand in candidates:
                X_train.append(features_to_array(cand))
                y_train.append(1 if cand['candidate'] == tr['ord2'] else 0)

        if len(X_train) < 10:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # 모델 학습
        if use_xgb:
            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            model.fit(X_train, y_train)

            # 피처 중요도 누적
            for idx, importance in enumerate(model.feature_importances_):
                feature_importance_sum[FEATURE_COLS[idx]] += importance
            feature_importance_count += 1
        else:
            model = SimpleScorer()
            model.fit(X_train, y_train)

        # 예측
        extractor = FeatureExtractor(train_data)
        candidates = extractor.get_all_candidates(target['ord1'], target['ord6'])

        if not candidates:
            continue

        X_test = np.array([features_to_array(c) for c in candidates])
        probs = model.predict_proba(X_test)[:, 1]

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
    output_path = OUTPUT_DIR / "backtest_results.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[저장] {output_path}")

    # 피처 중요도 저장 (XGBoost 사용 시)
    if use_xgb and feature_importance_count > 0:
        importance_path = OUTPUT_DIR / "feature_importance.csv"
        with open(importance_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['feature', 'importance'])
            avg_importance = {k: v / feature_importance_count
                            for k, v in feature_importance_sum.items()}
            for feat, imp in sorted(avg_importance.items(), key=lambda x: -x[1]):
                writer.writerow([feat, f"{imp:.6f}"])
        print(f"[저장] {importance_path}")

    # 통계 출력
    print_statistics(results)

    return results


def print_statistics(results: List[Dict]):
    """백테스트 결과 통계 출력"""
    print("\n" + "=" * 60)
    print("백테스트 결과 통계")
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
    range_stats = defaultdict(lambda: {'total': 0, 'exact': 0, 'top5': 0})
    for r in results:
        range_idx = r['actual_ord2'] // 10
        range_stats[range_idx]['total'] += 1
        range_stats[range_idx]['exact'] += r['exact_match']
        range_stats[range_idx]['top5'] += r['in_top5']

    for range_idx in sorted(range_stats.keys()):
        stats = range_stats[range_idx]
        total = stats['total']
        exact_pct = stats['exact'] / total * 100 if total > 0 else 0
        top5_pct = stats['top5'] / total * 100 if total > 0 else 0
        range_name = f"{range_idx*10:02d}-{range_idx*10+9:02d}"
        print(f"  {range_name}: {total:3d}회, 정확 {exact_pct:5.1f}%, Top-5 {top5_pct:5.1f}%")


if __name__ == "__main__":
    run_backtest(min_train_size=50, use_xgb=HAS_XGBOOST)
