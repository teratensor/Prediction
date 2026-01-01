"""
3_predict/main.py - ML 예측 파이프라인

목표: 100개 조합 중 1개라도 6개 완전 일치

파이프라인:
1. (ord1, ord6) 쌍 후보 생성
2. 각 쌍에 대해 ord4 → ord2 → ord3 → ord5 예측
3. 6개 조합 생성 및 이상치 점수 계산
4. 다양성 필터링 (3자리 이상 중복 제거)
5. 최종 100개 선택

+ Ball 예측도 함께 출력
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from itertools import product

# 모듈 경로 추가 (3_predict 우선)
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    load_winning_data,
    Combination,
    calculate_outlier_score,
    select_diverse_combinations,
    compare_with_actual,
    format_combination,
    get_cluster_pattern,
)
# 숫자로 시작하는 폴더는 import 불가 → importlib 사용
import importlib.util

def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 각 예측 모듈 로드
_base = Path(__file__).parent
ord1ord6_ml = load_module("ord1ord6", _base / "1_ord1ord6_ml" / "predict.py")
ord2_ml = load_module("ord2", _base / "2_ord2_ml" / "predict.py")
ord3_ml = load_module("ord3", _base / "3_ord3_ml" / "predict.py")
ord4_ml = load_module("ord4", _base / "4_ord4_ml" / "predict.py")
ord5_ml = load_module("ord5", _base / "5_ord5_ml" / "predict.py")

predict_ord1ord6 = ord1ord6_ml.predict_ord1ord6
predict_ord1ord6_v2 = ord1ord6_ml.predict_ord1ord6_v2
get_frequency_filtered_pairs = ord1ord6_ml.get_frequency_filtered_pairs
predict_ord2 = ord2_ml.predict_ord2
predict_ord3 = ord3_ml.predict_ord3
predict_ord4 = ord4_ml.predict_ord4
predict_ord5 = ord5_ml.predict_ord5

# Ball 예측 모듈 로드
_ball_base = Path(__file__).parent.parent / "5_ball_predict"
try:
    ball_main = load_module("ball_main", _ball_base / "main.py")
    predict_ball_for_round = ball_main.predict_ball_for_round
    print_ball_combinations = ball_main.print_ball_combinations
    BALL_PREDICT_AVAILABLE = True
except Exception:
    BALL_PREDICT_AVAILABLE = False


def generate_all_combinations(
    target_round: int,
    all_data: List[Dict],
    top_pairs: int = 50,
    top_inner: int = 5,
    use_v2: bool = True
) -> List[Combination]:
    """
    모든 가능한 6개 조합 생성

    Args:
        target_round: 예측할 회차
        all_data: 전체 데이터
        top_pairs: (ord1, ord6) 상위 개수
        top_inner: ord2~5 각각의 상위 개수
        use_v2: v2 개별예측 방식 사용 여부 (기본: True)

    Returns:
        Combination 리스트
    """
    past_data = [d for d in all_data if d['round'] < target_round]
    if len(past_data) < 10:
        return []

    # 이전 회차 번호 (이월수 계산용)
    prev_round = past_data[-1]
    prev_numbers = tuple(prev_round[f'ord{i}'] for i in range(1, 7))

    if use_v2:
        # v2: 개별 예측 후 조합 (20×20=400개 쌍, 범위 가중치 적용)
        import math
        top_each = int(math.ceil(math.sqrt(top_pairs)))  # top_pairs=400 → 20×20
        pairs = predict_ord1ord6_v2(target_round, all_data, top_ord1=top_each, top_ord6=top_each)
        pairs = pairs[:top_pairs]  # 상위 top_pairs개만
    else:
        # 기존 방식: 빈도 필터링 + 점수 기반 병합
        freq_pairs = get_frequency_filtered_pairs(target_round, all_data, min_freq=1, top_k=top_pairs)
        all_pairs = predict_ord1ord6(target_round, all_data, top_k=top_pairs * 2)

        seen = set()
        pairs = []
        for p in freq_pairs + all_pairs:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                pairs.append(p)
                if len(pairs) >= top_pairs:
                    break

        if not pairs:
            pairs = predict_ord1ord6(target_round, all_data, top_k=top_pairs)

    all_combos = []

    for ord1, ord6, pair_score in pairs:
        # 2. ord4 예측
        ord4_candidates = predict_ord4(ord1, ord6, past_data, top_k=top_inner)
        if not ord4_candidates:
            continue

        for ord4, _ in ord4_candidates:
            # 3. ord2 예측
            ord2_candidates = predict_ord2(ord1, ord4, past_data, top_k=top_inner)
            if not ord2_candidates:
                continue

            for ord2, _ in ord2_candidates:
                # 4. ord3 예측
                ord3_candidates = predict_ord3(ord2, ord4, past_data, top_k=top_inner)
                if not ord3_candidates:
                    continue

                for ord3, _ in ord3_candidates:
                    # 5. ord5 예측
                    ord5_candidates = predict_ord5(ord4, ord6, past_data, top_k=top_inner)
                    if not ord5_candidates:
                        continue

                    for ord5, _ in ord5_candidates:
                        # 유효성 검사
                        if not (ord1 < ord2 < ord3 < ord4 < ord5 < ord6):
                            continue

                        numbers = (ord1, ord2, ord3, ord4, ord5, ord6)

                        # 이상치 점수 계산
                        score = calculate_outlier_score(numbers, prev_numbers)

                        combo = Combination(
                            numbers=numbers,
                            score=score,
                            cluster_pattern=get_cluster_pattern(numbers)
                        )
                        all_combos.append(combo)

    return all_combos


def predict_for_round(
    target_round: int,
    all_data: List[Dict],
    max_combinations: int = 100,
    top_pairs: int = 400,  # 다양성을 위해 더 많은 쌍
    top_inner: int = 3,    # 각 위치당 적은 후보
    verbose: bool = False
) -> List[Combination]:
    """
    특정 회차 예측

    Args:
        target_round: 예측할 회차
        all_data: 전체 데이터
        max_combinations: 최종 조합 수
        top_pairs: (ord1, ord6) 상위 개수
        top_inner: ord2~5 각각의 상위 개수
        verbose: 상세 출력

    Returns:
        다양성 필터링된 조합 리스트
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"  {target_round}회차 예측")
        print(f"{'='*50}")

    # 모든 조합 생성
    all_combos = generate_all_combinations(
        target_round, all_data, top_pairs, top_inner
    )

    if verbose:
        print(f"생성된 전체 조합: {len(all_combos):,}개")

    if not all_combos:
        return []

    # 다양성 필터링
    diverse_combos = select_diverse_combinations(all_combos, max_combinations)

    if verbose:
        print(f"다양성 필터링 후: {len(diverse_combos)}개")
        print(f"평균 이상치 점수: {sum(c.score for c in diverse_combos)/len(diverse_combos):.2f}")

    return diverse_combos


def backtest(
    all_data: List[Dict],
    start_round: int = 1150,
    end_round: int = 1204,
    max_combinations: int = 100,
    verbose: bool = True
) -> Dict:
    """
    백테스트 실행

    Args:
        all_data: 전체 데이터
        start_round: 시작 회차
        end_round: 종료 회차
        max_combinations: 회차당 최대 조합 수
        verbose: 상세 출력

    Returns:
        백테스트 결과
    """
    results = []
    perfect_matches = []
    match_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    for target_round in range(start_round, end_round + 1):
        # 실제 당첨번호
        actual_data = next((d for d in all_data if d['round'] == target_round), None)
        if not actual_data:
            continue

        actual = tuple(actual_data[f'ord{i}'] for i in range(1, 7))

        # 예측
        predictions = predict_for_round(
            target_round, all_data, max_combinations,
            top_pairs=50, top_inner=5, verbose=False
        )

        if not predictions:
            continue

        # 비교
        comparison = compare_with_actual(predictions, actual)
        results.append({
            'round': target_round,
            'actual': actual,
            'best_match': comparison['best_match_count'],
            'best_combo': comparison['best_combo'],
            'total_combos': len(predictions)
        })

        match_distribution[comparison['best_match_count']] += 1

        if comparison['is_perfect']:
            perfect_matches.append(target_round)

        if verbose:
            status = "✓ 6개 일치!" if comparison['is_perfect'] else f"최대 {comparison['best_match_count']}개"
            print(f"{target_round}회: {format_combination(actual)} → {status}")

    # 요약
    total = len(results)
    summary = {
        'total_rounds': total,
        'perfect_matches': len(perfect_matches),
        'perfect_rounds': perfect_matches,
        'match_distribution': match_distribution,
        'avg_best_match': sum(r['best_match'] for r in results) / total if total > 0 else 0,
        'results': results
    }

    return summary


def print_backtest_summary(summary: Dict):
    """백테스트 결과 출력"""
    print(f"\n{'='*50}")
    print("  백테스트 결과 요약")
    print(f"{'='*50}")
    print(f"테스트 회차: {summary['total_rounds']}개")
    print(f"6개 완전 일치: {summary['perfect_matches']}회")
    print(f"평균 최대 일치: {summary['avg_best_match']:.2f}개")

    print("\n일치 분포:")
    for match, count in sorted(summary['match_distribution'].items()):
        pct = count / summary['total_rounds'] * 100 if summary['total_rounds'] > 0 else 0
        bar = '█' * int(pct / 5)
        print(f"  {match}개 일치: {count:3d}회 ({pct:5.1f}%) {bar}")

    if summary['perfect_rounds']:
        print(f"\n6개 일치 회차: {summary['perfect_rounds']}")


def print_all_combinations(predictions: List[Combination], actual: tuple = None):
    """100개 조합 전체 출력 (일치 수 표시)"""
    print(f"\n{'='*70}")
    print(f"  전체 {len(predictions)}개 조합")
    print(f"{'='*70}")

    match_counts = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    best_match = 0
    best_combo = None

    for i, combo in enumerate(predictions, 1):
        if actual:
            matched_nums = set(combo.numbers) & set(actual)
            match = len(matched_nums)
            match_counts[match].append(sorted(matched_nums))
            if match > best_match:
                best_match = match
                best_combo = combo

        print(f"  {i:3d}. {format_combination(combo.numbers)}")

    # 요약
    if actual:
        print(f"\n{'='*70}")
        print(f"  일치 분포 요약")
        print(f"{'='*70}")
        print(f"  실제 당첨번호: {format_combination(actual)}")
        print(f"  최대 일치: {best_match}개")
        if best_combo:
            print(f"  최근접 조합: {format_combination(best_combo.numbers)}")
        print()
        for m in range(6, -1, -1):
            if match_counts[m]:
                # 일치된 숫자 조합별로 그룹화
                from collections import Counter
                nums_counter = Counter(tuple(nums) for nums in match_counts[m])
                nums_str = ", ".join(f"{list(nums)}" for nums, _ in nums_counter.most_common(5))
                if len(nums_counter) > 5:
                    nums_str += f" 외 {len(nums_counter)-5}개"
                print(f"  {m}개 일치: {len(match_counts[m]):3d}개 → {nums_str}")


def main():
    """메인 실행"""
    import argparse

    parser = argparse.ArgumentParser(description='로또 예측 파이프라인')
    parser.add_argument('rounds', type=int, nargs='*', help='예측할 회차 (1개: 단일, 2개: 범위)')
    parser.add_argument('--backtest', action='store_true', help='백테스트 실행')
    parser.add_argument('--start', type=int, default=1150, help='백테스트 시작 회차')
    parser.add_argument('--end', type=int, default=1204, help='백테스트 종료 회차')
    parser.add_argument('--top', type=int, default=100, help='최종 조합 수')
    parser.add_argument('--top_pairs', type=int, default=400, help='(ord1,ord6) 상위 개수')
    parser.add_argument('--top_inner', type=int, default=3, help='ord2~5 상위 개수')

    args = parser.parse_args()

    # 데이터 로드
    all_data = load_winning_data()
    print(f"총 {len(all_data)}개 회차 로드")

    if args.backtest:
        # 백테스트 모드
        summary = backtest(all_data, args.start, args.end, args.top, verbose=True)
        print_backtest_summary(summary)

    elif len(args.rounds) == 2:
        # 범위 예측 모드 (시작~끝)
        start_round, end_round = args.rounds[0], args.rounds[1]
        print(f"\n{start_round}~{end_round}회차 예측 시작...")
        print(f"파라미터: top_pairs={args.top_pairs}, top_inner={args.top_inner}")

        for target_round in range(start_round, end_round + 1):
            # Ord 예측
            predictions = predict_for_round(
                target_round, all_data, args.top,
                top_pairs=args.top_pairs, top_inner=args.top_inner,
                verbose=True
            )

            # 실제 당첨번호 확인
            actual_data = next((d for d in all_data if d['round'] == target_round), None)
            actual_ord = tuple(actual_data[f'ord{i}'] for i in range(1, 7)) if actual_data else None

            # Ord 100개 전체 출력
            print_all_combinations(predictions, actual_ord)

            # Ball 예측
            if BALL_PREDICT_AVAILABLE:
                ball_predictions = predict_ball_for_round(
                    target_round, all_data, args.top,
                    top_pairs=args.top_pairs, top_inner=args.top_inner,
                    verbose=True
                )
                actual_ball = tuple(actual_data[f'ball{i}'] for i in range(1, 7)) if actual_data else None
                print_ball_combinations(ball_predictions, actual_ball)

    elif len(args.rounds) == 1:
        # 단일 회차 예측
        target_round = args.rounds[0]
        print(f"\n{target_round}회차 예측 시작...")
        print(f"파라미터: top_pairs={args.top_pairs}, top_inner={args.top_inner}")

        # Ord 예측
        predictions = predict_for_round(
            target_round, all_data, args.top,
            top_pairs=args.top_pairs, top_inner=args.top_inner,
            verbose=True
        )

        # 실제 당첨번호 확인
        actual_data = next((d for d in all_data if d['round'] == target_round), None)
        actual_ord = tuple(actual_data[f'ord{i}'] for i in range(1, 7)) if actual_data else None

        # Ord 100개 전체 출력
        print_all_combinations(predictions, actual_ord)

        # Ball 예측
        if BALL_PREDICT_AVAILABLE:
            ball_predictions = predict_ball_for_round(
                target_round, all_data, args.top,
                top_pairs=args.top_pairs, top_inner=args.top_inner,
                verbose=True
            )
            actual_ball = tuple(actual_data[f'ball{i}'] for i in range(1, 7)) if actual_data else None
            print_ball_combinations(ball_predictions, actual_ball)

    else:
        # 기본: 가장 최근 회차 + 1 예측
        latest = all_data[-1]['round']
        next_round = latest + 1
        print(f"\n다음 회차 ({next_round}회) 예측 시작...")
        print(f"파라미터: top_pairs={args.top_pairs}, top_inner={args.top_inner}")

        # Ord 예측
        predictions = predict_for_round(
            next_round, all_data, args.top,
            top_pairs=args.top_pairs, top_inner=args.top_inner,
            verbose=True
        )

        # Ord 100개 전체 출력 (실제 번호 없음)
        print_all_combinations(predictions, None)

        # Ball 예측
        if BALL_PREDICT_AVAILABLE:
            ball_predictions = predict_ball_for_round(
                next_round, all_data, args.top,
                top_pairs=args.top_pairs, top_inner=args.top_inner,
                verbose=True
            )
            print_ball_combinations(ball_predictions, None)


if __name__ == '__main__':
    main()
