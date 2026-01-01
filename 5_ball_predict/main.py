"""
5_ball_predict/main.py - Ball 기반 ML 예측 파이프라인

목표: 100개 조합 중 1개라도 6개 완전 일치

파이프라인:
1. (ball1, ball6) 쌍 후보 생성
2. 각 쌍에 대해 ball2 → ball3 → ball4 → ball5 예측
3. 6개 조합 생성 및 이상치 점수 계산
4. 다양성 필터링 (3자리 이상 중복 제거)
5. 최종 100개 선택
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import importlib.util

# 명시적 모듈 로드 (경로 충돌 방지)
def _load_local_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_base = Path(__file__).parent
_ball_common = _load_local_module("ball_common", _base / "common.py")

load_winning_data = _ball_common.load_winning_data
BallCombination = _ball_common.BallCombination
calculate_ball_outlier_score = _ball_common.calculate_ball_outlier_score
select_diverse_ball_combinations = _ball_common.select_diverse_ball_combinations
compare_ball_with_actual = _ball_common.compare_ball_with_actual
format_ball_combination = _ball_common.format_ball_combination
convert_ball_combo_to_ord = _ball_common.convert_ball_combo_to_ord

# 각 예측 모듈 로드
ball1ball6_ml = _load_local_module("ball1ball6", _base / "1_ball1ball6_ml" / "predict.py")
ball2_ml = _load_local_module("ball2", _base / "2_ball2_ml" / "predict.py")
ball3_ml = _load_local_module("ball3", _base / "3_ball3_ml" / "predict.py")
ball4_ml = _load_local_module("ball4", _base / "4_ball4_ml" / "predict.py")
ball5_ml = _load_local_module("ball5", _base / "5_ball5_ml" / "predict.py")

predict_ball1ball6_v2 = ball1ball6_ml.predict_ball1ball6_v2
predict_ball2 = ball2_ml.predict_ball2
predict_ball3 = ball3_ml.predict_ball3
predict_ball4 = ball4_ml.predict_ball4
predict_ball5 = ball5_ml.predict_ball5


def generate_all_ball_combinations(
    target_round: int,
    all_data: List[Dict],
    top_pairs: int = 400,
    top_inner: int = 3
) -> List[BallCombination]:
    """
    모든 가능한 ball 6개 조합 생성

    Args:
        target_round: 예측할 회차
        all_data: 전체 데이터
        top_pairs: (ball1, ball6) 상위 개수
        top_inner: ball2~5 각각의 상위 개수

    Returns:
        BallCombination 리스트
    """
    past_data = [d for d in all_data if d['round'] < target_round]
    if len(past_data) < 10:
        return []

    # 이전 회차 번호 (이월수 계산용)
    prev_round = past_data[-1]
    prev_numbers = tuple(prev_round[f'ball{i}'] for i in range(1, 7))

    # (ball1, ball6) 쌍 예측
    import math
    top_each = int(math.ceil(math.sqrt(top_pairs)))
    pairs = predict_ball1ball6_v2(target_round, all_data, top_ball1=top_each, top_ball6=top_each)
    pairs = pairs[:top_pairs]

    all_combos = []

    for ball1, ball6, pair_score in pairs:
        used = {ball1, ball6}

        # ball2 예측
        ball2_candidates = predict_ball2(ball1, ball6, past_data, top_k=top_inner)
        if not ball2_candidates:
            continue

        for ball2, _ in ball2_candidates:
            used2 = used | {ball2}

            # ball3 예측
            ball3_candidates = predict_ball3(used2, past_data, top_k=top_inner)
            if not ball3_candidates:
                continue

            for ball3, _ in ball3_candidates:
                used3 = used2 | {ball3}

                # ball4 예측
                ball4_candidates = predict_ball4(used3, past_data, top_k=top_inner)
                if not ball4_candidates:
                    continue

                for ball4, _ in ball4_candidates:
                    used4 = used3 | {ball4}

                    # ball5 예측
                    ball5_candidates = predict_ball5(used4, past_data, top_k=top_inner)
                    if not ball5_candidates:
                        continue

                    for ball5, _ in ball5_candidates:
                        # 6개 번호 (ball 순서)
                        numbers = (ball1, ball2, ball3, ball4, ball5, ball6)

                        # 이상치 점수 계산
                        score = calculate_ball_outlier_score(numbers, prev_numbers)

                        combo = BallCombination(
                            numbers=numbers,
                            score=score
                        )
                        all_combos.append(combo)

    return all_combos


def predict_ball_for_round(
    target_round: int,
    all_data: List[Dict],
    max_combinations: int = 100,
    top_pairs: int = 400,
    top_inner: int = 3,
    verbose: bool = False
) -> List[BallCombination]:
    """
    특정 회차 Ball 예측

    Args:
        target_round: 예측할 회차
        all_data: 전체 데이터
        max_combinations: 최종 조합 수
        top_pairs: (ball1, ball6) 상위 개수
        top_inner: ball2~5 각각의 상위 개수
        verbose: 상세 출력

    Returns:
        다양성 필터링된 조합 리스트
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"  {target_round}회차 Ball 예측")
        print(f"{'='*50}")

    # 모든 조합 생성
    all_combos = generate_all_ball_combinations(
        target_round, all_data, top_pairs, top_inner
    )

    if verbose:
        print(f"생성된 전체 Ball 조합: {len(all_combos):,}개")

    if not all_combos:
        return []

    # 다양성 필터링
    diverse_combos = select_diverse_ball_combinations(all_combos, max_combinations)

    if verbose:
        print(f"다양성 필터링 후: {len(diverse_combos)}개")

    return diverse_combos


def print_ball_combinations(predictions: List[BallCombination], actual: tuple = None, round_data: Dict = None, start_num: int = 1):
    """Ball 조합 전체 출력 (Ball→Ord 변환, 오름차순 정렬) - 요약 데이터 반환"""
    actual_ord = None
    if round_data:
        actual_ord = tuple(round_data[f'ord{i}'] for i in range(1, 7))

    match_counts = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    ord_match_counts = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    best_match = 0
    best_combo = None
    best_ord_match = 0
    best_ord_combo = None

    for i, combo in enumerate(predictions, start_num):
        # Ball 일치 계산 (요약용)
        if actual:
            matched_nums = set(combo.numbers) & set(actual)
            match = len(matched_nums)
            match_counts[match].append(sorted(matched_nums))
            if match > best_match:
                best_match = match
                best_combo = combo

        # Ball→Ord 변환 (오름차순 정렬)
        if round_data:
            ord_combo = convert_ball_combo_to_ord(combo.numbers, round_data)
            ord_combo_sorted = tuple(sorted(ord_combo))  # 오름차순 정렬
            actual_ord_set = set(actual_ord)
            matched = set(ord_combo_sorted) & actual_ord_set
            ord_match_count = len(matched)
            ord_match_counts[ord_match_count].append(sorted(matched))

            if ord_match_count > best_ord_match:
                best_ord_match = ord_match_count
                best_ord_combo = ord_combo_sorted

            # Ord 일치 개수 및 일치 번호 표시
            if ord_match_count >= 3:
                ord_mark = "★" * (ord_match_count - 2)
            else:
                ord_mark = f"({ord_match_count}개)"
            ord_matched_str = f"{sorted(matched)}" if ord_match_count > 0 else ""
            line = f"  {i:3d}. {format_ball_combination(ord_combo_sorted)} {ord_mark} {ord_matched_str}"
        else:
            # round_data 없으면 Ball 조합 표시
            line = f"  {i:3d}. {format_ball_combination(combo.numbers)}"

        print(line)

    # 요약 데이터 반환
    return {
        'actual': actual,
        'actual_ord': actual_ord,
        'match_counts': match_counts,
        'ord_match_counts': ord_match_counts,
        'best_match': best_match,
        'best_combo': best_combo,
        'best_ord_match': best_ord_match,
        'best_ord_combo': best_ord_combo
    }


def print_ball_summary(summary_data: dict):
    """Ball 일치 분포 요약 출력"""
    if not summary_data or not summary_data.get('actual'):
        return

    actual = summary_data['actual']
    match_counts = summary_data['match_counts']
    best_match = summary_data['best_match']
    best_combo = summary_data['best_combo']

    print(f"\n{'='*90}")
    print(f"  Ball 일치 분포 요약")
    print(f"{'='*90}")
    print(f"  실제 당첨번호: {format_ball_combination(actual)}")
    print(f"  최대 일치: {best_match}개")
    if best_combo:
        print(f"  최근접 조합: {format_ball_combination(best_combo.numbers)}")
    print()
    for m in range(6, -1, -1):
        if match_counts[m]:
            from collections import Counter
            nums_counter = Counter(tuple(nums) for nums in match_counts[m])
            nums_str = ", ".join(f"{list(nums)}" for nums, _ in nums_counter.most_common(5))
            if len(nums_counter) > 5:
                nums_str += f" 외 {len(nums_counter)-5}개"
            print(f"  {m}개 일치: {len(match_counts[m]):3d}개 → {nums_str}")


def print_ball_ord_summary(summary_data: dict):
    """Ball→Ord 일치 분포 요약 출력"""
    if not summary_data or not summary_data.get('actual_ord'):
        return

    actual_ord = summary_data['actual_ord']
    ord_match_counts = summary_data['ord_match_counts']
    best_ord_match = summary_data['best_ord_match']
    best_ord_combo = summary_data['best_ord_combo']

    print(f"\n{'='*90}")
    print(f"  Ball→Ord(판매순위) 일치 분포 요약")
    print(f"{'='*90}")
    print(f"  실제 ord1-ord6: {format_ball_combination(actual_ord)}")
    print(f"  최대 일치: {best_ord_match}개")
    if best_ord_combo:
        print(f"  최근접 조합: {format_ball_combination(best_ord_combo)}")
    print()
    for m in range(6, -1, -1):
        if ord_match_counts[m]:
            from collections import Counter
            nums_counter = Counter(tuple(nums) for nums in ord_match_counts[m])
            nums_str = ", ".join(f"{list(nums)}" for nums, _ in nums_counter.most_common(5))
            if len(nums_counter) > 5:
                nums_str += f" 외 {len(nums_counter)-5}개"
            print(f"  {m}개 일치: {len(ord_match_counts[m]):3d}개 → {nums_str}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ball 예측 파이프라인')
    parser.add_argument('rounds', type=int, nargs='*', help='예측할 회차 (1개: 단일, 2개: 범위)')
    parser.add_argument('--top', type=int, default=100, help='최종 조합 수')
    parser.add_argument('--top_pairs', type=int, default=400, help='(ball1,ball6) 상위 개수')
    parser.add_argument('--top_inner', type=int, default=3, help='ball2~5 상위 개수')

    args = parser.parse_args()

    # 데이터 로드
    all_data = load_winning_data()
    print(f"총 {len(all_data)}개 회차 로드")

    if len(args.rounds) == 2:
        start_round, end_round = args.rounds[0], args.rounds[1]
        print(f"\n{start_round}~{end_round}회차 Ball 예측 시작...")

        for target_round in range(start_round, end_round + 1):
            predictions = predict_ball_for_round(
                target_round, all_data, args.top,
                top_pairs=args.top_pairs, top_inner=args.top_inner,
                verbose=True
            )

            actual_data = next((d for d in all_data if d['round'] == target_round), None)
            actual = tuple(actual_data[f'ball{i}'] for i in range(1, 7)) if actual_data else None

            print_ball_combinations(predictions, actual, actual_data)

    elif len(args.rounds) == 1:
        target_round = args.rounds[0]
        print(f"\n{target_round}회차 Ball 예측 시작...")

        predictions = predict_ball_for_round(
            target_round, all_data, args.top,
            top_pairs=args.top_pairs, top_inner=args.top_inner,
            verbose=True
        )

        actual_data = next((d for d in all_data if d['round'] == target_round), None)
        actual = tuple(actual_data[f'ball{i}'] for i in range(1, 7)) if actual_data else None

        print_ball_combinations(predictions, actual, actual_data)

    else:
        latest = all_data[-1]['round']
        next_round = latest + 1
        print(f"\n다음 회차 ({next_round}회) Ball 예측 시작...")

        predictions = predict_ball_for_round(
            next_round, all_data, args.top,
            top_pairs=args.top_pairs, top_inner=args.top_inner,
            verbose=True
        )

        print_ball_combinations(predictions, None, None)
