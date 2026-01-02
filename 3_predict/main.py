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
import csv
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


def print_all_combinations(predictions: List[Combination], actual: tuple = None, start_num: int = 1):
    """100개 조합 전체 출력 (일치 수 표시) - 요약 데이터 반환"""
    if start_num == 1:
        print(f"\n{'='*70}")
        print(f"  전체 200개 Ord 조합 (Ball→Ord 포함)")
        print(f"{'='*70}")

    match_counts = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    best_match = 0
    best_combo = None

    for i, combo in enumerate(predictions, start_num):
        line = f"  {i:3d}. {format_combination(combo.numbers)}"

        if actual:
            matched_nums = set(combo.numbers) & set(actual)
            match = len(matched_nums)
            match_counts[match].append(sorted(matched_nums))
            if match > best_match:
                best_match = match
                best_combo = combo

            # 일치 개수 및 일치 번호 표시
            if match >= 3:
                mark = "★" * (match - 2)
            else:
                mark = f"({match}개)"
            matched_str = f"{sorted(matched_nums)}" if match > 0 else ""
            line += f" {mark} {matched_str}"

        print(line)

    # 요약 데이터 반환
    return {
        'match_counts': match_counts,
        'best_match': best_match,
        'best_combo': best_combo,
        'actual': actual
    }


def save_frequency_to_csv(target_round: int, ord_predictions: List, ball_predictions: List, actual_ord: tuple, actual_data: dict = None):
    """
    회차별 실제 당첨번호의 위치별 빈도수를 CSV로 저장

    Args:
        target_round: 예측 회차
        ord_predictions: Ord 예측 조합 리스트
        ball_predictions: Ball 예측 조합 리스트
        actual_ord: 실제 당첨번호 (ord1~ord6)
        actual_data: 전체 회차 데이터 (Ball→Ord 변환용)
    """
    from collections import Counter

    result_dir = Path(__file__).parent.parent / "result"
    result_dir.mkdir(exist_ok=True)
    csv_path = result_dir / "onehot.csv"

    # 기존 데이터 로드 (해당 회차 제외)
    existing_rows = []
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['round']) != target_round:
                    existing_rows.append(row)

    # 위치별 숫자 수집
    position_numbers = [[] for _ in range(6)]

    # Ord 조합
    for combo in ord_predictions:
        for j in range(6):
            position_numbers[j].append(combo.numbers[j])

    # Ball→Ord 조합
    if ball_predictions and actual_data:
        _ball_base = Path(__file__).parent.parent / "5_ball_predict"
        ball_common = load_module("ball_common", _ball_base / "common.py")
        convert_ball_combo_to_ord = ball_common.convert_ball_combo_to_ord

        for combo in ball_predictions:
            ord_combo = convert_ball_combo_to_ord(combo.numbers, actual_data)
            ord_combo_sorted = tuple(sorted(ord_combo))
            for j in range(6):
                position_numbers[j].append(ord_combo_sorted[j])

    # 위치별 빈도수 계산
    position_counters = [Counter(nums) for nums in position_numbers]

    # 실제 당첨번호의 각 위치별 빈도수 추출
    freq_values = []
    for j in range(6):
        actual_num = actual_ord[j]
        freq = position_counters[j].get(actual_num, 0)
        freq_values.append(freq)

    freq_str = ','.join(str(f) for f in freq_values)

    # 새 데이터 (1줄)
    new_row = {
        'round': target_round,
        'frequency': freq_str
    }

    # 기존 + 새 데이터 병합 후 정렬
    all_rows = existing_rows + [new_row]
    all_rows.sort(key=lambda x: int(x['round']))

    # CSV 저장
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'frequency'])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  → CSV 저장 완료: {csv_path} (빈도수: {freq_str})")


def print_position_frequency(ord_predictions: List, ball_predictions: List = None, actual_data: dict = None):
    """각 위치(ord1~ord6)별 숫자 빈도수 표 출력"""
    from collections import Counter

    # 위치별 숫자 수집
    position_numbers = [[] for _ in range(6)]

    # Ord 조합
    for combo in ord_predictions:
        for j in range(6):
            position_numbers[j].append(combo.numbers[j])

    # Ball→Ord 조합
    if ball_predictions and actual_data:
        _ball_base = Path(__file__).parent.parent / "5_ball_predict"
        ball_common = load_module("ball_common", _ball_base / "common.py")
        convert_ball_combo_to_ord = ball_common.convert_ball_combo_to_ord

        for combo in ball_predictions:
            ord_combo = convert_ball_combo_to_ord(combo.numbers, actual_data)
            ord_combo_sorted = tuple(sorted(ord_combo))
            for j in range(6):
                position_numbers[j].append(ord_combo_sorted[j])

    # 위치별 빈도수 계산
    position_counters = [Counter(nums) for nums in position_numbers]

    # 표 출력
    print(f"\n{'='*70}")
    print(f"  위치별 숫자 빈도수 (총 {len(ord_predictions) + (len(ball_predictions) if ball_predictions else 0)}개 조합)")
    print(f"{'='*70}")

    for pos in range(6):
        counter = position_counters[pos]
        # 빈도수 높은 순으로 정렬하여 전체 출력
        sorted_items = sorted(counter.items(), key=lambda x: -x[1])
        freq_str = ", ".join(f"{num}({cnt})" for num, cnt in sorted_items)
        print(f"  ord{pos+1}: {freq_str}")


def print_combined_summary(ord_summary: dict, ball_summary: dict = None):
    """Ord + Ball→Ord 통합 일치 분포 요약 출력 (200개 통합)"""
    if not ord_summary or not ord_summary.get('actual'):
        return

    from collections import Counter

    actual = ord_summary['actual']
    ord_match_counts = ord_summary['match_counts']
    ord_best_match = ord_summary['best_match']
    ord_best_combo = ord_summary['best_combo']

    # Ball→Ord 데이터 추출 및 통합
    ball_ord_match_counts = ball_summary.get('ord_match_counts') if ball_summary else None
    ball_ord_best_match = ball_summary.get('best_ord_match') if ball_summary else None
    ball_ord_best_combo = ball_summary.get('best_ord_combo') if ball_summary else None

    # 통합 카운트 및 일치번호 리스트 계산
    combined_counts = {}
    combined_matched_nums = {}
    for m in range(7):
        ord_list = ord_match_counts[m] if ord_match_counts[m] else []
        ball_ord_list = ball_ord_match_counts[m] if ball_ord_match_counts and ball_ord_match_counts[m] else []
        combined_counts[m] = len(ord_list) + len(ball_ord_list)
        combined_matched_nums[m] = ord_list + ball_ord_list

    # 최대 일치 (통합)
    best_match = ord_best_match
    best_combo = ord_best_combo
    if ball_ord_best_match is not None and ball_ord_best_match > best_match:
        best_match = ball_ord_best_match
        best_combo = ball_ord_best_combo

    # 총 조합 수
    total = sum(combined_counts.values())

    print(f"\n{'='*70}")
    print(f"  일치 분포 요약 (총 {total}개 조합)")
    print(f"{'='*70}")
    print(f"  실제 당첨번호: {format_combination(actual)}")
    print(f"  최대 일치: {best_match}개")
    if best_combo:
        if isinstance(best_combo, tuple):
            print(f"  최근접 조합: {format_combination(best_combo)}")
        else:
            print(f"  최근접 조합: {format_combination(best_combo.numbers)}")
    print()

    for m in range(6, -1, -1):
        if combined_counts[m] > 0:
            # 일치 번호 패턴 집계
            nums_counter = Counter(tuple(nums) for nums in combined_matched_nums[m])
            nums_str = ", ".join(f"{list(nums)}" for nums, _ in nums_counter.most_common(5))
            if len(nums_counter) > 5:
                nums_str += f" 외 {len(nums_counter)-5}개"
            print(f"  {m}개 일치: {combined_counts[m]:3d}개 → {nums_str}")


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
            # 실제 당첨번호 확인
            actual_data = next((d for d in all_data if d['round'] == target_round), None)
            actual_ord = tuple(actual_data[f'ord{i}'] for i in range(1, 7)) if actual_data else None

            # Ord 예측
            predictions = predict_for_round(
                target_round, all_data, args.top,
                top_pairs=args.top_pairs, top_inner=args.top_inner,
                verbose=True
            )

            # Ball 예측 (verbose=False로 생성만)
            ball_predictions = None
            actual_ball = None
            if BALL_PREDICT_AVAILABLE:
                ball_predictions = predict_ball_for_round(
                    target_round, all_data, args.top,
                    top_pairs=args.top_pairs, top_inner=args.top_inner,
                    verbose=False
                )
                actual_ball = tuple(actual_data[f'ball{i}'] for i in range(1, 7)) if actual_data else None

            # 1. Ord 100개 조합 출력 (1번부터)
            ord_summary = print_all_combinations(predictions, actual_ord, start_num=1)

            # 2. Ball→Ord 100개 조합 출력 (101번부터)
            ball_summary = None
            if ball_predictions:
                ball_summary = print_ball_combinations(ball_predictions, actual_ball, actual_data, start_num=101)

            # 3. 통합 일치 분포 요약
            print_combined_summary(ord_summary, ball_summary)

            # 4. 위치별 빈도수 표
            print_position_frequency(predictions, ball_predictions, actual_data)

            # 5. CSV 저장
            if actual_ord:
                save_frequency_to_csv(target_round, predictions, ball_predictions, actual_ord, actual_data)

    elif len(args.rounds) == 1:
        # 단일 회차 예측
        target_round = args.rounds[0]
        print(f"\n{target_round}회차 예측 시작...")
        print(f"파라미터: top_pairs={args.top_pairs}, top_inner={args.top_inner}")

        # 실제 당첨번호 확인
        actual_data = next((d for d in all_data if d['round'] == target_round), None)
        actual_ord = tuple(actual_data[f'ord{i}'] for i in range(1, 7)) if actual_data else None

        # Ord 예측
        predictions = predict_for_round(
            target_round, all_data, args.top,
            top_pairs=args.top_pairs, top_inner=args.top_inner,
            verbose=True
        )

        # Ball 예측 (verbose=False로 생성만)
        ball_predictions = None
        actual_ball = None
        if BALL_PREDICT_AVAILABLE:
            ball_predictions = predict_ball_for_round(
                target_round, all_data, args.top,
                top_pairs=args.top_pairs, top_inner=args.top_inner,
                verbose=False
            )
            actual_ball = tuple(actual_data[f'ball{i}'] for i in range(1, 7)) if actual_data else None

        # 1. Ord 100개 조합 출력 (1번부터)
        ord_summary = print_all_combinations(predictions, actual_ord, start_num=1)

        # 2. Ball→Ord 100개 조합 출력 (101번부터)
        ball_summary = None
        if ball_predictions:
            ball_summary = print_ball_combinations(ball_predictions, actual_ball, actual_data, start_num=101)

        # 3. 통합 일치 분포 요약
        print_combined_summary(ord_summary, ball_summary)

        # 4. 위치별 빈도수 표
        print_position_frequency(predictions, ball_predictions, actual_data)

        # 5. CSV 저장
        if actual_ord:
            save_frequency_to_csv(target_round, predictions, ball_predictions, actual_ord, actual_data)

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

        # Ball 예측 (verbose=False로 생성만)
        ball_predictions = None
        if BALL_PREDICT_AVAILABLE:
            ball_predictions = predict_ball_for_round(
                next_round, all_data, args.top,
                top_pairs=args.top_pairs, top_inner=args.top_inner,
                verbose=False
            )

        # 1. Ord 100개 조합 출력 (1번부터)
        print_all_combinations(predictions, None, start_num=1)

        # 2. Ball→Ord 100개 조합 출력 (101번부터)
        if ball_predictions:
            print_ball_combinations(ball_predictions, None, None, start_num=101)


if __name__ == '__main__':
    main()
