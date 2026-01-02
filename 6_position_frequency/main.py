#!/usr/bin/env python3
"""
위치별 빈도수 패턴 기반 예측 모듈

positionf.csv의 ord1~ord6 위치별 빈도 순위 패턴을 분석하여
유사한 조합 100개를 생성합니다.

사용법:
    python 6_position_frequency/main.py           # 분석 및 100개 조합 생성
    python 6_position_frequency/main.py --top 50  # 50개 조합 생성
"""

import csv
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple
from dataclasses import dataclass

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
POSITIONF_PATH = Path(__file__).parent / "positionf.csv"
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"


@dataclass
class PositionStats:
    """위치별 통계"""
    position: int  # 0~5 (ord1~ord6)
    freq_distribution: Dict[int, int]  # 빈도순위 → 출현횟수
    mean_freq: float  # 평균 빈도순위
    std_freq: float  # 표준편차
    percentiles: Dict[int, int]  # 백분위수


def load_positionf() -> List[Dict]:
    """positionf.csv 로드"""
    results = []
    with open(POSITIONF_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = {
                'round': int(row['round']),
                'ord1': int(row['ord1']),
                'ord2': int(row['ord2']),
                'ord3': int(row['ord3']),
                'ord4': int(row['ord4']),
                'ord5': int(row['ord5']),
                'ord6': int(row['ord6']),
            }
            results.append(data)
    return results


def load_winning_numbers() -> Dict[int, List[int]]:
    """당첨번호 데이터 로드 (회차 → [ord1~ord6])"""
    results = {}
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                round_num = int(row['round'])
                nums = [int(row[f'ord{i}']) for i in range(1, 7)]
                if all(1 <= n <= 45 for n in nums):
                    results[round_num] = nums
            except (ValueError, KeyError):
                continue
    return results


def analyze_position_stats(data: List[Dict]) -> List[PositionStats]:
    """각 위치별 빈도순위 통계 분석"""
    stats_list = []

    for pos in range(6):
        col_name = f'ord{pos + 1}'
        freqs = [row[col_name] for row in data]

        # 0 제외 (0은 200개 조합에 없는 경우)
        valid_freqs = [f for f in freqs if f > 0]

        if not valid_freqs:
            continue

        # 빈도 분포
        freq_dist = Counter(valid_freqs)

        # 평균, 표준편차
        mean = sum(valid_freqs) / len(valid_freqs)
        variance = sum((f - mean) ** 2 for f in valid_freqs) / len(valid_freqs)
        std = variance ** 0.5

        # 백분위수 계산
        sorted_freqs = sorted(valid_freqs)
        n = len(sorted_freqs)
        percentiles = {
            10: sorted_freqs[int(n * 0.1)],
            25: sorted_freqs[int(n * 0.25)],
            50: sorted_freqs[int(n * 0.5)],
            75: sorted_freqs[int(n * 0.75)],
            90: sorted_freqs[int(n * 0.9)],
        }

        stats = PositionStats(
            position=pos,
            freq_distribution=dict(freq_dist),
            mean_freq=mean,
            std_freq=std,
            percentiles=percentiles
        )
        stats_list.append(stats)

    return stats_list


def get_optimal_freq_ranges(stats_list: List[PositionStats]) -> List[Tuple[int, int]]:
    """각 위치별 최적 빈도순위 범위 (25~75 백분위)"""
    ranges = []
    for stats in stats_list:
        low = max(1, stats.percentiles[25])
        high = stats.percentiles[75]
        ranges.append((low, high))
    return ranges


def get_freq_weights(stats: PositionStats, max_freq: int = 100) -> Dict[int, float]:
    """빈도순위별 가중치 계산 (출현 빈도 기반)"""
    weights = {}
    total = sum(stats.freq_distribution.values())

    for freq in range(1, max_freq + 1):
        count = stats.freq_distribution.get(freq, 0)
        # 스무딩: 출현하지 않은 빈도에도 최소 가중치 부여
        weights[freq] = (count + 0.1) / (total + max_freq * 0.1)

    return weights


def load_current_frequency(target_round: int) -> Dict[int, List[Tuple[int, int]]]:
    """현재 예측의 위치별 빈도수 로드 (result/{round}_frequency.csv)"""
    freq_path = BASE_DIR / "result" / f"{target_round}_frequency.csv"

    if not freq_path.exists():
        return None

    position_freqs = {i: [] for i in range(6)}

    with open(freq_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for i, col in enumerate(['ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6']):
                val = row[col]
                if val and '(' in val:
                    # "3(51)" 형태 파싱
                    num = int(val.split('(')[0])
                    cnt = int(val.split('(')[1].rstrip(')'))
                    position_freqs[i].append((num, cnt))

    return position_freqs


def generate_combinations_from_pattern(
    stats_list: List[PositionStats],
    current_freq: Dict[int, List[Tuple[int, int]]],
    n_combinations: int = 100
) -> List[Tuple[int, ...]]:
    """패턴 기반 조합 생성"""

    combinations = set()

    # 각 위치별 최적 범위
    optimal_ranges = get_optimal_freq_ranges(stats_list)

    # 각 위치별 가중치
    position_weights = [get_freq_weights(stats) for stats in stats_list]

    # 현재 빈도수에서 상위 숫자들 추출
    top_numbers_by_pos = {}
    if current_freq:
        for pos in range(6):
            # 빈도수 상위 10개 숫자
            top_numbers_by_pos[pos] = [num for num, cnt in current_freq[pos][:10]]

    attempts = 0
    max_attempts = n_combinations * 100

    while len(combinations) < n_combinations and attempts < max_attempts:
        attempts += 1
        combo = []

        for pos in range(6):
            low, high = optimal_ranges[pos]
            weights = position_weights[pos]

            # 현재 빈도수 기반 숫자 선택
            if current_freq and pos in top_numbers_by_pos:
                candidates = top_numbers_by_pos[pos]

                # 빈도순위 범위 내에서 가중치 적용 선택
                target_freq = random.choices(
                    range(low, min(high + 1, len(candidates) + 1)),
                    weights=[weights.get(f, 0.01) for f in range(low, min(high + 1, len(candidates) + 1))]
                )[0]

                # 해당 빈도순위의 숫자 선택
                if target_freq <= len(candidates):
                    num = candidates[target_freq - 1]
                else:
                    num = random.choice(candidates[:5]) if candidates else random.randint(1, 45)
            else:
                # fallback: 랜덤 선택
                num = random.randint(1, 45)

            # 중복 방지
            while num in combo:
                if current_freq and pos in top_numbers_by_pos:
                    candidates = [n for n in top_numbers_by_pos[pos] if n not in combo]
                    if candidates:
                        num = random.choice(candidates)
                    else:
                        available = [n for n in range(1, 46) if n not in combo]
                        num = random.choice(available)
                else:
                    available = [n for n in range(1, 46) if n not in combo]
                    num = random.choice(available)

            combo.append(num)

        # 정렬하여 저장
        combo_sorted = tuple(sorted(combo))

        # 유효성 검사
        if len(set(combo_sorted)) == 6 and all(1 <= n <= 45 for n in combo_sorted):
            combinations.add(combo_sorted)

    return sorted(combinations)


def generate_weighted_combinations(
    stats_list: List[PositionStats],
    current_freq: Dict[int, List[Tuple[int, int]]],
    n_combinations: int = 100
) -> List[Tuple[int, ...]]:
    """가중치 기반 조합 생성 (더 정교한 방법)"""

    combinations = set()

    # 각 위치별 숫자→가중치 매핑
    position_num_weights = {}

    if current_freq:
        for pos in range(6):
            num_weights = {}
            total_cnt = sum(cnt for num, cnt in current_freq[pos])

            for rank, (num, cnt) in enumerate(current_freq[pos], 1):
                # 빈도수 + 순위 기반 가중치
                freq_weight = cnt / total_cnt if total_cnt > 0 else 0.01
                rank_weight = 1.0 / (rank ** 0.5)  # 순위가 낮을수록 높은 가중치
                num_weights[num] = freq_weight * rank_weight

            position_num_weights[pos] = num_weights

    attempts = 0
    max_attempts = n_combinations * 200

    while len(combinations) < n_combinations and attempts < max_attempts:
        attempts += 1
        combo = []
        used_nums = set()

        for pos in range(6):
            if pos in position_num_weights:
                # 아직 사용하지 않은 숫자 중에서 선택
                available = {n: w for n, w in position_num_weights[pos].items() if n not in used_nums}

                if available:
                    nums = list(available.keys())
                    weights = list(available.values())
                    num = random.choices(nums, weights=weights)[0]
                else:
                    # fallback
                    available_all = [n for n in range(1, 46) if n not in used_nums]
                    num = random.choice(available_all)
            else:
                available_all = [n for n in range(1, 46) if n not in used_nums]
                num = random.choice(available_all)

            combo.append(num)
            used_nums.add(num)

        combo_sorted = tuple(sorted(combo))

        if len(set(combo_sorted)) == 6:
            combinations.add(combo_sorted)

    return sorted(combinations)


def print_stats(stats_list: List[PositionStats]):
    """위치별 통계 출력"""
    print("\n" + "=" * 80)
    print("  위치별 빈도순위 통계 분석")
    print("=" * 80)

    print(f"\n  {'위치':<8}{'평균':>10}{'표준편차':>10}{'P25':>8}{'P50':>8}{'P75':>8}{'P90':>8}")
    print("  " + "-" * 70)

    for stats in stats_list:
        print(f"  ord{stats.position + 1:<4}{stats.mean_freq:>10.1f}{stats.std_freq:>10.1f}"
              f"{stats.percentiles[25]:>8}{stats.percentiles[50]:>8}"
              f"{stats.percentiles[75]:>8}{stats.percentiles[90]:>8}")

    print("\n  [해석]")
    print("  - 평균/P50이 낮을수록: 해당 위치의 숫자가 빈도 상위권에서 자주 나옴")
    print("  - 표준편차가 작을수록: 예측 범위가 좁아 신뢰도 높음")


def print_top_patterns(data: List[Dict], n: int = 10):
    """가장 자주 나오는 빈도순위 패턴 출력"""
    print("\n" + "=" * 80)
    print("  상위 빈도순위 패턴 (최근 회차 기준)")
    print("=" * 80)

    # 최근 100회차만 분석
    recent = data[-100:] if len(data) > 100 else data

    for pos in range(6):
        col_name = f'ord{pos + 1}'
        freqs = [row[col_name] for row in recent if row[col_name] > 0]
        freq_counter = Counter(freqs)

        top5 = freq_counter.most_common(5)
        top5_str = ", ".join(f"{f}위({c}회)" for f, c in top5)
        print(f"  ord{pos + 1}: {top5_str}")


def print_combinations(combinations: List[Tuple[int, ...]], title: str = "생성된 조합"):
    """조합 출력"""
    print("\n" + "=" * 80)
    print(f"  {title} ({len(combinations)}개)")
    print("=" * 80)

    for i, combo in enumerate(combinations, 1):
        print(f"  {i:3d}. ({combo[0]:2d}, {combo[1]:2d}, {combo[2]:2d}, "
              f"{combo[3]:2d}, {combo[4]:2d}, {combo[5]:2d})")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='위치별 빈도수 패턴 기반 예측')
    parser.add_argument('--top', type=int, default=100, help='생성할 조합 수')
    parser.add_argument('--round', type=int, default=None, help='대상 회차')
    args = parser.parse_args()

    print("=" * 80)
    print("  위치별 빈도수 패턴 기반 예측 시스템")
    print("=" * 80)

    # 1. positionf.csv 로드
    print("\n1. positionf.csv 로드 중...")
    positionf_data = load_positionf()
    print(f"  → {len(positionf_data)}개 회차 로드 완료")

    # 2. 통계 분석
    print("\n2. 위치별 통계 분석 중...")
    stats_list = analyze_position_stats(positionf_data)
    print_stats(stats_list)

    # 3. 상위 패턴 출력
    print_top_patterns(positionf_data)

    # 4. 현재 빈도수 로드 (있는 경우)
    target_round = args.round
    if target_round is None:
        # 마지막 회차 + 1
        winning = load_winning_numbers()
        target_round = max(winning.keys()) + 1

    print(f"\n3. {target_round}회차 빈도수 로드 중...")
    current_freq = load_current_frequency(target_round)

    if current_freq:
        print(f"  → result/{target_round}_frequency.csv 로드 완료")
    else:
        print(f"  → result/{target_round}_frequency.csv 없음 (기본 패턴 사용)")

    # 5. 조합 생성
    print(f"\n4. 패턴 기반 {args.top}개 조합 생성 중...")

    # 방법 1: 패턴 기반
    combinations1 = generate_combinations_from_pattern(
        stats_list, current_freq, args.top // 2
    )

    # 방법 2: 가중치 기반
    combinations2 = generate_weighted_combinations(
        stats_list, current_freq, args.top // 2
    )

    # 통합 (중복 제거)
    all_combinations = list(set(combinations1 + combinations2))
    all_combinations = sorted(all_combinations)[:args.top]

    print_combinations(all_combinations, f"Position Frequency 기반 예측")

    # 6. 요약
    print("\n" + "=" * 80)
    print("  요약")
    print("=" * 80)
    print(f"  - 분석 데이터: {len(positionf_data)}개 회차")
    print(f"  - 생성 조합: {len(all_combinations)}개")
    print(f"  - 대상 회차: {target_round}회")


if __name__ == '__main__':
    main()
