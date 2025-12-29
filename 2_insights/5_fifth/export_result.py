"""
fifth 결과 내보내기 - ord5 추가

실행: python export_result.py
입력: result/result.csv (ord1, ord2, ord3, ord4, ord6 있음)
출력: result/result.csv (ord5 추가, 복수 추천시 행 복제)

ord5 선택 전략:
1. ★ Sum이 가장 중요 (121-160 범위 = 48.5%)
2. Shortcode 반영 (Top24/Mid14/Rest7 패턴)
3. 원핫인코딩 (ord5 포지션 비트 빈도, 핫/콜드 비트)
4. 5자리/6자리가 기존 당첨번호와 동일하면 무조건 제외
"""

import csv
from pathlib import Path
from itertools import combinations

RESULT_PATH = Path(__file__).parent.parent.parent / "result" / "result.csv"
DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"

# ball5 빈도 통계 (내장)
BALL5_STATS = {
    33: {'freq': 26, 'rank': 1},
    34: {'freq': 24, 'rank': 2},
    38: {'freq': 23, 'rank': 3},
    37: {'freq': 22, 'rank': 4},
    39: {'freq': 21, 'rank': 5},
    31: {'freq': 20, 'rank': 6},
    36: {'freq': 20, 'rank': 7},
    35: {'freq': 19, 'rank': 8},
    32: {'freq': 18, 'rank': 9},
    28: {'freq': 18, 'rank': 10},
    40: {'freq': 17, 'rank': 11},
    29: {'freq': 17, 'rank': 12},
    30: {'freq': 16, 'rank': 13},
    27: {'freq': 15, 'rank': 14},
    42: {'freq': 14, 'rank': 15},
    26: {'freq': 13, 'rank': 16},
    41: {'freq': 13, 'rank': 17},
    25: {'freq': 12, 'rank': 18},
    24: {'freq': 11, 'rank': 19},
    43: {'freq': 10, 'rank': 20},
    23: {'freq': 9, 'rank': 21},
    22: {'freq': 8, 'rank': 22},
    21: {'freq': 7, 'rank': 23},
    19: {'freq': 6, 'rank': 24},
    20: {'freq': 5, 'rank': 25},
    18: {'freq': 4, 'rank': 26},
    17: {'freq': 3, 'rank': 27},
    16: {'freq': 2, 'rank': 28},
    44: {'freq': 2, 'rank': 29},
}

# ball4-ball5 간격 통계 (내장)
GAP45_STATS = {
    1: {'freq': 56},   # 연속수 14.8%
    2: {'freq': 51},   # 13.5%
    3: {'freq': 42},   # 11.1%
    4: {'freq': 35},
    5: {'freq': 33},
    6: {'freq': 29},
    7: {'freq': 26},
    8: {'freq': 19},
    9: {'freq': 18},
    10: {'freq': 14},
    11: {'freq': 12},
    12: {'freq': 10},
    13: {'freq': 9},
    14: {'freq': 8},
    15: {'freq': 5},
    16: {'freq': 4},
    17: {'freq': 3},
}

# 원핫인코딩 - ord5 포지션 비트 빈도
ORD5_BIT_FREQ = {
    33: 26,  # 1위
    34: 24,  # 2위
    38: 23,  # 3위
    37: 22,  # 4위
    39: 21,  # 5위
    31: 20, 36: 20,  # 공동 6위
    35: 19,
    32: 18, 28: 18,
    40: 17, 29: 17,
    30: 16,
    27: 15,
    42: 14,
    26: 13, 41: 13,
    25: 12,
    24: 11,
    43: 10,
    23: 9,
    22: 8,
    21: 7,
    19: 6,
    20: 5,
}

# 원핫인코딩 - 핫/콜드 비트 (전역)
HOT_BITS = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31, 37, 33, 10, 2, 32}
COLD_BITS = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7, 30, 15, 11, 40, 6}

# Range 구간 정의
def get_range(num):
    """번호의 구간 반환 (0-4)"""
    if num <= 9: return 0
    elif num <= 19: return 1
    elif num <= 29: return 2
    elif num <= 39: return 3
    else: return 4

# ord5 포지션별 구간 분포 (position_range_distribution.csv)
# 구간3(30-39): 54.6%, 구간2(20-29): 25.9%, 구간4(40-45): 16.1%
ORD5_RANGE_FREQ = {
    3: 54.6,  # 30-39: 최빈
    2: 25.9,  # 20-29: 차빈
    4: 16.1,  # 40-45: 가능
    1: 3.4,   # 10-19: 드물
    0: 0.0,   # 01-09: 없음
}

# Sum 최적 범위 (121-160 = 48.5%)
SUM_OPTIMAL_MIN = 121
SUM_OPTIMAL_MAX = 160
SUM_GOOD_MIN = 100
SUM_GOOD_MAX = 170  # 73.6%

# Shortcode - Top24/Mid14/Rest7 세그먼트
# 평균: Top24=3.22, Mid14=1.86, Rest7=0.92
# 최빈 패턴: 321 (Top24에서 3, Mid14에서 2, Rest7에서 1)
TOP24_THRESHOLD = 24  # 빈도 순위 1-24
MID14_THRESHOLD = 38  # 빈도 순위 25-38
# REST7: 순위 39-45

# 확장 헤더
FIELDNAMES = [
    'ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6',
    '분류', '빈도',
    'range_code', 'unique_ranges', 'consecutive', 'hot_count', 'cold_count', 'score'
]


def load_winning_numbers():
    """기존 당첨번호 로드 (중복 체크용)"""
    winning_sets = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = tuple(sorted([
                int(row['ball1']), int(row['ball2']), int(row['ball3']),
                int(row['ball4']), int(row['ball5']), int(row['ball6'])
            ]))
            winning_sets.append(balls)
    return winning_sets


def check_duplicate(numbers, winning_sets, min_match=5):
    """기존 당첨번호와 min_match개 이상 일치하면 True 반환"""
    num_set = set(numbers)
    for winning in winning_sets:
        match_count = len(num_set & set(winning))
        if match_count >= min_match:
            return True
    return False


def get_shortcode_segment(num, num_to_rank):
    """번호의 shortcode 세그먼트 반환 (Top24/Mid14/Rest7)"""
    rank = num_to_rank.get(num, 45)  # 없으면 최하위
    if rank <= TOP24_THRESHOLD:
        return 'top24'
    elif rank <= MID14_THRESHOLD:
        return 'mid14'
    else:
        return 'rest7'


def get_shortcode_score(numbers, num_to_rank):
    """현재 조합의 shortcode 패턴에 대한 점수"""
    segments = {'top24': 0, 'mid14': 0, 'rest7': 0}
    for num in numbers:
        seg = get_shortcode_segment(num, num_to_rank)
        segments[seg] += 1

    # 최적 패턴: Top24에서 3-4개, Mid14에서 1-2개, Rest7에서 0-1개
    top_cnt = segments['top24']
    mid_cnt = segments['mid14']
    rest_cnt = segments['rest7']

    score = 0

    # Top24에서 3-4개가 최적 (평균 3.22)
    if 3 <= top_cnt <= 4:
        score += 20
    elif top_cnt == 2 or top_cnt == 5:
        score += 10

    # Mid14에서 1-2개가 최적 (평균 1.86)
    if 1 <= mid_cnt <= 2:
        score += 15
    elif mid_cnt == 3:
        score += 8

    # Rest7에서 0-1개가 최적 (평균 0.92)
    if rest_cnt <= 1:
        score += 10
    elif rest_cnt == 2:
        score += 5

    return score


def build_num_to_rank():
    """번호 -> 빈도 순위 매핑 (간단히 ball5 통계 기반)"""
    # 실제로는 전체 빈도 통계가 필요하지만, 간단히 ball5 기준으로 근사
    # 더 정확한 구현 위해서는 별도 통계 파일 필요
    num_to_rank = {}
    for num, stats in BALL5_STATS.items():
        num_to_rank[num] = stats['rank']
    # 나머지 번호는 중간 순위
    for num in range(1, 46):
        if num not in num_to_rank:
            num_to_rank[num] = 35  # 하위권으로 추정
    return num_to_rank


def select_ord5_candidates(ord1, ord2, ord3, ord4, ord6, winning_sets, num_to_rank, max_candidates=2):
    """주어진 (ord4, ord6)에 대해 ord5 후보 선택"""

    # ord5 가능 범위: ord4+1 ~ ord6-1
    min_ord5 = ord4 + 1
    max_ord5 = ord6 - 1

    if min_ord5 > max_ord5:
        return []

    candidates = []
    current_sum = ord1 + ord2 + ord3 + ord4 + ord6
    current_numbers = [ord1, ord2, ord3, ord4, ord6]

    for ord5 in range(min_ord5, max_ord5 + 1):
        # === 5자리/6자리 중복 체크 (무조건 제외) ===
        full_combo = sorted([ord1, ord2, ord3, ord4, ord5, ord6])

        # 6자리 완전 일치 체크
        if check_duplicate(full_combo, winning_sets, min_match=6):
            continue  # 6개 일치 - 제외

        # 5자리 일치 체크
        if check_duplicate(full_combo, winning_sets, min_match=5):
            continue  # 5개 일치 - 제외

        score = 0

        # === 1. Sum이 가장 중요 ★★★ ===
        total_sum = current_sum + ord5

        # 최적 범위 (121-160): 48.5%
        if SUM_OPTIMAL_MIN <= total_sum <= SUM_OPTIMAL_MAX:
            score += 50  # 최고 가중치
            # 중앙에 가까울수록 보너스
            mid = (SUM_OPTIMAL_MIN + SUM_OPTIMAL_MAX) / 2  # 140.5
            dist = abs(total_sum - mid)
            score += max(0, 20 - int(dist / 2))  # 최대 20점 추가
        # 양호 범위 (100-170): 73.6%
        elif SUM_GOOD_MIN <= total_sum <= SUM_GOOD_MAX:
            score += 25
        # 범위 밖
        else:
            score -= 30  # 큰 감점

        # === 2. Shortcode 반영 ===
        shortcode_score = get_shortcode_score(full_combo, num_to_rank)
        score += shortcode_score

        # === 3. ball5 빈도 점수 ===
        if ord5 in BALL5_STATS:
            freq = BALL5_STATS[ord5]['freq']
            score += freq  # 최대 26점

        # === 4. ball4-ball5 간격 점수 ===
        gap = ord5 - ord4
        if gap in GAP45_STATS:
            score += GAP45_STATS[gap]['freq'] // 4  # 최대 14점

        # === 5. 연속수 패턴 (14.8%) ===
        if gap == 1:
            score += 8

        # === 6. 원핫인코딩 - ord5 포지션 비트 빈도 ===
        if ord5 in ORD5_BIT_FREQ:
            score += ORD5_BIT_FREQ[ord5] // 2  # 최대 13점

        # === 7. 원핫인코딩 - 핫/콜드 비트 ===
        if ord5 in HOT_BITS:
            score += 10  # 핫 비트 보너스
        elif ord5 in COLD_BITS:
            score -= 3   # 콜드 비트 약간 감점

        # === 8. Range 점수 ===
        ord5_range = get_range(ord5)
        range_freq = ORD5_RANGE_FREQ.get(ord5_range, 0)
        score += int(range_freq * 0.3)  # 최대 16점

        # === 9. 구간 다양성 보너스 (4개 구간 = 51.2%) ===
        ranges = {get_range(ord1), get_range(ord2), get_range(ord3),
                  get_range(ord4), get_range(ord5), get_range(ord6)}
        if len(ranges) == 4:
            score += 10  # 최빈
        elif len(ranges) == 3:
            score += 5   # 두 번째
        elif len(ranges) == 5:
            score += 3   # 세 번째

        # === 10. 소수 보너스 (22.4%) ===
        if ord5 in {17, 19, 23, 29, 31, 37, 41, 43}:
            score += 3

        candidates.append({
            'ord5': ord5,
            'score': score,
            'sum': total_sum,
        })

    # 점수순 정렬 후 상위 N개 선택
    candidates.sort(key=lambda x: -x['score'])
    return candidates[:max_candidates]


def process_result():
    """result.csv 읽어서 ord5 추가"""

    # 기존 당첨번호 로드
    winning_sets = load_winning_numbers()
    print(f"기존 당첨번호 로드: {len(winning_sets)}회차")

    # 번호 -> 순위 매핑
    num_to_rank = build_num_to_rank()

    # 기존 result.csv 읽기
    rows = []
    with open(RESULT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"기존 행 수: {len(rows)}")

    # ord5 추가하여 새 행 생성
    new_rows = []
    expanded_count = 0
    excluded_count = 0

    for row in rows:
        ord1 = int(row['ord1'])
        ord2 = int(row['ord2']) if row['ord2'] else None
        ord3 = int(row['ord3']) if row['ord3'] else None
        ord4 = int(row['ord4']) if row['ord4'] else None
        ord6 = int(row['ord6'])

        if ord4 is None:
            new_rows.append(row)
            continue

        # ord5 후보 선택
        candidates = select_ord5_candidates(
            ord1, ord2, ord3, ord4, ord6,
            winning_sets, num_to_rank,
            max_candidates=2
        )

        def update_row_info(new_row, ord5_val, score_val):
            """행에 추가 정보 업데이트"""
            new_row['ord5'] = ord5_val
            # range_code 완성
            r1 = get_range(ord1)
            r2 = get_range(ord2) if ord2 else '-'
            r3 = get_range(ord3) if ord3 else '-'
            r4 = get_range(ord4)
            r5 = get_range(ord5_val)
            r6 = get_range(ord6)
            new_row['range_code'] = f"{r1}{r2}{r3}{r4}{r5}{r6}"

            # unique_ranges 계산
            ranges = {r1, r6}
            if ord2: ranges.add(r2)
            if ord3: ranges.add(r3)
            ranges.add(r4)
            ranges.add(r5)
            new_row['unique_ranges'] = len(ranges)

            # hot/cold 카운트
            nums = [ord1, ord2, ord3, ord4, ord5_val, ord6]
            nums = [x for x in nums if x]
            new_row['hot_count'] = sum(1 for x in nums if x in HOT_BITS)
            new_row['cold_count'] = sum(1 for x in nums if x in COLD_BITS)

            # 연속수 카운트
            consec = 0
            if ord2 and ord2 - ord1 == 1: consec += 1
            if ord2 and ord3 and ord3 - ord2 == 1: consec += 1
            if ord3 and ord4 - ord3 == 1: consec += 1
            if ord5_val - ord4 == 1: consec += 1
            if ord6 - ord5_val == 1: consec += 1
            new_row['consecutive'] = consec
            new_row['score'] = score_val
            return new_row

        if not candidates:
            # 가능한 ord5가 없거나 모두 제외됨
            excluded_count += 1
            continue  # 이 행은 버림
        elif len(candidates) == 1:
            new_row = row.copy()
            new_row = update_row_info(new_row, candidates[0]['ord5'], candidates[0]['score'])
            new_rows.append(new_row)
        else:
            for cand in candidates:
                new_row = row.copy()
                new_row = update_row_info(new_row, cand['ord5'], cand['score'])
                new_rows.append(new_row)
            expanded_count += len(candidates) - 1

    print(f"새 행 수: {len(new_rows)} (복제로 {expanded_count}개 추가, 중복제외 {excluded_count}개)")

    # 저장
    with open(RESULT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"저장 완료: {RESULT_PATH}")

    # 샘플 출력
    print("\n[샘플 출력 (상위 10개)]")
    for row in new_rows[:10]:
        total_sum = int(row['ord1']) + int(row['ord2']) + int(row['ord3']) + \
                    int(row['ord4']) + int(row['ord5']) + int(row['ord6'])
        print(f"  ({row['ord1']},{row['ord2']},{row['ord3']},{row['ord4']},{row['ord5']},{row['ord6']}) "
              f"sum={total_sum} score={row['score']} - {row['분류']}")


if __name__ == "__main__":
    process_result()
