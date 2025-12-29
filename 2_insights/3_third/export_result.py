"""
third 결과 내보내기 - ord3 추가

실행: python export_result.py
입력: result/result.csv (ord1, ord2, ord6 있음)
출력: result/result.csv (ord3 추가, 복수 추천시 행 복제)

ord3 선택 조건:
1. ord2 < ord3 < ord6 - 3 (ord4,5 자리 확보)
2. ball2-ball3 간격 통계 기반 점수화
3. 원핫인코딩 반영
4. range 반영 (80% 잘 나오는 구간, 20% 나올 가능성 있는 구간)
"""

import csv
from pathlib import Path

RESULT_PATH = Path(__file__).parent.parent.parent / "result" / "result.csv"

# ball3 빈도 통계 (내장)
BALL3_STATS = {
    12: {'freq': 22, 'rank': 1},
    16: {'freq': 20, 'rank': 2},
    17: {'freq': 20, 'rank': 3},
    13: {'freq': 19, 'rank': 4},
    19: {'freq': 19, 'rank': 5},
    18: {'freq': 19, 'rank': 6},
    15: {'freq': 19, 'rank': 7},
    22: {'freq': 18, 'rank': 8},
    24: {'freq': 16, 'rank': 9},
    28: {'freq': 16, 'rank': 10},
    20: {'freq': 16, 'rank': 11},
    11: {'freq': 15, 'rank': 12},
    30: {'freq': 14, 'rank': 13},
    29: {'freq': 14, 'rank': 14},
    23: {'freq': 14, 'rank': 15},
    25: {'freq': 13, 'rank': 16},
    26: {'freq': 12, 'rank': 17},
    21: {'freq': 12, 'rank': 18},
    14: {'freq': 11, 'rank': 19},
    31: {'freq': 10, 'rank': 20},
    9: {'freq': 9, 'rank': 21},
    32: {'freq': 8, 'rank': 22},
    10: {'freq': 8, 'rank': 23},
    27: {'freq': 8, 'rank': 24},
    34: {'freq': 5, 'rank': 25},
    35: {'freq': 3, 'rank': 26},
    8: {'freq': 3, 'rank': 27},
    33: {'freq': 3, 'rank': 28},
    6: {'freq': 3, 'rank': 29},
    37: {'freq': 2, 'rank': 30},
    4: {'freq': 2, 'rank': 31},
    7: {'freq': 2, 'rank': 32},
}

# ball2-ball3 간격 통계 (내장)
GAP23_STATS = {
    1: {'freq': 41},   # 연속수 10.8%
    2: {'freq': 46},   # 최빈 간격 12.1%
    3: {'freq': 36},
    4: {'freq': 22},
    5: {'freq': 39},   # 10.3%
    6: {'freq': 25},
    7: {'freq': 29},
    8: {'freq': 22},
    9: {'freq': 21},
    10: {'freq': 15},
    11: {'freq': 12},
    12: {'freq': 13},
    13: {'freq': 10},
    14: {'freq': 11},
    15: {'freq': 9},
}

# (ball1, ball2) 조합별 ball3 최빈값 (상위 조합만)
PAIR12_BALL3_MODE = {
    (3, 6): 17, (3, 7): 9, (2, 6): 11, (6, 7): 12, (3, 4): 12,
    (1, 9): 12, (1, 4): 13, (1, 3): 24, (7, 9): 24, (4, 7): 13,
    (1, 2): 11, (4, 8): 18, (3, 13): 29, (5, 12): 25, (6, 14): 16,
    (7, 11): 12, (10, 16): 19, (1, 6): 13,
}

# 원핫인코딩 - ord3 포지션 비트 빈도
ORD3_BIT_FREQ = {
    16: 23,  # 1위
    13: 20,  # 2위
    21: 19, 19: 19,  # 공동 3위
    25: 18, 24: 18, 18: 18, 17: 18,  # 공동 5위
    20: 17,
    23: 16, 22: 16,
    27: 15, 15: 15,
    29: 14, 11: 14,
    12: 13,
    10: 12,
    26: 11, 14: 11,
    31: 9,
    9: 8, 7: 8,
    30: 7, 28: 7,
    32: 6,
}

# 원핫인코딩 - 핫/콜드 비트
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

# ord3 포지션별 구간 분포 (position_range_distribution.csv)
# 80% 잘 나오는 구간: 구간1(45.4%), 구간2(36.7%) = 82.1%
# 20% 나올 가능성: 구간0(5.0%), 구간3(12.7%), 구간4(0.3%)
ORD3_RANGE_FREQ = {
    1: 45.4,  # 10-19: 최빈
    2: 36.7,  # 20-29: 차빈
    0: 5.0,   # 01-09: 가능
    3: 12.7,  # 30-39: 가능
    4: 0.3,   # 40-45: 희귀
}

# 4개 구간 사용이 51.2%로 최빈 → 새로운 구간 추가시 보너스
UNIQUE_RANGE_BONUS = {
    3: 10,  # 3개 구간 → 4개 되면 좋음
    4: 5,   # 4개 구간 유지도 좋음
    5: 3,   # 5개 구간도 가능
}

# 확장 헤더
FIELDNAMES = [
    'ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6',
    '분류', '빈도',
    'range_code', 'unique_ranges', 'consecutive', 'hot_count', 'cold_count', 'score'
]


def select_ord3_candidates(ord1, ord2, ord6, max_candidates=2):
    """주어진 (ord1, ord2, ord6)에 대해 ord3 후보 선택"""

    # ord3 가능 범위: ord2+1 ~ ord6-3 (ord4,5 자리 확보)
    min_ord3 = ord2 + 1
    max_ord3 = ord6 - 3

    if min_ord3 > max_ord3:
        return []

    candidates = []

    for ord3 in range(min_ord3, max_ord3 + 1):
        score = 0

        # 1. ball3 빈도 점수
        if ord3 in BALL3_STATS:
            score += BALL3_STATS[ord3]['freq'] * 2

        # 2. ball2-ball3 간격 점수
        gap = ord3 - ord2
        if gap in GAP23_STATS:
            score += GAP23_STATS[gap]['freq'] // 2

        # 3. (ball1, ball2) 조합별 ball3 최빈값 보너스
        if (ord1, ord2) in PAIR12_BALL3_MODE:
            if ord3 == PAIR12_BALL3_MODE[(ord1, ord2)]:
                score += 20

        # 4. 최빈 구간 보너스 (10-19: 45.4%, 20-29: 36.7%)
        if 10 <= ord3 <= 19:
            score += 12
        elif 20 <= ord3 <= 29:
            score += 8

        # 5. 연속수 패턴 (10.8%)
        if gap == 1:
            score += 5

        # 6. 원핫인코딩 - ord3 포지션 비트 빈도
        if ord3 in ORD3_BIT_FREQ:
            score += ORD3_BIT_FREQ[ord3]  # 최대 23점

        # 7. 원핫인코딩 - 핫/콜드 비트
        if ord3 in HOT_BITS:
            score += 10
        elif ord3 in COLD_BITS:
            score -= 5

        # === Range 반영 (80/20 전략) ===
        ord3_range = get_range(ord3)
        ord1_range = get_range(ord1)
        ord2_range = get_range(ord2)
        ord6_range = get_range(ord6)

        # 8. ord3 구간별 빈도 점수 (80% 잘 나오는 구간)
        range_freq = ORD3_RANGE_FREQ.get(ord3_range, 0)
        if range_freq >= 30:  # 구간1, 구간2 (82.1%)
            score += int(range_freq * 0.3)  # 최대 13점
        elif range_freq >= 5:  # 구간0, 구간3 (17.7%)
            # 20% 나올 가능성 있는 구간 - 약간의 보너스
            score += int(range_freq * 0.5)  # 최대 6점

        # 9. 새로운 구간 추가 보너스 (4개 구간 = 51.2%)
        existing_ranges = {ord1_range, ord2_range, ord6_range}
        if ord3_range not in existing_ranges:
            new_unique_count = len(existing_ranges) + 1
            if new_unique_count in UNIQUE_RANGE_BONUS:
                score += UNIQUE_RANGE_BONUS[new_unique_count]

        # 10. 연속 구간 패턴 (57.5% 연속 구간 사용)
        # ord2와 ord3가 같은 구간이거나 인접 구간이면 보너스
        if abs(ord3_range - ord2_range) <= 1:
            score += 5

        candidates.append({
            'ord3': ord3,
            'score': score,
        })

    candidates.sort(key=lambda x: -x['score'])
    return candidates[:max_candidates]


def process_result():
    """result.csv 읽어서 ord3 추가"""

    # 기존 result.csv 읽기
    rows = []
    with open(RESULT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"기존 행 수: {len(rows)}")

    # ord3 추가하여 새 행 생성
    new_rows = []
    expanded_count = 0

    for row in rows:
        ord1 = int(row['ord1'])
        ord2 = int(row['ord2']) if row['ord2'] else None
        ord6 = int(row['ord6'])

        if ord2 is None:
            new_rows.append(row)
            continue

        # ord3 후보 선택
        candidates = select_ord3_candidates(ord1, ord2, ord6, max_candidates=2)

        def update_row_info(new_row, ord3_val, score_val):
            """행에 추가 정보 업데이트"""
            new_row['ord3'] = ord3_val
            # range_code 업데이트
            r1, r2, r3, r6 = get_range(ord1), get_range(ord2), get_range(ord3_val), get_range(ord6)
            new_row['range_code'] = f"{r1}{r2}{r3}--{r6}"
            new_row['unique_ranges'] = len({r1, r2, r3, r6})
            # hot/cold 카운트
            nums = [ord1, ord2, ord3_val, ord6]
            new_row['hot_count'] = sum(1 for x in nums if x in HOT_BITS)
            new_row['cold_count'] = sum(1 for x in nums if x in COLD_BITS)
            # 연속수 카운트
            consec = 0
            if ord2 - ord1 == 1: consec += 1
            if ord3_val - ord2 == 1: consec += 1
            new_row['consecutive'] = consec
            new_row['score'] = score_val
            return new_row

        if not candidates:
            new_rows.append(row)
        elif len(candidates) == 1:
            new_row = row.copy()
            new_row = update_row_info(new_row, candidates[0]['ord3'], candidates[0]['score'])
            new_rows.append(new_row)
        else:
            for cand in candidates:
                new_row = row.copy()
                new_row = update_row_info(new_row, cand['ord3'], cand['score'])
                new_rows.append(new_row)
            expanded_count += len(candidates) - 1

    print(f"새 행 수: {len(new_rows)} (복제로 {expanded_count}개 추가)")

    # 저장
    with open(RESULT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"저장 완료: {RESULT_PATH}")

    # 샘플 출력
    print("\n[샘플 출력 (상위 10개)]")
    for row in new_rows[:10]:
        print(f"  ({row['ord1']}, {row['ord2']}, {row['ord3']}, ..., {row['ord6']}) - {row['분류']}")


if __name__ == "__main__":
    process_result()
