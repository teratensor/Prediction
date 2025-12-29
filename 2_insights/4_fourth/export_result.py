"""
fourth 결과 내보내기 - ord4 추가 (뜻밖의 수 중심)

실행: python export_result.py
입력: result/result.csv (ord1, ord2, ord3, ord6 있음)
출력: result/result.csv (ord4 추가, 복수 추천시 행 복제)

ord4 선택 전략 - "뜻밖의 수":
1. 기본: ord3 < ord4 < ord6 - 1 (ord5 자리 확보)
2. 빈도 하위 번호에 가중치 부여 (역발상)
3. 최근 콜드 트렌드 번호 보너스
4. 원핫인코딩 콜드 비트에 역가산
5. range 반영 (80% 잘 나오는 구간, 20% 나올 가능성 있는 구간)
"""

import csv
from pathlib import Path

RESULT_PATH = Path(__file__).parent.parent.parent / "result" / "result.csv"

# ball4 빈도 통계 (내장) - 순위 역순 가중치 부여용
BALL4_STATS = {
    30: {'freq': 26, 'rank': 1},
    26: {'freq': 19, 'rank': 2},
    29: {'freq': 18, 'rank': 3},
    35: {'freq': 18, 'rank': 4},
    27: {'freq': 18, 'rank': 5},
    23: {'freq': 18, 'rank': 6},
    33: {'freq': 17, 'rank': 7},
    28: {'freq': 16, 'rank': 8},
    22: {'freq': 16, 'rank': 9},
    32: {'freq': 16, 'rank': 10},
    31: {'freq': 16, 'rank': 11},
    21: {'freq': 15, 'rank': 12},
    24: {'freq': 15, 'rank': 13},
    25: {'freq': 14, 'rank': 14},
    19: {'freq': 14, 'rank': 15},
    20: {'freq': 13, 'rank': 16},
    38: {'freq': 12, 'rank': 17},
    18: {'freq': 11, 'rank': 18},
    16: {'freq': 11, 'rank': 19},
    14: {'freq': 10, 'rank': 20},
    37: {'freq': 9, 'rank': 21},
    15: {'freq': 8, 'rank': 22},
    36: {'freq': 7, 'rank': 23},
    13: {'freq': 7, 'rank': 24},
    34: {'freq': 7, 'rank': 25},
    17: {'freq': 7, 'rank': 26},
    40: {'freq': 5, 'rank': 27},
    39: {'freq': 4, 'rank': 28},
    41: {'freq': 4, 'rank': 29},
    11: {'freq': 3, 'rank': 30},
    12: {'freq': 2, 'rank': 31},
    43: {'freq': 1, 'rank': 32},
    42: {'freq': 1, 'rank': 33},
    8: {'freq': 1, 'rank': 34},
}

# ball3-ball4 간격 통계 (내장)
GAP34_STATS = {
    1: {'freq': 46},   # 연속수 12.1%
    2: {'freq': 54},   # 최빈 간격 14.2%
    3: {'freq': 44},   # 11.6%
    4: {'freq': 37},
    5: {'freq': 24},
    6: {'freq': 30},
    7: {'freq': 25},
    8: {'freq': 22},
    9: {'freq': 18},
    10: {'freq': 18},
    11: {'freq': 9},
    12: {'freq': 7},
    13: {'freq': 7},
    14: {'freq': 4},
    15: {'freq': 4},
    16: {'freq': 9},
    17: {'freq': 6},
    18: {'freq': 4},
    19: {'freq': 5},
}

# 원핫인코딩 - ord4 포지션 비트 빈도
ORD4_BIT_FREQ = {
    27: 23,  # 1위
    28: 22,  # 2위
    25: 20,  # 3위
    24: 19, 29: 19,  # 공동 4위
    31: 18,
    21: 17, 17: 17, 32: 17, 33: 17,
    23: 16,
    19: 15, 26: 15, 30: 15, 35: 15,
    22: 12, 18: 12,
    20: 10,
    16: 9,
    37: 8, 34: 8,
    36: 7, 38: 7,
    40: 6,
}

# 원핫인코딩 - 핫/콜드 비트 (전역)
HOT_BITS = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31, 37, 33, 10, 2, 32}
COLD_BITS = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7, 30, 15, 11, 40, 6}

# 뜻밖의 수 후보 - 빈도 하위 그룹 (rank 20~34)
UNEXPECTED_NUMBERS = {14, 37, 15, 36, 13, 34, 17, 40, 39, 41, 11, 12, 43, 42, 8}

# Range 구간 정의
def get_range(num):
    """번호의 구간 반환 (0-4)"""
    if num <= 9: return 0
    elif num <= 19: return 1
    elif num <= 29: return 2
    elif num <= 39: return 3
    else: return 4

# ord4 포지션별 구간 분포 (position_range_distribution.csv)
# 80% 잘 나오는 구간: 구간2(42.7%), 구간3(34.8%) = 77.5%
# 20% 나올 가능성: 구간1(19.3%), 구간4(2.9%), 구간0(0.3%)
ORD4_RANGE_FREQ = {
    2: 42.7,  # 20-29: 최빈
    3: 34.8,  # 30-39: 차빈
    1: 19.3,  # 10-19: 가능
    4: 2.9,   # 40-45: 드물
    0: 0.3,   # 01-09: 희귀
}

# 4개 구간 사용이 51.2%로 최빈 → 새로운 구간 추가시 보너스
UNIQUE_RANGE_BONUS = {
    3: 8,   # 3개 구간 → 4개 되면 좋음
    4: 5,   # 4개 구간 유지
    5: 3,   # 5개 구간도 가능
}

# 확장 헤더
FIELDNAMES = [
    'ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6',
    '분류', '빈도',
    'range_code', 'unique_ranges', 'consecutive', 'hot_count', 'cold_count', 'score'
]


def select_ord4_candidates(ord1, ord2, ord3, ord6, max_candidates=2):
    """주어진 (ord3, ord6)에 대해 ord4 후보 선택 - 뜻밖의 수 중심"""

    # ord4 가능 범위: ord3+1 ~ ord6-1 (ord5 자리 확보)
    min_ord4 = ord3 + 1
    max_ord4 = ord6 - 1

    if min_ord4 > max_ord4:
        return []

    candidates = []

    for ord4 in range(min_ord4, max_ord4 + 1):
        score = 0

        # === 뜻밖의 수 전략 ===

        # 1. 빈도 역순 점수 (빈도 낮을수록 높은 점수)
        if ord4 in BALL4_STATS:
            rank = BALL4_STATS[ord4]['rank']
            # 순위 20 이상이면 역발상 보너스
            if rank >= 20:
                score += (rank - 19) * 3  # 최대 45점 (rank 34)
            else:
                # 상위 번호도 기본 점수는 줌
                score += BALL4_STATS[ord4]['freq']

        # 2. 뜻밖의 수 그룹 보너스
        if ord4 in UNEXPECTED_NUMBERS:
            score += 15

        # 3. 최빈 구간도 고려 (완전히 무시하면 안됨)
        # 20-29: 42.7%, 30-39: 34.8%
        if 20 <= ord4 <= 29:
            score += 8
        elif 30 <= ord4 <= 39:
            score += 6

        # 4. ball3-ball4 간격 점수
        gap = ord4 - ord3
        if gap in GAP34_STATS:
            score += GAP34_STATS[gap]['freq'] // 4

        # 5. 연속수 패턴 (12.1%)
        if gap == 1:
            score += 5

        # 6. 원핫인코딩 - ord4 포지션 비트 빈도 (역발상)
        if ord4 in ORD4_BIT_FREQ:
            bit_freq = ORD4_BIT_FREQ[ord4]
            # 원핫 빈도가 낮을수록 보너스 (뜻밖)
            if bit_freq <= 10:
                score += (15 - bit_freq)
            else:
                score += bit_freq // 3

        # 7. 원핫인코딩 - 핫/콜드 비트 (역발상!)
        # 콜드 비트에 보너스, 핫 비트에 패널티
        if ord4 in COLD_BITS:
            score += 12  # 뜻밖의 수니까 콜드 비트 선호
        elif ord4 in HOT_BITS:
            score -= 3   # 핫 비트는 약간 감점 (너무 많이 안함)

        # 8. 소수 보너스 (25.6%인데 뜻밖이므로 약간 가산)
        if ord4 in {11, 13, 17, 19, 23, 29, 31, 37, 41, 43}:
            score += 5

        # === Range 반영 (80/20 전략, 뜻밖의 수와 조합) ===
        ord4_range = get_range(ord4)
        ord1_range = get_range(ord1)
        ord2_range = get_range(ord2)
        ord3_range = get_range(ord3)
        ord6_range = get_range(ord6)

        # 9. ord4 구간별 빈도 점수
        range_freq = ORD4_RANGE_FREQ.get(ord4_range, 0)
        if range_freq >= 30:  # 구간2, 구간3 (77.5%)
            score += int(range_freq * 0.2)  # 최대 8점 (뜻밖의 수라서 낮게)
        elif range_freq >= 2:  # 구간1, 구간4 (22.2%)
            # 20% 나올 가능성 있는 구간 - 뜻밖의 수 전략에 맞음
            score += int(range_freq * 0.4)  # 최대 7점

        # 10. 새로운 구간 추가 보너스 (4개 구간 = 51.2%)
        existing_ranges = {ord1_range, ord2_range, ord3_range, ord6_range}
        if ord4_range not in existing_ranges:
            new_unique_count = len(existing_ranges) + 1
            if new_unique_count in UNIQUE_RANGE_BONUS:
                score += UNIQUE_RANGE_BONUS[new_unique_count]

        # 11. 연속 구간 패턴 (57.5%)
        # 뜻밖의 수 전략: 연속 구간보다 점프 구간에 약간 보너스
        if abs(ord4_range - ord3_range) >= 2:
            score += 3  # 구간 점프 보너스 (뜻밖)
        elif abs(ord4_range - ord3_range) <= 1:
            score += 2  # 연속 구간도 나쁘지 않음

        candidates.append({
            'ord4': ord4,
            'score': score,
        })

    # 점수순 정렬 후 상위 N개 선택
    candidates.sort(key=lambda x: -x['score'])
    return candidates[:max_candidates]


def process_result():
    """result.csv 읽어서 ord4 추가"""

    # 기존 result.csv 읽기
    rows = []
    with open(RESULT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"기존 행 수: {len(rows)}")

    # ord4 추가하여 새 행 생성
    new_rows = []
    expanded_count = 0

    for row in rows:
        ord1 = int(row['ord1'])
        ord2 = int(row['ord2']) if row['ord2'] else None
        ord3 = int(row['ord3']) if row['ord3'] else None
        ord6 = int(row['ord6'])

        if ord3 is None:
            new_rows.append(row)
            continue

        # ord4 후보 선택
        candidates = select_ord4_candidates(ord1, ord2, ord3, ord6, max_candidates=2)

        def update_row_info(new_row, ord4_val, score_val):
            """행에 추가 정보 업데이트"""
            new_row['ord4'] = ord4_val
            # range_code 업데이트
            r1 = get_range(ord1)
            r2 = get_range(ord2) if ord2 else '-'
            r3 = get_range(ord3)
            r4 = get_range(ord4_val)
            r6 = get_range(ord6)
            new_row['range_code'] = f"{r1}{r2}{r3}{r4}-{r6}"
            new_row['unique_ranges'] = len({r1, r2 if ord2 else 99, r3, r4, r6}) - (0 if ord2 else 1)
            # hot/cold 카운트
            nums = [ord1, ord2, ord3, ord4_val, ord6]
            nums = [x for x in nums if x]  # None 제거
            new_row['hot_count'] = sum(1 for x in nums if x in HOT_BITS)
            new_row['cold_count'] = sum(1 for x in nums if x in COLD_BITS)
            # 연속수 카운트
            consec = 0
            if ord2 and ord2 - ord1 == 1: consec += 1
            if ord3 - ord2 == 1 if ord2 else False: consec += 1
            if ord4_val - ord3 == 1: consec += 1
            new_row['consecutive'] = consec
            new_row['score'] = score_val
            return new_row

        if not candidates:
            new_rows.append(row)
        elif len(candidates) == 1:
            new_row = row.copy()
            new_row = update_row_info(new_row, candidates[0]['ord4'], candidates[0]['score'])
            new_rows.append(new_row)
        else:
            for cand in candidates:
                new_row = row.copy()
                new_row = update_row_info(new_row, cand['ord4'], cand['score'])
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
        print(f"  ({row['ord1']}, {row['ord2']}, {row['ord3']}, {row['ord4']}, ..., {row['ord6']}) - {row['분류']}")


if __name__ == "__main__":
    process_result()
