"""
second 결과 내보내기 - ord2 추가

실행: python export_result.py
입력: result/result.csv (ord1, ord6 있음)
출력: result/result.csv (ord2 추가, 복수 추천시 행 복제)

ord2 선택 조건:
1. ord1 < ord2 < ord6 - 4 (ord3,4,5 자리 확보)
2. ball1별 ball2 통계 기반 점수화
3. 연속수/소수 패턴 반영
"""

import csv
from pathlib import Path

RESULT_PATH = Path(__file__).parent.parent.parent / "result" / "result.csv"

# ball2 빈도 통계 (내장)
# 형식: {ball2: {'freq': 빈도, 'is_prime': 소수여부, 'rank': 순위}}
BALL2_STATS = {
    7: {'freq': 25, 'is_prime': True, 'rank': 1},
    16: {'freq': 23, 'is_prime': False, 'rank': 2},
    11: {'freq': 23, 'is_prime': True, 'rank': 3},
    6: {'freq': 23, 'is_prime': False, 'rank': 4},
    9: {'freq': 23, 'is_prime': False, 'rank': 5},
    12: {'freq': 21, 'is_prime': False, 'rank': 6},
    14: {'freq': 20, 'is_prime': False, 'rank': 7},
    18: {'freq': 20, 'is_prime': False, 'rank': 8},
    13: {'freq': 20, 'is_prime': True, 'rank': 9},
    10: {'freq': 17, 'is_prime': False, 'rank': 10},
    4: {'freq': 17, 'is_prime': False, 'rank': 11},
    8: {'freq': 16, 'is_prime': False, 'rank': 12},
    15: {'freq': 16, 'is_prime': False, 'rank': 13},
    5: {'freq': 13, 'is_prime': True, 'rank': 14},
    21: {'freq': 12, 'is_prime': False, 'rank': 15},
    17: {'freq': 12, 'is_prime': True, 'rank': 16},
    19: {'freq': 11, 'is_prime': True, 'rank': 17},
    3: {'freq': 9, 'is_prime': True, 'rank': 18},
    26: {'freq': 8, 'is_prime': False, 'rank': 19},
    20: {'freq': 8, 'is_prime': False, 'rank': 20},
    23: {'freq': 8, 'is_prime': True, 'rank': 21},
    24: {'freq': 6, 'is_prime': False, 'rank': 22},
    25: {'freq': 5, 'is_prime': False, 'rank': 23},
    2: {'freq': 5, 'is_prime': True, 'rank': 24},
    31: {'freq': 4, 'is_prime': True, 'rank': 25},
    22: {'freq': 3, 'is_prime': False, 'rank': 26},
    27: {'freq': 3, 'is_prime': False, 'rank': 27},
    34: {'freq': 2, 'is_prime': False, 'rank': 28},
    30: {'freq': 2, 'is_prime': False, 'rank': 29},
    33: {'freq': 2, 'is_prime': False, 'rank': 30},
    32: {'freq': 1, 'is_prime': False, 'rank': 31},
    36: {'freq': 1, 'is_prime': False, 'rank': 32},
}

# ball1별 ball2 통계 (내장)
# 형식: {ball1: {'ball2_avg': 평균, 'ball2_mode': 최빈값, 'ball2_min': 최소, 'ball2_max': 최대}}
BALL1_BALL2_STATS = {
    1: {'ball2_avg': 7.6, 'ball2_mode': 9, 'ball2_min': 2, 'ball2_max': 23},
    2: {'ball2_avg': 9.1, 'ball2_mode': 6, 'ball2_min': 3, 'ball2_max': 25},
    3: {'ball2_avg': 9.2, 'ball2_mode': 6, 'ball2_min': 4, 'ball2_max': 20},
    4: {'ball2_avg': 10.4, 'ball2_mode': 7, 'ball2_min': 5, 'ball2_max': 24},
    5: {'ball2_avg': 11.3, 'ball2_mode': 12, 'ball2_min': 6, 'ball2_max': 18},
    6: {'ball2_avg': 12.4, 'ball2_mode': 7, 'ball2_min': 7, 'ball2_max': 24},
    7: {'ball2_avg': 12.4, 'ball2_mode': 9, 'ball2_min': 8, 'ball2_max': 24},
    8: {'ball2_avg': 14.3, 'ball2_mode': 11, 'ball2_min': 9, 'ball2_max': 23},
    9: {'ball2_avg': 14.4, 'ball2_mode': 14, 'ball2_min': 10, 'ball2_max': 21},
    10: {'ball2_avg': 16.5, 'ball2_mode': 16, 'ball2_min': 11, 'ball2_max': 34},
    11: {'ball2_avg': 17.5, 'ball2_mode': 17, 'ball2_min': 13, 'ball2_max': 23},
    12: {'ball2_avg': 18.9, 'ball2_mode': 18, 'ball2_min': 14, 'ball2_max': 30},
    13: {'ball2_avg': 17.5, 'ball2_mode': 14, 'ball2_min': 14, 'ball2_max': 24},
    14: {'ball2_avg': 20.2, 'ball2_mode': 16, 'ball2_min': 15, 'ball2_max': 33},
    15: {'ball2_avg': 21.0, 'ball2_mode': 23, 'ball2_min': 16, 'ball2_max': 26},
    16: {'ball2_avg': 21.1, 'ball2_mode': 20, 'ball2_min': 18, 'ball2_max': 26},
    17: {'ball2_avg': 21.6, 'ball2_mode': 18, 'ball2_min': 18, 'ball2_max': 26},
    19: {'ball2_avg': 24.5, 'ball2_mode': 21, 'ball2_min': 21, 'ball2_max': 32},
    20: {'ball2_avg': 25.7, 'ball2_mode': 25, 'ball2_min': 21, 'ball2_max': 31},
    21: {'ball2_avg': 26.3, 'ball2_mode': 25, 'ball2_min': 22, 'ball2_max': 33},
}

# ord1-ord2 간격 통계 (내장)
# 형식: {gap: {'freq': 빈도}}
GAP_STATS = {
    1: {'freq': 47},   # 연속수 12.4%
    2: {'freq': 38},
    3: {'freq': 40},
    4: {'freq': 50},   # 최빈 간격
    5: {'freq': 32},
    6: {'freq': 33},
    7: {'freq': 18},
    8: {'freq': 26},
    9: {'freq': 13},
    10: {'freq': 15},
    11: {'freq': 15},
    12: {'freq': 14},
    13: {'freq': 8},
    14: {'freq': 8},
    15: {'freq': 6},
    16: {'freq': 3},
    17: {'freq': 4},
    18: {'freq': 3},
    19: {'freq': 1},
    20: {'freq': 2},
}

# 원핫인코딩 - ord2 포지션 비트 빈도 (상위)
# bit_frequency_by_position.csv의 ord2_freq 컬럼
ORD2_BIT_FREQ = {
    9: 28,   # 1위
    10: 26,  # 2위
    5: 23,   # 3위
    17: 22,  # 4위
    6: 21,   # 5위
    7: 20, 11: 20, 12: 20, 15: 20,  # 공동 6위
    19: 19,
    13: 18,
    14: 17,
    18: 15, 20: 14,
    3: 12, 8: 13, 4: 11,
    16: 9, 21: 8, 22: 8,
    2: 7,
}

# 원핫인코딩 - 핫/콜드 비트
HOT_BITS = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31, 37, 33, 10, 2, 32}
COLD_BITS = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7, 30, 15, 11, 40, 6}

# 확장 헤더
FIELDNAMES = [
    'ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6',
    '분류', '빈도',
    'range_code', 'unique_ranges', 'consecutive', 'hot_count', 'cold_count', 'score'
]

def get_range(num):
    """번호의 구간 반환 (0-4)"""
    if num <= 9: return 0
    elif num <= 19: return 1
    elif num <= 29: return 2
    elif num <= 39: return 3
    else: return 4


def select_ord2_candidates(ord1, ord6, ball2_stats, ball1_stats, gap_stats, max_candidates=3):
    """주어진 (ord1, ord6)에 대해 ord2 후보 선택"""

    # ord2 가능 범위: ord1+1 ~ ord6-4 (ord3,4,5 자리 확보)
    min_ord2 = ord1 + 1
    max_ord2 = ord6 - 4

    if min_ord2 > max_ord2:
        return []  # 가능한 ord2 없음

    candidates = []

    for ord2 in range(min_ord2, max_ord2 + 1):
        score = 0

        # 1. ball2 빈도 점수 (높을수록 좋음)
        if ord2 in ball2_stats:
            score += ball2_stats[ord2]['freq'] * 2

        # 2. ball1별 ball2 통계 반영
        if ord1 in ball1_stats:
            b1_stats = ball1_stats[ord1]
            # mode (최빈값) 보너스
            if ord2 == b1_stats['ball2_mode']:
                score += 15
            # 평균 근처 보너스
            if abs(ord2 - b1_stats['ball2_avg']) <= 2:
                score += 10
            # min-max 범위 내 보너스
            if b1_stats['ball2_min'] <= ord2 <= b1_stats['ball2_max']:
                score += 5

        # 3. 간격(gap) 점수
        gap = ord2 - ord1
        if gap in gap_stats:
            score += gap_stats[gap]['freq'] // 2

        # 4. 최빈 구간 보너스 (10-19: 48.3%)
        if 10 <= ord2 <= 19:
            score += 10
        elif 1 <= ord2 <= 9:
            score += 5  # 구간0도 34.6%로 괜찮음

        # 5. 연속수 패턴 (12.4% 연속)
        if gap == 1:
            score += 5  # 연속수 약간 보너스

        # 6. 원핫인코딩 - ord2 포지션 비트 빈도
        if ord2 in ORD2_BIT_FREQ:
            score += ORD2_BIT_FREQ[ord2]  # 최대 28점

        # 7. 원핫인코딩 - 핫/콜드 비트
        if ord2 in HOT_BITS:
            score += 10  # 핫 비트 보너스
        elif ord2 in COLD_BITS:
            score -= 5   # 콜드 비트 패널티

        candidates.append({
            'ord2': ord2,
            'score': score,
        })

    # 점수순 정렬 후 상위 N개 선택
    candidates.sort(key=lambda x: -x['score'])
    return candidates[:max_candidates]


def process_result():
    """result.csv 읽어서 ord2 추가"""

    # 내장 통계 사용
    ball2_stats = BALL2_STATS
    ball1_stats = BALL1_BALL2_STATS
    gap_stats = GAP_STATS

    # 기존 result.csv 읽기
    rows = []
    with open(RESULT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"기존 행 수: {len(rows)}")

    # ord2 추가하여 새 행 생성
    new_rows = []
    expanded_count = 0

    for row in rows:
        ord1 = int(row['ord1'])
        ord6 = int(row['ord6'])

        # ord2 후보 선택
        candidates = select_ord2_candidates(
            ord1, ord6, ball2_stats, ball1_stats, gap_stats,
            max_candidates=2  # 상위 2개 후보
        )

        if not candidates:
            # 가능한 ord2 없으면 그대로 유지
            new_rows.append(row)
        elif len(candidates) == 1:
            # 1개만 있으면 그냥 추가
            new_row = row.copy()
            ord2 = candidates[0]['ord2']
            new_row['ord2'] = ord2
            # range_code 업데이트
            r1, r2, r6 = get_range(ord1), get_range(ord2), get_range(ord6)
            new_row['range_code'] = f"{r1}{r2}---{r6}"
            new_row['unique_ranges'] = len({r1, r2, r6})
            # hot/cold 카운트
            hot_cnt = sum(1 for x in [ord1, ord2, ord6] if x in HOT_BITS)
            cold_cnt = sum(1 for x in [ord1, ord2, ord6] if x in COLD_BITS)
            new_row['hot_count'] = hot_cnt
            new_row['cold_count'] = cold_cnt
            new_row['consecutive'] = 1 if ord2 - ord1 == 1 else 0
            new_rows.append(new_row)
        else:
            # 복수 후보면 행 복제
            for cand in candidates:
                new_row = row.copy()
                ord2 = cand['ord2']
                new_row['ord2'] = ord2
                # range_code 업데이트
                r1, r2, r6 = get_range(ord1), get_range(ord2), get_range(ord6)
                new_row['range_code'] = f"{r1}{r2}---{r6}"
                new_row['unique_ranges'] = len({r1, r2, r6})
                # hot/cold 카운트
                hot_cnt = sum(1 for x in [ord1, ord2, ord6] if x in HOT_BITS)
                cold_cnt = sum(1 for x in [ord1, ord2, ord6] if x in COLD_BITS)
                new_row['hot_count'] = hot_cnt
                new_row['cold_count'] = cold_cnt
                new_row['consecutive'] = 1 if ord2 - ord1 == 1 else 0
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
        print(f"  ({row['ord1']}, {row['ord2']}, ..., {row['ord6']}) - {row['분류']}")


if __name__ == "__main__":
    process_result()
