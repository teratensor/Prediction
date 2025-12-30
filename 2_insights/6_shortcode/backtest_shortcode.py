"""
숏코드 백테스트 - 각 회차별 숏코드와 출현 회차 저장

출력: statistics/shortcode_by_round.csv
- round: 회차
- winning: 당첨번호
- shortcode: 6자리 숏코드
- ord_code: Ord 3자리
- ball_code: Ball 3자리

Ord vs Ball 차이:
- Ord 세그먼트: 번호 빈도 기반 Top24/Mid14/Rest7
- Ball 세그먼트: 번호 빈도 기반 Top24/Mid14/Rest7 (다를 수 있음)

- Ord (ABC): 당첨번호가 Ord 세그먼트 어디에 속하는지
- Ball (DEF): 당첨번호 위치(o1~o45)의 값이 Ball 세그먼트 어디에 속하는지
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent / "statistics"


def load_winning_numbers():
    """당첨번호 로드 (o1~o45 포함)"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = sorted([int(row[f'ball{i}']) for i in range(1, 7)])
            # o1~o45 값 로드
            o_values = {i: int(row[f'o{i}']) for i in range(1, 46)}
            results.append({
                'round': int(row['round']),
                'balls': balls,
                'o_values': o_values,
            })
    return sorted(results, key=lambda x: x['round'])


def get_segments(all_data, target_round, lookback=10):
    """빈도 기반 Top24/Mid14/Rest7 세그먼트 계산

    Args:
        all_data: 전체 당첨번호 데이터
        target_round: 목표 회차
        lookback: 참조할 최근 회차 수 (기본 10)

    Returns:
        tuple: (TOP_24, MID_14, REST_7) 세트
    """
    train_data = [r for r in all_data if r['round'] < target_round]

    if len(train_data) < lookback:
        lookback = len(train_data)

    if lookback == 0:
        return set(), set(), set(range(1, 46))

    # 최근 N회차 빈도 계산
    recent_data = train_data[-lookback:]
    freq = Counter()
    for r in recent_data:
        for b in r['balls']:
            freq[b] += 1

    # 빈도순 정렬
    sorted_nums = sorted(range(1, 46), key=lambda x: -freq.get(x, 0))

    TOP_24 = set(sorted_nums[:24])
    MID_14 = set(sorted_nums[24:38])
    REST_7 = set(sorted_nums[38:])

    return TOP_24, MID_14, REST_7


def calculate_shortcode(balls, o_values, ord_segments, ball_segments):
    """당첨번호의 숏코드 계산

    Args:
        balls: 당첨번호 리스트 (정렬됨)
        o_values: o1~o45 값 딕셔너리
        ord_segments: (TOP_24, MID_14, REST_7) for Ord
        ball_segments: (TOP_24, MID_14, REST_7) for Ball

    Returns:
        tuple: (shortcode, ord_code, ball_code, details)

    Ord vs Ball 차이:
    - Ord (ABC): 당첨번호(12,16,19,28,33,42)가 Ord 세그먼트 어디에 속하는지
    - Ball (DEF): 당첨번호 위치의 o값(o12,o16,o19,o28,o33,o42)이 Ball 세그먼트 어디에 속하는지
    """
    ORD_TOP24, ORD_MID14, ORD_REST7 = ord_segments
    BALL_TOP24, BALL_MID14, BALL_REST7 = ball_segments

    # Ord 기준: 당첨번호가 Ord 세그먼트 어디에 속하는지
    ord_top24 = sum(1 for b in balls if b in ORD_TOP24)
    ord_mid14 = sum(1 for b in balls if b in ORD_MID14)
    ord_rest7 = sum(1 for b in balls if b in ORD_REST7)

    # Ball 기준: 당첨번호 위치의 o값이 Ball 세그먼트 어디에 속하는지
    # balls = [12, 16, 19, 28, 33, 42] → o12, o16, o19, o28, o33, o42 값 확인
    ball_values = [o_values[b] for b in balls]
    ball_top24 = sum(1 for v in ball_values if v in BALL_TOP24)
    ball_mid14 = sum(1 for v in ball_values if v in BALL_MID14)
    ball_rest7 = sum(1 for v in ball_values if v in BALL_REST7)

    ord_code = f"{ord_top24}{ord_mid14}{ord_rest7}"
    ball_code = f"{ball_top24}{ball_mid14}{ball_rest7}"
    shortcode = ord_code + ball_code

    return shortcode, ord_code, ball_code, {
        'ord_top24': ord_top24,
        'ord_mid14': ord_mid14,
        'ord_rest7': ord_rest7,
        'ball_top24': ball_top24,
        'ball_mid14': ball_mid14,
        'ball_rest7': ball_rest7,
    }


def main():
    print("=" * 60)
    print("숏코드 백테스트 (Ord/Ball 분리)")
    print("=" * 60)

    all_data = load_winning_numbers()
    print(f"데이터: {len(all_data)}회차")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 각 회차별 숏코드 계산
    results = []

    for r in all_data:
        target_round = r['round']
        balls = r['balls']
        o_values = r['o_values']

        # 해당 회차 기준 세그먼트 계산
        ord_segments = get_segments(all_data, target_round)
        ball_segments = get_segments(all_data, target_round)

        if not ord_segments[0]:  # 첫 회차는 스킵
            continue

        shortcode, ord_code, ball_code, details = calculate_shortcode(
            balls, o_values, ord_segments, ball_segments
        )

        results.append({
            'round': target_round,
            'winning': balls,
            'shortcode': shortcode,
            'ord_code': ord_code,
            'ball_code': ball_code,
            **details,
        })

    print(f"분석 회차: {len(results)}회")

    # 1. shortcode_by_round.csv 저장
    output_path = OUTPUT_DIR / "shortcode_by_round.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'winning', 'shortcode', 'ord_code', 'ball_code',
                         'ord_top24', 'ord_mid14', 'ord_rest7',
                         'ball_top24', 'ball_mid14', 'ball_rest7'])
        for r in results:
            writer.writerow([
                r['round'],
                str(r['winning']),
                r['shortcode'],
                r['ord_code'],
                r['ball_code'],
                r['ord_top24'],
                r['ord_mid14'],
                r['ord_rest7'],
                r['ball_top24'],
                r['ball_mid14'],
                r['ball_rest7'],
            ])

    print(f"\n[저장] {output_path}")

    # 2. shortcode_rounds.csv 저장 - 각 숏코드별 출현 회차
    shortcode_rounds = defaultdict(list)
    for r in results:
        shortcode_rounds[r['shortcode']].append(r['round'])

    output_path2 = OUTPUT_DIR / "shortcode_rounds.csv"
    with open(output_path2, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['shortcode', 'count', 'rounds'])

        # 빈도순 정렬
        for code, rounds in sorted(shortcode_rounds.items(), key=lambda x: -len(x[1])):
            writer.writerow([
                code,
                len(rounds),
                ','.join(map(str, rounds)),
            ])

    print(f"[저장] {output_path2}")

    # 3. 통계 출력
    shortcode_freq = Counter(r['shortcode'] for r in results)

    print("\n" + "=" * 60)
    print("숏코드 분포 (상위 20개)")
    print("=" * 60)

    for code, count in shortcode_freq.most_common(20):
        pct = count / len(results) * 100
        rounds = shortcode_rounds[code]
        recent = rounds[-3:] if len(rounds) >= 3 else rounds
        # 코드 해석
        ord_part = f"Ord({code[0]},{code[1]},{code[2]})"
        ball_part = f"Ball({code[3]},{code[4]},{code[5]})"
        print(f"  {code}: {count:3d}회 ({pct:5.1f}%) {ord_part} {ball_part} - 최근: {recent}")

    print(f"\n고유 숏코드 수: {len(shortcode_freq)}개")

    # Ord와 Ball이 다른 경우 통계
    diff_count = sum(1 for r in results if r['ord_code'] != r['ball_code'])
    same_count = len(results) - diff_count
    print(f"\nOrd≠Ball: {diff_count}회 ({diff_count/len(results)*100:.1f}%)")
    print(f"Ord=Ball: {same_count}회 ({same_count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
