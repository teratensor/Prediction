"""
firstend 쌍 분석

ord1(첫수)과 ord6(끝수)의 모든 가능한 쌍에 대해:
1. 실제로 나온 쌍 (seen pairs)
2. 아직 나오지 않은 쌍 (unseen pairs)
을 분류하고 빈도 분석

출력 형식: ord1,ord2,ord3,ord4,ord5,ord6,분류,빈도
(ord2-5는 빈칸)
"""

import csv
from pathlib import Path
from collections import Counter

DATA_PATH = Path(__file__).parent.parent / "1_data" / "winning_numbers.csv"


def load_data():
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'round': int(row['round']),
                'balls': sorted([int(row[f'ball{i}']) for i in range(1, 7)]),
            })
    return results


def analyze_and_save(data):
    """모든 가능한 (ord1, ord6) 쌍 분석 및 통합 CSV 저장"""

    # 실제 나온 쌍 빈도 계산
    pair_freq = Counter()
    for r in data:
        ord1 = r['balls'][0]  # 첫수 (가장 작은 번호)
        ord6 = r['balls'][5]  # 끝수 (가장 큰 번호)
        pair_freq[(ord1, ord6)] += 1

    # 실제 데이터 기반 범위 확인
    actual_ord1_max = max(r['balls'][0] for r in data)
    actual_ord6_min = min(r['balls'][5] for r in data)

    # 모든 가능한 쌍 생성 (실제 데이터 범위 기반)
    # ord1: 1 ~ 실제최대값, ord6: ord1+5 ~ 45
    all_possible_pairs = []
    for o1 in range(1, actual_ord1_max + 1):
        for o6 in range(o1 + 5, 46):
            all_possible_pairs.append((o1, o6))

    total_possible = len(all_possible_pairs)
    total_rounds = len(data)

    print("=" * 70)
    print(f"firstend (ord1, ord6) 쌍 분석")
    print(f"총 회차: {total_rounds}회")
    print("=" * 70)

    # 나온 쌍 vs 안 나온 쌍
    seen_pairs = set(pair_freq.keys())
    unseen_pairs = set(all_possible_pairs) - seen_pairs

    # 빈도순 정렬
    sorted_pairs = sorted(pair_freq.items(), key=lambda x: -x[1])

    # 상위 30개를 최빈도로 분류
    top_freq_pairs = set(p for p, _ in sorted_pairs[:30])

    # 유력후보 (안나온 쌍 중 최적 범위)
    promising_pairs = set(
        (o1, o6) for (o1, o6) in unseen_pairs
        if 1 <= o1 <= 10 and 38 <= o6 <= 45
    )

    print(f"\n[전체 통계]")
    print(f"  가능한 모든 쌍: {total_possible}개")
    print(f"  나온 쌍 (seen): {len(seen_pairs)}개 ({len(seen_pairs)/total_possible*100:.1f}%)")
    print(f"  안 나온 쌍 (unseen): {len(unseen_pairs)}개 ({len(unseen_pairs)/total_possible*100:.1f}%)")

    # 나온 쌍 상세 (빈도순)
    print(f"\n[나온 쌍 (seen) - 빈도순 상위 30개]")
    print(f"  {'순위':>4} {'(ord1, ord6)':>15} {'빈도':>6} {'비율':>8}")
    print(f"  {'-'*4} {'-'*15} {'-'*6} {'-'*8}")

    for rank, ((o1, o6), freq) in enumerate(sorted_pairs[:30], 1):
        pct = freq / total_rounds * 100
        print(f"  {rank:4d} ({o1:2d}, {o6:2d})        {freq:6d} {pct:7.2f}%")

    # 유력후보 출력
    print(f"\n[안 나온 쌍 (unseen) 중 유력 후보]")
    print(f"  조건: ord1 (1-10) AND ord6 (38-45)")

    promising_list = sorted(promising_pairs)
    print(f"  총 {len(promising_list)}개:")
    for i, (o1, o6) in enumerate(promising_list):
        if i % 5 == 0:
            print(f"    ", end="")
        print(f"({o1:2d},{o6:2d}) ", end="")
        if (i + 1) % 5 == 0:
            print()
    if len(promising_list) % 5 != 0:
        print()

    # 통합 CSV 저장 (result 폴더)
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)

    # 구간 계산 함수
    def get_range(num):
        if num <= 9: return 0
        elif num <= 19: return 1
        elif num <= 29: return 2
        elif num <= 39: return 3
        else: return 4

    output_csv = output_dir / "result.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 확장된 헤더
        writer.writerow([
            'ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6',
            '분류', '빈도',
            'range_code', 'unique_ranges', 'consecutive', 'hot_count', 'cold_count', 'score'
        ])

        # 모든 가능한 쌍을 정렬하여 저장
        all_pairs_list = sorted(all_possible_pairs)
        for (o1, o6) in all_pairs_list:
            freq = pair_freq.get((o1, o6), 0)

            # 분류 결정
            if (o1, o6) in top_freq_pairs:
                cls = "top_freq"
            elif (o1, o6) in promising_pairs:
                cls = "promising"
            elif (o1, o6) in seen_pairs:
                cls = "seen"
            else:
                cls = "unseen"

            # 초기 range_code (ord1, ord6만 있음)
            r1, r6 = get_range(o1), get_range(o6)
            range_code = f"{r1}----{r6}"
            unique_ranges = len({r1, r6})

            # ord2-5는 빈칸, 나머지 정보도 초기값
            writer.writerow([
                o1, '', '', '', '', o6,
                cls, freq,
                range_code, unique_ranges, '', '', '', ''
            ])

    print(f"\n통합 CSV 저장: {output_csv}")

    # 분류별 요약
    print(f"\n[쌍 분류 요약]")
    print(f"  top_freq (최빈도 상위30): {len(top_freq_pairs)}개")
    print(f"  seen (나온쌍, top_freq 제외): {len(seen_pairs) - len(top_freq_pairs)}개")
    print(f"  promising (유력후보): {len(promising_pairs)}개")
    print(f"  unseen (안나온쌍, promising 제외): {len(unseen_pairs) - len(promising_pairs)}개")

    return output_csv


def main():
    data = load_data()
    analyze_and_save(data)


if __name__ == "__main__":
    main()
