"""
Consecutive(연속수) 인사이트 생성 - 전문가 분석

분석 항목:
1. 연속수 개수 분포 (0개, 2개, 3개, ...)
2. 연속수 쌍 개수 분포 (0쌍, 1쌍, 2쌍)
3. 연속수 시작 위치 분석
4. 연속수 시작 번호 분포
5. 연속수 길이별 분포
6. 다중 연속수 패턴
7. 최근 트렌드 분석
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict
import statistics

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent / "statistics"


def load_data() -> list:
    """당첨번호 데이터 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = sorted([int(row[f'ball{i}']) for i in range(1, 7)])
            results.append({
                'round': int(row['round']),
                'balls': balls,
            })
    return results


def find_consecutive_sequences(balls: list) -> list:
    """연속수 시퀀스 찾기"""
    sequences = []
    i = 0
    while i < len(balls):
        start = i
        while i < len(balls) - 1 and balls[i + 1] == balls[i] + 1:
            i += 1
        if i > start:  # 연속수 발견
            sequences.append({
                'start_value': balls[start],
                'length': i - start + 1,
                'start_position': start + 1,  # 1-indexed
                'values': balls[start:i + 1]
            })
        i += 1
    return sequences


def generate_insight():
    """Consecutive 인사이트 생성"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    total_rounds = len(data)

    # 각 회차별 연속수 정보 수집
    all_sequences = []
    round_info = []

    for r in data:
        seqs = find_consecutive_sequences(r['balls'])
        total_consec_nums = sum(s['length'] for s in seqs)
        num_pairs = sum(s['length'] - 1 for s in seqs)

        round_info.append({
            'round': r['round'],
            'balls': r['balls'],
            'sequences': seqs,
            'num_sequences': len(seqs),
            'total_consecutive_nums': total_consec_nums,
            'num_pairs': num_pairs
        })
        all_sequences.extend(seqs)

    # ========== 1. 연속수 쌍 개수 분포 ==========
    pair_count_freq = Counter(ri['num_pairs'] for ri in round_info)

    with open(OUTPUT_DIR / "consecutive_pair_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_pairs', 'frequency', 'ratio', 'probability'])
        for count in sorted(pair_count_freq.keys()):
            freq = pair_count_freq[count]
            writer.writerow([count, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 2. 연속수 시퀀스 개수 분포 ==========
    seq_count_freq = Counter(ri['num_sequences'] for ri in round_info)

    with open(OUTPUT_DIR / "sequence_count_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_sequences', 'frequency', 'ratio', 'probability', 'description'])
        for count in sorted(seq_count_freq.keys()):
            freq = seq_count_freq[count]
            desc = f"{count}개 연속수 그룹" if count > 0 else "연속수 없음"
            writer.writerow([count, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4), desc])

    # ========== 3. 연속수 길이별 분포 ==========
    length_freq = Counter(s['length'] for s in all_sequences)

    with open(OUTPUT_DIR / "consecutive_length_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['length', 'frequency', 'ratio', 'description'])
        total_seqs = len(all_sequences)
        for length in sorted(length_freq.keys()):
            freq = length_freq[length]
            desc = f"{length}연속 (예: n,n+1,...)" if length == 2 else f"{length}연속"
            writer.writerow([length, freq, round(freq/total_seqs*100, 1) if total_seqs > 0 else 0, desc])

    # ========== 4. 연속수 시작 번호 분포 ==========
    start_value_freq = Counter(s['start_value'] for s in all_sequences)

    with open(OUTPUT_DIR / "consecutive_start_value_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_value', 'frequency', 'ratio'])
        total_seqs = len(all_sequences)
        for val in sorted(start_value_freq.keys()):
            freq = start_value_freq[val]
            writer.writerow([val, freq, round(freq/total_seqs*100, 1) if total_seqs > 0 else 0])

    # ========== 5. 연속수 시작 위치 분포 ==========
    start_pos_freq = Counter(s['start_position'] for s in all_sequences)

    with open(OUTPUT_DIR / "consecutive_start_position_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_position', 'frequency', 'ratio', 'description'])
        total_seqs = len(all_sequences)
        for pos in sorted(start_pos_freq.keys()):
            freq = start_pos_freq[pos]
            desc = f"ball{pos}-ball{pos+1}부터 시작"
            writer.writerow([pos, freq, round(freq/total_seqs*100, 1) if total_seqs > 0 else 0, desc])

    # ========== 6. 연속수 패턴 상세 ==========
    pattern_freq = Counter()
    for ri in round_info:
        if ri['sequences']:
            pattern = '-'.join([f"{s['length']}연속" for s in ri['sequences']])
        else:
            pattern = "없음"
        pattern_freq[pattern] += 1

    with open(OUTPUT_DIR / "consecutive_pattern_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pattern', 'frequency', 'ratio', 'probability'])
        for pattern, freq in pattern_freq.most_common():
            writer.writerow([pattern, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 7. 최근 트렌드 분석 ==========
    recent_50 = round_info[-50:]
    recent_100 = round_info[-100:]

    recent_50_pair_freq = Counter(ri['num_pairs'] for ri in recent_50)
    recent_100_pair_freq = Counter(ri['num_pairs'] for ri in recent_100)

    with open(OUTPUT_DIR / "consecutive_trend_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_pairs', 'all_time_freq', 'all_time_ratio', 'recent_100_freq',
                        'recent_100_ratio', 'recent_50_freq', 'recent_50_ratio', 'trend'])
        for count in sorted(pair_count_freq.keys()):
            all_freq = pair_count_freq[count]
            r100_freq = recent_100_pair_freq.get(count, 0)
            r50_freq = recent_50_pair_freq.get(count, 0)

            all_ratio = all_freq / total_rounds
            r100_ratio = r100_freq / 100 if r100_freq else 0
            r50_ratio = r50_freq / 50 if r50_freq else 0

            if r50_ratio > all_ratio * 1.3:
                trend = '상승'
            elif r50_ratio < all_ratio * 0.7:
                trend = '하락'
            else:
                trend = '보합'

            writer.writerow([count, all_freq, round(all_ratio*100, 1), r100_freq,
                           round(r100_ratio*100, 1), r50_freq, round(r50_ratio*100, 1), trend])

    # ========== Summary ==========
    has_consecutive = sum(1 for ri in round_info if ri['num_pairs'] > 0)
    avg_pairs = statistics.mean(ri['num_pairs'] for ri in round_info)

    with open(OUTPUT_DIR / "summary.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'description'])
        writer.writerow(['total_rounds', total_rounds, '분석 회차 수'])
        writer.writerow(['rounds_with_consecutive', has_consecutive, '연속수 있는 회차'])
        writer.writerow(['consecutive_ratio', round(has_consecutive/total_rounds*100, 1), '연속수 출현 비율(%)'])
        writer.writerow(['avg_pairs_per_round', round(avg_pairs, 2), '회차당 평균 연속쌍 수'])
        writer.writerow(['total_sequences', len(all_sequences), '전체 연속수 시퀀스 수'])
        writer.writerow(['most_common_length', length_freq.most_common(1)[0][0] if length_freq else 0, '최빈 연속수 길이'])

    # ========== 출력 ==========
    print("=" * 60)
    print("Consecutive(연속수) 전문가 인사이트")
    print("=" * 60)
    print(f"\n총 {total_rounds}회차 분석\n")

    print("[1. 연속수 출현 현황]")
    print(f"    연속수 있는 회차: {has_consecutive}회 ({has_consecutive/total_rounds*100:.1f}%)")
    print(f"    연속수 없는 회차: {total_rounds - has_consecutive}회 ({(total_rounds-has_consecutive)/total_rounds*100:.1f}%)")
    print(f"    회차당 평균 연속쌍: {avg_pairs:.2f}개")

    print("\n[2. 연속쌍 개수 분포]")
    for count in sorted(pair_count_freq.keys()):
        freq = pair_count_freq[count]
        bar = '█' * int(freq/total_rounds*50)
        print(f"    {count}쌍: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {bar}")

    print("\n[3. 연속수 길이 분포]")
    total_seqs = len(all_sequences)
    for length in sorted(length_freq.keys()):
        freq = length_freq[length]
        ratio = freq/total_seqs*100 if total_seqs > 0 else 0
        print(f"    {length}연속: {freq:3d}개 ({ratio:5.1f}%)")

    print("\n[4. 연속수 시작 위치]")
    for pos in sorted(start_pos_freq.keys()):
        freq = start_pos_freq[pos]
        ratio = freq/total_seqs*100 if total_seqs > 0 else 0
        print(f"    위치 {pos}: {freq:3d}개 ({ratio:5.1f}%) - ball{pos}에서 시작")

    print("\n[5. 연속수 시작 번호 상위 10개]")
    for val, freq in start_value_freq.most_common(10):
        ratio = freq/total_seqs*100 if total_seqs > 0 else 0
        print(f"    {val:2d}부터: {freq:3d}개 ({ratio:5.1f}%)")

    print("\n[6. 연속수 패턴 상위 5개]")
    for pattern, freq in pattern_freq.most_common(5):
        print(f"    {pattern}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n" + "=" * 60)
    print(f"CSV 파일 저장 완료: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    generate_insight()
