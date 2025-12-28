"""
One-Hot Encoding 인사이트 생성 - 전문가 분석

ord1~ord6를 45비트 원핫인코딩으로 변환하여 패턴 분석

분석 항목:
1. ord별 45비트 원핫인코딩 생성
2. 포지션별(ord1~6) 비트 활성화 빈도
3. 비트 간 상관관계 분석 (동시 출현)
4. 비트 클러스터 분석 (구간별 활성화)
5. 희소성(Sparsity) 분석
6. 비트 시퀀스 패턴 분석
7. 머신러닝용 피처 매트릭스 생성
8. 최근 트렌드 분석
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
            results.append({
                'round': int(row['round']),
                'ord1': int(row['ord1']),
                'ord2': int(row['ord2']),
                'ord3': int(row['ord3']),
                'ord4': int(row['ord4']),
                'ord5': int(row['ord5']),
                'ord6': int(row['ord6']),
            })
    return results


def to_onehot(value: int, size: int = 45) -> list:
    """값을 원핫인코딩으로 변환 (1-indexed)"""
    vec = [0] * size
    if 1 <= value <= size:
        vec[value - 1] = 1
    return vec


def onehot_to_string(vec: list) -> str:
    """원핫 벡터를 문자열로 변환"""
    return ''.join(map(str, vec))


def get_active_bits(vec: list) -> list:
    """활성화된 비트 인덱스 반환 (1-indexed)"""
    return [i + 1 for i, v in enumerate(vec) if v == 1]


def get_cluster_label(bit: int) -> str:
    """비트의 클러스터 라벨"""
    if bit <= 9: return 'C1(01-09)'
    elif bit <= 18: return 'C2(10-18)'
    elif bit <= 27: return 'C3(19-27)'
    elif bit <= 36: return 'C4(28-36)'
    else: return 'C5(37-45)'


def generate_insight():
    """One-Hot Encoding 인사이트 생성"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    total_rounds = len(data)

    # ========== 1. 원핫인코딩 매트릭스 생성 ==========
    onehot_data = []
    for r in data:
        row = {'round': r['round']}
        full_vector = []
        for pos in range(1, 7):
            ord_val = r[f'ord{pos}']
            vec = to_onehot(ord_val)
            full_vector.extend(vec)
            # 각 포지션별 원핫도 저장
            row[f'ord{pos}'] = ord_val
            row[f'ord{pos}_onehot'] = onehot_to_string(vec)
        row['full_onehot'] = onehot_to_string(full_vector)
        row['active_bits'] = ','.join(map(str, [r[f'ord{i}'] for i in range(1, 7)]))
        onehot_data.append(row)

    with open(OUTPUT_DIR / "onehot_matrix.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6',
                        'ord1_onehot', 'ord2_onehot', 'ord3_onehot',
                        'ord4_onehot', 'ord5_onehot', 'ord6_onehot', 'active_bits'])
        for row in onehot_data:
            writer.writerow([row['round'], row['ord1'], row['ord2'], row['ord3'],
                           row['ord4'], row['ord5'], row['ord6'],
                           row['ord1_onehot'], row['ord2_onehot'], row['ord3_onehot'],
                           row['ord4_onehot'], row['ord5_onehot'], row['ord6_onehot'],
                           row['active_bits']])

    # ========== 2. 포지션별 비트 활성화 빈도 ==========
    bit_freq_by_pos = {pos: Counter() for pos in range(1, 7)}
    for r in data:
        for pos in range(1, 7):
            bit_freq_by_pos[pos][r[f'ord{pos}']] += 1

    with open(OUTPUT_DIR / "bit_frequency_by_position.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bit', 'ord1_freq', 'ord1_ratio', 'ord2_freq', 'ord2_ratio',
                        'ord3_freq', 'ord3_ratio', 'ord4_freq', 'ord4_ratio',
                        'ord5_freq', 'ord5_ratio', 'ord6_freq', 'ord6_ratio',
                        'total_freq', 'total_ratio'])
        for bit in range(1, 46):
            row = [bit]
            total = 0
            for pos in range(1, 7):
                freq = bit_freq_by_pos[pos].get(bit, 0)
                total += freq
                row.extend([freq, round(freq/total_rounds*100, 1)])
            row.extend([total, round(total/(total_rounds*6)*100, 1)])
            writer.writerow(row)

    # ========== 3. 비트 쌍 동시 활성화 분석 ==========
    bit_pair_freq = Counter()
    for r in data:
        ords = [r[f'ord{i}'] for i in range(1, 7)]
        # 모든 쌍 조합
        for i in range(len(ords)):
            for j in range(i + 1, len(ords)):
                pair = tuple(sorted([ords[i], ords[j]]))
                bit_pair_freq[pair] += 1

    with open(OUTPUT_DIR / "bit_pair_frequency.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bit1', 'bit2', 'frequency', 'ratio', 'rank'])
        for rank, ((b1, b2), freq) in enumerate(bit_pair_freq.most_common(100), 1):
            writer.writerow([b1, b2, freq, round(freq/total_rounds*100, 1), rank])

    # ========== 4. 클러스터별 활성화 패턴 ==========
    cluster_pattern_freq = Counter()
    cluster_by_pos = {pos: Counter() for pos in range(1, 7)}

    for r in data:
        pattern = []
        for pos in range(1, 7):
            cluster = get_cluster_label(r[f'ord{pos}'])
            pattern.append(cluster[0:2])  # C1, C2, ...
            cluster_by_pos[pos][cluster] += 1
        cluster_pattern_freq[tuple(pattern)] += 1

    with open(OUTPUT_DIR / "cluster_pattern_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pattern', 'frequency', 'ratio', 'rank'])
        for rank, (pattern, freq) in enumerate(cluster_pattern_freq.most_common(50), 1):
            writer.writerow(['-'.join(pattern), freq, round(freq/total_rounds*100, 1), rank])

    with open(OUTPUT_DIR / "cluster_by_position.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cluster', 'ord1_freq', 'ord1_ratio', 'ord2_freq', 'ord2_ratio',
                        'ord3_freq', 'ord3_ratio', 'ord4_freq', 'ord4_ratio',
                        'ord5_freq', 'ord5_ratio', 'ord6_freq', 'ord6_ratio'])
        for cluster in ['C1(01-09)', 'C2(10-18)', 'C3(19-27)', 'C4(28-36)', 'C5(37-45)']:
            row = [cluster]
            for pos in range(1, 7):
                freq = cluster_by_pos[pos].get(cluster, 0)
                row.extend([freq, round(freq/total_rounds*100, 1)])
            writer.writerow(row)

    # ========== 5. 희소성(Sparsity) 분석 ==========
    # 45비트 중 1비트만 활성화 = 97.8% 희소성
    sparsity_stats = {
        'bits_per_position': 45,
        'active_per_position': 1,
        'sparsity_per_position': round((1 - 1/45) * 100, 2),
        'total_bits': 270,  # 6 positions × 45 bits
        'total_active': 6,
        'total_sparsity': round((1 - 6/270) * 100, 2)
    }

    # 비트별 활성화 빈도의 분산 분석
    total_bit_freq = Counter()
    for r in data:
        for pos in range(1, 7):
            total_bit_freq[r[f'ord{pos}']] += 1

    freq_values = list(total_bit_freq.values())
    freq_variance = statistics.variance(freq_values) if len(freq_values) > 1 else 0
    freq_std = statistics.stdev(freq_values) if len(freq_values) > 1 else 0

    with open(OUTPUT_DIR / "sparsity_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'description'])
        writer.writerow(['bits_per_position', 45, '포지션당 비트 수'])
        writer.writerow(['active_per_position', 1, '포지션당 활성 비트 수'])
        writer.writerow(['sparsity_per_position', f"{sparsity_stats['sparsity_per_position']}%", '포지션당 희소성'])
        writer.writerow(['total_bits', 270, '총 비트 수 (6×45)'])
        writer.writerow(['total_active', 6, '총 활성 비트 수'])
        writer.writerow(['total_sparsity', f"{sparsity_stats['total_sparsity']}%", '전체 희소성'])
        writer.writerow(['freq_variance', round(freq_variance, 2), '비트 빈도 분산'])
        writer.writerow(['freq_std', round(freq_std, 2), '비트 빈도 표준편차'])
        writer.writerow(['freq_cv', round(freq_std/statistics.mean(freq_values)*100, 2), '변동계수(%)'])

    # ========== 6. 비트 시퀀스 패턴 (연속 비트) ==========
    consecutive_bits_freq = Counter()
    bit_gap_freq = Counter()

    for r in data:
        ords = sorted([r[f'ord{i}'] for i in range(1, 7)])
        # 연속 비트 카운트
        consecutive_count = 0
        for i in range(len(ords) - 1):
            gap = ords[i + 1] - ords[i]
            bit_gap_freq[gap] += 1
            if gap == 1:
                consecutive_count += 1
        consecutive_bits_freq[consecutive_count] += 1

    with open(OUTPUT_DIR / "bit_sequence_pattern.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['consecutive_pairs', 'frequency', 'ratio'])
        for count in sorted(consecutive_bits_freq.keys()):
            freq = consecutive_bits_freq[count]
            writer.writerow([count, freq, round(freq/total_rounds*100, 1)])

    with open(OUTPUT_DIR / "bit_gap_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gap', 'frequency', 'ratio', 'cumulative_ratio'])
        cumsum = 0
        total_gaps = sum(bit_gap_freq.values())
        for gap in sorted(bit_gap_freq.keys()):
            freq = bit_gap_freq[gap]
            cumsum += freq
            writer.writerow([gap, freq, round(freq/total_gaps*100, 1), round(cumsum/total_gaps*100, 1)])

    # ========== 7. 포지션 간 비트 차이 분석 ==========
    pos_diff_stats = defaultdict(list)
    for r in data:
        for i in range(1, 6):
            diff = r[f'ord{i+1}'] - r[f'ord{i}']
            pos_diff_stats[f'ord{i}_to_ord{i+1}'].append(diff)

    with open(OUTPUT_DIR / "position_bit_difference.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['transition', 'avg_diff', 'std_diff', 'min_diff', 'max_diff', 'median_diff'])
        for trans, diffs in pos_diff_stats.items():
            writer.writerow([trans, round(statistics.mean(diffs), 2),
                           round(statistics.stdev(diffs), 2),
                           min(diffs), max(diffs), statistics.median(diffs)])

    # ========== 8. 핫 비트 vs 콜드 비트 ==========
    total_bit_freq_sorted = sorted(total_bit_freq.items(), key=lambda x: -x[1])
    hot_bits = [b for b, _ in total_bit_freq_sorted[:15]]  # Top 15
    cold_bits = [b for b, _ in total_bit_freq_sorted[-15:]]  # Bottom 15

    with open(OUTPUT_DIR / "hot_cold_bits.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'bit', 'frequency', 'ratio', 'rank'])
        for rank, (bit, freq) in enumerate(total_bit_freq_sorted[:15], 1):
            writer.writerow(['HOT', bit, freq, round(freq/(total_rounds*6)*100, 2), rank])
        for rank, (bit, freq) in enumerate(total_bit_freq_sorted[-15:], 31):
            writer.writerow(['COLD', bit, freq, round(freq/(total_rounds*6)*100, 2), rank])

    # ========== 9. 최근 트렌드 분석 ==========
    recent_50 = data[-50:]
    recent_100 = data[-100:]

    recent_50_bit_freq = Counter()
    recent_100_bit_freq = Counter()

    for r in recent_50:
        for pos in range(1, 7):
            recent_50_bit_freq[r[f'ord{pos}']] += 1

    for r in recent_100:
        for pos in range(1, 7):
            recent_100_bit_freq[r[f'ord{pos}']] += 1

    with open(OUTPUT_DIR / "bit_trend_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bit', 'all_time_freq', 'all_time_ratio', 'recent_100_freq',
                        'recent_100_ratio', 'recent_50_freq', 'recent_50_ratio', 'trend'])
        for bit in range(1, 46):
            all_freq = total_bit_freq.get(bit, 0)
            r100_freq = recent_100_bit_freq.get(bit, 0)
            r50_freq = recent_50_bit_freq.get(bit, 0)

            all_ratio = all_freq / (total_rounds * 6)
            r100_ratio = r100_freq / (100 * 6) if r100_freq else 0
            r50_ratio = r50_freq / (50 * 6) if r50_freq else 0

            if r50_ratio > all_ratio * 1.3:
                trend = 'HOT'
            elif r50_ratio < all_ratio * 0.7:
                trend = 'COLD'
            else:
                trend = 'STABLE'

            writer.writerow([bit, all_freq, round(all_ratio*100, 2), r100_freq,
                           round(r100_ratio*100, 2), r50_freq, round(r50_ratio*100, 2), trend])

    # ========== 10. ML용 피처 매트릭스 (간소화) ==========
    with open(OUTPUT_DIR / "ml_feature_matrix.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 헤더: round, 270 비트 (ord1_bit1~45, ord2_bit1~45, ...)
        header = ['round']
        for pos in range(1, 7):
            for bit in range(1, 46):
                header.append(f'ord{pos}_b{bit}')
        writer.writerow(header)

        for r in data:
            row = [r['round']]
            for pos in range(1, 7):
                vec = to_onehot(r[f'ord{pos}'])
                row.extend(vec)
            writer.writerow(row)

    # ========== Summary ==========
    with open(OUTPUT_DIR / "summary.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'description'])
        writer.writerow(['total_rounds', total_rounds, '분석 회차 수'])
        writer.writerow(['bits_per_round', 270, '회차당 비트 수 (6×45)'])
        writer.writerow(['active_bits_per_round', 6, '회차당 활성 비트'])
        writer.writerow(['sparsity', f"{sparsity_stats['total_sparsity']}%", '희소성'])
        writer.writerow(['most_freq_bit', total_bit_freq.most_common(1)[0][0], '최다 빈도 비트'])
        writer.writerow(['least_freq_bit', total_bit_freq.most_common()[-1][0], '최소 빈도 비트'])
        writer.writerow(['freq_range', f"{total_bit_freq.most_common()[-1][1]}-{total_bit_freq.most_common(1)[0][1]}", '빈도 범위'])
        writer.writerow(['most_freq_pair', f"{bit_pair_freq.most_common(1)[0][0]}", '최다 빈도 비트쌍'])
        writer.writerow(['hot_bits_count', len([b for b, f in total_bit_freq.items() if f > statistics.mean(freq_values) + freq_std]), '핫 비트 수'])
        writer.writerow(['cold_bits_count', len([b for b, f in total_bit_freq.items() if f < statistics.mean(freq_values) - freq_std]), '콜드 비트 수'])

    # ========== 출력 ==========
    print("=" * 70)
    print("One-Hot Encoding 전문가 인사이트")
    print("=" * 70)
    print(f"\n총 {total_rounds}회차 분석\n")

    print("[1. 원핫인코딩 구조]")
    print(f"    포지션당 비트 수: 45")
    print(f"    총 비트 수: 270 (6 포지션 × 45 비트)")
    print(f"    활성 비트: 6개/회차")
    print(f"    희소성: {sparsity_stats['total_sparsity']}%")

    print("\n[2. 포지션별 가장 빈번한 비트]")
    for pos in range(1, 7):
        top3 = bit_freq_by_pos[pos].most_common(3)
        print(f"    ord{pos}: {', '.join([f'{b}({c}회)' for b, c in top3])}")

    print("\n[3. 전체 핫 비트 (상위 10개)]")
    for bit, freq in total_bit_freq.most_common(10):
        ratio = freq / (total_rounds * 6) * 100
        print(f"    비트 {bit:2d}: {freq:3d}회 ({ratio:5.2f}%)")

    print("\n[4. 전체 콜드 비트 (하위 10개)]")
    for bit, freq in total_bit_freq.most_common()[-10:]:
        ratio = freq / (total_rounds * 6) * 100
        print(f"    비트 {bit:2d}: {freq:3d}회 ({ratio:5.2f}%)")

    print("\n[5. 가장 빈번한 비트 쌍 (상위 5개)]")
    for (b1, b2), freq in bit_pair_freq.most_common(5):
        print(f"    ({b1:2d}, {b2:2d}): {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[6. 클러스터별 분포]")
    for cluster in ['C1(01-09)', 'C2(10-18)', 'C3(19-27)', 'C4(28-36)', 'C5(37-45)']:
        total = sum(cluster_by_pos[pos].get(cluster, 0) for pos in range(1, 7))
        print(f"    {cluster}: {total:4d}회 ({total/(total_rounds*6)*100:5.1f}%)")

    print("\n[7. 연속 비트 패턴]")
    for count in sorted(consecutive_bits_freq.keys()):
        freq = consecutive_bits_freq[count]
        print(f"    연속 {count}쌍: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[8. 비트 간격 분포 (상위 5개)]")
    for gap, freq in sorted(bit_gap_freq.items(), key=lambda x: -x[1])[:5]:
        print(f"    간격 {gap:2d}: {freq:3d}회 ({freq/sum(bit_gap_freq.values())*100:5.1f}%)")

    print("\n[9. 최근 트렌드]")
    trending_hot = []
    trending_cold = []
    for bit in range(1, 46):
        all_ratio = total_bit_freq.get(bit, 0) / (total_rounds * 6)
        r50_ratio = recent_50_bit_freq.get(bit, 0) / (50 * 6)
        if r50_ratio > all_ratio * 1.3:
            trending_hot.append((bit, r50_ratio * 100, all_ratio * 100))
        elif r50_ratio < all_ratio * 0.7:
            trending_cold.append((bit, r50_ratio * 100, all_ratio * 100))

    print("    [HOT 트렌드]")
    for bit, r50, all_r in sorted(trending_hot, key=lambda x: -x[1])[:5]:
        print(f"        비트 {bit:2d}: 최근50회 {r50:.1f}% (전체 {all_r:.1f}%) ↑")

    print("    [COLD 트렌드]")
    for bit, r50, all_r in sorted(trending_cold, key=lambda x: x[1])[:5]:
        print(f"        비트 {bit:2d}: 최근50회 {r50:.1f}% (전체 {all_r:.1f}%) ↓")

    print("\n" + "=" * 70)
    print(f"CSV 파일 저장 완료: {OUTPUT_DIR}")
    print("ML 피처 매트릭스: ml_feature_matrix.csv (270 features)")
    print("=" * 70)


if __name__ == "__main__":
    generate_insight()
