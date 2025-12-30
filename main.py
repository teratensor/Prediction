"""
로또 예측 시스템 - ML 기반 메인 실행 파일

전체 파이프라인:
1. ord1, ord6 예측 (13_ord1ord6_ml v2 앙상블) - Top-100
2. ord4 예측 (11_ord4_ml)
3. ord2 예측 (9_ord2_ml)
4. ord3 예측 (10_ord3_ml)
5. ord5 예측 (12_ord5_ml)
6. 결과 저장 및 당첨번호 비교

실행:
    python main.py --round 1204
    python main.py --round 1204 --top_firstend 100 --top_inner 10
"""

import csv
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"
RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)

# 상수
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
RANGES = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 45)]


def get_range_index(num: int) -> int:
    for i, (start, end) in enumerate(RANGES):
        if start <= num <= end:
            return i
    return 4


def load_winning_numbers() -> List[Dict]:
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
    return sorted(results, key=lambda x: x['round'])


# ============================================================
# ord1, ord6 예측 (v2 앙상블)
# ============================================================

def get_all_pairs() -> List[Tuple[int, int]]:
    """모든 가능한 (ord1, ord6) 쌍"""
    pairs = []
    for ord1 in range(1, 41):
        for ord6 in range(ord1 + 5, 46):
            pairs.append((ord1, ord6))
    return pairs

ALL_PAIRS = get_all_pairs()

FIRSTEND_FEATURES = [
    'ord1', 'ord6', 'span',
    'ord1_freq', 'ord1_is_prime', 'ord1_is_odd',
    'ord1_recent_5', 'ord1_recent_10', 'ord1_recent_20', 'ord1_gap',
    'ord6_freq', 'ord6_is_prime', 'ord6_is_odd',
    'ord6_recent_5', 'ord6_recent_10', 'ord6_recent_20', 'ord6_gap',
    'pair_freq', 'pair_recent_20', 'span_freq', 'span_recent_10',
    'ord6_given_ord1_freq', 'ord1_given_ord6_freq',
    'ord1_trend', 'ord6_trend', 'span_trend',
    'sum_ord1_ord6', 'both_prime', 'both_odd', 'parity_match',
]


class FirstEndPredictor:
    """ord1, ord6 쌍 예측기 (v2 앙상블)"""

    def __init__(self, train_data: List[Dict]):
        self.train_data = train_data
        self.n = len(train_data)
        self._compute_statistics()

    def _compute_statistics(self):
        self.ord1_freq = Counter(r['ord1'] for r in self.train_data)
        self.ord6_freq = Counter(r['ord6'] for r in self.train_data)
        self.pair_freq = Counter((r['ord1'], r['ord6']) for r in self.train_data)
        self.span_freq = Counter(r['ord6'] - r['ord1'] for r in self.train_data)

        self.ord1_recent = {5: Counter(), 10: Counter(), 20: Counter()}
        self.ord6_recent = {5: Counter(), 10: Counter(), 20: Counter()}
        self.pair_recent_20 = Counter()
        self.span_recent_10 = Counter()

        for i, r in enumerate(self.train_data):
            if i >= self.n - 5:
                self.ord1_recent[5][r['ord1']] += 1
                self.ord6_recent[5][r['ord6']] += 1
            if i >= self.n - 10:
                self.ord1_recent[10][r['ord1']] += 1
                self.ord6_recent[10][r['ord6']] += 1
                self.span_recent_10[r['ord6'] - r['ord1']] += 1
            if i >= self.n - 20:
                self.ord1_recent[20][r['ord1']] += 1
                self.ord6_recent[20][r['ord6']] += 1
                self.pair_recent_20[(r['ord1'], r['ord6'])] += 1

        self.ord1_last = {}
        self.ord6_last = {}
        for i, r in enumerate(self.train_data):
            self.ord1_last[r['ord1']] = i
            self.ord6_last[r['ord6']] = i

        self.ord6_given_ord1 = defaultdict(Counter)
        self.ord1_given_ord6 = defaultdict(Counter)
        for r in self.train_data:
            self.ord6_given_ord1[r['ord1']][r['ord6']] += 1
            self.ord1_given_ord6[r['ord6']][r['ord1']] += 1

        if self.n >= 20:
            r10 = Counter(r['ord1'] for r in self.train_data[-10:])
            p10 = Counter(r['ord1'] for r in self.train_data[-20:-10])
            self.ord1_trend = {v: r10.get(v, 0) - p10.get(v, 0) for v in range(1, 46)}

            r10 = Counter(r['ord6'] for r in self.train_data[-10:])
            p10 = Counter(r['ord6'] for r in self.train_data[-20:-10])
            self.ord6_trend = {v: r10.get(v, 0) - p10.get(v, 0) for v in range(1, 46)}

            self.span_trend = {}
            for span in range(5, 45):
                rc = sum(1 for r in self.train_data[-10:] if r['ord6'] - r['ord1'] == span)
                pc = sum(1 for r in self.train_data[-20:-10] if r['ord6'] - r['ord1'] == span)
                self.span_trend[span] = rc - pc
        else:
            self.ord1_trend = defaultdict(int)
            self.ord6_trend = defaultdict(int)
            self.span_trend = defaultdict(int)

    def extract_features(self, ord1: int, ord6: int) -> np.ndarray:
        span = ord6 - ord1
        f = {}
        f['ord1'] = ord1
        f['ord6'] = ord6
        f['span'] = span
        f['ord1_freq'] = self.ord1_freq.get(ord1, 0) / self.n
        f['ord1_is_prime'] = 1 if ord1 in PRIMES else 0
        f['ord1_is_odd'] = ord1 % 2
        f['ord1_recent_5'] = self.ord1_recent[5].get(ord1, 0) / 5
        f['ord1_recent_10'] = self.ord1_recent[10].get(ord1, 0) / 10
        f['ord1_recent_20'] = self.ord1_recent[20].get(ord1, 0) / 20
        f['ord1_gap'] = (self.n - self.ord1_last.get(ord1, 0)) / self.n
        f['ord6_freq'] = self.ord6_freq.get(ord6, 0) / self.n
        f['ord6_is_prime'] = 1 if ord6 in PRIMES else 0
        f['ord6_is_odd'] = ord6 % 2
        f['ord6_recent_5'] = self.ord6_recent[5].get(ord6, 0) / 5
        f['ord6_recent_10'] = self.ord6_recent[10].get(ord6, 0) / 10
        f['ord6_recent_20'] = self.ord6_recent[20].get(ord6, 0) / 20
        f['ord6_gap'] = (self.n - self.ord6_last.get(ord6, 0)) / self.n
        f['pair_freq'] = self.pair_freq.get((ord1, ord6), 0) / self.n
        f['pair_recent_20'] = self.pair_recent_20.get((ord1, ord6), 0) / 20
        f['span_freq'] = self.span_freq.get(span, 0) / self.n
        f['span_recent_10'] = self.span_recent_10.get(span, 0) / 10

        og = self.ord6_given_ord1[ord1]
        t = sum(og.values())
        f['ord6_given_ord1_freq'] = og.get(ord6, 0) / t if t > 0 else 0

        og = self.ord1_given_ord6[ord6]
        t = sum(og.values())
        f['ord1_given_ord6_freq'] = og.get(ord1, 0) / t if t > 0 else 0

        f['ord1_trend'] = self.ord1_trend.get(ord1, 0)
        f['ord6_trend'] = self.ord6_trend.get(ord6, 0)
        f['span_trend'] = self.span_trend.get(span, 0)
        f['sum_ord1_ord6'] = ord1 + ord6
        f['both_prime'] = 1 if (ord1 in PRIMES and ord6 in PRIMES) else 0
        f['both_odd'] = 1 if (ord1 % 2 == 1 and ord6 % 2 == 1) else 0
        f['parity_match'] = 1 if (ord1 % 2 == ord6 % 2) else 0

        return np.array([f.get(c, 0) for c in FIRSTEND_FEATURES])

    def freq_score(self, ord1: int, ord6: int) -> float:
        score = 0
        span = ord6 - ord1
        score += (self.ord1_freq.get(ord1, 0) / self.n) * 30
        score += (self.ord6_freq.get(ord6, 0) / self.n) * 30
        score += (self.span_freq.get(span, 0) / self.n) * 20
        score += (self.pair_freq.get((ord1, ord6), 0) / self.n) * 100
        score += (self.ord1_recent[20].get(ord1, 0) / 20) * 5
        score += (self.ord6_recent[20].get(ord6, 0) / 20) * 5
        return score


# ============================================================
# ord2, ord3, ord4, ord5 예측기
# ============================================================

class InnerPositionPredictor:
    """ord2, ord3, ord4, ord5 공통 예측기"""

    def __init__(self, train_data: List[Dict]):
        self.train_data = train_data
        self.n = len(train_data)
        self._compute_statistics()

    def _compute_statistics(self):
        self.ball_freq = Counter()
        self.pos_freq = {i: Counter() for i in range(1, 7)}
        for r in self.train_data:
            for i in range(1, 7):
                ball = r[f'ord{i}']
                self.ball_freq[ball] += 1
                self.pos_freq[i][ball] += 1

        recent = self.train_data[-10:]
        freq = Counter()
        for r in recent:
            for i in range(1, 7):
                freq[r[f'ord{i}']] += 1
        sorted_nums = sorted(range(1, 46), key=lambda x: -freq.get(x, 0))
        self.hot_bits = set(sorted_nums[:10])
        self.cold_bits = set(sorted_nums[-10:])

        recent_3 = self.train_data[-3:]
        self.recent_balls = set()
        for r in recent_3:
            for i in range(1, 7):
                self.recent_balls.add(r[f'ord{i}'])

        self.range_freq = {i: Counter() for i in range(2, 6)}
        for r in self.train_data:
            for i in range(2, 6):
                self.range_freq[i][get_range_index(r[f'ord{i}'])] += 1

    def get_common_features(self, candidate: int, pos: int) -> Dict:
        f = {}
        f['candidate'] = candidate
        f['range_idx'] = get_range_index(candidate)

        total = sum(self.pos_freq[pos].values())
        f[f'pos{pos}_freq'] = self.pos_freq[pos].get(candidate, 0) / total if total > 0 else 0

        f['is_hot'] = 1 if candidate in self.hot_bits else 0
        f['is_cold'] = 1 if candidate in self.cold_bits else 0
        f['is_prime'] = 1 if candidate in PRIMES else 0

        total = sum(self.ball_freq.values())
        f['overall_freq'] = self.ball_freq.get(candidate, 0) / total if total > 0 else 0
        f['in_recent_3'] = 1 if candidate in self.recent_balls else 0

        return f


# ============================================================
# 메인 예측 함수
# ============================================================

def predict_round(target_round: int, top_firstend: int = 100, top_inner: int = 10):
    """특정 회차 예측"""
    print("=" * 70)
    print(f"ML 기반 로또 예측: {target_round}회차")
    print("=" * 70)

    data = load_winning_numbers()
    train_data = [r for r in data if r['round'] < target_round]
    target = next((r for r in data if r['round'] == target_round), None)

    print(f"학습 데이터: {len(train_data)}회차")

    if target:
        actual = (target['ord1'], target['ord2'], target['ord3'],
                  target['ord4'], target['ord5'], target['ord6'])
        print(f"실제 당첨번호: {actual}")

    # ========================================
    # Step 1: ord1, ord6 예측 (Top-N)
    # ========================================
    print(f"\n[Step 1] ord1, ord6 예측 (Top-{top_firstend})...")

    fe_predictor = FirstEndPredictor(train_data)

    # 학습
    train_window = train_data[-100:] if len(train_data) > 100 else train_data
    X_fe, y_fe = [], []
    for j, tr in enumerate(train_window):
        if j < 10:
            continue
        prev = train_window[:j]
        ext = FirstEndPredictor(prev)
        for pair in ALL_PAIRS:
            X_fe.append(ext.extract_features(pair[0], pair[1]))
            y_fe.append(1 if (pair[0] == tr['ord1'] and pair[1] == tr['ord6']) else 0)

    X_fe = np.array(X_fe)
    y_fe = np.array(y_fe)

    if HAS_XGBOOST:
        fe_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.08,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'
        )
        fe_model.fit(X_fe, y_fe)

    # 예측
    X_test = np.array([fe_predictor.extract_features(p[0], p[1]) for p in ALL_PAIRS])
    freq_scores = np.array([fe_predictor.freq_score(p[0], p[1]) for p in ALL_PAIRS])

    if HAS_XGBOOST:
        xgb_probs = fe_model.predict_proba(X_test)[:, 1]
    else:
        xgb_probs = np.zeros(len(ALL_PAIRS))

    # 정규화 및 앙상블
    if freq_scores.max() > freq_scores.min():
        freq_probs = (freq_scores - freq_scores.min()) / (freq_scores.max() - freq_scores.min())
    else:
        freq_probs = np.ones_like(freq_scores) / len(freq_scores)

    ensemble_weight = 0.3
    final_probs = (1 - ensemble_weight) * xgb_probs + ensemble_weight * freq_probs

    sorted_idx = np.argsort(-final_probs)
    top_pairs = [ALL_PAIRS[i] for i in sorted_idx[:top_firstend]]

    print(f"  Top-{top_firstend} (ord1, ord6) 쌍 선택 완료")

    # ========================================
    # Step 2-5: ord2, ord3, ord4, ord5 예측
    # ========================================
    print(f"\n[Step 2-5] ord2, ord3, ord4, ord5 예측 (각 Top-{top_inner})...")

    inner_pred = InnerPositionPredictor(train_data)

    # 각 위치별 모델 학습
    models = {}

    for pos in [2, 3, 4, 5]:
        X_train, y_train = [], []
        for j, tr in enumerate(train_window):
            if j < 10:
                continue

            if pos == 2:
                candidates = list(range(tr['ord1'] + 1, tr['ord6']))
            elif pos == 3:
                candidates = list(range(tr['ord2'] + 1, tr['ord6']))
            elif pos == 4:
                formula = round(tr['ord1'] + (tr['ord6'] - tr['ord1']) * 0.60)
                candidates = list(range(max(tr['ord1'] + 3, formula - 10),
                                        min(tr['ord6'] - 2, formula + 11)))
            else:  # pos == 5
                candidates = list(range(tr['ord4'] + 1, tr['ord6']))

            for cand in candidates:
                feat = inner_pred.get_common_features(cand, pos)
                feat_arr = [feat['candidate'], feat['range_idx'], feat[f'pos{pos}_freq'],
                           feat['is_hot'], feat['is_cold'], feat['is_prime'],
                           feat['overall_freq'], feat['in_recent_3']]
                X_train.append(feat_arr)
                y_train.append(1 if cand == tr[f'ord{pos}'] else 0)

        if len(X_train) > 10 and HAS_XGBOOST:
            models[pos] = xgb.XGBClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'
            )
            models[pos].fit(np.array(X_train), np.array(y_train))

    # ========================================
    # 전체 조합 생성
    # ========================================
    print(f"\n[Step 6] 전체 조합 생성...")

    all_combinations = []

    for ord1, ord6 in top_pairs:
        # ord4 예측
        formula = round(ord1 + (ord6 - ord1) * 0.60)
        ord4_cands = list(range(max(ord1 + 3, formula - 10), min(ord6 - 2, formula + 11)))

        if 4 in models and ord4_cands:
            X4 = []
            for c in ord4_cands:
                feat = inner_pred.get_common_features(c, 4)
                X4.append([feat['candidate'], feat['range_idx'], feat['pos4_freq'],
                          feat['is_hot'], feat['is_cold'], feat['is_prime'],
                          feat['overall_freq'], feat['in_recent_3']])
            probs4 = models[4].predict_proba(np.array(X4))[:, 1]
            top4_idx = np.argsort(-probs4)[:top_inner]
            ord4_list = [ord4_cands[i] for i in top4_idx]
        else:
            ord4_list = ord4_cands[:top_inner]

        # ord2 예측
        ord2_cands = list(range(ord1 + 1, ord6))
        if 2 in models and ord2_cands:
            X2 = []
            for c in ord2_cands:
                feat = inner_pred.get_common_features(c, 2)
                X2.append([feat['candidate'], feat['range_idx'], feat['pos2_freq'],
                          feat['is_hot'], feat['is_cold'], feat['is_prime'],
                          feat['overall_freq'], feat['in_recent_3']])
            probs2 = models[2].predict_proba(np.array(X2))[:, 1]
            top2_idx = np.argsort(-probs2)[:top_inner]
            ord2_list = [ord2_cands[i] for i in top2_idx]
        else:
            ord2_list = ord2_cands[:top_inner]

        for ord4 in ord4_list:
            for ord2 in ord2_list:
                if ord2 >= ord4:
                    continue

                # ord3 예측 (ord2 < ord3 < ord4)
                ord3_cands = list(range(ord2 + 1, ord4))
                if 3 in models and ord3_cands:
                    X3 = []
                    for c in ord3_cands:
                        feat = inner_pred.get_common_features(c, 3)
                        X3.append([feat['candidate'], feat['range_idx'], feat['pos3_freq'],
                                  feat['is_hot'], feat['is_cold'], feat['is_prime'],
                                  feat['overall_freq'], feat['in_recent_3']])
                    probs3 = models[3].predict_proba(np.array(X3))[:, 1]
                    top3_idx = np.argsort(-probs3)[:top_inner]
                    ord3_list = [ord3_cands[i] for i in top3_idx]
                else:
                    ord3_list = ord3_cands[:top_inner]

                # ord5 예측 (ord4 < ord5 < ord6)
                ord5_cands = list(range(ord4 + 1, ord6))
                if 5 in models and ord5_cands:
                    X5 = []
                    for c in ord5_cands:
                        feat = inner_pred.get_common_features(c, 5)
                        X5.append([feat['candidate'], feat['range_idx'], feat['pos5_freq'],
                                  feat['is_hot'], feat['is_cold'], feat['is_prime'],
                                  feat['overall_freq'], feat['in_recent_3']])
                    probs5 = models[5].predict_proba(np.array(X5))[:, 1]
                    top5_idx = np.argsort(-probs5)[:top_inner]
                    ord5_list = [ord5_cands[i] for i in top5_idx]
                else:
                    ord5_list = ord5_cands[:top_inner]

                for ord3 in ord3_list:
                    for ord5 in ord5_list:
                        if ord1 < ord2 < ord3 < ord4 < ord5 < ord6:
                            all_combinations.append((ord1, ord2, ord3, ord4, ord5, ord6))

    print(f"  생성된 조합 수: {len(all_combinations):,}개")

    # ========================================
    # 결과 저장
    # ========================================
    output_path = RESULT_DIR / "ml_result.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6'])
        for comb in all_combinations:
            writer.writerow(comb)
    print(f"\n[저장] {output_path}")

    # ========================================
    # 당첨번호 비교
    # ========================================
    if target:
        print("\n" + "=" * 70)
        print("당첨번호 비교 결과")
        print("=" * 70)

        actual_set = set(actual)
        match_counts = Counter()
        best_match = 0
        best_combs = []

        for comb in all_combinations:
            match = len(set(comb) & actual_set)
            match_counts[match] += 1
            if match > best_match:
                best_match = match
                best_combs = [comb]
            elif match == best_match:
                best_combs.append(comb)

        print(f"\n실제 당첨: {actual}")
        print(f"최고 적중: {best_match}개 일치")
        print(f"최고 조합 수: {len(best_combs)}개")
        if best_combs:
            print(f"최고 조합 예시: {best_combs[0]}")

        print(f"\n[매치 분포]")
        for i in range(7):
            cnt = match_counts.get(i, 0)
            pct = cnt / len(all_combinations) * 100 if all_combinations else 0
            bar = '█' * int(pct / 2)
            print(f"  {i}개 일치: {cnt:>10,}개 ({pct:6.2f}%) {bar}")

        has_6 = match_counts.get(6, 0) > 0
        print(f"\n{'🎯 6개 일치 조합: 있음! ✓' if has_6 else '❌ 6개 일치 조합: 없음'}")

        # 5개 이상 일치 조합 출력
        if best_match >= 5:
            print(f"\n[{best_match}개 일치 조합 목록]")
            for i, comb in enumerate(best_combs[:10]):
                matched = set(comb) & actual_set
                print(f"  {i+1}. {comb} - 일치: {sorted(matched)}")

    return all_combinations


def main():
    parser = argparse.ArgumentParser(description='ML 기반 로또 예측')
    parser.add_argument('--round', type=int, required=True, help='예측할 회차')
    parser.add_argument('--top_firstend', type=int, default=100, help='(ord1,ord6) Top-K (기본: 100)')
    parser.add_argument('--top_inner', type=int, default=10, help='ord2,3,4,5 각 Top-K (기본: 10)')

    args = parser.parse_args()
    predict_round(args.round, args.top_firstend, args.top_inner)


if __name__ == "__main__":
    main()
