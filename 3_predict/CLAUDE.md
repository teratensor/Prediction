# 3_predict - 예측 모듈 (레거시)

**참고**: 현재 main.py에서 통합 처리됨. 이 폴더의 스크립트들은 레거시 버전.

## 폴더 구조

| 폴더 | 설명 | main.py 함수 |
|------|------|-------------|
| 1_firstend | (ord1, ord6) 쌍 생성 | `generate_firstend_pairs()` |
| 2_146 | ord4 계산 | `fill_ord4()` |
| 3_235 | ord2, ord3, ord5 채우기 | `fill_ord235()` |

## 현재 파이프라인 (main.py v1.01)

```
1. generate_firstend_pairs() → 477개 쌍
2. fill_ord4()              → ~9,000개 행 (±10 offset)
3. fill_ord235()            → ~560만개 조합 (top_n=15)
```

## 실행

```bash
# 통합 예측 (권장)
python main.py --round 1205

# 백테스트
python main.py --backtest --start 900 --end 1000

# 매치 분포 분석 (NEW in v1.01)
python main.py --distribution --start 1195 --end 1204
```

## 현재 설정 (main.py)

| 파라미터 | 값 |
|----------|-----|
| FORMULA_RATIO | 0.60 |
| OFFSET_RANGE | ±10 |
| TOP_N | 15 |
