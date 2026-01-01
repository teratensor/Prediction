#!/usr/bin/env python
"""
로또 6/45 예측 시스템 - 메인 진입점

사용법:
    python main.py --round 1205              # 특정 회차 예측
    python main.py --backtest                # 백테스트 실행
    python main.py                           # 다음 회차 예측
"""

import sys
from pathlib import Path

# 3_predict 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "3_predict"))

from main import main

if __name__ == '__main__':
    main()
