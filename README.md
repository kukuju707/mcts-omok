# mcts-omok
**Monte Carlo Tree Search 기반 오목 AI**

Python으로 Monte Carlo Tree Search(MCTS) 알고리즘을 구현하여 오목 게임 AI를 개발했습니다.  
무작위 시뮬레이션과 트리 탐색을 결합해 **최적 수를 선택**하도록 설계했습니다.

---

## 사용 기술

- **언어**: Python  
- **알고리즘**: Monte Carlo Tree Search  
- **기법**: Simulation, Tree Search

---

## 주요 기능

### 1. MCTS (Monte Carlo Tree Search)
- 확률적 시뮬레이션 기반 수 읽기
- 탐색 깊이 제한 후 랜덤 시뮬레이션

### 2. 도메인 규칙 기반 휴리스틱
- 즉승 수(Immediate Win) 찾기
- 상대 즉승 차단
- 쌍사(Double Four) 생성/차단
- 열린3(Open Three) 확장 및 방어

### 3. 클릭 GUI
- `matplotlib` 기반 오목판
- 마우스 클릭 시 가장 가까운 교차점에 자동 착수
- 최근 수 빨간 테두리 표시
- AI 계산 중 상태 표시

---

## 기대 효과
1. 게임 AI 분야에서의 MCTS 동작 원리 이해
2. NLP 분야(다음 단어 예측, 기계 번역 후보 선택, 대화 응답 결정 등)와의 구조적 유사성 체감
3. 확률 기반 탐색 구조 학습 및 실습 경험 축적

---

## 플레이 방법

1.보드 위를 클릭하면 가장 가까운 교차점에 착수됩니다.
2.AI가 자동으로 다음 수를 계산해 둡니다.
3.최근 수에는 빨간 테두리가 표시됩니다.

## 프로젝트 구조
```
mcts-omok/
├── main.py           # MCTS 기반 오목 AI 메인 코드
├── README.md         # 프로젝트 설명 문서
└── screenshot.png    # 실행 화면 캡처
```

