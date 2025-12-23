## Assessment Outcome | 평가 결과 - 탄탄스웨

**EN**  
First of all, thank you for providing this assessment opportunity.  
I would like to share my **deliverable outcomes** below.

**KR**  
먼저 본 평가 과제를 제공해 주셔서 진심으로 감사드립니다.  
아래에 본 과제를 통해 제출한 **산출물(Deliverables)** 을 공유드립니다.

---

### Delivered Artifacts | 제출 산출물

1. **Final Result Report | 최종 결과 보고서**  
   **EN**
   - Provided in **both Korean and English versions**
   - Includes quantitative results and interpretation based on the **NASM Overhead Squat Assessment**

   **KR**
   - **한국어 및 영어 버전** 모두 제공  
   - **NASM 오버헤드 스쿼트 평가(NASM Overhead Squat Assessment)** 기준에 따른 정량적 결과 및 해석 포함  

---

2. **Algorithm Explanation Document | 알고리즘 설명 문서**  
   **EN**
   - Provided in **both Korean and English versions**
   - Describes the deterministic biomechanical pipeline, signal-processing methodology, and NASM-aligned decision logic

   **KR**
   - **한국어 및 영어 버전** 모두 제공  
   - 결정론적 생체역학 분석 파이프라인, 신호 처리 방법론, NASM 기준에 정렬된 판단 로직을 상세히 설명  

---

3. **Python-Based Project Source Code | Python 기반 프로젝트 소스 코드**  
   **EN**
   - Main implementation file: `main.py`
   - Fully deterministic, rule-based biomechanical analysis
   - Includes 3D skeletal visualization and NASM classification logic

   **KR**
   - 주요 구현 파일: `main.py`  
   - 완전한 결정론적(rule-based) 생체역학 분석 구조  
   - 3D 스켈레톤 시각화 및 NASM 분류 로직 포함  

---

4. **Unit Test Code | 유닛 테스트 코드**  
   **EN**
   - Python-based unit tests validating:
     - Signal filtering
     - Geometric joint-angle computation
     - NASM rule-based classification
     - 3D skeletal visualization execution

   **KR**
   - Python 기반 유닛 테스트를 통해 다음 항목 검증:
     - 신호 필터링
     - 기하학적 관절 각도 계산
     - NASM 규칙 기반 분류
     - 3D 스켈레톤 시각화 실행 여부

---

## Key Technical Characteristics | 주요 기술적 특징

**EN**
- Deterministic, rule-based NASM Overhead Squat Assessment
- Crenna-aligned zero-phase Butterworth signal filtering
- Vector-based geometric joint-angle modeling
- Phase-specific analysis focused on the squat bottom position
- Fully interpretable numerical outputs with 3D visual validation
- No machine learning used, ensuring transparency and reproducibility

**KR**
- 결정론적(rule-based) NASM 오버헤드 스쿼트 평가 구조
- Crenna 기준에 정렬된 영위상(Zero-phase) Butterworth 신호 필터링
- 벡터 기반 기하학적 관절 각도 모델링
- 스쿼트 최하단 구간에 집중한 동작 단계별 분석
- 3D 시각화를 포함한 완전한 해석 가능 수치 출력
- 머신러닝을 사용하지 않아 높은 투명성과 재현성 확보

---

## Repository Structure | 저장소 구조  
**Owner: Than Than Swe (탄탄스웨)**

```text
RnDTestProject/
├── data/                  # Input motion-capture datasets | 입력 모션 캡처 데이터
├── 탄탄스웨_output/        # Generated results & visualizations | 결과 및 시각화
├── src/
│   ├── main.py            # Core NASM OHS analysis pipeline
│   └── unit_test.py       # Unit tests
├── docs/
│   ├── Algorithm_EN.pdf
│   ├── Algorithm_KR.pdf
│   ├── Report_EN.pdf
│   └── Report_KR.pdf
└── README.md
