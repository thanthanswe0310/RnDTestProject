# RnDTestProject

## 과제 안내

이 github repo를 fork하고 MGsolutions-dev를 Collaborator로 추가하여 과제를 수행하십시오.

이후 서류지원시 자신의 fork한 github repo link를 제시하시면 됩니다.

주의!: github repo link를 제시하지 않으면 서류 불합격 처리됩니다


## 과제 내용

"NASM 기준으로 주어진 데이터를 분석하십시오"

의도는 nasm overhead squat assessment를 기준으로 피측정자를 평가하기 위해 노이즈가 낀 데이터에서 제시된 값을 추정할 수 있는지 확인하는 것입니다

## 과제 시나리오

3D Motion Capture 장비에서 추출된 5명의 동작 데이터가 주어집니다

하지만 센서 오류로 인해 좌표 값에 심한 노이즈가 섞여있습니다

이 데이터들을 평가하고자하는 데이터의 고유 특성을 해치지 않으면서 데이터를 필터링하여 nasm overhead squat assessment 기준으로 피측정자를 평가하십시오

## 요구 사항

### NASM 기준 Overhead squat 평가

nasm overhead squat assessment 중 주어진 데이터로 다음 항목을 평가하십시오

**Anterior View**
- knees move inward, outward
- 무릎 돌아감 각도 출력

**Lateral View**
- low back arch
- 허리 휘어짐 정도 출력
- torso lean forward
- 상체 숙여짐 정도 출력

## 결과물

- 결과 보고서
- 알고리즘 설명 문서
- Python 기반 프로젝트 코드 및 개발 시 사용한 유닛 테스트 코드