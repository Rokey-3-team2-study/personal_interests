# Robotics Transformer for Real-World Control at Scale
[link](https://robotics-transformer1.github.io/)

### 1. Introduction
1.1 연구 배경 및 동기
- 기존 연구들은 여러 가지 방식으로 로봇 학습을 시도했으나, 한계를 가짐
    - Single-Task Learning → 확장성 부족
    - Multi-Task Learning → 새로운 작업에 대한 적응력 부족
    - 로봇 데이터는 수집 비용이 높고, 다양성이 부족하여 실제 환경에서의 적응력이 떨어짐.

1.2 문제 정의 및 연구 목표
- 로봇을 위한 범용 모델을 개발
    - 다양한 로봇 작업을 하나의 모델이 수행할 수 있는 능력.
    - 새로운 작업, 환경, 객체에 대한 generalization 능력 향상.
    - zero-shot 또는 few-shot으로 새로운 작업 수행 가능.
- 해결해야 할 두 가지 핵심 요소
    - 데이터 : 기존 로봇 학습은 특정 작업을 위한 데이터 수집에 의존적 → 다양한 환경에서 학습할 수 있는 광범위한 데이터셋 필요.
    - 모델 : 로봇 정책을 학습하기 위해서는 high-capacity모델이 필요하지만, 동시에 실시간 제어가 가능해야 함.

1.3 연구 기여
- 대규모 데이터셋 활용
- Transformer 기반 아키텍처 적용
- 실시간 실행 가능
- 강력한 일반화 성능

### 2. Related Work
2.1 Transformer 기반 로봇 정책 학습
- 기존 연구들은 Transformer를 활용하여 자연어 명령을 해석하고 로봇 제어 정책을 학습하려고 시도함.
- 그러나 대부분 시뮬레이션 데이터에 집중하거나, 특정 작업에만 한정된 모델을 학습함.

2.2 범용 로봇 정책 연구
- Gato (Reed et al., 2022): 멀티태스킹 AI로 다양한 작업을 수행하지만, 로봇 작업 범위가 제한적이고 실시간성 부족.
- Perceiver-Actor (Shridhar et al., 2022): 비전-언어 모델을 활용하지만, 고속 실시간 제어 불가능.

2.3. 대규모 데이터 기반 로봇 학습
- 기존 연구들은 보통 작은 규모의 특정 로봇 작업 데이터를 사용하여 훈련됨.

2.4. 로봇 행동 정책을 위한 Transformer 모델
- 기존 Transformer 기반 로봇 모델들은 효율적인 실시간 추론을 제공하지 못함.

2.5. 본 연구의 차별점
- RT-1은 대규모 실세계 데이터에서 학습한 Transformer 모델로, 다양한 로봇 작업을 수행할 수 있도록 최적화되었으며, 기존 연구들보다 더 높은 일반화 성능과 실시간 제어 속도를 제공한다는 점에서 차별화됨.

### 3. Preliminaries

### 4. System Overview

### 5. RT-1: Robotics Transformer

### 6. Experiments

### 7. Conclusions, Limitations and Future Work

### Appendix