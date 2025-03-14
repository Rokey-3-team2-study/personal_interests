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
3.1 로봇 학습 (Robot Learning)
- RT-1의 목표는 비전과 언어 기반의 로봇 제어 정책을 학습하여, 다양한 작업을 수행할 수 있는 범용 모델을 만드는 것
- 이를 위해, 로봇이 환경에서 데이터를 수집하고 행동을 결정하는 과정을 수학적으로 정의
    - 시점 ${t}$에서 정책 ${\pi}$는 자연어 명령 ${i}$와 이미지 ${x_0}$로 나타냄
    - 정책은 액션 분포 ${\pi (\cdot | i, x_0)}$를 생성하며, 이 분포에서 ${a_0}$를 샘플링하여 로봇에 적용
        - 주어진 조건 ${(i, x_t)}$에서 로봇이 취할 수 있는 모든 액션에 대한 확률분포
    - 한 명령어에 대한 전체 상호작용 : _episode_, ${e = i, { \{(x_t, a_t)\} }^T_{0}}$
    - 에피소드 종료 후, 성공 여부를 나타내는 이진 보상 ${\tau \in \{ 0,1 \}}$
    - 목표는 average reward를 최대화 하는 정책 ${\pi}$학습.

3.2 트랜스포머 모델 (Transformers)
- RT-1은 트랜스포머(Transformer) 기반 정책 모델을 활용하여 비전-언어 정보를 행동으로 변환
    - 입력 시퀀스 ${\{ \xi_h \}^H_{h=0}}$를 받아, 출력 시퀀스 ${\{ y_k \}^K_{k=0}}$로 변환 with Self-attention, FC
    - parameterization : 정책 ${\pi}$ by first mapping inputs ${i, {\{ x_j \}}^t_{j=0}}$ → 입력 시퀀스, 다음 행동 ${a_t}$ → 출력 시퀀스

3.3 모방 학습 (Imitation Learning)
- RT-1은 인간이 직접 수행한 로봇 시연 데이터를 학습하는 모방 학습 방식을 사용
- 기본 원리
    - Dataset 활용 : ${\mathcal{D} = {\{(i^{(n)},{\{(x^{(n)}_{t},a^{(n)}_{t})\}}^{T(n)}_{t=0})\}}^{N}_{n=0}}$
    - Behavioral Cloning으로 ${\pi}$ 학습
- 목표 함수
    - 로봇이 인간의 행동을 모방하도록 로그 우도 최대화
    - ${L(\pi) = - \displaystyle\sum_{(x_i, a_i) \in \mathcal{D}}{\log{\pi(a_i|i, x_i)}}}$

### 4. System Overview
- RT-1은 대규모 데이터 학습을 통해 범용 로봇 학습 모델을 구축하고, 이를 실제 로봇에 적용하여 실시간 제어가 가능하도록 설계하는 것이 목표

4.1 로봇 하드웨어 및 환경
<div align="center">

![Figure 2](/hojun/images/(RT-1)%20Figure_2.png)

</div>
- Everyday Robots의 모바일 매니퓰레이터가 사용 (Fig. 2 (d))
    - 7축 로봇 팔
    - 2-핑거 그리퍼
    - 모바일 베이스
- 훈련 환경 : three kitchen-based environments (Fig. 2 (a,b,c))
    - real office kitchens 1
    - real office kitchens 2
    - a training environment modelled off these real kitchens.

4.2 데이터 수집 및 구성
- RT-1의 학습을 위해 130,000개 이상의 인간 시연 데이터를 활용했으며, 이는 700개 이상의 다양한 로봇 작업을 포함
    - 시연 데이터 + 자연어 명령
    - 작업 : verbs such as “pick”, “open” or “place upright”
    - 목표 : nouns such as “coke can”, “apple”, or “drawer”

4.3 네트워크 아키텍처: RT-1 모델 구조
<div align="center">

![Figure 1](/hojun/images/(RT-1)%20Figure_1(a).png)

</div>
- 입력
    - 비전
    - 자연어 명령
- 모델
    - EfficientNet
        -  a pretrained embedding of the instruction via FiLM
    - TokenLearner
    - Transformer
- 출력 = actions
    - 7D for the arm movement (x, y, z, roll, pitch, yaw, opening of the gripper)
    - 3D for base movement (x, y, yaw)
    - modes (controlling the arm | the base | terminating the episode)

4.4 실시간 제어 및 추론 속도
- RT-1은 3Hz(초당 3번) 속도로 동작하며, 실시간 로봇 제어가 가능하도록 설계
- until : a “terminate” action or hits a pre-set time step limit.

### 5. RT-1: Robotics Transformer

### 6. Experiments

### 7. Conclusions, Limitations and Future Work

### Appendix