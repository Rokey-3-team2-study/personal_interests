# RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

[link](https://robotics-transformer2.github.io/)

### ✅ **Why: 연구 목적 및 이유**
- **로봇의 일반화 능력 향상**: 기존 로봇 정책은 좁은 범위의 데이터에 한정되어 일반화 능력이 부족함.
- **웹 기반 VLM의 지식 활용**: 인터넷 규모의 시각-언어 모델(VLM)이 지닌 **지식과 추론 능력**을 **로봇 제어에 직접 이식**함으로써, 새로운 명령·객체·환경에 대한 **일반화 및 추론 능력 향상**을 목표로 함.

### ✅ **What: 연구의 기여**
1. **RT-2 모델 제안**:
   - 인터넷 기반 VLM을 로봇 제어에 통합한 **Vision-Language-Action (VLA)** 모델.
   - 로봇 동작을 **텍스트 토큰**으로 표현하여 VLM 훈련 포맷에 통합.

2. **Co-Fine-Tuning 전략**:
   - 로봇 데이터와 웹 데이터(VQA 등)를 **동시에 미세조정**해 일반화 성능 향상.

3. **Emergent Capabilities 관찰**:
   - 훈련에 없던 개념(예: 숫자 근처에 물건 놓기, 유사 객체 추론, 즉흥 도구 선택 등)을 **추론하여 행동**하는 능력 실험적으로 입증.

4. **최대 55B 파라미터 모델을 실시간 제어에 활용**:
   - 클라우드 TPU를 통한 실시간 로봇 제어 구현(1–3 Hz).

### ✅ **How: 연구의 방법론**
1. **모델 구조**:
   - 기존 VLM (PaLI-X, PaLM-E 등)을 기반으로 구성.
   - 로봇 동작은 8개의 디스크리트 텍스트 토큰으로 표현됨 (ex: 6-DoF + gripper + 종료신호).

2. **훈련 전략**:
   - **Co-fine-tuning**: 로봇 데이터와 웹 기반 시각-언어 데이터(VQA, caption 등)를 함께 학습.
   - 행동 출력을 위한 **토큰 제한(Valid Token Constraint)** 적용으로 로봇 제어 정확성 보장.

3. **실험 및 평가**:
   - 총 6,000개의 로봇 평가 사례.
   - **기존 모델(RT-1, VC-1, R3M, MOO 등)과 비교**: 새로운 객체·배경·환경·명령에 대해 RT-2가 **2~6배 성능 향상**.
   - **Emergent Evaluation**:
     - 심볼 이해, 다단계 추론, 사람 인식, 수학적 판단 등의 능력을 실험적으로 검증.
   - **Chain-of-Thought Reasoning**:
     - 행동 전 **계획 문장**을 생성하고 그 후 행동 토큰을 생성함으로써 복잡한 지시 수행 향상.

4. **실시간 제어**:
   - 최대 55B 파라미터 모델은 TPU 클라우드 서비스에서 구동.
   - 소형 모델(5B)은 5 Hz로 동작.

---
### 1. Introduction

1.1 배경 및 연구 목적
- 최근 대규모 비전-언어 모델(Vision-Language Models, VLMs) 은 텍스트와 이미지 이해를 기반으로 강력한 일반화 능력(= 텍스트 생성, 개념적 추론, 문제 해결, 시각적 의미 이해)을 보이고 있음
- 하지만, 로봇 제어(Robotic Control) 로 확장하는 것은 어려운 과제
    1. 데이터 부족 : 웹 데이터와 달리 로봇 학습 데이터는 수집이 어렵고 제한적적
    2. 물리적 세계의 복잡성 : 로봇은 단순한 인식이 아니라 물리적인 상호작용이 가능한 행동을 학습해야 함
    3. 웹 기반 지식 활용 어려움 : 기존 VLMs는 주어진 명령을 해석할 수 있지만, 이를 실제 로봇 행동으로 변환하는 것은 별개의 문제

1.2 연구 목표
1. 웹 데이터 기반의 일반화된 지식 활용
    - 기존의 VLM들이 웹에서 학습한 객체 인식, 관계 추론, 의미 이해 능력을 로봇 제어에도 적용

2. 로봇 행동을 VLM의 일부로 통합
    - 기존 로봇 모델들은 언어 해석과 행동 예측이 분리되어 있었지만,
    - RT-2는 로봇 행동을 텍스트 토큰(token) 형식으로 변환하여 VLM이 자연스럽게 행동을 출력하도록 설계

3. Zero-shot 및 Few-shot 일반화 가능
    - 기존 로봇 모델들은 특정한 작업(Task)에 대해 학습된 후에야 실행 가능
    - RT-2는 웹에서 학습한 개념을 활용하여 새로운 환경과 사물에 대해 제로샷(Zero-shot)으로 수행 가능

1.3 RT-2의 핵심 기여
1. VLM을 로봇 제어로 직접 확장
2. 비전-언저 데이터를 활용한 행동 학습
3. Zero-shot 및 Multi-task 수행 능력

1.4 기존 연구와 차별점
- RT-2는 기존 연구들과 비교해 **로봇 행동을 자연어 모델과 동일한 방식으로 표현하는 비전-언어-행동(Vision-Language-Action, VLA)** 모델을 최초로 도입

### 2. Related Work

2.1 Vision-Language Models (VLMs)
1. Representation-learning models : CLIP (Contrastive Language-Image Pre-Training) – 이미지와 텍스트를 공통 임베딩 공간에 맵핑

<div align="center">

![CLIP](/hojun/images/(RT-2)%20Figure_1.png)

</div>

2. Vision-to-text generation models : Flamingo, PaLI – 시각 정보와 언어를 입력받아 텍스트 생성

<div align="center">

![Flamingo](/hojun/images/(RT-2)%20Figure_2.png)

</div>

- 이런 모델들은 이미지 캡셔닝, VQA(Visual Question Answering), 자유로운 언어 생성에 능숙하고,
- RT-2는 이 중 텍스트 생성 계열을 활용하여 로봇 액션을 직접 생성하는 방식으로 확장
- VLM 자체가 로봇 제어(액션)까지 담당하도록 훈련시켜, 하나의 통합 모델을 구성

2.2 Generalization in Robot Learning
- 기존 로봇 연구들은 일반화 능력 부족이 주요한 한계
- 웹 기반 사전학습(VLM) 모델로부터 일반화된 개념과 표현을 로봇 제어로 직접 이전함으로써, 보다 광범위한 일반화(객체, 장면, 언어 지시어, 추론 등)를 가능케 함.

2.3 Pre-training for Robotic Manipulation
- 기존 로봇 분야에서의 사전학습(pre-training)은 주로 시각 표현
- 최근에는 언어 모델 또는 VLM을 계획자(planner)로 활용하는 접근
- RT-2는 VLM의 출력 자체를 로봇 액션 토큰으로 확장 = 시각+언어+행동까지 단일 모델 내에서 통합

### 3. Vision-Language-Action Models

<div align="center">

![alt text](/hojun/images/(RT-2)%20Figure_3.png)

</div>

3.1 Pre-trained Vision-Language Models
- RT-2는 기존의 대규모 비전-언어 모델(VLM)인 PaLI-X와 PaLM-E를 기반으로 함
- 이들 모델은 원래 시각적 질문 응답(VQA), 캡셔닝, 이미지-텍스트 대응 등 인터넷 규모의 데이터로 훈련
- RT-2는 이 모델들을 기반으로 로봇 제어 능력을 추가하여 VLA (Vision-Language-Action) 모델로 확장

3.2 Robot-Action Fine-tuning
- 로봇 액션을 텍스트 토큰으로 표현해 언어 출력처럼 학습 가능
- 액션 공간
   - 6-DoF end-effector 위치 및 회전
   - gripper의 확장 정도
   - 종료 명령 (terminate)
- 각 액션 차원을 256개로 이산화 하여 정수 시퀀스로 표현
- 액션 토큰을 기존의 언어 토큰과 동일하게 모델에 입력/출력하여 훈련

- Co-Fine-Tuning (공동 미세조정)
   - 로봇 데이터만으로 미세조정하면 성능이 제한되므로, 원래의 웹 데이터(VQA 등)와 로봇 시연 데이터를 동시에 사용해 학습
   - 훈련 배치 내에서 로봇 데이터의 샘플링 가중치를 증가시켜 균형 유지

- Output Constraints
   - 로봇 명령 수행 시에는 출력 가능한 토큰을 액션 토큰으로 제한
   - 이를 통해 모델이 실행 가능한 액션만 생성하도록 유도
   - 일반 VLM 태스크에서는 전체 어휘 출력이 가능하지만,
   - 로봇 제어 시에는 액션 토큰만 사용

3.3 Real-Time Inference
- 55B 파라미터 모델을 직접 로봇에 내장하기엔 너무 무거움
- 클라우드 TPU 인프라에 모델을 띄우고 로봇이 네트워크를 통해 질의 → 응답 받는 방식
- RT-2는 1–3Hz의 제어 주파수(실시간 제어 수준)를 달성함
- 더 작은 5B 모델은 약 5Hz의 속도로 동작 가능

### 4. Experiments

4.1 목적
- RT-2가 기존 학습 태스크는 물론이고, 새로운 물체/배경/환경에 대해 얼마나 잘 일반화하는가?
- 웹에서 학습한 지식으로부터 어떤 emergent capability가 발생하는가?
- 모델 크기 및 학습 방식이 일반화 성능에 어떤 영향을 미치는가?
- RT-2도 체인 오브 소트(chain-of-thought) reasoning을 보여줄 수 있는가?

4.2 실험 구성 요약
1. Generalization 성능 측정
2. Emergent Capabilities 분석
3. Ablation: 모델 크기 및 학습 전략 영향
4. Chain-of-Thought Reasoning

### 5. Limitaions

- 물리적 기술 한계
   - **완전히 새로운 동작(motion) 을 생성하거나 학습하지는 못한다**
   - 로봇 시연 데이터에서 본 동작 분포 안에서만 물리적 행동이 가능
   - 로봇 데이터의 스킬 다양성 부족에 기인하며, 향후 사람의 비디오 등을 활용한 새로운 데이터 수집 방식이 필요하다고 지적
- 고비용/낮은 추론 속도
   - RT-2는 대규모 VLM 기반이기 때문에, 추론을 위한 계산 비용이 매우 높음
   - 로봇이 고빈도 실시간 제어를 해야 하는 상황에서는 추론 속도 자체가 병목
   - 해결법 : 경량화(quantization), 지식 증류(distillation)
- VLM 모델 접근성 제한

### 6. Conclusion

- RT-2는 vision-language-action (VLA) 모델로,
   - **웹 기반 대규모 비전-언어 모델**(VLM)을 로봇 제어에 직접적으로 통합하는 접근을 제시
   - 로봇 데이터를 VLM과 함께 co-fine-tuning 하여, 로봇 행동을 텍스트 토큰으로 표현하고 예측할 수 있게 훈련
- 이 방법은 단순하면서도 강력한 일반화 성능을 보여주며, 로봇 정책(policy)의 품질 향상
- 이 접근은 VLM의 발전이 로봇 제어에 실질적인 이점을 줄 수 있다는 것을 보여주며, 로봇 학습 연구가 VLM의 발전과 함께 상호 발전할 수 있음을 시사

### Appendices