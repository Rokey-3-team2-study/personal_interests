# Foundation Models in Robotics

## 2. Foundation Models in Background
<details>
  <summary>Foundation Models in Background</summary>
  <div markdown="1">

### A. Terminology and Mathematical Preliminaries
- Tokenization : 텍스트 쪼개기
- Generative Models **vs** Discriminative Models
  - 생성형 : 데이터의 확률 분포를 학습하고, 이로부터 새로운 샘플을 생성 - GPT, DALL-E
  - 판별형 : 입력 데이터에 대해 분류, 예측 - CNN, ML, ...
- Transformer Architecture : 순차 데이터도 모든 위치 쌍에 대해 병렬로 attention 연산
  - self-attention : 입력 시퀀스 내 토큰들끼리의 관련성 학습
  - multi-head : 어텐션 연산을 병렬로 여러번 수행
  - 위치 인코딩
- Contrastive Learning : 양의 샘플 쌍은 가깝게, 음의 샘플 쌍은 멀어지게
  - CLIP, BLIP : 이미지-텍스트 쌍 데이터 > 로봇 응용
- Diffusion Models
  - 정방향(Forward) 과정 : 데이터에 노이즈를 입힘
  - 역방향(Reverse) 과정 : 노이즈를 제거하고 복원

### B. Large Language Model (LLM) Examples and Historical Context
- 대표 모델
  - BERT, Masked Language Model : 문장 중 일부 단어(토큰)을 마스킹(mask)하고, 이를 예측하도록 학습
  - GPT, Autoregressive Language Model : 앞 단어들이 주어졌을 때 다음 단어를 예측하도록 학습
  - 초거대 모델
    - Fine-tuning & Prompting
    - RLHF(Reinforcement Learning with Human Feedback)
  - 추론 모델 (Reasoning, Test-Time Computing)
    - Chain of Thought
  - 경량화
    - 가지치기, Prunning
    - 양자화, Quantization
    - 지식 증류, Knowledge Distillation
- How LLM can be Foundation Model?
  - Generalization
  - Emergent
  - Plug-and-Play
- 로보틱스 분야와의 접점
  - 로봇에게 고수준 명령(“물건 집어서 옮겨 둬” 등)을 텍스트로 전달하고, 로봇이 이를 적절히 해석·실행하도록 하는 청사진을 제시합니다.
  - 로봇이 언어 명령을 이해하려면, LLM이 가진 개념적 지식, 추론 능력이 중요하고, 이를 로봇의 감각 정보(이미지·센서)와 연동해야 한다는 점에서, LLM은 로봇용 파운데이션 모델과 자연스럽게 연결되는 토대가 되고 있습니다.

### C. Vision Transformers
기존에는 합성곱 신경망(CNN)을 주로 사용하던 이미지 기반 문제들을, Transformer 기반으로 해결하려는 시도가 ViT 계열에서 시작
- 개념
  1. 이미지를 '토큰(패치)' 시퀀스로 처리
    - NLP에서 단어를 ‘토큰(token)’으로 간주하듯, ViT는 이미지를 일정 크기의 패치(Patch)로 분할하고, 각 패치를 ‘토큰’처럼 취급합니다.
    - 이처럼 생성된 패치를 1D로 펼친 뒤, 고정 차원의 벡터로 임베딩(embedding)하여 Transformer에 입력합니다.
  2. 위치 인코딩
  3. 멀티 헤드 셀프 어텐션
  4. 확장성

### D. Multimodal Vision-Language Models (VLMs)
- 개념
  - 비전(Vision)과 언어(Language)를 동시에 다루는 멀티모달 학습 기법
  - 단순히 텍스트만 이해하거나 이미지만 분석해서는 얻을 수 없는 풍부한 ‘상호연결 의미’를 학습하는 것이 목표
  - 이미지에는 언어가 지니는 추상적 의미나 순차적 맥락이 부족할 수 있고, 언어에는 구체적인 시각적 정보가 결핍될 수 있음, 상호보완 목적적
- 예시
  - Contrastive Language-Image Pre-training (CLIP)
    - 대규모 웹 이미지와 해당 이미지에 대한 캡션(설명문) 쌍을 모아, 대조학습(contrastive learning)을 수행합니다.
  - 멀티모달 트랜스포머 기반 학습
    - 시각 모델과 언어 모델을 병렬로 두고, 두 임베딩을 합치는 병합(Merge) 구조나, 교차어텐션(Cross-attention)으로 상호작용하는 구조 등 다양한 아키텍처들이 존재
    - 공통적으로 이미지·텍스트를 같은 차원의 “토큰 시퀀스”로 보고, 트랜스포머로 처리하는 방식이 주가 되며, 이 과정에서 언어-시각 간 의미적 일치를 극대화하는 방향으로 학습
- 로봇에서의 응용 예시
  - 시각적 의미 추출 + 언어 지시 기반 행동
  - 정확한 3D Task에 활용

### E. Embodied Multimodal Language Models
- 개념
  - An embodied agent is an AI system that interacts with a virtual or physical world
  -  Embodied language models are foundation models that incorporate real-world sensor and actuation modalities into pretrained large language models. 
  - 로봇이 언어로 주어진 목표나 설명을 이해하고 실제 동작까지 이어가게 하는 것을 목표로 합니다.
- PaLE-E
  - 구조 : 대규모 언어 모델과 ViT 등을 결합해, 언어·이미지·센서 정보를 동시에 입력받을 수 있도록 설계
  - 학습 : 대규모 LLM을 바탕으로, 추가적인 로봇 데이터(시뮬레이션 혹은 실제 로봇 환경)를 함께 학습
  - 출력 : 다음에 취해야 할 조치를 자연어(혹은 프로그래밍 형태)로 기술하거나, 로봇 제어 신호를 직간접적으로 표현

### F. Visual Generative Models
- Diffusion 기반 생성 모델
  - 순방향 프로세스(forward process) : 이미지에 점진적으로 노이즈를 추가해 백색 잡음(white noise)에 가깝게 만들고,
  - 역방향 프로세스(reverse process) : 모델이 이 노이즈를 단계별로 되돌려가는 식으로 최종 이미지를 복원·합성하는 방식
  - 노이즈 -> 이미지 역과정 학습

  </div>
</details>

## 3. Robotics

<details>
  <summary>Robotics</summary>
  <div markdown='1'>

### A. Robot Policy Learning for Decision Making and Control

#### 1) Language-conditioned Imitation Learning for Manipulation
- 개념
  - 언어 조건 모방 학습에서는 자연어 지시 ${(l)}$ 와 현재 상태 ${(s_t)}$ 를 기반으로 적절한 행동을 예측하는 목표 조건 정책을 학습
  - 주어진 예제 데이터를 그대로 따라 학습하는 방식으로 작동, 최대 우도 목적 함수 사용 ${L_{GCIL}=\mathbb{E_{(\tau, l) \sim \mathcal{D}}} \displaystyle{\sum_{t=0}^{|r|} \log \pi_\theta (a_t|s_t, l)}}$
    - ${\pi_\theta (a_t|s_t, l)}$ : a goal-conditioned policy
    - ${\mathcal{D} = \{ \tau_i \}^N_i}$ : the language-annotated demonstration dataset
    - ${\tau}$ : trajectories, or sequences of images

- 주요 연구
  - Play-LMP (Play Latent Motor Policies)
  - CLIPort (CLIP (Contrastive Language-Image Pretraining) + Transport Network)
  - PerAct (Perceiver-Actor)
  - MCIL (Multi-Context Imitation Learning)
  - CACTI (Context-Aware Consistent Task Imitation)
  - Voltron

#### 2) Language-Assisted Reinforcement Learning
- 개념
  - Reinforcement Learning은 로봇이 reward를 최대화하는 방향으로 행동을 학습하는 방법
  - 하지만, 새로운 환경 탐색이 오래 걸리며 데이터 효율성이 낮음
  - LLM / VLM을 RL과 결합하여 탐색 효율을 개선하는 방식

- 주요연구
  - AdA (Adaptive Agent)
  - Palo et al.

### B. Language-Image Goal-Conditioned Value Learning
- 개념
  - 기존 RL에서는 명시적인 보상함수가 필요하지만,
  - 언어와 이미지를 활용한 가치 학습을 하면 보상을 직접 학습할 수 있고, 더 나아가 인간의 언어 지시를 반영하는 로봇 행동을 설계 가능능
  - 로봇이 학습하지 않은 새로운 작업도 수행할 수 있도록 일반화(generalization) 능력을 향상시키는 데 기여

  - Value Learning, 가치학습
    - 특정 상태에서 특정 행동 시, 얼마나 좋은 결과를 낼 것인지를 예측
    - 로봇이 미래의 목표를 고려해, 최적의 행동을 선택하도록 가치함수 학습
      - 상태 가치 함수 : 특정 상태에서 기대할 수 있는 보상의 총합
      - 행동 가치 함수 : 특정 상태에서 특정 행동을 했을 때 기대할 수 있는 보상의 총합

  - Language-Image goal-conditioned, 언어-이미지 목표-조건
    - 목표를 언어 또는 이미지로 지정
    - 로봇이 이를 달성할 수 있는지 여부를 학습, 
    - 이를 기반으로 행동 선택

- 주요 연구
  - R3M (Reusable Representations for Robotic Manipulations)
  - VIP (Value-Implicit Pretraining)
  - LIV (Language-Image Value Learning)
  - SayCan
  - VoxPoser

### C. Robot Task Planning using Large Language Models
- 개념
  - Robot Task Planning : 로봇이 주어진 목표를 달성하기 위해 일련의 작업을 계획하고 실행하는 과정
  - 최근 LLM을 로봇 작업 계획에 활용

#### 1) Language Instructions for Task Specification
- NL2TL (Natural Language to Temporal Logic)
  - translation from natural language (NL) to temporal logic (TL).
- AutoTAMP (Task and Motion Planning)
  - LLM을 사용하여 자연어에서 Task and Motion Planning (TAMP)을 자동 생성.
  - 전통적인 로봇 계획에서는 고수준의 작업 계획(Task Planning)과 저수준의 모션 계획(Motion Planning)이 분리되어 있었지만, AutoTAMP는 LLM을 활용하여 이 두 가지를 자동으로 통합

#### 2) Code Generation using Language Models for Task Planning
- ProgPrompt
  - LLM을 활용하여 자연어에서 Python과 같은 프로그램 코드로 변환하여 로봇이 실행할 수 있도록 함.
- Code-as-Policies
  - 기존의 RL이나 행동 학습과 달리, LLM이 프로그래밍 언어를 사용해 로봇 행동을 직접 정의.
- ChatGPT-Robotics
  - ChatGPT를 활용하여 사용자가 직접 로봇 프로그래밍을 할 필요 없이 프롬프트 입력만으로 코드 자동 생성.

### D. In-context Learning (ICL) for Decision-Making
- 개념
  - In-context Learning : 모델이 과거의 데이터에서 학습한 일반적인 패턴을 기반으로, 새로운 입력을 처리하는 방식
    - Traditional method
      - 모델이 특정 데이터셋으로 학습되고, 테스트 데이터에 대한 예측을 수행.
      - 새로운 태스크가 주어지면 추가적인 재학습(fine-tuning)이 필요.
    - ICL
      - 모델이 특정 데이터를 학습한 후, 새로운 데이터가 주어지면 학습된 패턴을 이용해 즉석에서 해결.
      - 모델 자체를 변경하지 않고 프롬프트(prompt) 내에서 학습하여 태스크를 해결.
  - 대형 언어 모델(LLM)에서 강력한 성능을 보이며, 로보틱스에서도 ICL을 활용하여 학습 시간을 단축하고 적응성을 높이는 연구가 활발히 진행

### E. Robot Transformers
- 개념
  - RT (Robot Transformers) : 대형 트랜스포머 모델을 활용하여 로봇의 감각 입력을 행동으로 변환하는 프레임워크
  - 최근 대형 기초 모델(Foundation Models)이 NLP 및 컴퓨터 비전 분야에서 성공적으로 적용된 것을 바탕으로, 로봇 공학에서도 이러한 모델을 활용하려는 연구들에 의해 주도

- 주요 연구
  - RT-1 (Robotic Transformer 1)
  - RT-2 (Robotic Transformer 2)
  - RT-X (Cross-Embodiment Robotic Transformer)

- 기타 연구
  - PACT (Perception-Action Causal Transformer)
  - SMART (Self-supervised Multi-task pretrAining with contRol Transformer)
  - LATTE (LAnguage Trajectory TransformEr)

### F. Open-Vocabulary Robot Navigation and Manipulation
- 개념
  - Open-Vocabulary : 기존의 로봇 시스템은 미리 정의된 객체(예: 특정 가구, 도구)에 대해 훈련되지만, 개방형 어휘 로봇은 이전에 본 적 없는 객체나 환경에서도 작동할 수 있어야 한다.
  - 즉, 새로운 명령어를 이해하고, 새로운 물체나 장소를 탐색하는 능력이 필요

#### 1) Open-Vocabulary Navigation
- 개념
  - 로봇이 기존의 지도나 사전 학습된 목표 없이, 텍스트나 이미지로 주어진 목표를 찾아 이동하는 능력.
  - 사전 정의된 경로나 지도 없이 자연어, 시각적 단서, 경험을 바탕으로 환경을 탐색.
  - 인간의 명령을 이해하고 목표 지점까지 이동할 수 있어야 함.

- 주요 연구
  - LM-Nav
  - ViNT
  - AVLMaps
  - Object-Based Navigation

#### 2) Open-Vocabulary Manipulation
- 개념
  - 로봇이 새로운 물체를 이해하고 조작할 수 있도록 하는 연구.
  - 새로운 객체의 조작 가능성(Affordance)을 학습.
  - 기존에 학습된 객체가 아니더라도 유사한 특성을 이용해 조작 가능.

- 주요 연구
  - VIMA 
  - RoboCat
  - StructDiffusion
  - DALL-E-Bot

  </div>
</details>

## 4. Perception

<details>
  <summary>Perception</summary>
  <div markdown='1'>
  </div>
</details>

## 5. Embodied AI

<details>
  <summary>Embodied AI</summary>
  <div markdown='1'>
  </div>
</details>

## 6. Challenges and Future Directions

<details>
  <summary>Challenges and Future Directions</summary>
  <div markdown='1'>
  </div>
</details>

## 7. Conclusion

<details>
  <summary>Conclusion</summary>
  <div markdown='1'>
  </div>
</details>
