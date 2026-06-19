# Deeplearning-final

NVIDIA Isaac Sim과 IsaacLab을 활용하여 4족보행 로봇의 물리 기반 시뮬레이션 환경을 구축하고,
자연어 명령으로 로봇을 제어하는 Language-Conditioned Quadruped Control 프로젝트입니다.


## 팀원

| 이름 | GitHub |
|------|--------|
| 김주영 | juyoung020 |
| 박성현 | psh030917 |
| 한준태 | Han1371 |
| 변민석 | bynminsuk-creator |

## 주차별 계획

### [1주차 — Notion 페이지](https://plausible-hallway-e4f.notion.site/Week-1-Isaac-Lab-352454b08de281c18542f72adf82ec91)
> 📝 [회의록 #1 (2026-05-02)](https://plausible-hallway-e4f.notion.site/1-2026-05-02-352454b08de281cc90a8e3e0a9032a42?source=copy_link)
- 팀 전체 환경 셋업 및 기술 조사
- Isaac Sim / IsaacLab 설치 및 실행 확인
- WASD 시뮬레이터 실행 해보기

### [2주차 — Notion 페이지](https://plausible-hallway-e4f.notion.site/Week-2-rule-based-352454b08de281a88c8dfce2651e080a)
> 📝 [회의록 #2 (2026-05-17)](https://plausible-hallway-e4f.notion.site/2-2026-05-17-363454b08de281b589e3fd2c8d051d1d)
- 자연어 명령을 velocity command로 변환하는 rule-based 파서 구현
- env.step()에서 velocity command 지속 유지 구현
- 공용 컴퓨터에서 시뮬레이터 실행 및 시연

### [3주차 — Notion 페이지](https://plausible-hallway-e4f.notion.site/Week-3-Text-Velocity-352454b08de28118a3aafb1c0c614117)
> 📝 [회의록 #3 (2026-05-26)](https://plausible-hallway-e4f.notion.site/36c454b08de281f2bb91d89e3f026a21)
- SigLIP2 텍스트 인코더(freeze) + MLP로 자연어 → velocity command 예측 모델 구현
- 학습 데이터 46개 구성 (forward / backward / turn / stop + paraphrase 확장)
- 시뮬레이터 연동 및 학습 모델 자동 감지 버그 수정

### [4주차 — Notion 페이지](https://plausible-hallway-e4f.notion.site/Week-4-5-VLN-37f454b08de281fcadc3ed0d3ed67359)
- Qwen3-VL-2B + LoRA(q_proj·v_proj) 미세조정 — 이미지 2장 + 지시문 → 다음 행동 예측
- 데이터 불균형(전진 67% · 정지 2%) 대응 — balance_power 재샘플링(균형형) · 정보·용량 ablation(정확도형) 두 갈래 병행 학습
- IsaacSim closed-loop 평가 · W&B 손실곡선 기록 · 최종보고서/발표자료 작성
- 
## 모델 체크포인트

- **모델2** — Qwen3-VL-2B + LoRA(q_proj·v_proj), 6 epoch (best: epoch 1), 회전·정지 클래스 재샘플링(balance_power 0.7): [Google Drive 다운로드](https://drive.google.com/file/d/1HZ5uGvO1A-6W_6pLetn_w-CSoFMPSW7N/view?usp=drive_link)
  - 사용법: 베이스 모델 `Qwen/Qwen3-VL-2B-Instruct`에 위 LoRA 어댑터를 적용해 로드
  - 
## 환경

- NVIDIA Isaac Sim
- IsaacLab
