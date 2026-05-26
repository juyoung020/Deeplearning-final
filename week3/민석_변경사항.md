# C 변경사항

## 담당 범위
`language_velocity_model.py` (LanguageVelocityPredictor) + `infer_language_velocity.py` + `go2_physics_teleop.py`

---

## 변경 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `language_velocity_model.py` | `predict_batch()` 메서드 추가 |
| `infer_language_velocity.py` | `--all`, `--output` 옵션 추가 |
| `go2_physics_teleop.py` | 기본 checkpoint 경로 자동 감지 추가 |

---

## 상세 변경 내용

### 1. `language_velocity_model.py` — `predict_batch()` 추가

**변경 전:** 텍스트 1개씩만 predict() 가능

**변경 후:**
```python
def predict_batch(self, texts: list[str]) -> list[tuple[float, float, float]]:
    tensors = self.predict_tensor(texts).detach().cpu().float()
    return [tuple(float(v) for v in row.tolist()) for row in tensors]
```

여러 command를 한 번에 배치로 추론 가능 → 전체 11개 command 일괄 평가에 활용.

---

### 2. `infer_language_velocity.py` — `--all`, `--output` 추가

**추가된 옵션:**

| 옵션 | 설명 |
|------|------|
| `--all` | 11개 기본 command 전체 일괄 추론, target과 prediction 비교 출력 |
| `--output <path>` | `--all` 결과를 JSON 파일로 저장 |

**사용 예시:**
```bash
# 11개 전체 확인
python infer_language_velocity.py --all

# JSON 저장
python infer_language_velocity.py --all --output runs/inference_result.json

# 단일 command
python infer_language_velocity.py --text "turn left 45 degree"

# 대화형
python infer_language_velocity.py --interactive
```

**`--all` 출력 예시:**
```
Command                   Target                         Predicted
-------------------------------------------------------------------------------------
move forward 25cm         [+0.2500, +0.0000, +0.0000]   [+0.2501, +0.0000, +0.0000]
move forward 50cm         [+0.5000, +0.0000, +0.0000]   [+0.4999, +0.0000, +0.0000]
...
```

---

### 3. `go2_physics_teleop.py` — 기본 checkpoint 자동 감지

**변경 전:** `--go2-physics-language-model-checkpoint` 미지정 시 즉시 ValueError 발생

**변경 후:**
```python
if not checkpoint:
    default = os.path.join(os.path.dirname(__file__), "checkpoints", "language_velocity_mlp.pt")
    if os.path.exists(default):
        checkpoint = default  # 자동 사용
    else:
        raise ValueError(...)
```

`checkpoints/language_velocity_mlp.pt`가 있으면 플래그 없이도 자동으로 학습된 모델 사용.

**시뮬레이터 실행 (checkpoint 자동 감지):**
```bash
python demo.py --task fine --index 0 --work_dir ./myresults \
  --agent go2 --go2-controller physics
```

**시뮬레이터 실행 (명시적 지정):**
```bash
python demo.py --task fine --index 0 --work_dir ./myresults \
  --agent go2 --go2-controller physics \
  --go2-physics-language-model-checkpoint checkpoints/language_velocity_mlp.pt
```
