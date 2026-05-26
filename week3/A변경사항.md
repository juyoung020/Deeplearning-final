# A 변경사항

## 담당 범위
`language_velocity_model.py` 전반부 — 데이터셋 + 모델 아키텍처

---

## 변경 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `language_velocity_model.py` | `VelocityMLP`에 `num_layers` 파라미터 추가, `LanguageVelocityPredictor`에 반영 |
| `train_language_velocity.py` | `DEFAULT_CONFIG`에 `"num_layers": 1` 추가, `VelocityMLP` 생성 시 전달 |
| `configs/week3_siglip_velocity.json` | `"num_layers": 1` 추가 |
| `CLAUDE.md` | 4인 분담표 신규 생성 |

---

## 상세 변경 내용

### 1. `language_velocity_model.py`

#### `VelocityMLP` — `num_layers` 파라미터 추가

**변경 전**
```python
class VelocityMLP(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 128, output_dim: int = 3, dropout: float = 0.0):
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
```

**변경 후**
```python
class VelocityMLP(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 128, output_dim: int = 3,
                 dropout: float = 0.0, num_layers: int = 1):
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
```

- `num_layers=1` (기본값): 기존과 동일 (`1024 → 128 → 3`)
- `num_layers=2`: `1024 → 128 → 128 → 3`

#### `LanguageVelocityPredictor` — checkpoint에서 `num_layers` 로드

**변경 전**
```python
self.model = VelocityMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(self.device)
```

**변경 후**
```python
num_layers = int(config.get("num_layers", 1))
self.model = VelocityMLP(
    input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers
).to(self.device)
```

---

### 2. `train_language_velocity.py`

`DEFAULT_CONFIG`에 `"num_layers": 1` 추가:
```python
"hidden_dim": 128,
"num_layers": 1,   # 추가
"dropout": 0.0,
```

`VelocityMLP` 생성 시 전달:
```python
model = VelocityMLP(
    input_dim=int(config["embedding_dim"]),
    hidden_dim=int(config["hidden_dim"]),
    dropout=float(config["dropout"]),
    num_layers=int(config.get("num_layers", 1)),   # 추가
).to(device)
```

---

### 3. `configs/week3_siglip_velocity.json`

```json
"hidden_dim": 128,
"num_layers": 1,
"dropout": 0.0,
```

---

## 기존 구현 (변경 없음)

A 담당 범위 중 아래 항목은 기존 구현이 완성된 상태로 유지:

- `VelocityExample` dataclass
- `default_velocity_examples()` — 11개 command 정의
- `examples_from_config()` — JSON에서 dataset 로드
- `FrozenSigLIPTextEncoder` — SigLIP2 freeze + 1024-dim 임베딩 추출
- `resolve_device()`, `resolve_torch_dtype()`
