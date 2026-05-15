# Week 2 Go2 Language Command Implementation

## 실행 명령어

처음 새 터미널을 열었을 때는 `week1/RUN_WEEK1.md`와 같은 방식으로 환경을 먼저 준비합니다. 이 작업 환경에서는 마지막 실행을 반드시 `IAmGoodNavigator` 폴더 안에서 진행해야 합니다.

```bash
source /home/psh/deeplearning/miniconda3/etc/profile.d/conda.sh
conda activate goodnav

cd /home/psh/isaacsim
source setup_conda_env.sh

cd /home/psh/deeplearning/IAmGoodNavigator
export PYTHONNOUSERSITE=1
```

Week2 과제 실행은 아래 명령을 사용합니다. 제출 녹화에는 Week1과 동일하게 `first_person` 카메라가 가장 무난합니다.

```bash
python demo.py --agent go2 --task fine --index 1 --work_dir ./myresults --isaaclab-root /home/psh/isaacsim/IsaacLab --go2-physics-camera first_person
```

로봇에 붙은 USD 카메라를 viewport에 직접 쓰고 싶으면 아래처럼 실행합니다.

```bash
python demo.py --agent go2 --task fine --index 1 --work_dir ./myresults --isaaclab-root /home/psh/isaacsim/IsaacLab --go2-physics-camera rgbd_camera --go2-physics-use-usd-camera-prim
```

한 번에 복사해서 실행하려면 아래 블록을 사용합니다.

```bash
source /home/psh/deeplearning/miniconda3/etc/profile.d/conda.sh
conda activate goodnav

cd /home/psh/isaacsim
source setup_conda_env.sh

cd /home/psh/deeplearning/IAmGoodNavigator
export PYTHONNOUSERSITE=1

python demo.py --agent go2 --task fine --index 1 --work_dir ./myresults --isaaclab-root /home/psh/isaacsim/IsaacLab --go2-physics-camera first_person
```

실행 후 터미널에 `Command:` 프롬프트가 나오면 자연어 명령을 한 줄씩 입력합니다.

```text
move forward 3m
turn right 60
move forward 3
turn left 90
stop
finish
```

`finish`를 입력하면 정상 종료로 처리되어 결과 CSV 저장 및 평가 단계로 넘어갑니다. 기본 속도는 과제 조건에 맞춰 전진 `0.5 m/s`, 회전 `30 deg/s`입니다. `demo.py`는 수정하지 않고, 이 기본값은 `go2_physics_teleop.py` 내부에서 처리합니다.

## 지원 명령 형식

지원되는 자연어 명령은 과제에서 요구한 `"action type + argument value"` 형식입니다.

| 입력 예시 | velocity command | 지속 시간 |
|---|---:|---:|
| `move forward 3m` | `[0.5, 0, 0]` | `3 / 0.5 = 6.0s` |
| `move forward 2` | `[0.5, 0, 0]` | `2 / 0.5 = 4.0s` |
| `turn left 90` | `[0, 0, pi/6]` | `90 / 30 = 3.0s` |
| `turn right 60degrees` | `[0, 0, -pi/6]` | `60 / 30 = 2.0s` |
| `stop` | `[0, 0, 0]` | 즉시 정지 |

추가 제어 명령도 있습니다.

- `finish`: 정상 종료 및 결과 저장/평가 단계로 이동합니다.
- `quit`, `exit`, `abort`: 실행을 중단합니다.

## 핵심 수정 내용

이번 Week2 구현은 `go2_physics_teleop.py`에만 추가했습니다. `demo.py`에는 언어 명령용 CLI 인자를 추가하지 않았고, 기존 실행 명령을 그대로 사용합니다. 새로 넣은 코드에는 요청대로 `###2` 주석을 붙였습니다.

### 1. 정규식 사용을 위한 import

자연어 명령을 `"action type + argument value"` 형식으로 판별하기 위해 `re`를 추가했습니다.

```python
import re  # ###2
```

이 import는 `move forward 3m`, `turn right 60 degrees`처럼 숫자와 단위가 섞인 문자열을 안정적으로 파싱하기 위해 사용합니다.

### 2. Control step 시간 저장

`configure_env_from_episode()` 안에서 IsaacLab 환경의 control step 시간을 계산해 `args._go2_physics_control_dt`에 저장했습니다.

```python
args._go2_physics_control_dt = float(env_cfg.sim.dt) * float(getattr(env_cfg, "decimation", 1))  # ###2
```

이 값이 필요한 이유는 과제가 요구한 “계산된 지속 시간 동안 velocity command 유지”를 실제 `env.step()` 반복 횟수로 바꿔야 하기 때문입니다.

```text
duration_s = 명령 지속 시간
control_dt = env.step() 한 번에 해당하는 제어 시간
num_steps = ceil(duration_s / control_dt)
```

예를 들어 `control_dt = 0.02s`라면 `move forward 3m`의 6초 명령은 약 `300 step` 동안 유지됩니다.

### 3. 자연어 명령 정규화

`normalize_language_command_text()`를 추가해 입력 문자열을 파싱하기 쉬운 형태로 바꿉니다.

처리 내용:

- 앞뒤 공백 제거
- 대문자를 소문자로 변환
- `°`를 `degrees`로 변환
- `_`, `-`를 공백으로 변환
- 여러 공백을 하나로 압축

따라서 아래 입력들은 같은 계열로 처리됩니다.

```text
turn right 60°
turn right 60 degrees
turn_right_60_degrees
turn-right-60-degrees
```

과제에서 요구한 주 형식은 `move forward`, `turn left`, `turn right`, `stop`이며, 정규화는 사용자가 조금 다르게 입력해도 같은 의도로 받아들이기 위한 보조 처리입니다.

### 4. 자연어 명령 파서

`parse_language_velocity_command()`를 추가했습니다. 이 함수가 과제 핵심인 자연어 명령을 velocity command와 지속 시간으로 바꾸는 부분입니다.

지원하는 주 명령:

```text
move forward <meters>
turn left <degrees>
turn right <degrees>
stop
```

보조 명령:

```text
finish
quit
exit
abort
```

반환값은 `SimpleNamespace`이며, 주요 필드는 다음과 같습니다.

```text
kind       motion / stop / finish / quit
label      출력용 명령 이름
command    [v_x, v_y, v_yaw]
duration_s velocity를 유지할 시간
```

전진 명령은 아래처럼 처리됩니다.

```text
입력: move forward D
속도: [0.5, 0, 0]
시간: D / 0.5
```

예시:

```text
move forward 3m
distance_m = 3
linear_speed_mps = 0.5
duration_s = 3 / 0.5 = 6.0
command = [0.5, 0.0, 0.0]
```

회전 명령은 아래처럼 처리됩니다.

```text
입력: turn left A
속도: [0, 0, +pi/6]
시간: A / 30

입력: turn right A
속도: [0, 0, -pi/6]
시간: A / 30
```

예시:

```text
turn right 60
degrees = 60
yaw_speed_deg_s = 30
duration_s = 60 / 30 = 2.0
yaw_speed_rad_s = radians(30) = pi/6 = 0.5236
command = [0.0, 0.0, -0.5236]
```

`stop`은 duration 없이 즉시 `[0, 0, 0]`을 반환합니다.

### 5. Timed velocity executor

`LanguageVelocityCommander` 클래스를 추가했습니다. 이 클래스는 파서 결과를 실제 제어 루프에서 사용할 수 있도록 관리합니다.

역할:

- 터미널에서 `Command:` 입력을 받음
- 입력된 문자열을 `parse_language_velocity_command()`로 파싱
- motion 명령이면 duration을 step 수로 변환
- 해당 step 수 동안 같은 velocity command를 계속 반환
- 명령 시간이 끝나면 자동으로 stop command로 되돌림

핵심 상태 변수:

```text
self._command          현재 유지 중인 velocity command
self._remaining_steps  command를 더 유지해야 하는 step 수
self._active_label     현재 명령 표시용 label
self.mission_complete  finish 입력 여부
self.quit_requested    quit 입력 여부
```

motion 명령을 받으면 다음 흐름으로 처리합니다.

```text
1. parsed = parse_language_velocity_command(raw_text)
2. self._command = parsed.command
3. self._remaining_steps = ceil(parsed.duration_s / self.control_dt)
4. 매 step마다 self._command 반환
5. remaining_steps가 0이 되면 [0, 0, 0]으로 초기화
```

이 구조 덕분에 사용자는 한 번만 `move forward 3m`을 입력하면 되고, 코드는 내부적으로 여러 번의 `env.step()` 동안 같은 command를 유지합니다.

### 6. 메인 제어 루프 연결

`run_go2_physics_episode()`의 기존 제어 루프에 언어 명령 분기를 추가했습니다.

기존 구조에는 크게 세 제어 방식이 있었습니다.

```text
GT follower
scripted command
keyboard WASD teleop
```

Week2 구현은 여기에 language command를 추가한 것입니다.

```text
GT follower
scripted command
language command
keyboard WASD teleop
```

언어 명령 모드는 기본으로 켜집니다.

```python
language_commands_enabled = bool(getattr(args, "go2_physics_language_commands", True))  # ###2
```

`demo.py`에 새 인자를 추가하지 않았기 때문에 `args.go2_physics_language_commands`가 없어도 `getattr(..., True)`에 의해 자동으로 활성화됩니다.

언어 명령이 활성화되는 조건:

```text
scripted_command is None
gt_follower is None
language_commands_enabled is True
```

즉 scripted test나 GT follower가 켜져 있지 않을 때 터미널 자연어 명령 모드로 들어갑니다.

### 7. env.step()에서 velocity 유지

실제 루프의 핵심 순서는 다음과 같습니다.

```text
1. command_np = language_commander.advance()
2. command = torch.tensor(command_np, device=tensor.device, dtype=tensor.dtype)
3. inject_velocity_command(obs, env, command)
4. actions = policy(obs)
5. obs, _, dones, _ = env.step(actions)
```

중요한 점은 `language_commander.advance()`가 duration이 끝나기 전까지 같은 command를 반환한다는 것입니다. 그래서 `inject_velocity_command()`에는 매 step 같은 velocity가 들어가고, 그 상태로 `env.step(actions)`가 반복됩니다.

예시 흐름:

```text
입력: move forward 3m
계산: [0.5, 0, 0], 6.0s
변환: 6.0s / control_dt = N steps
실행: N번의 env.step() 동안 [0.5, 0, 0] 주입
완료: 자동으로 [0, 0, 0] 복귀
```

### 8. WASD 키보드와의 관계

언어 명령 모드가 활성화되면 기존 WASD keyboard teleop은 생성하지 않도록 했습니다.

```python
keyboard = None if args.headless or language_commander is not None else WASDVelocityKeyboard(args)  # ###2
```

이렇게 한 이유는 터미널 자연어 명령과 키보드 입력이 동시에 velocity command를 바꾸면 실험 결과가 헷갈릴 수 있기 때문입니다. Week2 과제 목적은 자연어 명령 기반 velocity 제어이므로, 언어 명령 모드에서는 터미널 입력만 사용합니다.

### 9. 종료와 중단 처리

`finish`를 입력하면 `language_commander.mission_complete`가 `True`가 되고, 메인 루프에서 성공 종료로 처리됩니다.

```text
Command: finish
-> success = True
-> 결과 CSV 저장
-> 평가 단계 진행
```

`quit`, `exit`, `abort`, `Ctrl+C`, EOF는 실행 중단으로 처리합니다.

```text
Command: quit
-> quit_requested = True
-> 루프 종료
```

환경이 넘어지거나 termination이 발생한 경우에는 language-command run을 중단합니다.

```text
[ABORT] Language-command Go2 physics run stopped because the environment terminated.
```

### 10. 수정 위치 요약

수정 위치는 모두 `go2_physics_teleop.py` 안의 `###2` 주석으로 찾을 수 있습니다.

| 위치 | 내용 |
|---|---|
| import 영역 | `re` 추가 |
| `configure_env_from_episode()` | control step time 저장 |
| `parse_language_velocity_command()` 근처 | 자연어 명령 파서 추가 |
| `LanguageVelocityCommander` | duration 기반 command 유지 executor 추가 |
| `warmup_policy()` | 언어 명령 모드 안내 문구 추가 |
| `run_go2_physics_episode()` | language command 생성 및 제어 루프 연결 |

## 핵심 구현 원리 요약

이번 구현의 핵심은 자연어 명령을 바로 위치 이동으로 바꾸는 것이 아니라, Go2 policy가 이해하는 base velocity command로 바꾸는 것입니다.

```text
natural language
-> action type + argument value parsing
-> velocity command [v_x, v_y, v_yaw]
-> duration 계산
-> duration 동안 env.step() 반복 주입
```

과제 조건과 코드의 대응 관계는 다음과 같습니다.

| 과제 조건 | 구현 방식 |
|---|---|
| 자연어는 `"action type + argument value"`만 지원 | 정규식으로 `move forward`, `turn left`, `turn right`, `stop`만 허용 |
| 전진 속도는 `0.5 m/s` | `linear_speed_mps = 0.5` 기본값 |
| 회전 속도는 `30 deg/s` | `yaw_speed_deg_s = 30.0`, 내부에서 `pi/6 rad/s`로 변환 |
| 지속 시간 계산 | `distance / 0.5`, `angle / 30` |
| `env.step()`에서 velocity 유지 | `remaining_steps`가 0이 될 때까지 같은 command 반환 |
| 완료 명령 | `finish` 입력 시 정상 종료 |

따라서 `move forward 3m`을 입력하면 6초 동안 `[0.5, 0, 0]`이 유지되고, `turn right 60`을 입력하면 2초 동안 `[0, 0, -pi/6]`이 유지됩니다.

## 검증 명령

문법 체크:

```bash
cd /home/psh/deeplearning/IAmGoodNavigator
python -m py_compile go2_physics_teleop.py demo.py
```

파서 동작 확인:

```bash
cd /home/psh/deeplearning/IAmGoodNavigator
python -c "from go2_physics_teleop import parse_language_velocity_command as p; tests=['move forward 3m','move forward 2','turn left 90','turn right 60degrees','stop']; print([(t, p(t).label, tuple(round(float(x), 4) for x in p(t).command), round(p(t).duration_s, 2)) for t in tests])"
```

확인된 결과:

```text
move forward 3m -> [0.5, 0.0, 0.0], 6.0s
move forward 2 -> [0.5, 0.0, 0.0], 4.0s
turn left 90 -> [0.0, 0.0, 0.5236], 3.0s
turn right 60degrees -> [0.0, 0.0, -0.5236], 2.0s
stop -> [0.0, 0.0, 0.0], 0.0s
```

## 주의 사항

velocity command는 Go2 policy의 목표 속도 입력입니다. 실제 위치 변화는 policy, 물리 시뮬레이션, 충돌, 바닥 상태에 의해 약간 달라질 수 있습니다. 과제에서 요구하는 구현 포인트는 자연어 명령을 velocity command와 duration으로 변환하고, 그 duration 동안 `env.step()` 루프에서 동일 command를 유지하는 부분입니다.

## 실행 중 자주 보이는 로그

아래처럼 `Rigid Body ... missing xformstack reset`, `ScaleOrientation is not supported`, `MDLC ... unused let temporary`가 많이 출력될 수 있습니다.

```text
[Error] [omni.physicsschema.plugin] Rigid Body ... missing xformstack reset ...
[Warning] [omni.physicsschema.plugin] ScaleOrientation is not supported ...
[Warning] [rtx.neuraylib.plugin] [MDLC:COMPILER] ... unused let temporary ...
```

이 로그들은 Kujiale 씬 USD의 기존 rigid body hierarchy, scale, material compiler 경고라서 보통은 무시해도 됩니다. 실패 여부는 마지막에 Python `Traceback`, `[ERROR] Go2 physics controller failed`, Isaac Sim 창 강제 종료가 나오는지로 판단합니다.

정상 흐름이면 조금 기다린 뒤 아래 로그가 이어지고, 터미널에 `Command:` 프롬프트가 나타납니다.

```text
[INFO] Warmup complete. Enter language commands in the terminal to drive Go2.
[INFO] Interactive language prompt ready.
Command:
```

만약 `Command:`가 뜨지 않고 멈춘 것처럼 보이면 1~2분 정도 더 기다려봅니다. 그래도 진행되지 않으면 터미널 마지막 30줄, 특히 `Traceback` 또는 `[ERROR]`가 있는 부분을 확인합니다.
