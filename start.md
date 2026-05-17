# 실행 방법

## 풀 실행 코드 (한 번에 복붙)

```bash
source /home/ad06/miniconda3/etc/profile.d/conda.sh && \
conda activate goodnav && \
cd /home/ad06/isaacsim/isaac-sim-standalone-4.5.0-linux-x86_64 && \
source setup_conda_env.sh && \
export PYTHONPATH=/home/ad06/IsaacLab/source/isaaclab:$PYTHONPATH && \
cd /home/ad06/isaacsim/IAmGoodNavigator && \
python demo.py --task fine --index 0 --work_dir ./myresults --agent go2 --go2-controller physics
```

- `--task`: `fine` 또는 `coarse`
- `--index`: `0` ~ `9`

## 조작

- **W/S/A/D**: 전진 / 후진 / 좌회전 / 우회전
- **Enter**: 미션 완료 → 평가 결과 팝업 표시

뷰포트에서 카메라 전환: Perspective → Cameras → FloatingCamera
