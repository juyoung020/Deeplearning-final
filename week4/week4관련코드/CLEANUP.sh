#!/usr/bin/env bash
# Week 4 정리(cleanup) 스크립트 — 공용 컴퓨터에서 받은 대용량 데이터/캐시/산출물 삭제용.
#
# 기본은 DRY-RUN (크기만 보여주고 삭제 안 함).
# 실제 삭제하려면:   bash week4/CLEANUP.sh --yes
# pip 패키지까지 제거: bash week4/CLEANUP.sh --yes --purge-pip
#
# 코드/문서(아래 KEEP 목록)는 자동 삭제하지 않습니다. 필요시 직접 지우세요.

set -u
YES=0
PURGE_PIP=0
for a in "$@"; do
  case "$a" in
    --yes) YES=1 ;;
    --purge-pip) PURGE_PIP=1 ;;
  esac
done

# 주의: Windows(NTFS) 파티션 /media/ad06/시스템 에는 아무것도 받지 않습니다.
MODEL_CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct"
REPO="/home/ad06/isaacsim/git/Deeplearning-final"
DATA="$REPO/data"           # VLN-VERSE 데이터 (root 드라이브)
OUTPUTS="$REPO/outputs"     # 학습 체크포인트/로그

DELETE_TARGETS=(
  "$DATA"                  # VLN-VERSE 데이터
  "$MODEL_CACHE"           # Qwen3-VL-2B 모델 캐시 (~4GB)
  "$OUTPUTS"               # 학습 체크포인트/로그
)

echo "================ Week 4 cleanup ================"
echo "(mode: $([ $YES -eq 1 ] && echo DELETE || echo DRY-RUN))"
echo ""
echo "[삭제 대상]"
total=0
for t in "${DELETE_TARGETS[@]}"; do
  if [ -e "$t" ]; then
    sz=$(du -sh "$t" 2>/dev/null | cut -f1)
    echo "  - $t   ($sz)"
    if [ $YES -eq 1 ]; then
      rm -rf "$t" && echo "      -> 삭제됨"
    fi
  else
    echo "  - $t   (없음)"
  fi
done
echo ""

if [ $PURGE_PIP -eq 1 ]; then
  echo "[pip 패키지 제거] peft accelerate pyarrow"
  if [ $YES -eq 1 ]; then
    pip uninstall -y peft accelerate pyarrow
  else
    echo "  (dry-run: --yes 와 함께 실행해야 제거)"
  fi
  echo ""
fi

echo "[유지(KEEP) — 코드/문서, 자동 삭제 안 함]"
echo "  - $REPO/week4/                 (Week 4 코드)"
echo "  - $REPO/week3/demo.py, go2_physics_teleop.py  (additive 변경)"
echo "  - /home/ad06/isaacsim/change/  (변경본 사본)"
echo "  - /home/ad06/isaacsim/deeplearning_project/클로드/  (정리 문서)"
echo ""
echo "[참고] HF 모델 캐시 전체를 비우려면:  rm -rf ~/.cache/huggingface/hub"
echo "       (다른 모델도 같이 지워질 수 있으니 주의)"
echo "================================================"
