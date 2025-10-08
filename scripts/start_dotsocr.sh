#!/bin/bash

echo "🚀 DotsOCR 서버 시작 중..."

# 이미 실행 중인지 확인
if pgrep -f "vllm serve.*DotsOCR" > /dev/null; then
    echo "⚠️ DotsOCR 서버가 이미 실행 중입니다. 건너뜁니다."
    exit 0
fi

# 포트 8000이 사용 중인지 확인
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️ 포트 8000이 이미 사용 중입니다. 건너뜁니다."
    exit 0
fi

# 환경 설정
export hf_model_path=./weights/DotsOCR
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH

# vllm 파일 수정
sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from DotsOCR import modeling_dots_ocr_vllm' `which vllm`

echo "✅ 환경 설정 완료"

# conda 환경 활성화 및 서버 시작
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 0924

echo "🔄 DotsOCR 서버 시작 중... (포트 8000)"
cd /home/cywell/project6/dots.ocr

vllm serve ./weights/DotsOCR \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --chat-template-content-format string \
    --served-model-name model \
    --trust-remote-code \
    --port 8000
