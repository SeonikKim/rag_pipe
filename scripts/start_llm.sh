#!/bin/bash

echo "🚀 LLM 서버 시작 중..."

# 이미 실행 중인지 확인
if pgrep -f "vllm serve.*EEVE" > /dev/null; then
    echo "⚠️ EEVE 서버가 이미 실행 중입니다. 건너뜁니다."
    exit 0
fi

# 포트 8003이 사용 중인지 확인
if lsof -Pi :8003 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️ 포트 8003이 이미 사용 중입니다. 건너뜁니다."
    exit 0
fi

# conda 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3_rag

echo "🔄 EEVE LLM 서버 시작 중... (포트 8003)"

# EEVE 모델 서버 시작
vllm serve yanolja/EEVE-Korean-Instruct-2.8B-v1.0 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.80 \
    --served-model-name eeve \
    --trust-remote-code \
    --port 8003
