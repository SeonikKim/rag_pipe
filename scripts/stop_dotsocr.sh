#!/bin/bash

echo "🛑 DotsOCR 서버 종료 중..."

# DotsOCR 관련 프로세스 종료
pkill -f "vllm.*DotsOCR" 2>/dev/null
pkill -f "python.*DotsOCR" 2>/dev/null

# 잠시 대기
sleep 2

# GPU 메모리 확인
echo "📊 GPU 메모리 상태:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1

echo "✅ DotsOCR 서버 종료 완료!"
