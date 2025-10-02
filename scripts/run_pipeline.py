#!/usr/bin/env python3
"""
OCR 파이프라인 메인 컨트롤러
전체 7단계 파이프라인을 순차 실행
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# 설정
BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent

PIPELINE_STEPS = [
    {
        'name': '1차 OCR',
        'script': 'step1_primary_ocr.py',
        'description': 'PDF를 DotsOCR로 전 페이지 OCR'
    },
    {
        'name': '저신뢰 블록 크롭',
        'script': 'step2_crop_low_confidence.py',
        'description': 'confidence < 0.80 블록들을 크롭'
    },
    {
        'name': '2차 OCR',
        'script': 'step3_secondary_ocr.py',
        'description': '고정 전처리 옵션으로 2차 OCR'
    },
    {
        'name': '섹션 병합',
        'script': 'step4_layout_merge.py',
        'description': 'BeautifulSoup4로 섹션별 병합'
    },
    {
        'name': 'Kiwi 교정',
        'script': 'step5_kiwi_correction.py',
        'description': '형태소 분석 기반 1차 교정'
    },
    {
        'name': 'LLM 교정',
        'script': 'step6_llm_correction.py',
        'description': 'EEVE 모델로 문맥 교정'
    },
    {
        'name': '최종 JSON',
        'script': 'step7_final_json.py',
        'description': 'RAG 친화적 최종 JSON 생성'
    }
]

def check_prerequisites():
    """사전 요구사항 확인"""
    print("🔍 사전 요구사항 확인...")
    
    issues = []
    
    # PDF 파일 확인
    pdf_dir = BASE_DIR / "pdf_in"
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        issues.append(f"PDF 파일이 없습니다: {pdf_dir}")
    else:
        print(f"   ✅ PDF 파일: {len(pdf_files)}개")
    
    # DotsOCR 서버 확인
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        print("   ✅ DotsOCR 서버 연결됨")
    except:
        issues.append("DotsOCR 서버에 연결할 수 없습니다 (포트 8000)")
    
    # LLM 서버 확인 (선택사항)
    try:
        import requests
        response = requests.get("http://localhost:8003/health", timeout=5)
        print("   ✅ LLM 서버 연결됨")
    except:
        print("   ⚠️ LLM 서버 연결 실패 (포트 8003) - 기본 교정만 수행")
    
    # 필수 라이브러리 확인
    required_libs = ['cv2', 'requests', 'fitz', 'bs4']
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            issues.append(f"필수 라이브러리 없음: {lib}")
    
    if not issues:
        print("   ✅ 모든 사전 요구사항 충족")
    
    return issues

def run_step(step_info, step_num, total_steps):
    """개별 단계 실행"""
    script_path = SCRIPTS_DIR / step_info['script']
    
    print(f"\n{'='*60}")
    print(f"[{step_num}/{total_steps}] {step_info['name']}")
    print(f"{'='*60}")
    print(f"📋 {step_info['description']}")
    print(f"🚀 실행: {step_info['script']}")
    
    start_time = time.time()
    
    try:
        # Python 스크립트 실행
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd=BASE_DIR)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 완료 ({duration:.1f}초)")
            if result.stdout:
                # 출력에서 중요한 정보만 표시
                lines = result.stdout.strip().split('\n')
                important_lines = [line for line in lines[-10:] if any(marker in line for marker in ['✅', '❌', '🎉', '📊', '📁'])]
                for line in important_lines:
                    print(f"   {line}")
            return True
        else:
            print(f"❌ 실패 ({duration:.1f}초)")
            print("오류 출력:")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-5:]:  # 마지막 5줄만
                    print(f"   {line}")
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-5:]:  # 마지막 5줄만
                    print(f"   {line}")
            return False
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ 실행 오류 ({duration:.1f}초): {e}")
        return False

def run_pipeline(start_step=1, end_step=None):
    """파이프라인 실행"""
    if end_step is None:
        end_step = len(PIPELINE_STEPS)
    
    print("🚀 OCR 파이프라인 시작")
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 실행 단계: {start_step} ~ {end_step}")
    
    # 사전 요구사항 확인
    issues = check_prerequisites()
    if issues:
        print("\n❌ 사전 요구사항 미충족:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    
    # 단계별 실행
    total_start_time = time.time()
    failed_steps = []
    
    for i in range(start_step - 1, end_step):
        step_info = PIPELINE_STEPS[i]
        success = run_step(step_info, i + 1, len(PIPELINE_STEPS))
        
        if not success:
            failed_steps.append(f"{i+1}. {step_info['name']}")
            
            # 실패 시 계속할지 물어보기
            print(f"\n⚠️ {step_info['name']} 단계가 실패했습니다.")
            response = input("계속 진행하시겠습니까? (y/N): ").strip().lower()
            if response != 'y':
                break
    
    # 완료 보고
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("🎉 파이프라인 완료")
    print(f"{'='*60}")
    print(f"⏱️ 총 소요 시간: {total_duration/60:.1f}분")
    
    if failed_steps:
        print(f"❌ 실패한 단계: {len(failed_steps)}개")
        for step in failed_steps:
            print(f"   • {step}")
    else:
        print("✅ 모든 단계 성공")
        
        # 최종 결과 위치 안내
        final_dir = BASE_DIR / "final_outputs"
        final_files = list(final_dir.glob("*_final.json"))
        if final_files:
            print(f"\n📁 최종 결과:")
            for file in final_files:
                print(f"   📄 {file}")
    
    return len(failed_steps) == 0

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR 파이프라인 실행')
    parser.add_argument('--start', type=int, default=1, help='시작 단계 (1-7)')
    parser.add_argument('--end', type=int, help='종료 단계 (1-7)')
    parser.add_argument('--list', action='store_true', help='단계 목록 표시')
    
    args = parser.parse_args()
    
    if args.list:
        print("📋 파이프라인 단계:")
        for i, step in enumerate(PIPELINE_STEPS, 1):
            print(f"   {i}. {step['name']}: {step['description']}")
        return
    
    # 인수 검증
    if args.start < 1 or args.start > len(PIPELINE_STEPS):
        print(f"❌ 시작 단계는 1-{len(PIPELINE_STEPS)} 범위여야 합니다.")
        return
    
    if args.end and (args.end < args.start or args.end > len(PIPELINE_STEPS)):
        print(f"❌ 종료 단계는 {args.start}-{len(PIPELINE_STEPS)} 범위여야 합니다.")
        return
    
    # 파이프라인 실행
    success = run_pipeline(args.start, args.end)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
