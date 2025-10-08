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
        'name': '레이아웃 감지',
        'script': 'step1_primary_ocr.py',
        'description': 'DotsOCR로 레이아웃 추출 + Picture 크롭 (원본 이미지)'
    },
    {
        'name': '이중 OCR',
        'script': 'step1b_dual_ocr.py',
        'description': '텍스트 블록 전처리 전/후 2회 OCR'
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
        'name': 'LLM OCR 선택',
        'script': 'step6_llm_correction.py',
        'description': 'LLM이 이중 OCR 결과 비교 및 최적 값 선택'
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
    llm_server_available = False
    try:
        import requests
        response = requests.get("http://localhost:8003/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ LLM 서버 연결됨")
            llm_server_available = True
        else:
            print("   ⚠️ LLM 서버 응답 이상 (포트 8003)")
    except:
        print("   ⚠️ LLM 서버 연결 실패 (포트 8003) - 자동 시작 시도")
        # LLM 서버 자동 시작 시도
        try:
            print("   🔄 LLM 서버 시작 중...")
            subprocess.Popen(["bash", str(BASE_DIR / "scripts" / "start_llm.sh")], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=BASE_DIR)
            print("   ⏳ LLM 서버 준비 대기 (30초)...")
            time.sleep(30)
            
            # 다시 확인
            response = requests.get("http://localhost:8003/health", timeout=10)
            if response.status_code == 200:
                print("   ✅ LLM 서버 자동 시작 완료")
                llm_server_available = True
            else:
                print("   ❌ LLM 서버 자동 시작 실패")
        except Exception as e:
            print(f"   ❌ LLM 서버 시작 오류: {e}")
    
    if not llm_server_available:
        print("   ⚠️ LLM 교정 단계는 기본 교정만 수행됩니다")
    
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

def check_next_step_requirements(step_num):
    """다음 스텝이 정상 작동할 수 있는지 확인"""
    try:
        # 현재 세션 디렉토리 찾기
        session_dirs = list((BASE_DIR / "ocr_results").glob("*/202*"))
        if not session_dirs:
            print("   ❌ 세션 디렉토리를 찾을 수 없습니다")
            return False
        
        latest_session = max(session_dirs, key=lambda x: x.name)
        print(f"   📁 세션 디렉토리: {latest_session}")
        
        # 각 스텝별 요구사항 확인
        if step_num == 1:  # 이중 OCR
            step1_dir = latest_session / "step1_primary"
            json_files = list(step1_dir.glob("page_*.json"))
            if not json_files:
                print("   ❌ step1 JSON 파일이 없습니다")
                return False
            print(f"   ✅ step1 JSON 파일 {len(json_files)}개 발견")
            
        elif step_num == 2:  # 섹션 병합
            step1b_dir = latest_session / "step1b_dual"
            combined_file = step1b_dir / "combined.json"
            if not combined_file.exists():
                print("   ❌ step1b 통합 파일이 없습니다")
                return False
            print(f"   ✅ step1b 통합 파일 발견")
            
        elif step_num == 3:  # Kiwi 교정
            layout_dir = latest_session / "step4_layout"
            json_files = list(layout_dir.glob("*_sections.json"))
            if not json_files:
                print("   ❌ 섹션 병합 결과가 없습니다")
                return False
            print(f"   ✅ 섹션 병합 결과 {len(json_files)}개 발견")
            
        elif step_num == 4:  # LLM OCR 선택
            kiwi_dir = latest_session / "step5_kiwi"
            json_files = list(kiwi_dir.glob("kiwi_*.json"))
            if not json_files:
                print("   ❌ Kiwi 교정 결과가 없습니다")
                return False
            print(f"   ✅ Kiwi 교정 결과 {len(json_files)}개 발견")
            
        elif step_num == 5:  # 최종 JSON
            llm_dir = latest_session / "step6_llm"
            json_files = list(llm_dir.glob("llm_*.json"))
            if not json_files:
                print("   ❌ LLM OCR 선택 결과가 없습니다")
                return False
            print(f"   ✅ LLM OCR 선택 결과 {len(json_files)}개 발견")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 요구사항 확인 오류: {e}")
        return False

def run_step(step_info, step_num, total_steps, max_pages=None):
    """개별 단계 실행"""
    script_path = SCRIPTS_DIR / step_info['script']
    
    print(f"\n{'='*60}")
    print(f"[{step_num}/{total_steps}] {step_info['name']}")
    print(f"{'='*60}")
    print(f"📋 {step_info['description']}")
    print(f"🚀 실행: {step_info['script']}")
    
    start_time = time.time()
    
    try:
        # 명령줄 인수 준비
        cmd = [sys.executable, str(script_path)]
        
        # step1에 max_pages 전달
        if step_num == 1 and max_pages:
            cmd.extend(['--max-pages', str(max_pages)])
        
        # Python 스크립트 실행
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=BASE_DIR)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {step_info['name']} 완료 ({duration:.1f}초)")
            return True
        else:
            print(f"\n❌ {step_info['name']} 실패 ({duration:.1f}초)")
            return False
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ 실행 오류 ({duration:.1f}초): {e}")
        return False

def run_pipeline(start_step=1, end_step=None, max_pages=None):
    """파이프라인 실행"""
    if end_step is None:
        end_step = len(PIPELINE_STEPS)
    
    print("🚀 OCR 파이프라인 시작")
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 실행 단계: {start_step} ~ {end_step}")
    if max_pages:
        print(f"📄 페이지 제한: 최대 {max_pages}페이지")
    
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
        
        # 서버 전환 로직
        if i + 1 == 3:  # 3단계(섹션 병합) 시작 전에 DotsOCR 종료
            print("\n🔄 DotsOCR 서버 종료 중...")
            subprocess.run(["bash", str(BASE_DIR / "scripts" / "stop_dotsocr.sh")], 
                         capture_output=False, cwd=BASE_DIR)
        
        if i + 1 == 5:  # 5단계(LLM OCR 선택) 시작 전에 LLM 서버 시작
            print("\n🔄 LLM 서버 시작 중...")
            try:
                subprocess.Popen(["bash", str(BASE_DIR / "scripts" / "start_llm.sh")], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=BASE_DIR)
                print("⏳ LLM 서버 준비 대기 (30초)...")
                time.sleep(30)
                
                # LLM 서버 상태 확인
                try:
                    import requests
                    response = requests.get("http://localhost:8003/health", timeout=10)
                    if response.status_code == 200:
                        print("✅ LLM 서버 준비 완료")
                    else:
                        print("⚠️ LLM 서버 응답 이상, 계속 진행")
                except:
                    print("⚠️ LLM 서버 상태 확인 실패, 계속 진행")
            except Exception as e:
                print(f"❌ LLM 서버 시작 실패: {e}, 기본 교정으로 진행")
        
        success = run_step(step_info, i + 1, len(PIPELINE_STEPS), max_pages=max_pages)
        
        if success:
            # 다음 스텝이 정상 작동할 수 있는지 확인
            if i + 1 < end_step:
                next_step = PIPELINE_STEPS[i + 1]
                print(f"\n🔍 다음 스텝 '{next_step['name']}' 준비 상태 확인 중...")
                
                # 필요한 입력 파일들이 존재하는지 확인
                if not check_next_step_requirements(i + 1):
                    print(f"⚠️ 다음 스텝 '{next_step['name']}' 요구사항 미충족")
                    failed_steps.append(f"{i+1}. {step_info['name']} (다음 스텝 준비 실패)")
                    success = False
                else:
                    print(f"✅ 다음 스텝 '{next_step['name']}' 준비 완료")
        
        if not success:
            failed_steps.append(f"{i+1}. {step_info['name']}")
            
            # 실패 시 자동으로 계속 진행
            print(f"\n⚠️ {step_info['name']} 단계가 실패했습니다.")
            print("⚠️ 실패한 단계를 건너뛰고 계속 진행합니다.")
            # response = input("계속 진행하시겠습니까? (y/N): ").strip().lower()
            # if response != 'y':
            #     break
    
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
    parser.add_argument('--max-pages', type=int, help='처리할 최대 페이지 수 (step1용)')
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
    success = run_pipeline(args.start, args.end, max_pages=args.max_pages)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
