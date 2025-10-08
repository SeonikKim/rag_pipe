#!/usr/bin/env python3
"""
⑥ 이중 OCR 비교 및 LLM 선택
입력: step1b_dual/combined.json (이중 OCR 결과)
처리: LLM이 두 OCR 값(원본/전처리) 비교 + 문맥 기반으로 최적 값 선택
출력: step6_llm/llm_selected.json
"""

import json
import requests
from pathlib import Path
from datetime import datetime
import time
import re

# 설정
BASE_DIR = Path(__file__).parent.parent
CURRENT_SESSION_DIR = None
EEVE_URL = "http://localhost:8003/v1/chat/completions"  # EEVE 서버

# Kiwi 초기화 (lazy loading)
kiwi = None
KIWI_AVAILABLE = False

def init_kiwi():
    """Kiwi 형태소 분석기 초기화"""
    global kiwi, KIWI_AVAILABLE
    if kiwi is not None:
        return True
    
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        KIWI_AVAILABLE = True
        print("✅ Kiwi 형태소 분석기 로드됨")
        return True
    except Exception as e:
        KIWI_AVAILABLE = False
        print(f"⚠️ Kiwi 사용 불가: {e}")
        return False

# 원 안의 한글 문자 → 원 안의 숫자 매핑
CIRCLED_HANGUL_TO_NUMBER = {
    '㉮': '①',  # 가 → 1
    '㉯': '②',  # 나 → 2
    '㉰': '③',  # 다 → 3
    '㉱': '④',  # 라 → 4
    '㉲': '⑤',  # 마 → 5
    '㉳': '⑥',  # 바 → 6
    '㉴': '⑦',  # 사 → 7
    '㉵': '⑧',  # 아 → 8
    '㉶': '⑨',  # 자 → 9
    '㉷': '⑩',  # 차 → 10
    '㉸': '⑪',  # 카 → 11
    '㉹': '⑫',  # 타 → 12
    '㉺': '⑬',  # 파 → 13
    '㉻': '⑭',  # 하 → 14
}

def normalize_circled_characters(text):
    """원 안의 한글 문자를 원 안의 숫자로 정규화"""
    if not text:
        return text
    
    for hangul, number in CIRCLED_HANGUL_TO_NUMBER.items():
        text = text.replace(hangul, number)
    
    return text

def calculate_kiwi_score(text):
    """Kiwi 형태소 분석으로 텍스트의 자연스러움 점수 계산"""
    if not text or not KIWI_AVAILABLE:
        return 0.5  # 기본 점수
    
    try:
        # 형태소 분석
        result = kiwi.analyze(text)
        if not result or len(result) == 0:
            return 0.3
        
        tokens = result[0][0]  # 첫 번째 분석 결과
        if not tokens:
            return 0.3
        
        total_tokens = len(tokens)
        if total_tokens == 0:
            return 0.3
        
        # 1. UNK(알 수 없는 단어) 비율 계산
        unk_count = sum(1 for token in tokens if token.tag == 'UNK' or token.tag == 'UNKNOWN')
        unk_ratio = unk_count / total_tokens
        
        # 2. 명사/동사/형용사 비율 (의미 있는 품사)
        meaningful_tags = ['NNG', 'NNP', 'NNB', 'VV', 'VA', 'MAG', 'MM']
        meaningful_count = sum(1 for token in tokens if token.tag in meaningful_tags)
        meaningful_ratio = meaningful_count / total_tokens
        
        # 3. 점수 계산 (높을수록 자연스러움)
        score = (1.0 - unk_ratio) * 0.7 + meaningful_ratio * 0.3
        
        return score
    except Exception as e:
        return 0.5


def apply_known_corrections(text):
    """알려진 OCR 오타를 수정"""
    if not text:
        return text
    
    corrections = {
        '선휴수준': '성취수준',
        '보장도시': '보장지도', 
        '떠산을': '덕산읍',
        '고용EK정책원': '교육과정평가원',
        '체목': '체육',
        '고교학정제': '고교학점제',
        '연관력관': '연구협력관',
        '조기회': '조기희',
        '연 구 진': '연구진',
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text


def select_best_ocr_with_llm(ocr_raw, ocr_preprocessed, context_before="", context_after="", max_retries=2):
    """Kiwi 형태소 분석 기반으로 두 OCR 결과 중 더 나은 것을 선택하고 교정"""
    # 먼저 원 안의 한글 문자 정규화
    ocr_raw = normalize_circled_characters(ocr_raw)
    ocr_preprocessed = normalize_circled_characters(ocr_preprocessed)
    
    # 두 값이 동일하거나 하나만 있으면 간단하게 처리
    if not ocr_raw and not ocr_preprocessed:
        return "", False, "both_empty"
    
    if not ocr_raw:
        return apply_known_corrections(ocr_preprocessed), False, "only_preprocessed"
    
    if not ocr_preprocessed:
        return apply_known_corrections(ocr_raw), False, "only_raw"
    
    if ocr_raw == ocr_preprocessed:
        return apply_known_corrections(ocr_raw), False, "same"
    
    # 두 값이 다를 때: Kiwi 형태소 분석 기반 선택
    len_raw = len(ocr_raw.strip())
    len_prep = len(ocr_preprocessed.strip())
    
    # 전처리가 너무 이상하면 원본 사용
    if len_prep < 3 or (len_raw > 10 and len_prep > len_raw * 3):
        return apply_known_corrections(ocr_raw), True, "kiwi_selected"
    
    # URL, 이메일, 전화번호가 포함된 경우 원본 우선
    if any(pattern in ocr_raw for pattern in ['http://', 'https://', 'www.', '@', '043)', '02)']):
        # 전처리에 이런 패턴이 없으면 원본 사용
        if not any(pattern in ocr_preprocessed for pattern in ['http://', 'https://', 'www.', '@']):
            return apply_known_corrections(ocr_raw), True, "kiwi_selected"
    
    # 한자나 이상한 기호가 많으면 배제
    weird_chars = sum(1 for c in ocr_preprocessed if ord(c) > 0x4E00 and ord(c) < 0x9FFF)  # 중국어 한자
    if weird_chars > len_prep * 0.3:  # 30% 이상이 한자면 이상함
        return apply_known_corrections(ocr_raw), True, "kiwi_selected"
    
    # Kiwi로 자연스러움 점수 계산
    score_raw = calculate_kiwi_score(ocr_raw)
    score_prep = calculate_kiwi_score(ocr_preprocessed)
    
    # 점수 차이가 명확하면 (0.15 이상) 높은 쪽 선택
    if abs(score_raw - score_prep) > 0.15:
        if score_raw > score_prep:
            selected_text = ocr_raw
        else:
            selected_text = ocr_preprocessed
    else:
        # 점수가 비슷하면 기존 규칙 적용
        korean_ratio_raw = sum(1 for c in ocr_raw if '\uac00' <= c <= '\ud7a3') / max(len_raw, 1)
        korean_ratio_prep = sum(1 for c in ocr_preprocessed if '\uac00' <= c <= '\ud7a3') / max(len_prep, 1)
        
        # 한글 비율이 더 높은 것 선택
        if korean_ratio_prep > korean_ratio_raw * 1.2:
            selected_text = ocr_preprocessed
        elif korean_ratio_raw > korean_ratio_prep * 1.2:
            selected_text = ocr_raw
        # 한글 비율이 비슷하면 더 긴 것 선택
        elif len_prep > len_raw * 1.1:
            selected_text = ocr_preprocessed
        else:
            selected_text = ocr_raw
    
    return apply_known_corrections(selected_text), True, "kiwi_selected"

def process_blocks_with_dual_ocr(pages_data):
    """블록별 이중 OCR 비교 및 LLM 선택"""
    # LLM 서버 확인
    try:
        response = requests.get("http://localhost:8003/health", timeout=5)
        print("✅ LLM 서버 연결됨")
    except:
        print("❌ LLM 서버에 연결할 수 없습니다.")
        print("   원본 OCR 결과만 사용합니다.")
        # LLM 없이 원본 사용
        for page in pages_data:
            for block in page.get('blocks', []):
                if 'ocr_raw' in block:
                    block['text'] = block['ocr_raw']
                    block['selection_method'] = 'raw_only'
        return pages_data
    
    total_blocks = 0
    total_changed = 0
    selection_stats = {
        'same': 0,
        'only_raw': 0,
        'only_preprocessed': 0,
        'llm_selected': 0,
        'llm_failed': 0
    }
    
    for page_idx, page in enumerate(pages_data, 1):
        page_num = page.get('page_number', page_idx)
        blocks = page.get('blocks', [])
        
        print(f"\n📄 페이지 {page_num}: {len(blocks)}개 블록 처리 중...")
        
        for block_idx, block in enumerate(blocks):
            category = block.get('category', '').lower()
            
            # Picture/Image는 건너뜀
            if category in ['picture', 'image']:
                block['text'] = ""
                block['selection_method'] = 'picture'
                continue
            
            ocr_raw = block.get('ocr_raw', '')
            ocr_preprocessed = block.get('ocr_preprocessed', '')
            
            # 문맥 정보 (이전/이후 블록)
            context_before = ""
            context_after = ""
            
            if block_idx > 0:
                prev_block = blocks[block_idx - 1]
                context_before = prev_block.get('text', prev_block.get('ocr_raw', ''))[:50]
            
            if block_idx < len(blocks) - 1:
                next_block = blocks[block_idx + 1]
                context_after = next_block.get('text', next_block.get('ocr_raw', ''))[:50]
            
            # LLM으로 최적 값 선택
            selected_text, changed, method = select_best_ocr_with_llm(
                ocr_raw, ocr_preprocessed, context_before, context_after
            )
            
            block['text'] = selected_text
            block['selection_method'] = method
            
            selection_stats[method] = selection_stats.get(method, 0) + 1
            total_blocks += 1
            
            if changed:
                total_changed += 1
                print(f"    [{block_idx+1:03d}] {category:15s} ✏️ 변경: '{ocr_raw[:30]}...' → '{selected_text[:30]}...' ({method})")
            else:
                print(f"    [{block_idx+1:03d}] {category:15s} ℹ️ 유지: '{selected_text[:30]}...' ({method})")
            
            # API 과부하 방지
            if method == 'llm_selected':
                time.sleep(1.5)
    
    print(f"\n📊 이중 OCR 선택 통계:")
    print(f"   • 총 블록: {total_blocks}개")
    print(f"   • 변경됨: {total_changed}개")
    print(f"   • 동일: {selection_stats.get('same', 0)}개")
    print(f"   • 원본만: {selection_stats.get('only_raw', 0)}개")
    print(f"   • 전처리만: {selection_stats.get('only_preprocessed', 0)}개")
    print(f"   • Kiwi 선택: {selection_stats.get('kiwi_selected', 0)}개")
    
    return pages_data

def main():
    print("=" * 60)
    print("🤖 Step 6: 이중 OCR 비교 및 Kiwi 선택")
    print("=" * 60)
    
    # Kiwi 초기화
    init_kiwi()
    
    # 현재 세션 정보 읽기
    session_file = BASE_DIR / "current_session.json"
    if not session_file.exists():
        print("❌ 현재 세션 정보가 없습니다.")
        return
    
    with open(session_file, 'r', encoding='utf-8') as f:
        session_info = json.load(f)
    
    global CURRENT_SESSION_DIR
    CURRENT_SESSION_DIR = Path(session_info['session_dir'])
    
    # step1b 결과 파일 로드 (이중 OCR 결과)
    step1b_dir = CURRENT_SESSION_DIR / "step1b_dual"
    combined_file = step1b_dir / "combined.json"
    
    if not combined_file.exists():
        print(f"❌ step1b 결과를 찾을 수 없습니다: {combined_file}")
        print("   step1b를 먼저 실행하세요.")
        return
    
    with open(combined_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pages = data.get('pages', [])
    if not pages:
        print("❌ 페이지 데이터가 없습니다.")
        return
    
    print(f"📄 페이지 수: {len(pages)}개")
    
    # 이중 OCR 비교 및 LLM 선택
    print("\n🤖 이중 OCR 비교 및 LLM 선택 시작...")
    processed_pages = process_blocks_with_dual_ocr(pages)
    
    # 결과 저장
    output_data = {
        "metadata": {
            "document_name": data.get('metadata', {}).get('document_name', 'unknown'),
            "total_pages": len(processed_pages),
            "pipeline_stage": "이중 OCR 비교 및 Kiwi 선택",
            "timestamp": datetime.now().isoformat(),
            "selection_method": "Kiwi 형태소 분석 + 규칙 기반",
            "ocr_comparison": {
                "ocr_raw": "원본 이미지 OCR",
                "ocr_preprocessed": "전처리 이미지 OCR",
                "selection": "Kiwi 형태소 분석으로 자연스러움 점수 계산 후 선택"
            }
        },
        "pages": processed_pages
    }
    
    step6_dir = CURRENT_SESSION_DIR / "step6_llm"
    step6_dir.mkdir(parents=True, exist_ok=True)
    output_file = step6_dir / "llm_selected.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 이중 OCR 비교 및 LLM 선택 완료!")
    print(f"📁 결과: {output_file}")

if __name__ == "__main__":
    main()
