#!/usr/bin/env python3
"""
⑤ 1차 교정 (형태소 분석)
입력: layout_combined/doc1_sections.json
처리: Kiwi로 띄어쓰기, 자모 분리/합침, 간단 조사 오류 보정
출력: corrections/kiwi_intermediate.json
"""

import json
import re
from pathlib import Path
from datetime import datetime

# 설정
BASE_DIR = Path(__file__).parent.parent
LAYOUT_DIR = BASE_DIR / "layout_combined"
CORRECTIONS_DIR = BASE_DIR / "corrections"

# Kiwi 초기화 (lazy loading)
kiwi = None
KIWI_AVAILABLE = False

def init_kiwi():
    """Kiwi 초기화"""
    global kiwi, KIWI_AVAILABLE
    if kiwi is not None:
        return True
    
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        KIWI_AVAILABLE = True
        print("✅ Kiwi 형태소 분석기 로드")
        return True
    except Exception as e:
        KIWI_AVAILABLE = False
        print(f"⚠️ Kiwi 사용 불가: {e}")
        return False

def normalize_spacing(text):
    """기본 공백 정규화"""
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 문장부호 앞뒤 공백 정리
    text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
    
    # 괄호 안쪽 공백 제거
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    return text.strip()

def fix_jamo_separation(text):
    """자모 분리/합침 오류 수정"""
    # 자주 발생하는 자모 분리 패턴들
    jamo_fixes = {
        'ㄱ ㅏ': '가', 'ㄴ ㅏ': '나', 'ㄷ ㅏ': '다', 'ㄹ ㅏ': '라',
        'ㅁ ㅏ': '마', 'ㅂ ㅏ': '바', 'ㅅ ㅏ': '사', 'ㅇ ㅏ': '아',
        'ㅈ ㅏ': '자', 'ㅊ ㅏ': '차', 'ㅋ ㅏ': '카', 'ㅌ ㅏ': '타',
        'ㅍ ㅏ': '파', 'ㅎ ㅏ': '하',
        
        'ㄱ ㅓ': '거', 'ㄴ ㅓ': '너', 'ㄷ ㅓ': '더', 'ㄹ ㅓ': '러',
        'ㅁ ㅓ': '머', 'ㅂ ㅓ': '버', 'ㅅ ㅓ': '서', 'ㅇ ㅓ': '어',
        
        'ㄱ ㅗ': '고', 'ㄴ ㅗ': '노', 'ㄷ ㅗ': '도', 'ㄹ ㅗ': '로',
        'ㅁ ㅗ': '모', 'ㅂ ㅗ': '보', 'ㅅ ㅗ': '소', 'ㅇ ㅗ': '오',
        
        'ㄱ ㅜ': '구', 'ㄴ ㅜ': '누', 'ㄷ ㅜ': '두', 'ㄹ ㅜ': '루',
        'ㅁ ㅜ': '무', 'ㅂ ㅜ': '부', 'ㅅ ㅜ': '수', 'ㅇ ㅜ': '우',
    }
    
    for wrong, correct in jamo_fixes.items():
        text = text.replace(wrong, correct)
    
    return text

def correct_common_ocr_errors(text):
    """일반적인 OCR 오류 수정"""
    corrections = {
        # 숫자/문자 혼동
        '0': 'O', '1': 'l', '5': 'S',
        
        # 한글 오타
        '무름': '무릎', '무abar': '무릎', '무bar': '무릎',
        '윗몸 말아올리기': '윗몸일으키기',
        '팔굽혀펴기': '팔굽혀펴기',
        
        # 단위 정리
        '°': '도', '℃': '도', '%': '퍼센트',
        
        # 공백 오류
        '90 °': '90도', '90°': '90도',
        '1 0': '10', '2 0': '20', '3 0': '30',
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text

def kiwi_spacing_correction(text):
    """Kiwi를 사용한 띄어쓰기 교정"""
    if not KIWI_AVAILABLE:
        return text
    
    try:
        # 문장 단위로 분할
        sentences = re.split(r'[.!?]\s*', text)
        corrected_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 3:
                corrected_sentences.append(sentence)
                continue
            
            # Kiwi 분석
            result = kiwi.analyze(sentence.strip())
            if result:
                # 형태소 기반 띄어쓰기 재구성
                tokens = []
                for token in result[0]:
                    form = token.form
                    tag = token.tag
                    
                    # 조사는 앞 단어와 붙임
                    if tag.startswith('J'):  # 조사
                        if tokens:
                            tokens[-1] += form
                        else:
                            tokens.append(form)
                    else:
                        tokens.append(form)
                
                corrected = ' '.join(tokens)
                corrected_sentences.append(corrected)
            else:
                corrected_sentences.append(sentence)
        
        return '. '.join(corrected_sentences).strip()
        
    except Exception as e:
        print(f"⚠️ Kiwi 교정 실패: {e}")
        return text

def correct_section_text(text):
    """섹션 텍스트 종합 교정"""
    if not text or len(text.strip()) < 3:
        return text
    
    # 1. 기본 정규화
    corrected = normalize_spacing(text)
    
    # 2. 자모 분리 수정
    corrected = fix_jamo_separation(corrected)
    
    # 3. 일반적인 OCR 오류 수정
    corrected = correct_common_ocr_errors(corrected)
    
    # 4. Kiwi 띄어쓰기 교정
    corrected = kiwi_spacing_correction(corrected)
    
    return corrected

def process_sections(sections_data):
    """섹션별 텍스트 교정"""
    if not init_kiwi():
        print("⚠️ Kiwi 없이 기본 교정만 수행")
    
    corrected_sections = []
    
    for i, section in enumerate(sections_data, 1):
        print(f"[{i}/{len(sections_data)}] 섹션 교정: {section['title'][:30]}...")
        
        original_text = section.get('text_combined', '')
        corrected_text = correct_section_text(original_text)
        
        # 교정 결과 저장
        corrected_section = section.copy()
        corrected_section['text_combined'] = corrected_text
        corrected_section['correction_applied'] = True
        corrected_section['correction_method'] = 'kiwi_morpheme'
        
        # 변경 사항 기록
        if corrected_text != original_text:
            corrected_section['text_changed'] = True
            print(f"    ✏️ 교정됨: {len(original_text)} → {len(corrected_text)}자")
        else:
            corrected_section['text_changed'] = False
            print(f"    ℹ️ 변경 없음")
        
        corrected_sections.append(corrected_section)
    
    return corrected_sections

def main():
    print("=" * 60)
    print("🔤 1차 교정 (형태소 분석)")
    print("=" * 60)
    
    # 디렉토리 생성
    CORRECTIONS_DIR.mkdir(exist_ok=True)
    
    # 섹션 데이터 로드
    section_files = list(LAYOUT_DIR.glob("*_sections.json"))
    if not section_files:
        print("❌ 섹션 파일을 찾을 수 없습니다.")
        return
    
    section_file = section_files[0]  # 첫 번째 파일 사용
    print(f"📂 로드: {section_file.name}")
    
    with open(section_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sections = data.get('sections', [])
    if not sections:
        print("❌ 섹션 데이터가 없습니다.")
        return
    
    print(f"📄 섹션 수: {len(sections)}개")
    
    # 교정 처리
    print("\n🔤 형태소 기반 교정 시작...")
    corrected_sections = process_sections(sections)
    
    # 결과 저장
    output_data = data.copy()
    output_data['sections'] = corrected_sections
    output_data['metadata']['pipeline_stage'] = '1차 교정 (형태소)'
    output_data['metadata']['timestamp'] = datetime.now().isoformat()
    output_data['metadata']['kiwi_available'] = KIWI_AVAILABLE
    
    output_file = CORRECTIONS_DIR / "kiwi_intermediate.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 통계
    changed_count = sum(1 for s in corrected_sections if s.get('text_changed', False))
    
    print(f"\n🎉 1차 교정 완료!")
    print(f"📄 교정된 섹션: {changed_count}/{len(corrected_sections)}개")
    print(f"📁 결과: {output_file}")

if __name__ == "__main__":
    main()
