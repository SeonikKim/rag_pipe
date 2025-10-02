#!/usr/bin/env python3
"""
⑥ 2차 교정 (문맥 보정)
입력: corrections/kiwi_intermediate.json
처리: LLM으로 문맥 의미 교정
출력: corrections/llm_intermediate.json
"""

import json
import requests
from pathlib import Path
from datetime import datetime
import time

# 설정
BASE_DIR = Path(__file__).parent.parent
CORRECTIONS_DIR = BASE_DIR / "corrections"
EEVE_URL = "http://localhost:8003/v1/chat/completions"  # EEVE 서버

def correct_text_with_llm(text, max_retries=3):
    """LLM으로 문맥 교정"""
    if len(text.strip()) < 5:
        return text, False
    
    prompt = f"""너는 OCR 텍스트 교정 전문가다.
다음 규칙을 엄격히 따라라:
- 출력은 반드시 교정된 한국어 문장만 포함한다
- 맞춤법, 띄어쓰기, 문맥 오류를 자연스럽게 고친다
- 설명, 번역, 해설은 절대 하지 않는다
- 원문의 의미를 유지하면서 자연스러운 한국어로 만든다

교정할 텍스트:
{text}"""

    for attempt in range(max_retries):
        try:
            response = requests.post(EEVE_URL, json={
                "model": "yanolja/EEVE-Korean-Instruct-2.8B-v1.0",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": min(len(text) * 2, 500),
                "repetition_penalty": 1.1
            }, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                corrected = result['choices'][0]['message']['content'].strip()
                
                # 응답 검증
                if corrected and len(corrected) > 3:
                    # 불필요한 설명 제거
                    lines = corrected.split('\n')
                    clean_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        # 설명성 문장 제거
                        if any(phrase in line for phrase in ['교정된', '수정된', '다음과', '아래는', '결과는']):
                            continue
                        if line and not line.startswith(('*', '-', '•')):
                            clean_lines.append(line)
                    
                    if clean_lines:
                        final_text = ' '.join(clean_lines).strip()
                        
                        # 길이 검증 (너무 짧거나 길면 원문 사용)
                        if len(final_text) >= len(text) * 0.5 and len(final_text) <= len(text) * 2:
                            changed = final_text != text
                            return final_text, changed
                
            print(f"    ⚠️ 시도 {attempt + 1} 실패")
            time.sleep(1)  # 재시도 전 대기
            
        except Exception as e:
            print(f"    ❌ LLM 교정 오류 (시도 {attempt + 1}): {e}")
            time.sleep(2)
    
    return text, False

def process_sections_with_llm(sections_data):
    """섹션별 LLM 교정"""
    corrected_sections = []
    total_changed = 0
    
    # LLM 서버 확인
    try:
        response = requests.get("http://localhost:8003/health", timeout=5)
        print("✅ LLM 서버 연결됨")
    except:
        print("❌ LLM 서버에 연결할 수 없습니다.")
        print("   기본 교정만 수행합니다.")
        return sections_data
    
    for i, section in enumerate(sections_data, 1):
        print(f"[{i}/{len(sections_data)}] LLM 교정: {section['title'][:30]}...")
        
        original_text = section.get('text_combined', '')
        
        # 텍스트가 너무 길면 분할 처리
        if len(original_text) > 1000:
            # 문장 단위로 분할
            import re
            sentences = re.split(r'(?<=[.!?])\s+', original_text)
            corrected_parts = []
            section_changed = False
            
            for j, sentence in enumerate(sentences):
                if len(sentence.strip()) > 10:
                    corrected_sentence, changed = correct_text_with_llm(sentence.strip())
                    corrected_parts.append(corrected_sentence)
                    if changed:
                        section_changed = True
                else:
                    corrected_parts.append(sentence)
            
            corrected_text = ' '.join(corrected_parts)
        else:
            corrected_text, section_changed = correct_text_with_llm(original_text)
        
        # 결과 저장
        corrected_section = section.copy()
        corrected_section['text_combined'] = corrected_text
        corrected_section['llm_correction_applied'] = True
        corrected_section['llm_correction_method'] = 'EEVE-2.8B'
        
        if section_changed:
            corrected_section['llm_text_changed'] = True
            total_changed += 1
            print(f"    ✏️ 교정됨: {len(original_text)} → {len(corrected_text)}자")
        else:
            corrected_section['llm_text_changed'] = False
            print(f"    ℹ️ 변경 없음")
        
        corrected_sections.append(corrected_section)
        
        # API 호출 제한을 위한 짧은 대기
        time.sleep(0.5)
    
    print(f"\n📊 LLM 교정 통계: {total_changed}/{len(sections_data)}개 섹션 변경")
    return corrected_sections

def main():
    print("=" * 60)
    print("🤖 2차 교정 (LLM 문맥 보정)")
    print("=" * 60)
    
    # Kiwi 교정 결과 로드
    kiwi_file = CORRECTIONS_DIR / "kiwi_intermediate.json"
    if not kiwi_file.exists():
        print(f"❌ Kiwi 교정 결과를 찾을 수 없습니다: {kiwi_file}")
        return
    
    with open(kiwi_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sections = data.get('sections', [])
    if not sections:
        print("❌ 섹션 데이터가 없습니다.")
        return
    
    print(f"📄 섹션 수: {len(sections)}개")
    
    # LLM 교정 처리
    print("\n🤖 LLM 문맥 교정 시작...")
    corrected_sections = process_sections_with_llm(sections)
    
    # 결과 저장
    output_data = data.copy()
    output_data['sections'] = corrected_sections
    output_data['metadata']['pipeline_stage'] = '2차 교정 (LLM)'
    output_data['metadata']['timestamp'] = datetime.now().isoformat()
    output_data['metadata']['llm_model'] = 'EEVE-Korean-Instruct-2.8B-v1.0'
    
    output_file = CORRECTIONS_DIR / "llm_intermediate.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 통계
    kiwi_changed = sum(1 for s in corrected_sections if s.get('text_changed', False))
    llm_changed = sum(1 for s in corrected_sections if s.get('llm_text_changed', False))
    
    print(f"\n🎉 2차 교정 완료!")
    print(f"📊 교정 통계:")
    print(f"   • Kiwi 교정: {kiwi_changed}개 섹션")
    print(f"   • LLM 교정: {llm_changed}개 섹션")
    print(f"📁 결과: {output_file}")

if __name__ == "__main__":
    main()
