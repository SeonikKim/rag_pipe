#!/usr/bin/env python3
"""
⑦ 최종 JSON 생성
입력: corrections/llm_intermediate.json
처리: 섹션별 본문을 하나의 문자열로 결합
출력: final_outputs/doc1_final.json
"""

import json
from pathlib import Path
from datetime import datetime

# 설정
BASE_DIR = Path(__file__).parent.parent
CORRECTIONS_DIR = BASE_DIR / "corrections"
FINAL_DIR = BASE_DIR / "final_outputs"

def clean_final_text(text):
    """최종 텍스트 정리"""
    if not text:
        return ""
    
    # 연속 공백 정리
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # 문장 부호 정리
    text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
    text = re.sub(r'\s*([,;:])\s*', r'\1 ', text)
    
    # 괄호 정리
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    # 끝 공백 제거
    text = text.strip()
    
    # 마지막에 마침표가 없으면 추가
    if text and not text.endswith(('.', '!', '?')):
        # 한국어 종결어미로 끝나는지 확인
        if text.endswith(('다', '요', '니다', '습니다', '세요', '죠')):
            text += '.'
    
    return text

def create_final_json(data):
    """최종 JSON 구조 생성"""
    sections = data.get('sections', [])
    metadata = data.get('metadata', {})
    
    # 문서 ID 결정
    document_id = data.get('document_id', 'unknown')
    if document_id == 'unknown':
        # 메타데이터에서 추출 시도
        source_pdf = metadata.get('source_pdf', '')
        if source_pdf:
            document_id = Path(source_pdf).stem
        else:
            document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 섹션 처리
    final_sections = []
    
    for section in sections:
        # 텍스트 정리
        text_combined = clean_final_text(section.get('text_combined', ''))
        
        # 빈 섹션 제외
        if not text_combined or len(text_combined.strip()) < 10:
            continue
        
        final_section = {
            "id": section.get('id', f"section_{len(final_sections)+1:03d}"),
            "title": section.get('title', '').strip(),
            "category": section.get('category', ''),
            "pages": sorted(section.get('pages', [])),
            "text_combined": text_combined
        }
        
        final_sections.append(final_section)
    
    # 파이프라인 정보 구성
    pipeline_steps = [
        "DotsOCR 1차",
        "2차 OCR(고정 옵션: Gaussian 5×5 → Unsharp r=1.3, p=150 → 업스케일 ×2.8)",
        "형태소 교정(Kiwi)",
        "문맥 교정(LLM)",
        "섹션 병합(BeautifulSoup4)"
    ]
    
    # 최종 JSON 구조
    final_json = {
        "document_id": document_id,
        "sections": final_sections,
        "metadata": {
            "source_pdf": metadata.get('source_pdf', f"pdf_in/{document_id}.pdf"),
            "pipeline": pipeline_steps,
            "timestamp": datetime.now().isoformat(),
            "total_sections": len(final_sections),
            "processing_stats": {
                "kiwi_available": metadata.get('kiwi_available', False),
                "llm_model": metadata.get('llm_model', 'EEVE-Korean-Instruct-2.8B-v1.0'),
                "pipeline_stage": "최종 완료"
            }
        }
    }
    
    return final_json

def validate_final_json(final_json):
    """최종 JSON 검증"""
    issues = []
    
    # 필수 필드 확인
    if not final_json.get('document_id'):
        issues.append("document_id가 없습니다")
    
    if not final_json.get('sections'):
        issues.append("sections가 없습니다")
    
    # 섹션 검증
    sections = final_json.get('sections', [])
    for i, section in enumerate(sections):
        if not section.get('text_combined'):
            issues.append(f"섹션 {i+1}: text_combined가 비어있습니다")
        
        if not section.get('id'):
            issues.append(f"섹션 {i+1}: id가 없습니다")
    
    return issues

def generate_summary(final_json):
    """최종 결과 요약"""
    sections = final_json.get('sections', [])
    
    total_chars = sum(len(s.get('text_combined', '')) for s in sections)
    total_pages = set()
    
    for section in sections:
        total_pages.update(section.get('pages', []))
    
    categories = {}
    for section in sections:
        cat = section.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        'total_sections': len(sections),
        'total_characters': total_chars,
        'total_pages': len(total_pages),
        'categories': categories,
        'avg_section_length': total_chars // len(sections) if sections else 0
    }

def main():
    print("=" * 60)
    print("📄 최종 JSON 생성")
    print("=" * 60)
    
    # 디렉토리 생성
    FINAL_DIR.mkdir(exist_ok=True)
    
    # LLM 교정 결과 로드
    llm_file = CORRECTIONS_DIR / "llm_intermediate.json"
    if not llm_file.exists():
        print(f"❌ LLM 교정 결과를 찾을 수 없습니다: {llm_file}")
        return
    
    with open(llm_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📂 로드: {llm_file.name}")
    
    # 최종 JSON 생성
    print("🏗️ 최종 JSON 구조 생성...")
    final_json = create_final_json(data)
    
    # 검증
    print("🔍 JSON 검증...")
    issues = validate_final_json(final_json)
    if issues:
        print("⚠️ 검증 이슈:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ 검증 통과")
    
    # 요약 생성
    summary = generate_summary(final_json)
    
    # 파일 저장
    document_id = final_json['document_id']
    output_file = FINAL_DIR / f"{document_id}_final.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 최종 JSON 생성 완료!")
    print(f"📁 결과: {output_file}")
    print(f"\n📊 최종 통계:")
    print(f"   • 총 섹션: {summary['total_sections']}개")
    print(f"   • 총 문자: {summary['total_characters']:,}자")
    print(f"   • 총 페이지: {summary['total_pages']}페이지")
    print(f"   • 평균 섹션 길이: {summary['avg_section_length']}자")
    
    if summary['categories']:
        print(f"   • 카테고리별:")
        for cat, count in summary['categories'].items():
            print(f"     - {cat}: {count}개")
    
    # 샘플 섹션 미리보기
    sections = final_json.get('sections', [])
    if sections:
        print(f"\n📋 섹션 미리보기 (처음 3개):")
        for i, section in enumerate(sections[:3]):
            title = section['title'][:40] + ('...' if len(section['title']) > 40 else '')
            text_preview = section['text_combined'][:80] + ('...' if len(section['text_combined']) > 80 else '')
            print(f"   {i+1}. [{section['id']}] {title}")
            print(f"      {text_preview}")

if __name__ == "__main__":
    main()
