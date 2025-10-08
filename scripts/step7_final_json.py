#!/usr/bin/env python3
"""
⑦ 최종 JSON 생성 (BGE-m3-ko 최적화)
입력: step6_llm/llm_selected.json
처리: BGE-m3-ko 임베딩 모델에 최적화된 JSON 구조 생성
출력: final_outputs/doc1_final.json, final_outputs/doc1_bge_optimized.json
"""

import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.append('/home/cywell/project5')
from bge_m3_ko_optimized_schema import create_bge_optimized_json
from context_aware_schema import create_context_aware_json, create_bge_optimized_context_groups

# 설정
BASE_DIR = Path(__file__).parent.parent
CURRENT_SESSION_DIR = None

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
    print("📄 최종 JSON 생성 (문맥 인식)")
    print("=" * 60)
    
    # 현재 세션 정보 읽기
    session_file = BASE_DIR / "current_session.json"
    if not session_file.exists():
        print("❌ 현재 세션 정보가 없습니다.")
        return
    
    with open(session_file, 'r', encoding='utf-8') as f:
        session_info = json.load(f)
    
    global CURRENT_SESSION_DIR
    CURRENT_SESSION_DIR = Path(session_info['session_dir'])
    
    # 모든 페이지의 step6 결과 로드 (문맥 보존용)
    print("📂 모든 페이지의 교정 결과 로드 중...")
    all_pages_data = []
    
    # step6 디렉토리 확인
    step6_dir = CURRENT_SESSION_DIR / "step6_llm"
    if not step6_dir.exists():
        print(f"❌ LLM 교정 결과 디렉토리를 찾을 수 없습니다: {step6_dir}")
        return
    
    # intermediate JSON 파일 사용
    intermediate_file = step6_dir / "llm_selected.json"
    if not intermediate_file.exists():
        print(f"❌ LLM 교정 결과를 찾을 수 없습니다: {intermediate_file}")
        return
    
    print(f"📄 LLM 교정 결과 로드 중...")
    
    with open(intermediate_file, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
    
    print(f"   ✅ {len(llm_data.get('sections', []))}개 섹션 로드 완료")
    
    # llm_data를 그대로 최종 JSON으로 사용 (이미 완성된 구조)
    print("🏗️ 최종 JSON 생성...")
    final_json = create_final_json(llm_data)
    
    # 문맥 인식 JSON 생성
    print("🔗 문맥 인식 JSON 생성 중...")
    
    # 모든 페이지의 블록 데이터 추출
    all_pages_blocks = []
    for page_data in all_pages_data:
        page_blocks = []
        for section in page_data.get('sections', []):
            for block in section.get('blocks', []):
                page_blocks.append(block)
        all_pages_blocks.append(page_blocks)
    
    # 문맥 인식 JSON 생성
    context_aware_json = create_context_aware_json(all_pages_blocks)
    
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
    
    # 파일 저장 (세션 디렉토리 사용)
    step7_dir = CURRENT_SESSION_DIR / "step7_final"
    step7_dir.mkdir(parents=True, exist_ok=True)
    
    document_id = final_json['document_id']
    output_file = step7_dir / f"{document_id}_final.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)
    
    # 문맥 인식 JSON 저장
    context_output_file = step7_dir / f"{document_id}_context_aware.json"
    with open(context_output_file, 'w', encoding='utf-8') as f:
        json.dump(context_aware_json, f, ensure_ascii=False, indent=2)
    
    # BGE-m3-ko 최적화 JSON 생성 (문맥 보존)
    print(f"\n🤖 BGE-m3-ko 최적화 JSON 생성 중...")
    
    # 문맥 그룹을 BGE-m3-ko 최적화 형태로 변환
    bge_context_groups = create_bge_optimized_context_groups(context_aware_json)
    
    # 페이지 메타데이터
    page_metadata = {
        "document_name": final_json.get("document_id", "unknown"),
        "page_number": final_json.get("page_number", 0),
        "timestamp": final_json.get("timestamp", "")
    }
    
    # 기존 방식 BGE 최적화도 유지 (호환성)
    blocks_for_bge = []
    for section in final_json.get('sections', []):
        for block in section.get('blocks', []):
            blocks_for_bge.append({
                "bbox": block.get("bbox", [0, 0, 0, 0]),
                "category": block.get("category", "Text"),
                "text": block.get("text", ""),
                "confidence": block.get("confidence", 0.9)
            })
    
    bge_optimized_json = create_bge_optimized_json(blocks_for_bge, page_metadata)
    
    # 문맥 인식 BGE 최적화 JSON 생성
    context_bge_json = {
        "document_metadata": {
            "document_id": final_json.get("document_id", "unknown"),
            "total_pages": len(all_pages_data),
            "total_context_groups": len(bge_context_groups),
            "has_multi_page_tables": any(g["group_type"] == "table_continuation" for g in bge_context_groups),
            "language": "ko",
            "embedding_model": "dragonkue/bge-m3-ko"
        },
        "context_groups": bge_context_groups,
        "document_text": "\n".join([g["embedding_text"] for g in bge_context_groups]),
        "embedding_config": {
            "model_name": "dragonkue/bge-m3-ko",
            "embedding_dimension": 1024,
            "similarity_function": "cosine",
            "chunk_strategy": "context_aware",
            "max_chunk_tokens": 4000,
            "preserves_context": True,
            "handles_table_continuations": True
        }
    }
    
    # BGE 최적화 JSON 저장
    bge_output_file = step7_dir / f"{document_id}_bge_optimized.json"
    with open(bge_output_file, 'w', encoding='utf-8') as f:
        json.dump(bge_optimized_json, f, ensure_ascii=False, indent=2)
    
    # 문맥 인식 BGE 최적화 JSON 저장
    context_bge_output_file = step7_dir / f"{document_id}_context_bge_optimized.json"
    with open(context_bge_output_file, 'w', encoding='utf-8') as f:
        json.dump(context_bge_json, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 최종 JSON 생성 완료!")
    print(f"📁 일반 JSON: {output_file}")
    print(f"📁 문맥 인식 JSON: {context_output_file}")
    print(f"📁 BGE-m3-ko 최적화 JSON: {bge_output_file}")
    print(f"📁 문맥 인식 BGE 최적화 JSON: {context_bge_output_file}")
    
    print(f"\n📊 최종 통계:")
    print(f"   • 총 섹션: {summary['total_sections']}개")
    print(f"   • 총 문자: {summary['total_characters']:,}자")
    print(f"   • 총 페이지: {summary['total_pages']}페이지")
    print(f"   • 평균 섹션 길이: {summary['avg_section_length']}자")
    
    if summary['categories']:
        print(f"   • 카테고리별:")
        for cat, count in summary['categories'].items():
            print(f"     - {cat}: {count}개")
    
    # 문맥 인식 통계
    print(f"\n🔗 문맥 인식 통계:")
    print(f"   • 총 페이지: {len(all_pages_data)}개")
    print(f"   • 문맥 그룹: {len(bge_context_groups)}개")
    print(f"   • 멀티페이지 표: {len([g for g in bge_context_groups if g['group_type'] == 'table_continuation'])}개")
    print(f"   • 페이지 간 연결: {len(context_aware_json.get('cross_page_connections', []))}개")
    
    # BGE-m3-ko 최적화 통계
    print(f"\n🤖 BGE-m3-ko 최적화 통계:")
    print(f"   • 임베딩 가능 블록: {len([b for b in bge_optimized_json['blocks'] if b['embedding_ready']])}개")
    print(f"   • 문맥 그룹 임베딩: {len([g for g in bge_context_groups if g['embedding_ready']])}개")
    if len(bge_optimized_json['blocks']) > 0:
        print(f"   • 평균 토큰 수: {sum(b['token_estimate'] for b in bge_optimized_json['blocks']) / len(bge_optimized_json['blocks']):.1f}")
    print(f"   • 문서 텍스트 길이: {len(bge_optimized_json['document_text']):,}자")
    print(f"   • 문맥 보존 여부: {'✅' if context_bge_json['embedding_config']['preserves_context'] else '❌'}")
    print(f"   • 표 연속 처리: {'✅' if context_bge_json['embedding_config']['handles_table_continuations'] else '❌'}")
    
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
