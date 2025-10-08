#!/usr/bin/env python3
"""
④ 레이아웃 병합 (섹션 단위)
입력: ocr_refined/refined_combined.json
처리: BeautifulSoup4로 DOM 유사 구조 구성 → 섹션 병합
출력: layout_combined/doc1_sections.json
"""

import json
import re
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup, Tag

# 설정
BASE_DIR = Path(__file__).parent.parent
OCR_DIR = BASE_DIR / "ocr_results"
LAYOUT_DIR = OCR_DIR / "step4_layout"
CURRENT_SESSION_DIR = None

def html_table_to_text(html_text):
    """HTML 테이블을 읽기 쉬운 텍스트로 변환"""
    if not html_text or '<table' not in html_text.lower():
        return html_text
    
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        tables = soup.find_all('table')
        
        if not tables:
            return html_text
        
        result_parts = []
        
        for table in tables:
            table_text = []
            
            # 테이블 행 처리
            rows = table.find_all('tr')
            for row in rows:
                # 헤더 셀 (th)과 데이터 셀 (td) 모두 처리
                cells = row.find_all(['th', 'td'])
                cell_texts = [cell.get_text().strip() for cell in cells]
                
                # 빈 셀 제거 후 병합
                cell_texts = [c for c in cell_texts if c]
                if cell_texts:
                    row_text = ' | '.join(cell_texts)
                    table_text.append(row_text)
            
            # 테이블을 텍스트로 변환
            if table_text:
                result_parts.append('\n'.join(table_text))
        
        # 테이블 외 텍스트도 포함
        for elem in soup.children:
            if elem.name != 'table' and hasattr(elem, 'get_text'):
                text = elem.get_text().strip()
                if text:
                    result_parts.append(text)
        
        return '\n'.join(result_parts)
        
    except Exception as e:
        print(f"    ⚠️ 테이블 변환 오류: {e}")
        return html_text

def create_dom_structure(pages_data):
    """페이지 데이터를 DOM 구조로 변환 (HTML 테이블은 텍스트로 변환)"""
    soup = BeautifulSoup('<document></document>', 'html.parser')
    doc_root = soup.document
    
    for page_data in pages_data:
        page_num = page_data.get('page_number', 0)
        page_tag = soup.new_tag('page', number=page_num)
        
        blocks = page_data.get('blocks', [])
        for i, block in enumerate(blocks):
            category = block.get('category', '').lower()
            text = block.get('text', '').strip()
            
            # Table 카테고리는 HTML을 텍스트로 변환
            if category == 'table':
                text = html_table_to_text(text)
                print(f"    📊 Table 변환: {len(block.get('text', ''))}자 → {len(text)}자")
            
            block_tag = soup.new_tag('block', 
                                   index=i,
                                   category=category,
                                   confidence=block.get('confidence', 0.0))
            block_tag.string = text
            page_tag.append(block_tag)
        
        doc_root.append(page_tag)
    
    return soup

def detect_section_headers(soup):
    """섹션 헤더 패턴 감지"""
    header_patterns = [
        r'^\d+\.\s*',  # "1. 제목"
        r'^[가-힣]\.\s*',  # "가. 제목"
        r'^\(\d+\)\s*',  # "(1) 제목"
        r'^[①-⑳]\s*',  # "① 제목"
        r'^제\s*\d+\s*장',  # "제1장"
        r'^[0-9]+\s*[장절항]',  # "1장", "2절"
    ]
    
    sections = []
    
    for block in soup.find_all('block'):
        text = block.get_text().strip()
        category = block.get('category', '').lower()
        
        # 카테고리 기반 판단
        if 'header' in category or 'title' in category:
            sections.append({
                'element': block,
                'text': text,
                'type': 'category_header',
                'level': 1
            })
            continue
        
        # 패턴 기반 판단
        for i, pattern in enumerate(header_patterns):
            if re.match(pattern, text):
                level = i + 1  # 패턴 순서로 레벨 결정
                sections.append({
                    'element': block,
                    'text': text,
                    'type': 'pattern_header',
                    'level': level
                })
                break
    
    return sections

def merge_sections(soup, section_headers):
    """섹션별로 블록들을 병합"""
    sections = []
    current_section = None
    orphan_blocks = []  # 헤더 없이 시작하는 블록들
    
    all_blocks = soup.find_all('block')
    header_blocks = {header['element'] for header in section_headers}
    
    for block in all_blocks:
        if block in header_blocks:
            # 첫 섹션 전에 고아 블록이 있으면 섹션으로 만들기
            if not current_section and orphan_blocks:
                orphan_section = {
                    'id': f"section_{len(sections)+1:03d}",
                    'title': '(머리말)',  # 기본 제목
                    'category': 'orphan',
                    'level': 0,
                    'pages': list(set(int(b.find_parent('page')['number']) for b in orphan_blocks)),
                    'blocks': orphan_blocks.copy(),
                    'text_parts': [b.get_text().strip() for b in orphan_blocks if b.get_text().strip() and len(b.get_text().strip()) > 2]
                }
                sections.append(orphan_section)
                orphan_blocks = []
            
            # 새 섹션 시작
            if current_section:
                sections.append(current_section)
            
            # 헤더 정보 찾기
            header_info = next(h for h in section_headers if h['element'] == block)
            
            current_section = {
                'id': f"section_{len(sections)+1:03d}",
                'title': header_info['text'],
                'category': block.get('category', ''),
                'level': header_info['level'],
                'pages': [int(block.find_parent('page')['number'])],
                'blocks': [block],
                'text_parts': []
            }
        else:
            # 기존 섹션에 블록 추가
            if current_section:
                page_num = int(block.find_parent('page')['number'])
                if page_num not in current_section['pages']:
                    current_section['pages'].append(page_num)
                
                current_section['blocks'].append(block)
                
                # 텍스트 추가 (빈 텍스트 제외)
                text = block.get_text().strip()
                if text and len(text) > 2:
                    current_section['text_parts'].append(text)
            else:
                # 아직 섹션이 시작 안됐으면 고아 블록으로 저장
                orphan_blocks.append(block)
    
    # 마지막 섹션 추가
    if current_section:
        sections.append(current_section)
    
    # 끝에 남은 고아 블록들도 섹션으로 추가
    if orphan_blocks:
        orphan_section = {
            'id': f"section_{len(sections)+1:03d}",
            'title': '(추가 내용)',
            'category': 'orphan',
            'level': 0,
            'pages': list(set(int(b.find_parent('page')['number']) for b in orphan_blocks)),
            'blocks': orphan_blocks.copy(),
            'text_parts': [b.get_text().strip() for b in orphan_blocks if b.get_text().strip() and len(b.get_text().strip()) > 2]
        }
        sections.append(orphan_section)
    
    return sections

def combine_section_text(section):
    """섹션 내 텍스트들을 자연스럽게 결합"""
    text_parts = section['text_parts']
    if not text_parts:
        return ""
    
    combined = []
    
    for i, part in enumerate(text_parts):
        # 문장 끝 처리
        if i > 0:
            prev_part = combined[-1]
            # 이전 문장이 완전하지 않으면 공백으로 연결
            if not prev_part.endswith(('.', '!', '?', '다', '요', '니다')):
                combined.append(' ')
        
        combined.append(part)
        
        # 문장 끝에 마침표 추가 (필요한 경우)
        if not part.endswith(('.', '!', '?')) and i < len(text_parts) - 1:
            next_part = text_parts[i + 1]
            # 다음 문장이 대문자나 숫자로 시작하면 마침표 추가
            if next_part and (next_part[0].isupper() or next_part[0].isdigit()):
                combined.append('.')
    
    return ''.join(combined).strip()

def process_refined_data():
    """정제된 데이터를 섹션별로 병합"""
    # 현재 세션 정보 읽기
    session_file = BASE_DIR / "current_session.json"
    if not session_file.exists():
        print("❌ 현재 세션 정보가 없습니다.")
        return None
    
    with open(session_file, 'r', encoding='utf-8') as f:
        session_info = json.load(f)
    
    global CURRENT_SESSION_DIR
    CURRENT_SESSION_DIR = Path(session_info['session_dir'])
    
    # step1 결과 파일들 로드 (통합된 파이프라인)
    step1_dir = CURRENT_SESSION_DIR / "step1_primary"
    combined_file = step1_dir / "combined.json"
    
    if not combined_file.exists():
        print(f"❌ 통합 파일을 찾을 수 없습니다: {combined_file}")
        return None
    
    print(f"📂 사용할 파일: {combined_file.name}")
    
    with open(combined_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pages_data = data.get('pages', [])
    if not pages_data:
        print("❌ 페이지 데이터가 없습니다.")
        return None
    
    print(f"📄 로드된 페이지: {len(pages_data)}개")
    
    # DOM 구조 생성
    print("🏗️ DOM 구조 생성...")
    soup = create_dom_structure(pages_data)
    
    # 섹션 헤더 감지
    print("🔍 섹션 헤더 감지...")
    section_headers = detect_section_headers(soup)
    print(f"   발견된 헤더: {len(section_headers)}개")
    
    # 섹션 병합
    print("🔗 섹션 병합...")
    sections = merge_sections(soup, section_headers)
    
    # 텍스트 결합
    print("📝 텍스트 결합...")
    for section in sections:
        section['text_combined'] = combine_section_text(section)
        # DOM 요소 제거 (JSON 직렬화를 위해)
        del section['blocks']
        del section['text_parts']
    
    print(f"✅ {len(sections)}개 섹션 생성")
    
    return {
        'document_id': data.get('document_id', 'unknown'),
        'sections': sections,
        'metadata': {
            'pipeline_stage': '섹션 병합',
            'timestamp': datetime.now().isoformat(),
            'total_sections': len(sections),
            'source_pages': len(pages_data)
        }
    }

def main():
    print("=" * 60)
    print("🔗 레이아웃 병합 (섹션 단위)")
    print("=" * 60)
    
    # 디렉토리 생성
    LAYOUT_DIR.mkdir(exist_ok=True)
    
    # 데이터 처리
    result = process_refined_data()
    if not result:
        return
    
    # 결과 저장 (세션 디렉토리 사용)
    step4_dir = CURRENT_SESSION_DIR / "step4_layout"
    step4_dir.mkdir(parents=True, exist_ok=True)
    
    doc_id = result['document_id']
    output_file = step4_dir / f"{doc_id}_sections.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 섹션 병합 완료!")
    print(f"📄 총 섹션: {len(result['sections'])}개")
    print(f"📁 결과: {output_file}")
    
    # 섹션 미리보기
    print(f"\n📋 섹션 미리보기:")
    for i, section in enumerate(result['sections'][:5]):  # 처음 5개만
        title = section['title'][:50] + ('...' if len(section['title']) > 50 else '')
        text_preview = section['text_combined'][:100] + ('...' if len(section['text_combined']) > 100 else '')
        print(f"   {i+1}. {title}")
        print(f"      {text_preview}")

if __name__ == "__main__":
    main()
