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
REFINED_DIR = BASE_DIR / "ocr_refined"
LAYOUT_DIR = BASE_DIR / "layout_combined"

def create_dom_structure(pages_data):
    """페이지 데이터를 DOM 구조로 변환"""
    soup = BeautifulSoup('<document></document>', 'html.parser')
    doc_root = soup.document
    
    for page_data in pages_data:
        page_num = page_data.get('page_number', 0)
        page_tag = soup.new_tag('page', number=page_num)
        
        blocks = page_data.get('blocks', [])
        for i, block in enumerate(blocks):
            block_tag = soup.new_tag('block', 
                                   index=i,
                                   category=block.get('category', ''),
                                   confidence=block.get('confidence', 0.0))
            block_tag.string = block.get('text', '').strip()
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
    
    all_blocks = soup.find_all('block')
    header_blocks = {header['element'] for header in section_headers}
    
    for block in all_blocks:
        if block in header_blocks:
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
    
    # 마지막 섹션 추가
    if current_section:
        sections.append(current_section)
    
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
    # 통합 파일 로드
    combined_file = REFINED_DIR / "refined_combined.json"
    if not combined_file.exists():
        print(f"❌ 통합 파일을 찾을 수 없습니다: {combined_file}")
        return None
    
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
    
    # 결과 저장
    doc_id = result['document_id']
    output_file = LAYOUT_DIR / f"{doc_id}_sections.json"
    
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
