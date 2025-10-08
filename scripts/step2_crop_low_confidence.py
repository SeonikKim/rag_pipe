#!/usr/bin/env python3
"""
② 저신뢰 블록 선별·크롭
입력: 1차 OCR JSON
처리: confidence < 임계치 블록만 좌표 크롭
출력: crops/low_conf/*.png, crops/images/*.png
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import glob

# 설정
BASE_DIR = Path(__file__).parent.parent
OCR_DIR = BASE_DIR / "ocr_results"
DEBUG_DIR = OCR_DIR / "debug"
# 세션별 디렉토리는 런타임에 설정
CURRENT_SESSION_DIR = None

CONFIDENCE_THRESHOLD = 0.95  # 단어별 confidence 임계치
BLOCK_CONFIDENCE_THRESHOLD = 0.80  # 블록 전체 confidence 임계치
MARGIN = 15  # 크롭 시 여백
WORD_MARGIN = 5  # 단어 크롭 시 여백

def load_ocr_results():
    """1차 OCR 결과 로드 (현재 세션)"""
    # 현재 세션 정보 읽기
    session_file = BASE_DIR / "current_session.json"
    if not session_file.exists():
        print("❌ 현재 세션 정보가 없습니다. step1을 먼저 실행하세요.")
        return []
    
    with open(session_file, 'r', encoding='utf-8') as f:
        session_info = json.load(f)
    
    session_dir = Path(session_info['session_dir'])
    step1_dir = session_dir / "step1_primary"
    
    if not step1_dir.exists():
        return []
    
    # 페이지별 JSON 파일 로드
    results = []
    page_files = sorted(step1_dir.glob("page_*.json"))
    
    for page_file in page_files:
        with open(page_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)
    
    # 글로벌 세션 정보 저장 (다른 단계에서 사용)
    global CURRENT_SESSION_DIR
    CURRENT_SESSION_DIR = session_dir
    
    return results

def is_low_confidence_block(block, threshold=BLOCK_CONFIDENCE_THRESHOLD):
    """저신뢰 블록 판단 (블록 전체 기준)"""
    confidence = block.get('confidence', 0.0)
    
    # confidence가 임계치 미만
    if confidence < threshold:
        return True
    
    # 추가 조건들
    text = block.get('text', '').strip()
    
    # 텍스트가 너무 짧음
    if len(text) < 3:
        return True
    
    # 한글 비율이 낮음
    korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
    if len(text) > 0 and korean_chars / len(text) < 0.6:
        return True
    
    # 특수 문자가 많음
    special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t')
    if len(text) > 0 and special_chars / len(text) > 0.3:
        return True
    
    return False

def has_low_confidence_words(block):
    """저신뢰 단어가 있는지 확인"""
    low_conf_tokens = block.get('low_confidence_tokens', [])
    if not low_conf_tokens:
        return False
    
    # confidence < 0.95인 단어가 있는지 확인
    for token_info in low_conf_tokens:
        confidence = token_info.get('confidence', 0.0)
        if confidence < CONFIDENCE_THRESHOLD:
            return True
    
    return False

def crop_block(image, bbox, margin=MARGIN):
    """블록 크롭 (여백 포함)"""
    h, w = image.shape[:2]
    
    if len(bbox) == 4:
        x, y, width, height = bbox
        x2, y2 = x + width, y + height
    elif len(bbox) == 8:  # 4점 좌표
        xs = [bbox[i] for i in range(0, 8, 2)]
        ys = [bbox[i] for i in range(1, 8, 2)]
        x, y = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
    else:
        return None
    
    # 여백 추가
    x1 = max(0, int(x - margin))
    y1 = max(0, int(y - margin))
    x2 = min(w, int(x2 + margin))
    y2 = min(h, int(y2 + margin))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return image[y1:y2, x1:x2]

def estimate_word_bbox(block_bbox, text, word_index, total_words):
    """단어의 대략적인 bbox 추정"""
    x1, y1, x2, y2 = block_bbox
    
    # 블록의 너비를 단어 수로 나누어 각 단어의 너비 추정
    block_width = x2 - x1
    word_width = block_width / max(total_words, 1)
    
    # 단어의 x 좌표 계산
    word_x1 = x1 + (word_index * word_width)
    word_x2 = min(x2, word_x1 + word_width)
    
    return [int(word_x1), y1, int(word_x2), y2]

def crop_low_confidence_words(image, block, margin=WORD_MARGIN):
    """저신뢰 단어들을 개별적으로 크롭"""
    low_conf_tokens = block.get('low_confidence_tokens', [])
    if not low_conf_tokens:
        return []
    
    block_bbox = block.get('bbox', [0, 0, 100, 100])
    if len(block_bbox) != 4:
        return []
    
    cropped_words = []
    
    for i, token_info in enumerate(low_conf_tokens):
        token = token_info.get('token', '')
        confidence = token_info.get('confidence', 0.0)
        
        # confidence가 임계치 미만인 경우만 크롭
        if confidence < CONFIDENCE_THRESHOLD:
            # 단어의 대략적인 bbox 추정
            word_bbox = estimate_word_bbox(block_bbox, block.get('text', ''), i, len(low_conf_tokens))
            
            # 단어 크롭
            cropped_word = crop_block(image, word_bbox, margin)
            if cropped_word is not None and cropped_word.size > 0:
                cropped_words.append({
                    'cropped_image': cropped_word,
                    'token': token,
                    'confidence': confidence,
                    'word_bbox': word_bbox,
                    'original_bbox': block_bbox
                })
    
    return cropped_words

def categorize_block(block):
    """블록 카테고리 분류"""
    category = block.get('category', '').lower()
    text = block.get('text', '').strip()
    
    # 비텍스트 카테고리
    non_text_categories = ['figure', 'table', 'image', 'chart', 'graph']
    if any(cat in category for cat in non_text_categories):
        return 'images'
    
    # 텍스트가 거의 없는 경우
    if len(text) < 5:
        return 'images'
    
    return 'low_conf'

def process_page(page_data):
    """페이지별 저신뢰 블록 처리"""
    page_num = page_data['page_number']
    doc_name = page_data['document_name']
    
    # 디버그 이미지 로드
    debug_image_path = DEBUG_DIR / f"page_{page_num:03d}.png"
    if not debug_image_path.exists():
        print(f"❌ 디버그 이미지 없음: {debug_image_path}")
        return 0
    
    image = cv2.imread(str(debug_image_path))
    if image is None:
        print(f"❌ 이미지 로드 실패: {debug_image_path}")
        return 0
    
    crop_count = 0
    blocks = page_data.get('blocks', [])
    
    # 세션별 크롭 디렉토리 설정
    step2_dir = CURRENT_SESSION_DIR / "step2_crops"
    low_conf_dir = step2_dir / "low_conf"
    words_dir = step2_dir / "words"
    images_dir = step2_dir / "images"
    
    for i, block in enumerate(blocks):
        if 'bbox' not in block:
            continue
        
        category = block.get('category', '').lower()
        
        # 1. 텍스트 블록에서 저신뢰 단어 크롭
        if category in ['text', 'title', 'section-header'] and has_low_confidence_words(block):
            cropped_words = crop_low_confidence_words(image, block)
            
            for j, word_data in enumerate(cropped_words):
                words_dir.mkdir(parents=True, exist_ok=True)
                
                # 파일명 생성
                filename = f"{doc_name}_page{page_num:03d}_block{i:03d}_word{j:03d}.png"
                output_path = words_dir / filename
                
                # 저장
                cv2.imwrite(str(output_path), word_data['cropped_image'])
                
                # 메타데이터 저장
                meta_path = output_path.with_suffix('.json')
                meta_data = {
                    "source_page": page_num,
                    "block_index": i,
                    "word_index": j,
                    "token": word_data['token'],
                    "confidence": word_data['confidence'],
                    "word_bbox": word_data['word_bbox'],
                    "original_bbox": word_data['original_bbox'],
                    "category": block.get('category', ''),
                    "text_preview": block.get('text', '')[:100],
                    "reason": "low_confidence_word"
                }
                
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta_data, f, ensure_ascii=False, indent=2)
                
                crop_count += 1
                print(f"    🔤 Block {i:03d} Word {j:03d}: '{word_data['token']}' ({word_data['confidence']:.3f})")
        
        # 2. 저신뢰 블록 전체 크롭 (기존 로직)
        elif is_low_confidence_block(block):
            cropped = crop_block(image, block['bbox'])
            if cropped is None or cropped.size == 0:
                continue
            
            # 카테고리별 저장
            block_category = categorize_block(block)
            if block_category == 'low_conf':
                output_dir = low_conf_dir
            else:
                output_dir = images_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 파일명 생성
            filename = f"{doc_name}_page{page_num:03d}_block{i:03d}.png"
            output_path = output_dir / filename
            
            # 저장
            cv2.imwrite(str(output_path), cropped)
            
            # 메타데이터 저장
            meta_path = output_path.with_suffix('.json')
            meta_data = {
                "source_page": page_num,
                "block_index": i,
                "original_bbox": block['bbox'],
                "confidence": block.get('confidence', 0.0),
                "category": block.get('category', ''),
                "text_preview": block.get('text', '')[:100],
                "reason": "low_confidence_block" if block_category == 'low_conf' else "non_text"
            }
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)
            
            crop_count += 1
            print(f"    📦 Block {i:03d}: {block_category} ({block.get('confidence', 0.0):.2f})")
    
    return crop_count

def main():
    print("=" * 60)
    print("🔍 저신뢰 블록 및 단어 선별 및 크롭")
    print("=" * 60)
    
    # 1차 OCR 결과 로드
    print("📂 1차 OCR 결과 로드...")
    ocr_results = load_ocr_results()
    
    if not ocr_results:
        print("❌ 1차 OCR 결과를 찾을 수 없습니다.")
        return
    
    print(f"✅ {len(ocr_results)}페이지 로드됨")
    print(f"🎯 단어별 confidence 임계치: {CONFIDENCE_THRESHOLD}")
    print(f"🎯 블록별 confidence 임계치: {BLOCK_CONFIDENCE_THRESHOLD}")
    
    # 페이지별 처리
    total_crops = 0
    for page_data in ocr_results:
        page_num = page_data['page_number']
        print(f"\n📄 페이지 {page_num} 처리 중...")
        
        crops = process_page(page_data)
        total_crops += crops
        
        if crops > 0:
            print(f"    ✅ {crops}개 항목 크롭됨")
        else:
            print(f"    ℹ️ 저신뢰 항목 없음")
    
    print(f"\n🎉 크롭 완료!")
    print(f"📦 총 {total_crops}개 항목 크롭됨")
    if CURRENT_SESSION_DIR:
        step2_dir = CURRENT_SESSION_DIR / "step2_crops"
        print(f"📁 저장 위치: {step2_dir}")
        print(f"  ├── words/ (저신뢰 단어)")
        print(f"  ├── low_conf/ (저신뢰 블록)")
        print(f"  └── images/ (이미지 블록)")

if __name__ == "__main__":
    main()
