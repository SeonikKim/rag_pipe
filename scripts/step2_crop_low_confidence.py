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
CROPS_DIR = BASE_DIR / "crops"
LOW_CONF_DIR = CROPS_DIR / "low_conf"
IMAGES_DIR = CROPS_DIR / "images"

CONFIDENCE_THRESHOLD = 0.80  # 임계치
MARGIN = 15  # 크롭 시 여백

def load_ocr_results():
    """1차 OCR 결과 로드"""
    ocr_files = list(OCR_DIR.glob("page_*.json"))
    results = []
    
    for ocr_file in sorted(ocr_files):
        with open(ocr_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)
    
    return results

def is_low_confidence_block(block, threshold=CONFIDENCE_THRESHOLD):
    """저신뢰 블록 판단"""
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
    
    low_conf_count = 0
    blocks = page_data.get('blocks', [])
    
    for i, block in enumerate(blocks):
        if 'bbox' not in block:
            continue
        
        # 저신뢰 블록 판단
        if is_low_confidence_block(block):
            # 블록 크롭
            cropped = crop_block(image, block['bbox'])
            if cropped is None or cropped.size == 0:
                continue
            
            # 카테고리별 저장
            category = categorize_block(block)
            if category == 'low_conf':
                output_dir = LOW_CONF_DIR
            else:
                output_dir = IMAGES_DIR
            
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
                "reason": "low_confidence" if category == 'low_conf' else "non_text"
            }
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)
            
            low_conf_count += 1
            print(f"    📦 Block {i:03d}: {category} ({block.get('confidence', 0.0):.2f})")
    
    return low_conf_count

def main():
    print("=" * 60)
    print("🔍 저신뢰 블록 선별 및 크롭")
    print("=" * 60)
    
    # 디렉토리 생성
    LOW_CONF_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1차 OCR 결과 로드
    print("📂 1차 OCR 결과 로드...")
    ocr_results = load_ocr_results()
    
    if not ocr_results:
        print("❌ 1차 OCR 결과를 찾을 수 없습니다.")
        return
    
    print(f"✅ {len(ocr_results)}페이지 로드됨")
    print(f"🎯 신뢰도 임계치: {CONFIDENCE_THRESHOLD}")
    
    # 페이지별 처리
    total_crops = 0
    for page_data in ocr_results:
        page_num = page_data['page_number']
        print(f"\n📄 페이지 {page_num} 처리 중...")
        
        crops = process_page(page_data)
        total_crops += crops
        
        if crops > 0:
            print(f"    ✅ {crops}개 블록 크롭됨")
        else:
            print(f"    ℹ️ 저신뢰 블록 없음")
    
    print(f"\n🎉 크롭 완료!")
    print(f"📦 총 {total_crops}개 블록 크롭됨")
    print(f"📁 텍스트 블록: {LOW_CONF_DIR}")
    print(f"📁 이미지 블록: {IMAGES_DIR}")

if __name__ == "__main__":
    main()
