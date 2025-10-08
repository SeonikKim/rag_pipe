#!/usr/bin/env python3
"""
③ 2차 OCR (전처리 고정 옵션 적용)
입력: crops/low_conf/*.png
전처리: Gaussian 5×5 → Unsharp r=1.0~1.5, p=120~200 → 업스케일 ×2.8
처리: 전처리 이미지를 다시 OCR → 원래 블록에 값 치환
출력: ocr_refined/page_XXX_refined.json, ocr_refined/refined_combined.json
"""

import os
import json
import cv2
import numpy as np
import requests
import base64
from pathlib import Path
from datetime import datetime
import glob

# 설정
BASE_DIR = Path(__file__).parent.parent
OCR_DIR = BASE_DIR / "ocr_results"
# 세션별 디렉토리는 런타임에 설정
CURRENT_SESSION_DIR = None
DOTSOCR_URL = "http://localhost:8000/v1/chat/completions"

# 전처리 고정 옵션
GAUSSIAN_KERNEL = (5, 5)
UNSHARP_RADIUS = 1.3
UNSHARP_PERCENT = 150
UPSCALE_FACTOR = 2.8

def preprocess_image_fixed(image):
    """고정 전처리 옵션 적용"""
    # 1. Gaussian Blur (5×5)
    blurred = cv2.GaussianBlur(image, GAUSSIAN_KERNEL, 0)
    
    # 2. Unsharp Mask (r=1.3, p=150)
    unsharp = cv2.addWeighted(image, 1 + UNSHARP_PERCENT/100, blurred, -UNSHARP_PERCENT/100, 0)
    
    # 3. 업스케일 (×2.8)
    height, width = unsharp.shape[:2]
    new_size = (int(width * UPSCALE_FACTOR), int(height * UPSCALE_FACTOR))
    upscaled = cv2.resize(unsharp, new_size, interpolation=cv2.INTER_CUBIC)
    
    return upscaled

def ocr_with_dotsocr(image, confidence_threshold=0.8):
    """DotsOCR로 2차 OCR 수행"""
    try:
        # 이미지를 base64로 인코딩
        _, buffer = cv2.imencode('.png', image)
        image_data = base64.b64encode(buffer).decode('utf-8')
        
        response = requests.post(DOTSOCR_URL, json={
            "model": "model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                    {"type": "text", "text": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""}
                ]
            }],
            "max_tokens": 4096,
            "temperature": 0.0
        }, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return None
            
    except Exception as e:
        print(f"❌ OCR 실패: {e}")
        return None

def parse_crop_filename(filename):
    """크롭 파일명에서 정보 추출"""
    # 예: doc1_page001_block005.png
    parts = filename.stem.split('_')
    if len(parts) >= 3:
        doc_name = parts[0]
        page_str = parts[1].replace('page', '')
        block_str = parts[2].replace('block', '')
        
        try:
            page_num = int(page_str)
            block_idx = int(block_str)
            return doc_name, page_num, block_idx
        except ValueError:
            pass
    
    return None, None, None

def load_original_ocr_results():
    """1차 OCR 결과를 딕셔너리로 로드"""
    results = {}
    ocr_files = list(OCR_DIR.glob("page_*.json"))
    
    for ocr_file in ocr_files:
        with open(ocr_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            page_num = data['page_number']
            results[page_num] = data
    
    return results

def process_crops():
    """크롭된 이미지들을 2차 OCR 처리"""
    # 현재 세션 정보 읽기
    session_file = BASE_DIR / "current_session.json"
    if not session_file.exists():
        print("❌ 현재 세션 정보가 없습니다.")
        return {}
    
    with open(session_file, 'r', encoding='utf-8') as f:
        session_info = json.load(f)
    
    global CURRENT_SESSION_DIR
    CURRENT_SESSION_DIR = Path(session_info['session_dir'])
    
    # 크롭 디렉토리에서 이미지 찾기
    crops_dir = CURRENT_SESSION_DIR / "step2_crops" / "low_conf"
    crop_files = list(crops_dir.glob("*.png"))
    if not crop_files:
        print("❌ 크롭된 이미지를 찾을 수 없습니다.")
        return {}
    
    print(f"📦 크롭된 이미지: {len(crop_files)}개")
    
    # 1차 OCR 결과 로드
    original_results = load_original_ocr_results()
    
    # 2차 OCR 결과 저장용
    refined_results = {}
    
    for i, crop_file in enumerate(crop_files, 1):
        print(f"[{i}/{len(crop_files)}] {crop_file.name}")
        
        # 파일명에서 정보 추출
        doc_name, page_num, block_idx = parse_crop_filename(crop_file)
        if not all([doc_name, page_num is not None, block_idx is not None]):
            print(f"    ⚠️ 파일명 파싱 실패")
            continue
        
        # 메타데이터 로드
        meta_file = crop_file.with_suffix('.json')
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
        else:
            meta_data = {}
        
        # 이미지 로드
        image = cv2.imread(str(crop_file))
        if image is None:
            print(f"    ❌ 이미지 로드 실패")
            continue
        
        # 전처리 적용
        preprocessed = preprocess_image_fixed(image)
        
        # 2차 OCR 수행
        ocr_result = ocr_with_dotsocr(preprocessed, confidence_threshold=0.8)
        
        if ocr_result:
            # 결과 저장
            if page_num not in refined_results:
                # 원본 페이지 데이터 복사
                if page_num in original_results:
                    refined_results[page_num] = original_results[page_num].copy()
                    refined_results[page_num]['blocks'] = [block.copy() for block in original_results[page_num]['blocks']]
                else:
                    refined_results[page_num] = {
                        "page_number": page_num,
                        "blocks": []
                    }
            
            # 해당 블록의 텍스트 업데이트
            if block_idx < len(refined_results[page_num]['blocks']):
                old_text = refined_results[page_num]['blocks'][block_idx].get('text', '')
                refined_results[page_num]['blocks'][block_idx]['text'] = ocr_result
                refined_results[page_num]['blocks'][block_idx]['ocr_method'] = '2nd_pass_fixed_preprocess'
                refined_results[page_num]['blocks'][block_idx]['preprocessing'] = {
                    "gaussian": f"{GAUSSIAN_KERNEL}",
                    "unsharp": f"r={UNSHARP_RADIUS}, p={UNSHARP_PERCENT}",
                    "upscale": f"×{UPSCALE_FACTOR}"
                }
                
                print(f"    ✅ {len(old_text)} → {len(ocr_result)}자")
            else:
                print(f"    ⚠️ 블록 인덱스 오류: {block_idx}")
        else:
            print(f"    ❌ OCR 실패")
    
    return refined_results

def save_refined_results(refined_results):
    """정제된 결과 저장"""
    REFINED_DIR.mkdir(exist_ok=True)
    
    # 페이지별 저장
    for page_num, page_data in refined_results.items():
        # 메타데이터 업데이트
        page_data['pipeline_stage'] = '2차 OCR (고정 전처리)'
        page_data['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 페이지별 파일 저장 (세션 디렉토리 사용)
        step3_dir = CURRENT_SESSION_DIR / "step3_refined"
        step3_dir.mkdir(parents=True, exist_ok=True)
        output_path = step3_dir / f"page_{page_num:03d}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)
    
    # 통합 파일 저장
    combined_data = {
        "timestamp": session_info['session_timestamp'],
        "pipeline_stage": "2차 OCR 통합",
        "preprocessing": {
            "gaussian": f"{GAUSSIAN_KERNEL}",
            "unsharp": f"r={UNSHARP_RADIUS}, p={UNSHARP_PERCENT}",
            "upscale": f"×{UPSCALE_FACTOR}"
        },
        "pages": list(refined_results.values())
    }
    
    combined_path = step3_dir / "refined_combined.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    return combined_path

def main():
    print("=" * 60)
    print("🔧 2차 OCR (고정 전처리 옵션)")
    print("=" * 60)
    print(f"📋 전처리 설정:")
    print(f"   • Gaussian: {GAUSSIAN_KERNEL}")
    print(f"   • Unsharp: r={UNSHARP_RADIUS}, p={UNSHARP_PERCENT}")
    print(f"   • 업스케일: ×{UPSCALE_FACTOR}")
    
    # DotsOCR 서버 확인
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print("✅ DotsOCR 서버 연결됨")
    except:
        print("❌ DotsOCR 서버에 연결할 수 없습니다.")
        return
    
    # 크롭 이미지 처리
    print("\n🔍 크롭 이미지 2차 OCR 처리...")
    refined_results = process_crops()
    
    if not refined_results:
        print("❌ 처리할 결과가 없습니다.")
        return
    
    # 결과 저장
    print(f"\n💾 결과 저장...")
    combined_path = save_refined_results(refined_results)
    
    print(f"\n🎉 2차 OCR 완료!")
    print(f"📄 처리된 페이지: {len(refined_results)}개")
    print(f"📁 결과: {REFINED_DIR}")
    print(f"📄 통합 파일: {combined_path}")

if __name__ == "__main__":
    main()
