#!/usr/bin/env python3
"""
① 레이아웃 감지 + 원본 OCR
입력: pdf_in/*.pdf
처리: DotsOCR로 레이아웃(bbox) + 원본 OCR 추출 → Picture는 원본에서 크롭
출력: step1_primary/page_XXX.json (ocr_raw 포함), crops/*.png (원본 이미지 크롭)
Note: step1b에서 전처리 OCR(ocr_preprocessed)을 추가로 수행
"""

import os
import json
import requests
import base64
import cv2
import numpy as np
import time
from datetime import datetime
import fitz  # PyMuPDF
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import torch
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import re

# 설정
DOTSOCR_URL = "http://localhost:8000/v1/chat/completions"
BASE_DIR = Path(__file__).parent.parent
PDF_DIR = BASE_DIR / "pdf_in"
OCR_DIR = BASE_DIR / "ocr_results"
DEBUG_DIR = OCR_DIR / "debug"

def pdf_to_images(pdf_path, dpi=300):
    """PDF를 페이지별 이미지로 변환"""
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(dpi/72, dpi/72)  # 1차 OCR용 기본 DPI
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # OpenCV로 변환
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        images.append((page_num + 1, img))
        
        # 디버그 이미지 저장 (선택사항)
        # debug_path = DEBUG_DIR / f"page_{page_num+1:03d}.png"
        # cv2.imwrite(str(debug_path), img)
    
    doc.close()
    return images

def load_embedding_models(load_clip=False):
    """임베딩 모델 로드"""
    try:
        # BGE-m3-ko 텍스트 임베딩 모델
        bge_model = SentenceTransformer('dragonkue/bge-m3-ko')
        print("✅ BGE-m3-ko 모델 로드 완료")
        
        # CLIP 멀티모달 모델 (이미지 + 텍스트) - 선택적
        clip_model = None
        if load_clip:
            try:
                clip_model = SentenceTransformer('clip-ViT-B-32')
                print("✅ CLIP 모델 로드 완료")
            except:
                print("⚠️ CLIP 모델 로드 실패, 이미지 임베딩 비활성화")
        else:
            print("⏭️ CLIP 모델 건너뜀 (빠른 테스트 모드)")
        
        return bge_model, clip_model
    except Exception as e:
        print(f"❌ 임베딩 모델 로드 실패: {e}")
        return None, None

def generate_text_embedding(text, bge_model):
    """텍스트 임베딩 생성"""
    if bge_model is None or not text:
        return None
    
    try:
        embedding = bge_model.encode(text)
        return embedding.tolist()  # numpy array를 list로 변환
    except Exception as e:
        print(f"    ⚠️ 텍스트 임베딩 생성 실패: {e}")
        return None

def generate_image_embedding(image_path, clip_model):
    """이미지 임베딩 생성"""
    if clip_model is None or not os.path.exists(image_path):
        return None
    
    try:
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # CLIP으로 이미지 임베딩
        embedding = clip_model.encode(image)
        return embedding.tolist()  # numpy array를 list로 변환
    except Exception as e:
        print(f"    ⚠️ 이미지 임베딩 생성 실패: {e}")
        return None

def generate_image_caption(image_path):
    """이미지 캡션 생성 (간단한 휴리스틱 방법)"""
    try:
        # 이미지 분석을 통한 간단한 캡션 생성
        image = cv2.imread(image_path)
        if image is None:
            return "이미지를 로드할 수 없습니다"
        
        height, width = image.shape[:2]
        
        # 이미지 크기 기반 설명
        size_desc = f"{width}x{height} 픽셀 이미지"
        
        # 색상 분석
        mean_color = np.mean(image, axis=(0, 1))
        if np.mean(mean_color) > 200:
            color_desc = "밝은 이미지"
        elif np.mean(mean_color) < 50:
            color_desc = "어두운 이미지"
        else:
            color_desc = "중간 톤 이미지"
        
        # 간단한 캡션 생성
        caption = f"문서 내 {color_desc} ({size_desc})"
        
        return caption
    except Exception as e:
        print(f"    ⚠️ 이미지 캡션 생성 실패: {e}")
        return "이미지 캡션을 생성할 수 없습니다"

def correct_incomplete_text(blocks):
    """불완전한 텍스트를 다른 텍스트와 비교하여 교정"""
    if len(blocks) < 2:
        return blocks
    
    # 불완전한 텍스트 패턴 감지
    incomplete_patterns = [
        r'.*도\s*clo$',  # "보장도 clo"
        r'.*지\s*clo$',  # "보장지 clo" 
        r'.*는\s*clo$',  # "보장는 clo"
        r'.*이\s*clo$',  # "보장이 clo"
        r'.*을\s*clo$',  # "보장을 clo"
        r'.*를\s*clo$',  # "보장를 clo"
        r'.*에\s*clo$',  # "보장에 clo"
        r'.*에서\s*clo$', # "보장에서 clo"
        r'^clo.*',       # "clo"로 시작하는 경우
        r'.*clo$',       # "clo"로 끝나는 경우
    ]
    
    corrected_blocks = []
    
    for i, block in enumerate(blocks):
        text = block.get('text', '').strip()
        is_incomplete = any(re.search(pattern, text, re.IGNORECASE) for pattern in incomplete_patterns)
        
        if is_incomplete and len(text) > 3:  # 최소 3글자 이상
            print(f"    🔍 불완전한 텍스트 감지: '{text}'")
            
            # 다른 텍스트들과 유사도 비교
            best_match = None
            best_similarity = 0.0
            
            for j, other_block in enumerate(blocks):
                if i != j:
                    other_text = other_block.get('text', '').strip()
                    if len(other_text) > len(text):  # 더 긴 텍스트와 비교
                        # 유사도 계산 (공통 부분 기반)
                        similarity = SequenceMatcher(None, text, other_text).ratio()
                        
                        # 부분 문자열 매칭도 확인
                        if text in other_text:
                            similarity += 0.3  # 부분 매칭 보너스
                        
                        if similarity > best_similarity and similarity > 0.6:
                            best_similarity = similarity
                            best_match = other_text
            
            if best_match:
                # 가장 유사한 텍스트에서 해당 부분 추출
                corrected_text = extract_similar_text(text, best_match)
                if corrected_text != text:
                    print(f"    ✅ 텍스트 교정: '{text}' → '{corrected_text}' (유사도: {best_similarity:.2f})")
                    block['text'] = corrected_text
                else:
                    print(f"    ⚠️ 교정 실패: '{text}'")
            else:
                print(f"    ⚠️ 유사한 텍스트 없음: '{text}'")
        
        corrected_blocks.append(block)
    
    return corrected_blocks

def extract_similar_text(incomplete_text, full_text):
    """불완전한 텍스트를 전체 텍스트에서 유사한 부분으로 교정"""
    # "보장도 clo" → "보장지도" 같은 패턴 교정
    if incomplete_text.endswith(' clo') or incomplete_text.endswith('clo'):
        # "clo" 부분을 제거하고 유사한 어미 찾기
        base_text = incomplete_text.replace(' clo', '').replace('clo', '').strip()
        
        # 전체 텍스트에서 해당 부분 찾기
        if base_text in full_text:
            # 해당 부분 주변의 텍스트 추출
            start_idx = full_text.find(base_text)
            if start_idx >= 0:
                # 앞뒤로 몇 글자 더 포함해서 추출
                start = max(0, start_idx - 2)
                end = min(len(full_text), start_idx + len(base_text) + 5)
                extracted = full_text[start:end]
                
                # 불필요한 문자 제거
                extracted = re.sub(r'[^\w\s가-힣]', '', extracted).strip()
                return extracted
    
    # 기본적으로 전체 텍스트에서 가장 유사한 부분 반환
    return incomplete_text

def apply_preprocessing(image):
    """전처리 파이프라인 적용 (업스케일링 + 타일링 포함)"""
    # OpenCV 이미지를 PIL로 변환
    if len(image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(image)
    
    # 1. Gaussian Blur (5×5)
    gaussian_kernel = ImageFilter.GaussianBlur(radius=2.5)  # 5×5 커널에 해당
    blurred = pil_image.filter(gaussian_kernel)
    
    # 2. Unsharp Mask (r=1.0~1.5, p=120~200)
    unsharp_filter = ImageFilter.UnsharpMask(radius=1.2, percent=160, threshold=3)
    sharpened = blurred.filter(unsharp_filter)
    
    # 3. 업스케일링 (2.8배)
    original_size = sharpened.size
    scale_factor = 2.8
    new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
    upscaled = sharpened.resize(new_size, Image.Resampling.LANCZOS)
    
    # PIL을 OpenCV로 다시 변환
    upscaled_cv = cv2.cvtColor(np.array(upscaled), cv2.COLOR_RGB2BGR)
    
    return upscaled_cv

def extract_layout_only(original_image, timeout=300):
    """
    레이아웃만 추출 (텍스트 OCR 없음)
    1. 원본 이미지로 레이아웃(bbox)만 추출
    2. Picture/Image 블록 감지
    → 텍스트는 step1b에서 별도로 처리
    """
    print(f"    📐 레이아웃 감지 중 (텍스트 없음)...")
    
    # 원본 이미지로 레이아웃만 추출
    h_orig, w_orig = original_image.shape[:2]
    
    print(f"       원본 이미지: {w_orig}×{h_orig}")
    
    # DotsOCR로 레이아웃만 추출 (텍스트 없음)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    _, buffer = cv2.imencode('.jpg', original_image, encode_param)
    image_data = base64.b64encode(buffer).decode('utf-8')
    
    try:
        response = requests.post(DOTSOCR_URL, json={
            "model": "model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
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
        }, timeout=timeout)
        
        if response.status_code != 200:
            print(f"       ❌ 레이아웃 추출 실패: HTTP {response.status_code}")
            return None
            
        result = response.json()
        layout_content = result['choices'][0]['message']['content']
        print(f"       ✅ 레이아웃 추출 완료: {len(layout_content)}자")
        
        # JSON 파싱
        import json
        layout_data = json.loads(layout_content)
        
        # blocks 추출
        if isinstance(layout_data, dict):
            blocks = layout_data.get('blocks') or layout_data.get('layout') or layout_data.get('elements', [])
            if not blocks and len(layout_data) == 1:
                blocks = list(layout_data.values())[0]
        else:
            blocks = layout_data
            
        if not blocks:
            print(f"       ❌ 레이아웃 블록을 찾을 수 없습니다")
            return None
        
        print(f"    📐 {len(blocks)}개 레이아웃 블록 감지 (원본 OCR 포함)")
        
        # 레이아웃 + 원본 OCR 텍스트 반환
        for i, block in enumerate(blocks):
            category = block.get('category', 'Unknown')
            bbox = block.get('bbox', [])
            text = block.get('text', '')
            
            if len(bbox) != 4:
                continue
            
            # 원본 OCR 텍스트 저장
            block['ocr_raw'] = text  # step1에서 추출한 텍스트 = 원본 OCR
            block['ocr_pending'] = True  # step1b에서 전처리 OCR 추가 필요
            
            if category.lower() in ['picture', 'image']:
                print(f"       [{i+1:03d}] {category:15s} - Picture (원본 크롭 예정)")
            else:
                text_preview = text[:30] if text else '(텍스트 없음)'
                print(f"       [{i+1:03d}] {category:15s} - 원본 OCR: '{text_preview}...'")
        
        print(f"    ✅ 레이아웃 + 원본 OCR 완료: {len(blocks)}개 블록")
        return {"blocks": blocks}
        
    except Exception as e:
        print(f"       ❌ 레이아웃 추출 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def ocr_tile_direct(tile_image, timeout=20):
    """타일 이미지를 전처리 없이 직접 OCR (무한재귀 방지)"""
    try:
        # 이미지 인코딩 (전처리 없이!)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', tile_image, encode_param)
        image_data = base64.b64encode(buffer).decode('utf-8')
        
        # DotsOCR 요청
        response = requests.post(DOTSOCR_URL, json={
            "model": "model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
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
        }, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            ocr_content = result['choices'][0]['message']['content']
            
            # JSON 파싱
            import json
            ocr_data = json.loads(ocr_content)
            
            # blocks 추출
            if isinstance(ocr_data, dict):
                blocks = ocr_data.get('blocks') or ocr_data.get('layout') or ocr_data.get('elements', [])
                if not blocks and len(ocr_data) == 1:
                    blocks = list(ocr_data.values())[0]
            else:
                blocks = ocr_data
                
            if isinstance(blocks, list) and blocks:
                return {"blocks": blocks}
            else:
                return {"blocks": []}
        else:
            return None
            
    except Exception as e:
        print(f"        ⚠️ 타일 OCR 오류: {e}")
        return None

def apply_tiling_ocr(image, tile_size=(1024, 1024), overlap=100):
    """이미지를 타일로 나누어 OCR 수행 후 결과 병합"""
    h, w = image.shape[:2]
    
    # 타일 크기 조정 (이미지보다 클 경우 전체 이미지 사용)
    tile_h, tile_w = min(tile_size[0], h), min(tile_size[1], w)
    
    # 오버랩을 고려한 스텝 크기
    step_h = tile_h - overlap
    step_w = tile_w - overlap
    
    all_blocks = []
    
    print(f"    🧩 타일링 OCR: {tile_h}×{tile_w}, 오버랩 {overlap}px")
    
    # 타일별로 처리
    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            # 타일 경계 조정
            y_end = min(y + tile_h, h)
            x_end = min(x + tile_w, w)
            
            # 실제 타일 크기
            actual_h = y_end - y
            actual_w = x_end - x
            
            # 너무 작은 타일은 건너뛰기
            if actual_h < 100 or actual_w < 100:
                continue
                
            tile = image[y:y_end, x:x_end]
            
            print(f"      📦 타일 ({x},{y})-({x_end},{y_end}) 크기: {actual_w}×{actual_h}")
            
            # 타일별 OCR 수행 (이미 전처리된 이미지이므로 직접 OCR)
            tile_result = ocr_tile_direct(tile, timeout=20)
            
            if tile_result and 'blocks' in tile_result:
                # 좌표를 전체 이미지 기준으로 변환
                for block in tile_result['blocks']:
                    if 'bbox' in block:
                        bbox = block['bbox']
                        # 타일 내 좌표를 전체 이미지 좌표로 변환
                        adjusted_bbox = [
                            bbox[0] + x,  # x1
                            bbox[1] + y,  # y1
                            bbox[2] + x,  # x2
                            bbox[3] + y   # y2
                        ]
                        block['bbox'] = adjusted_bbox
                        
                all_blocks.extend(tile_result['blocks'])
    
    # 중복 제거 및 블록 병합 (오버랩 영역에서 발생할 수 있음)
    filtered_blocks = remove_overlapping_blocks(all_blocks)
    
    print(f"    ✅ 타일링 완료: {len(all_blocks)}개 → {len(filtered_blocks)}개 블록")
    
    # 전처리된 이미지도 함께 반환
    return {"blocks": filtered_blocks, "processed_image": image}

def remove_overlapping_blocks(blocks, iou_threshold=0.3):
    """중복되는 블록 제거 및 타일 경계에서 잘린 블록 병합"""
    if len(blocks) <= 1:
        return blocks
    
    # IoU 계산 함수
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    # 블록을 y좌표 기준으로 정렬 (위에서 아래로)
    sorted_blocks = sorted(blocks, key=lambda b: (b.get('bbox', [0,0,0,0])[1], b.get('bbox', [0,0,0,0])[0]))
    
    filtered = []
    
    for i, block in enumerate(sorted_blocks):
        is_duplicate = False
        merged = False
        
        for j, existing_block in enumerate(filtered):
            if 'bbox' not in block or 'bbox' not in existing_block:
                continue
                
            iou = calculate_iou(block['bbox'], existing_block['bbox'])
            
            # 1. 중복 블록 제거 (IoU가 높으면 중복)
            if iou > iou_threshold:
                # 텍스트가 더 긴 것을 선택
                text1 = block.get('text', '')
                text2 = existing_block.get('text', '')
                
                if len(text1) > len(text2):
                    filtered[j] = block
                is_duplicate = True
                break
            
            # 2. 타일 경계에서 잘린 블록 병합 (IoU는 낮지만 인접한 경우)
            if should_merge_adjacent_blocks(block, existing_block):
                # 블록 병합
                merged_block = merge_two_blocks(block, existing_block)
                filtered[j] = merged_block
                merged = True
                
                # 병합 방향 표시
                y1 = (existing_block['bbox'][1] + existing_block['bbox'][3]) / 2
                y2 = (block['bbox'][1] + block['bbox'][3]) / 2
                merge_type = "좌우" if abs(y1 - y2) < 30 else "상하"
                
                print(f"      🔗 블록 병합({merge_type}): '{existing_block.get('text', '')[:15]}' + '{block.get('text', '')[:15]}' → '{merged_block.get('text', '')[:30]}'")
                break
        
        if not is_duplicate and not merged:
            filtered.append(block)
    
    return filtered

def should_merge_adjacent_blocks(block1, block2, max_gap=50, y_threshold=20):
    """두 블록이 타일 경계에서 잘린 것인지 확인 (x축 및 y축)"""
    bbox1 = block1.get('bbox', [])
    bbox2 = block2.get('bbox', [])
    
    if len(bbox1) != 4 or len(bbox2) != 4:
        return False
    
    # 같은 카테고리인지 확인
    if block1.get('category') != block2.get('category'):
        return False
    
    # 케이스 1: x축으로 잘린 경우 (같은 줄에서 좌우로 인접)
    y1_center = (bbox1[1] + bbox1[3]) / 2
    y2_center = (bbox2[1] + bbox2[3]) / 2
    
    if abs(y1_center - y2_center) <= y_threshold:
        # x좌표가 인접한지 확인
        x_gap_right = bbox2[0] - bbox1[2]  # block2가 block1의 오른쪽
        x_gap_left = bbox1[0] - bbox2[2]   # block1이 block2의 오른쪽
        
        if -max_gap < x_gap_right < max_gap or -max_gap < x_gap_left < max_gap:
            return True
    
    # 케이스 2: y축으로 잘린 경우 (위아래로 인접, x축 범위 겹침)
    x1_center = (bbox1[0] + bbox1[2]) / 2
    x2_center = (bbox2[0] + bbox2[2]) / 2
    
    # x축 범위가 겹치는지 확인 (세로로 이어진 블록)
    x_overlap = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    x_min_width = min(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0])
    
    # x축이 50% 이상 겹치면 같은 칼럼으로 간주
    if x_overlap > x_min_width * 0.5:
        # y좌표가 인접한지 확인
        y_gap_bottom = bbox2[1] - bbox1[3]  # block2가 block1의 아래
        y_gap_top = bbox1[1] - bbox2[3]     # block1이 block2의 아래
        
        if -max_gap < y_gap_bottom < max_gap or -max_gap < y_gap_top < max_gap:
            return True
    
    return False

def merge_two_blocks(block1, block2):
    """두 블록을 하나로 병합 (x축 또는 y축)"""
    bbox1 = block1.get('bbox', [])
    bbox2 = block2.get('bbox', [])
    
    # 새로운 bbox 계산 (두 블록을 포함하는 최소 bbox)
    merged_bbox = [
        min(bbox1[0], bbox2[0]),  # x1
        min(bbox1[1], bbox2[1]),  # y1
        max(bbox1[2], bbox2[2]),  # x2
        max(bbox1[3], bbox2[3])   # y2
    ]
    
    # 텍스트 병합 방향 판단
    y1_center = (bbox1[1] + bbox1[3]) / 2
    y2_center = (bbox2[1] + bbox2[3]) / 2
    
    # y축 차이가 작으면 x축(좌우) 병합, 크면 y축(위아래) 병합
    if abs(y1_center - y2_center) < 30:
        # x축 병합 (좌우로 이어진 경우)
        if bbox1[0] < bbox2[0]:
            merged_text = block1.get('text', '') + block2.get('text', '')
        else:
            merged_text = block2.get('text', '') + block1.get('text', '')
    else:
        # y축 병합 (위아래로 이어진 경우) - 줄바꿈 추가
        if bbox1[1] < bbox2[1]:  # block1이 위
            merged_text = block1.get('text', '') + '\n' + block2.get('text', '')
        else:  # block2가 위
            merged_text = block2.get('text', '') + '\n' + block1.get('text', '')
    
    # 병합된 블록 생성
    merged_block = block1.copy()
    merged_block['bbox'] = merged_bbox
    merged_block['text'] = merged_text
    
    # confidence는 평균값 사용
    conf1 = block1.get('confidence', 0)
    conf2 = block2.get('confidence', 0)
    merged_block['confidence'] = (conf1 + conf2) / 2
    
    return merged_block

def ocr_with_dotsocr(image, timeout=300, max_retries=3):
    """DotsOCR로 레이아웃만 추출 (텍스트 없음, step1b에서 처리)"""
    for attempt in range(max_retries):
        try:
            print(f"    🔄 레이아웃 감지 중... (시도 {attempt + 1}/{max_retries})")
            
            # 레이아웃만 추출 (텍스트 없음)
            result = extract_layout_only(image, timeout=timeout)
            if result:
                # 원본 이미지를 반환 (Picture 크롭용)
                result['original_image'] = image
                return result
            else:
                print(f"    ❌ 레이아웃 추출 실패")
                if attempt < max_retries - 1:
                    print(f"    🔄 {2}초 후 재시도...")
                    time.sleep(2)
                    continue
                return None
                
        except requests.exceptions.Timeout:
            print(f"    ⏰ OCR 타임아웃 ({timeout}초)")
            if attempt < max_retries - 1:
                print(f"    🔄 {2}초 후 재시도...")
                time.sleep(2)
                continue
            return None
        except requests.exceptions.ConnectionError:
            print(f"    🔌 서버 연결 실패")
            if attempt < max_retries - 1:
                print(f"    🔄 {2}초 후 재시도...")
                time.sleep(2)
                continue
            return None
        except Exception as e:
            print(f"    ❌ OCR 실패: {e}")
            if attempt < max_retries - 1:
                print(f"    🔄 {2}초 후 재시도...")
                time.sleep(2)
                continue
            return None
    
    return None

def crop_picture_blocks(image, picture_blocks, doc_name, page_num, bge_model=None, clip_model=None):
    """Picture/Image 블록들을 크롭하여 저장하고 임베딩 생성"""
    cropped_images = []
    
    # 크롭 이미지 저장 디렉토리 (세션 디렉토리 내)
    crop_dir = OCR_DIR / doc_name / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    for i, block in enumerate(picture_blocks):
        bbox = block['bbox']
        block_id = f"picture_{i:03d}"
        
        # 블록 크롭
        x1, y1, x2, y2 = bbox
        cropped_image = image[y1:y2, x1:x2]
        
        # 파일명 생성
        filename = f"{doc_name}_page{page_num:03d}_{block_id}.png"
        file_path = crop_dir / filename
        
        # 크롭된 이미지 저장
        cv2.imwrite(str(file_path), cropped_image)
        
        # 이미지 캡션 생성
        caption = generate_image_caption(str(file_path))
        
        # 임베딩 생성
        text_embedding = None
        image_embedding = None
        
        if bge_model:
            text_embedding = generate_text_embedding(caption, bge_model)
        
        if clip_model:
            image_embedding = generate_image_embedding(str(file_path), clip_model)
        
        # 메타데이터 생성
        cropped_info = {
            "block_id": block_id,
            "filename": filename,
            "file_path": str(file_path),
            "relative_path": f"ocr_results/{doc_name}/crops/{filename}",
            "bbox": bbox,
            "category": block['category'],
            "confidence": block.get('confidence', 0.9),
            "size": [x2-x1, y2-y1],
            "page_number": page_num,
            "document_name": doc_name,
            "caption": caption,
            "embeddings": {
                "text_embedding": text_embedding,
                "image_embedding": image_embedding,
                "embedding_models": {
                    "text_model": "dragonkue/bge-m3-ko" if bge_model else None,
                    "image_model": "clip-ViT-B-32" if clip_model else None
                }
            },
            "search_ready": bool(text_embedding or image_embedding)
        }
        
        cropped_images.append(cropped_info)
        print(f"      📦 {block_id}: {block['category']} → {filename} (임베딩: {'✅' if cropped_info['search_ready'] else '❌'})")
    
    return cropped_images

def draw_bbox_debug(image, blocks):
    """bbox 디버그 이미지 생성 (demo 스타일)"""
    debug_img = image.copy()
    
    # demo와 동일한 카테고리별 색상 정의 (BGR 형식)
    colors = {
        'Page-header': (0, 0, 255),     # 빨간색
        'Page-footer': (0, 0, 255),     # 빨간색  
        'Section-header': (0, 255, 0),  # 초록색
        'Table': (255, 0, 0),           # 파란색
        'Text': (0, 255, 255),          # 노란색
        'Title': (255, 0, 255),         # 마젠타
        'List-item': (255, 255, 0),     # 시안
        'Picture': (128, 0, 128),       # 보라색
        'Formula': (0, 128, 255),       # 주황색
        'Caption': (128, 128, 128),     # 회색
        'Footnote': (64, 64, 64)        # 어두운 회색
    }
    
    for i, block in enumerate(blocks):
        bbox = block.get('bbox', [0, 0, 100, 100])
        category = block.get('category', 'Text')
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            color = colors.get(category, (128, 128, 128))
            
            # demo 스타일: 반투명 배경 채우기
            overlay = debug_img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.3, debug_img, 0.7, 0, debug_img)
            
            # 테두리 그리기
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            
            # 블록 번호와 카테고리 표시 (demo 스타일)
            label = f"{i:03d}:{category}"
            
            # 텍스트 배경 그리기
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(debug_img, (x1, y1-25), (x1+text_width+10, y1), color, -1)
            
            # 텍스트 그리기
            cv2.putText(debug_img, label, (x1+5, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return debug_img

def process_pdf(pdf_path, bge_model=None, clip_model=None, max_pages=None):
    """PDF 파일 처리 (원래 데모 방식 + Picture 크롭 + 임베딩)"""
    print(f"📄 처리 중: {pdf_path.name}")
    
    # 세션 타임스탬프 생성 (한 번만)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_dir = OCR_DIR / pdf_path.stem / session_timestamp
    step1_dir = doc_dir / "step1_primary"
    step1_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📁 세션 디렉토리: {doc_dir}")
    
    # 세션 정보를 파일로 저장 (다른 단계에서 사용)
    session_info = {
        "document_name": pdf_path.stem,
        "session_timestamp": session_timestamp,
        "session_dir": str(doc_dir),
        "embedding_models": {
            "text_model": "dragonkue/bge-m3-ko" if bge_model else None,
            "image_model": "clip-ViT-B-32" if clip_model else None
        }
    }
    session_file = BASE_DIR / "current_session.json"
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session_info, f, ensure_ascii=False, indent=2)
    
    # PDF를 이미지로 변환
    print("  🖼️ PDF → 이미지 변환...")
    images = pdf_to_images(pdf_path)
    print(f"  ✅ {len(images)}페이지 변환 완료")
    
    # 페이지 제한 적용
    if max_pages and max_pages > 0:
        images = images[:max_pages]
        print(f"  📄 처음 {len(images)}페이지만 처리합니다")
    else:
        print(f"  📄 전체 {len(images)}페이지를 처리합니다")
    
    # 각 페이지 OCR
    for page_num, image in images:
        print(f"  📝 페이지 {page_num} OCR 중... ({image.shape})")
        
        # 서버 과부하 방지를 위한 대기
        if page_num > 1:
            time.sleep(2)
        
        ocr_result = ocr_with_dotsocr(image)
        if ocr_result:
            # 레이아웃 결과 파싱
            blocks = ocr_result.get("blocks", [])
            original_image = ocr_result.get("original_image", image)  # 원본 이미지
            
            # Picture/Image 블록 찾기 및 크롭
            picture_blocks = []
            for block in blocks:
                category = block.get('category', '').lower()
                if category in ['picture', 'image']:
                    picture_blocks.append(block)
            
            # Picture 블록 크롭 및 저장 (원본 이미지 사용, 임베딩 포함)
            cropped_images = []
            if picture_blocks:
                print(f"    🖼️ Picture/Image 블록 {len(picture_blocks)}개 크롭 중 (원본 이미지 사용)...")
                # 원본 이미지에서 크롭 (전처리 없음)
                cropped_images = crop_picture_blocks(original_image, picture_blocks, pdf_path.stem, page_num, bge_model, clip_model)
            
            # 메타데이터 추가
            result_data = {
                "document_name": pdf_path.stem,
                "page_number": page_num,
                "timestamp": session_timestamp,
                "original_image_size": image.shape[:2],
                "pipeline_stage": "레이아웃 감지 + 원본 OCR",
                "preprocessing_applied": {
                    "layout_detection": "원본 이미지 사용",
                    "picture_crop": "원본 이미지 사용 (전처리 없음)",
                    "ocr_raw": "원본 이미지 OCR (DotsOCR)",
                    "ocr_preprocessed": "step1b에서 전처리 OCR 추가 예정"
                },
                "blocks": blocks,
                "picture_blocks": cropped_images,
                "total_picture_blocks": len(cropped_images),
                "embedding_models": {
                    "text_model": "dragonkue/bge-m3-ko" if bge_model else None,
                    "image_model": "clip-ViT-B-32" if clip_model else None
                }
            }
            
            # 결과 저장 (세션 디렉토리 사용)
            output_path = step1_dir / f"page_{page_num:03d}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            # bbox 디버그 이미지 생성 및 저장 (원본 이미지 사용)
            debug_image = draw_bbox_debug(original_image, result_data['blocks'])
            debug_path = step1_dir / f"page_{page_num:03d}_bbox_debug.png"
            cv2.imwrite(str(debug_path), debug_image)
            
            print(f"    ✅ {len(result_data['blocks'])}개 레이아웃 블록 감지, {len(cropped_images)}개 Picture 크롭 (원본) → JSON + 디버그 이미지 저장")
        else:
            print(f"    ❌ OCR 실패")
    
    # 모든 페이지 결과를 하나의 combined.json으로 통합
    print("  🔗 페이지 결과 통합 중...")
    combined_data = {
        "metadata": {
            "document_name": pdf_path.stem,
            "total_pages": len(images),
            "processed_pages": len(images),
            "preprocessing": {
                "gaussian_blur": {"kernel_size": 5, "sigma": 1.0},
                "unsharp_mask": {"radius": 1.2, "percent": 160, "threshold": 3},
                "upscale": {"scale_factor": 1.0, "method": "removed"}
            },
            "text_model": "dragonkue/bge-m3-ko" if bge_model else None,
            "image_model": "clip-ViT-B-32" if clip_model else None
        },
        "pages": []
    }
    
    for page_num, _ in images:
        page_file = step1_dir / f"page_{page_num:03d}.json"
        if page_file.exists():
            with open(page_file, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
                combined_data["pages"].append(page_data)
    
    # combined.json 저장
    combined_file = step1_dir / "combined.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ 통합 파일 생성: {combined_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='통합 OCR 파이프라인')
    parser.add_argument('--max-pages', type=int, default=None, 
                        help='처리할 최대 페이지 수 (기본값: 전체)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Step 1: 레이아웃 감지 (텍스트 없음, Picture는 원본 크롭)")
    print("=" * 60)
    
    # 디렉토리 확인
    OCR_DIR.mkdir(exist_ok=True)
    DEBUG_DIR.mkdir(exist_ok=True)
    
    # PDF 파일 찾기
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❌ PDF 파일을 찾을 수 없습니다.")
        print(f"   경로: {PDF_DIR}")
        return
    
    print(f"📁 발견된 PDF: {len(pdf_files)}개")
    
    # DotsOCR 서버 확인
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print("✅ DotsOCR 서버 연결됨")
    except:
        print("❌ DotsOCR 서버에 연결할 수 없습니다.")
        print("   서버를 먼저 시작해주세요.")
        return
    
    # 임베딩 모델 로드 (선택적)
    import os
    enable_embeddings = os.getenv('ENABLE_EMBEDDINGS', 'false').lower() == 'true'
    
    if enable_embeddings:
        print("\n🤖 임베딩 모델 로드 중...")
        try:
            bge_model, clip_model = load_embedding_models(load_clip=False)  # CLIP은 나중에 필요할 때
        except Exception as e:
            print(f"   ❌ 임베딩 모델 로드 실패: {e}")
            bge_model, clip_model = None, None
    else:
        print("\n⏭️ 임베딩 기능 비활성화 (빠른 테스트 모드)")
        print("   (임베딩을 활성화하려면: export ENABLE_EMBEDDINGS=true)")
        bge_model, clip_model = None, None
    
    if bge_model is None:
        print("⚠️ BGE 모델 로드 실패, 텍스트 임베딩 비활성화")
    
    if clip_model is None:
        print("⚠️ CLIP 모델 로드 실패, 이미지 임베딩 비활성화")
    
    # PDF 처리
    for pdf_path in pdf_files:
        try:
            process_pdf(pdf_path, bge_model, clip_model, max_pages=args.max_pages)
        except Exception as e:
            print(f"❌ {pdf_path.name} 처리 실패: {e}")
    
    print("\n🎉 레이아웃 감지 완료!")
    print(f"📁 결과: {OCR_DIR}")
    if args.max_pages:
        print(f"📄 처리된 페이지: 최대 {args.max_pages}페이지")
    else:
        print(f"📄 처리된 페이지: 전체 페이지")
    print(f"📐 레이아웃: 원본 이미지로 감지 (텍스트 없음)")
    print(f"🖼️ Picture 크롭: 원본 이미지에서 크롭 (전처리 없음)")
    print(f"🤖 임베딩 모델: BGE-m3-ko {'✅' if bge_model else '❌'}, CLIP {'✅' if clip_model else '❌'}")
    print(f"⏭️ 다음 단계: step1b에서 텍스트 블록 전처리 전/후 2회 OCR 수행")

def search_images_by_query(query, json_data, clip_model):
    """이미지 검색 예시 함수"""
    if clip_model is None:
        print("❌ CLIP 모델이 로드되지 않았습니다.")
        return []
    
    try:
        # 질문을 임베딩으로 변환
        query_embedding = clip_model.encode(query)
        
        results = []
        for page_data in json_data.get('pages', []):
            for picture_block in page_data.get('picture_blocks', []):
                if picture_block.get('search_ready') and picture_block['embeddings'].get('image_embedding'):
                    # 유사도 계산
                    image_embedding = np.array(picture_block['embeddings']['image_embedding'])
                    similarity = np.dot(query_embedding, image_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(image_embedding))
                    
                    results.append({
                        'image_path': picture_block['file_path'],
                        'relative_path': picture_block['relative_path'],
                        'caption': picture_block['caption'],
                        'similarity': float(similarity),
                        'page_number': picture_block['page_number'],
                        'bbox': picture_block['bbox']
                    })
        
        # 유사도 순으로 정렬
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results
    
    except Exception as e:
        print(f"❌ 이미지 검색 실패: {e}")
        return []

# 사용 예시
def example_image_search():
    """이미지 검색 사용 예시"""
    print("\n🔍 이미지 검색 예시:")
    print("질문: 'AA사진이 뭐야?'")
    print("→ 해당 이미지 파일과 유사도 점수를 반환")
    print("→ 이미지 파일 경로, 캡션, 페이지 정보 포함")

if __name__ == "__main__":
    main()
