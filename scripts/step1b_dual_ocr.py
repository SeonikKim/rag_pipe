#!/usr/bin/env python3
"""
①-b 이중 OCR (전처리 전/후 2회 OCR)
입력: step1_primary/page_XXX.json (레이아웃만 있는 파일)
처리: 텍스트 블록만 전처리 전/후 2회 OCR → 두 값 모두 저장
출력: step1b_dual/page_XXX.json (ocr_raw, ocr_preprocessed 두 필드 모두 포함)
"""

import os
import json
import re
import requests
import base64
import cv2
import numpy as np
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

# 설정
DOTSOCR_URL = "http://localhost:8000/v1/chat/completions"
BASE_DIR = Path(__file__).parent.parent
OCR_DIR = BASE_DIR / "ocr_results"
CURRENT_SESSION_DIR = None

# 원 안의 한글 문자 → 원 안의 숫자 매핑
CIRCLED_HANGUL_TO_NUMBER = {
    '㉮': '①',  # 가 → 1
    '㉯': '②',  # 나 → 2
    '㉰': '③',  # 다 → 3
    '㉱': '④',  # 라 → 4
    '㉲': '⑤',  # 마 → 5
    '㉳': '⑥',  # 바 → 6
    '㉴': '⑦',  # 사 → 7
    '㉵': '⑧',  # 아 → 8
    '㉶': '⑨',  # 자 → 9
    '㉷': '⑩',  # 차 → 10
    '㉸': '⑪',  # 카 → 11
    '㉹': '⑫',  # 타 → 12
    '㉺': '⑬',  # 파 → 13
    '㉻': '⑭',  # 하 → 14
}

def normalize_circled_characters(text):
    """원 안의 한글 문자를 원 안의 숫자로 정규화"""
    if not text:
        return text

    for hangul, number in CIRCLED_HANGUL_TO_NUMBER.items():
        text = text.replace(hangul, number)
    
    return text


def evaluate_ocr_consistency(raw_text, preprocessed_text):
    """원본 OCR과 전처리 OCR의 유사도를 계산해 불일치 여부 판단"""

    if not raw_text or not preprocessed_text:
        return {
            "similarity": 0.0,
            "overlap": 0.0,
            "mismatch": False,
        }

    def normalize(text):
        # 영문/숫자/한글 중심으로 비교 (특수문자 제거)
        return re.sub(r"[^0-9A-Za-z가-힣]", "", text)

    raw_norm = normalize(raw_text)
    pre_norm = normalize(preprocessed_text)

    if not raw_norm or not pre_norm:
        return {
            "similarity": 0.0,
            "overlap": 0.0,
            "mismatch": False,
        }

    similarity = SequenceMatcher(None, raw_norm, pre_norm).ratio()

    raw_set = set(raw_norm)
    pre_set = set(pre_norm)
    overlap = len(raw_set & pre_set) / max(len(raw_set | pre_set), 1)

    mismatch = similarity < 0.25 and overlap < 0.35 and len(pre_norm) > 6 and len(raw_norm) > 6

    return {
        "similarity": similarity,
        "overlap": overlap,
        "mismatch": mismatch,
    }

def apply_preprocessing(image):
    """전처리 파이프라인 적용 (텍스트 OCR용)"""
    # OpenCV 이미지를 PIL로 변환
    if len(image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(image)
    
    # 1. Gaussian Blur (5×5)
    gaussian_kernel = ImageFilter.GaussianBlur(radius=2.5)
    blurred = pil_image.filter(gaussian_kernel)
    
    # 2. Unsharp Mask (r=1.2, p=160)
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

def ocr_single_block(image, bbox, use_preprocessing=False, timeout=300):
    """단일 블록 OCR 수행"""
    try:
        # bbox로 블록 크롭
        x1, y1, x2, y2 = bbox
        cropped = image[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return None
        
        # 전처리 적용 여부
        if use_preprocessing:
            cropped = apply_preprocessing(cropped)
        
        # 이미지 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', cropped, encode_param)
        image_data = base64.b64encode(buffer).decode('utf-8')
        
        # DotsOCR 요청
        response = requests.post(DOTSOCR_URL, json={
            "model": "model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    {"type": "text", "text": """Extract all text from this image in reading order. Output ONLY the extracted text, without any formatting, explanation, or translation."""}
                ]
            }],
            "max_tokens": 2048,
            "temperature": 0.0
        }, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            text = result['choices'][0]['message']['content'].strip()
            return text
        else:
            return None
            
    except Exception as e:
        print(f"      ❌ OCR 오류: {e}")
        return None

def process_page(page_data, original_image):
    """페이지별 이중 OCR 처리"""
    page_num = page_data['page_number']
    blocks = page_data.get('blocks', [])
    
    print(f"  📄 페이지 {page_num}: {len(blocks)}개 블록 처리 중...")
    
    text_blocks_count = 0
    picture_blocks_count = 0
    
    for i, block in enumerate(blocks):
        category = block.get('category', '').lower()
        bbox = block.get('bbox', [])
        
        if len(bbox) != 4:
            continue
        
        # Picture/Image 블록은 건너뜀 (이미 step1에서 처리됨)
        if category in ['picture', 'image']:
            picture_blocks_count += 1
            print(f"    [{i+1:03d}] {category:15s} - 건너뜀 (step1에서 처리됨)")
            continue
        
        # 텍스트 블록만 전처리 OCR 수행 (원본은 step1에서 이미 추출됨)
        print(f"    [{i+1:03d}] {category:15s} - 전처리 OCR 수행 중...")
        
        # 1. 원본 OCR은 step1에서 추출된 것 사용
        ocr_raw = block.get('ocr_raw', '')
        print(f"          1️⃣ 원본 OCR (step1): '{(ocr_raw[:30] + '...') if len(ocr_raw) > 30 else ocr_raw}'")
        
        # 2. 전처리 후 OCR만 추가로 수행
        print(f"          2️⃣ 전처리 이미지 OCR...")
        ocr_preprocessed = ocr_single_block(original_image, bbox, use_preprocessing=True)
        
        # 원 안의 한글 문자 정규화 적용
        ocr_raw = normalize_circled_characters(ocr_raw) if ocr_raw else ""
        ocr_preprocessed = normalize_circled_characters(ocr_preprocessed) if ocr_preprocessed else ""
        
        # 원본/전처리 결과 비교 및 상태 기록
        status = {
            "state": "empty" if not ocr_preprocessed else "unknown",
            "similarity": 0.0,
            "overlap": 0.0,
        }

        if ocr_preprocessed:
            consistency = evaluate_ocr_consistency(ocr_raw, ocr_preprocessed)
            status.update({
                "similarity": round(consistency["similarity"], 3),
                "overlap": round(consistency["overlap"], 3),
            })

            if consistency["mismatch"] and ocr_raw:
                status["state"] = "discarded_mismatch"
                status["discarded_preview"] = (ocr_preprocessed[:40] + '...') if len(ocr_preprocessed) > 40 else ocr_preprocessed
                print(
                    f"          ⚠️ 전처리 OCR 불일치 → 폐기 (유사도 {consistency['similarity']:.2f}, 중복비율 {consistency['overlap']:.2f})"
                )
                ocr_preprocessed = ""
            else:
                status["state"] = "kept"
        else:
            status["state"] = "empty"

        # 결과 저장
        block['ocr_raw'] = ocr_raw
        block['ocr_preprocessed'] = ocr_preprocessed
        block['ocr_preprocessed_status'] = status
        block['ocr_pending'] = False

        # 미리보기
        raw_preview = (ocr_raw[:30] + '...') if ocr_raw and len(ocr_raw) > 30 else (ocr_raw or "(없음)")
        prep_preview = (ocr_preprocessed[:30] + '...') if ocr_preprocessed and len(ocr_preprocessed) > 30 else (ocr_preprocessed or "(없음)")

        print(f"          ✅ 원본: {raw_preview}")
        print(f"          ✅ 전처리: {prep_preview}")
        
        text_blocks_count += 1
        
        # API 과부하 방지
        time.sleep(0.5)
    
    print(f"    ✅ 텍스트 블록 {text_blocks_count}개 이중 OCR 완료, Picture {picture_blocks_count}개 건너뜀")
    
    return page_data

def main():
    print("=" * 60)
    print("🔄 Step 1b: 이중 OCR (전처리 전/후 2회)")
    print("=" * 60)
    
    # 현재 세션 정보 읽기
    session_file = BASE_DIR / "current_session.json"
    if not session_file.exists():
        print("❌ 현재 세션 정보가 없습니다. step1을 먼저 실행하세요.")
        return
    
    with open(session_file, 'r', encoding='utf-8') as f:
        session_info = json.load(f)
    
    global CURRENT_SESSION_DIR
    CURRENT_SESSION_DIR = Path(session_info['session_dir'])
    
    # step1 결과 파일 로드
    step1_dir = CURRENT_SESSION_DIR / "step1_primary"
    page_files = sorted(step1_dir.glob("page_*.json"))
    
    if not page_files:
        print(f"❌ step1 결과를 찾을 수 없습니다: {step1_dir}")
        return
    
    print(f"📂 발견된 페이지: {len(page_files)}개")
    
    # DotsOCR 서버 확인
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print("✅ DotsOCR 서버 연결됨")
    except:
        print("❌ DotsOCR 서버에 연결할 수 없습니다.")
        return
    
    # 출력 디렉토리 생성
    step1b_dir = CURRENT_SESSION_DIR / "step1b_dual"
    step1b_dir.mkdir(parents=True, exist_ok=True)
    
    # PDF 원본 이미지 로드를 위한 정보
    import fitz
    pdf_name = session_info.get('document_name', 'unknown')
    pdf_path = BASE_DIR / "pdf_in" / f"{pdf_name}.pdf"
    
    if not pdf_path.exists():
        print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return
    
    # PDF 열기
    doc = fitz.open(pdf_path)
    
    # 페이지별 처리
    for page_file in page_files:
        with open(page_file, 'r', encoding='utf-8') as f:
            page_data = json.load(f)
        
        page_num = page_data['page_number']
        
        # PDF에서 해당 페이지 이미지 추출
        page = doc.load_page(page_num - 1)  # 0-based index
        mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # OpenCV로 변환
        nparr = np.frombuffer(img_data, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 이중 OCR 처리
        processed_page = process_page(page_data, original_image)
        
        # 메타데이터 업데이트
        processed_page['pipeline_stage'] = '이중 OCR (전처리 전/후)'
        processed_page['dual_ocr_applied'] = True
        processed_page['preprocessing_info'] = {
            'gaussian_blur': '5x5',
            'unsharp_mask': 'r=1.2, p=160',
            'upscale': 'x2.8'
        }
        
        # 결과 저장
        output_path = step1b_dir / f"page_{page_num:03d}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_page, f, ensure_ascii=False, indent=2)
        
        print(f"  💾 저장: {output_path.name}")
    
    doc.close()
    
    # 통합 파일 생성
    print("\n🔗 페이지 결과 통합 중...")
    combined_data = {
        "metadata": {
            "document_name": pdf_name,
            "total_pages": len(page_files),
            "pipeline_stage": "이중 OCR",
            "timestamp": datetime.now().isoformat(),
            "ocr_methods": {
                "ocr_raw": "원본 이미지 (전처리 없음)",
                "ocr_preprocessed": "전처리 이미지 (Gaussian + Unsharp + Upscale)"
            }
        },
        "pages": []
    }
    
    for page_file in sorted(step1b_dir.glob("page_*.json")):
        with open(page_file, 'r', encoding='utf-8') as f:
            page_data = json.load(f)
            combined_data["pages"].append(page_data)
    
    combined_file = step1b_dir / "combined.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 통합 파일 생성: {combined_file}")
    
    print(f"\n🎉 이중 OCR 완료!")
    print(f"📄 처리된 페이지: {len(page_files)}개")
    print(f"📁 결과: {step1b_dir}")
    print(f"📊 각 텍스트 블록에 ocr_raw, ocr_preprocessed 두 값 저장됨")
    print(f"⏭️ 다음 단계: step6에서 LLM이 두 값 비교하여 최적 값 선택")

if __name__ == "__main__":
    main()

