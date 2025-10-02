#!/usr/bin/env python3
"""
① 1차 OCR
입력: pdf_in/*.pdf
처리: DotsOCR로 전 페이지 OCR
출력: ocr_results/page_XXX.json, ocr_results/debug/page_XXX.png
"""

import os
import json
import requests
import base64
import cv2
import numpy as np
from datetime import datetime
import fitz  # PyMuPDF
import sys
from pathlib import Path

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
        mat = fitz.Matrix(dpi/72, dpi/72)  # DPI 설정
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # OpenCV로 변환
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        images.append((page_num + 1, img))
        
        # 디버그 이미지 저장
        debug_path = DEBUG_DIR / f"page_{page_num+1:03d}.png"
        cv2.imwrite(str(debug_path), img)
    
    doc.close()
    return images

def ocr_with_dotsocr(image, timeout=60):
    """DotsOCR로 OCR 수행"""
    try:
        # 이미지를 base64로 인코딩
        _, buffer = cv2.imencode('.png', image)
        image_data = base64.b64encode(buffer).decode('utf-8')
        
        response = requests.post(DOTSOCR_URL, json={
            "model": "dots",
            "messages": [{"role": "user", "content": image_data}],
            "confidence_threshold": 0.1  # 1차에서는 낮게 설정
        }, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"❌ DotsOCR 오류: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ OCR 실패: {e}")
        return None

def parse_ocr_result(ocr_text):
    """OCR 결과를 구조화된 데이터로 파싱"""
    # DotsOCR 결과가 JSON 형태라고 가정
    try:
        if isinstance(ocr_text, str):
            # 문자열인 경우 JSON 파싱 시도
            import re
            json_match = re.search(r'\{.*\}', ocr_text, re.DOTALL)
            if json_match:
                ocr_data = json.loads(json_match.group())
            else:
                # JSON이 아닌 경우 간단한 구조로 변환
                ocr_data = {
                    "blocks": [{"text": ocr_text, "bbox": [0, 0, 100, 100], "confidence": 0.5}]
                }
        else:
            ocr_data = ocr_text
            
        return ocr_data
        
    except Exception as e:
        print(f"⚠️ OCR 결과 파싱 실패: {e}")
        return {"blocks": []}

def process_pdf(pdf_path):
    """PDF 파일 처리"""
    print(f"📄 처리 중: {pdf_path.name}")
    
    # PDF를 이미지로 변환
    print("  🖼️ PDF → 이미지 변환...")
    images = pdf_to_images(pdf_path)
    print(f"  ✅ {len(images)}페이지 변환 완료")
    
    # 각 페이지 OCR
    for page_num, image in images:
        print(f"  📝 페이지 {page_num} OCR 중...")
        
        ocr_result = ocr_with_dotsocr(image)
        if ocr_result:
            # OCR 결과 파싱
            parsed_result = parse_ocr_result(ocr_result)
            
            # 메타데이터 추가
            result_data = {
                "document_name": pdf_path.stem,
                "page_number": page_num,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "image_size": image.shape[:2],
                "pipeline_stage": "1차 OCR",
                "blocks": parsed_result.get("blocks", [])
            }
            
            # 결과 저장
            output_path = OCR_DIR / f"page_{page_num:03d}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            print(f"    ✅ {len(result_data['blocks'])}개 블록 추출")
        else:
            print(f"    ❌ OCR 실패")

def main():
    print("=" * 60)
    print("🚀 1차 OCR 파이프라인")
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
    
    # PDF 처리
    for pdf_path in pdf_files:
        try:
            process_pdf(pdf_path)
        except Exception as e:
            print(f"❌ {pdf_path.name} 처리 실패: {e}")
    
    print("\n🎉 1차 OCR 완료!")
    print(f"📁 결과: {OCR_DIR}")
    print(f"🐛 디버그: {DEBUG_DIR}")

if __name__ == "__main__":
    main()
