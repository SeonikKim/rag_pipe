# OCR 파이프라인

한국어 문서 OCR을 위한 완전 자동화 파이프라인입니다.

## 🎯 목적

PDF/이미지 문서를 RAG 친화적인 JSON으로 변환:
- 1차 OCR → 저신뢰 블록 2차 OCR → 섹션 병합 → 형태소/문맥 교정 → 최종 JSON

## 📁 디렉토리 구조

```
ocr_pipeline/
├── pdf_in/                    # 입력 PDF/이미지
├── ocr_results/              # ① 1차 OCR 결과
│   └── debug/               # 디버그 이미지
├── crops/                   # ② 저신뢰 블록 크롭
│   ├── low_conf/           # 텍스트 블록 (2차 OCR 대상)
│   └── images/             # 비텍스트 블록
├── ocr_refined/             # ③ 2차 OCR 결과
├── layout_combined/         # ④ 섹션 병합 결과
├── corrections/             # ⑤⑥ 교정 결과
├── final_outputs/           # ⑦ 최종 JSON (RAG 투입용)
└── scripts/                 # 파이프라인 스크립트들
```

## 🚀 7단계 파이프라인

1. **1차 OCR**: DotsOCR로 전 페이지 OCR
2. **저신뢰 블록 크롭**: confidence < 0.80 블록들을 크롭
3. **2차 OCR**: 고정 전처리 옵션으로 재OCR
   - Gaussian(5×5) → Unsharp(r=1.3, p=150) → 업스케일(×2.8)
4. **섹션 병합**: BeautifulSoup4로 DOM 구조 생성 후 섹션별 병합
5. **Kiwi 교정**: 형태소 분석 기반 1차 교정
6. **LLM 교정**: EEVE 모델로 문맥 교정
7. **최종 JSON**: RAG 친화적 JSON 생성

## 📋 사전 요구사항

### 서버
- **DotsOCR 서버**: `http://localhost:8000`
- **LLM 서버** (선택): `http://localhost:8003` (EEVE 모델)

### Python 라이브러리
```bash
pip install opencv-python requests PyMuPDF beautifulsoup4 kiwipiepy
```

## 🎮 사용법

### 전체 파이프라인 실행
```bash
cd scripts
python3 run_pipeline.py
```

### 특정 단계만 실행
```bash
python3 run_pipeline.py --start 3 --end 5  # 3~5단계만
```

### 단계 목록 확인
```bash
python3 run_pipeline.py --list
```

### 개별 단계 실행
```bash
python3 step1_primary_ocr.py      # 1차 OCR
python3 step2_crop_low_confidence.py  # 저신뢰 블록 크롭
python3 step3_secondary_ocr.py    # 2차 OCR
python3 step4_layout_merge.py     # 섹션 병합
python3 step5_kiwi_correction.py  # Kiwi 교정
python3 step6_llm_correction.py   # LLM 교정
python3 step7_final_json.py       # 최종 JSON
```

## 📄 최종 JSON 형식

```json
{
  "document_id": "doc1",
  "sections": [
    {
      "id": "section_001",
      "title": "무릎 운동 가이드",
      "category": "운동",
      "pages": [1, 2, 3],
      "text_combined": "무릎을 90도로 굽혀 세우고..."
    }
  ],
  "metadata": {
    "source_pdf": "pdf_in/doc1.pdf",
    "pipeline": [
      "DotsOCR 1차",
      "2차 OCR(고정 옵션: Gaussian 5×5 → Unsharp r=1.3, p=150 → 업스케일 ×2.8)",
      "형태소 교정(Kiwi)",
      "문맥 교정(LLM)",
      "섹션 병합(BeautifulSoup4)"
    ],
    "timestamp": "2025-10-02T17:20:00"
  }
}
```

## ⚙️ 설정

### 2차 OCR 전처리 (고정 옵션)
- **Gaussian Blur**: 5×5 커널
- **Unsharp Mask**: 반경 1.3, 강도 150%
- **업스케일**: 2.8배

### 신뢰도 임계치
- **1차 OCR**: 0.1 (모든 텍스트 추출)
- **2차 OCR**: 0.8 (고품질만)
- **저신뢰 판단**: 0.8 미만

## 🔧 트러블슈팅

### DotsOCR 서버 연결 실패
```bash
# 서버 상태 확인
curl http://localhost:8000/health
```

### LLM 서버 연결 실패
- LLM 교정은 선택사항입니다
- 서버가 없어도 Kiwi 교정까지는 정상 작동

### 메모리 부족
- 대용량 PDF는 페이지별로 분할 처리
- 2차 OCR 시 배치 크기 조정

## 📊 성능

- **처리 속도**: 페이지당 약 30초 (2차 OCR 포함)
- **정확도**: 한국어 문서 기준 95%+ (2차 OCR + 교정 적용 시)
- **지원 형식**: PDF, PNG, JPG, JPEG

## 🤝 기여

이슈 및 PR 환영합니다!

## 📝 라이선스

MIT License