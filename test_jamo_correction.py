#!/usr/bin/env python3
"""
자모 복원 기능 테스트 스크립트
"""

import sys
from pathlib import Path

# 스크립트 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from step5_kiwi_correction import (
    init_kiwi,
    decompose_hangul,
    compose_hangul,
    try_compose_jamo_sequence,
    fix_mixed_english_in_korean,
    fix_jamo_separation,
    correct_context_based_words,
    correct_section_text,
)

def test_decompose_compose():
    """한글 분해/조합 테스트"""
    print("=" * 60)
    print("1. 한글 분해/조합 테스트")
    print("=" * 60)
    
    test_chars = ['한', '글', '무', '릎']
    for char in test_chars:
        decomposed = decompose_hangul(char)
        print(f"'{char}' -> {decomposed}")
        if decomposed:
            recomposed = compose_hangul(*decomposed)
            print(f"  재조합: {recomposed} {'✅' if recomposed == char else '❌'}")
    print()

def test_jamo_composition():
    """자모 조합 테스트"""
    print("=" * 60)
    print("2. 자모 조합 테스트")
    print("=" * 60)
    
    test_cases = [
        'ㅎㅏㄴㄱㅡㄹ',  # 한글
        'ㅁㅜㄹㅡㅂ',    # 무릎
        'ㄱㅗㅇㅂㅜ',    # 공부
    ]
    
    for jamo_text in test_cases:
        print(f"\n입력: '{jamo_text}'")
        candidates = try_compose_jamo_sequence(jamo_text)
        if candidates:
            print(f"후보: {candidates[:5]}")
        else:
            print("후보 없음")
    print()

def test_mixed_english():
    """영어 혼합 패턴 테스트"""
    print("=" * 60)
    print("3. 영어 혼합 패턴 테스트")
    print("=" * 60)
    
    test_cases = [
        '무les를',
        '무abar',
        '무bar',
        '재merc게',
        '재mer게',
        '측정measurement',
        '시험test',
        '학생student',
        '교육education과정',
        '연구research원',
        # '윗몸말아올리기',  # 실제 운동 이름이므로 테스트 제외
        # '팔굽펴기',  # 확인 필요
    ]
    
    for text in test_cases:
        print(f"\n'{text}'")
        fixed = fix_mixed_english_in_korean(text, debug=True)
        print(f"  결과: '{fixed}' {'✅' if text != fixed else 'ℹ️'}")
    print()

def test_jamo_separation():
    """자모 분리 수정 테스트"""
    print("=" * 60)
    print("4. 자모 분리 수정 테스트")
    print("=" * 60)
    
    test_cases = [
        'ㄱ ㅏ ㄴ ㅏ 다',
        '한글 ㅎㅏㄴㄱㅡㄹ 테스트',
        'ㅁ ㅜ 릎',
    ]
    
    for text in test_cases:
        fixed = fix_jamo_separation(text)
        print(f"'{text}'")
        print(f"  -> '{fixed}' {'✅' if text != fixed else 'ℹ️'}")
    print()

def test_context_based_correction():
    """문맥 기반 단어 교정 테스트"""
    print("=" * 60)
    print("5. 문맥 기반 단어 교정 테스트 (Kiwi 형태소 분석)")
    print("=" * 60)
    
    test_cases = [
        ('무름을 구부리고 펴는 운동', '무릎 문맥'),
        ('뼈의 무름 증상이 나타난다', '무름 문맥'),
        ('체육 시험 무름 측정 90도', '무릎 문맥'),
        ('골다공증으로 인한 무름 현상', '무름 문맥'),
    ]
    
    for text, expected_context in test_cases:
        print(f"{expected_context:12s} | '{text}'")
        corrected = correct_context_based_words(text, debug=True)
        changed = '✅' if text != corrected else 'ℹ️'
        print(f"             → '{corrected}' {changed}")
        print()

def test_full_correction():
    """종합 교정 테스트"""
    print("=" * 60)
    print("6. 종합 교정 테스트")
    print("=" * 60)
    
    test_cases = [
        '무les를 구부리고   90 ° 회전',
        'ㅎㅏㄴㄱㅡㄹ  교육 과정',
        'ㄱ ㅏ ㄴ ㅏ 다',
        '체육 시험  ㅁㅜㄹㅡㅂ',
        '무름 구부리기 체육 시험',
    ]
    
    for text in test_cases:
        corrected = correct_section_text(text)
        print(f"원본: '{text}'")
        print(f"교정: '{corrected}'")
        print(f"상태: {'✅ 변경됨' if text != corrected else 'ℹ️ 변경 없음'}")
        print()

def main():
    print("\n🧪 자모 복원 기능 테스트\n")
    
    # Kiwi 초기화
    if init_kiwi():
        print("✅ Kiwi 로드 성공\n")
    else:
        print("⚠️ Kiwi 없이 테스트 진행 (기본 기능만)\n")
    
    # 테스트 실행
    test_decompose_compose()
    test_jamo_composition()
    test_mixed_english()
    test_jamo_separation()
    test_context_based_correction()
    test_full_correction()
    
    print("=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()

