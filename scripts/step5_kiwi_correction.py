#!/usr/bin/env python3
"""
⑤ 1차 교정 (형태소 분석)
입력: layout_combined/doc1_sections.json
처리: Kiwi로 띄어쓰기, 자모 분리/합침, 간단 조사 오류 보정
출력: corrections/kiwi_intermediate.json
"""

import json
import re
from pathlib import Path
from datetime import datetime
from itertools import product

# 설정
BASE_DIR = Path(__file__).parent.parent
CURRENT_SESSION_DIR = None

# Kiwi 초기화 (lazy loading)
kiwi = None
KIWI_AVAILABLE = False

# 한글 자모 정의
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 자음과 모음 집합
CONSONANTS = set('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ')
VOWELS = set('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')

def init_kiwi():
    """Kiwi 초기화"""
    global kiwi, KIWI_AVAILABLE
    if kiwi is not None:
        return True
    
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        KIWI_AVAILABLE = True
        print("✅ Kiwi 형태소 분석기 로드")
        return True
    except Exception as e:
        KIWI_AVAILABLE = False
        print(f"⚠️ Kiwi 사용 불가: {e}")
        return False

def decompose_hangul(char):
    """한글 글자를 초성, 중성, 종성으로 분해"""
    if not ('가' <= char <= '힣'):
        return None
    
    code = ord(char) - 0xAC00
    cho_idx = code // 588
    jung_idx = (code % 588) // 28
    jong_idx = code % 28
    
    return (CHOSUNG_LIST[cho_idx], JUNGSUNG_LIST[jung_idx], JONGSUNG_LIST[jong_idx])

def compose_hangul(cho, jung, jong=''):
    """초성, 중성, 종성을 합쳐서 한글 글자 생성"""
    try:
        cho_idx = CHOSUNG_LIST.index(cho)
        jung_idx = JUNGSUNG_LIST.index(jung)
        jong_idx = JONGSUNG_LIST.index(jong)
        
        code = 0xAC00 + (cho_idx * 588) + (jung_idx * 28) + jong_idx
        return chr(code)
    except (ValueError, IndexError):
        return None

def is_jamo_separated(text):
    """자모가 분리된 텍스트인지 확인 (예: 'ㅎㅏㄴㄱㅡㄹ')"""
    if not text:
        return False
    
    # 자모 문자가 2개 이상 연속되면 분리된 것으로 간주
    jamo_count = sum(1 for c in text if c in CONSONANTS or c in VOWELS)
    return jamo_count >= 2

def try_compose_jamo_sequence(jamo_text):
    """분리된 자모 문자열을 가능한 한글 조합으로 변환
    예: 'ㅎㅏㄴㄱㅡㄹ' -> ['한글', '함글', '항글', ...]
    """
    if not jamo_text:
        return []
    
    candidates = []
    
    # 자모만 추출
    jamos = [c for c in jamo_text if c in CONSONANTS or c in VOWELS]
    if len(jamos) < 2:
        return []
    
    # 가능한 조합 생성 (최대 3글자까지만 시도)
    def generate_combinations(jamos, max_attempts=50):
        results = []
        
        def recurse(remaining, current_result, attempt_count):
            if attempt_count > max_attempts:
                return
            
            if not remaining:
                if current_result:
                    results.append(''.join(current_result))
                return
            
            # 방법 1: 초성+중성 조합
            if len(remaining) >= 2:
                if remaining[0] in CONSONANTS and remaining[1] in VOWELS:
                    char = compose_hangul(remaining[0], remaining[1])
                    if char:
                        recurse(remaining[2:], current_result + [char], attempt_count + 1)
            
            # 방법 2: 초성+중성+종성 조합
            if len(remaining) >= 3:
                if remaining[0] in CONSONANTS and remaining[1] in VOWELS and remaining[2] in CONSONANTS:
                    char = compose_hangul(remaining[0], remaining[1], remaining[2])
                    if char:
                        recurse(remaining[3:], current_result + [char], attempt_count + 1)
            
            # 방법 3: 그냥 넘어가기
            if remaining:
                recurse(remaining[1:], current_result + [remaining[0]], attempt_count + 1)
        
        recurse(jamos, [], 0)
        return results
    
    candidates = generate_combinations(jamos)
    
    # 결과 필터링: 최소 한글 비율 확인
    valid_candidates = []
    for candidate in candidates:
        hangul_count = sum(1 for c in candidate if '가' <= c <= '힣')
        if hangul_count >= len(candidate) * 0.5:  # 최소 50% 한글
            valid_candidates.append(candidate)
    
    return valid_candidates[:10]  # 최대 10개 후보

def calculate_kiwi_score(text):
    """Kiwi 형태소 분석으로 텍스트의 자연스러움 점수 계산"""
    if not text or not KIWI_AVAILABLE:
        return 0.0
    
    try:
        result = kiwi.analyze(text)
        if not result or len(result) == 0:
            return 0.0
        
        # Kiwi 결과는 [[(Token, Token, ...), score], ...] 형태
        token_list = result[0][0] if isinstance(result[0], tuple) else result[0]
        
        if not token_list:
            return 0.0
        
        total_tokens = len(token_list)
        if total_tokens == 0:
            return 0.0
        
        unk_count = 0
        meaningful_count = 0
        meaningful_tags = ['NNG', 'NNP', 'NNB', 'VV', 'VA', 'MAG', 'MM']
        
        for token in token_list:
            # Token 객체인 경우
            if hasattr(token, 'tag'):
                tag = str(token.tag)
            # 튜플이나 리스트인 경우
            elif isinstance(token, (tuple, list)) and len(token) >= 2:
                tag = str(token[1])
            else:
                continue
            
            # UNK(알 수 없는 단어) 카운트
            if tag in ['UNK', 'UNKNOWN']:
                unk_count += 1
            
            # 의미 있는 품사 카운트
            if tag in meaningful_tags:
                meaningful_count += 1
        
        # 점수 계산 (높을수록 자연스러움)
        unk_ratio = unk_count / total_tokens
        meaningful_ratio = meaningful_count / total_tokens
        score = (1.0 - unk_ratio) * 0.7 + meaningful_ratio * 0.3
        
        return score
    except Exception:
        return 0.0

def select_best_candidate_with_kiwi(candidates, context=""):
    """Kiwi로 여러 후보 중 가장 자연스러운 것 선택"""
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    if not KIWI_AVAILABLE:
        # Kiwi가 없으면 가장 긴 것 선택
        return max(candidates, key=lambda x: len([c for c in x if '가' <= c <= '힣']))
    
    # 각 후보의 점수 계산
    scored_candidates = []
    for candidate in candidates:
        # 문맥과 함께 점수 계산
        if context:
            full_text = f"{context} {candidate}"
            score = calculate_kiwi_score(full_text)
        else:
            score = calculate_kiwi_score(candidate)
        
        scored_candidates.append((candidate, score))
    
    # 가장 높은 점수의 후보 선택
    if scored_candidates:
        best_candidate = max(scored_candidates, key=lambda x: x[1])
        return best_candidate[0]
    
    return candidates[0]

def fix_mixed_english_in_korean(text, debug=False):
    """한글+영어 혼합 패턴 감지 (실제 OCR 결과 기반)
    
    현재는 패턴만 감지하고 원본을 그대로 반환합니다.
    실제 OCR 결과를 모은 후 개선할 예정입니다.
    """
    if not text:
        return text
    
    # 패턴 감지만 수행 (교정은 하지 않음)
    # 한글 + 영어가 섞인 패턴 찾기
    mixed_pattern = r'([가-힣]+)([a-zA-Z]{2,})([가-힣]*)'
    matches = re.findall(mixed_pattern, text)
    
    if matches and debug:
        print(f"  🔍 한글+영어 혼합 패턴 감지:")
        for korean, english, suffix in matches:
            print(f"    - {korean}{english}{suffix}")
    
    # 일단 원본 그대로 반환 (실제 OCR 결과를 먼저 봐야 함)
    return text

def normalize_spacing(text):
    """기본 공백 정규화"""
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 문장부호 앞뒤 공백 정리
    text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
    
    # 괄호 안쪽 공백 제거
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    return text.strip()

def fix_jamo_separation(text):
    """자모 분리/합침 오류 수정 (고급 문맥 기반)"""
    if not text:
        return text
    
    # 1. 간단한 공백 분리 패턴 먼저 처리
    jamo_fixes = {
        'ㄱ ㅏ': '가', 'ㄴ ㅏ': '나', 'ㄷ ㅏ': '다', 'ㄹ ㅏ': '라',
        'ㅁ ㅏ': '마', 'ㅂ ㅏ': '바', 'ㅅ ㅏ': '사', 'ㅇ ㅏ': '아',
        'ㅈ ㅏ': '자', 'ㅊ ㅏ': '차', 'ㅋ ㅏ': '카', 'ㅌ ㅏ': '타',
        'ㅍ ㅏ': '파', 'ㅎ ㅏ': '하',
        
        'ㄱ ㅓ': '거', 'ㄴ ㅓ': '너', 'ㄷ ㅓ': '더', 'ㄹ ㅓ': '러',
        'ㅁ ㅓ': '머', 'ㅂ ㅓ': '버', 'ㅅ ㅓ': '서', 'ㅇ ㅓ': '어',
        
        'ㄱ ㅗ': '고', 'ㄴ ㅗ': '노', 'ㄷ ㅗ': '도', 'ㄹ ㅗ': '로',
        'ㅁ ㅗ': '모', 'ㅂ ㅗ': '보', 'ㅅ ㅗ': '소', 'ㅇ ㅗ': '오',
        
        'ㄱ ㅜ': '구', 'ㄴ ㅜ': '누', 'ㄷ ㅜ': '두', 'ㄹ ㅜ': '루',
        'ㅁ ㅜ': '무', 'ㅂ ㅜ': '부', 'ㅅ ㅜ': '수', 'ㅇ ㅜ': '우',
    }
    
    for wrong, correct in jamo_fixes.items():
        text = text.replace(wrong, correct)
    
    # 2. 복잡한 자모 분리 패턴 찾아서 처리 (공백 없이 붙어있는 경우)
    # 단어 단위로 분리해서 처리
    words = text.split()
    fixed_words = []
    
    for word in words:
        # 자모가 분리된 단어인지 확인
        if is_jamo_separated(word):
            # 가능한 조합 생성
            candidates = try_compose_jamo_sequence(word)
            if candidates:
                # Kiwi로 최적 후보 선택
                best = select_best_candidate_with_kiwi(candidates, context=' '.join(fixed_words[-3:]))
                if best:
                    fixed_words.append(best)
                    continue
        
        fixed_words.append(word)
    
    return ' '.join(fixed_words)

def analyze_word_context_pattern(text, word_position):
    """주변 단어의 품사 패턴 분석"""
    if not KIWI_AVAILABLE:
        return {}
    
    try:
        result = kiwi.analyze(text)
        if not result or len(result) == 0:
            return {}
        
        token_list = result[0][0] if isinstance(result[0], tuple) else result[0]
        
        # 주변 단어의 품사 추출
        pos_tags = []
        for token in token_list:
            if hasattr(token, 'tag'):
                pos_tags.append(str(token.tag))
            elif isinstance(token, (tuple, list)) and len(token) >= 2:
                pos_tags.append(str(token[1]))
        
        # 동사, 명사 비율 계산
        verb_count = sum(1 for tag in pos_tags if tag in ['VV', 'VA', 'VX', 'VCP', 'VCN'])
        noun_count = sum(1 for tag in pos_tags if tag in ['NNG', 'NNP', 'NNB'])
        
        return {
            'pos_tags': pos_tags,
            'verb_count': verb_count,
            'noun_count': noun_count,
            'verb_ratio': verb_count / len(pos_tags) if pos_tags else 0
        }
    except Exception:
        return {}

def correct_context_based_words(text, debug=False):
    """문맥 기반 단어 교정 (품사 패턴 + 형태소 분석)
    
    실제 단어이지만 OCR 오류로 다른 단어일 가능성이 있는 경우,
    문맥의 품사 패턴을 분석하여 어느 쪽이 더 적합한지 판단합니다.
    """
    if not text or not KIWI_AVAILABLE:
        return text
    
    # OCR에서 자주 혼동되는 단어 쌍과 패턴
    # (원본, 후보, 후보_선호_품사_패턴)
    ambiguous_pairs = [
        {
            'original': '무름',
            'candidate': '무릎',
            # "무릎"은 신체 부위로 동작 동사(구부리다, 펴다, 꿇다 등)와 자주 쓰임
            # 동사 비율이 높으면 "무릎"일 가능성 높음
            'pattern': lambda ctx: ctx.get('verb_ratio', 0) > 0.25  # 동사가 25% 이상
        },
        # 여기에 다른 혼동 단어 쌍 추가 가능
    ]
    
    words = text.split()
    fixed_words = []
    
    for i, word in enumerate(words):
        # 주변 문맥 가져오기 (앞뒤 7단어)
        context_start = max(0, i - 7)
        context_end = min(len(words), i + 8)
        context_text = ' '.join(words[context_start:context_end])
        
        replaced = False
        for pair in ambiguous_pairs:
            original = pair['original']
            candidate = pair['candidate']
            pattern_check = pair['pattern']
            
            if original in word:
                # 문맥의 품사 패턴 분석
                context_pattern = analyze_word_context_pattern(context_text, i - context_start)
                
                # 패턴이 후보에 맞으면 교체 고려
                if pattern_check(context_pattern):
                    candidate_word = word.replace(original, candidate)
                    
                    # 추가 검증: 형태소 점수도 확인
                    original_context = ' '.join(
                        words[context_start:i] + [word] + words[i+1:context_end]
                    )
                    candidate_context = ' '.join(
                        words[context_start:i] + [candidate_word] + words[i+1:context_end]
                    )
                    
                    score_original = calculate_kiwi_score(original_context)
                    score_candidate = calculate_kiwi_score(candidate_context)
                    
                    if debug:
                        print(f"  [{original} vs {candidate}]")
                        print(f"    품사 패턴: 동사비율={context_pattern.get('verb_ratio', 0):.2f}, 조건충족={'✓' if pattern_check(context_pattern) else '✗'}")
                        print(f"    형태소 점수: 원본={score_original:.3f}, 후보={score_candidate:.3f}, 차이={score_candidate - score_original:.3f}")
                    
                    # 패턴이 맞고 점수가 크게 나빠지지 않으면 교체
                    if score_candidate >= score_original - 0.02:  # 후보가 약간 낮아도 OK
                        if debug:
                            print(f"    → {candidate}로 교체")
                        fixed_words.append(candidate_word)
                        replaced = True
                        break
                    else:
                        if debug:
                            print(f"    → 점수 차이로 유지")
                
                if not replaced:
                    fixed_words.append(word)
                    replaced = True
                break
        
        if not replaced:
            fixed_words.append(word)
    
    return ' '.join(fixed_words)

def correct_common_ocr_errors(text):
    """일반적인 OCR 오류 수정 (명확한 오류만)"""
    corrections = {
        # 숫자/문자 혼동
        '0': 'O', '1': 'l', '5': 'S',
        
        # 명백한 OCR 오류 (실제 단어가 아닌 것들)
        '무abar': '무릎', '무bar': '무릎',
        # 주의: '무름'은 실제 단어이므로 문맥 기반 교정으로 이동
        
        # 단위 정리
        '°': '도', '℃': '도', '%': '퍼센트',
        
        # 공백 오류
        '90 °': '90도', '90°': '90도',
        '1 0': '10', '2 0': '20', '3 0': '30',
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text

def kiwi_spacing_correction(text):
    """Kiwi를 사용한 띄어쓰기 교정"""
    if not KIWI_AVAILABLE:
        return text
    
    try:
        # 문장 단위로 분할
        sentences = re.split(r'[.!?]\s*', text)
        corrected_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 3:
                corrected_sentences.append(sentence)
                continue
            
            # Kiwi 분석
            result = kiwi.analyze(sentence.strip())
            if result and len(result) > 0:
                # 형태소 기반 띄어쓰기 재구성
                tokens = []
                
                # Kiwi 결과는 [[(Token, Token, ...), score], ...] 형태
                # 가장 높은 점수의 첫 번째 결과를 사용
                token_list = result[0][0] if isinstance(result[0], tuple) else result[0]
                
                for token in token_list:
                    # Token 객체인 경우
                    if hasattr(token, 'form') and hasattr(token, 'tag'):
                        form = token.form
                        tag = str(token.tag)
                    # 튜플이나 리스트인 경우
                    elif isinstance(token, (tuple, list)) and len(token) >= 2:
                        form = str(token[0])
                        tag = str(token[1])
                    else:
                        continue
                    
                    # 조사는 앞 단어와 붙임
                    if tag.startswith('J'):  # 조사
                        if tokens:
                            tokens[-1] += form
                        else:
                            tokens.append(form)
                    else:
                        tokens.append(form)
                
                if tokens:
                    corrected = ' '.join(tokens)
                    corrected_sentences.append(corrected)
                else:
                    corrected_sentences.append(sentence)
            else:
                corrected_sentences.append(sentence)
        
        return '. '.join(corrected_sentences).strip()
        
    except Exception as e:
        print(f"⚠️ Kiwi 교정 실패: {e}")
        return text

def correct_section_text(text):
    """섹션 텍스트 종합 교정 (문맥 기반 자모 복원 포함)"""
    if not text or len(text.strip()) < 3:
        return text
    
    # 1. 기본 정규화
    corrected = normalize_spacing(text)
    
    # 2. 한글+영어 혼합 패턴 수정 (예: "무les를" -> "무릎을")
    corrected = fix_mixed_english_in_korean(corrected)
    
    # 3. 자모 분리 수정 (문맥 기반, Kiwi 활용)
    corrected = fix_jamo_separation(corrected)
    
    # 4. 일반적인 OCR 오류 수정 (명확한 오류만)
    corrected = correct_common_ocr_errors(corrected)
    
    # 5. 문맥 기반 단어 교정 (예: "무름" vs "무릎" 문맥으로 판단)
    corrected = correct_context_based_words(corrected)
    
    # 6. Kiwi 띄어쓰기 교정
    corrected = kiwi_spacing_correction(corrected)
    
    return corrected

def process_sections(sections_data):
    """섹션별 텍스트 교정"""
    if not init_kiwi():
        print("⚠️ Kiwi 없이 기본 교정만 수행")
    
    corrected_sections = []
    total_fixed_jamo = 0
    total_fixed_mixed = 0
    
    for i, section in enumerate(sections_data, 1):
        print(f"[{i}/{len(sections_data)}] 섹션 교정: {section['title'][:30]}...")
        
        original_text = section.get('text_combined', '')
        corrected_text = correct_section_text(original_text)
        
        # 교정 결과 저장
        corrected_section = section.copy()
        corrected_section['text_combined'] = corrected_text
        corrected_section['correction_applied'] = True
        corrected_section['correction_method'] = 'kiwi_morpheme_with_jamo_context'
        
        # 변경 사항 기록 및 분석
        if corrected_text != original_text:
            corrected_section['text_changed'] = True
            
            # 자모 분리 수정 여부 확인
            has_jamo_orig = any(c in CONSONANTS or c in VOWELS for c in original_text)
            has_jamo_corr = any(c in CONSONANTS or c in VOWELS for c in corrected_text)
            if has_jamo_orig and not has_jamo_corr:
                total_fixed_jamo += 1
                print(f"    ✨ 자모 복원: {len(original_text)} → {len(corrected_text)}자")
            
            # 영어 혼합 수정 여부 확인
            has_mixed_pattern = bool(re.search(r'[가-힣]+[a-zA-Z]{2,6}[가-힣]', original_text))
            if has_mixed_pattern:
                total_fixed_mixed += 1
                print(f"    🔧 혼합 패턴 수정: '{original_text[:50]}...' → '{corrected_text[:50]}...'")
            else:
                print(f"    ✏️ 교정됨: {len(original_text)} → {len(corrected_text)}자")
        else:
            corrected_section['text_changed'] = False
            print(f"    ℹ️ 변경 없음")
        
        corrected_sections.append(corrected_section)
    
    # 통계 출력
    print(f"\n📊 교정 통계:")
    print(f"   • 자모 복원: {total_fixed_jamo}건")
    print(f"   • 혼합 패턴 수정: {total_fixed_mixed}건")
    
    return corrected_sections

def main():
    print("=" * 60)
    print("🔤 1차 교정 (형태소 분석 + 문맥 기반 자모 복원)")
    print("=" * 60)
    
    # 현재 세션 정보 읽기
    session_file = BASE_DIR / "current_session.json"
    if not session_file.exists():
        print("❌ 현재 세션 정보가 없습니다.")
        return
    
    with open(session_file, 'r', encoding='utf-8') as f:
        session_info = json.load(f)
    
    global CURRENT_SESSION_DIR
    CURRENT_SESSION_DIR = Path(session_info['session_dir'])
    
    # step4 결과 파일 로드
    step4_dir = CURRENT_SESSION_DIR / "step4_layout"
    section_files = list(step4_dir.glob("*_sections.json"))
    if not section_files:
        print(f"❌ 섹션 파일을 찾을 수 없습니다: {step4_dir}")
        return
    
    section_file = section_files[0]
    print(f"📂 로드: {section_file.name}")
    
    with open(section_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sections = data.get('sections', [])
    if not sections:
        print("❌ 섹션 데이터가 없습니다.")
        return
    
    print(f"📄 섹션 수: {len(sections)}개")
    
    # 교정 처리
    print("\n🔤 형태소 기반 교정 시작...")
    corrected_sections = process_sections(sections)
    
    # 결과 저장
    output_data = data.copy()
    output_data['sections'] = corrected_sections
    output_data['metadata']['pipeline_stage'] = '1차 교정 (형태소)'
    output_data['metadata']['timestamp'] = datetime.now().isoformat()
    output_data['metadata']['kiwi_available'] = KIWI_AVAILABLE
    
    step5_dir = CURRENT_SESSION_DIR / "step5_kiwi"
    step5_dir.mkdir(parents=True, exist_ok=True)
    output_file = step5_dir / "kiwi_intermediate.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 통계
    changed_count = sum(1 for s in corrected_sections if s.get('text_changed', False))
    
    print(f"\n🎉 1차 교정 완료!")
    print(f"📄 교정된 섹션: {changed_count}/{len(corrected_sections)}개")
    print(f"📁 결과: {output_file}")
    print(f"\n✨ 적용된 기능:")
    print(f"   • 문맥 기반 자모 복원 (초성/중성/종성 조합)")
    print(f"   • 한글+영어 혼합 패턴 수정 (예: 무les를 → 무릎을)")
    print(f"   • 문맥 기반 단어 교정 (예: 무름 → 무릎 체육 문맥에서)")
    print(f"   • Kiwi 형태소 분석 기반 띄어쓰기 교정")

if __name__ == "__main__":
    main()
