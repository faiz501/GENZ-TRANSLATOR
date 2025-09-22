"""
Morphology module for slang and regular word correction
Handles typos, morphological analysis, and word normalization
"""

import re
import editdistance
from typing import Set, Tuple, List, Dict

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None

try:
    import nltk
    from nltk.corpus import words
    nltk.download('words', quiet=True)
    ENGLISH_WORDS = set(word.lower() for word in words.words())
except ImportError:
    ENGLISH_WORDS = set()

class MorphologyCorrector:
    """Handles morphological analysis and typo correction"""
    
    def __init__(self, slang_terms: Set[str]):
        self.slang_terms = slang_terms
        
        # Initialize spell checker
        self.spell_checker = SpellChecker() if SpellChecker else None
        
        # Load common English words
        self.common_words = self._load_common_words()
        self.all_correct_words = self.common_words | self.slang_terms | ENGLISH_WORDS
        
        # Morphological rules
        self.morphology_rules = self._build_morphology_rules()
        
        # Common typo patterns
        self.typo_patterns = self._build_typo_patterns()
    
    def _load_common_words(self) -> Set[str]:
        """Load essential English words"""
        return {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'good', 'make', 'they', 'time', 'very', 'when', 'what', 'with', 'have',
            'from', 'this', 'that', 'will', 'would', 'there', 'their', 'about'
        }
    
    def _build_morphology_rules(self) -> List[Dict]:
        """Build morphological transformation rules"""
        return [
            # Slang-specific rules (highest priority)
            {'pattern': r'^(.+)ed$', 'stem': r'\1', 'priority': 10, 'type': 'slang_past'},
            {'pattern': r'^(.+)ing$', 'stem': r'\1', 'priority': 10, 'type': 'slang_gerund'},
            {'pattern': r'^(.+)s$', 'stem': r'\1', 'priority': 9, 'type': 'slang_plural', 'condition': 'not_ss_ending'},
            {'pattern': r'^(.+)er$', 'stem': r'\1', 'priority': 8, 'type': 'slang_agent'},
            {'pattern': r'^(.+)y$', 'stem': r'\1', 'priority': 7, 'type': 'slang_adjective'},
            
            # Standard English morphology
            {'pattern': r'^(.+)ies$', 'stem': r'\1y', 'priority': 6, 'type': 'standard_plural_y'},
            {'pattern': r'^(.+)ied$', 'stem': r'\1y', 'priority': 6, 'type': 'standard_past_y'},
            {'pattern': r'^(.+)ly$', 'stem': r'\1', 'priority': 5, 'type': 'adverb'},
            {'pattern': r'^(.+)est$', 'stem': r'\1', 'priority': 4, 'type': 'superlative'},
        ]
    
    def _build_typo_patterns(self) -> Dict[str, str]:
        """Build common typo correction patterns"""
        return {
            # Common English typos
            'teh': 'the', 'adn': 'and', 'wierd': 'weird', 'freind': 'friend',
            'definately': 'definitely', 'seperate': 'separate', 'occured': 'occurred',
            
            # Contractions
            'youre': "you're", 'cant': "can't", 'dont': "don't", 'wont': "won't",
            'thats': "that's", 'whats': "what's", 'isnt': "isn't", 'hasnt': "hasn't",
            
            # Internet shorthand
            'ur': 'your', 'u': 'you', 'ppl': 'people', 'msg': 'message',
            'txt': 'text', 'thru': 'through', 'tho': 'though', 'cuz': 'because'
        }
    
    def apply_morphology(self, word: str) -> Tuple[str, str, float]:
        """Apply morphological analysis to find base form"""
        word_lower = word.lower()
        
        # Check contractions and patterns first
        if word_lower in self.typo_patterns:
            return self.typo_patterns[word_lower], "pattern_correction", 0.95
        
        # Apply morphological rules
        for rule in sorted(self.morphology_rules, key=lambda x: x['priority'], reverse=True):
            match = re.match(rule['pattern'], word_lower)
            if match:
                stem = re.sub(rule['pattern'], rule['stem'], word_lower)
                
                # Check conditions
                if rule.get('condition') == 'not_ss_ending' and word_lower.endswith('ss'):
                    continue
                
                # For slang rules, only apply if stem is actual slang
                if rule['type'].startswith('slang') and stem in self.slang_terms:
                    confidence = 0.9 - (10 - rule['priority']) * 0.05  # Higher priority = higher confidence
                    return stem, rule['type'], confidence
                
                # For standard rules, apply if stem is valid English word
                elif not rule['type'].startswith('slang') and stem in self.all_correct_words:
                    confidence = 0.8 - (10 - rule['priority']) * 0.05
                    return stem, rule['type'], confidence
        
        return word_lower, "no_morphology", 0.0
    
    def correct_typo(self, word: str) -> Tuple[str, str, float]:
        """Correct typos using edit distance"""
        word_lower = word.lower()
        
        # Skip if already correct
        if word_lower in self.all_correct_words:
            return word, "already_correct", 1.0
        
        # Find edit distance candidates
        candidates = []
        for correct_word in self.all_correct_words:
            if abs(len(word_lower) - len(correct_word)) <= 2:  # Similar length
                distance = editdistance.eval(word_lower, correct_word)
                if 0 < distance <= 2:
                    confidence = 1 - (distance / max(len(word_lower), len(correct_word)))
                    candidates.append((correct_word, distance, confidence))
        
        if not candidates:
            return word, "no_correction", 0.0
        
        # Sort by distance, then by slang preference
        candidates.sort(key=lambda x: (x[1], x[0] not in self.slang_terms))
        
        best_word, distance, confidence = candidates[0]
        
        # Apply threshold based on word type
        threshold = 0.6 if best_word in self.slang_terms else 0.7
        
        if confidence >= threshold:
            correction_type = "slang_typo" if best_word in self.slang_terms else f"typo_edit_{distance}"
            return best_word, correction_type, confidence
        
        return word, "no_correction", 0.0
    
    def correct_word(self, word: str) -> Tuple[str, str, float]:
        """Main word correction function"""
        # Skip very short words
        if len(word) <= 1:
            return word, "skip_short", 1.0
        
        # Try morphological analysis first
        morph_result, morph_type, morph_confidence = self.apply_morphology(word)
        
        # If morphology found valid result, use it
        if morph_confidence > 0.0:
            return morph_result, morph_type, morph_confidence
        
        # Otherwise try typo correction
        return self.correct_typo(word)
    
    def correct_sentence(self, text: str) -> Tuple[str, List[Dict]]:
        """Correct all words in a sentence"""
        words = text.split()
        corrected_words = []
        corrections = []
        
        for i, word in enumerate(words):
            # Extract punctuation
            punctuation = ''
            clean_word = word
            
            # Trailing punctuation
            while clean_word and clean_word[-1] in '.,!?;:)"\'':
                punctuation = clean_word[-1] + punctuation
                clean_word = clean_word[:-1]
            
            # Leading punctuation
            leading_punct = ''
            while clean_word and clean_word[0] in '("\'':
                leading_punct += clean_word[0]
                clean_word = clean_word[1:]
            
            if clean_word:
                corrected, correction_type, confidence = self.correct_word(clean_word)
                
                if (correction_type not in ["already_correct", "skip_short", "no_correction"] and 
                    confidence > 0.6 and corrected != clean_word.lower()):
                    
                    corrections.append({
                        'original': clean_word,
                        'corrected': corrected,
                        'type': correction_type,
                        'confidence': confidence,
                        'position': i
                    })
                
                corrected_words.append(leading_punct + corrected + punctuation)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words), corrections
