const UniversalBPETokenizer = require('./tokenizer');

/**
 * Multilingual-ready BPE Tokenizer
 */
class MultilingualUniversalTokenizer extends UniversalBPETokenizer {
  constructor(options = {}) {
    super({
      ...options,
      // Default multilingual-specific options
      vocabSize: options.vocabSize || 50000,
      caseSensitive: options.caseSensitive !== undefined ? options.caseSensitive : true,
      guaranteeAscii: true
    });
    
    // Script-specific character sets
    this.scriptCharSets = {
      latin: this.generateCharSet(0x0020, 0x007F), // Basic Latin
      latinExtended: this.generateCharSet(0x00A0, 0x024F), // Latin Extended
      cyrillic: this.generateCharSet(0x0400, 0x04FF), // Cyrillic
      greek: this.generateCharSet(0x0370, 0x03FF), // Greek
      arabic: this.generateCharSet(0x0600, 0x06FF), // Arabic
      hebrew: this.generateCharSet(0x0590, 0x05FF), // Hebrew
      devanagari: this.generateCharSet(0x0900, 0x097F), // Devanagari (Hindi, Sanskrit)
      hiragana: this.generateCharSet(0x3040, 0x309F), // Japanese Hiragana
      katakana: this.generateCharSet(0x30A0, 0x30FF), // Japanese Katakana
      hangul: this.generateCharSet(0xAC00, 0xD7AF, 100), // Korean Hangul (subset)
      cjk: this.generateCharSet(0x4E00, 0x9FFF, 100), // CJK Unified Ideographs (subset)
    };
    
    // Language detection patterns
    this.langPatterns = {
      ja: /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]/, // Japanese
      ko: /[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]/, // Korean
      zh: /[\u4e00-\u9fff\u3400-\u4dbf]/, // Chinese
      ru: /[\u0400-\u04FF]/, // Russian/Cyrillic
      ar: /[\u0600-\u06FF]/, // Arabic
      hi: /[\u0900-\u097F]/, // Hindi/Devanagari
      he: /[\u0590-\u05FF]/, // Hebrew
      th: /[\u0E00-\u0E7F]/, // Thai
      el: /[\u0370-\u03FF]/, // Greek
    };
  }
  
  /**
   * Generate a set of characters for a Unicode range
   * For large ranges like CJK, we sample to avoid massive vocabularies
   * @param {number} start - Starting code point
   * @param {number} end - Ending code point
   * @param {number|null} maxChars - Maximum number of characters to include (for sampling)
   * @returns {string[]} - Array of characters
   */
  generateCharSet(start, end, maxChars = null) {
    const chars = [];
    const range = end - start + 1;
    
    if (maxChars && range > maxChars) {
      // Sample characters across the range
      const step = Math.floor(range / maxChars);
      for (let i = 0; i < maxChars; i++) {
        const codePoint = start + (i * step);
        chars.push(String.fromCodePoint(codePoint));
      }
    } else {
      // Include all characters in the range
      for (let codePoint = start; codePoint <= end; codePoint++) {
        chars.push(String.fromCodePoint(codePoint));
      }
    }
    
    return chars;
  }
  
  /**
   * Override to add multilingual character sets to initial vocabulary
   */
  buildInitialVocabulary(corpus) {
    super.buildInitialVocabulary(corpus);
    
    // Detect scripts used in the corpus
    const scripts = this.detectScriptsInCorpus(corpus);
    
    // Add common characters from detected scripts
    for (const script of scripts) {
      if (this.scriptCharSets[script]) {
        for (const char of this.scriptCharSets[script]) {
          if (!this.vocab.has(char)) {
            const id = this.vocab.size;
            this.vocab.set(char, id);
            this.invertedVocab.set(id, char);
          }
        }
      }
    }
    
    console.log(`Enhanced vocabulary with ${scripts.join(', ')} script characters. Total size: ${this.vocab.size}`);
    return this;
  }
  
  /**
   * Detect scripts used in the corpus
   * @param {string|string[]} corpus - Text corpus
   * @returns {string[]} - Array of script names
   */
  detectScriptsInCorpus(corpus) {
    const scripts = new Set();
    const textCorpus = Array.isArray(corpus) ? corpus : [corpus];
    
    // Check each text for script patterns
    for (const text of textCorpus) {
      if (/[\u0020-\u007F]/.test(text)) scripts.add('latin');
      if (/[\u00A0-\u024F]/.test(text)) scripts.add('latinExtended');
      if (this.langPatterns.ru.test(text)) scripts.add('cyrillic');
      if (this.langPatterns.el.test(text)) scripts.add('greek');
      if (this.langPatterns.ar.test(text)) scripts.add('arabic');
      if (this.langPatterns.he.test(text)) scripts.add('hebrew');
      if (this.langPatterns.hi.test(text)) scripts.add('devanagari');
      if (this.langPatterns.ja.test(text)) {
        scripts.add('hiragana');
        scripts.add('katakana');
        scripts.add('cjk');
      }
      if (this.langPatterns.ko.test(text)) scripts.add('hangul');
      if (this.langPatterns.zh.test(text)) scripts.add('cjk');
    }
    
    return Array.from(scripts);
  }
  
  /**
   * Detect the primary language of a text
   * @param {string} text - Input text
   * @returns {string} - Detected language code
   */
  detectLanguage(text) {
    // Count matches for each language pattern
    const langScores = {};
    for (const [lang, pattern] of Object.entries(this.langPatterns)) {
      // Count matches for this language pattern
      const matches = (text.match(pattern) || []).length;
      langScores[lang] = matches;
    }
    
    // Find language with highest score
    let bestLang = 'en'; // Default to English
    let maxScore = 0;
    
    for (const [lang, score] of Object.entries(langScores)) {
      if (score > maxScore) {
        maxScore = score;
        bestLang = lang;
      }
    }
    
    return bestLang;
  }
  
  /**
   * Tokenize based on detected language
   * @param {string} text - Input text
   * @returns {number[]} - Array of token IDs
   */
  smartTokenize(text) {
    const lang = this.detectLanguage(text);
    // In a more sophisticated implementation, we could use language-specific preprocessing
    return this.tokenize(text);
  }
}

module.exports = MultilingualUniversalTokenizer;