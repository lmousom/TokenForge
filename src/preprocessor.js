/**
 * Text preprocessor with multilingual support
 */
class UniversalTextPreprocessor {
    constructor(options = {}) {
      this.options = {
        caseSensitive: options.caseSensitive || false,
        normalizeWhitespace: options.normalizeWhitespace !== undefined ? options.normalizeWhitespace : true,
        stripAccents: options.stripAccents || false,
        normalizeUnicode: options.normalizeUnicode || true
      };
    }
    
    /**
     * Process text according to configuration options
     * @param {string} text - Input text
     * @returns {string} - Processed text
     */
    process(text) {
      if (!text) return '';
      
      let processed = text;
      
      // Unicode normalization (important for multilingual)
      if (this.options.normalizeUnicode) {
        processed = processed.normalize('NFC');
      }
      
      // Case sensitivity
      if (!this.options.caseSensitive) {
        processed = processed.toLowerCase();
      }
      
      // Normalize whitespace
      if (this.options.normalizeWhitespace) {
        processed = processed.replace(/\s+/g, ' ');
      }
      
      // Strip accents
      if (this.options.stripAccents) {
        processed = processed.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
      }
      
      return processed;
    }
  }
  
  module.exports = UniversalTextPreprocessor;