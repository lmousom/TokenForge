const UniversalTextPreprocessor = require('./preprocessor');

/**
 * Universal BPE Tokenizer
 */
class UniversalBPETokenizer {
    constructor(options = {}) {
      // Default configuration
      this.config = {
        vocabSize: options.vocabSize || 10000,
        minFrequency: options.minFrequency || 2,
        specialTokens: options.specialTokens || ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
        caseSensitive: options.caseSensitive !== undefined ? options.caseSensitive : true,
        useCache: options.useCache !== undefined ? options.useCache : true,
        maxCacheSize: options.maxCacheSize || 10000,
        guaranteeAscii: options.guaranteeAscii !== undefined ? options.guaranteeAscii : true,
        preprocessText: options.preprocessText !== undefined ? options.preprocessText : true
      };
      
      // Core data structures
      this.vocab = new Map(); // token → id
      this.invertedVocab = new Map(); // id → token
      this.merges = []; // ordered merge operations
      this.mergeLookup = new Map(); // Fast lookup for merges
      this.specialTokenIds = new Map(); // special token → id
      this.initialized = false;
      this.tokenCache = new Map(); // Caching for fast tokenization
      
      // Preprocessing
      this.preprocessor = new UniversalTextPreprocessor({
        caseSensitive: this.config.caseSensitive,
        normalizeWhitespace: true
      });
    }
    
    /**
     * Initialize the tokenizer with special tokens
     */
    init() {
      console.log("Initializing tokenizer...");
      // Add special tokens first to ensure they have the lowest IDs
      for (const token of this.config.specialTokens) {
        const id = this.vocab.size;
        this.vocab.set(token, id);
        this.invertedVocab.set(id, token);
        this.specialTokenIds.set(token, id);
      }
      
      this.unkTokenId = this.specialTokenIds.get('[UNK]');
      this.initialized = true;
      return this;
    }
  
    /**
     * Build initial vocabulary from text corpus
     * Guarantees all ASCII characters are included
     */
    buildInitialVocabulary(corpus) {
      console.log("Building initial vocabulary...");
      const charFreq = new Map();
      
      // First pass: count character frequencies
      for (const text of corpus) {
        const processed = this.config.preprocessText ? 
          this.preprocessor.process(text) : text;
        
        for (const char of processed) {
          charFreq.set(char, (charFreq.get(char) || 0) + 1);
        }
      }
      
      // Sort characters by frequency
      const sortedChars = Array.from(charFreq.entries())
        .sort((a, b) => b[1] - a[1]); // Sort by frequency, descending
      
      // Calculate how many characters we can add
      const availableSlots = this.config.vocabSize - this.config.specialTokens.length;
      
      // Add most frequent characters first
      let addedChars = 0;
      for (const [char, freq] of sortedChars) {
        if (addedChars >= availableSlots) break;
        if (freq >= this.config.minFrequency) {
          const id = this.vocab.size;
          this.vocab.set(char, id);
          this.invertedVocab.set(id, char);
          addedChars++;
        }
      }
      
      // If we still have space and guaranteeAscii is true, add remaining ASCII characters
      if (this.config.guaranteeAscii && addedChars < availableSlots) {
        // Add all printable ASCII characters (32-126)
        for (let i = 32; i <= 126; i++) {
          const char = String.fromCharCode(i);
          if (!this.vocab.has(char) && addedChars < availableSlots) {
            const id = this.vocab.size;
            this.vocab.set(char, id);
            this.invertedVocab.set(id, char);
            addedChars++;
          }
        }
        
        // Add common control characters
        const controlChars = ['\n', '\t', '\r'];
        for (const char of controlChars) {
          if (!this.vocab.has(char) && addedChars < availableSlots) {
            const id = this.vocab.size;
            this.vocab.set(char, id);
            this.invertedVocab.set(id, char);
            addedChars++;
          }
        }
      }
      
      console.log(`Initial vocabulary built with ${this.vocab.size - this.config.specialTokens.length} characters`);
      return this;
    }
    
    /**
     * Train the tokenizer on a corpus
     * @param {string|string[]} corpus - Text or array of texts
     * @param {number} maxIterations - Maximum BPE iterations
     */
    async train(corpus, maxIterations = 10000) {
      console.time('Training');
      if (!this.initialized) {
        this.init();
        
        // Handle both single string and array of strings
        const textCorpus = Array.isArray(corpus) ? corpus : [corpus];
        this.buildInitialVocabulary(textCorpus);
      }
      
      // Convert corpus to tokens (initially characters)
      let tokenizedCorpus = this.tokenizeCorpus(corpus);
      const SEPARATOR = '\u0000'; // Use same separator as in getPairStatistics
      
      console.log('Starting training loop with vocab size:', this.vocab.size);
      console.log('Target vocab size:', this.config.vocabSize);
      console.log('Max iterations:', maxIterations);
      
      // Learn merges until reaching target vocabulary size or max iterations
      let iteration = 0;
      while (iteration < maxIterations) {
        console.log('Iteration', iteration, 'current vocab size:', this.vocab.size);
        
        // Find most frequent pair
        const pairStats = this.getPairStatistics(tokenizedCorpus);
        console.log('Found', pairStats.size, 'unique pairs');
        
        if (pairStats.size === 0) {
          console.log('No pairs found, breaking');
          break;
        }
        
        let bestPair = null;
        let maxFreq = 0;
        
        for (const [pair, freq] of pairStats.entries()) {
          if (freq > maxFreq) {
            maxFreq = freq;
            bestPair = pair;
          }
        }
        
        console.log('Best pair:', bestPair, 'with frequency:', maxFreq);
        console.log('Min frequency required:', this.config.minFrequency);
        
        if (maxFreq < this.config.minFrequency) {
          console.log('Best pair frequency below minimum required');
          break;
        }
        
        // Create new token from pair
        const [first, second] = bestPair.split(SEPARATOR);
        const newToken = first + second;
        const newId = this.vocab.size;
        
        console.log('Creating new token:', newToken, 'with ID:', newId);
        
        // Add to vocabulary
        this.vocab.set(newToken, newId);
        this.invertedVocab.set(newId, newToken);
        this.merges.push([first, second]);
        
        // Apply the merge
        tokenizedCorpus = this.applyMerge(tokenizedCorpus, first, second, newToken);
        
        iteration++;
        if (iteration % 100 === 0 || iteration === 1) {
          console.log(`Iteration ${iteration}: Merged "${first}" + "${second}" → "${newToken}", vocab size: ${this.vocab.size}`);
        }
        
        // Break if we've reached the target vocabulary size
        if (this.vocab.size >= this.config.vocabSize) {
          console.log('Reached target vocabulary size');
          break;
        }
      }
      
      // Build the merge lookup for fast tokenization
      this.buildMergeLookup();
      
      console.log(`Training completed after ${iteration} iterations. Final vocab size: ${this.vocab.size}`);
      console.timeEnd('Training');
      return this;
    }
    
    /**
     * Convert a corpus to tokens
     */
    tokenizeCorpus(corpus) {
      const textCorpus = Array.isArray(corpus) ? corpus : [corpus];
      console.log('Tokenizing corpus:', textCorpus);
      const result = textCorpus.map(text => {
        const processed = this.config.preprocessText ? 
          this.preprocessor.process(text) : text;
        console.log('Processed text:', processed);
        const tokens = processed.split('');
        console.log('Initial tokens:', tokens);
        return tokens;
      });
      console.log('Tokenized corpus:', result);
      return result;
    }
    
    /**
     * Count pair frequencies in tokenized corpus
     */
    getPairStatistics(tokenizedCorpus) {
      const pairCounts = new Map();
      const SEPARATOR = '\u0000'; // Use same separator as in train method
      
      console.log('Processing tokenized corpus:', tokenizedCorpus);
      
      for (const sequence of tokenizedCorpus) {
        console.log('Processing sequence:', sequence);
        for (let i = 0; i < sequence.length - 1; i++) {
          const pair = `${sequence[i]}${SEPARATOR}${sequence[i+1]}`;
          const count = (pairCounts.get(pair) || 0) + 1;
          pairCounts.set(pair, count);
          console.log('Found pair:', sequence[i], '+', sequence[i+1], 'count:', count);
        }
      }
      
      console.log('All pair counts:', Array.from(pairCounts.entries()));
      return pairCounts;
    }
    
    /**
     * Apply a merge operation to all token sequences
     */
    applyMerge(tokenizedCorpus, first, second, newToken) {
      return tokenizedCorpus.map(sequence => {
        const result = [];
        let i = 0;
        
        while (i < sequence.length) {
          if (i < sequence.length - 1 && sequence[i] === first && sequence[i+1] === second) {
            result.push(newToken);
            i += 2;
          } else {
            result.push(sequence[i]);
            i += 1;
          }
        }
        
        return result;
      });
    }
    
    /**
     * Build lookup table for fast merging
     */
    buildMergeLookup() {
      this.mergeLookup.clear();
      
      for (let i = 0; i < this.merges.length; i++) {
        const [first, second] = this.merges[i];
        
        if (!this.mergeLookup.has(first)) {
          this.mergeLookup.set(first, []);
        }
        
        this.mergeLookup.get(first).push({
          second,
          newToken: first + second,
          priority: i
        });
      }
      
      // Sort potential merges by priority
      for (const [_, merges] of this.mergeLookup.entries()) {
        merges.sort((a, b) => a.priority - b.priority);
      }
    }
    
    /**
     * Tokenize text into subword tokens
     * @param {string} text - Input text
     * @returns {number[]} - Array of token IDs
     */
    tokenize(text) {
      if (!this.initialized) {
        throw new Error("Tokenizer not initialized. Call init() and train() first.");
      }
      
      // Check cache if enabled
      const cacheKey = this.config.useCache ? text : null;
      if (cacheKey && this.tokenCache.has(cacheKey)) {
        return this.tokenCache.get(cacheKey);
      }
      
      // Preprocess text
      const processed = this.config.preprocessText ? 
        this.preprocessor.process(text) : text;
      
      // Initial character splitting
      let tokens = processed.split('');
      
      // Apply merges using the efficient lookup structure
      tokens = this.applyMergesEfficiently(tokens);
      
      // Convert to IDs, handling OOV tokens
      const tokenIds = tokens.map(token => {
        const id = this.vocab.get(token);
        return id !== undefined ? id : this.unkTokenId;
      });
      
      // Cache result if caching is enabled
      if (cacheKey && this.tokenCache.size < this.config.maxCacheSize) {
        this.tokenCache.set(cacheKey, tokenIds);
      }
      
      return tokenIds;
    }
    
    /**
     * Apply merges efficiently using lookup structure
     */
    applyMergesEfficiently(tokens) {
      let changed = true;
      while (changed) {
        changed = false;
        const result = [];
        let i = 0;
        
        while (i < tokens.length) {
          const token = tokens[i];
          const potentialMerges = this.mergeLookup.get(token);
          
          let merged = false;
          if (potentialMerges && i < tokens.length - 1) {
            const nextToken = tokens[i + 1];
            
            for (const merge of potentialMerges) {
              if (merge.second === nextToken) {
                result.push(merge.newToken);
                i += 2;
                merged = true;
                changed = true;
                break;
              }
            }
          }
          
          if (!merged) {
            result.push(token);
            i += 1;
          }
        }
        
        tokens = result;
      }
      
      return tokens;
    }
    
    /**
     * Decode token IDs back to text
     * @param {number[]} tokenIds - Array of token IDs
     * @returns {string} - Decoded text
     */
    decode(tokenIds) {
      return tokenIds.map(id => {
        if (id === undefined || id === null) return '[UNK]';
        const token = this.invertedVocab.get(id);
        return token !== undefined ? token : '[UNK]';
      }).join('');
    }
    
    /**
     * Get vocabulary for inspection
     */
    getVocabulary() {
      return {
        size: this.vocab.size,
        tokens: Array.from(this.vocab.keys()),
        specialTokens: Array.from(this.specialTokenIds.keys())
      };
    }
    
    /**
     * Save the tokenizer model
     */
    save() {
      return {
        config: this.config,
        vocab: Array.from(this.vocab.entries()),
        merges: this.merges,
        specialTokenIds: Array.from(this.specialTokenIds.entries())
      };
    }
    
    /**
     * Load a saved tokenizer model
     */
    load(modelData) {
      this.config = modelData.config;
      this.vocab = new Map(modelData.vocab);
      this.merges = modelData.merges;
      this.specialTokenIds = new Map(modelData.specialTokenIds);
      
      // Rebuild inverted vocabulary
      this.invertedVocab = new Map();
      for (const [token, id] of this.vocab.entries()) {
        this.invertedVocab.set(id, token);
      }
      
      // Set UNK token ID
      this.unkTokenId = this.specialTokenIds.get('[UNK]');
      
      // Build merge lookup for fast tokenization
      this.buildMergeLookup();
      
      this.initialized = true;
      this.preprocessor = new UniversalTextPreprocessor({
        caseSensitive: this.config.caseSensitive
      });
      
      return this;
    }
  }
  
  module.exports = UniversalBPETokenizer;