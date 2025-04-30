const { UniversalBPETokenizer } = require('../src');

describe('UniversalBPETokenizer', () => {
  let tokenizer;
  const sampleText = "The quick brown fox jumps over the lazy dog.";

  beforeEach(() => {
    // Initialize tokenizer before each test
    tokenizer = new UniversalBPETokenizer({
      vocabSize: 100,
      minFrequency: 1
    });
  });

  test('should initialize with special tokens', () => {
    tokenizer.init();
    const vocab = tokenizer.getVocabulary();
    
    // Check special tokens exist
    expect(vocab.specialTokens).toContain('[PAD]');
    expect(vocab.specialTokens).toContain('[UNK]');
    expect(vocab.specialTokens).toContain('[CLS]');
    expect(vocab.specialTokens).toContain('[SEP]');
    expect(vocab.specialTokens).toContain('[MASK]');
  });

  test('should build initial vocabulary from corpus', () => {
    tokenizer.init();
    tokenizer.buildInitialVocabulary([sampleText]);
    const vocab = tokenizer.getVocabulary();
    
    // Check characters from sample text exist in vocabulary
    'ThabcdefghijklmnopqrstuvwxyzBCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.split('').forEach(char => {
      // Only check basic ASCII characters that should be in our text or guaranteed ASCII
      if (sampleText.includes(char) || (char.charCodeAt(0) >= 32 && char.charCodeAt(0) <= 126)) {
        expect(vocab.tokens).toContain(char);
      }
    });
  });

  test('should train on corpus and create merges', async () => {
    await tokenizer.train(sampleText);
    
    // Verify that we have merges
    expect(tokenizer.merges.length).toBeGreaterThan(0);
    
    // Verify that vocabulary size increased beyond just characters
    expect(tokenizer.vocab.size).toBeGreaterThan(
      new Set(sampleText.split('')).size + tokenizer.config.specialTokens.length
    );
  });

  test('should tokenize and decode text correctly', async () => {
    await tokenizer.train(sampleText);
    
    const tokenIds = tokenizer.tokenize(sampleText);
    expect(tokenIds).toBeInstanceOf(Array);
    expect(tokenIds.length).toBeGreaterThan(0);
    
    const decoded = tokenizer.decode(tokenIds);
    // The decoded text should match the original text
    // (There might be whitespace differences depending on preprocessing)
    const normalizedOriginal = tokenizer.preprocessor.process(sampleText);
    const normalizedDecoded = tokenizer.preprocessor.process(decoded);
    
    expect(normalizedDecoded).toBe(normalizedOriginal);
  });

  test('should handle OOV tokens with UNK token', async () => {
    await tokenizer.train("simple text");
    
    // Tokenize text with characters not in training data
    const tokenIds = tokenizer.tokenize("£€¥");
    
    // All tokens should be replaced with UNK token ID
    expect(tokenIds.every(id => id === tokenizer.unkTokenId)).toBe(true);
  });

  test('should save and load model state', async () => {
    await tokenizer.train(sampleText);
    
    const savedModel = tokenizer.save();
    expect(savedModel).toHaveProperty('config');
    expect(savedModel).toHaveProperty('vocab');
    expect(savedModel).toHaveProperty('merges');
    
    const loadedTokenizer = new UniversalBPETokenizer().load(savedModel);
    
    // Test that tokenization works the same on both tokenizers
    const originalTokenIds = tokenizer.tokenize(sampleText);
    const loadedTokenIds = loadedTokenizer.tokenize(sampleText);
    
    expect(loadedTokenIds).toEqual(originalTokenIds);
  });

  test('should use cache for repeated tokenization', async () => {
    tokenizer = new UniversalBPETokenizer({
      vocabSize: 100,
      useCache: true
    });
    
    await tokenizer.train(sampleText);
    
    // First tokenization
    const tokenIds1 = tokenizer.tokenize(sampleText);
    expect(tokenizer.tokenCache.size).toBe(1);
    
    // Second tokenization should use cache
    const tokenIds2 = tokenizer.tokenize(sampleText);
    expect(tokenIds2).toEqual(tokenIds1);
    
    // Spy on a method that would be called if cache wasn't used
    const spy = jest.spyOn(tokenizer, 'applyMergesEfficiently');
    tokenizer.tokenize(sampleText);
    
    // Verify method wasn't called (cache was used)
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });
});