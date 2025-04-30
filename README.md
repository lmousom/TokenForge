# TokenForge

A high-performance, production-ready BPE (Byte-Pair Encoding) tokenizer implementation with universal character support and multilingual capabilities. TokenForge provides robust tokenization for natural language processing tasks across multiple languages.

## Features

- **Universal Character Support**: Works with all Unicode characters
- **Multilingual Capabilities**: Built-in support for multiple scripts and languages
- **High-Performance Tokenization**: Optimized for speed with caching
- **Robust Edge Case Handling**: Properly manages special tokens and unknown characters
- **Configurable**: Extensive options for customization

## Classes

- `UniversalBPETokenizer`: Core tokenizer implementation with BPE algorithm
- `MultilingualUniversalTokenizer`: Extended version with better multilingual support
- `UniversalTextPreprocessor`: Helper for text normalization and preprocessing

## Usage

### Basic Usage

```javascript
// Initialize a basic tokenizer
const tokenizer = new UniversalBPETokenizer({
  vocabSize: 1000, 
  caseSensitive: true
});

// Train on a corpus
await tokenizer.train(myCorpus);

// Tokenize text
const tokenIds = tokenizer.tokenize("Hello, world!");

// Decode back to text
const decoded = tokenizer.decode(tokenIds);
```

### Multilingual Usage

```javascript
// Initialize a multilingual tokenizer
const multiTokenizer = new MultilingualUniversalTokenizer({
  vocabSize: 5000,
  caseSensitive: true
});

// Train on multilingual corpus
await multiTokenizer.train(multilingualCorpus);

// Detect language
const lang = multiTokenizer.detectLanguage("こんにちは世界");

// Smart tokenize with language detection
const tokenIds = multiTokenizer.smartTokenize("こんにちは世界");
```

### Save and Load

```javascript
// Save tokenizer state
const savedModel = tokenizer.save();

// Save to disk (requires Node.js)
fs.writeFileSync('tokenizer.json', JSON.stringify(savedModel));

// Load tokenizer state
const loadedTokenizer = new UniversalBPETokenizer().load(savedModel);
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| vocabSize | 10000 | Maximum vocabulary size |
| minFrequency | 2 | Minimum frequency for merge operations |
| specialTokens | ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] | Special tokens to include |
| caseSensitive | true | Whether to preserve case |
| useCache | true | Enable tokenization caching |
| maxCacheSize | 10000 | Maximum cache size |
| guaranteeAscii | true | Ensure all ASCII characters are in vocabulary |
| preprocessText | true | Apply text preprocessing |

## How It Works

The tokenizer uses the BPE algorithm to learn subword units:

1. Initialize vocabulary with individual characters
2. Count frequencies of adjacent pairs
3. Merge most frequent pair and add to vocabulary
4. Repeat until target vocabulary size is reached
5. Use learned merges to tokenize new text

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.