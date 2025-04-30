/**
 * TokenForge Demo
 * 
 * This script demonstrates the usage of TokenForge's universal and multilingual tokenizers
 */

const { 
    UniversalBPETokenizer, 
    MultilingualUniversalTokenizer 
  } = require('../src');
  
  /**
   * Demonstrate the universal tokenizer
   */
  async function demonstrateTokenForge() {
    // Sample training text (expanded to include uppercase and more punctuation)
    const trainingText = `
      Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
      concerned with the interactions between computers and human language, in particular how to program computers 
      to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" 
      the contents of documents, including the contextual nuances of the language within them.
      
      The technology can then accurately extract information and insights contained in the documents as well as 
      categorize and organize the documents themselves. NLP is used in many everyday applications, such as email filters, 
      digital assistants, voice-operated GPS systems, and home automation systems. NLP is also used in search engines, 
      social media monitoring, and for analysis of customer feedback.
      
      Challenges in natural language processing frequently involve speech recognition, natural language understanding, 
      and natural-language generation. AI models like GPT, BERT, and others have revolutionized how we approach these tasks.
      
      Deep learning has transformed the field of NLP. Models can now understand context, semantics, and even generate
      human-like text. The future of NLP looks promising with innovations in few-shot learning and multimodal models.
      
      Both small startups and big tech companies are investing in AI & NLP research. Examples include OpenAI, Google, 
      Microsoft, and Anthropic. The field continues to advance rapidly in 2023-2024.
      
      自然言語処理（NLP）は、言語学、コンピュータサイエンス、人工知能の分野であり、
      コンピュータと人間の言語の相互作用に関するものです。特に、コンピュータがどのように
      大量の自然言語データを処理し分析するかに焦点を当てています。目標は、文書の内容を
      「理解」できるコンピュータを作ることであり、言語の文脈的なニュアンスも含まれます。
      
      ディープラーニングはNLPの分野を変革しました。モデルは現在、文脈や意味を理解し、
      さらに人間のようなテキストを生成することができます。少数ショット学習やマルチモーダル
      モデルのイノベーションにより、NLPの未来は有望です。
    `;
    
    console.log("Training text length:", trainingText.length);
    
    // Initialize universal tokenizer
    console.log("\n1. Initializing standard TokenForge tokenizer");
    const tokenizer = new UniversalBPETokenizer({
      vocabSize: 500,
      caseSensitive: true,
      guaranteeAscii: true
    });
    
    // Train tokenizer
    console.log("\n2. Training standard tokenizer...");
    await tokenizer.train(trainingText, 500);
    
    // Initialize multilingual tokenizer
    console.log("\n3. Initializing multilingual TokenForge tokenizer");
    const multiTokenizer = new MultilingualUniversalTokenizer({
      vocabSize: 500,
      caseSensitive: true
    });
    
    // Train with same text for comparison
    console.log("\n4. Training multilingual tokenizer...");
    await multiTokenizer.train(trainingText, 500);
    
    // Tokenize test sample
    const sampleText = "Natural language processing is important for AI. Deep learning has revolutionized NLP.";
    
    // Benchmark standard vs multilingual
    console.log("\n--- Benchmarking ---");
    const iterations = 1000;
    
    console.time('Universal tokenizer');
    for (let i = 0; i < iterations; i++) {
      tokenizer.tokenize(sampleText);
    }
    console.timeEnd('Universal tokenizer');
    
    console.time('Multilingual tokenizer');
    for (let i = 0; i < iterations; i++) {
      multiTokenizer.tokenize(sampleText);
    }
    console.timeEnd('Multilingual tokenizer');
    
    // Show results
    const universalTokenIds = tokenizer.tokenize(sampleText);
    const multiTokenIds = multiTokenizer.tokenize(sampleText);
    
    console.log("\n--- Results ---");
    console.log("Sample text:", sampleText);
    
    console.log("\nUniversal tokenizer:");
    console.log("Token IDs:", universalTokenIds);
    console.log("Decoded:", tokenizer.decode(universalTokenIds));
    
    console.log("\nMultilingual tokenizer:");
    console.log("Token IDs:", multiTokenIds);
    console.log("Decoded:", multiTokenizer.decode(multiTokenIds));
    
    // Test multilingual support
    const japaneseSample = `自然言語処理（NLP）は、言語学、コンピュータサイエンス、人工知能の分野であり、
      コンピュータと人間の言語の相互作用に関するものです。特に、コンピュータがどのように
      大量の自然言語データを処理し分析するかに焦点を当てています。`;
    
    console.log("\n--- Multilingual Support Test ---");
    console.log("Japanese sample text:", japaneseSample.slice(0, 50) + "...");
    
    // Get token IDs for Japanese text
    const japaneseTokenIds = multiTokenizer.smartTokenize(japaneseSample);
    console.log("Token count:", japaneseTokenIds.length);
    console.log("First 10 token IDs:", japaneseTokenIds.slice(0, 10));
    console.log("Decoded first 10 tokens:", multiTokenizer.decode(japaneseTokenIds.slice(0, 10)));
    console.log("Detected language:", multiTokenizer.detectLanguage(japaneseSample));
    
    // Check vocabulary
    const universalVocab = tokenizer.getVocabulary();
    const multiVocab = multiTokenizer.getVocabulary();
    
    console.log("\n--- Vocabulary Stats ---");
    console.log("Universal vocabulary size:", universalVocab.size);
    console.log("Multilingual vocabulary size:", multiVocab.size);
    
    // Print some example tokens (first 10 after special tokens)
    console.log("\nSample Universal tokens:", universalVocab.tokens.slice(5, 15));
    console.log("Sample Multilingual tokens:", multiVocab.tokens.slice(5, 15));
    
    // Test save and load functionality
    console.log("\n--- Save & Load Test ---");
    
    const savedModel = tokenizer.save();
    console.log("Model saved, size:", JSON.stringify(savedModel).length + " bytes");
    
    const loadedTokenizer = new UniversalBPETokenizer().load(savedModel);
    const loadedTokenIds = loadedTokenizer.tokenize(sampleText);
    
    console.log("Original token IDs:", universalTokenIds);
    console.log("Loaded token IDs:  ", loadedTokenIds);
    console.log("Token IDs match:", JSON.stringify(universalTokenIds) === JSON.stringify(loadedTokenIds));
  }
  
  // Run the demonstration
  demonstrateTokenForge().catch(console.error);
  
  // Export for testing
  module.exports = { demonstrateTokenForge };