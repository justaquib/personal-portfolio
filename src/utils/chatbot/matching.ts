import { QAItem, chatbotDataset } from "./dataset";

// Tokenize text into words
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter(word => word.length > 0);
}

// Calculate TF (Term Frequency)
function calculateTF(tokens: string[]): Map<string, number> {
  const tf = new Map<string, number>();
  const total = tokens.length;
  
  for (const token of tokens) {
    tf.set(token, (tf.get(token) || 0) + 1);
  }
  
  // Normalize by total
  for (const [key, value] of tf) {
    tf.set(key, value / total);
  }
  
  return tf;
}

// Calculate IDF (Inverse Document Frequency)
function calculateIDF(documents: string[]): Map<string, number> {
  const idf = new Map<string, number>();
  const N = documents.length;
  
  // Count documents containing each term
  const docFreq = new Map<string, number>();
  
  for (const doc of documents) {
    const tokens = new Set(tokenize(doc));
    for (const token of tokens) {
      docFreq.set(token, (docFreq.get(token) || 0) + 1);
    }
  }
  
  // Calculate IDF
  for (const [term, freq] of docFreq) {
    idf.set(term, Math.log((N + 1) / (freq + 1)) + 1);
  }
  
  return idf;
}

// Pre-compute IDF for the dataset
const allDocuments = chatbotDataset.flatMap(item => [
  item.question,
  item.answer,
  ...item.keywords
]);
const idf = calculateIDF(allDocuments);

// Calculate TF-IDF score between query and text
function calculateTFIDFScore(query: string, text: string): number {
  const queryTokens = tokenize(query);
  const textTokens = tokenize(text);
  
  const queryTF = calculateTF(queryTokens);
  const textTF = calculateTF(textTokens);
  
  let score = 0;
  
  // Calculate cosine similarity
  const textTF_normalized = new Map<string, number>();
  let textMagnitude = 0;
  for (const [term, tf] of textTF) {
    const idfValue = idf.get(term) || 1;
    const weighted = tf * idfValue;
    textTF_normalized.set(term, weighted);
    textMagnitude += weighted * weighted;
  }
  textMagnitude = Math.sqrt(textMagnitude);
  
  let queryMagnitude = 0;
  for (const [term, tf] of queryTF) {
    const idfValue = idf.get(term) || 1;
    const weighted = tf * idfValue;
    queryMagnitude += weighted * weighted;
  }
  queryMagnitude = Math.sqrt(queryMagnitude);
  
  // Cosine similarity
  for (const [term, queryWeight] of queryTF) {
    const termIDF = idf.get(term) || 1;
    const textWeight = textTF_normalized.get(term) || 0;
    score += (queryWeight * termIDF) * (textWeight * termIDF);
  }
  
  if (queryMagnitude * textMagnitude === 0) return 0;
  
  return score / (queryMagnitude * textMagnitude);
}

// Keyword matching score
function calculateKeywordScore(query: string, keywords: string[]): number {
  const queryLower = query.toLowerCase();
  const queryTokens = tokenize(query);
  
  let score = 0;
  
  for (const keyword of keywords) {
    const keywordLower = keyword.toLowerCase();
    
    // Exact match
    if (queryLower.includes(keywordLower)) {
      score += 1.0;
    }
    // Partial match
    else {
      for (const token of queryTokens) {
        if (keywordLower.includes(token) || token.includes(keywordLower)) {
          score += 0.5;
        }
      }
    }
  }
  
  return score / Math.max(keywords.length, 1);
}

// Combined scoring
function calculateScore(query: string, item: QAItem): number {
  const questionScore = calculateTFIDFScore(query, item.question) * 2; // Weight questions higher
  const answerScore = calculateTFIDFScore(query, item.answer) * 0.5;
  const keywordScore = calculateKeywordScore(query, item.keywords);
  
  return questionScore + answerScore + keywordScore;
}

// Find best matching Q&A
export function findBestMatch(userQuery: string, history: Array<{ role: string; content: string }> = []): { answer: string; confidence: number } {
  const query = userQuery.trim();
  
  if (!query) {
    return {
      answer: "Please ask me something about justaquib.com!",
      confidence: 0
    };
  }
  
  // Score all items
  const scored = chatbotDataset.map(item => ({
    item,
    score: calculateScore(query, item)
  }));
  
  // Sort by score descending
  scored.sort((a, b) => b.score - a.score);
  
  // Get top matches
  const topMatches = scored.slice(0, 3);
  
  // Check if we have a good match (threshold)
  const bestMatch = topMatches[0];
  
  if (bestMatch.score > 0.1) {
    return {
      answer: bestMatch.item.answer,
      confidence: Math.min(bestMatch.score * 100, 100)
    };
  }
  
  // No good match found - provide helpful response based on context
  return {
    answer: "I'm not sure I understand that specific question, but I can tell you about:\n\n• Aquib Shahbaz - the developer\n• All 10 projects on this website\n• The tech stack used\n• How to contact Aquib\n\nTry asking something like 'Who is Aquib?' or 'Tell me about the AI Document Intelligence Workspace'!",
    confidence: 10
  };
}

// Get suggested questions
export function getSuggestedQuestions(): string[] {
  return [
    "Who is Aquib Shahbaz?",
    "What projects are available?",
    "What is AI Document Intelligence Workspace?",
    "How can I contact Aquib?"
  ];
}
