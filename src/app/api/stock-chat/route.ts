import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextRequest, NextResponse } from "next/server";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

// List of models to try in order (primary first, then fallbacks)
const MODEL_FALLBACKS = [
  "gemini-3-pro",
  "gemini-2.0-pro",
  "gemini-2.5-flash",
  "gemini-2.0-flash"
];

// Helper function to try generating content with model fallback
async function generateWithFallback(
  prompt: string,
  modelName: string
): Promise<{ text: string; usedModel: string }> {
  const model = genAI.getGenerativeModel({ model: modelName });
  const result = await model.generateContent(prompt);
  const response = await result.response;
  return { text: response.text(), usedModel: modelName };
}

// Helper function to try chat with model fallback
async function chatWithFallback(
  stockContext: string,
  question: string,
  chatHistory: Array<{ role: string; parts: Array<{ text: string }> }>,
  modelName: string
): Promise<{ text: string; usedModel: string }> {
  const model = genAI.getGenerativeModel({ model: modelName });
  
  const chat = model.startChat({
    history: [
      {
        role: "user",
        parts: [{ text: `You are a professional stock market analyst and financial advisor. You have deep knowledge of stock markets, financial analysis, technical indicators, and investment strategies. 

You will answer questions about the following stock:

${stockContext}

Important guidelines:
- Provide accurate, data-driven answers based on the stock information provided
- If the user asks about a specific stock, always reference the data provided
- Give practical investment insights when asked
- Explain financial terms when needed
- Be concise but thorough in your responses
- Always cite specific metrics from the provided data when relevant` }],
      },
      {
        role: "model",
        parts: [{ text: "I understand. I'm a professional stock market analyst and I'll use the stock data provided to answer your questions. I'll give you accurate, data-driven insights about the stock, including its current price, performance, technical indicators, and investment potential. What would you like to know about this stock?" }],
      },
      ...chatHistory,
    ],
  });

  const result = await chat.sendMessage(question);
  const response = await result.response;
  return { text: response.text(), usedModel: modelName };
}

// Wrapper function to handle model fallback logic
async function tryWithFallbacks<T>(
  operation: (modelName: string) => Promise<T>,
  operationName: string
): Promise<T> {
  const errors: Array<{ model: string; error: Error }> = [];

  for (const modelName of MODEL_FALLBACKS) {
    try {
      console.log(`Trying ${operationName} with model: ${modelName}`);
      const result = await operation(modelName);
      console.log(`Success with model: ${modelName}`);
      return result;
    } catch (error) {
      console.error(`Failed with model ${modelName}:`, error);
      errors.push({ model: modelName, error: error as Error });
    }
  }

  // All models failed, throw the last error with details
  const errorDetails = errors.map(e => `${e.model}: ${e.error.message}`).join("; ");
  throw new Error(`All models failed for ${operationName}. Errors: ${errorDetails}`);
}

export async function POST(request: NextRequest) {
  console.log('Stock Chat API called');
  console.log('GEMINI_API_KEY present:', !!process.env.GEMINI_API_KEY);
  console.log('GEMINI_API_KEY value:', process.env.GEMINI_API_KEY?.substring(0, 10) + '...');
  
  try {
    const { stock, question, history } = await request.json();
    console.log('Received request:', { stock: stock?.symbol, question: question?.substring(0, 50) });

    if (!process.env.GEMINI_API_KEY) {
      console.error('Missing Gemini API key');
      return NextResponse.json(
        { error: "Gemini API key not configured. Please add GEMINI_API_KEY to .env.local" },
        { status: 500 }
      );
    }

    if (!stock || !question) {
      return NextResponse.json(
        { error: "Stock data and question are required" },
        { status: 400 }
      );
    }

    // Build stock context from the selected stock
    const currency = stock.marketCap > 1e12 ? '$' : '₹';
    const marketCapValue = stock.marketCap > 1e12 ? (stock.marketCap / 1e12).toFixed(2) : (stock.marketCap / 1e9).toFixed(2);
    const marketCapUnit = stock.marketCap > 1e12 ? 'Trillion' : 'Billion';
    const dividendYield = stock.dividend > 0 ? ((stock.dividend / stock.price) * 100).toFixed(2) : 'N/A';

    const stockContext = `
STOCK INFORMATION:
- Symbol: ${stock.symbol}
- Company Name: ${stock.name}
- Sector: ${stock.sector}
- Current Price: ${currency}${stock.price.toFixed(2)}
- Price Change: ${stock.change >= 0 ? '+' : ''}${stock.change.toFixed(2)} (${stock.changePercent >= 0 ? '+' : ''}${stock.changePercent.toFixed(2)}%)
- 52-Week High: ${currency}${stock.high52w.toFixed(2)}
- 52-Week Low: ${currency}${stock.low52w.toFixed(2)}
- Market Cap: ${currency}${marketCapValue} ${marketCapUnit}
- P/E Ratio: ${stock.pe.toFixed(2)}
- EPS: ${currency}${stock.eps.toFixed(2)}
- Dividend Yield: ${dividendYield}%
- Trading Volume: ${stock.volume.toLocaleString()}

Please provide a detailed analysis and answer the user's question based on this data.
`;

    // Convert history to Gemini format
    const chatHistory = history?.map((msg: { role: string; content: string }) => ({
      role: msg.role === "assistant" ? "model" : "user",
      parts: [{ text: msg.content }],
    })) || [];

    const { text, usedModel } = await tryWithFallbacks(
      (modelName) => chatWithFallback(stockContext, question, chatHistory, modelName),
      "stock-chat"
    );

    return NextResponse.json({ 
      answer: text, 
      usedModel,
      stock: {
        symbol: stock.symbol,
        name: stock.name
      }
    });
  } catch (error) {
    console.error("Stock Chat API error:", error);
    return NextResponse.json(
      { error: "Failed to get AI response. Please try again.", details: (error as Error).message },
      { status: 500 }
    );
  }
}
