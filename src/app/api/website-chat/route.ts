import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextRequest, NextResponse } from "next/server";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

// Website knowledge base - all information about the website
const WEBSITE_CONTEXT = `
You are a helpful AI assistant for justaquib.com - Aquib Shahbaz's personal portfolio website.

## ABOUT THE OWNER
- Name: Aquib Shahbaz
- Email: developer@justaquib.com
- Location: Has worked across India and the Middle East
- Profession: Software Engineer and Creative Frontend Developer
- Focus: Building smooth, scalable web apps with human-friendly design
- Philosophy: Loves building things that are clean, clever, and genuinely helpful
- Interests: Designing user flows, polishing layouts, side projects, product ideas, music playlists

## WEBSITE PROJECTS/PROTOTYPES (10 total)

1. **Hello World** (hello-world)
   - Stack: React + Motion
   - Description: A visually engaging entry point with typewriter effect using Framer Motion
   - Features: Typewriter animation, lightweight implementation

2. **Ping Pong** (ping-pong)
   - Stack: Canvas + React
   - Description: Classic ping pong arcade game with AI opponent
   - Features: Real-time rendering at 60fps, collision physics, predictive AI tracking

3. **AI Document Intelligence Workspace** (doc-reader)
   - Stack: GPT-Wrapper
   - Description: AI-powered document reader supporting PDF, DOCX, XLSX, CSV
   - Features: AI summarization, Q&A about documents, Gemini AI integration

4. **PDF Generator** (pdf-generator)
   - Stack: jsPDF + mammoth + xlsx
   - Description: Convert images, DOCX, XLSX, CSV to PDF
   - Features: Client-side processing, modular converter architecture

5. **AI Chatbot** (ai-chatbot)
   - Stack: Socket + NLP
   - Description: Service chat bot with natural language processing
   - Features: Real-time messaging, intent recognition, conversation context

6. **Doodle Predictor** (doodle-predictor)
   - Stack: Canvas + TensorFlow.js
   - Description: Draw-and-guess AI using TensorFlow.js
   - Features: CNN model, real-time recognition, QuickDraw dataset

7. **Code Playground** (code-playground)
   - Stack: Monaco + Sandbox
   - Description: Live code editor with HTML/CSS/JS/PHP support
   - Features: Monaco Editor (VS Code-like), backend execution for PHP/Python/SQL, live preview

8. **TaskFlow** (taskflow)
   - Stack: SQLite + React
   - Description: Task management with persistent SQLite storage
   - Features: CRUD operations, board and list views, persistent data

9. **Mood Journal** (mood-journal)
   - Stack: AI, Wrapper
   - Description: Daily emotion tracker with AI insights
   - Features: Mood tracking, Gemini AI for insights, trend visualization

10. **Stock Tracker** (stock-tracker)
    - Stack: React Native
    - Description: Real-time stock dashboard
    - Features: Live price updates via WebSocket, portfolio management, watchlists

## TECHNICAL STACK USED ON WEBSITE
- Next.js (React framework)
- TypeScript
- Tailwind CSS
- Framer Motion (animations)
- TensorFlow.js (machine learning)
- Google Gemini AI
- SQLite (database)
- Monaco Editor (code editing)
- WebSocket (real-time)

## ANSWERING RULES
- Only answer questions related to this website, Aquib Shahbaz, or the projects listed above
- If asked about topics unrelated to this website, politely redirect to the website
- Be helpful, concise, and friendly
- Use the information from the context above to answer questions
- Do not make up information that isn't provided above
`;

// List of models to try in order (primary first, then fallbacks)
const MODEL_FALLBACKS = [
  "gemini-3-pro",
  "gemini-2.5-pro",
  "gemini-3-flash",
  "gemini-2.5-flash",
  "gemini-2.0-flash-exp"
];

// Helper function to try chat with model fallback
async function chatWithFallback(
  question: string,
  chatHistory: Array<{ role: string; parts: Array<{ text: string }> }>,
  modelName: string
): Promise<{ text: string; usedModel: string }> {
  const model = genAI.getGenerativeModel({ model: modelName });
  
  const chat = model.startChat({
    history: [
      {
        role: "user",
        parts: [{ text: WEBSITE_CONTEXT }],
      },
      {
        role: "model",
        parts: [{ text: "I've analyzed all the information about justaquib.com. I'm ready to answer questions about Aquib Shahbaz's website, projects, and background. What would you like to know?" }],
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
  try {
    const { question, history } = await request.json();

    if (!question) {
      return NextResponse.json(
        { error: "Question is required" },
        { status: 400 }
      );
    }

    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        { error: "Gemini API key not configured" },
        { status: 500 }
      );
    }

    const chatHistory = history?.map((msg: { role: string; content: string }) => ({
      role: msg.role === "assistant" ? "model" : "user",
      parts: [{ text: msg.content }],
    })) || [];

    const { text, usedModel } = await tryWithFallbacks(
      (modelName) => chatWithFallback(question, chatHistory, modelName),
      "website-chat"
    );

    return NextResponse.json({ answer: text, usedModel });

  } catch (error) {
    console.error("Website Chat API error:", error);
    return NextResponse.json(
      { error: "Failed to process request", details: (error as Error).message },
      { status: 500 }
    );
  }
}
