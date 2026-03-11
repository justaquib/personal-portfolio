import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextRequest, NextResponse } from "next/server";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

// List of models to try in order (primary first, then fallbacks)
const MODEL_FALLBACKS = [
  "gemini-3-pro",        // Best reasoning
  "gemini-2.5-pro",      // Best stability/context
  "gemini-3-flash",      // Balanced speed/smarts
  "gemini-2.5-flash",    // Your current baseline
  "gemini-2.5-flash-lite" // Highest quota/cheapest
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
  content: string,
  question: string,
  chatHistory: Array<{ role: string; parts: Array<{ text: string }> }>,
  modelName: string
): Promise<{ text: string; usedModel: string }> {
  const model = genAI.getGenerativeModel({ model: modelName });
  
  const chat = model.startChat({
    history: [
      {
        role: "user",
        parts: [{ text: `You are a helpful AI assistant that answers questions about documents. Here is the document content you will be answering questions about:\n\n${content}` }],
      },
      {
        role: "model",
        parts: [{ text: "I understand. I've analyzed the document and I'm ready to answer questions about it. Please ask me anything related to the document content." }],
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
    const { action, content, question, history, prompt } = await request.json();

    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        { error: "Gemini API key not configured" },
        { status: 500 }
      );
    }

    if (action === "summarize") {
      const prompt = `Please provide a comprehensive summary of the following document content. Structure your response with:
- A brief overview
- Key points (as bullet points)
- Main takeaways
- Any important conclusions

Document content:
${content}`;

      const { text, usedModel } = await tryWithFallbacks(
        (modelName) => generateWithFallback(prompt, modelName),
        "summarize"
      );

      return NextResponse.json({ summary: text, usedModel });
    }

    if (action === "enhance") {
      if (!prompt) {
        return NextResponse.json({ error: "Prompt is required for enhance action" }, { status: 400 });
      }

      const { text, usedModel } = await tryWithFallbacks(
        (modelName) => generateWithFallback(prompt, modelName),
        "enhance"
      );

      return NextResponse.json({ answer: text, usedModel });
    }

    if (action === "chat") {
      const chatHistory = history?.map((msg: { role: string; content: string }) => ({
        role: msg.role === "assistant" ? "model" : "user",
        parts: [{ text: msg.content }],
      })) || [];

      const { text, usedModel } = await tryWithFallbacks(
        (modelName) => chatWithFallback(content, question, chatHistory, modelName),
        "chat"
      );

      return NextResponse.json({ answer: text, usedModel });
    }

    if (action === "parse_resume") {
      const prompt = `You are a professional resume parser. Parse the following resume text and extract the information into a structured JSON format.

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation, no code blocks. The JSON must have this exact structure:

{
  "personalInfo": {
    "name": "Full Name",
    "email": "email@example.com",
    "phone": "phone number",
    "location": "city, state",
    "linkedin": "linkedin url",
    "portfolio": "portfolio url"
  },
  "summary": "professional summary text",
  "experience": [
    {
      "company": "Company Name",
      "role": "Job Title",
      "startDate": "Jan 2020",
      "endDate": "Dec 2023",
      "current": false,
      "description": "job responsibilities and achievements"
    }
  ],
  "education": [
    {
      "institution": "University Name",
      "degree": "Bachelor of Science",
      "field": "Computer Science",
      "graduationDate": "May 2020"
    }
  ],
  "skills": "comma, separated, skills",
  "projects": [
    {
      "name": "Project Name",
      "description": "project description",
      "technologies": "technologies used"
    }
  ]
}

Resume text to parse:
${content}

Return ONLY the JSON object. No markdown formatting. No explanations.`;

      const { text, usedModel } = await tryWithFallbacks(
        (modelName) => generateWithFallback(prompt, modelName),
        "parse_resume"
      );

      // Parse the JSON response
      try {
        // Try to extract JSON from the response (in case there's any extra text)
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const parsedData = JSON.parse(jsonMatch[0]);
          return NextResponse.json({ parsedData, usedModel });
        }
        // If no JSON found, try parsing the whole text
        const parsedData = JSON.parse(text);
        return NextResponse.json({ parsedData, usedModel });
      } catch (parseError) {
        console.error("Failed to parse JSON from AI response:", text);
        return NextResponse.json({ 
          error: "Failed to parse resume data", 
          rawResponse: text 
        }, { status: 500 });
      }
    }

    return NextResponse.json({ error: "Invalid action" }, { status: 400 });
  } catch (error) {
    console.error("Gemini API error:", error);
    return NextResponse.json(
      { error: "Failed to process request", details: (error as Error).message },
      { status: 500 }
    );
  }
}
