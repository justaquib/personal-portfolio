import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextRequest, NextResponse } from "next/server";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

export async function POST(request: NextRequest) {
  try {
    const { action, content, question, history } = await request.json();

    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        { error: "Gemini API key not configured" },
        { status: 500 }
      );
    }

    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

    if (action === "summarize") {
      const prompt = `Please provide a comprehensive summary of the following document content. Structure your response with:
- A brief overview
- Key points (as bullet points)
- Main takeaways
- Any important conclusions

Document content:
${content}`;

      const result = await model.generateContent(prompt);
      const response = await result.response;
      const text = response.text();

      return NextResponse.json({ summary: text });
    }

    if (action === "chat") {
      const chatHistory = history?.map((msg: { role: string; content: string }) => ({
        role: msg.role === "assistant" ? "model" : "user",
        parts: [{ text: msg.content }],
      })) || [];

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
      const text = response.text();

      return NextResponse.json({ answer: text });
    }

    return NextResponse.json({ error: "Invalid action" }, { status: 400 });
  } catch (error) {
    console.error("Gemini API error:", error);
    return NextResponse.json(
      { error: "Failed to process request" },
      { status: 500 }
    );
  }
}
