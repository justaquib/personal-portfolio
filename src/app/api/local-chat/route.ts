import { NextRequest, NextResponse } from "next/server";
import { findBestMatch, getSuggestedQuestions } from "@/utils/chatbot/matching";

export async function POST(request: NextRequest) {
  try {
    const { question, history } = await request.json();

    if (!question) {
      return NextResponse.json(
        { error: "Question is required" },
        { status: 400 }
      );
    }

    // Use the local matching engine
    const result = findBestMatch(question, history);

    return NextResponse.json({ 
      answer: result.answer, 
      confidence: result.confidence,
      usedModel: "local-matching"
    });

  } catch (error) {
    console.error("Local Chat API error:", error);
    return NextResponse.json(
      { error: "Failed to process request", details: (error as Error).message },
      { status: 500 }
    );
  }
}

export async function GET() {
  // Return suggested questions
  return NextResponse.json({
    suggestions: getSuggestedQuestions()
  });
}
