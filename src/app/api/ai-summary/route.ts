import { NextRequest, NextResponse } from 'next/server'
import { GoogleGenerativeAI } from '@google/generative-ai'

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '')

// Fallback summary when AI fails
function getFallbackSummary(data: any): string {
  const { jobTitle, skills, experience } = data
  
  const years = experience?.length > 0 
    ? `${experience.length} year${experience.length > 1 ? 's' : ''} of professional experience` 
    : 'professional experience'
  
  const skillList = skills 
    ? `skilled in ${skills.split(',').slice(0, 5).join(', ')}` 
    : ''

  return `${jobTitle || 'Professional'} with ${years}${skillList ? ` and ${skillList}` : ''}. Proven track record of delivering results and driving business growth.`
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { experience, education, skills, jobTitle, industry } = body

    // Build prompt for AI
    const prompt = `You are a professional resume writer. Generate a compelling professional summary for a resume based on the following information:

${jobTitle ? `Target Job Title: ${jobTitle}` : ''}
${industry ? `Industry: ${industry}` : ''}

Work Experience:
${experience?.map((exp: any) => `- ${exp.role} at ${exp.company}: ${exp.description}`).join('\n') || 'Not provided'}

Education:
${education?.map((edu: any) => `- ${edu.degree} in ${edu.field} from ${edu.institution}`).join('\n') || 'Not provided'}

Skills:
${skills || 'Not provided'}

Please generate a 2-3 sentence professional summary that:
1. Highlights years of experience and key expertise
2. Mentions specific skills and accomplishments
3. Is ATS-friendly (no special characters, plain text)
4. Is professional and impactful
5. Does not use bullet points or special formatting

Write only the summary, nothing else.`

    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-pro' })
    
    const result = await model.generateContent(prompt)
    const response = result.response
    const summary = response.text()

    return NextResponse.json({ summary: summary.trim() })
  } catch (error) {
    console.error('Error generating AI summary:', error)
    return NextResponse.json({ 
      error: 'Failed to generate summary'
    }, { status: 500 })
  }
}
