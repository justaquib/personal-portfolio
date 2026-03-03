// Dataset of Q&A about justaquib.com
export interface QAItem {
  question: string;
  answer: string;
  keywords: string[];
  category: string;
}

export const chatbotDataset: QAItem[] = [
  // About Aquib Shahbaz
  {
    question: "Who is Aquib Shahbaz?",
    answer: "Aquib Shahbaz is a Software Engineer and Creative Frontend Developer based in India. He has worked across India and the Middle East. He specializes in building smooth, scalable web apps with human-friendly design.",
    keywords: ["aquib", "shahbaz", "owner", "developer", "creator", "founder", "about", "who", "person", "engineer"],
    category: "about"
  },
  {
    question: "What is Aquib's email?",
    answer: "You can reach Aquib Shahbaz at developer@justaquib.com",
    keywords: ["email", "contact", "mail", "reach", "address"],
    category: "contact"
  },
  {
    question: "What are Aquib's interests?",
    answer: "Aquib is interested in designing user flows, polishing layouts, building side projects, exploring product ideas, and creating music playlists.",
    keywords: ["interests", "hobbies", "likes", "passion", "enjoys"],
    category: "about"
  },
  {
    question: "What is Aquib's profession?",
    answer: "Aquib Shahbaz is a Software Engineer and Creative Frontend Developer. His philosophy is building things that are clean, clever, and genuinely helpful.",
    keywords: ["profession", "job", "career", "work", "role", "title"],
    category: "about"
  },
  
  // Projects - Hello World
  {
    question: "What is Hello World project?",
    answer: "Hello World is a visually engaging entry point for the website with a typewriter effect using Framer Motion. It's a lightweight implementation that showcases animation skills.",
    keywords: ["hello world", "hello-world", "typewriter", "animation", "framer motion", "entry", "landing"],
    category: "projects"
  },
  {
    question: "Tell me about the Hello World prototype",
    answer: "Hello World uses React + Motion stack. It's a beautiful entry point with typewriter animation created using Framer Motion library.",
    keywords: ["hello world", "hello-world", "typewriter", "framer motion", "react"],
    category: "projects"
  },
  
  // Projects - Ping Pong
  {
    question: "What is Ping Pong game?",
    answer: "Ping Pong is a classic arcade game with an AI opponent, built using Canvas + React. It features real-time rendering at 60fps, collision physics, and predictive AI tracking.",
    keywords: ["ping pong", "ping-pong", "game", "arcade", "canvas", "ai opponent", "pong"],
    category: "projects"
  },
  {
    question: "Tell me about the Ping Pong prototype",
    answer: "Ping Pong is built with Canvas and React. It features a classic arcade game with an AI opponent, 60fps rendering, collision physics, and predictive AI that tracks the ball.",
    keywords: ["ping pong", "ping-pong", "game", "arcade", "canvas", "ai"],
    category: "projects"
  },
  
  // Projects - AI Document Intelligence Workspace
  {
    question: "What is AI Document Intelligence Workspace?",
    answer: "AI Document Intelligence Workspace is an AI-powered document reader supporting PDF, DOCX, XLSX, and CSV files. It features AI summarization and allows you to ask questions about your documents using Gemini AI.",
    keywords: ["doc reader", "doc-reader", "document", "pdf", "docx", "xlsx", "csv", "read", "upload", "summarize", "ai summary", "ai document intelligence workspace"],
    category: "projects"
  },
  {
    question: "Tell me about the AI Document Intelligence Workspace prototype",
    answer: "AI Document Intelligence Workspace uses GPT-Wrapper with Gemini AI integration. It can read PDF, DOCX, XLSX, and CSV files, provide AI summaries, and answer questions about the document content.",
    keywords: ["doc reader", "doc-reader", "document", "pdf", "ai", "gemini", "summarize", "ai document intelligence workspace"],
    category: "projects"
  },
  
  // Projects - PDF Generator
  {
    question: "What is PDF Generator?",
    answer: "PDF Generator is a tool to convert images, DOCX, XLSX, and CSV files to PDF. It features client-side processing with a modular converter architecture using jsPDF, mammoth, and xlsx libraries.",
    keywords: ["pdf generator", "pdf-generator", "pdf", "convert", "converter", "image", "docx", "xlsx", "csv"],
    category: "projects"
  },
  {
    question: "Tell me about the PDF Generator prototype",
    answer: "PDF Generator uses jsPDF, mammoth, and xlsx libraries. It converts images, DOCX, XLSX, and CSV files to PDF entirely on the client side with a modular architecture.",
    keywords: ["pdf generator", "pdf-generator", "convert", "jspdf", "xlsx"],
    category: "projects"
  },
  
  // Projects - AI Chatbot
  {
    question: "What is AI Chatbot?",
    answer: "AI Chatbot is a service chat bot with natural language processing capabilities. It features real-time messaging, intent recognition, and conversation context. Built with Socket and NLP.",
    keywords: ["ai chatbot", "ai-chatbot", "chatbot", "chat", "bot", "nlp", "natural language", "socket"],
    category: "projects"
  },
  {
    question: "Tell me about the AI Chatbot prototype",
    answer: "AI Chatbot uses Socket for real-time messaging and NLP for intent recognition. It can understand natural language and maintain conversation context.",
    keywords: ["ai chatbot", "ai-chatbot", "chatbot", "nlp", "socket", "real-time"],
    category: "projects"
  },
  
  // Projects - Doodle Predictor
  {
    question: "What is Doodle Predictor?",
    answer: "Doodle Predictor is a draw-and-guess AI using TensorFlow.js. It uses a CNN (Convolutional Neural Network) model trained on the QuickDraw dataset for real-time recognition of hand-drawn doodles.",
    keywords: ["doodle predictor", "doodle-predictor", "doodle", "draw", "guess", "tensorflow", "cnn", "machine learning", "ai", "quickdraw"],
    category: "projects"
  },
  {
    question: "Tell me about the Doodle Predictor prototype",
    answer: "Doodle Predictor uses Canvas and TensorFlow.js. It's a CNN-based machine learning model that recognizes hand-drawn doodles in real-time, trained on the QuickDraw dataset.",
    keywords: ["doodle predictor", "doodle-predictor", "tensorflow", "cnn", "canvas", "machine learning"],
    category: "projects"
  },
  
  // Projects - Code Playground
  {
    question: "What is Code Playground?",
    answer: "Code Playground is a live code editor with support for HTML, CSS, JS, PHP, Python, and SQL. It uses Monaco Editor (the same editor as VS Code) with backend execution for PHP, Python, and SQL.",
    keywords: ["code playground", "code-playground", "editor", "code", "html", "css", "javascript", "php", "python", "sql", "monaco", "live"],
    category: "projects"
  },
  {
    question: "Tell me about the Code Playground prototype",
    answer: "Code Playground features Monaco Editor (VS Code-like), live preview, and backend execution. It supports HTML, CSS, JavaScript, PHP, Python, and SQL with real-time results.",
    keywords: ["code playground", "code-playground", "monaco", "editor", "live preview", "execute"],
    category: "projects"
  },
  
  // Projects - TaskFlow
  {
    question: "What is TaskFlow?",
    answer: "TaskFlow is a task management application with persistent SQLite storage. It features full CRUD operations with both board and list views, and all data is stored persistently.",
    keywords: ["taskflow", "task", "tasks", "todo", "management", "board", "list", "sqlite", "database", "crud"],
    category: "projects"
  },
  {
    question: "Tell me about the TaskFlow prototype",
    answer: "TaskFlow uses SQLite with React. It's a task management app with full CRUD operations, supporting both board and list views with persistent data storage.",
    keywords: ["taskflow", "task", "sqlite", "database", "crud", "board"],
    category: "projects"
  },
  
  // Projects - Mood Journal
  {
    question: "What is Mood Journal?",
    answer: "Mood Journal is a daily emotion tracker with AI insights. It tracks your daily moods and uses Gemini AI to provide insights and visualize mood trends over time.",
    keywords: ["mood journal", "mood-journal", "mood", "emotion", "tracker", "journal", "ai", "insights", "feelings"],
    category: "projects"
  },
  {
    question: "Tell me about the Mood Journal prototype",
    answer: "Mood Journal uses AI (Gemini) and React wrapper. It's a daily emotion tracker that provides AI-powered insights and visualizes mood trends over time.",
    keywords: ["mood journal", "mood-journal", "mood", "emotion", "ai", "insights"],
    category: "projects"
  },
  
  // Projects - Stock Tracker
  {
    question: "What is Stock Tracker?",
    answer: "Stock Tracker is a real-time stock dashboard built with React Native. It features live price updates via WebSocket, portfolio management, and watchlists.",
    keywords: ["stock tracker", "stock-tracker", "stock", "stocks", "market", "react native", "realtime", "websocket", "portfolio", "watchlist"],
    category: "projects"
  },
  {
    question: "Tell me about the Stock Tracker prototype",
    answer: "Stock Tracker uses React Native for a mobile experience. It provides real-time stock price updates via WebSocket, with portfolio management and watchlist features.",
    keywords: ["stock tracker", "stock-tracker", "react native", "realtime", "websocket", "stocks"],
    category: "projects"
  },
  
  // Technical Stack
  {
    question: "What tech stack is used on this website?",
    answer: "The website uses Next.js (React framework), TypeScript, Tailwind CSS, Framer Motion for animations, TensorFlow.js for machine learning, Google Gemini AI, SQLite for database, Monaco Editor for code editing, and WebSocket for real-time features.",
    keywords: ["tech", "stack", "technology", "technologies", "used", "framework", "tools", "nextjs", "react", "typescript", "tailwind"],
    category: "tech"
  },
  {
    question: "What frameworks are used?",
    answer: "The website uses Next.js as the main framework, React for UI components, and Tailwind CSS for styling. Additional libraries include Framer Motion for animations, TensorFlow.js for ML, and Monaco Editor for code editing.",
    keywords: ["framework", "frameworks", "nextjs", "react", "tailwind", "framer motion"],
    category: "tech"
  },
  
  // General
  {
    question: "How many projects are on this website?",
    answer: "There are 10 projects/prototypes on justaquib.com: Hello World, Ping Pong, AI Document Intelligence Workspace, PDF Generator, AI Chatbot, Doodle Predictor, Code Playground, TaskFlow, Mood Journal, and Stock Tracker.",
    keywords: ["how many", "projects", "prototypes", "apps", "demos", "count", "number"],
    category: "general"
  },
  {
    question: "What projects are available?",
    answer: "The website has 10 projects: Hello World (animation), Ping Pong (game), AI Document Intelligence Workspace (AI document), PDF Generator (conversion), AI Chatbot (NLP), Doodle Predictor (ML), Code Playground (editor), TaskFlow (task management), Mood Journal (tracking), and Stock Tracker (finance).",
    keywords: ["projects", "prototypes", "apps", "demos", "available", "list", "what"],
    category: "general"
  },
  {
    question: "What can this website do?",
    answer: "justaquib.com showcases 10 creative web projects including a document AI reader, PDF converter, code editor, doodle recognizer, task manager, stock tracker, mood journal, chat bot, arcade game, and more. It's a portfolio demonstrating full-stack development skills.",
    keywords: ["website", "do", "can", "able", "what", "about"],
    category: "general"
  },
  {
    question: "Who built this website?",
    answer: "This website was built by Aquib Shahbaz, a Software Engineer and Creative Frontend Developer. It's his personal portfolio showcasing various web projects and prototypes.",
    keywords: ["built", "made", "created", "who", "developed", "designed"],
    category: "about"
  },
  {
    question: "How can I contact Aquib?",
    answer: "You can contact Aquib Shahbaz at developer@justaquib.com",
    keywords: ["contact", "email", "reach", "connect", "message", "how"],
    category: "contact"
  },
  
  // Fallback responses
  {
    question: "Hello",
    answer: "Hello! I'm the justaquib.com assistant. I can tell you about Aquib Shahbaz and his projects. What would you like to know?",
    keywords: ["hello", "hi", "hey", "greetings", "good morning", "good evening"],
    category: "greeting"
  },
  {
    question: "Help",
    answer: "I can help you learn about justaquib.com! Ask me about:\n\n• Aquib Shahbaz (who he is, his background, interests)\n• Any of the 10 projects (Hello World, Ping Pong, AI Document Intelligence Workspace, PDF Generator, AI Chatbot, Doodle Predictor, Code Playground, TaskFlow, Mood Journal, Stock Tracker)\n• The tech stack used on this website\n• How to contact Aquib\n\nWhat would you like to know?",
    keywords: ["help", "help me", "what can you do", "commands", "features", "assist", "support"],
    category: "help"
  }
];

// Get all unique categories
export const categories = [...new Set(chatbotDataset.map(item => item.category))];
