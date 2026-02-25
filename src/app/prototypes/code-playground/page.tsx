'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Editor from '@monaco-editor/react';
import { 
  Play, 
  RotateCcw, 
  ChevronDown, 
  ChevronUp, 
  Terminal, 
  Code2,
  FileCode,
  FileType,
  Braces,
  Home,
  X,
  AlertCircle,
  CheckCircle,
  Info,
  Loader2,
  Database,
  TerminalSquare
} from 'lucide-react';

// Default code templates
const defaultCodes = {
  html: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>My Page</title>
</head>
<body>
  <div class="container">
    <h1>Welcome to Code Playground</h1>
    <p>Start editing to see your changes live!</p>
    <button id="clickMe">Click Me</button>
  </div>
</body>
</html>`,
  css: `* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
}

.container {
  text-align: center;
  padding: 2rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

h1 {
  color: #e94560;
  margin-bottom: 1rem;
  font-size: 2.5rem;
}

p {
  color: #a8a8b3;
  margin-bottom: 1.5rem;
}

button {
  background: linear-gradient(135deg, #e94560, #ff6b6b);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  transition: transform 0.2s, box-shadow 0.2s;
}

button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(233, 69, 96, 0.4);
}`,
  javascript: `// JavaScript code
document.addEventListener('DOMContentLoaded', function() {
  const button = document.getElementById('clickMe');
  let clickCount = 0;
  
  button.addEventListener('click', function() {
    clickCount++;
    console.log('Button clicked ' + clickCount + ' time(s)');
    
    // Create a nice animation effect
    button.style.transform = 'scale(0.95)';
    setTimeout(() => {
      button.style.transform = 'scale(1)';
    }, 100);
    
    // Show a message
    const container = document.querySelector('.container');
    let message = document.getElementById('clickMessage');
    if (!message) {
      message = document.createElement('p');
      message.id = 'clickMessage';
      message.style.color = '#4ade80';
      message.style.marginTop = '1rem';
      container.appendChild(message);
    }
    message.textContent = 'You clicked ' + clickCount + ' times! Keep going!';
  });
  
  console.log('JavaScript loaded successfully!');
});`,
  php: `<?php
// PHP Code Playground
$message = "Hello from PHP!";
$numbers = [1, 2, 3, 4, 5];
$sum = array_sum($numbers);

echo "<h2>$message</h2>";
echo "<p>The sum of numbers (1-5) is: $sum</p>";

// For loop example
echo "<h3>Number List (for):</h3><ul>";
for ($i = 0; $i < count($numbers); $i++) { echo "<li>Number: " . $numbers[$i] . "</li>"; }
echo "</ul>";

// Foreach loop example
echo "<h3>Number List (foreach):</h3><ul>";
foreach ($numbers as $num) { echo "<li>Number: $num</li>"; }
echo "</ul>";

// Date example
echo "<p>Current date and time: " . date('Y-m-d H:i:s') . "</p>";

// Associative array
$user = ['name' => 'John Doe', 'email' => 'john@example.com', 'role' => 'Developer'];

echo "<h3>User Info:</h3>";
echo "<p>Name: " . $user['name'] . "</p>";
echo "<p>Email: " . $user['email'] . "</p>";
echo "<p>Role: " . $user['role'] . "</p>";
?>`,
  python: `# Python Code Playground
# Write your Python code here

# Variables and basic operations
message = "Hello from Python!"
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)

print(message)
print(f"Numbers: {numbers}")
print(f"Sum of numbers: {total}")

# Dictionary example
user = {
    "name": "John Doe",
    "email": "john@example.com",
    "age": 28,
    "skills": ["Python", "JavaScript", "SQL"]
}

print(f"\\nUser Info:")
print(f"Name: {user['name']}")
print(f"Email: {user['email']}")
print(f"Age: {user['age']}")
print(f"Skills: {', '.join(user['skills'])}")

# Loop example
print("\\nCounting from 1 to 5:")
for i in range(1, 6):
    print(f"  Count: {i}")

# List comprehension
squares = [x**2 for x in range(1, 6)]
print(f"\\nSquares of 1-5: {squares}")`,
  sql: `-- SQL Playground (SQLite-like syntax)
-- Sample tables: users, products, orders, employees

SHOW TABLES;

SELECT * FROM users;

SELECT name, email, country FROM users WHERE country = 'USA';

SELECT category, COUNT(*) as count FROM products GROUP BY category;

SELECT COUNT(*) as total_orders, SUM(total) as total_revenue, AVG(total) as avg_order_value FROM orders;

SELECT department, COUNT(*) as employees, AVG(salary) as avg_salary FROM employees GROUP BY department ORDER BY avg_salary DESC;

SELECT * FROM products WHERE price > 100;

UPDATE users SET age = 30 WHERE name = 'John Doe';

SELECT * FROM users;`
};

// Tab configuration
const tabs = [
  { id: 'html', label: 'HTML', icon: FileType, language: 'html' },
  { id: 'css', label: 'CSS', icon: FileCode, language: 'css' },
  { id: 'javascript', label: 'JavaScript', icon: Braces, language: 'javascript' },
  { id: 'php', label: 'PHP', icon: Code2, language: 'php' },
  { id: 'python', label: 'Python', icon: TerminalSquare, language: 'python' },
  { id: 'sql', label: 'SQL', icon: Database, language: 'sql' },
];

// Console log type
interface ConsoleLog {
  id: number;
  type: 'log' | 'error' | 'warn' | 'info';
  message: string;
  timestamp: Date;
}

// Backend languages that need server execution
const backendLanguages = ['php', 'python', 'sql'];

export default function CodePlayground() {
  // State management
  const router = useRouter();
  const [activeTab, setActiveTab] = useState('html');
  const [codes, setCodes] = useState(defaultCodes);
  const [consoleLogs, setConsoleLogs] = useState<ConsoleLog[]>([]);
  const [isConsoleOpen, setIsConsoleOpen] = useState(true);
  const [isConsoleMinimized, setIsConsoleMinimized] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [splitPosition, setSplitPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const [backendOutput, setBackendOutput] = useState<string>('');
  
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const dragRef = useRef<{ startX: number; startPos: number } | null>(null);
  const logIdRef = useRef(0);

  // Add console log
  const addLog = useCallback((type: ConsoleLog['type'], message: string) => {
    const newLog: ConsoleLog = {
      id: ++logIdRef.current,
      type,
      message,
      timestamp: new Date()
    };
    setConsoleLogs(prev => [...prev, newLog]);
  }, []);

  // Clear console
  const clearConsole = useCallback(() => {
    setConsoleLogs([]);
  }, []);

  // Handle code change
  const handleCodeChange = useCallback((value: string | undefined) => {
    if (value !== undefined) {
      setCodes(prev => ({ ...prev, [activeTab]: value }));
    }
  }, [activeTab]);

  // Generate combined HTML for iframe
  const generatePreviewHTML = useCallback(() => {
    const htmlCode = codes.html;
    const cssCode = codes.css;
    const jsCode = codes.javascript;
    
    // Inject CSS into HTML
    let combinedHTML = htmlCode;
    
    // Add style tag if not present
    if (cssCode.trim()) {
      const styleTag = `<style>${cssCode}</style>`;
      if (combinedHTML.includes('</head>')) {
        combinedHTML = combinedHTML.replace('</head>', `${styleTag}</head>`);
      } else if (combinedHTML.includes('<body>')) {
        combinedHTML = combinedHTML.replace('<body>', `<head>${styleTag}</head><body>`);
      } else {
        combinedHTML = `${styleTag}${combinedHTML}`;
      }
    }
    
    // Add script tag for JavaScript
    if (jsCode.trim()) {
      const scriptTag = `<script>
        // Override console methods to send messages to parent
        (function() {
          const originalConsole = {
            log: console.log,
            error: console.error,
            warn: console.warn,
            info: console.info
          };
          
          function sendToParent(type, args) {
            const message = args.map(arg => {
              if (typeof arg === 'object') {
                try {
                  return JSON.stringify(arg, null, 2);
                } catch (e) {
                  return String(arg);
                }
              }
              return String(arg);
            }).join(' ');
            
            window.parent.postMessage({
              type: 'console',
              logType: type,
              message: message
            }, '*');
          }
          
          console.log = function(...args) {
            originalConsole.log.apply(console, args);
            sendToParent('log', args);
          };
          
          console.error = function(...args) {
            originalConsole.error.apply(console, args);
            sendToParent('error', args);
          };
          
          console.warn = function(...args) {
            originalConsole.warn.apply(console, args);
            sendToParent('warn', args);
          };
          
          console.info = function(...args) {
            originalConsole.info.apply(console, args);
            sendToParent('info', args);
          };
          
          // Catch errors
          window.onerror = function(msg, url, line, col, error) {
            sendToParent('error', ['Error: ' + msg + ' (line ' + line + ')']);
            return false;
          };
        })();
        
        ${jsCode}
      </script>`;
      
      if (combinedHTML.includes('</body>')) {
        combinedHTML = combinedHTML.replace('</body>', `${scriptTag}</body>`);
      } else {
        combinedHTML = `${combinedHTML}${scriptTag}`;
      }
    }
    
    return combinedHTML;
  }, [codes]);

  // Run backend code (PHP, Python, SQL)
  const runBackendCode = useCallback(async () => {
    setIsRunning(true);
    const language = activeTab;
    addLog('info', `Running ${language.toUpperCase()} code...`);
    
    try {
      const endpoint = `/api/${language}-executor`;
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code: codes[language as keyof typeof codes] }),
      });
      
      const result = await response.json();
      
      if (result.error) {
        addLog('error', `${language.toUpperCase()} Error: ${result.error}`);
        setBackendOutput(`<div class="error">${result.error}</div>`);
      } else {
        addLog('log', `${language.toUpperCase()} executed successfully`);
        
        // Format output based on language
        if (language === 'sql') {
          // SQL output is already formatted as a table
          setBackendOutput(`<pre class="sql-output">${result.output || 'Query executed successfully'}</pre>`);
        } else if (language === 'python') {
          // Python output
          setBackendOutput(`<pre class="python-output">${result.output || 'Code executed successfully'}</pre>`);
        } else {
          // PHP output (HTML)
          setBackendOutput(result.output || '');
        }
      }
    } catch (error) {
      addLog('error', `Failed to execute ${language.toUpperCase()}: ${error}`);
      setBackendOutput(`<div class="error">Failed to execute code: ${error}</div>`);
    } finally {
      setIsRunning(false);
    }
  }, [activeTab, codes, addLog]);

  // Run code (frontend or backend)
  const runCode = useCallback(() => {
    clearConsole();
    addLog('info', 'Running code...');
    
    if (backendLanguages.includes(activeTab)) {
      runBackendCode();
    } else {
      // Run frontend code
      const previewHTML = generatePreviewHTML();
      if (iframeRef.current) {
        iframeRef.current.srcdoc = previewHTML;
      }
      setTimeout(() => {
        addLog('log', 'Frontend code rendered successfully');
      }, 100);
    }
  }, [activeTab, generatePreviewHTML, runBackendCode, clearConsole, addLog]);

  // Reset code
  const resetCode = useCallback(() => {
    setCodes(defaultCodes);
    setBackendOutput('');
    clearConsole();
    addLog('info', 'Code reset to default');
  }, [clearConsole, addLog]);

  // Listen for console messages from iframe
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.data && event.data.type === 'console') {
        addLog(event.data.logType, event.data.message);
      }
    };
    
    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [addLog]);

  // Handle split pane dragging
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    dragRef.current = { startX: e.clientX, startPos: splitPosition };
  }, [splitPosition]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging || !dragRef.current) return;
    
    const containerWidth = window.innerWidth;
    const deltaX = e.clientX - dragRef.current.startX;
    const deltaPercent = (deltaX / containerWidth) * 100;
    const newPosition = Math.max(20, Math.min(80, dragRef.current.startPos + deltaPercent));
    setSplitPosition(newPosition);
  }, [isDragging]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    dragRef.current = null;
  }, []);

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Generate output HTML for backend languages
  const generateBackendOutputHTML = useCallback(() => {
    const baseStyles = `
      body {
        font-family: 'Fira Code', 'Cascadia Code', Consolas, monospace;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e4e4e7;
        padding: 20px;
        min-height: 100vh;
        margin: 0;
      }
      h2, h3 { color: #e94560; margin: 1rem 0 0.5rem; }
      p { color: #a8a8b3; margin: 0.5rem 0; }
      ul { list-style: none; padding: 0; }
      li { 
        padding: 5px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
      }
      pre {
        background: rgba(0, 0, 0, 0.3);
        padding: 15px;
        border-radius: 8px;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
        font-size: 13px;
        line-height: 1.5;
      }
      .error {
        color: #ff6b6b;
        background: rgba(255, 107, 107, 0.1);
        padding: 15px;
        border-radius: 8px;
        border-left: 3px solid #ff6b6b;
      }
      .sql-output, .python-output {
        color: #4ade80;
      }
    `;
    
    return `<!DOCTYPE html>
      <html>
      <head>
        <style>${baseStyles}</style>
      </head>
      <body>
        ${backendOutput}
      </body>
      </html>`;
  }, [backendOutput]);

  // Get log icon
  const getLogIcon = (type: ConsoleLog['type']) => {
    switch (type) {
      case 'error': return <AlertCircle className="w-4 h-4 text-red-400" />;
      case 'warn': return <AlertCircle className="w-4 h-4 text-yellow-400" />;
      case 'info': return <Info className="w-4 h-4 text-blue-400" />;
      default: return <CheckCircle className="w-4 h-4 text-green-400" />;
    }
  };

  // Get output label based on active tab
  const getOutputLabel = () => {
    switch (activeTab) {
      case 'php': return 'PHP Output';
      case 'python': return 'Python Output';
      case 'sql': return 'SQL Results';
      default: return 'Live Preview';
    }
  };

  return (
    <div className="h-screen bg-[#0d0d0d] text-white flex flex-col overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 bg-[#161616] border-b border-[#2a2a2a]">
        <div className="flex items-center gap-3">
          <Code2 className="w-6 h-6 text-[#e94560]" />
          <h1 className="text-lg font-semibold">Code Playground</h1>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={runCode}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-[#e94560] to-[#ff6b6b] rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {isRunning ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            <span>Run</span>
          </button>
          <button
            onClick={resetCode}
            className="flex items-center gap-2 px-4 py-2 bg-[#2a2a2a] rounded-lg hover:bg-[#3a3a3a] transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset</span>
          </button>
          <button
            onClick={() => router.push('/prototypes')}
            className="flex items-center gap-2 px-4 py-2 bg-[#2a2a2a] rounded-lg hover:bg-[#3a3a3a] transition-colors"
          >
            <Home className="w-4 h-4" />
            <span>Home</span>
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Editor Panel */}
        <div 
          className="flex flex-col bg-[#1a1a1a] overflow-hidden"
          style={{ width: `${splitPosition}%` }}
        >
          {/* Tabs */}
          <div className="flex items-center bg-[#161616] border-b border-[#2a2a2a] overflow-x-auto">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 text-sm transition-colors border-b-2 whitespace-nowrap ${
                    activeTab === tab.id
                      ? 'text-white border-[#e94560] bg-[#1a1a1a]'
                      : 'text-gray-400 border-transparent hover:text-gray-200 hover:bg-[#1a1a1a]/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Monaco Editor */}
          <div className="flex-1 overflow-hidden">
            <Editor
              height="100%"
              language={tabs.find(t => t.id === activeTab)?.language || 'html'}
              value={codes[activeTab as keyof typeof codes]}
              onChange={handleCodeChange}
              theme="vs-dark"
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                fontFamily: "'Fira Code', 'Cascadia Code', Consolas, monospace",
                fontLigatures: true,
                lineNumbers: 'on',
                roundedSelection: true,
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 2,
                wordWrap: 'on',
                padding: { top: 10 },
                scrollbar: {
                  vertical: 'auto',
                  horizontal: 'auto',
                  verticalScrollbarSize: 10,
                  horizontalScrollbarSize: 10,
                },
                renderLineHighlight: 'all',
                cursorBlinking: 'smooth',
                cursorSmoothCaretAnimation: 'on',
                smoothScrolling: true,
                bracketPairColorization: { enabled: true },
              }}
            />
          </div>
        </div>

        {/* Resizer */}
        <div
          className={`w-1 bg-[#2a2a2a] hover:bg-[#e94560] cursor-col-resize transition-colors ${
            isDragging ? 'bg-[#e94560]' : ''
          }`}
          onMouseDown={handleMouseDown}
        />

        {/* Output Panel */}
        <div 
          className="flex flex-col bg-[#0d0d0d] overflow-hidden"
          style={{ width: `${100 - splitPosition}%` }}
        >
          {/* Output Header */}
          <div className="flex items-center justify-between px-4 py-2 bg-[#161616] border-b border-[#2a2a2a]">
            <span className="text-sm font-medium text-gray-300">Output Preview</span>
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">
                {getOutputLabel()}
              </span>
            </div>
          </div>

          {/* Iframe Container */}
          <div className="flex-1 bg-white overflow-hidden">
            <iframe
              ref={iframeRef}
              title="Preview"
              sandbox="allow-scripts allow-modals"
              className="w-full h-full border-0"
              srcDoc={backendLanguages.includes(activeTab) ? generateBackendOutputHTML() : generatePreviewHTML()}
            />
          </div>
        </div>
      </div>

      {/* Console Drawer */}
      <div 
        className={`bg-[#161616] border-t border-[#2a2a2a] transition-all duration-300 ${
          isConsoleMinimized ? 'h-10' : 'h-48'
        } ${!isConsoleOpen ? 'h-0 border-0' : ''}`}
      >
        {/* Console Header */}
        <div 
          className="flex items-center justify-between px-4 py-2 cursor-pointer hover:bg-[#1a1a1a]"
          onClick={() => setIsConsoleMinimized(!isConsoleMinimized)}
        >
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-[#e94560]" />
            <span className="text-sm font-medium">Console</span>
            {consoleLogs.length > 0 && (
              <span className="px-2 py-0.5 text-xs bg-[#2a2a2a] rounded-full">
                {consoleLogs.length}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                clearConsole();
              }}
              className="px-2 py-1 text-xs text-gray-400 hover:text-white transition-colors"
            >
              Clear
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsConsoleOpen(false);
              }}
              className="p-1 hover:bg-[#2a2a2a] rounded transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
            {isConsoleMinimized ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </div>
        </div>

        {/* Console Content */}
        {!isConsoleMinimized && (
          <div className="flex-1 overflow-auto px-4 pb-2 font-mono text-sm">
            {consoleLogs.length === 0 ? (
              <div className="text-gray-500 text-center py-4">
                No logs yet. Run your code to see output.
              </div>
            ) : (
              <div className="space-y-1">
                {consoleLogs.map((log) => (
                  <div 
                    key={log.id}
                    className={`flex items-start gap-2 py-1 px-2 rounded ${
                      log.type === 'error' 
                        ? 'bg-red-500/10' 
                        : log.type === 'warn' 
                        ? 'bg-yellow-500/10' 
                        : ''
                    }`}
                  >
                    {getLogIcon(log.type)}
                    <span className={`flex-1 ${
                      log.type === 'error' 
                        ? 'text-red-300' 
                        : log.type === 'warn' 
                        ? 'text-yellow-300' 
                        : log.type === 'info'
                        ? 'text-blue-300'
                        : 'text-gray-300'
                    }`}>
                      {log.message}
                    </span>
                    <span className="text-xs text-gray-500">
                      {log.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Console Toggle Button (when closed) */}
      {!isConsoleOpen && (
        <button
          onClick={() => {
            setIsConsoleOpen(true);
            setIsConsoleMinimized(false);
          }}
          className="fixed bottom-4 right-4 flex items-center gap-2 px-4 py-2 bg-[#2a2a2a] rounded-lg hover:bg-[#3a3a3a] transition-colors shadow-lg"
        >
          <Terminal className="w-4 h-4 text-[#e94560]" />
          <span className="text-sm">Show Console</span>
          {consoleLogs.length > 0 && (
            <span className="px-2 py-0.5 text-xs bg-[#e94560] rounded-full">
              {consoleLogs.length}
            </span>
          )}
        </button>
      )}
    </div>
  );
}
