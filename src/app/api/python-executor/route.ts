import { NextRequest, NextResponse } from 'next/server';

// Python executor sandbox API
// Provides a simulated Python execution environment

interface PythonExecutionResult {
  output: string;
  error?: string;
  executionTime?: number;
}

// Security: List of dangerous Python functions/modules that should be blocked
const BLOCKED_PATTERNS = [
  // File operations
  'open(', 'file(', 'os.remove', 'os.unlink', 'os.rmdir', 'os.mkdir',
  'os.rename', 'os.chmod', 'os.chown', 'shutil.', 'pathlib.Path',
  
  // System operations
  'os.system', 'os.popen', 'subprocess.', 'commands.', 'exec(',
  'eval(', 'compile(', '__import__', 'importlib',
  
  // Network operations
  'socket.', 'urllib.', 'requests.', 'http.', 'ftplib.', 'smtplib.',
  'telnetlib.', 'poplib.', 'imaplib.', 'nntplib.',
  
  // Dangerous modules
  'pickle.', 'marshal.', 'shelve.', 'ctypes.', 'multiprocessing.',
  
  // System info
  'os.environ', 'os.getcwd', 'os.getuid', 'os.getpid', 'platform.',
  
  // Other dangerous
  'input(', 'raw_input(', 'breakpoint(',
];

// Check if code contains blocked patterns
function containsBlockedPatterns(code: string): string | null {
  const codeWithoutStrings = code
    .replace(/'''[\s\S]*?'''/g, '""')
    .replace(/"""[\s\S]*?"""/g, '""')
    .replace(/#[^\n]*/g, '');
  
  for (const pattern of BLOCKED_PATTERNS) {
    if (codeWithoutStrings.includes(pattern)) {
      return pattern;
    }
  }
  
  return null;
}

// Simulate Python execution
function simulatePythonExecution(code: string): PythonExecutionResult {
  const startTime = Date.now();
  
  // Security check
  const blockedPattern = containsBlockedPatterns(code);
  if (blockedPattern) {
    return {
      output: '',
      error: `Security Error: Pattern "${blockedPattern}" is not allowed in the sandbox.`,
    };
  }
  
  try {
    const output = simulatePythonParse(code);
    
    return {
      output,
      executionTime: Date.now() - startTime,
    };
  } catch (error) {
    return {
      output: '',
      error: `Python Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
    };
  }
}

// Parse multi-line structures (dicts, lists)
function parseMultiLineStructure(lines: string[], startIndex: number): { content: string; endIndex: number } {
  let content = '';
  let depth = 0;
  let started = false;
  let i = startIndex;
  
  while (i < lines.length) {
    const line = lines[i];
    
    for (const char of line) {
      if (char === '{' || char === '[' || char === '(') {
        if (!started) started = true;
        depth++;
      } else if (char === '}' || char === ']' || char === ')') {
        depth--;
      }
    }
    
    content += (content ? '\n' : '') + line.trim();
    
    if (started && depth === 0) {
      break;
    }
    i++;
  }
  
  return { content, endIndex: i };
}

// Basic Python parser/simulator
function simulatePythonParse(code: string): string {
  let output = '';
  const variables: Record<string, unknown> = {};
  const lines = code.split('\n');
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    // Skip empty lines and comments
    if (!line || line.startsWith('#')) {
      continue;
    }
    
    // Handle print statements
    if (line.startsWith('print(')) {
      const printMatch = line.match(/print\s*\((.*)\)$/);
      if (printMatch) {
        const content = printMatch[1].trim();
        output += evaluatePythonExpression(content, variables) + '\n';
      }
    }
    
    // Handle variable assignments
    else if (line.includes('=') && !line.startsWith('if') && !line.startsWith('elif') && !line.startsWith('while') && !line.startsWith('for')) {
      // Handle dictionary access assignments separately
      const dictAssignMatch = line.match(/^(\w+)\[['"](\w+)['"]\]\s*=\s*(.+)$/);
      if (dictAssignMatch) {
        const [, dictName, key, value] = dictAssignMatch;
        if (variables[dictName] && typeof variables[dictName] === 'object') {
          (variables[dictName] as Record<string, unknown>)[key] = evaluatePythonValue(value.trim(), variables);
        }
        continue;
      }
      
      const eqIndex = line.indexOf('=');
      const varName = line.substring(0, eqIndex).trim();
      let value = line.substring(eqIndex + 1).trim();
      
      // Check if this is a multi-line structure
      if ((value.startsWith('{') || value.startsWith('[')) && !isCompleteStructure(value)) {
        const { content, endIndex } = parseMultiLineStructure(lines, i);
        // Extract just the value part
        const eqPos = content.indexOf('=');
        value = content.substring(eqPos + 1).trim();
        i = endIndex;
      }
      
      if (varName && value && !value.startsWith('=')) {
        variables[varName] = evaluatePythonValue(value, variables);
      }
    }
    
    // Handle for loops
    else if (line.startsWith('for ')) {
      const forMatch = line.match(/for\s+(\w+)\s+in\s+(?:range\(([^)]+)\)|(\w+))/);
      if (forMatch) {
        const loopVar = forMatch[1];
        const rangeArgs = forMatch[2];
        const listVar = forMatch[3];
        
        let loopValues: (number | string)[] = [];
        
        if (rangeArgs) {
          const args = rangeArgs.split(',').map(a => parseInt(a.trim()));
          if (args.length === 1) {
            loopValues = Array.from({ length: args[0] }, (_, i) => i);
          } else if (args.length === 2) {
            loopValues = Array.from({ length: args[1] - args[0] }, (_, i) => args[0] + i);
          } else if (args.length === 3) {
            for (let j = args[0]; j < args[1]; j += args[2]) {
              loopValues.push(j);
            }
          }
        } else if (listVar && Array.isArray(variables[listVar])) {
          loopValues = variables[listVar] as (number | string)[];
        }
        
        // Find loop body (indented lines)
        const loopBody: string[] = [];
        let j = i + 1;
        while (j < lines.length && (lines[j].startsWith('    ') || lines[j].startsWith('\t') || lines[j].trim() === '')) {
          if (lines[j].trim()) {
            loopBody.push(lines[j]);
          }
          j++;
        }
        
        // Execute loop
        for (const val of loopValues) {
          const loopVars = { ...variables, [loopVar]: val };
          const loopOutput = executeBlock(loopBody, loopVars);
          output += loopOutput;
        }
        
        i = j - 1;
      }
    }
    
    // Handle function definitions (skip)
    else if (line.startsWith('def ')) {
      let j = i + 1;
      while (j < lines.length && (lines[j].startsWith('    ') || lines[j].startsWith('\t') || lines[j].trim() === '')) {
        j++;
      }
      i = j - 1;
    }
    
    // Handle class definitions (skip)
    else if (line.startsWith('class ')) {
      let j = i + 1;
      while (j < lines.length && (lines[j].startsWith('    ') || lines[j].startsWith('\t') || lines[j].trim() === '')) {
        j++;
      }
      i = j - 1;
    }
  }
  
  if (!output) {
    output = generatePythonSummary(variables);
  }
  
  return output;
}

// Check if a structure (dict/list) is complete
function isCompleteStructure(s: string): boolean {
  let depth = 0;
  let inString = false;
  let stringChar = '';
  
  for (let i = 0; i < s.length; i++) {
    const char = s[i];
    
    if (!inString) {
      if (char === '"' || char === "'") {
        inString = true;
        stringChar = char;
      } else if (char === '{' || char === '[' || char === '(') {
        depth++;
      } else if (char === '}' || char === ']' || char === ')') {
        depth--;
      }
    } else {
      if (char === stringChar && s[i-1] !== '\\') {
        inString = false;
      }
    }
  }
  
  return depth === 0 && !inString;
}

// Execute a block of code
function executeBlock(lines: string[], variables: Record<string, unknown>): string {
  let output = '';
  
  for (const line of lines) {
    const trimmedLine = line.trim();
    
    if (!trimmedLine || trimmedLine.startsWith('#')) {
      continue;
    }
    
    if (trimmedLine.startsWith('print(')) {
      const printMatch = trimmedLine.match(/print\s*\((.*)\)$/);
      if (printMatch) {
        const content = printMatch[1].trim();
        output += evaluatePythonExpression(content, variables) + '\n';
      }
    }
    else if (trimmedLine.includes('=') && !trimmedLine.startsWith('if')) {
      const eqIndex = trimmedLine.indexOf('=');
      const varName = trimmedLine.substring(0, eqIndex).trim();
      const value = trimmedLine.substring(eqIndex + 1).trim();
      
      if (varName && value && !value.startsWith('=')) {
        variables[varName] = evaluatePythonValue(value, variables);
      }
    }
  }
  
  return output;
}

// Evaluate Python expression
function evaluatePythonExpression(expr: string, variables: Record<string, unknown>): string {
  // Handle f-strings
  if (expr.startsWith('f"') || expr.startsWith("f'")) {
    const quoteChar = expr[1];
    let fstring = expr.slice(2);
    if (fstring.endsWith(quoteChar)) {
      fstring = fstring.slice(0, -1);
    }
    
    // Replace {expression} with evaluated values
    let result = '';
    let i = 0;
    while (i < fstring.length) {
      if (fstring[i] === '{') {
        let depth = 1;
        let j = i + 1;
        let inString = false;
        let stringChar = '';
        
        while (j < fstring.length && depth > 0) {
          const char = fstring[j];
          
          if (!inString) {
            if (char === '{') depth++;
            else if (char === '}') depth--;
            else if (char === '"' || char === "'") {
              inString = true;
              stringChar = char;
            }
          } else {
            if (char === stringChar && fstring[j-1] !== '\\') {
              inString = false;
            }
          }
          j++;
        }
        
        const innerExpr = fstring.slice(i + 1, j - 1);
        const evaluated = evaluatePythonValue(innerExpr, variables);
        result += evaluated !== undefined ? String(evaluated) : 'undefined';
        i = j;
      } else {
        result += fstring[i];
        i++;
      }
    }
    
    return result;
  }
  
  // Handle string concatenation
  if (expr.includes('+') && !expr.startsWith('"') && !expr.startsWith("'")) {
    const parts = splitByOperator(expr, '+');
    if (parts.length > 1) {
      return parts.map(p => evaluatePythonExpression(p.trim(), variables)).join('');
    }
  }
  
  // Handle comma-separated values
  if (expr.includes(',') && !expr.includes('(')) {
    const parts = splitByOperator(expr, ',');
    if (parts.length > 1) {
      return parts.map(p => evaluatePythonExpression(p.trim(), variables)).join(' ');
    }
  }
  
  return String(evaluatePythonValue(expr, variables));
}

// Split by operator respecting strings and brackets
function splitByOperator(expr: string, operator: string): string[] {
  const parts: string[] = [];
  let current = '';
  let depth = 0;
  let inString = false;
  let stringChar = '';
  
  for (let i = 0; i < expr.length; i++) {
    const char = expr[i];
    
    if (!inString) {
      if (char === '"' || char === "'") {
        inString = true;
        stringChar = char;
        current += char;
      } else if (char === '(' || char === '[' || char === '{') {
        depth++;
        current += char;
      } else if (char === ')' || char === ']' || char === '}') {
        depth--;
        current += char;
      } else if (char === operator && depth === 0) {
        parts.push(current);
        current = '';
      } else {
        current += char;
      }
    } else {
      current += char;
      if (char === stringChar && expr[i-1] !== '\\') {
        inString = false;
      }
    }
  }
  
  if (current) {
    parts.push(current);
  }
  
  return parts;
}

// Evaluate Python value
function evaluatePythonValue(value: string, variables: Record<string, unknown>): unknown {
  value = value.trim();
  
  if (!value) return '';
  
  // String with double quotes
  if (value.startsWith('"') && value.endsWith('"') && value.length >= 2) {
    return value.slice(1, -1);
  }
  
  // String with single quotes
  if (value.startsWith("'") && value.endsWith("'") && value.length >= 2) {
    return value.slice(1, -1);
  }
  
  // Number
  if (!isNaN(Number(value)) && value !== '') {
    return Number(value);
  }
  
  // Boolean
  if (value === 'True') return true;
  if (value === 'False') return false;
  if (value === 'None') return null;
  
  // List - handle multi-line
  if (value.startsWith('[')) {
    const content = extractBracketContent(value, '[', ']');
    if (!content.trim()) return [];
    const items = splitByOperator(content, ',');
    return items.map(item => evaluatePythonValue(item.trim(), variables));
  }
  
  // Dictionary - handle multi-line
  if (value.startsWith('{')) {
    const content = extractBracketContent(value, '{', '}');
    if (!content.trim()) return {};
    const obj: Record<string, unknown> = {};
    const pairs = splitByOperator(content, ',');
    for (const pair of pairs) {
      const colonIndex = pair.indexOf(':');
      if (colonIndex > 0) {
        const key = pair.substring(0, colonIndex).trim();
        const val = pair.substring(colonIndex + 1).trim();
        const cleanKey = evaluatePythonValue(key, variables);
        obj[String(cleanKey)] = evaluatePythonValue(val, variables);
      }
    }
    return obj;
  }
  
  // Dictionary access: var['key'] or var["key"]
  const dictAccessMatch = value.match(/^(\w+)\[['"](\w+)['"]\]$/);
  if (dictAccessMatch) {
    const [, varName, key] = dictAccessMatch;
    const dict = variables[varName];
    if (dict && typeof dict === 'object' && !Array.isArray(dict)) {
      return (dict as Record<string, unknown>)[key];
    }
    return undefined;
  }
  
  // Variable
  if (variables.hasOwnProperty(value)) {
    return variables[value];
  }
  
  // Method calls like ', '.join(list)
  const methodMatch = value.match(/^['"]([^'"]*)['"]\.join\(([^)]+)\)$/);
  if (methodMatch) {
    const separator = methodMatch[1];
    const listExpr = methodMatch[2].trim();
    const list = evaluatePythonValue(listExpr, variables);
    if (Array.isArray(list)) {
      return list.join(separator);
    }
    return String(list);
  }
  
  // Function calls
  if (value.includes('(') && value.includes(')')) {
    const funcMatch = value.match(/^(\w+)\(([^)]*)\)$/);
    if (funcMatch) {
      const funcName = funcMatch[1];
      const args = funcMatch[2];
      
      if (funcName === 'len') {
        const val = evaluatePythonValue(args.trim(), variables);
        if (Array.isArray(val) || typeof val === 'string') return val.length;
        if (typeof val === 'object' && val !== null) return Object.keys(val).length;
        return 0;
      }
      
      if (funcName === 'sum') {
        const val = evaluatePythonValue(args.trim(), variables);
        if (Array.isArray(val)) {
          return val.reduce((a: number, b: number) => a + (typeof b === 'number' ? b : 0), 0);
        }
        return 0;
      }
      
      if (funcName === 'min' || funcName === 'max') {
        const val = evaluatePythonValue(args.trim(), variables);
        if (Array.isArray(val) && val.length > 0) {
          const numVals = val.filter(v => typeof v === 'number') as number[];
          return funcName === 'min' ? Math.min(...numVals) : Math.max(...numVals);
        }
        return val;
      }
      
      if (funcName === 'str') {
        return String(evaluatePythonValue(args.trim(), variables));
      }
      
      if (funcName === 'int') {
        return parseInt(String(evaluatePythonValue(args.trim(), variables)));
      }
      
      if (funcName === 'float') {
        return parseFloat(String(evaluatePythonValue(args.trim(), variables)));
      }
      
      if (funcName === 'type') {
        const val = evaluatePythonValue(args.trim(), variables);
        if (Array.isArray(val)) return '<class \'list\'>';
        if (typeof val === 'object' && val !== null) return '<class \'dict\'>';
        if (typeof val === 'string') return '<class \'str\'>';
        if (typeof val === 'number') return Number.isInteger(val) ? '<class \'int\'>' : '<class \'float\'>';
        return '<class \'NoneType\'>';
      }
      
      if (funcName === 'range') {
        const argList = splitByOperator(args, ',').map(a => parseInt(a.trim()));
        if (argList.length === 1) {
          return Array.from({ length: argList[0] }, (_, i) => i);
        } else if (argList.length === 2) {
          return Array.from({ length: argList[1] - argList[0] }, (_, i) => argList[0] + i);
        } else if (argList.length === 3) {
          const result: number[] = [];
          for (let j = argList[0]; j < argList[1]; j += argList[2]) {
            result.push(j);
          }
          return result;
        }
        return [];
      }
    }
  }
  
  return value;
}

// Extract content between matching brackets
function extractBracketContent(s: string, openChar: string, closeChar: string): string {
  if (!s.startsWith(openChar)) return '';
  
  let depth = 0;
  let start = 0;
  
  for (let i = 0; i < s.length; i++) {
    if (s[i] === openChar) {
      if (depth === 0) start = i + 1;
      depth++;
    } else if (s[i] === closeChar) {
      depth--;
      if (depth === 0) {
        return s.slice(start, i);
      }
    }
  }
  
  return s.slice(1); // Fallback
}

// Generate summary output
function generatePythonSummary(variables: Record<string, unknown>): string {
  let output = '';
  
  const varNames = Object.keys(variables);
  if (varNames.length > 0) {
    output += 'Variables:\n';
    for (const name of varNames) {
      const val = variables[name];
      if (Array.isArray(val)) {
        output += `  ${name} = [${val.join(', ')}]\n`;
      } else if (typeof val === 'object' && val !== null) {
        output += `  ${name} = ${JSON.stringify(val)}\n`;
      } else {
        output += `  ${name} = ${val}\n`;
      }
    }
  }
  
  return output || 'Code executed successfully (no output)';
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { code } = body;
    
    if (!code || typeof code !== 'string') {
      return NextResponse.json(
        { error: 'Invalid request: code is required' },
        { status: 400 }
      );
    }
    
    if (code.length > 50000) {
      return NextResponse.json(
        { error: 'Code size exceeds maximum limit (50KB)' },
        { status: 400 }
      );
    }
    
    const result = simulatePythonExecution(code);
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('Python execution error:', error);
    return NextResponse.json(
      { error: 'Internal server error during Python execution' },
      { status: 500 }
    );
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}
