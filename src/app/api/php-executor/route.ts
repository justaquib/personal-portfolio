import { NextRequest, NextResponse } from 'next/server';

// PHP executor sandbox API

interface PHPExecutionResult {
  output: string;
  error?: string;
  executionTime?: number;
}

// Security: List of dangerous PHP functions that should be blocked
const BLOCKED_FUNCTIONS = [
  'exec', 'shell_exec', 'system', 'passthru', 'popen', 'proc_open',
  'pcntl_exec', 'show_source', 'highlight_file', 'file_get_contents',
  'file_put_contents', 'fopen', 'fwrite', 'fread', 'fputs',
  'move_uploaded_file', 'copy', 'rename', 'unlink', 'rmdir', 'mkdir',
  'chmod', 'chown', 'chgrp', 'touch', 'symlink', 'link',
  'fsockopen', 'pfsockopen', 'socket_create', 'curl_init', 'curl_exec',
  'stream_socket_client', 'stream_socket_server', 'ftp_connect',
  'eval', 'assert', 'create_function', 'call_user_func', 'call_user_func_array',
  'preg_replace',
  'phpinfo', 'getenv', 'putenv', 'get_current_user', 'getmyuid', 'getmypid',
  'getmyinode', 'getlastmod', 'disk_free_space', 'disk_total_space',
  'pcntl_fork', 'pcntl_signal', 'pcntl_waitpid', 'pcntl_alarm',
  'mail', 'header', 'setcookie', 'setrawcookie', 'header_remove',
  'apache_setenv', 'virtual', 'apache_note', 'apache_child_terminate',
];

function containsBlockedFunctions(code: string): string | null {
  const codeWithoutStrings = code
    .replace(/'[^']*'/g, '""')
    .replace(/"[^"]*"/g, '""')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/.*$/gm, '')
    .replace(/#.*$/gm, '');
  
  for (const func of BLOCKED_FUNCTIONS) {
    const regex = new RegExp(`\\b${func}\\s*\\(`, 'i');
    if (regex.test(codeWithoutStrings)) {
      return func;
    }
  }
  
  return null;
}

function simulatePHPExecution(code: string): PHPExecutionResult {
  const startTime = Date.now();
  
  const blockedFunction = containsBlockedFunctions(code);
  if (blockedFunction) {
    return {
      output: '',
      error: `Security Error: Function "${blockedFunction}" is not allowed in the sandbox.`,
    };
  }
  
  try {
    const output = executePHPCode(code);
    
    return {
      output,
      executionTime: Date.now() - startTime,
    };
  } catch (error) {
    return {
      output: '',
      error: `PHP Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
    };
  }
}

// Main execution function
function executePHPCode(code: string): string {
  // Remove PHP tags
  let processedCode = code
    .replace(/<\?php\s*/g, '')
    .replace(/\?>/g, '')
    .trim();
  
  const variables: Record<string, unknown> = {};
  let output = '';
  
  // Split into lines and process
  const lines = processedCode.split('\n');
  let i = 0;
  
  while (i < lines.length) {
    let line = lines[i].trim();
    
    // Skip empty lines and comments
    if (!line || line.startsWith('//') || line.startsWith('#')) {
      i++;
      continue;
    }
    
    // Handle multi-line arrays
    if (line.includes('[') && !line.includes(']')) {
      // Multi-line array - collect all lines until we find closing bracket
      let arrayLine = line;
      while (i < lines.length && !arrayLine.includes(']')) {
        i++;
        arrayLine += ' ' + lines[i].trim();
      }
      line = arrayLine;
    }
    
    // Process the line
    const result = processStatement(line, variables);
    if (result.output) {
      output += result.output;
    }
    
    i++;
  }
  
  return output;
}

// Process a single statement
function processStatement(line: string, variables: Record<string, unknown>): { output: string; variables: Record<string, unknown> } {
  let output = '';
  
  // Handle echo statements
  if (line.startsWith('echo ')) {
    const content = line.substring(5).trim();
    // Remove trailing semicolon
    const contentWithoutSemicolon = content.replace(/;$/, '').trim();
    output = evaluateExpression(contentWithoutSemicolon, variables);
    return { output: output + '\n', variables };
  }
  
  // Handle variable assignments
  const assignMatch = line.match(/^\$(\w+)\s*=\s*(.+?);?$/);
  if (assignMatch) {
    const varName = assignMatch[1];
    const value = assignMatch[2].trim().replace(/;$/, '');
    variables[varName] = evaluateValue(value, variables);
    return { output: '', variables };
  }
  
  // Handle for loops: for ($i = 0; $i < 5; $i++) { ... }
  const forMatch = line.match(/^for\s*\(\s*\$(\w+)\s*=\s*(\d+)\s*;\s*\$(\w+)\s*<\s*(.+?)\s*;\s*\$(\w+)\+\+\s*\)\s*\{(.+)\}$/);
  if (forMatch) {
    const counterVar = forMatch[1];
    const startVal = parseInt(forMatch[2]);
    const conditionVar = forMatch[3];
    const endVal = forMatch[4];
    const body = forMatch[6].replace(/\}.*$/, '').trim();
    
    // Get actual end value if it's a count() or variable
    let actualEnd = parseInt(endVal);
    if (endVal.startsWith('count(')) {
      const varMatch = endVal.match(/count\(\$(\w+)\)/);
      if (varMatch) {
        const arr = variables[varMatch[1]];
        if (Array.isArray(arr)) {
          actualEnd = arr.length;
        }
      }
    } else if (endVal.startsWith('$')) {
      const arr = variables[endVal.substring(1)];
      if (Array.isArray(arr)) {
        actualEnd = arr.length;
      }
    }
    
    // Execute the for loop
    for (let i = startVal; i < actualEnd; i++) {
      const loopVars = { ...variables, [counterVar]: i };
      // Process body statements
      const bodyStatements = body.split(';').filter(s => s.trim());
      for (const bodyStmt of bodyStatements) {
        const trimmed = bodyStmt.trim();
        if (trimmed.startsWith('echo ')) {
          const echoContent = trimmed.substring(5).trim();
          output += evaluateExpression(echoContent, loopVars);
        }
      }
    }
    
    return { output, variables };
  }
  
  // Handle foreach loops: foreach ($arr as $item) { ... }
  // This handles both single and multi-line foreach loops
  const foreachMatch = line.match(/^foreach\s*\(\s*\$(\w+)\s+as\s+\$(\w+)\s*\)\s*\{(.+)$/);
  if (foreachMatch) {
    const arrayName = foreachMatch[1];
    const itemVar = foreachMatch[2];
    let body = foreachMatch[3];
    
    // Find the closing brace
    let braceCount = 1;
    let i = 0;
    while (i < body.length && braceCount > 0) {
      if (body[i] === '{') braceCount++;
      if (body[i] === '}') braceCount--;
      i++;
    }
    
    // Extract body content
    if (braceCount === 0) {
      body = body.substring(0, i - 1).trim();
    }
    
    const array = variables[arrayName];
    
    if (Array.isArray(array)) {
      for (const item of array) {
        const loopVars = { ...variables, [itemVar]: item };
        // Process body statements - split by ; but be careful with nested braces
        const bodyStatements = splitByStatement(body);
        for (const bodyStmt of bodyStatements) {
          const trimmed = bodyStmt.trim();
          if (trimmed.startsWith('echo ')) {
            const echoContent = trimmed.substring(5).trim().replace(/;$/, '');
            output += evaluateExpression(echoContent, loopVars);
          }
        }
      }
    }
    
    return { output, variables };
  }
  
  return { output, variables };
}

// Evaluate a value (string, number, array, etc.)
function evaluateValue(value: string, variables: Record<string, unknown>): unknown {
  value = value.trim();
  
  // Remove trailing semicolon
  value = value.replace(/;$/, '').trim();
  
  // String with double quotes
  if (value.startsWith('"') && value.endsWith('"')) {
    let content = value.slice(1, -1);
    content = content.replace(/\$(\w+)/g, (_, varName) => {
      const val = variables[varName];
      return val !== undefined ? String(val) : '';
    });
    return content;
  }
  
  // String with single quotes
  if (value.startsWith("'") && value.endsWith("'")) {
    return value.slice(1, -1);
  }
  
  // Number
  if (!isNaN(Number(value)) && value !== '') {
    return Number(value);
  }
  
  // Boolean
  if (value.toLowerCase() === 'true') return true;
  if (value.toLowerCase() === 'false') return false;
  if (value.toLowerCase() === 'null') return null;
  
  // Array with short syntax [...]
  if (value.startsWith('[') && value.endsWith(']')) {
    return parseArray(value, variables);
  }
  
  // Function calls
  if (value.includes('(') && value.endsWith(')')) {
    return evaluateFunctionCall(value, variables);
  }
  
  // Variable reference
  if (value.startsWith('$')) {
    return variables[value.substring(1)];
  }
  
  return value;
}

// Parse an array
function parseArray(value: string, variables: Record<string, unknown>): unknown {
  const content = value.slice(1, -1).trim();
  if (!content) return [];
  
  const items: string[] = [];
  let current = '';
  let depth = 0;
  let inString = false;
  let stringChar = '';
  
  for (let i = 0; i < content.length; i++) {
    const char = content[i];
    const prevChar = i > 0 ? content[i - 1] : '';
    
    if (!inString) {
      if (char === '"' || char === "'") {
        inString = true;
        stringChar = char;
        current += char;
      } else if (char === '[' || char === '(' || char === '{') {
        depth++;
        current += char;
      } else if (char === ']' || char === ')' || char === '}') {
        depth--;
        current += char;
      } else if (char === ',' && depth === 0) {
        items.push(current);
        current = '';
      } else {
        current += char;
      }
    } else {
      current += char;
      if (char === stringChar && prevChar !== '\\') {
        inString = false;
      }
    }
  }
  
  if (current.trim()) {
    items.push(current);
  }
  
  // Parse each item
  const parsedItems: unknown[] = [];
  const assocItems: Record<string, unknown> = {};
  let isAssociative = false;
  
  for (const item of items) {
    const trimmed = item.trim();
    if (!trimmed) continue;
    
    // Check for => (associative array)
    const arrowIndex = trimmed.indexOf('=>');
    if (arrowIndex !== -1) {
      isAssociative = true;
      const key = trimmed.substring(0, arrowIndex).trim();
      const val = trimmed.substring(arrowIndex + 2).trim();
      const cleanKey = evaluateValue(key, variables);
      const cleanVal = evaluateValue(val, variables);
      assocItems[String(cleanKey)] = cleanVal;
    } else {
      parsedItems.push(evaluateValue(trimmed, variables));
    }
  }
  
  if (isAssociative) {
    return assocItems;
  }
  return parsedItems;
}

// Evaluate a PHP function call
function evaluateFunctionCall(expr: string, variables: Record<string, unknown>): unknown {
  // Extract function name and arguments
  const funcMatch = expr.match(/^(\w+)\((.*)\)$/);
  if (!funcMatch) return expr;
  
  const funcName = funcMatch[1];
  const args = funcMatch[2];
  
  if (funcName === 'array_sum') {
    const varName = args.replace('$', '').trim();
    const arr = variables[varName];
    if (Array.isArray(arr)) {
      return arr.reduce((sum: number, val) => sum + (typeof val === 'number' ? val : 0), 0);
    }
    return 0;
  }
  
  if (funcName === 'date') {
    // Extract format string
    let format = args.trim();
    if (format.startsWith("'") && format.endsWith("'")) {
      format = format.slice(1, -1);
    } else if (format.startsWith('"') && format.endsWith('"')) {
      format = format.slice(1, -1);
    }
    return formatDate(format, new Date());
  }
  
  if (funcName === 'count') {
    const varName = args.replace('$', '').trim();
    const arr = variables[varName];
    if (Array.isArray(arr)) {
      return arr.length;
    }
    if (typeof arr === 'object' && arr !== null) {
      return Object.keys(arr).length;
    }
    return 0;
  }
  
  return expr;
}

// Format date like PHP's date() function
function formatDate(format: string, date: Date): string {
  const year = date.getFullYear().toString();
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const day = date.getDate().toString().padStart(2, '0');
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const seconds = date.getSeconds().toString().padStart(2, '0');
  
  let result = format;
  result = result.replace(/Y/g, year);
  result = result.replace(/\bm\b/g, month);
  result = result.replace(/\bd\b/g, day);
  result = result.replace(/\bH\b/g, hours);
  result = result.replace(/\bi\b/g, minutes);
  result = result.replace(/\bs\b/g, seconds);
  
  return result;
}

// Evaluate an expression (for echo statements)
function evaluateExpression(expr: string, variables: Record<string, unknown>): string {
  // Check for concatenation with . operator (including without spaces)
  // Match . not preceded or followed by digit (to avoid decimal points)
  const concatIndex = findConcatOperator(expr);
  if (concatIndex !== -1) {
    const left = expr.substring(0, concatIndex).trim();
    const right = expr.substring(concatIndex + 1).trim();
    return evaluateExpression(left, variables) + evaluateExpression(right, variables);
  }
  
  // Handle strings with variable interpolation
  if (expr.startsWith('"') && expr.endsWith('"')) {
    let content = expr.slice(1, -1);
    
    // Replace array access: $arr[$idx] or $arr['key']
    content = content.replace(/\$(\w+)\[(\w+)\]/g, (_, arrName, idx) => {
      const arr = variables[arrName];
      if (Array.isArray(arr)) {
        const index = parseInt(idx);
        if (!isNaN(index) && index < arr.length) {
          return String(arr[index]);
        }
      }
      if (arr && typeof arr === 'object' && !Array.isArray(arr)) {
        return String((arr as Record<string, unknown>)[idx] ?? '');
      }
      return '';
    });
    
    // Replace simple variables
    content = content.replace(/\$(\w+)/g, (_, varName) => {
      const val = variables[varName];
      if (val !== undefined) {
        return String(val);
      }
      return '';
    });
    return content;
  }
  
  if (expr.startsWith("'") && expr.endsWith("'")) {
    return expr.slice(1, -1);
  }
  
  // Handle function calls
  if (expr.includes('(') && expr.endsWith(')')) {
    return String(evaluateFunctionCall(expr, variables));
  }
  
  // Handle array access: $var[$idx] or $var['key']
  const arrayAccessMatch = expr.match(/^\$(\w+)\[(.+)\]$/);
  if (arrayAccessMatch) {
    const varName = arrayAccessMatch[1];
    const keyOrIndex = arrayAccessMatch[2].trim();
    const arr = variables[varName];
    
    // Check if key/index is a variable
    if (keyOrIndex.startsWith('$')) {
      const idx = variables[keyOrIndex.substring(1)];
      if (Array.isArray(arr) && typeof idx === 'number') {
        return String(arr[idx] ?? '');
      }
      return '';
    }
    
    // Check if key is a number
    const numericIndex = parseInt(keyOrIndex);
    if (!isNaN(numericIndex) && Array.isArray(arr)) {
      return String(arr[numericIndex] ?? '');
    }
    
    // String key (for associative arrays)
    const key = keyOrIndex.replace(/['"]/g, '');
    if (arr && typeof arr === 'object' && !Array.isArray(arr)) {
      return String((arr as Record<string, unknown>)[key] ?? '');
    }
    return '';
  }
  
  // Handle variables
  if (expr.startsWith('$')) {
    // Simple variable
    const varName = expr.substring(1);
    const val = variables[varName];
    if (val !== undefined) {
      return String(val);
    }
    return '';
  }
  
  return expr;
}

// Find the concatenation operator (.) at depth 0
function findConcatOperator(expr: string): number {
  let depth = 0;
  let inString = false;
  let stringChar = '';
  
  for (let i = 0; i < expr.length; i++) {
    const char = expr[i];
    const prevChar = i > 0 ? expr[i - 1] : '';
    
    if (!inString) {
      if (char === '"' || char === "'") {
        inString = true;
        stringChar = char;
      } else if (char === '(' || char === '[' || char === '{') {
        depth++;
      } else if (char === ')' || char === ']' || char === '}') {
        depth--;
      } else if (char === '.' && depth === 0 && !/\d/.test(prevChar) && !/\d/.test(expr[i + 1] || '')) {
        return i;
      }
    } else {
      if (char === stringChar && prevChar !== '\\') {
        inString = false;
      }
    }
  }
  
  return -1;
}

// Split a string into statements by semicolon, respecting strings and brackets
function splitByStatement(s: string): string[] {
  const parts: string[] = [];
  let current = '';
  let depth = 0;
  let inString = false;
  let stringChar = '';
  
  for (let i = 0; i < s.length; i++) {
    const char = s[i];
    const prevChar = i > 0 ? s[i - 1] : '';
    
    if (!inString) {
      if (char === '"' || char === "'") {
        inString = true;
        stringChar = char;
        current += char;
      } else if (char === '[' || char === '(' || char === '{') {
        depth++;
        current += char;
      } else if (char === ']' || char === ')' || char === '}') {
        depth--;
        current += char;
      } else if (char === ';' && depth === 0) {
        parts.push(current);
        current = '';
      } else {
        current += char;
      }
    } else {
      current += char;
      if (char === stringChar && prevChar !== '\\') {
        inString = false;
      }
    }
  }
  
  if (current.trim()) {
    parts.push(current);
  }
  
  return parts;
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
    
    const result = simulatePHPExecution(code);
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('PHP execution error:', error);
    return NextResponse.json(
      { error: 'Internal server error during PHP execution' },
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
