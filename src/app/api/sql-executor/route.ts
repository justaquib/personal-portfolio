import { NextRequest, NextResponse } from 'next/server';

// SQL executor sandbox API
// Provides a simulated SQLite execution environment

interface SQLExecutionResult {
  output: string;
  error?: string;
  executionTime?: number;
  data?: {
    columns: string[];
    rows: Record<string, unknown>[];
  };
}

// In-memory database simulation
interface Table {
  name: string;
  columns: string[];
  rows: Record<string, unknown>[];
}

// Security: Blocked SQL patterns
const BLOCKED_PATTERNS = [
  'DROP DATABASE',
  'DROP SCHEMA',
  'TRUNCATE TABLE',
  'LOAD_FILE',
  'INTO OUTFILE',
  'INTO DUMPFILE',
  'BENCHMARK',
  'SLEEP',
  'WAITFOR',
  'PG_SLEEP',
  'EXEC(',
  'EXECUTE(',
  'xp_cmdshell',
  'sp_',
  'information_schema',
  'mysql.',
  'pg_',
  'sqlite_master',
  'sys.',
];

// Check for blocked patterns
function containsBlockedPatterns(sql: string): string | null {
  const upperSQL = sql.toUpperCase();
  for (const pattern of BLOCKED_PATTERNS) {
    if (upperSQL.includes(pattern.toUpperCase())) {
      return pattern;
    }
  }
  return null;
}

// SQL Parser and Executor
class SQLDatabase {
  private tables: Map<string, Table> = new Map();
  
  constructor() {
    // Initialize with sample data
    this.initializeSampleData();
  }
  
  private initializeSampleData() {
    // Users table
    this.tables.set('users', {
      name: 'users',
      columns: ['id', 'name', 'email', 'age', 'country'],
      rows: [
        { id: 1, name: 'John Doe', email: 'john@example.com', age: 28, country: 'USA' },
        { id: 2, name: 'Jane Smith', email: 'jane@example.com', age: 34, country: 'UK' },
        { id: 3, name: 'Bob Johnson', email: 'bob@example.com', age: 45, country: 'Canada' },
        { id: 4, name: 'Alice Brown', email: 'alice@example.com', age: 23, country: 'Australia' },
        { id: 5, name: 'Charlie Wilson', email: 'charlie@example.com', age: 31, country: 'USA' },
      ]
    });
    
    // Products table
    this.tables.set('products', {
      name: 'products',
      columns: ['id', 'name', 'category', 'price', 'stock'],
      rows: [
        { id: 1, name: 'Laptop', category: 'Electronics', price: 999.99, stock: 50 },
        { id: 2, name: 'Smartphone', category: 'Electronics', price: 699.99, stock: 100 },
        { id: 3, name: 'Headphones', category: 'Electronics', price: 149.99, stock: 200 },
        { id: 4, name: 'Book', category: 'Books', price: 19.99, stock: 500 },
        { id: 5, name: 'Desk Chair', category: 'Furniture', price: 299.99, stock: 30 },
      ]
    });
    
    // Orders table
    this.tables.set('orders', {
      name: 'orders',
      columns: ['id', 'user_id', 'product_id', 'quantity', 'total', 'order_date'],
      rows: [
        { id: 1, user_id: 1, product_id: 1, quantity: 1, total: 999.99, order_date: '2024-01-15' },
        { id: 2, user_id: 2, product_id: 2, quantity: 2, total: 1399.98, order_date: '2024-01-16' },
        { id: 3, user_id: 1, product_id: 3, quantity: 1, total: 149.99, order_date: '2024-01-17' },
        { id: 4, user_id: 3, product_id: 4, quantity: 5, total: 99.95, order_date: '2024-01-18' },
        { id: 5, user_id: 4, product_id: 5, quantity: 1, total: 299.99, order_date: '2024-01-19' },
      ]
    });
    
    // Employees table
    this.tables.set('employees', {
      name: 'employees',
      columns: ['id', 'name', 'department', 'salary', 'hire_date'],
      rows: [
        { id: 1, name: 'Alice Johnson', department: 'Engineering', salary: 85000, hire_date: '2020-03-15' },
        { id: 2, name: 'Bob Smith', department: 'Marketing', salary: 65000, hire_date: '2019-07-22' },
        { id: 3, name: 'Carol Davis', department: 'Engineering', salary: 92000, hire_date: '2018-11-01' },
        { id: 4, name: 'David Wilson', department: 'Sales', salary: 72000, hire_date: '2021-02-10' },
        { id: 5, name: 'Eva Martinez', department: 'HR', salary: 58000, hire_date: '2022-05-18' },
      ]
    });
  }
  
  execute(sql: string): { result?: SQLExecutionResult['data']; message?: string; error?: string } {
    try {
      // Parse and execute SQL
      const statements = sql.split(';').filter(s => s.trim());
      let lastResult: SQLExecutionResult['data'] | undefined;
      let messages: string[] = [];
      
      for (const statement of statements) {
        const trimmed = statement.trim();
        if (!trimmed) continue;
        
        const parsed = this.parseStatement(trimmed);
        
        if (parsed.error) {
          return { error: parsed.error };
        }
        
        if (parsed.result) {
          lastResult = parsed.result;
        }
        
        if (parsed.message) {
          messages.push(parsed.message);
        }
      }
      
      return { 
        result: lastResult, 
        message: messages.length > 0 ? messages.join('\n') : undefined 
      };
    } catch (error) {
      return { error: `SQL Error: ${error instanceof Error ? error.message : 'Unknown error'}` };
    }
  }
  
  private parseStatement(sql: string): { result?: SQLExecutionResult['data']; message?: string; error?: string } {
    // Remove comments
    let cleanSQL = sql;
    
    // Remove single-line comments (-- ...)
    cleanSQL = cleanSQL.replace(/--.*$/gm, '');
    
    // Remove multi-line comments (/* ... */)
    cleanSQL = cleanSQL.replace(/\/\*[\s\S]*?\*\//g, '');
    
    // Trim and check if empty
    const trimmedSQL = cleanSQL.trim();
    if (!trimmedSQL) {
      return {}; // Skip empty statements (comments)
    }
    
    const upperSQL = trimmedSQL.toUpperCase();
    
    // SELECT statement
    if (upperSQL.startsWith('SELECT')) {
      return this.executeSelect(trimmedSQL);
    }
    
    // CREATE TABLE statement
    if (upperSQL.startsWith('CREATE TABLE')) {
      return this.executeCreateTable(trimmedSQL);
    }
    
    // INSERT statement
    if (upperSQL.startsWith('INSERT INTO')) {
      return this.executeInsert(trimmedSQL);
    }
    
    // UPDATE statement
    if (upperSQL.startsWith('UPDATE')) {
      return this.executeUpdate(trimmedSQL);
    }
    
    // DELETE statement
    if (upperSQL.startsWith('DELETE FROM')) {
      return this.executeDelete(trimmedSQL);
    }
    
    // DROP TABLE statement
    if (upperSQL.startsWith('DROP TABLE')) {
      return this.executeDropTable(trimmedSQL);
    }
    
    // SHOW TABLES
    if (upperSQL === 'SHOW TABLES' || upperSQL === 'SHOW TABLES;') {
      return this.showTables();
    }
    
    // DESCRIBE/DESC table
    if (upperSQL.startsWith('DESCRIBE ') || upperSQL.startsWith('DESC ')) {
      return this.describeTable(trimmedSQL);
    }
    
    return { error: `Unsupported SQL statement: ${trimmedSQL.substring(0, 50)}...` };
  }
  
  private executeSelect(sql: string): { result?: SQLExecutionResult['data']; error?: string } {
    // Try to match complex SELECT with all clauses
    const selectMatch = sql.match(/SELECT\s+(.+?)\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+?))?(?:\s+GROUP\s+BY\s+(.+?))?(?:\s+ORDER\s+BY\s+(.+?))?(?:\s+LIMIT\s+(\d+))?$/i);
    
    if (selectMatch) {
      const [, columns, tableName, whereClause, groupBy, orderBy, limit] = selectMatch;
      return this.executeSimpleSelect(columns, tableName, whereClause, groupBy, orderBy, limit);
    }
    
    // Try simpler pattern: SELECT columns FROM table
    const simpleMatch = sql.match(/SELECT\s+(.+?)\s+FROM\s+(\w+)/i);
    if (simpleMatch) {
      return this.executeSimpleSelect(simpleMatch[1], simpleMatch[2], null, null, null, null);
    }
    
    return { error: 'Invalid SELECT statement syntax' };
  }
  
  private executeSimpleSelect(
    columnsStr: string, 
    tableName: string, 
    whereClause: string | null,
    groupBy: string | null,
    orderBy: string | null,
    limit: string | null
  ): { result?: SQLExecutionResult['data']; error?: string } {
    const table = this.tables.get(tableName.toLowerCase());
    
    if (!table) {
      return { error: `Table '${tableName}' does not exist` };
    }
    
    let rows = [...table.rows];
    
    // Apply WHERE clause
    if (whereClause) {
      rows = this.applyWhere(rows, whereClause);
    }
    
    // Apply ORDER BY
    if (orderBy) {
      const orderMatch = orderBy.match(/(\w+)(?:\s+(ASC|DESC))?/i);
      if (orderMatch) {
        const col = orderMatch[1];
        const dir = (orderMatch[2] || 'ASC').toUpperCase();
        rows.sort((a, b) => {
          const aVal = a[col];
          const bVal = b[col];
          if (typeof aVal === 'number' && typeof bVal === 'number') {
            return dir === 'ASC' ? aVal - bVal : bVal - aVal;
          }
          return dir === 'ASC' 
            ? String(aVal).localeCompare(String(bVal))
            : String(bVal).localeCompare(String(aVal));
        });
      }
    }
    
    // Apply LIMIT
    if (limit) {
      rows = rows.slice(0, parseInt(limit));
    }
    
    // Select columns
    let columns: string[];
    if (columnsStr.trim() === '*') {
      columns = table.columns;
    } else {
      columns = columnsStr.split(',').map(c => c.trim());
    }
    
    // Handle aggregate functions
    const aggregateResult = this.handleAggregates(columnsStr, rows);
    if (aggregateResult) {
      return { result: aggregateResult };
    }
    
    // Filter columns in rows
    const filteredRows = rows.map(row => {
      const filtered: Record<string, unknown> = {};
      for (const col of columns) {
        if (row.hasOwnProperty(col)) {
          filtered[col] = row[col];
        }
      }
      return filtered;
    });
    
    return { 
      result: { 
        columns: columns.filter(c => table.columns.includes(c)), 
        rows: filteredRows 
      } 
    };
  }
  
  private handleAggregates(columnsStr: string, rows: Record<string, unknown>[]): SQLExecutionResult['data'] | null {
    const upperColumns = columnsStr.toUpperCase();
    
    // COUNT
    const countMatch = columnsStr.match(/COUNT\s*\(\s*(\*|\w+)\s*\)/i);
    if (countMatch) {
      return { 
        columns: ['COUNT'], 
        rows: [{ COUNT: rows.length }] 
      };
    }
    
    // SUM
    const sumMatch = columnsStr.match(/SUM\s*\(\s*(\w+)\s*\)/i);
    if (sumMatch) {
      const col = sumMatch[1];
      const sum = rows.reduce((acc, row) => acc + (Number(row[col]) || 0), 0);
      return { columns: ['SUM'], rows: [{ SUM: sum }] };
    }
    
    // AVG
    const avgMatch = columnsStr.match(/AVG\s*\(\s*(\w+)\s*\)/i);
    if (avgMatch) {
      const col = avgMatch[1];
      const sum = rows.reduce((acc, row) => acc + (Number(row[col]) || 0), 0);
      return { columns: ['AVG'], rows: [{ AVG: rows.length > 0 ? sum / rows.length : 0 }] };
    }
    
    // MAX
    const maxMatch = columnsStr.match(/MAX\s*\(\s*(\w+)\s*\)/i);
    if (maxMatch) {
      const col = maxMatch[1];
      const values = rows.map(r => Number(r[col]) || 0);
      return { columns: ['MAX'], rows: [{ MAX: Math.max(...values) }] };
    }
    
    // MIN
    const minMatch = columnsStr.match(/MIN\s*\(\s*(\w+)\s*\)/i);
    if (minMatch) {
      const col = minMatch[1];
      const values = rows.map(r => Number(r[col]) || 0);
      return { columns: ['MIN'], rows: [{ MIN: Math.min(...values) }] };
    }
    
    return null;
  }
  
  private applyWhere(rows: Record<string, unknown>[], whereClause: string): Record<string, unknown>[] {
    // Simple WHERE clause parsing
    const conditions = whereClause.split(/\s+AND\s+/i);
    
    return rows.filter(row => {
      for (const condition of conditions) {
        if (!this.evaluateCondition(row, condition.trim())) {
          return false;
        }
      }
      return true;
    });
  }
  
  private evaluateCondition(row: Record<string, unknown>, condition: string): boolean {
    // Handle different operators
    const operators = ['>=', '<=', '!=', '<>', '=', '>', '<', ' LIKE ', ' IN '];
    
    for (const op of operators) {
      if (condition.toUpperCase().includes(op.toUpperCase())) {
        const [left, right] = condition.split(new RegExp(op.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i')).map(s => s.trim());
        const leftVal = row[left];
        let rightVal = right.replace(/['"]/g, '');
        
        if (op.toUpperCase() === ' LIKE ') {
          const pattern = rightVal.replace(/%/g, '.*').replace(/_/g, '.');
          return new RegExp(`^${pattern}$`, 'i').test(String(leftVal));
        }
        
        if (op.toUpperCase() === ' IN ') {
          const values = rightVal.replace(/[()]/g, '').split(',').map(v => v.trim().replace(/['"]/g, ''));
          return values.includes(String(leftVal));
        }
        
        // Convert to number if possible
        const numRight = parseFloat(rightVal);
        const numLeft = typeof leftVal === 'number' ? leftVal : parseFloat(String(leftVal));
        
        if (!isNaN(numRight) && !isNaN(numLeft)) {
          switch (op.trim()) {
            case '=': return numLeft === numRight;
            case '!=': 
            case '<>': return numLeft !== numRight;
            case '>': return numLeft > numRight;
            case '<': return numLeft < numRight;
            case '>=': return numLeft >= numRight;
            case '<=': return numLeft <= numRight;
          }
        }
        
        // String comparison
        switch (op.trim()) {
          case '=': return String(leftVal) === rightVal;
          case '!=':
          case '<>': return String(leftVal) !== rightVal;
          case '>': return String(leftVal) > rightVal;
          case '<': return String(leftVal) < rightVal;
          case '>=': return String(leftVal) >= rightVal;
          case '<=': return String(leftVal) <= rightVal;
        }
      }
    }
    
    return true;
  }
  
  private executeCreateTable(sql: string): { message?: string; error?: string } {
    const match = sql.match(/CREATE\s+TABLE\s+(\w+)\s*\((.+)\)/i);
    
    if (!match) {
      return { error: 'Invalid CREATE TABLE syntax' };
    }
    
    const [, tableName, columnsStr] = match;
    const columns = columnsStr.split(',').map(c => c.trim().split(/\s+/)[0]);
    
    this.tables.set(tableName.toLowerCase(), {
      name: tableName.toLowerCase(),
      columns,
      rows: []
    });
    
    return { message: `Table '${tableName}' created successfully` };
  }
  
  private executeInsert(sql: string): { message?: string; error?: string } {
    const match = sql.match(/INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)/i);
    
    if (!match) {
      return { error: 'Invalid INSERT syntax' };
    }
    
    const [, tableName, columnsStr, valuesStr] = match;
    const table = this.tables.get(tableName.toLowerCase());
    
    if (!table) {
      return { error: `Table '${tableName}' does not exist` };
    }
    
    const columns = columnsStr.split(',').map(c => c.trim());
    const values = valuesStr.split(',').map(v => {
      const trimmed = v.trim();
      if (trimmed.startsWith("'") || trimmed.startsWith('"')) {
        return trimmed.slice(1, -1);
      }
      const num = parseFloat(trimmed);
      return isNaN(num) ? trimmed : num;
    });
    
    const row: Record<string, unknown> = {};
    columns.forEach((col, i) => {
      row[col] = values[i];
    });
    
    table.rows.push(row);
    
    return { message: `1 row inserted into '${tableName}'` };
  }
  
  private executeUpdate(sql: string): { message?: string; error?: string } {
    const match = sql.match(/UPDATE\s+(\w+)\s+SET\s+(.+?)\s+WHERE\s+(.+)/i);
    
    if (!match) {
      return { error: 'Invalid UPDATE syntax' };
    }
    
    const [, tableName, setClause, whereClause] = match;
    const table = this.tables.get(tableName.toLowerCase());
    
    if (!table) {
      return { error: `Table '${tableName}' does not exist` };
    }
    
    // Parse SET clause
    const setPairs = setClause.split(',').map(pair => {
      const [col, val] = pair.split('=').map(s => s.trim());
      return { col, val: val.replace(/['"]/g, '') };
    });
    
    // Find and update rows
    let updated = 0;
    for (let i = 0; i < table.rows.length; i++) {
      if (this.evaluateCondition(table.rows[i], whereClause)) {
        for (const { col, val } of setPairs) {
          const numVal = parseFloat(val);
          table.rows[i][col] = isNaN(numVal) ? val : numVal;
        }
        updated++;
      }
    }
    
    return { message: `${updated} row(s) updated` };
  }
  
  private executeDelete(sql: string): { message?: string; error?: string } {
    const match = sql.match(/DELETE\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+))?/i);
    
    if (!match) {
      return { error: 'Invalid DELETE syntax' };
    }
    
    const [, tableName, whereClause] = match;
    const table = this.tables.get(tableName.toLowerCase());
    
    if (!table) {
      return { error: `Table '${tableName}' does not exist` };
    }
    
    if (!whereClause) {
      const count = table.rows.length;
      table.rows = [];
      return { message: `${count} row(s) deleted` };
    }
    
    const originalLength = table.rows.length;
    table.rows = table.rows.filter(row => !this.evaluateCondition(row, whereClause));
    
    return { message: `${originalLength - table.rows.length} row(s) deleted` };
  }
  
  private executeDropTable(sql: string): { message?: string; error?: string } {
    const match = sql.match(/DROP\s+TABLE\s+(\w+)/i);
    
    if (!match) {
      return { error: 'Invalid DROP TABLE syntax' };
    }
    
    const tableName = match[1].toLowerCase();
    
    if (!this.tables.has(tableName)) {
      return { error: `Table '${tableName}' does not exist` };
    }
    
    this.tables.delete(tableName);
    
    return { message: `Table '${tableName}' dropped successfully` };
  }
  
  private showTables(): { result?: SQLExecutionResult['data'] } {
    const tables = Array.from(this.tables.keys());
    return {
      result: {
        columns: ['Tables'],
        rows: tables.map(t => ({ Tables: t }))
      }
    };
  }
  
  private describeTable(sql: string): { result?: SQLExecutionResult['data']; error?: string } {
    const match = sql.match(/(?:DESCRIBE|DESC)\s+(\w+)/i);
    
    if (!match) {
      return { error: 'Invalid DESCRIBE syntax' };
    }
    
    const tableName = match[1].toLowerCase();
    const table = this.tables.get(tableName);
    
    if (!table) {
      return { error: `Table '${tableName}' does not exist` };
    }
    
    return {
      result: {
        columns: ['Field', 'Type'],
        rows: table.columns.map(col => ({ Field: col, Type: 'text' }))
      }
    };
  }
}

// Create a global database instance
const db = new SQLDatabase();

function executeSQL(sql: string): SQLExecutionResult {
  const startTime = Date.now();
  
  // Security check
  const blockedPattern = containsBlockedPatterns(sql);
  if (blockedPattern) {
    return {
      output: '',
      error: `Security Error: Pattern "${blockedPattern}" is not allowed.`,
    };
  }
  
  const result = db.execute(sql);
  
  let output = '';
  
  if (result.error) {
    return {
      output: '',
      error: result.error,
      executionTime: Date.now() - startTime,
    };
  }
  
  if (result.message) {
    output = result.message + '\n';
  }
  
  if (result.result) {
    // Format as table
    const { columns, rows } = result.result;
    
    // Calculate column widths
    const widths: number[] = columns.map(col => col.length);
    for (const row of rows) {
      for (let i = 0; i < columns.length; i++) {
        const val = String(row[columns[i]] ?? '');
        widths[i] = Math.max(widths[i], val.length);
      }
    }
    
    // Build table header
    const separator = '+' + widths.map(w => '-'.repeat(w + 2)).join('+') + '+';
    output += separator + '\n';
    output += '| ' + columns.map((col, i) => col.padEnd(widths[i])).join(' | ') + ' |\n';
    output += separator + '\n';
    
    // Build table rows
    for (const row of rows) {
      output += '| ' + columns.map((col, i) => String(row[col] ?? '').padEnd(widths[i])).join(' | ') + ' |\n';
    }
    output += separator + '\n';
    output += `${rows.length} row(s) returned\n`;
  }
  
  return {
    output,
    executionTime: Date.now() - startTime,
  };
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
    
    // Security: Limit code size
    if (code.length > 50000) {
      return NextResponse.json(
        { error: 'Code size exceeds maximum limit (50KB)' },
        { status: 400 }
      );
    }
    
    // Execute SQL
    const result = executeSQL(code);
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('SQL execution error:', error);
    return NextResponse.json(
      { error: 'Internal server error during SQL execution' },
      { status: 500 }
    );
  }
}

// Handle OPTIONS for CORS preflight
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
