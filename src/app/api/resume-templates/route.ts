import { NextRequest, NextResponse } from 'next/server'
import Database from 'better-sqlite3'
import path from 'path'

let db: Database.Database | null = null

const getDb = () => {
  if (!db) {
    db = new Database(path.join(process.cwd(), 'resumes.db'))
    db.exec(`
      CREATE TABLE IF NOT EXISTS resume_templates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        template_data TEXT NOT NULL,
        is_active INTEGER DEFAULT 1,
        created_by TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `)
    
    // Seed default templates if none exist
    const existing = db.prepare('SELECT COUNT(*) as count FROM resume_templates').get() as { count: number }
    if (existing.count === 0) {
      const defaultTemplates = [
        {
          name: 'Modern Purple',
          description: 'Clean and professional with a purple accent',
          template_data: JSON.stringify({
            headerBg: '#9333ea',
            headerText: '#ffffff',
            accentColor: '#9333ea',
            primaryColor: '#808080',
            fontFamily: 'helvetica',
            layout: 'modern',
            badgeStyle: 'rounded'
          })
        },
        {
          name: 'Classic Blue',
          description: 'Traditional corporate blue theme',
          template_data: JSON.stringify({
            headerBg: '#3b82f6',
            headerText: '#ffffff',
            accentColor: '#3b82f6',
            primaryColor: '#3c3c3c',
            fontFamily: 'times',
            layout: 'classic',
            badgeStyle: 'square'
          })
        },
        {
          name: 'Minimal Green',
          description: 'Simple and elegant with green accents',
          template_data: JSON.stringify({
            headerBg: '#059669',
            headerText: '#ffffff',
            accentColor: '#059669',
            primaryColor: '#065f46',
            fontFamily: 'helvetica',
            layout: 'minimal',
            badgeStyle: 'pill'
          })
        },
        {
          name: 'Creative Orange',
          description: 'Stand out with a unique orange design',
          template_data: JSON.stringify({
            headerBg: '#ea580c',
            headerText: '#ffffff',
            accentColor: '#ea580c',
            primaryColor: '#dc2626',
            fontFamily: 'georgia',
            layout: 'creative',
            badgeStyle: 'rounded'
          })
        }
      ]
      
      const insert = db.prepare('INSERT INTO resume_templates (name, description, template_data) VALUES (?, ?, ?)')
      for (const template of defaultTemplates) {
        insert.run(template.name, template.description, template.template_data)
      }
    }
  }
  return db
}

export async function GET() {
  try {
    const db = getDb()
    const templates = db.prepare('SELECT * FROM resume_templates WHERE is_active = 1 ORDER BY created_at DESC').all()
    return NextResponse.json(templates)
  } catch (error) {
    console.error('Error fetching templates:', error)
    return NextResponse.json({ error: 'Failed to fetch templates' }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { id, name, description, templateData, userId } = body
    
    const db = getDb()
    
    if (id) {
      // Update existing template
      db.prepare('UPDATE resume_templates SET name = ?, description = ?, template_data = ? WHERE id = ?')
        .run(name, description, JSON.stringify(templateData), id)
      return NextResponse.json({ success: true, id })
    } else {
      // Create new template
      const result = db.prepare('INSERT INTO resume_templates (name, description, template_data, created_by) VALUES (?, ?, ?, ?)')
        .run(name, description, JSON.stringify(templateData), userId)
      return NextResponse.json({ success: true, id: result.lastInsertRowid })
    }
  } catch (error) {
    console.error('Error saving template:', error)
    return NextResponse.json({ error: 'Failed to save template' }, { status: 500 })
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')
    
    if (!id) {
      return NextResponse.json({ error: 'Template ID required' }, { status: 400 })
    }
    
    const db = getDb()
    db.prepare('UPDATE resume_templates SET is_active = 0 WHERE id = ?').run(id)
    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Error deleting template:', error)
    return NextResponse.json({ error: 'Failed to delete template' }, { status: 500 })
  }
}
