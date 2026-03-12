import { NextRequest, NextResponse } from 'next/server'
import Database from 'better-sqlite3'
import path from 'path'

// Initialize SQLite database
const dbPath = path.join(process.cwd(), 'resumes.db')
const db = new Database(dbPath)

// Migration: Add new columns if they don't exist
const migrateDatabase = () => {
  try {
    // Check if certifications column exists
    const tableInfo = db.prepare('PRAGMA table_info(resumes)').all()
    const columnNames = tableInfo.map((col: any) => col.name)
    
    if (!columnNames.includes('certifications')) {
      db.exec('ALTER TABLE resumes ADD COLUMN certifications TEXT NOT NULL DEFAULT \'[]\'')
    }
    if (!columnNames.includes('websites')) {
      db.exec('ALTER TABLE resumes ADD COLUMN websites TEXT NOT NULL DEFAULT \'[]\'')
    }
    if (!columnNames.includes('languages')) {
      db.exec('ALTER TABLE resumes ADD COLUMN languages TEXT NOT NULL DEFAULT \'[]\'')
    }
    console.log('Migration completed successfully')
  } catch (error) {
    console.error('Migration error:', error)
  }
}

// Run migration
migrateDatabase()

// Create resumes table if it doesn't exist
db.exec(`
  CREATE TABLE IF NOT EXISTS resumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    template TEXT NOT NULL DEFAULT 'modern',
    personal_info TEXT NOT NULL DEFAULT '{}',
    summary TEXT,
    experience TEXT NOT NULL DEFAULT '[]',
    education TEXT NOT NULL DEFAULT '[]',
    skills TEXT,
    projects TEXT NOT NULL DEFAULT '[]',
    certifications TEXT NOT NULL DEFAULT '[]',
    websites TEXT NOT NULL DEFAULT '[]',
    languages TEXT NOT NULL DEFAULT '[]',
    is_default INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
  )
`)

// GET - Fetch all resumes for a user or single resume
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')
    const userId = searchParams.get('userId')

    if (id) {
      // Get single resume
      const stmt = db.prepare('SELECT * FROM resumes WHERE id = ?')
      const resume = stmt.get(id)
      
      if (!resume) {
        return NextResponse.json({ error: 'Resume not found' }, { status: 404 })
      }

      return NextResponse.json(resume)
    }

    // Get all resumes for user
    if (!userId) {
      return NextResponse.json({ error: 'User ID is required' }, { status: 400 })
    }

    const stmt = db.prepare('SELECT * FROM resumes WHERE user_id = ? ORDER BY created_at DESC')
    const resumes = stmt.all(userId)

    return NextResponse.json(resumes)
  } catch (error) {
    console.error('Error fetching resumes:', error)
    return NextResponse.json({ error: 'Failed to fetch resumes' }, { status: 500 })
  }
}

// POST - Create or update a resume
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { id, userId, name, template, personalInfo, summary, experience, education, skills, projects, certifications, websites, languages, isDefault } = body

    if (!userId) {
      return NextResponse.json({ error: 'User ID is required' }, { status: 400 })
    }

    // If setting as default, unset other defaults first
    if (isDefault) {
      db.prepare('UPDATE resumes SET is_default = 0 WHERE user_id = ?').run(userId)
    }

    // Serialize JSON fields
    const personalInfoJson = JSON.stringify(personalInfo || {})
    const experienceJson = JSON.stringify(experience || [])
    const educationJson = JSON.stringify(education || [])
    const projectsJson = JSON.stringify(projects || [])
    const certificationsJson = JSON.stringify(certifications || [])
    const websitesJson = JSON.stringify(websites || [])
    const languagesJson = JSON.stringify(languages || [])

    if (id) {
      // Update existing resume
      const stmt = db.prepare(`
        UPDATE resumes SET 
          name = ?, template = ?, personal_info = ?, summary = ?,
          experience = ?, education = ?, skills = ?, projects = ?, 
          certifications = ?, websites = ?, languages = ?,
          is_default = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND user_id = ?
      `)
      
      const result = stmt.run(
        name, template || 'modern', personalInfoJson, summary || '',
        experienceJson, educationJson, skills || '', projectsJson,
        certificationsJson, websitesJson, languagesJson,
        isDefault ? 1 : 0, id, userId
      )

      if (result.changes === 0) {
        return NextResponse.json({ error: 'Resume not found' }, { status: 404 })
      }

      const updated = db.prepare('SELECT * FROM resumes WHERE id = ?').get(id)
      return NextResponse.json(updated)
    }

    // Create new resume
    const stmt = db.prepare(`
      INSERT INTO resumes (user_id, name, template, personal_info, summary, experience, education, skills, projects, certifications, websites, languages, is_default)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `)

    const result = stmt.run(
      userId, name, template || 'modern', personalInfoJson, summary || '',
      experienceJson, educationJson, skills || '', projectsJson,
      certificationsJson, websitesJson, languagesJson,
      isDefault ? 1 : 0
    )

    const newResume = db.prepare('SELECT * FROM resumes WHERE id = ?').get(result.lastInsertRowid)
    return NextResponse.json(newResume, { status: 201 })
  } catch (error) {
    console.error('Error saving resume:', error)
    return NextResponse.json({ error: 'Failed to save resume' }, { status: 500 })
  }
}

// DELETE - Delete a resume
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')

    if (!id) {
      return NextResponse.json({ error: 'Resume ID is required' }, { status: 400 })
    }

    const result = db.prepare('DELETE FROM resumes WHERE id = ?').run(id)

    if (result.changes === 0) {
      return NextResponse.json({ error: 'Resume not found' }, { status: 404 })
    }

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Error deleting resume:', error)
    return NextResponse.json({ error: 'Failed to delete resume' }, { status: 500 })
  }
}
