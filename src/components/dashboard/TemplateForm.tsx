'use client'

import { useState } from 'react'
import type { MessageTemplate, TemplateFormData } from '@/types/database'
import '../../app/dashboard/dashboard.css'

interface TemplateFormProps {
  template?: MessageTemplate | null
  onSave: (data: TemplateFormData) => Promise<void>
  onCancel: () => void
}

export function TemplateForm({ template, onSave, onCancel }: TemplateFormProps) {
  const [formData, setFormData] = useState<TemplateFormData>(
    template
      ? { name: template.name, content: template.content }
      : { name: '', content: '' }
  )
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true)
    setError(null)

    try {
      await onSave(formData)
    } catch (err: any) {
      setError(err.message || 'Failed to save template')
    } finally {
      setSaving(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="form-section">
      <h3 className="form-title">
        {template ? 'Edit Template' : 'New Template'}
      </h3>

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      <div className="form-stack">
        <input
          type="text"
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          placeholder="Template Name"
          className="form-input"
          required
        />
        <textarea
          value={formData.content}
          onChange={(e) => setFormData({ ...formData, content: e.target.value })}
          rows={4}
          placeholder="Template message content..."
          className="form-textarea"
          required
        />
      </div>

      <div className="form-actions">
        <button
          type="submit"
          disabled={saving}
          className="btn btn-primary"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="btn btn-secondary"
        >
          Cancel
        </button>
      </div>
    </form>
  )
}
