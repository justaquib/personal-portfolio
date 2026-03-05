'use client'

import { useState } from 'react'
import type { Contact, ContactFormData } from '@/types/database'

interface ContactFormProps {
  contact?: Contact | null
  onSave: (data: ContactFormData) => Promise<void>
  onCancel: () => void
}

const initialFormData: ContactFormData = {
  name: '',
  company: '',
  email: '',
  phone_number: '',
  address: '',
  notes: '',
  is_active: true,
}

export function ContactForm({ contact, onSave, onCancel }: ContactFormProps) {
  const [formData, setFormData] = useState<ContactFormData>(
    contact
      ? {
          name: contact.name,
          company: contact.company || '',
          email: contact.email || '',
          phone_number: contact.phone_number,
          address: contact.address || '',
          notes: contact.notes || '',
          is_active: contact.is_active,
        }
      : initialFormData
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
      setError(err.message || 'Failed to save contact')
    } finally {
      setSaving(false)
    }
  }

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value, type } = e.target
    const checked = (e.target as HTMLInputElement).checked
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }))
  }

  return (
    <form onSubmit={handleSubmit} className="form-section">
      <h3 className="form-title">
        {contact ? 'Edit Contact' : 'New Contact'}
      </h3>

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      <div className="form-grid">
        <input
          type="text"
          name="name"
          value={formData.name}
          onChange={handleChange}
          placeholder="Name *"
          className="form-input"
          required
        />
        <input
          type="text"
          name="company"
          value={formData.company}
          onChange={handleChange}
          placeholder="Company"
          className="form-input"
        />
        <input
          type="email"
          name="email"
          value={formData.email}
          onChange={handleChange}
          placeholder="Email"
          className="form-input"
        />
        <input
          type="tel"
          name="phone_number"
          value={formData.phone_number}
          onChange={handleChange}
          placeholder="Phone Number *"
          className="form-input"
          required
        />
        <input
          type="text"
          name="address"
          value={formData.address}
          onChange={handleChange}
          placeholder="Address"
          className="form-input"
        />
        <label className="form-checkbox-label">
          <input
            type="checkbox"
            name="is_active"
            checked={formData.is_active}
            onChange={handleChange}
            className="form-checkbox"
          />
          <span>Active</span>
        </label>
      </div>

      <div className="form-textarea-section">
        <textarea
          name="notes"
          value={formData.notes}
          onChange={handleChange}
          placeholder="Notes"
          rows={2}
          className="form-textarea"
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
