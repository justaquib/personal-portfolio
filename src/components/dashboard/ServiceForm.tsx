'use client'

import { useState, useEffect } from 'react'
import type { Service, ServiceFormData } from '@/types/database'

interface ServiceFormProps {
  service?: Service | null
  onSave: (data: ServiceFormData) => Promise<void>
  onCancel: () => void
}

const initialFormData: ServiceFormData = {
  name: '',
  amount: 0,
  actual_cost: 0,
  description: '',
}

export function ServiceForm({ service, onSave, onCancel }: ServiceFormProps) {
  const [formData, setFormData] = useState<ServiceFormData>(
    service
      ? {
          name: service.name,
          amount: service.amount,
          actual_cost: service.actual_cost || 0,
          description: service.description || '',
        }
      : initialFormData
  )
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Update form when service prop changes
  useEffect(() => {
    if (service) {
      setFormData({
        name: service.name,
        amount: service.amount,
        actual_cost: service.actual_cost || 0,
        description: service.description || '',
      })
    } else {
      setFormData(initialFormData)
    }
  }, [service])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true)
    setError(null)

    try {
      await onSave(formData)
    } catch (err: any) {
      setError(err.message || 'Failed to save service')
    } finally {
      setSaving(false)
    }
  }

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value, type } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) || 0 : value,
    }))
  }

  return (
    <form onSubmit={handleSubmit} className="form-section">
      <h3 className="form-title">
        {service ? 'Edit Service' : 'New Service'}
      </h3>

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      <div className="form-stack">
        <div>
          <label className="form-label">Service Name *</label>
          <input
            type="text"
            name="name"
            value={formData.name}
            onChange={handleChange}
            placeholder="e.g., Internet, Rent, Maintenance"
            className="form-input"
            required
          />
        </div>
        <div>
          <label className="form-label">Monthly Amount *</label>
          <input
            type="number"
            name="amount"
            value={formData.amount || ''}
            onChange={handleChange}
            placeholder="0.00"
            className="form-input"
            min="0"
            step="0.01"
            required
          />
        </div>
        <div>
          <label className="form-label">Actual Cost *</label>
          <input
            type="number"
            name="actual_cost"
            value={formData.actual_cost || ''}
            onChange={handleChange}
            placeholder="0.00"
            className="form-input"
            min="0"
            step="0.01"
            required
          />
          <p className="text-xs text-gray-500 mt-1">Your cost for this service</p>
        </div>
        <div>
          <label className="form-label">Description</label>
          <textarea
            name="description"
            value={formData.description}
            onChange={handleChange}
            placeholder="Service description (optional)"
            rows={2}
            className="form-textarea"
          />
        </div>
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
