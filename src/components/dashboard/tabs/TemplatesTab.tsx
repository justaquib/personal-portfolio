'use client'

import { useState, useEffect } from 'react'
import { Card, EmptyState, LoadingState, Button } from '../ui'
import { TemplateForm } from '../TemplateForm'
import { MessageTemplate } from '@/types/database'
import { useTemplates } from '@/hooks/useDashboardData'
import { Pencil, Trash2 } from 'lucide-react'

interface TemplatesTabProps {
  userId: string
}

export function TemplatesTab({ userId }: TemplatesTabProps) {
  const { templates, loading: templatesLoading, fetchTemplates, saveTemplate, deleteTemplate } = useTemplates()
  
  // Fetch data on mount
  useEffect(() => {
    fetchTemplates()
  }, [fetchTemplates])
  const [showForm, setShowForm] = useState(false)
  const [editingTemplate, setEditingTemplate] = useState<MessageTemplate | null>(null)

  const handleSave = async (data: any) => {
    await saveTemplate(data, userId, editingTemplate?.id)
    setEditingTemplate(null)
    setShowForm(false)
  }

  return (
    <Card 
      title="Message Templates"
      actions={
        <Button onClick={() => { setEditingTemplate(null); setShowForm(true); }}>
          Add Template
        </Button>
      }
    >
      {(showForm || editingTemplate) && (
        <div className="mb-6 p-4 bg-gray-50 rounded-xl">
          <TemplateForm
            template={editingTemplate}
            onSave={handleSave}
            onCancel={() => { setEditingTemplate(null); setShowForm(false); }}
          />
        </div>
      )}

      {templatesLoading ? (
        <LoadingState />
      ) : templates.length === 0 ? (
        <EmptyState 
          message="No templates yet. Click 'Add Template' to create one."
          action={
            <Button onClick={() => setShowForm(true)}>Add Template</Button>
          }
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {templates.map((template) => (
            <div key={template.id} className="p-4 bg-gray-50 rounded-xl">
              <div className="flex items-start justify-between">
                <div>
                  <h4 className="font-medium text-gray-900">{template.name}</h4>
                  <p className="text-sm text-gray-600 mt-1">{template.content}</p>
                </div>
                <div className="flex items-center gap-1">
                  <button 
                    onClick={() => { setEditingTemplate(template); setShowForm(true); }} 
                    className="p-1.5 text-purple-600 hover:bg-purple-50 rounded"
                    title="Edit"
                  >
                    <Pencil className="w-4 h-4" />
                  </button>
                  <button 
                    onClick={() => deleteTemplate(template.id)} 
                    className="p-1.5 text-red-600 hover:bg-red-50 rounded"
                    title="Delete"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}
