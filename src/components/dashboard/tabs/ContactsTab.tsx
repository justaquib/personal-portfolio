'use client'

import { useState, useEffect } from 'react'
import { Card, EmptyState, LoadingState, Button, Badge } from '../ui'
import { ContactForm } from '../ContactForm'
import { Contact } from '@/types/database'
import { useContacts, useSubscriptions } from '@/hooks/useDashboardData'
import { CURRENCY_SYMBOL } from '@/constants'
import { Pencil, Trash2 } from 'lucide-react'

interface ContactsTabProps {
  userId: string
}

export function ContactsTab({ userId }: ContactsTabProps) {
  const { contacts, loading: contactsLoading, fetchContacts, saveContact, deleteContact } = useContacts()
  const { subscriptions, loading: subscriptionsLoading, fetchSubscriptions } = useSubscriptions()
  
  const [showForm, setShowForm] = useState(false)
  const [editingContact, setEditingContact] = useState<Contact | null>(null)

  // Fetch data on mount
  useEffect(() => {
    fetchContacts()
    fetchSubscriptions()
  }, [fetchContacts, fetchSubscriptions])

  const getSubscriptionsForContact = (contactId: string) => {
    return subscriptions.filter(s => s.contact_id === contactId)
  }

  const handleSave = async (data: any) => {
    await saveContact(data, userId, editingContact?.id)
    setEditingContact(null)
    setShowForm(false)
  }

  const loading = contactsLoading || subscriptionsLoading

  return (
    <Card 
      title="Contacts"
      actions={
        <Button onClick={() => { setEditingContact(null); setShowForm(true); }}>
          Add Contact
        </Button>
      }
    >
      {(showForm || editingContact) && (
        <div className="mb-6 p-4 bg-gray-50 rounded-xl">
          <ContactForm
            contact={editingContact}
            onSave={handleSave}
            onCancel={() => { setEditingContact(null); setShowForm(false); }}
          />
        </div>
      )}

      {loading ? (
        <LoadingState />
      ) : contacts.length === 0 ? (
        <EmptyState 
          message="No contacts yet. Click 'Add Contact' to create one."
          action={
            <Button onClick={() => setShowForm(true)}>Add Contact</Button>
          }
        />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left py-3 px-4 font-medium text-gray-600">Name</th>
                <th className="text-left py-3 px-4 font-medium text-gray-600">Phone</th>
                <th className="text-left py-3 px-4 font-medium text-gray-600">Services</th>
                <th className="text-left py-3 px-4 font-medium text-gray-600">Status</th>
                <th className="text-left py-3 px-4 font-medium text-gray-600">Actions</th>
              </tr>
            </thead>
            <tbody>
              {contacts.map((contact) => (
                <tr key={contact.id} className="border-b hover:bg-gray-50">
                  <td className="py-3 px-4">{contact.name}</td>
                  <td className="py-3 px-4">{contact.phone_number}</td>
                  <td className="py-3 px-4">
                    {getSubscriptionsForContact(contact.id).length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {getSubscriptionsForContact(contact.id).map((sub) => (
                          <Badge key={sub.id} variant="info">
                            {sub.service?.name} ({CURRENCY_SYMBOL}{sub.service?.amount})
                          </Badge>
                        ))}
                      </div>
                    ) : (
                      <span className="text-gray-400 text-sm">No services</span>
                    )}
                  </td>
                  <td className="py-3 px-4">
                    <Badge variant={contact.is_active ? 'success' : 'default'}>
                      {contact.is_active ? 'Active' : 'Inactive'}
                    </Badge>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-1">
                      <button 
                        onClick={() => { setEditingContact(contact); setShowForm(true); }} 
                        className="p-1.5 text-purple-600 hover:bg-purple-50 rounded"
                        title="Edit"
                      >
                        <Pencil className="w-4 h-4" />
                      </button>
                      <button 
                        onClick={() => deleteContact(contact.id)} 
                        className="p-1.5 text-red-600 hover:bg-red-50 rounded"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}
