'use client'

import { useState, useEffect } from 'react'
import { Card, EmptyState, LoadingState, Button, Badge } from '../ui'
import { ServiceForm } from '../ServiceForm'
import { Service, Subscription } from '@/types/database'
import { useServices, useSubscriptions, useContacts } from '@/hooks/useDashboardData'
import { CURRENCY_SYMBOL } from '@/constants'
import { Pencil, Trash2, Unlink } from 'lucide-react'

const getCurrentMonth = () => {
  const now = new Date()
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`
}

const getYearOptions = () => {
  const years = []
  const currentYear = new Date().getFullYear()
  for (let year = currentYear; year >= currentYear - 5; year--) {
    years.push(year)
  }
  return years
}

const getMonthOptions = () => [
  { value: '01', label: 'January' },
  { value: '02', label: 'February' },
  { value: '03', label: 'March' },
  { value: '04', label: 'April' },
  { value: '05', label: 'May' },
  { value: '06', label: 'June' },
  { value: '07', label: 'July' },
  { value: '08', label: 'August' },
  { value: '09', label: 'September' },
  { value: '10', label: 'October' },
  { value: '11', label: 'November' },
  { value: '12', label: 'December' },
]

interface ServicesTabProps {
  userId: string
}

export function ServicesTab({ userId }: ServicesTabProps) {
  const { services, loading: servicesLoading, fetchServices, saveService, deleteService } = useServices()
  const { subscriptions, loading: subscriptionsLoading, fetchSubscriptions, saveSubscription, deleteSubscription } = useSubscriptions()
  const { contacts, loading: contactsLoading, fetchContacts } = useContacts()
  
  // Fetch data on mount
  useEffect(() => {
    fetchServices()
    fetchSubscriptions()
    fetchContacts()
  }, [fetchServices, fetchSubscriptions, fetchContacts])

  const loading = servicesLoading || subscriptionsLoading || contactsLoading
  
  const [activeSubTab, setActiveSubTab] = useState<'services' | 'linked'>('services')
  const [showForm, setShowForm] = useState(false)
  const [editingService, setEditingService] = useState<Service | null>(null)
  const [selectedContactForService, setSelectedContactForService] = useState('')
  const [selectedServiceToLink, setSelectedServiceToLink] = useState('')
  const [subscriptionStartMonth, setSubscriptionStartMonth] = useState(getCurrentMonth())
  const [linkLoading, setLinkLoading] = useState(false)

  const handleServiceSave = async (data: any) => {
    await saveService(data, userId, editingService?.id)
    setShowForm(false)
    setEditingService(null)
  }

  const handleLinkService = async () => {
    if (!selectedContactForService || !selectedServiceToLink) return
    
    setLinkLoading(true)
    try {
      await saveSubscription({
        contact_id: selectedContactForService,
        service_id: selectedServiceToLink,
        started_at: `${subscriptionStartMonth}-01`,
        user_id: userId
      })
      setSelectedContactForService('')
      setSelectedServiceToLink('')
      setSubscriptionStartMonth(getCurrentMonth())
    } catch (err: any) {
      console.error('Failed to link service:', err)
    } finally {
      setLinkLoading(false)
    }
  }

  const handleUnlinkService = async (subscriptionId: string) => {
    try {
      await deleteSubscription(subscriptionId)
    } catch (err: any) {
      console.error('Failed to unlink service:', err)
    }
  }

  return (
    <div className="space-y-6">
      {/* Sub-tabs */}
      <div className="flex bg-gray-100 rounded-lg p-1 w-fit">
        <button
          onClick={() => setActiveSubTab('services')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeSubTab === 'services'
              ? 'bg-white text-gray-900 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Services ({services.length})
        </button>
        <button
          onClick={() => setActiveSubTab('linked')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeSubTab === 'linked'
              ? 'bg-white text-gray-900 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Linked Services ({subscriptions.length})
        </button>
      </div>

      {/* Services Section */}
      {activeSubTab === 'services' && (
        <Card 
          title="Services"
          actions={
            <Button onClick={() => { setEditingService(null); setShowForm(true); }}>
              {editingService ? 'Edit Service' : 'Add Service'}
            </Button>
          }
        >
          {showForm && (
            <div className="mb-6 p-4 bg-gray-50 rounded-xl">
              <ServiceForm
                service={editingService}
                onSave={handleServiceSave}
                onCancel={() => { setShowForm(false); setEditingService(null); }}
              />
            </div>
          )}

          {servicesLoading ? (
            <LoadingState />
          ) : services.length === 0 ? (
            <EmptyState message="No services yet. Click 'Add Service' to create one." />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {services.map((service) => (
                <div key={service.id} className="p-4 bg-gray-50 rounded-xl">
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="font-medium text-gray-900">{service.name}</h4>
                      <p className="text-sm text-gray-600 mt-1">
                        {CURRENCY_SYMBOL}{service.amount} <span className="text-xs">/ {service.payment_cycle || 'month'}</span>
                      </p>
                      {service.actual_cost > 0 && (
                        <p className="text-xs text-red-500 mt-1">Cost: {CURRENCY_SYMBOL}{service.actual_cost}</p>
                      )}
                      <p className="text-xs text-gray-500 mt-1">{service.description}</p>
                    </div>
                    <div className="flex items-center gap-1">
                      <button 
                        onClick={() => { setEditingService(service); setShowForm(true); }} 
                        className="p-1.5 text-gray-600 hover:bg-gray-100 rounded"
                      >
                        <Pencil className="w-4 h-4" />
                      </button>
                      <button 
                        onClick={() => deleteService(service.id)} 
                        className="p-1.5 text-red-600 hover:bg-red-50 rounded"
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
      )}

      {/* Linked Services Section */}
      {activeSubTab === 'linked' && (
        <Card title="Link Service to Contact">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Select Contact</label>
              <select
                value={selectedContactForService}
                onChange={(e) => setSelectedContactForService(e.target.value)}
                className="w-full px-4 py-2 border rounded-lg focus:ring-2"
                style={{ borderColor: '#ced4da', backgroundColor: '#f8f9fa', color: '#212529' }}
              >
                <option value="">Choose a contact...</option>
                {contacts.map((contact) => (
                  <option key={contact.id} value={contact.id}>{contact.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Select Service</label>
              <select
                value={selectedServiceToLink}
                onChange={(e) => setSelectedServiceToLink(e.target.value)}
                className="w-full px-4 py-2 border rounded-lg focus:ring-2"
                style={{ borderColor: '#ced4da', backgroundColor: '#f8f9fa', color: '#212529' }}
              >
                <option value="">Choose a service...</option>
                {services.map((service) => (
                  <option key={service.id} value={service.id}>
                    {service.name} - {CURRENCY_SYMBOL}{service.amount}/{service.payment_cycle || 'month'}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Service Started</label>
              <select
                value={subscriptionStartMonth}
                onChange={(e) => setSubscriptionStartMonth(e.target.value)}
                className="w-full px-4 py-2 border rounded-lg focus:ring-2"
                style={{ borderColor: '#ced4da', backgroundColor: '#f8f9fa', color: '#212529' }}
              >
                {getYearOptions().map(year => 
                  getMonthOptions().map(month => (
                    <option key={`${year}-${month.value}`} value={`${year}-${month.value}`}>
                      {month.label} {year}
                    </option>
                  ))
                )}
              </select>
            </div>
            <Button
              onClick={handleLinkService}
              disabled={!selectedContactForService || !selectedServiceToLink || linkLoading}
              loading={linkLoading}
            >
              Link Service
            </Button>
          </div>

          {subscriptions.length > 0 ? (
            <div className="mt-6 overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-3 px-4 font-medium text-gray-600">Contact</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-600">Service</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-600">Amount</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-600">Since</th>
                    <th className="text-left py-3 px-4 font-medium text-gray-600">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {subscriptions.map((sub) => {
                    const contact = contacts.find(c => c.id === sub.contact_id)
                    const startedAt = sub.started_at ? new Date(sub.started_at).toISOString().slice(0, 7) : getCurrentMonth()
                    return (
                      <tr key={sub.id} className="border-b hover:bg-gray-50">
                        <td className="py-3 px-4">{contact?.name || 'Unknown'}</td>
                        <td className="py-3 px-4">{sub.service?.name || 'Unknown'}</td>
                        <td className="py-3 px-4">
                          {CURRENCY_SYMBOL}{sub.service?.amount || 0} 
                          <span className="text-xs text-gray-500">/ {sub.service?.payment_cycle || 'month'}</span>
                        </td>
                        <td className="py-3 px-4 text-sm text-gray-600">{startedAt}</td>
                        <td className="py-3 px-4">
                          <button 
                            onClick={() => handleUnlinkService(sub.id)} 
                            className="p-1.5 text-red-600 hover:bg-red-50 rounded"
                            title="Unlink"
                          >
                            <Unlink className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState message="No services linked to contacts yet." className="mt-6" />
          )}
        </Card>
      )}
    </div>
  )
}
