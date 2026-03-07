'use client'

import { useState, useEffect } from 'react'
import { Card, EmptyState, LoadingState, Badge } from '../ui'
import { useNotifications, useContacts } from '@/hooks/useDashboardData'

const getCurrentMonth = () => {
  const now = new Date()
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`
}

export function HistoryTab() {
  const { notifications, loading: notificationsLoading, fetchNotifications, deleteNotification } = useNotifications()
  const { contacts, loading: contactsLoading, fetchContacts } = useContacts()
  const [filterMonth, setFilterMonth] = useState(getCurrentMonth())
  const [deleteModalOpen, setDeleteModalOpen] = useState(false)
  const [notificationToDelete, setNotificationToDelete] = useState<string | null>(null)

  // Fetch data on mount
  useEffect(() => {
    fetchNotifications()
    fetchContacts()
  }, [fetchNotifications, fetchContacts])

  const handleDeleteClick = (id: string) => {
    setNotificationToDelete(id)
    setDeleteModalOpen(true)
  }

  const handleConfirmDelete = async () => {
    if (notificationToDelete) {
      try {
        await deleteNotification(notificationToDelete)
      } catch (err) {
        console.error('Failed to delete notification:', err)
      }
    }
    setDeleteModalOpen(false)
    setNotificationToDelete(null)
  }

  return (
    <Card title="Notification History">
      <div className="flex flex-wrap items-center gap-4 mb-4">
        <div className="flex items-center gap-2">
          <label className="text-sm text-gray-600">Month:</label>
          <input
            type="month"
            value={filterMonth}
            onChange={(e) => setFilterMonth(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500"
          />
        </div>
        <button
          onClick={fetchNotifications}
          className="p-2 text-gray-400 hover:text-gray-600 ml-auto"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>

      {notificationsLoading ? (
        <LoadingState />
      ) : notifications.length === 0 ? (
        <EmptyState message="No notifications sent yet." />
      ) : (
        <div className="space-y-3">
          {notifications
            .filter((n) => {
              if (filterMonth) {
                const notificationDate = new Date(n.timestamp).toISOString().slice(0, 7)
                return notificationDate === filterMonth
              }
              return true
            })
            .map((notification) => {
              const contact = contacts.find(c => c.phone_number === notification.phone_number)
              const displayName = contact?.name || notification.phone_number
              return (
                <div key={notification.id} className="p-4 bg-gray-50 rounded-xl group">
                  <div className="flex items-start justify-between mb-2">
                    <span className="font-medium text-gray-900">{displayName}</span>
                    <div className="flex items-center gap-2">
                      <Badge variant={notification.status === 'sent' ? 'success' : notification.status === 'pending' ? 'warning' : 'error'}>
                        {notification.status}
                      </Badge>
                      <button
                        onClick={() => handleDeleteClick(notification.id)}
                        className="p-1 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mb-2">{notification.message}</p>
                  <p className="text-xs text-gray-400">{new Date(notification.timestamp).toLocaleString()}</p>
                </div>
              )
            })}
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteModalOpen && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex min-h-screen items-center justify-center p-4">
            <div className="fixed inset-0 bg-black/50" onClick={() => setDeleteModalOpen(false)} />
            <div className="relative bg-white rounded-2xl shadow-xl p-6 max-w-md w-full">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Delete Notification</h3>
              <p className="text-gray-600 mb-4">Are you sure you want to delete this notification? This action cannot be undone.</p>
              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => setDeleteModalOpen(false)}
                  className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmDelete}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}
