'use client'

import { useState, useEffect } from 'react'
import { useAuth } from '@/context/AuthContext'
import { UserRoleRecord, UserRole } from '@/types/database'
import { Users, Plus, Trash2, Edit2, Shield, AlertCircle } from 'lucide-react'

export function TeamMembersTab() {
  const { user, isAdmin, role } = useAuth()
  const [teamMembers, setTeamMembers] = useState<UserRoleRecord[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [showAddForm, setShowAddForm] = useState(false)
  const [newEmail, setNewEmail] = useState('')
  const [newName, setNewName] = useState('')
  const [newRole, setNewRole] = useState<UserRole>('editor')
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  useEffect(() => {
    if (isAdmin) {
      fetchTeamMembers()
    }
  }, [isAdmin])

  const fetchTeamMembers = async () => {
    try {
      setIsLoading(true)
      const response = await fetch('/api/user-roles')
      const data = await response.json()
      
      if (response.ok) {
        setTeamMembers(data.teamMembers || [])
      } else {
        setError(data.error || 'Failed to fetch team members')
      }
    } catch (err) {
      setError('Failed to fetch team members')
    } finally {
      setIsLoading(false)
    }
  }

  const handleAddMember = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setSuccess(null)

    if (!newEmail.trim()) {
      setError('Email is required')
      return
    }

    try {
      const response = await fetch('/api/user-roles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: newEmail, name: newName, role: newRole })
      })

      const data = await response.json()

      if (response.ok) {
        setSuccess(data.message)
        setNewEmail('')
        setNewName('')
        setNewRole('editor')
        setShowAddForm(false)
        fetchTeamMembers()
      } else {
        setError(data.error || 'Failed to add team member')
      }
    } catch (err) {
      setError('Failed to add team member')
    }
  }

  const handleDeleteMember = async (id: string) => {
    if (!confirm('Are you sure you want to remove this team member?')) return

    try {
      const response = await fetch(`/api/user-roles?id=${id}`, {
        method: 'DELETE'
      })

      if (response.ok) {
        setSuccess('Team member removed successfully')
        fetchTeamMembers()
      } else {
        const data = await response.json()
        setError(data.error || 'Failed to remove team member')
      }
    } catch (err) {
      setError('Failed to remove team member')
    }
  }

  const handleUpdateRole = async (id: string, newRole: UserRole) => {
    try {
      const response = await fetch('/api/user-roles', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, role: newRole })
      })

      if (response.ok) {
        setSuccess('Role updated successfully')
        fetchTeamMembers()
      } else {
        const data = await response.json()
        setError(data.error || 'Failed to update role')
      }
    } catch (err) {
      setError('Failed to update role')
    }
  }

  // Show access denied for non-admins
  if (!isAdmin) {
    return (
      <div 
        className="rounded-xl p-8 text-center"
        style={{ backgroundColor: '#e9ecef', border: '1px solid #ced4da' }}
      >
        <Shield className="w-12 h-12 mx-auto mb-4" style={{ color: '#212529' }} />
        <h3 className="text-lg font-semibold mb-2" style={{ color: '#212529' }}>
          Access Restricted
        </h3>
        <p style={{ color: '#212529' }}>
          Only admins can manage team members. Your current role: <strong>{role || 'admin'}</strong>
        </p>
      </div>
    )
  }

  const roleLabels: Record<UserRole, string> = {
    super_admin: 'Super Admin',
    admin: 'Admin',
    editor: 'Editor',
    viewer: 'Viewer'
  }

  const roleColors: Record<UserRole, string> = {
    super_admin: '#6f42c1',
    admin: '#dc3545',
    editor: '#28a745',
    viewer: '#212529'
  }

  return (
    <div>
      {/* Error/Success Messages */}
      {error && (
        <div 
          className="mb-6 p-4 rounded-xl flex items-center gap-3"
          style={{ backgroundColor: '#f8d7da', color: '#721c24', border: '1px solid #f5c6cb' }}
        >
          <AlertCircle className="w-5 h-5" />
          {error}
        </div>
      )}

      {success && (
        <div 
          className="mb-6 p-4 rounded-xl flex items-center gap-3"
          style={{ backgroundColor: '#d4edda', color: '#155724', border: '1px solid #c3e6cb' }}
        >
          <AlertCircle className="w-5 h-5" />
          {success}
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div 
            className="w-12 h-12 rounded-xl flex items-center justify-center"
            style={{ backgroundColor: '#dee2e6' }}
          >
            <Users className="w-6 h-6" style={{ color: '#212529' }} />
          </div>
          <div>
            <p className="text-sm" style={{ color: '#212529' }}>
              Manage your team and their access levels
            </p>
          </div>
        </div>

        <button
          onClick={() => setShowAddForm(!showAddForm)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors"
          style={{ backgroundColor: '#212529', color: '#ffffff' }}
        >
          <Plus className="w-4 h-4" />
          Add Team Member
        </button>
      </div>

      {/* Add Form */}
      {showAddForm && (
        <div 
          className="mb-6 p-6 rounded-xl"
          style={{ backgroundColor: '#ffffff', border: '1px solid #dee2e6' }}
        >
          <h3 className="text-lg font-semibold mb-4" style={{ color: '#212529' }}>
            Add New Team Member
          </h3>
          <form onSubmit={handleAddMember} className="space-y-4">
            <div>
              <label 
                htmlFor="name" 
                className="block text-sm font-medium mb-1"
                style={{ color: '#212529' }}
              >
                Name
              </label>
              <input
                type="text"
                id="name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="John Doe"
                className="w-full px-4 py-2.5 rounded-lg"
                style={{ 
                  backgroundColor: '#f8f9fa', 
                  border: '1px solid #ced4da', 
                  color: '#212529',
                  outline: 'none'
                }}
              />
            </div>

            <div>
              <label 
                htmlFor="email" 
                className="block text-sm font-medium mb-1"
                style={{ color: '#212529' }}
              >
                Email Address
              </label>
              <input
                type="email"
                id="email"
                value={newEmail}
                onChange={(e) => setNewEmail(e.target.value)}
                placeholder="colleague@example.com"
                className="w-full px-4 py-2.5 rounded-lg"
                style={{ 
                  backgroundColor: '#f8f9fa', 
                  border: '1px solid #ced4da', 
                  color: '#212529',
                  outline: 'none'
                }}
              />
            </div>

            <div>
              <label 
                htmlFor="role" 
                className="block text-sm font-medium mb-1"
                style={{ color: '#212529' }}
              >
                Role
              </label>
              <select
                id="role"
                value={newRole}
                onChange={(e) => setNewRole(e.target.value as UserRole)}
                className="w-full px-4 py-2.5 rounded-lg"
                style={{ 
                  backgroundColor: '#f8f9fa', 
                  border: '1px solid #ced4da', 
                  color: '#212529',
                  outline: 'none'
                }}
              >
                <option value="admin">Admin</option>
                <option value="editor">Editor</option>
                <option value="viewer">Viewer</option>
              </select>
            </div>

            <div className="flex gap-3">
              <button
                type="submit"
                className="px-4 py-2 rounded-lg font-medium transition-colors"
                style={{ backgroundColor: '#212529', color: '#ffffff' }}
              >
                Add Member
              </button>
              <button
                type="button"
                onClick={() => setShowAddForm(false)}
                className="px-4 py-2 rounded-lg font-medium transition-colors"
                style={{ backgroundColor: '#e9ecef', color: '#212529' }}
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Team Members List */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <div 
            className="w-8 h-8 border-4 animate-spin"
            style={{ borderColor: '#212529', borderTopColor: 'transparent', borderRadius: '50%' }}
          />
        </div>
      ) : teamMembers.length === 0 ? (
        <div 
          className="rounded-xl p-12 text-center"
          style={{ backgroundColor: '#e9ecef', border: '1px solid #ced4da' }}
        >
          <Users className="w-12 h-12 mx-auto mb-4" style={{ color: '#adb5bd' }} />
          <h3 className="text-lg font-semibold mb-2" style={{ color: '#212529' }}>
            No Team Members Yet
          </h3>
          <p style={{ color: '#212529' }}>
            Add team members to give them access to your dashboard
          </p>
        </div>
      ) : (
        <div 
          className="rounded-xl overflow-hidden"
          style={{ backgroundColor: '#ffffff', border: '1px solid #dee2e6' }}
        >
          <table className="w-full">
            <thead>
              <tr style={{ backgroundColor: '#f8f9fa', borderBottom: '1px solid #dee2e6' }}>
                <th 
                  className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider"
                  style={{ color: '#212529' }}
                >
                  Email
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider"
                  style={{ color: '#212529' }}
                >
                  Role
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider"
                  style={{ color: '#212529' }}
                >
                  Added On
                </th>
                <th 
                  className="px-6 py-3 text-right text-xs font-semibold uppercase tracking-wider"
                  style={{ color: '#212529' }}
                >
                  Actions
                </th>
              </tr>
            </thead>
            <tbody style={{ borderBottom: '1px solid #dee2e6' }}>
              {teamMembers.map((member) => (
                <tr key={member.id} style={{ borderBottom: '1px solid #dee2e6' }}>
                  <td className="px-6 py-4">
                    <p className="font-medium" style={{ color: '#212529' }}>
                      {member.name || member.email}
                    </p>
                    {member.name && (
                      <p className="text-sm" style={{ color: '#6c757d' }}>
                        {member.email}
                      </p>
                    )}
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                      style={{ 
                        backgroundColor: `${roleColors[member.role]}20`,
                        color: roleColors[member.role]
                      }}
                    >
                      {roleLabels[member.role]}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <p className="text-sm" style={{ color: '#212529' }}>
                      {new Date(member.created_at).toLocaleDateString()}
                    </p>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end gap-2">
                      {member.role !== 'super_admin' ? (
                        <>
                          <select
                            value={member.role}
                            onChange={(e) => handleUpdateRole(member.id, e.target.value as UserRole)}
                            className="px-2 py-1 rounded text-sm"
                            style={{ 
                              backgroundColor: '#f8f9fa', 
                              border: '1px solid #ced4da', 
                              color: '#212529'
                            }}
                          >
                            <option value="admin">Admin</option>
                            <option value="editor">Editor</option>
                            <option value="viewer">Viewer</option>
                          </select>
                          <button
                            onClick={() => handleDeleteMember(member.id)}
                            className="p-2 rounded-lg transition-colors"
                            style={{ color: '#dc3545' }}
                            title="Remove team member"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </>
                      ) : (
                        <span className="text-sm" style={{ color: '#6c757d' }}>
                          Protected
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Role Descriptions */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div 
          className="p-4 rounded-xl"
          style={{ backgroundColor: '#fff5f5', border: '1px solid #fed7d7' }}
        >
          <h4 className="font-semibold mb-1" style={{ color: '#c53030' }}>Super Admin</h4>
          <p className="text-sm" style={{ color: '#742a2a' }}>
            Full access to everything, can manage all admins
          </p>
        </div>
        <div 
          className="p-4 rounded-xl"
          style={{ backgroundColor: '#fffaf0', border: '1px solid #feebc8' }}
        >
          <h4 className="font-semibold mb-1" style={{ color: '#c05621' }}>Admin</h4>
          <p className="text-sm" style={{ color: '#7c2d12' }}>
            Can manage team members and all features
          </p>
        </div>
        <div 
          className="p-4 rounded-xl"
          style={{ backgroundColor: '#f0fff4', border: '1px solid #c6f6d5' }}
        >
          <h4 className="font-semibold mb-1" style={{ color: '#276749' }}>Editor</h4>
          <p className="text-sm" style={{ color: '#22543d' }}>
            Can edit content and access development tools
          </p>
        </div>
        <div 
          className="p-4 rounded-xl"
          style={{ backgroundColor: '#f8f9fa', border: '1px solid #e9ecef' }}
        >
          <h4 className="font-semibold mb-1" style={{ color: '#212529' }}>Viewer</h4>
          <p className="text-sm" style={{ color: '#212529' }}>
            Read-only access to view projects and prototypes
          </p>
        </div>
      </div>
    </div>
  )
}