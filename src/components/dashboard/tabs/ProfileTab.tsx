'use client';

import { useState, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { createClient } from '@/lib/supabase/client';
import { User } from '@supabase/supabase-js';

interface ProfileTabProps {
  user: User;
}

export default function ProfileTab({ user }: ProfileTabProps) {
  const supabase = createClient();
  const queryClient = useQueryClient();
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [localUpdating, setLocalUpdating] = useState(false);
  const [localPasswordChanging, setLocalPasswordChanging] = useState(false);

  // Query for profile data
  const { data: profile } = useQuery({
    queryKey: ['profile'],
    queryFn: async () => {
      return user.user_metadata?.full_name || '';
    },
    initialData: user.user_metadata?.full_name || '',
  });

  // Update profile mutation
  const updateProfileMutation = useMutation({
    mutationFn: async (newName: string) => {
      const { error } = await supabase.auth.updateUser({
        data: {
          full_name: newName.trim(),
        }
      });
      if (error) throw error;
      return newName;
    },
    onSuccess: (data) => {
      queryClient.setQueryData(['profile'], data);
      setMessage({ type: 'success', text: 'Profile updated successfully!' });
    },
    onError: (error: any) => {
      setMessage({ type: 'error', text: error.message || 'Failed to update profile' });
    },
    onSettled: () => {
      setLocalUpdating(false);
    },
  });

  // Handle update with local state
  const handleProfileUpdate = useCallback((name: string) => {
    setLocalUpdating(true);
    setMessage(null);
    updateProfileMutation.mutate(name);
  }, [updateProfileMutation]);

  // Change password mutation
  const changePasswordMutation = useMutation({
    mutationFn: async ({ currentPassword, newPassword }: { currentPassword: string; newPassword: string }) => {
      const { error: signInError } = await supabase.auth.signInWithPassword({
        email: user.email!,
        password: currentPassword,
      });

      if (signInError) {
        throw new Error('Current password is incorrect');
      }

      const { error: updateError } = await supabase.auth.updateUser({
        password: newPassword,
      });

      if (updateError) throw updateError;
      return true;
    },
    onSuccess: () => {
      setMessage({ type: 'success', text: 'Password changed successfully!' });
    },
    onError: (error: any) => {
      setMessage({ type: 'error', text: error.message || 'Failed to change password' });
    },
    onSettled: () => {
      setLocalPasswordChanging(false);
    },
  });

  // Handle password change with local state
  const handlePasswordChange = useCallback((currentPassword: string, newPassword: string, confirmPassword: string) => {
    if (newPassword !== confirmPassword) {
      setMessage({ type: 'error', text: 'New passwords do not match' });
      return;
    }
    if (newPassword.length < 6) {
      setMessage({ type: 'error', text: 'Password must be at least 6 characters' });
      return;
    }
    
    setLocalPasswordChanging(true);
    setMessage(null);
    changePasswordMutation.mutate({ currentPassword, newPassword });
  }, [changePasswordMutation]);

  return (
    <ProfileForm
      user={user}
      profile={profile || ''}
      isUpdating={localUpdating}
      isChangingPassword={localPasswordChanging}
      message={message}
      onUpdate={handleProfileUpdate}
      onPasswordChange={handlePasswordChange}
    />
  );
}

// Separate form component
function ProfileForm({
  user,
  profile,
  isUpdating,
  isChangingPassword,
  message,
  onUpdate,
  onPasswordChange,
}: {
  user: User;
  profile: string;
  isUpdating: boolean;
  isChangingPassword: boolean;
  message: { type: 'success' | 'error'; text: string } | null;
  onUpdate: (name: string) => void;
  onPasswordChange: (current: string, newPass: string, confirm: string) => void;
}) {
  const [name, setName] = useState(profile);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  const handleSave = () => {
    onUpdate(name);
  };

  const handlePasswordClick = () => {
    onPasswordChange(currentPassword, newPassword, confirmPassword);
  };

  return (
    <div className="space-y-8">
      {message && (
        <div className={`p-4 rounded-lg ${message.type === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
          {message.text}
        </div>
      )}

      {/* Profile Section */}
      <div className="bg-white rounded-xl shadow-sm border p-6">
        <h3 className="text-lg font-semibold mb-4" style={{ color: '#212529' }}>Profile Information</h3>
        
        <div className="max-w-md">
          <div>
            <label className="block text-sm font-medium mb-2" style={{ color: '#212529' }}>
              Display Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter your name"
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              style={{ borderColor: '#dee2e6' }}
            />
            <p className="mt-1 text-sm" style={{ color: '#6c757d' }}>
              This is how your name will appear to team members
            </p>
          </div>
        </div>

        <div className="mt-6 flex justify-start">
          <button
            onClick={handleSave}
            disabled={isUpdating}
            className="px-6 py-2 rounded-lg font-medium text-white transition-colors disabled:opacity-50"
            style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}
          >
            {isUpdating ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>

      {/* Account Info Section */}
      <div className="bg-white rounded-xl shadow-sm border p-6">
        <h3 className="text-lg font-semibold mb-4" style={{ color: '#212529' }}>Account Information</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1" style={{ color: '#6c757d' }}>Email</label>
            <input
              type="email"
              value={user.email || ''}
              disabled
              className="w-full px-4 py-2 border rounded-lg bg-gray-50"
              style={{ borderColor: '#dee2e6' }}
            />
            <p className="mt-1 text-sm" style={{ color: '#6c757d' }}>Email cannot be changed</p>
          </div>
        </div>
      </div>

      {/* Password Section */}
      <div className="bg-white rounded-xl shadow-sm border p-6">
        <h3 className="text-lg font-semibold mb-4" style={{ color: '#212529' }}>Change Password</h3>
        
        <div className="space-y-4 max-w-md">
          <div>
            <label className="block text-sm font-medium mb-2" style={{ color: '#212529' }}>
              Current Password
            </label>
            <div className="relative">
              <input
                type={showPassword ? 'text' : 'password'}
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                placeholder="Enter current password"
                className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent pr-12"
                style={{ borderColor: '#dee2e6' }}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-sm"
                style={{ color: '#6c757d' }}
              >
                {showPassword ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2" style={{ color: '#212529' }}>
              New Password
            </label>
            <input
              type={showPassword ? 'text' : 'password'}
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="Enter new password"
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              style={{ borderColor: '#dee2e6' }}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2" style={{ color: '#212529' }}>
              Confirm New Password
            </label>
            <input
              type={showPassword ? 'text' : 'password'}
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Confirm new password"
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              style={{ borderColor: '#dee2e6' }}
            />
          </div>

          <div className="pt-2">
            <button
              onClick={handlePasswordClick}
              disabled={isChangingPassword || !currentPassword || !newPassword || !confirmPassword}
              className="px-6 py-2 rounded-lg font-medium text-white transition-colors disabled:opacity-50"
              style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}
            >
              {isChangingPassword ? 'Changing...' : 'Change Password'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
