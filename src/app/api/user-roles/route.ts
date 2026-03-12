import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import { UserRole } from '@/types/database'

// GET - Fetch all team members created by the current user
export async function GET(request: NextRequest) {
  const supabase = await createClient()
  
  // Get current user
  const { data: { user }, error: userError } = await supabase.auth.getUser()
  
  if (userError || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  // Check if current user is admin
  const { data: userRole } = await supabase
    .from('user_roles')
    .select('role')
    .eq('user_id', user.id)
    .single()

  // If not admin, only return their own info
  if (userRole?.role !== 'super_admin' && userRole?.role !== 'admin') {
    return NextResponse.json({ 
      error: 'Only admins can view team members' 
    }, { status: 403 })
  }

  // Get all users created by this admin
  const { data: teamMembers, error } = await supabase
    .from('user_roles')
    .select('*')
    .eq('created_by', user.id)
    .order('created_at', { ascending: false })

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  return NextResponse.json({ teamMembers })
}

// POST - Assign a role to an existing user by email
export async function POST(request: NextRequest) {
  const supabase = await createClient()
  
  // Get current user
  const { data: { user }, error: userError } = await supabase.auth.getUser()
  
  if (userError || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  // Check if current user is admin
  const { data: userRole } = await supabase
    .from('user_roles')
    .select('role')
    .eq('user_id', user.id)
    .single()

  if (userRole?.role !== 'super_admin' && userRole?.role !== 'admin') {
    return NextResponse.json({ 
      error: 'Only admins can assign roles to team members' 
    }, { status: 403 })
  }

  const body = await request.json()
  const { email, name, role } = body

  if (!email || !role) {
    return NextResponse.json({ 
      error: 'Email and role are required' 
    }, { status: 400 })
  }

  // Validate role
  const validRoles: UserRole[] = ['admin', 'editor', 'viewer']
  if (!validRoles.includes(role)) {
    return NextResponse.json({ 
      error: 'Invalid role. Must be one of: admin, editor, viewer' 
    }, { status: 400 })
  }

  try {
    // Check if user already has a role
    const { data: existingRole } = await supabase
      .from('user_roles')
      .select('*')
      .eq('email', email.toLowerCase())
      .single()

    if (existingRole) {
      // Update existing role
      const { error: updateError } = await supabase
        .from('user_roles')
        .update({ role, name: name || null })
        .eq('email', email.toLowerCase())

      if (updateError) {
        return NextResponse.json({ error: updateError.message }, { status: 500 })
      }

      return NextResponse.json({ 
        message: 'Role updated successfully',
        email,
        name,
        role
      })
    }

    // Create new role record
    const { error: insertError } = await supabase
      .from('user_roles')
      .insert({
        user_id: user.id,
        email: email.toLowerCase(),
        name: name || null,
        role,
        created_by: user.id
      })

    if (insertError) {
      return NextResponse.json({ error: insertError.message }, { status: 500 })
    }

    return NextResponse.json({ 
      message: 'Team member added. They need to sign up to activate their account.',
      email,
      role
    })
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}

// PUT - Update a team member's role
export async function PUT(request: NextRequest) {
  const supabase = await createClient()
  
  const { data: { user }, error: userError } = await supabase.auth.getUser()
  
  if (userError || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { data: userRole } = await supabase
    .from('user_roles')
    .select('role')
    .eq('user_id', user.id)
    .single()

  if (userRole?.role !== 'super_admin' && userRole?.role !== 'admin') {
    return NextResponse.json({ 
      error: 'Only admins can update team member roles' 
    }, { status: 403 })
  }

  const body = await request.json()
  const { id, role } = body

  if (!id || !role) {
    return NextResponse.json({ 
      error: 'ID and role are required' 
    }, { status: 400 })
  }

  // Check if the user being updated is a Super Admin
  const { data: targetUser } = await supabase
    .from('user_roles')
    .select('role')
    .eq('id', id)
    .single()

  if (targetUser?.role === 'super_admin') {
    return NextResponse.json({ 
      error: 'Cannot modify Super Admin role' 
    }, { status: 403 })
  }

  const validRoles: UserRole[] = ['admin', 'editor', 'viewer']
  if (!validRoles.includes(role)) {
    return NextResponse.json({ 
      error: 'Invalid role' 
    }, { status: 400 })
  }

  const { error: updateError } = await supabase
    .from('user_roles')
    .update({ role })
    .eq('id', id)
    .eq('created_by', user.id)

  if (updateError) {
    return NextResponse.json({ error: updateError.message }, { status: 500 })
  }

  return NextResponse.json({ message: 'Role updated successfully' })
}

// DELETE - Remove a team member
export async function DELETE(request: NextRequest) {
  const supabase = await createClient()
  
  const { data: { user }, error: userError } = await supabase.auth.getUser()
  
  if (userError || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { data: userRole } = await supabase
    .from('user_roles')
    .select('role')
    .eq('user_id', user.id)
    .single()

  if (userRole?.role !== 'super_admin' && userRole?.role !== 'admin') {
    return NextResponse.json({ 
      error: 'Only admins can remove team members' 
    }, { status: 403 })
  }

  const { searchParams } = new URL(request.url)
  const id = searchParams.get('id')

  if (!id) {
    return NextResponse.json({ 
      error: 'Team member ID is required' 
    }, { status: 400 })
  }

  // Check if the user being deleted is a Super Admin
  const { data: targetUser } = await supabase
    .from('user_roles')
    .select('role')
    .eq('id', id)
    .single()

  if (targetUser?.role === 'super_admin') {
    return NextResponse.json({ 
      error: 'Cannot delete Super Admin' 
    }, { status: 403 })
  }

  const { error: deleteError } = await supabase
    .from('user_roles')
    .delete()
    .eq('id', id)
    .eq('created_by', user.id)

  if (deleteError) {
    return NextResponse.json({ error: deleteError.message }, { status: 500 })
  }

  return NextResponse.json({ message: 'Team member removed successfully' })
}