'use client';

import { useState, useEffect, useCallback } from 'react';
import { 
  Plus, 
  CheckCircle2, 
  Circle, 
  Clock, 
  Trash2, 
  Edit3, 
  X,
  Filter,
  LayoutGrid,
  List,
  Tag,
  Calendar
} from 'lucide-react';
import HomeButton from '@/components/HomeButton';
import DetailsModal from '@/components/DetailsModal';
import prototypeLists from '@/utils/json/prototypeList.json';

interface Task {
  id: number;
  title: string;
  description: string | null;
  status: 'pending' | 'in-progress' | 'completed';
  priority: 'low' | 'medium' | 'high';
  category: string;
  dueDate: string | null;
  createdAt: string;
  updatedAt: string;
}

type ViewMode = 'board' | 'list';
type FilterStatus = 'all' | 'pending' | 'in-progress' | 'completed';

function TaskCard({ 
  task, 
  onToggle, 
  onEdit, 
  onDelete,
  getPriorityColor,
  getStatusIcon
}: {
  task: Task;
  onToggle: (task: Task) => void;
  onEdit: (task: Task) => void;
  onDelete: (id: number) => void;
  getPriorityColor: (priority: string) => string;
  getStatusIcon: (status: string) => React.ReactNode;
}) {
  return (
    <div className="bg-slate-700/50 rounded-xl p-4 hover:bg-slate-700/70 transition-colors group">
      <div className="flex items-start justify-between mb-2">
        <button onClick={() => onToggle(task)} className="flex-shrink-0">
          {getStatusIcon(task.status)}
        </button>
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => onEdit(task)}
            className="p-1.5 text-gray-400 hover:text-purple-400 transition-colors"
          >
            <Edit3 className="w-4 h-4" />
          </button>
          <button
            onClick={() => onDelete(task.id)}
            className="p-1.5 text-gray-400 hover:text-red-400 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>
      <h3 className={`font-semibold mb-2 ${task.status === 'completed' ? 'text-gray-400 line-through' : 'text-white'}`}>
        {task.title}
      </h3>
      {task.description && (
        <p className="text-gray-400 text-sm mb-3 line-clamp-2">{task.description}</p>
      )}
      <div className="flex items-center justify-between">
        <span className={`px-2 py-1 rounded-full text-xs ${getPriorityColor(task.priority)}`}>
          {task.priority}
        </span>
        {task.dueDate && (
          <span className="text-gray-500 text-xs flex items-center gap-1">
            <Calendar className="w-3 h-3" />
            {new Date(task.dueDate).toLocaleDateString()}
          </span>
        )}
      </div>
    </div>
  );
}

export default function TaskFlowPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('board');
  const [filterStatus, setFilterStatus] = useState<FilterStatus>('all');
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingTask, setEditingTask] = useState<Task | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [newTask, setNewTask] = useState({
    title: '',
    description: '',
    priority: 'medium' as 'low' | 'medium' | 'high',
    category: 'general',
    dueDate: ''
  });

  const fetchTasks = useCallback(async () => {
    try {
      const url = filterStatus === 'all' 
        ? '/api/taskflow' 
        : `/api/taskflow?status=${filterStatus}`;
      const response = await fetch(url);
      const data = await response.json();
      if (data.tasks) {
        setTasks(data.tasks);
      }
    } catch (error) {
      console.error('Error fetching tasks:', error);
    } finally {
      setLoading(false);
    }
  }, [filterStatus]);

  useEffect(() => {
    fetchTasks();
  }, [fetchTasks]);

  const handleCreateTask = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await fetch('/api/taskflow', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: newTask.title,
          description: newTask.description || null,
          priority: newTask.priority,
          category: newTask.category,
          dueDate: newTask.dueDate || null
        })
      });
      const data = await response.json();
      if (data.task) {
        setTasks([data.task, ...tasks]);
        setShowAddModal(false);
        setNewTask({
          title: '',
          description: '',
          priority: 'medium',
          category: 'general',
          dueDate: ''
        });
      }
    } catch (error) {
      console.error('Error creating task:', error);
    }
  };

  const handleToggleStatus = async (task: Task) => {
    const newStatus = task.status === 'completed' 
      ? 'pending' 
      : task.status === 'pending' 
        ? 'in-progress' 
        : 'completed';
    
    try {
      const response = await fetch('/api/taskflow', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: task.id, status: newStatus })
      });
      const data = await response.json();
      if (data.task) {
        setTasks(tasks.map(t => t.id === task.id ? data.task : t));
      }
    } catch (error) {
      console.error('Error updating task:', error);
    }
  };

  const handleDeleteTask = async (id: number) => {
    if (!confirm('Are you sure you want to delete this task?')) return;
    
    try {
      await fetch(`/api/taskflow?id=${id}`, { method: 'DELETE' });
      setTasks(tasks.filter(t => t.id !== id));
    } catch (error) {
      console.error('Error deleting task:', error);
    }
  };

  const handleUpdateTask = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingTask) return;

    try {
      const response = await fetch('/api/taskflow', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: editingTask.id,
          title: editingTask.title,
          description: editingTask.description,
          status: editingTask.status,
          priority: editingTask.priority,
          category: editingTask.category,
          dueDate: editingTask.dueDate
        })
      });
      const data = await response.json();
      if (data.task) {
        setTasks(tasks.map(t => t.id === editingTask.id ? data.task : t));
        setEditingTask(null);
      }
    } catch (error) {
      console.error('Error updating task:', error);
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'text-red-500 bg-red-50';
      case 'medium': return 'text-yellow-500 bg-yellow-50';
      case 'low': return 'text-green-500 bg-green-50';
      default: return 'text-gray-500 bg-gray-50';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle2 className="w-5 h-5 text-green-500" />;
      case 'in-progress': return <Clock className="w-5 h-5 text-yellow-500" />;
      default: return <Circle className="w-5 h-5 text-gray-400" />;
    }
  };

  const pendingTasks = tasks.filter(t => t.status === 'pending');
  const inProgressTasks = tasks.filter(t => t.status === 'in-progress');
  const completedTasks = tasks.filter(t => t.status === 'completed');

  // Get prototype details from JSON
  const prototype = prototypeLists.find(p => p.slug === 'taskflow');

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className='w-full flex flex-row justify-between'>
        {/* Home Button */}
        <div className="z-50">
          <HomeButton />
        </div>

        {/* Details Button */}
        <button
          onClick={() => setShowDetails(true)}
          className="z-50 p-2 bg-gray-800 hover:bg-gray-700 rounded-full transition-colors"
          title="View Details"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </button>
      </div>

      {/* Details Modal */}
      <DetailsModal
        isOpen={showDetails}
        onClose={() => setShowDetails(false)}
        title={prototype?.title || 'TaskFlow'}
        details={{
          problem: prototype?.problem || '',
          approach: prototype?.approach || '',
          challenges: prototype?.challenges || '',
          optimizations: prototype?.optimizations || '',
          improvements: prototype?.improvements || '',
        }}
      />

      <div className="max-w-7xl mx-auto pt-12">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">TaskFlow</h1>
            <p className="text-purple-300">Manage your tasks with persistent SQLite storage</p>
          </div>
          <button
            onClick={() => setShowAddModal(true)}
            className="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-xl font-semibold transition-all hover:scale-105"
          >
            <Plus className="w-5 h-5" />
            Add Task
          </button>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between mb-6 bg-slate-800/50 backdrop-blur-sm rounded-xl p-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-purple-400" />
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value as FilterStatus)}
                className="bg-slate-700 text-white px-4 py-2 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="all">All Tasks</option>
                <option value="pending">Pending</option>
                <option value="in-progress">In Progress</option>
                <option value="completed">Completed</option>
              </select>
            </div>
          </div>

          <div className="flex items-center gap-2 bg-slate-700 rounded-lg p-1">
            <button
              onClick={() => setViewMode('board')}
              className={`p-2 rounded-lg transition-colors ${viewMode === 'board' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'}`}
            >
              <LayoutGrid className="w-5 h-5" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded-lg transition-colors ${viewMode === 'list' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'}`}
            >
              <List className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Loading State */}
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
          </div>
        ) : viewMode === 'board' ? (
          /* Board View */
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Pending Column */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4">
              <div className="flex items-center gap-2 mb-4">
                <Circle className="w-5 h-5 text-gray-400" />
                <h2 className="text-lg font-semibold text-white">Pending</h2>
                <span className="bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full text-sm">{pendingTasks.length}</span>
              </div>
              <div className="space-y-3">
                {pendingTasks.map(task => (
                  <TaskCard 
                    key={task.id} 
                    task={task} 
                    onToggle={handleToggleStatus}
                    onEdit={setEditingTask}
                    onDelete={handleDeleteTask}
                    getPriorityColor={getPriorityColor}
                    getStatusIcon={getStatusIcon}
                  />
                ))}
              </div>
            </div>

            {/* In Progress Column */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4">
              <div className="flex items-center gap-2 mb-4">
                <Clock className="w-5 h-5 text-yellow-500" />
                <h2 className="text-lg font-semibold text-white">In Progress</h2>
                <span className="bg-yellow-500/20 text-yellow-400 px-2 py-0.5 rounded-full text-sm">{inProgressTasks.length}</span>
              </div>
              <div className="space-y-3">
                {inProgressTasks.map(task => (
                  <TaskCard 
                    key={task.id} 
                    task={task} 
                    onToggle={handleToggleStatus}
                    onEdit={setEditingTask}
                    onDelete={handleDeleteTask}
                    getPriorityColor={getPriorityColor}
                    getStatusIcon={getStatusIcon}
                  />
                ))}
              </div>
            </div>

            {/* Completed Column */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4">
              <div className="flex items-center gap-2 mb-4">
                <CheckCircle2 className="w-5 h-5 text-green-500" />
                <h2 className="text-lg font-semibold text-white">Completed</h2>
                <span className="bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full text-sm">{completedTasks.length}</span>
              </div>
              <div className="space-y-3">
                {completedTasks.map(task => (
                  <TaskCard 
                    key={task.id} 
                    task={task} 
                    onToggle={handleToggleStatus}
                    onEdit={setEditingTask}
                    onDelete={handleDeleteTask}
                    getPriorityColor={getPriorityColor}
                    getStatusIcon={getStatusIcon}
                  />
                ))}
              </div>
            </div>
          </div>
        ) : (
          /* List View */
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl overflow-hidden">
            <table className="w-full">
              <thead className="bg-slate-700/50">
                <tr>
                  <th className="text-left p-4 text-gray-300 font-medium">Status</th>
                  <th className="text-left p-4 text-gray-300 font-medium">Title</th>
                  <th className="text-left p-4 text-gray-300 font-medium">Priority</th>
                  <th className="text-left p-4 text-gray-300 font-medium">Category</th>
                  <th className="text-left p-4 text-gray-300 font-medium">Due Date</th>
                  <th className="text-left p-4 text-gray-300 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {tasks.map(task => (
                  <tr key={task.id} className="border-t border-slate-700/50 hover:bg-slate-700/30">
                    <td className="p-4">
                      <button onClick={() => handleToggleStatus(task)}>
                        {getStatusIcon(task.status)}
                      </button>
                    </td>
                    <td className="p-4 text-white font-medium">{task.title}</td>
                    <td className="p-4">
                      <span className={`px-3 py-1 rounded-full text-sm ${getPriorityColor(task.priority)}`}>
                        {task.priority}
                      </span>
                    </td>
                    <td className="p-4 text-gray-300">
                      <span className="flex items-center gap-1">
                        <Tag className="w-4 h-4" />
                        {task.category}
                      </span>
                    </td>
                    <td className="p-4 text-gray-300">
                      {task.dueDate ? (
                        <span className="flex items-center gap-1">
                          <Calendar className="w-4 h-4" />
                          {new Date(task.dueDate).toLocaleDateString()}
                        </span>
                      ) : '-'}
                    </td>
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setEditingTask(task)}
                          className="p-2 text-gray-400 hover:text-purple-400 transition-colors"
                        >
                          <Edit3 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDeleteTask(task.id)}
                          className="p-2 text-gray-400 hover:text-red-400 transition-colors"
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

        {/* Empty State */}
        {!loading && tasks.length === 0 && (
          <div className="text-center py-16">
            <div className="w-24 h-24 mx-auto mb-4 bg-slate-800/50 rounded-full flex items-center justify-center">
              <Plus className="w-12 h-12 text-purple-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">No tasks yet</h3>
            <p className="text-gray-400 mb-4">Create your first task to get started</p>
            <button
              onClick={() => setShowAddModal(true)}
              className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-2 rounded-lg font-semibold transition-colors"
            >
              Add Task
            </button>
          </div>
        )}
      </div>

      {/* Add Task Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-2xl p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white">Add New Task</h2>
              <button onClick={() => setShowAddModal(false)} className="text-gray-400 hover:text-white">
                <X className="w-6 h-6" />
              </button>
            </div>
            <form onSubmit={handleCreateTask} className="space-y-4">
              <div>
                <label className="block text-gray-300 mb-2">Title</label>
                <input
                  type="text"
                  value={newTask.title}
                  onChange={(e) => setNewTask({ ...newTask, title: e.target.value })}
                  className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder="Enter task title"
                  required
                />
              </div>
              <div>
                <label className="block text-gray-300 mb-2">Description</label>
                <textarea
                  value={newTask.description}
                  onChange={(e) => setNewTask({ ...newTask, description: e.target.value })}
                  className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder="Enter task description"
                  rows={3}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-300 mb-2">Priority</label>
                  <select
                    value={newTask.priority}
                    onChange={(e) => setNewTask({ ...newTask, priority: e.target.value as 'low' | 'medium' | 'high' })}
                    className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>
                <div>
                  <label className="block text-gray-300 mb-2">Category</label>
                  <input
                    type="text"
                    value={newTask.category}
                    onChange={(e) => setNewTask({ ...newTask, category: e.target.value })}
                    className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="e.g., work, personal"
                  />
                </div>
              </div>
              <div>
                <label className="block text-gray-300 mb-2">Due Date</label>
                <input
                  type="date"
                  value={newTask.dueDate}
                  onChange={(e) => setNewTask({ ...newTask, dueDate: e.target.value })}
                  className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>
              <button
                type="submit"
                className="w-full bg-purple-600 hover:bg-purple-700 text-white py-3 rounded-lg font-semibold transition-colors"
              >
                Create Task
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Edit Task Modal */}
      {editingTask && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-2xl p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white">Edit Task</h2>
              <button onClick={() => setEditingTask(null)} className="text-gray-400 hover:text-white">
                <X className="w-6 h-6" />
              </button>
            </div>
            <form onSubmit={handleUpdateTask} className="space-y-4">
              <div>
                <label className="block text-gray-300 mb-2">Title</label>
                <input
                  type="text"
                  value={editingTask.title}
                  onChange={(e) => setEditingTask({ ...editingTask, title: e.target.value })}
                  className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  required
                />
              </div>
              <div>
                <label className="block text-gray-300 mb-2">Description</label>
                <textarea
                  value={editingTask.description || ''}
                  onChange={(e) => setEditingTask({ ...editingTask, description: e.target.value })}
                  className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  rows={3}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-300 mb-2">Status</label>
                  <select
                    value={editingTask.status}
                    onChange={(e) => setEditingTask({ ...editingTask, status: e.target.value as Task['status'] })}
                    className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="pending">Pending</option>
                    <option value="in-progress">In Progress</option>
                    <option value="completed">Completed</option>
                  </select>
                </div>
                <div>
                  <label className="block text-gray-300 mb-2">Priority</label>
                  <select
                    value={editingTask.priority}
                    onChange={(e) => setEditingTask({ ...editingTask, priority: e.target.value as Task['priority'] })}
                    className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-300 mb-2">Category</label>
                  <input
                    type="text"
                    value={editingTask.category}
                    onChange={(e) => setEditingTask({ ...editingTask, category: e.target.value })}
                    className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-gray-300 mb-2">Due Date</label>
                  <input
                    type="date"
                    value={editingTask.dueDate || ''}
                    onChange={(e) => setEditingTask({ ...editingTask, dueDate: e.target.value })}
                    className="w-full bg-slate-700 text-white px-4 py-3 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
              </div>
              <button
                type="submit"
                className="w-full bg-purple-600 hover:bg-purple-700 text-white py-3 rounded-lg font-semibold transition-colors"
              >
                Update Task
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
