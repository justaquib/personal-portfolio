'use client';

import { FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  ArrowDownWideNarrow,
  CalendarDays,
  Check,
  CheckCircle2,
  Circle,
  Clock3,
  Filter,
  LayoutGrid,
  List,
  Plus,
  Search,
  Sparkles,
  Tag,
  Trash2,
  X,
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
type FilterStatus = 'all' | Task['status'];
type Priority = Task['priority'];

const emptyDraft = {
  title: '',
  description: '',
  status: 'pending' as Task['status'],
  priority: 'medium' as Priority,
  category: 'general',
  dueDate: '',
};

const statusLabels: Record<Task['status'], string> = {
  pending: 'To do',
  'in-progress': 'In progress',
  completed: 'Completed',
};

function formatDate(date: string) {
  return new Date(`${date.slice(0, 10)}T00:00:00`).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

function priorityStyles(priority: Priority) {
  if (priority === 'high') return 'bg-[#fff0ec] text-[#a94e38]';
  if (priority === 'medium') return 'bg-[#f8f2df] text-[#8a6a25]';
  return 'bg-[#eaf3ec] text-[#487354]';
}

function StatusMark({ status }: { status: Task['status'] }) {
  if (status === 'completed') return <CheckCircle2 className="h-[18px] w-[18px] text-[#4f8a60]" />;
  if (status === 'in-progress') return <Clock3 className="h-[18px] w-[18px] text-[#9b7834]" />;
  return <Circle className="h-[18px] w-[18px] text-[#9aa99e]" />;
}

function TaskCard({
  task,
  onToggle,
  onEdit,
  onDelete,
}: {
  task: Task;
  onToggle: (task: Task) => void;
  onEdit: (task: Task) => void;
  onDelete: (id: number) => void;
}) {
  return (
    <article className="group rounded-xl border border-[#e2e9e3] bg-white p-4 shadow-[0_5px_18px_-16px_rgba(29,62,41,.45)] transition hover:border-[#c8d9ca] hover:shadow-[0_12px_30px_-20px_rgba(29,62,41,.35)]">
      <div className="flex items-start gap-3">
        <button
          type="button"
          onClick={() => onToggle(task)}
          aria-label={`Move “${task.title}” to ${task.status === 'completed' ? 'to do' : 'completed'}`}
          title={task.status === 'completed' ? 'Mark as to do' : 'Mark completed'}
          className="mt-0.5 shrink-0 rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#94b39a]"
        >
          <StatusMark status={task.status} />
        </button>
        <div className="min-w-0 flex-1">
          <h3 className={`break-words text-sm font-semibold leading-5 ${task.status === 'completed' ? 'text-[#8a9990] line-through' : 'text-[#263b2e]'}`}>{task.title}</h3>
          {task.description && <p className="mt-1.5 line-clamp-2 text-xs leading-5 text-[#78877d]">{task.description}</p>}
        </div>
        <div className="flex shrink-0 items-center gap-0.5 opacity-100 transition sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100">
          <button type="button" onClick={() => onEdit(task)} aria-label={`Edit ${task.title}`} className="rounded-lg p-1.5 text-[#829087] hover:bg-[#f0f5f0] hover:text-[#315d3d]"><span className="text-xs font-medium">Edit</span></button>
          <button type="button" onClick={() => onDelete(task.id)} aria-label={`Delete ${task.title}`} className="rounded-lg p-1.5 text-[#9b9993] hover:bg-[#fff2ef] hover:text-[#a94e38]"><Trash2 className="h-4 w-4" /></button>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2 pl-7">
        <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold capitalize tracking-wide ${priorityStyles(task.priority)}`}>{task.priority} priority</span>
        {task.category && task.category !== 'general' && <span className="inline-flex items-center gap-1 text-[11px] text-[#79877d]"><Tag className="h-3 w-3" />{task.category}</span>}
        {task.dueDate && <span className="ml-auto inline-flex items-center gap-1 text-[11px] text-[#79877d]"><CalendarDays className="h-3 w-3" />{formatDate(task.dueDate)}</span>}
      </div>
    </article>
  );
}

export default function TaskFlowPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [actionError, setActionError] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('board');
  const [filterStatus, setFilterStatus] = useState<FilterStatus>('all');
  const [priorityFilter, setPriorityFilter] = useState<'all' | Priority>('all');
  const [query, setQuery] = useState('');
  const [showTaskModal, setShowTaskModal] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [draft, setDraft] = useState(emptyDraft);
  const [showDetails, setShowDetails] = useState(false);

  const fetchTasks = useCallback(async () => {
    setLoading(true);
    setLoadError('');
    try {
      const response = await fetch('/api/taskflow');
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not load tasks.');
      setTasks(Array.isArray(data.tasks) ? data.tasks : []);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : 'Could not load tasks.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void fetchTasks(); }, [fetchTasks]);

  const filteredTasks = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return tasks.filter((task) => {
      const matchesStatus = filterStatus === 'all' || task.status === filterStatus;
      const matchesPriority = priorityFilter === 'all' || task.priority === priorityFilter;
      const matchesQuery = !normalizedQuery || `${task.title} ${task.description || ''} ${task.category}`.toLowerCase().includes(normalizedQuery);
      return matchesStatus && matchesPriority && matchesQuery;
    });
  }, [tasks, filterStatus, priorityFilter, query]);

  const stats = useMemo(() => ({
    total: tasks.length,
    active: tasks.filter((task) => task.status !== 'completed').length,
    inProgress: tasks.filter((task) => task.status === 'in-progress').length,
    completed: tasks.filter((task) => task.status === 'completed').length,
  }), [tasks]);

  const openCreateModal = () => {
    setEditingId(null);
    setDraft(emptyDraft);
    setActionError('');
    setShowTaskModal(true);
  };

  const openEditModal = (task: Task) => {
    setEditingId(task.id);
    setDraft({
      title: task.title,
      description: task.description || '',
      status: task.status,
      priority: task.priority,
      category: task.category || 'general',
      dueDate: task.dueDate?.slice(0, 10) || '',
    });
    setActionError('');
    setShowTaskModal(true);
  };

  const closeTaskModal = () => {
    if (isSaving) return;
    setShowTaskModal(false);
    setEditingId(null);
  };

  const handleSaveTask = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsSaving(true);
    setActionError('');
    try {
      const response = await fetch('/api/taskflow', {
        method: editingId === null ? 'POST' : 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editingId === null ? {
          ...draft,
          description: draft.description.trim() || null,
          dueDate: draft.dueDate || null,
        } : {
          id: editingId,
          ...draft,
          description: draft.description.trim() || null,
          dueDate: draft.dueDate || null,
        }),
      });
      const data = await response.json();
      if (!response.ok || !data.task) throw new Error(data.error || 'Could not save this task.');
      setTasks((current) => editingId === null
        ? [data.task, ...current]
        : current.map((task) => task.id === editingId ? data.task : task));
      setShowTaskModal(false);
      setEditingId(null);
      setDraft(emptyDraft);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Could not save this task.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleToggleStatus = async (task: Task) => {
    const status: Task['status'] = task.status === 'completed' ? 'pending' : 'completed';
    setActionError('');
    try {
      const response = await fetch('/api/taskflow', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: task.id, status }),
      });
      const data = await response.json();
      if (!response.ok || !data.task) throw new Error(data.error || 'Could not update task status.');
      setTasks((current) => current.map((item) => item.id === task.id ? data.task : item));
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Could not update task status.');
    }
  };

  const handleDeleteTask = async (id: number) => {
    if (!window.confirm('Delete this task? This action cannot be undone.')) return;
    setActionError('');
    try {
      const response = await fetch(`/api/taskflow?id=${id}`, { method: 'DELETE' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not delete this task.');
      setTasks((current) => current.filter((task) => task.id !== id));
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Could not delete this task.');
    }
  };

  const columns: { status: Task['status']; icon: typeof Circle; color: string; tasks: Task[] }[] = [
    { status: 'pending', icon: Circle, color: 'text-[#85948a]', tasks: filteredTasks.filter((task) => task.status === 'pending') },
    { status: 'in-progress', icon: Clock3, color: 'text-[#9b7834]', tasks: filteredTasks.filter((task) => task.status === 'in-progress') },
    { status: 'completed', icon: CheckCircle2, color: 'text-[#4f8a60]', tasks: filteredTasks.filter((task) => task.status === 'completed') },
  ];
  const prototype = prototypeLists.find((item) => item.slug === 'taskflow');

  return (
    <main className="min-h-screen bg-[#f5f6f2] text-[#20352a]">
      <header className="border-b border-[#e2e6df] bg-white/90 backdrop-blur-xl">
        <div className="mx-auto flex h-[68px] max-w-[1440px] items-center justify-between px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-[#234a34] p-2 text-[#eff5ea]"><Check className="h-4 w-4" /></div>
            <div><p className="text-sm font-semibold tracking-tight text-[#263c2e]">TaskFlow</p><p className="text-[10px] text-[#8b978d]">A focused task workspace</p></div>
          </div>
          <div className="flex items-center gap-2">
            <div className="rounded-full border border-[#dce5dc] bg-white p-1"><HomeButton /></div>
            <button onClick={() => setShowDetails(true)} className="grid h-9 w-9 place-items-center rounded-full border border-[#dce5dc] bg-white text-[#6e7f72] transition hover:bg-[#f2f7f1] hover:text-[#345c3e]" title="About TaskFlow" aria-label="About TaskFlow"><Sparkles className="h-4 w-4" /></button>
          </div>
        </div>
      </header>

      <DetailsModal
        isOpen={showDetails}
        onClose={() => setShowDetails(false)}
        title={prototype?.title || 'TaskFlow'}
        details={{ problem: prototype?.problem || '', approach: prototype?.approach || '', challenges: prototype?.challenges || '', optimizations: prototype?.optimizations || '', improvements: prototype?.improvements || '' }}
      />

      <div className="mx-auto max-w-[1440px] px-4 pb-12 pt-7 sm:px-6 sm:pt-9 lg:px-8">
        <section className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
          <div>
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-[#718978]">Personal workspace</p>
            <h1 className="text-3xl font-semibold tracking-[-0.045em] text-[#20382b] sm:text-4xl">Your tasks, in flow.</h1>
            <p className="mt-2 text-sm text-[#77857b]">Keep priorities clear and make steady progress.</p>
          </div>
          <button onClick={openCreateModal} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-[#254f36] px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-[#1d412c] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#8fb49a] focus-visible:ring-offset-2">
            <Plus className="h-4 w-4" /> Add task
          </button>
        </section>

        <section aria-label="Task summary" className="mt-7 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { label: 'Total tasks', value: stats.total },
            { label: 'Active', value: stats.active },
            { label: 'In progress', value: stats.inProgress },
            { label: 'Completed', value: stats.completed },
          ].map((stat) => <div key={stat.label} className="rounded-xl border border-[#e2e8e0] bg-white px-4 py-3.5"><p className="text-[11px] font-medium text-[#839086]">{stat.label}</p><p className="mt-1 text-xl font-semibold tracking-tight text-[#2c4934]">{stat.value}</p></div>)}
        </section>

        <section className="mt-6 rounded-2xl border border-[#e1e7e0] bg-white p-3 sm:p-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <label className="flex h-10 min-w-0 flex-1 items-center gap-2 rounded-xl border border-[#e3e9e3] bg-[#fbfcfa] px-3 text-[#87948a] focus-within:border-[#a8c2a8] focus-within:ring-2 focus-within:ring-[#e7f0e6] lg:max-w-md">
              <Search className="h-4 w-4 shrink-0" />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search tasks, notes, categories…" className="min-w-0 flex-1 bg-transparent text-sm text-[#334a3a] outline-none placeholder:text-[#a0aaa2]" />
              {query && <button type="button" onClick={() => setQuery('')} aria-label="Clear search" className="rounded p-1 hover:bg-[#eef3ee]"><X className="h-3.5 w-3.5" /></button>}
            </label>
            <div className="flex flex-wrap items-center gap-2">
              <label className="flex h-10 items-center gap-2 rounded-xl border border-[#e3e9e3] bg-white px-3 text-[#718176]"><Filter className="h-3.5 w-3.5" /><select aria-label="Filter by status" value={filterStatus} onChange={(event) => setFilterStatus(event.target.value as FilterStatus)} className="bg-transparent text-xs font-medium text-[#435749] outline-none"><option value="all">All statuses</option><option value="pending">To do</option><option value="in-progress">In progress</option><option value="completed">Completed</option></select></label>
              <label className="flex h-10 items-center gap-2 rounded-xl border border-[#e3e9e3] bg-white px-3 text-[#718176]"><ArrowDownWideNarrow className="h-3.5 w-3.5" /><select aria-label="Filter by priority" value={priorityFilter} onChange={(event) => setPriorityFilter(event.target.value as 'all' | Priority)} className="bg-transparent text-xs font-medium text-[#435749] outline-none"><option value="all">All priorities</option><option value="high">High priority</option><option value="medium">Medium priority</option><option value="low">Low priority</option></select></label>
              <div className="flex h-10 items-center rounded-xl border border-[#e3e9e3] bg-[#f8faf7] p-1">
                <button onClick={() => setViewMode('board')} aria-label="Board view" aria-pressed={viewMode === 'board'} className={`grid h-8 w-8 place-items-center rounded-lg transition ${viewMode === 'board' ? 'bg-[#e5efe4] text-[#315c3c]' : 'text-[#8a968d] hover:text-[#425c49]'}`}><LayoutGrid className="h-4 w-4" /></button>
                <button onClick={() => setViewMode('list')} aria-label="List view" aria-pressed={viewMode === 'list'} className={`grid h-8 w-8 place-items-center rounded-lg transition ${viewMode === 'list' ? 'bg-[#e5efe4] text-[#315c3c]' : 'text-[#8a968d] hover:text-[#425c49]'}`}><List className="h-4 w-4" /></button>
              </div>
            </div>
          </div>
          {actionError && <div role="alert" className="mt-3 flex items-start gap-2 rounded-xl border border-[#efd2c9] bg-[#fff6f3] px-3 py-2.5 text-xs text-[#984d3d]"><AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />{actionError}<button onClick={() => setActionError('')} className="ml-auto shrink-0 underline">Dismiss</button></div>}
        </section>

        {loading ? (
          <div className="mt-8 grid min-h-64 place-items-center rounded-2xl border border-[#e2e8e0] bg-white"><div className="flex items-center gap-3 text-sm text-[#75847a]"><span className="h-5 w-5 animate-spin rounded-full border-2 border-[#dbe7dc] border-t-[#4f8057]" />Loading your tasks…</div></div>
        ) : loadError ? (
          <div role="alert" className="mt-8 rounded-2xl border border-[#efd2c9] bg-white p-8 text-center"><AlertCircle className="mx-auto h-7 w-7 text-[#ae5a45]" /><h2 className="mt-3 font-semibold text-[#3b493f]">Tasks didn’t load</h2><p className="mt-1 text-sm text-[#829087]">{loadError}</p><button onClick={() => void fetchTasks()} className="mt-4 rounded-lg bg-[#254f36] px-4 py-2 text-sm font-medium text-white hover:bg-[#1d412c]">Try again</button></div>
        ) : filteredTasks.length === 0 ? (
          <div className="mt-8 rounded-2xl border border-dashed border-[#cfdacf] bg-white px-5 py-16 text-center">
            <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-[#edf4ec] text-[#4f8057]"><CheckCircle2 className="h-5 w-5" /></div>
            <h2 className="mt-4 text-base font-semibold text-[#334a38]">{tasks.length === 0 ? 'Start with one clear next step' : 'No tasks match these filters'}</h2>
            <p className="mx-auto mt-1 max-w-sm text-sm leading-6 text-[#829087]">{tasks.length === 0 ? 'Create a task and keep the details, priority, and due date in one place.' : 'Try a different search or clear one of your filters.'}</p>
            {tasks.length === 0 ? <button onClick={openCreateModal} className="mt-5 inline-flex items-center gap-2 rounded-xl bg-[#254f36] px-4 py-2.5 text-sm font-semibold text-white hover:bg-[#1d412c]"><Plus className="h-4 w-4" />Add your first task</button> : <button onClick={() => { setQuery(''); setFilterStatus('all'); setPriorityFilter('all'); }} className="mt-4 text-sm font-semibold text-[#3c7049] hover:underline">Clear filters</button>}
          </div>
        ) : viewMode === 'board' ? (
          <section aria-label="Task board" className="mt-5 grid items-start gap-4 md:grid-cols-3">
            {columns.map(({ status, icon: Icon, color, tasks: columnTasks }) => (
              <div key={status} className="min-w-0 rounded-2xl border border-[#e2e8e0] bg-[#eef1eb] p-3 sm:p-4">
                <div className="mb-3 flex items-center gap-2 px-1">
                  <Icon className={`h-4 w-4 ${color}`} />
                  <h2 className="text-xs font-semibold text-[#435448]">{statusLabels[status]}</h2>
                  <span className="ml-auto rounded-full bg-white px-2 py-0.5 text-[10px] font-semibold text-[#78877b]">{columnTasks.length}</span>
                </div>
                <div className="space-y-2.5">
                  {columnTasks.length > 0 ? columnTasks.map((task) => <TaskCard key={task.id} task={task} onToggle={handleToggleStatus} onEdit={openEditModal} onDelete={handleDeleteTask} />) : <p className="rounded-xl border border-dashed border-[#d3ddd2] px-3 py-5 text-center text-xs text-[#96a197]">Nothing here yet</p>}
                </div>
              </div>
            ))}
          </section>
        ) : (
          <div className="mt-5 overflow-hidden rounded-2xl border border-[#e2e8e0] bg-white">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-left">
                <thead className="bg-[#f8faf7] text-[10px] uppercase tracking-[0.12em] text-[#859188]"><tr><th className="px-4 py-3 font-semibold">Task</th><th className="px-4 py-3 font-semibold">Status</th><th className="px-4 py-3 font-semibold">Priority</th><th className="px-4 py-3 font-semibold">Category</th><th className="px-4 py-3 font-semibold">Due date</th><th className="px-4 py-3 text-right font-semibold">Actions</th></tr></thead>
                <tbody className="divide-y divide-[#edf0ec]">
                  {filteredTasks.map((task) => <tr key={task.id} className="transition hover:bg-[#fbfcfa]"><td className="px-4 py-3.5"><div className="flex items-start gap-2.5"><button onClick={() => handleToggleStatus(task)} aria-label={`Toggle ${task.title} completion`} className="mt-0.5"><StatusMark status={task.status} /></button><div><p className={`text-sm font-medium ${task.status === 'completed' ? 'text-[#8a9990] line-through' : 'text-[#344a39]'}`}>{task.title}</p>{task.description && <p className="mt-1 max-w-sm truncate text-xs text-[#849087]">{task.description}</p>}</div></div></td><td className="px-4 py-3.5 text-xs text-[#68786d]">{statusLabels[task.status]}</td><td className="px-4 py-3.5"><span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold capitalize ${priorityStyles(task.priority)}`}>{task.priority}</span></td><td className="px-4 py-3.5 text-xs text-[#68786d]">{task.category || 'General'}</td><td className="px-4 py-3.5 text-xs text-[#68786d]">{task.dueDate ? formatDate(task.dueDate) : '—'}</td><td className="px-4 py-3.5"><div className="flex justify-end gap-1"><button onClick={() => openEditModal(task)} className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-[#52715a] hover:bg-[#eff5ef]">Edit</button><button onClick={() => handleDeleteTask(task.id)} aria-label={`Delete ${task.title}`} className="rounded-lg p-1.5 text-[#929b93] hover:bg-[#fff2ef] hover:text-[#a94e38]"><Trash2 className="h-4 w-4" /></button></div></td></tr>)}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {showTaskModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#17251f]/45 p-3 backdrop-blur-sm sm:p-6" onMouseDown={(event) => { if (event.target === event.currentTarget) closeTaskModal(); }}>
          <section role="dialog" aria-modal="true" aria-labelledby="task-modal-title" className="flex max-h-[92dvh] w-full max-w-xl flex-col overflow-hidden rounded-2xl border border-[#dce6df] bg-[#fbfcfa] shadow-[0_28px_90px_-28px_rgba(12,34,24,.42)]">
            <header className="flex items-center justify-between border-b border-[#e3eae4] bg-white px-5 py-4 sm:px-6"><div><p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#6e8c74]">Task details</p><h2 id="task-modal-title" className="mt-1 text-lg font-semibold tracking-tight text-[#263d2e]">{editingId === null ? 'Create a task' : 'Edit task'}</h2></div><button onClick={closeTaskModal} disabled={isSaving} aria-label="Close task form" className="rounded-lg p-2 text-[#7b897f] hover:bg-[#f0f5f0]"><X className="h-5 w-5" /></button></header>
            <form onSubmit={handleSaveTask} className="min-h-0 space-y-4 overflow-y-auto p-5 sm:p-6">
              <label className="block"><span className="mb-1.5 block text-xs font-semibold text-[#53665a]">Task name</span><input autoFocus required maxLength={160} value={draft.title} onChange={(event) => setDraft({ ...draft, title: event.target.value })} placeholder="What needs to get done?" className="h-11 w-full rounded-xl border border-[#dfe7df] bg-white px-3.5 text-sm text-[#33483a] outline-none placeholder:text-[#a2aca4] focus:border-[#a5c0a7] focus:ring-2 focus:ring-[#e8f0e7]" /></label>
              <label className="block"><span className="mb-1.5 block text-xs font-semibold text-[#53665a]">Notes <span className="font-normal text-[#9aa49c]">(optional)</span></span><textarea rows={3} maxLength={1500} value={draft.description} onChange={(event) => setDraft({ ...draft, description: event.target.value })} placeholder="Add useful context…" className="w-full resize-y rounded-xl border border-[#dfe7df] bg-white px-3.5 py-3 text-sm leading-5 text-[#33483a] outline-none placeholder:text-[#a2aca4] focus:border-[#a5c0a7] focus:ring-2 focus:ring-[#e8f0e7]" /></label>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <label className="block"><span className="mb-1.5 block text-xs font-semibold text-[#53665a]">Status</span><select value={draft.status} onChange={(event) => setDraft({ ...draft, status: event.target.value as Task['status'] })} className="h-11 w-full rounded-xl border border-[#dfe7df] bg-white px-3 text-sm text-[#405546] outline-none focus:border-[#a5c0a7]"><option value="pending">To do</option><option value="in-progress">In progress</option><option value="completed">Completed</option></select></label>
                <label className="block"><span className="mb-1.5 block text-xs font-semibold text-[#53665a]">Priority</span><select value={draft.priority} onChange={(event) => setDraft({ ...draft, priority: event.target.value as Priority })} className="h-11 w-full rounded-xl border border-[#dfe7df] bg-white px-3 text-sm text-[#405546] outline-none focus:border-[#a5c0a7]"><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label>
                <label className="block"><span className="mb-1.5 block text-xs font-semibold text-[#53665a]">Category</span><input maxLength={50} value={draft.category} onChange={(event) => setDraft({ ...draft, category: event.target.value })} placeholder="General" className="h-11 w-full rounded-xl border border-[#dfe7df] bg-white px-3 text-sm text-[#405546] outline-none placeholder:text-[#a2aca4] focus:border-[#a5c0a7]" /></label>
                <label className="block"><span className="mb-1.5 block text-xs font-semibold text-[#53665a]">Due date <span className="font-normal text-[#9aa49c]">(optional)</span></span><input type="date" value={draft.dueDate} onChange={(event) => setDraft({ ...draft, dueDate: event.target.value })} className="h-11 w-full rounded-xl border border-[#dfe7df] bg-white px-3 text-sm text-[#405546] outline-none focus:border-[#a5c0a7]" /></label>
              </div>
              {actionError && <p role="alert" className="rounded-lg border border-[#efd2c9] bg-[#fff6f3] px-3 py-2 text-xs text-[#984d3d]">{actionError}</p>}
              <div className="flex justify-end gap-2 border-t border-[#e8ede8] pt-4"><button type="button" onClick={closeTaskModal} disabled={isSaving} className="rounded-xl border border-[#dce5dc] bg-white px-4 py-2.5 text-sm font-medium text-[#66776b] hover:bg-[#f6f8f5]">Cancel</button><button type="submit" disabled={isSaving || !draft.title.trim()} className="inline-flex min-w-32 items-center justify-center gap-2 rounded-xl bg-[#254f36] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#1d412c] disabled:cursor-not-allowed disabled:opacity-60">{isSaving ? 'Saving…' : editingId === null ? <><Plus className="h-4 w-4" />Create task</> : 'Save changes'}</button></div>
            </form>
          </section>
        </div>
      )}
    </main>
  );
}
