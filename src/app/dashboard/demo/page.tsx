'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Activity, ArrowUpRight, Bell, Check, CircleDollarSign, CreditCard, LogOut, Users } from 'lucide-react';

const samplePayments = [
  { name: 'Northstar Studio', service: 'Product design', amount: '₹24,000', status: 'Paid', date: 'Today' },
  { name: 'Morrow Labs', service: 'Frontend development', amount: '₹18,500', status: 'Pending', date: 'Yesterday' },
  { name: 'Fieldnote Co.', service: 'Monthly support', amount: '₹12,000', status: 'Paid', date: 'Nov 16' },
  { name: 'Common Ground', service: 'Landing page', amount: '₹32,000', status: 'Paid', date: 'Nov 14' },
];

export default function DemoDashboardPage() {
  const [signingOut, setSigningOut] = useState(false);
  const router = useRouter();

  const signOut = async () => {
    setSigningOut(true);
    await fetch('/api/demo-auth', { method: 'DELETE' });
    router.replace('/login');
  };

  return (
    <main className="min-h-screen bg-[#f5f6f2] text-[#20352a]">
      <aside className="fixed inset-y-0 left-0 hidden w-60 flex-col border-r border-[#e2e7df] bg-white px-5 py-6 md:flex">
        <div className="flex items-center gap-3"><div className="grid h-9 w-9 place-items-center rounded-xl bg-[#254f36] text-white"><Check className="h-4 w-4" /></div><span className="text-sm font-semibold tracking-tight text-[#293e30]">TaskFlow Studio</span></div>
        <p className="mt-10 px-3 text-[10px] font-semibold uppercase tracking-[0.16em] text-[#9aa49b]">Workspace</p>
        <nav className="mt-3 space-y-1">
          <div className="flex items-center gap-3 rounded-xl bg-[#edf4ec] px-3 py-2.5 text-sm font-medium text-[#365d40]"><Activity className="h-4 w-4" />Overview</div>
          <div className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-[#7d8a80]"><Users className="h-4 w-4" />Clients</div>
          <div className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-[#7d8a80]"><CreditCard className="h-4 w-4" />Payments</div>
          <div className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-[#7d8a80]"><Bell className="h-4 w-4" />Notifications</div>
        </nav>
        <div className="mt-auto rounded-xl border border-[#e3e9e2] bg-[#f8faf7] p-3"><p className="text-xs font-semibold text-[#425747]">Demo workspace</p><p className="mt-1 text-[11px] leading-5 text-[#87938a]">Sample data only. No live account records are connected.</p></div>
      </aside>

      <div className="md:pl-60">
        <header className="flex h-[68px] items-center justify-between border-b border-[#e2e7df] bg-white px-4 sm:px-7">
          <div className="flex items-center gap-2"><span className="rounded-full border border-[#d5e4d5] bg-[#eff6ee] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#4f7957]">Demo mode</span><span className="hidden text-xs text-[#8b978e] sm:inline">Preview dashboard</span></div>
          <button onClick={signOut} disabled={signingOut} className="inline-flex items-center gap-2 rounded-lg border border-[#dfe6de] px-3 py-2 text-xs font-medium text-[#65756a] transition hover:bg-[#f7f9f6] disabled:opacity-60"><LogOut className="h-3.5 w-3.5" />{signingOut ? 'Signing out…' : 'Sign out'}</button>
        </header>

        <div className="mx-auto max-w-6xl px-4 py-7 sm:px-7 sm:py-9">
          <section className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end"><div><p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[#7b9180]">Workspace overview</p><h1 className="mt-2 text-3xl font-semibold tracking-[-0.045em] text-[#263d2e]">Good to see you, Aquib.</h1><p className="mt-2 text-sm text-[#7d8a80]">Here’s a sample of your payment workspace.</p></div><div className="inline-flex items-center gap-2 rounded-xl border border-[#e2e8e0] bg-white px-3 py-2 text-xs text-[#718075]"><CircleDollarSign className="h-4 w-4 text-[#56825c]" />Monthly overview</div></section>

          <div className="mt-7 rounded-xl border border-[#eadfc2] bg-[#fffaf0] px-4 py-3 text-xs leading-5 text-[#806b3c]">This is an isolated preview with sample data. It does not read or modify your Supabase account.</div>

          <section aria-label="Sample summary" className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {[
              { label: 'Collected this month', value: '₹86,500', note: '+12.8% from last month', icon: CircleDollarSign },
              { label: 'Outstanding', value: '₹18,500', note: 'Across 2 invoices', icon: CreditCard },
              { label: 'Active clients', value: '14', note: '3 added this month', icon: Users },
              { label: 'Payment reminders', value: '6', note: '2 need attention', icon: Bell },
            ].map(({ label, value, note, icon: Icon }) => <article key={label} className="rounded-2xl border border-[#e2e8e0] bg-white p-4 shadow-[0_8px_24px_-22px_rgba(34,64,42,.4)]"><div className="flex items-center justify-between"><p className="text-xs font-medium text-[#7c8a80]">{label}</p><span className="rounded-lg bg-[#eff5ed] p-2 text-[#608265]"><Icon className="h-4 w-4" /></span></div><p className="mt-4 text-2xl font-semibold tracking-tight text-[#2c4934]">{value}</p><p className="mt-1 text-[11px] text-[#92a095]">{note}</p></article>)}
          </section>

          <section className="mt-5 overflow-hidden rounded-2xl border border-[#e2e8e0] bg-white">
            <div className="flex items-center justify-between border-b border-[#edf0ec] px-4 py-4 sm:px-5"><div><h2 className="text-sm font-semibold text-[#344a38]">Recent payments</h2><p className="mt-1 text-xs text-[#8b978e]">Example activity for the demo workspace</p></div><span className="rounded-full bg-[#f2f5f1] px-2.5 py-1 text-[10px] font-medium text-[#78877b]">Sample records</span></div>
            <div className="overflow-x-auto"><table className="w-full min-w-[600px] text-left"><thead className="bg-[#fafbf9] text-[10px] uppercase tracking-[0.12em] text-[#8c988e]"><tr><th className="px-5 py-3 font-semibold">Client</th><th className="px-5 py-3 font-semibold">Service</th><th className="px-5 py-3 font-semibold">Date</th><th className="px-5 py-3 font-semibold">Status</th><th className="px-5 py-3 text-right font-semibold">Amount</th></tr></thead><tbody className="divide-y divide-[#edf0ec]">{samplePayments.map((payment) => <tr key={payment.name} className="text-xs text-[#65756a]"><td className="px-5 py-4 font-medium text-[#3a4f3d]">{payment.name}</td><td className="px-5 py-4">{payment.service}</td><td className="px-5 py-4">{payment.date}</td><td className="px-5 py-4"><span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ${payment.status === 'Paid' ? 'bg-[#edf5ed] text-[#4e7954]' : 'bg-[#faf2df] text-[#8c6e2e]'}`}>{payment.status}</span></td><td className="px-5 py-4 text-right font-semibold text-[#405645]">{payment.amount}</td></tr>)}</tbody></table></div>
          </section>

          <div className="mt-4 flex items-center gap-2 text-[11px] text-[#929d94]"><ArrowUpRight className="h-3.5 w-3.5" />Demo access is read-only and expires after four hours.</div>
        </div>
      </div>
    </main>
  );
}
