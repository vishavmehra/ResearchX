"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { api, authRedirect, type Conversation } from "@/lib/api";
import { clearTray, diffHref, toggleTray, TRAY_MAX, useTray } from "@/lib/diff-tray";
import { FEATURE } from "@/lib/features";
import { supabase } from "@/lib/supabase/client";
import { Alert, cx, Spinner, Wordmark } from "@/components/ui";
import * as I from "@/components/icons";

export type Me = { id: string; email: string | null; aal: string; mfa_enabled: boolean; profile: { display_name: string | null; created_at?: string } };
type Ctx = { refresh: () => Promise<void>; me: Me | null; reloadMe: () => Promise<void> };
const AppContext = createContext<Ctx>({ refresh: async () => {}, me: null, reloadMe: async () => {} });
export const useApp = () => useContext(AppContext);

const GROUPS = ["Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older"] as const;

function groupOf(iso: string): (typeof GROUPS)[number] {
  const day = 86_400_000;
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const t = new Date(iso).getTime();
  if (t >= start.getTime()) return "Today";
  if (t >= start.getTime() - day) return "Yesterday";
  if (t >= start.getTime() - 7 * day) return "Previous 7 days";
  if (t >= start.getTime() - 30 * day) return "Previous 30 days";
  return "Older";
}

export function AppShell({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [items, setItems] = useState<Conversation[] | null>(null);
  const [search, setSearch] = useState("");
  const [me, setMe] = useState<Me | null>(null);
  const [navOpen, setNavOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onError = useCallback(
    (e: unknown) => {
      const to = authRedirect(e);
      if (to) router.replace(to);
      else setError(e instanceof Error ? e.message : "Something went wrong.");
    },
    [router],
  );

  const refresh = useCallback(async () => {
    try {
      const q = search.trim();
      const r = await api<{ conversations: Conversation[] }>(`/api/conversations?limit=60${q ? `&q=${encodeURIComponent(q)}` : ""}`);
      setItems(r.conversations);
      setError(null);
    } catch (e) {
      onError(e);
    }
  }, [search, onError]);

  const reloadMe = useCallback(async () => {
    try {
      setMe(await api<Me>("/api/me"));
    } catch (e) {
      onError(e);
    }
  }, [onError]);

  useEffect(() => {
    const t = setTimeout(refresh, search ? 250 : 0);
    return () => clearTimeout(t);
  }, [refresh, search]);

  useEffect(() => {
    void reloadMe();
  }, [reloadMe]);

  useEffect(() => {
    // another tab signed out (or the session was revoked): leave the app everywhere
    const { data } = supabase().auth.onAuthStateChange((event) => {
      if (event === "SIGNED_OUT") router.replace("/login?notice=signedout");
    });
    return () => data.subscription.unsubscribe();
  }, [router]);

  useEffect(() => setNavOpen(false), [pathname]);

  async function signOut() {
    await supabase().auth.signOut();
    router.replace("/login?notice=signedout");
  }

  const grouped = GROUPS.map((g) => ({ g, rows: (items ?? []).filter((c) => groupOf(c.updated_at) === g) })).filter((x) => x.rows.length);
  const name = me?.profile.display_name || me?.email?.split("@")[0] || "";

  return (
    <div className="flex h-dvh overflow-hidden">
      {navOpen && <button aria-label="Close menu" className="fixed inset-0 z-30 bg-black/30 md:hidden" onClick={() => setNavOpen(false)} />}
      <aside
        aria-label="Research chats"
        className={cx(
          "fixed inset-y-0 left-0 z-40 flex w-[284px] flex-col border-r border-line bg-surface transition-transform md:static md:translate-x-0",
          navOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="px-4 pb-3 pt-4">
          <Link href="/" aria-label="ResearchX: new research">
            <Wordmark />
          </Link>
        </div>
        <div className="px-3">
          <Link
            href="/"
            className="flex items-center gap-2 rounded-lg border border-line bg-bg px-3 py-2 text-sm font-medium text-ink transition hover:border-accent hover:text-accent"
          >
            <I.Plus /> New research
          </Link>
        </div>
        <div className="px-3 pt-1.5">
          <NavLink href="/map" icon={<I.Graph />} label="Knowledge Map" active={pathname === "/map"} />
          <NavLink href="/analyze" icon={<I.Bars />} label="Analyze" active={pathname === "/analyze"} />
        </div>
        <div className="px-3 pt-3">
          <label className="relative block">
            <span className="sr-only">Search chats</span>
            <I.Search className="pointer-events-none absolute left-2.5 top-2.5 text-faint" width={16} height={16} />
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search chats"
              className="w-full rounded-lg border border-line bg-bg py-2 pl-8 pr-2 text-sm text-ink placeholder:text-faint focus:border-accent"
            />
          </label>
        </div>
        <nav className="mt-2 flex-1 overflow-y-auto px-2 pb-3" aria-label="Chats">
          {items === null ? (
            <div className="flex justify-center py-6 text-faint">
              <Spinner />
            </div>
          ) : grouped.length === 0 ? (
            <p className="px-2 py-4 text-sm text-muted">{search ? "No chats match that search." : "No research yet. Ask your first question."}</p>
          ) : (
            grouped.map(({ g, rows }) => (
              <section key={g} className="mt-3">
                <h2 className="px-2 pb-1 text-[0.7rem] font-medium uppercase tracking-wider text-faint">{g}</h2>
                <ul>
                  {rows.map((c) => (
                    <Row key={c.id} c={c} active={pathname === `/c/${c.id}`} onChanged={refresh} onError={onError} />
                  ))}
                </ul>
              </section>
            ))
          )}
        </nav>
        <div className="space-y-0.5 border-t border-line p-3">
          <div className="flex items-center gap-2.5 px-1.5 pb-2">
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-accent-soft text-sm font-semibold uppercase text-accent">
              {name.slice(0, 1) || "·"}
            </span>
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-ink">{name || "Loading…"}</p>
              <p className="truncate text-xs text-muted">{me?.email ?? ""}</p>
            </div>
          </div>
          <NavLink href="/settings" icon={<I.Gear />} label="Settings" active={pathname === "/settings"} />
          <NavLink
            href="/settings/security"
            icon={<I.Shield />}
            label="Security"
            active={pathname === "/settings/security"}
            extra={me ? <span className={cx("ml-auto text-xs", me.mfa_enabled ? "text-ok" : "text-faint")}>{me.mfa_enabled ? "2FA on" : "2FA off"}</span> : null}
          />
          <button onClick={signOut} className="flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-sm text-muted transition hover:bg-surface-2 hover:text-ink">
            <I.LogOut /> Sign out
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 border-b border-line bg-surface px-3 py-2 md:hidden">
          <button aria-label="Open menu" className="rounded-lg p-1.5 text-muted hover:bg-surface-2" onClick={() => setNavOpen(true)}>
            <I.Menu />
          </button>
          <Wordmark className="text-lg" />
        </header>
        {error && (
          <div className="px-4 pt-3">
            <Alert tone="bad">{error}</Alert>
          </div>
        )}
        <AppContext.Provider value={{ refresh, me, reloadMe }}>{children}</AppContext.Provider>
      </div>
      <DiffTray />
    </div>
  );
}

/** Papers collected for a diff from answers or the Knowledge Map, one click away from the comparison. */
function DiffTray() {
  const tray = useTray();
  const pathname = usePathname();
  if (!tray.length || pathname === "/analyze") return null;
  return (
    <div role="region" aria-label={FEATURE.papers} className="fixed right-4 top-4 z-40 w-80 rounded-xl border border-accent/40 bg-surface p-3 text-sm shadow-lg">
      <div className="flex items-center justify-between">
        <span className="font-semibold text-ink">{FEATURE.papers} · {tray.length}/{TRAY_MAX}</span>
        <button onClick={clearTray} className="text-xs text-muted hover:text-bad">Clear</button>
      </div>
      <ol className="mt-2 space-y-1">
        {tray.map((p, i) => (
          <li key={p.id} className="flex items-start gap-2">
            <span className="mt-0.5 font-mono text-xs text-faint">{i + 1}</span>
            <span className="min-w-0 flex-1 text-ink" title={p.title}>{p.title.length > 70 ? p.title.slice(0, 69) + "…" : p.title}</span>
            <button onClick={() => toggleTray(p)} aria-label={`Remove ${p.title}`} className="rounded p-0.5 text-faint hover:bg-surface-2 hover:text-bad"><I.Cross width={13} height={13} /></button>
          </li>
        ))}
      </ol>
      {tray.length >= 2 ? (
        <Link href={diffHref(tray)} className="mt-2 block rounded-lg bg-accent px-2.5 py-1.5 text-center font-medium text-accent-ink hover:opacity-90">Compare {tray.length} papers</Link>
      ) : (
        <p className="mt-2 text-xs text-muted">Add a passage from a different paper (or a paper from the Knowledge Map) to compare.</p>
      )}
    </div>
  );
}

function NavLink({ href, icon, label, active, extra }: { href: string; icon: ReactNode; label: string; active: boolean; extra?: ReactNode }) {
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={cx("flex items-center gap-2.5 rounded-lg px-2 py-1.5 text-sm transition", active ? "bg-surface-2 text-ink" : "text-muted hover:bg-surface-2 hover:text-ink")}
    >
      {icon} {label} {extra}
    </Link>
  );
}

function Row({ c, active, onChanged, onError }: { c: Conversation; active: boolean; onChanged: () => Promise<void>; onError: (e: unknown) => void }) {
  const router = useRouter();
  const [menu, setMenu] = useState(false);
  const [editing, setEditing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [title, setTitle] = useState(c.title ?? "Untitled research");
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLLIElement>(null);

  useEffect(() => {
    if (!menu && !confirming) return;
    const close = (e: MouseEvent) => {
      if (!box.current?.contains(e.target as Node)) {
        setMenu(false);
        setConfirming(false);
      }
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [menu, confirming]);

  async function rename(e: FormEvent) {
    e.preventDefault();
    const t = title.trim();
    if (!t || t === c.title) return setEditing(false);
    setBusy(true);
    try {
      await api(`/api/conversations/${c.id}`, { method: "PATCH", body: JSON.stringify({ title: t }) });
      setEditing(false);
      await onChanged();
    } catch (err) {
      onError(err);
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    setBusy(true);
    try {
      await api(`/api/conversations/${c.id}`, { method: "DELETE" });
      if (active) router.push("/");
      await onChanged();
    } catch (err) {
      onError(err);
      setBusy(false);
    }
  }

  if (editing)
    return (
      <li className="px-1 py-0.5">
        <form onSubmit={rename}>
          <label className="sr-only" htmlFor={`rename-${c.id}`}>
            Chat title
          </label>
          <input
            id={`rename-${c.id}`}
            autoFocus
            value={title}
            maxLength={120}
            disabled={busy}
            onChange={(e) => setTitle(e.target.value)}
            onBlur={rename}
            onKeyDown={(e) => e.key === "Escape" && (setTitle(c.title ?? ""), setEditing(false))}
            className="w-full rounded-md border border-accent bg-surface px-2 py-1.5 text-sm text-ink"
          />
        </form>
      </li>
    );

  return (
    <li ref={box} className="group relative">
      <Link
        href={`/c/${c.id}`}
        aria-current={active ? "page" : undefined}
        className={cx(
          "block truncate rounded-lg py-1.5 pl-2.5 pr-8 text-sm transition",
          active ? "bg-accent-soft font-medium text-ink" : "text-muted hover:bg-surface-2 hover:text-ink",
        )}
        title={c.title ?? undefined}
      >
        {c.title || "Untitled research"}
      </Link>
      <button
        aria-label={`Options for ${c.title ?? "chat"}`}
        aria-expanded={menu}
        onClick={() => setMenu((m) => !m)}
        className={cx(
          "absolute right-1 top-1 rounded-md p-1 text-faint transition hover:bg-surface hover:text-ink focus:opacity-100",
          menu || active ? "opacity-100" : "opacity-0 group-hover:opacity-100",
        )}
      >
        <I.Dots width={16} height={16} />
      </button>
      {menu && (
        <div role="menu" className="absolute right-1 top-8 z-20 w-40 rounded-lg border border-line bg-surface p-1 shadow-lg">
          <button role="menuitem" onClick={() => (setMenu(false), setEditing(true))} className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm text-ink hover:bg-surface-2">
            <I.Pencil width={15} height={15} /> Rename
          </button>
          <button role="menuitem" onClick={() => (setMenu(false), setConfirming(true))} className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm text-bad hover:bg-bad-soft">
            <I.Trash width={15} height={15} /> Delete
          </button>
        </div>
      )}
      {confirming && (
        <div role="alertdialog" aria-label="Delete chat" className="absolute right-1 top-8 z-20 w-60 rounded-lg border border-line bg-surface p-3 shadow-lg">
          <p className="text-sm text-ink">Delete this chat?</p>
          <p className="mt-1 text-xs text-muted">Its messages, research runs and traces are removed for good.</p>
          <div className="mt-3 flex justify-end gap-2">
            <button onClick={() => setConfirming(false)} className="rounded-md px-2.5 py-1 text-sm text-muted hover:bg-surface-2">
              Cancel
            </button>
            <button onClick={remove} disabled={busy} className="inline-flex items-center gap-1.5 rounded-md bg-bad px-2.5 py-1 text-sm font-medium text-white disabled:opacity-60">
              {busy && <Spinner className="h-3.5 w-3.5" />} Delete
            </button>
          </div>
        </div>
      )}
    </li>
  );
}
