"use client";

import type { AMREntry, Factor, Provider, UserIdentity } from "@supabase/supabase-js";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { api, ApiError, authRedirect } from "@/lib/api";
import { supabase } from "@/lib/supabase/client";
import { useApp } from "@/components/app-shell";
import { callbackUrl, noticeOf, notEnabledNote, OAUTH, offeredOf, useRedirectBusy } from "@/components/auth";
import { Alert, Button, cx, Field, Spinner } from "@/components/ui";
import * as I from "@/components/icons";

type Doc = { id: string; title: string; media_type: string; bytes: number; status: string; chunks: number; error: string | null; created_at: string };
type Limits = { documents: number; passages: number; bytes: number };
type Flash = { tone: "ok" | "bad" | "warn" | "info"; text: string } | null;

function Page({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-[720px] space-y-6 px-4 py-8">
        <header>
          <h1 className="font-serif text-[1.75rem] font-semibold text-ink">{title}</h1>
          <p className="mt-1 text-sm text-muted">{subtitle}</p>
        </header>
        {children}
      </div>
    </div>
  );
}

function Card({ title, description, children, tone }: { title: string; description?: ReactNode; children: ReactNode; tone?: "danger" }) {
  return (
    <section className={cx("rounded-2xl border bg-surface p-5", tone === "danger" ? "border-bad/40" : "border-line")}>
      <h2 className={cx("font-serif text-lg font-semibold", tone === "danger" ? "text-bad" : "text-ink")}>{title}</h2>
      {description && <p className="mt-1 text-sm leading-relaxed text-muted">{description}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

function bytes(n: number) {
  return n < 1024 ? `${n} B` : n < 1_048_576 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1_048_576).toFixed(1)} MB`;
}

function useAuthErrors() {
  const router = useRouter();
  return useCallback(
    (e: unknown, set: (f: Flash) => void) => {
      const to = authRedirect(e);
      if (to) router.replace(to);
      else set({ tone: "bad", text: e instanceof Error ? e.message : "Something went wrong." });
    },
    [router],
  );
}

export function SettingsPage() {
  const { me, reloadMe } = useApp();
  const handle = useAuthErrors();
  const [name, setName] = useState("");
  const [savingName, setSavingName] = useState(false);
  const [profileMsg, setProfileMsg] = useState<Flash>(null);
  const [docs, setDocs] = useState<Doc[] | null>(null);
  const [limits, setLimits] = useState<Limits | null>(null);
  const [docMsg, setDocMsg] = useState<Flash>(null);
  const [uploading, setUploading] = useState(false);
  const file = useRef<HTMLInputElement>(null);

  useEffect(() => setName(me?.profile.display_name ?? ""), [me]);

  const loadDocs = useCallback(async () => {
    try {
      const r = await api<{ documents: Doc[]; limits: Limits }>("/api/documents");
      setDocs(r.documents);
      setLimits(r.limits);
    } catch (e) {
      handle(e, setDocMsg);
    }
  }, [handle]);

  useEffect(() => {
    void loadDocs();
  }, [loadDocs]);

  async function saveName(e: FormEvent) {
    e.preventDefault();
    setSavingName(true);
    setProfileMsg(null);
    try {
      await api("/api/me", { method: "PATCH", body: JSON.stringify({ display_name: name.trim() || null }) });
      await reloadMe();
      setProfileMsg({ tone: "ok", text: "Profile saved." });
    } catch (err) {
      handle(err, setProfileMsg);
    } finally {
      setSavingName(false);
    }
  }

  async function upload(f: File) {
    setDocMsg(null);
    const type = f.type || (f.name.endsWith(".md") ? "text/markdown" : f.name.endsWith(".txt") ? "text/plain" : "");
    if (limits && f.size > limits.bytes) return setDocMsg({ tone: "bad", text: `Files must be at most ${bytes(limits.bytes)}.` });
    setUploading(true);
    try {
      await api("/api/documents", { method: "POST", body: f, headers: { "Content-Type": type, "X-Filename": encodeURIComponent(f.name) } });
      setDocMsg({ tone: "ok", text: `${f.name} is ready. ResearchX will search it alongside the shared corpus, for you only.` });
      await loadDocs();
    } catch (err) {
      handle(err, setDocMsg);
    } finally {
      setUploading(false);
      if (file.current) file.current.value = "";
    }
  }

  async function remove(d: Doc) {
    setDocMsg(null);
    try {
      await api(`/api/documents/${d.id}`, { method: "DELETE" });
      await loadDocs();
    } catch (err) {
      handle(err, setDocMsg);
    }
  }

  const used = (docs ?? []).reduce((n, d) => n + d.chunks, 0);
  return (
    <Page title="Settings" subtitle="Your profile and private research library.">
      <Card title="Profile">
        <form onSubmit={saveName} className="space-y-4">
          {profileMsg && <Alert tone={profileMsg.tone}>{profileMsg.text}</Alert>}
          <Field label="Email" value={me?.email ?? ""} readOnly disabled hint="Your sign-in address. Change it from your account provider's flow." />
          <Field label="Display name" name="display_name" maxLength={80} value={name} onChange={(e) => setName(e.target.value)} />
          <Button type="submit" busy={savingName}>
            Save profile
          </Button>
        </form>
      </Card>

      <Card
        title="Private library"
        description="Upload papers, notes or drafts to research them together with the ResearchX corpus. They are visible only to you, never added to the shared corpus, and deleted with your account."
      >
        <div className="space-y-4">
          {docMsg && <Alert tone={docMsg.tone}>{docMsg.text}</Alert>}
          <div className="flex flex-wrap items-center gap-3">
            <input ref={file} type="file" accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown" className="sr-only" id="doc-upload"
              onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
            <label
              htmlFor="doc-upload"
              aria-disabled={uploading}
              className={cx("inline-flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-surface px-3.5 py-2 text-sm font-medium text-ink transition hover:bg-surface-2", uploading && "pointer-events-none opacity-60")}
            >
              {uploading ? <Spinner /> : <I.File />} {uploading ? "Processing…" : "Upload document"}
            </label>
            {limits && (
              <span className="text-xs text-muted">
                PDF, TXT or Markdown up to {bytes(limits.bytes)} · {docs?.length ?? 0}/{limits.documents} documents · {used}/{limits.passages} passages
              </span>
            )}
          </div>
          {docs === null ? (
            <Spinner className="text-faint" />
          ) : docs.length === 0 ? (
            <p className="text-sm text-muted">No private documents yet.</p>
          ) : (
            <ul className="divide-y divide-line rounded-xl border border-line">
              {docs.map((d) => (
                <li key={d.id} className="flex items-center gap-3 px-3 py-2.5">
                  <I.File className="shrink-0 text-faint" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-ink">{d.title}</p>
                    <p className="text-xs text-muted">
                      {bytes(d.bytes)} · {d.chunks} passages · {d.status === "READY" ? "ready" : d.status.toLowerCase()}
                      {d.error ? ` · ${d.error}` : ""}
                    </p>
                  </div>
                  <button onClick={() => remove(d)} aria-label={`Delete ${d.title}`} className="rounded-md p-1.5 text-faint transition hover:bg-bad-soft hover:text-bad">
                    <I.Trash width={16} height={16} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </Card>
    </Page>
  );
}

const label = (provider: string) => OAUTH[provider as Provider]?.label ?? provider;

export function SecurityPage({ providers, notice }: { providers: string[]; notice?: string }) {
  const router = useRouter();
  const { me, reloadMe } = useApp();
  const handle = useAuthErrors();
  const [factors, setFactors] = useState<Factor[] | null>(null);
  const [aal, setAal] = useState<string | null>(null);
  const [identities, setIdentities] = useState<UserIdentity[] | null>(null);
  const [reauthed, setReauthed] = useState(false);
  const [redirecting, setRedirecting] = useRedirectBusy();
  const [idMsg, setIdMsg] = useState<Flash>(noticeOf(notice === "reauthed" ? null : notice) ?? null);
  const [mfaMsg, setMfaMsg] = useState<Flash>(null);
  const [enrol, setEnrol] = useState<{ factorId: string; qr: string; secret: string } | null>(null);
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [pw, setPw] = useState({ next: "", confirm: "" });
  const [pwMsg, setPwMsg] = useState<Flash>(null);
  const [del, setDel] = useState({ password: "", code: "", confirm: "" });
  const [delMsg, setDelMsg] = useState<Flash>(null);
  const [sessionMsg, setSessionMsg] = useState<Flash>(null);

  const load = useCallback(async () => {
    const client = supabase();
    const [{ data: f, error }, { data: level }, { data: ids }, { data: jwt }] = await Promise.all([
      client.auth.mfa.listFactors(),
      client.auth.mfa.getAuthenticatorAssuranceLevel(),
      client.auth.getUserIdentities(),
      client.auth.getClaims(),
    ]);
    if (error) setMfaMsg({ tone: "bad", text: "Could not load two-factor settings." });
    setFactors(f?.all ?? []);
    setAal(level?.currentLevel ?? null);
    setIdentities(ids?.identities ?? []);
    // back from "confirm with <provider>": the API accepts a deletion within 10 minutes of a sign-in (REAUTH_WINDOW_S)
    const amr = (jwt?.claims.amr ?? []) as (AMREntry | string)[];
    setReauthed(notice === "reauthed" && amr.some((m) => typeof m !== "string" && m.timestamp > Date.now() / 1000 - 540));
  }, [notice]);

  useEffect(() => {
    void load();
  }, [load]);

  const verified = (factors ?? []).filter((f) => f.status === "verified");
  const hasPassword = identities === null || identities.some((i) => i.provider === "email");
  const connected = (identities ?? []).filter((i) => i.provider !== "email");
  const connectable = offeredOf(providers).filter((p) => !connected.some((i) => i.provider === p));
  const notEnabled = connectable.filter((p) => !providers.includes(p));

  async function connect(provider: Provider) {
    setRedirecting(provider);
    setIdMsg(null);
    const { error } = await supabase().auth.linkIdentity({ provider, options: { redirectTo: callbackUrl("link", "/settings/security?notice=linked") } });
    if (!error) return; // on the way to the provider
    setRedirecting(null);
    setIdMsg(error.code === "manual_linking_disabled" ? (noticeOf("linking_disabled") ?? null) : { tone: "bad", text: `Could not start connecting ${label(provider)}.` });
  }

  async function disconnect(i: UserIdentity) {
    if (!window.confirm(`Disconnect ${label(i.provider)}? You will no longer be able to sign in with it.`)) return;
    setBusy(`unlink:${i.identity_id}`);
    setIdMsg(null);
    const client = supabase();
    const { error } = await client.auth.unlinkIdentity(i);
    setBusy(null);
    if (error)
      return setIdMsg({
        tone: "bad",
        text:
          error.code === "single_identity_not_deletable"
            ? "Add another sign-in method before disconnecting this one."
            : error.code === "insufficient_aal"
              ? "Sign in again with your authenticator code first."
              : "Could not disconnect that account.",
      });
    await client.auth.refreshSession();
    setIdMsg({ tone: "info", text: `${label(i.provider)} disconnected.` });
    await load();
  }

  async function confirmWith(provider: Provider) {
    setRedirecting(provider);
    setDelMsg(null);
    const { error } = await supabase().auth.signInWithOAuth({ provider, options: { redirectTo: callbackUrl("reauth", "/settings/security?notice=reauthed") } });
    if (!error) return;
    setRedirecting(null);
    setDelMsg({ tone: "bad", text: `Could not start signing in with ${label(provider)}.` });
  }

  async function startEnrol() {
    setBusy("enrol");
    setMfaMsg(null);
    const client = supabase();
    for (const f of (factors ?? []).filter((x) => x.status !== "verified")) await client.auth.mfa.unenroll({ factorId: f.id }); // abandoned attempts
    const { data, error } = await client.auth.mfa.enroll({ factorType: "totp", friendlyName: `ResearchX ${new Date().toISOString().slice(0, 16)}` });
    setBusy(null);
    if (error || !data) return setMfaMsg({ tone: "bad", text: error?.message ?? "Could not start enrolment." });
    setEnrol({ factorId: data.id, qr: data.totp.qr_code, secret: data.totp.secret });
    setCode("");
  }

  async function confirmEnrol(e: FormEvent) {
    e.preventDefault();
    if (!enrol || code.length !== 6) return;
    setBusy("verify");
    const { error } = await supabase().auth.mfa.challengeAndVerify({ factorId: enrol.factorId, code });
    setBusy(null);
    if (error) return setMfaMsg({ tone: "bad", text: "That code did not work. Check your device clock and enter the current code." });
    setEnrol(null);
    setMfaMsg({ tone: "ok", text: "Two-factor authentication is on. You will be asked for a code each time you sign in." });
    await Promise.all([load(), reloadMe()]);
  }

  async function cancelEnrol() {
    if (enrol) await supabase().auth.mfa.unenroll({ factorId: enrol.factorId });
    setEnrol(null);
    await load();
  }

  async function disable(f: Factor) {
    if (aal !== "aal2") return setMfaMsg({ tone: "warn", text: "Sign in again with your authenticator code before turning two-factor authentication off." });
    if (!window.confirm("Turn off two-factor authentication? Signing in will no longer ask for an authenticator code.")) return;
    setBusy("disable");
    const client = supabase();
    const { error } = await client.auth.mfa.unenroll({ factorId: f.id });
    setBusy(null);
    if (error) return setMfaMsg({ tone: "bad", text: error.message });
    await client.auth.refreshSession();
    setMfaMsg({ tone: "info", text: "Two-factor authentication is off." });
    await Promise.all([load(), reloadMe()]);
  }

  async function changePassword(e: FormEvent) {
    e.preventDefault();
    setPwMsg(null);
    if (pw.next.length < 10 || !/[a-z]/i.test(pw.next) || !/\d/.test(pw.next)) return setPwMsg({ tone: "bad", text: "Use at least 10 characters with letters and numbers." });
    if (pw.next !== pw.confirm) return setPwMsg({ tone: "bad", text: "Passwords do not match." });
    setBusy("password");
    const { error } = await supabase().auth.updateUser({ password: pw.next });
    setBusy(null);
    if (error)
      return setPwMsg({
        tone: "bad",
        text:
          error.code === "insufficient_aal"
            ? "Verify your authenticator code first (sign in again), then change the password."
            : error.code === "same_password"
              ? "Choose a password you have not used for this account."
              : error.code === "reauthentication_needed"
                ? "For security, sign out and sign in again before changing the password."
                : "Could not change the password.",
      });
    setPw({ next: "", confirm: "" });
    setPwMsg({ tone: "ok", text: "Password changed." });
  }

  async function signOut(scope: "local" | "global") {
    setBusy(scope);
    const { error } = await supabase().auth.signOut({ scope });
    setBusy(null);
    if (error) return setSessionMsg({ tone: "bad", text: "Could not sign out. Try again." });
    router.replace("/login?notice=signedout");
  }

  async function deleteAccount(e: FormEvent) {
    e.preventDefault();
    setDelMsg(null);
    if (del.confirm !== "DELETE MY ACCOUNT") return setDelMsg({ tone: "bad", text: "Type DELETE MY ACCOUNT to confirm." });
    setBusy("delete");
    const client = supabase();
    try {
      if (!reauthed) {
        const { error } = await client.auth.signInWithPassword({ email: me?.email ?? "", password: del.password }); // fresh proof of identity
        if (error) throw new ApiError(401, "reauth_failed", "Incorrect password.");
      }
      if (verified[0]) {
        const r = await client.auth.mfa.challengeAndVerify({ factorId: verified[0].id, code: del.code });
        if (r.error) throw new ApiError(401, "reauth_failed", "That authenticator code did not work.");
      }
      await api("/api/account", { method: "DELETE", body: JSON.stringify({ confirm: "DELETE MY ACCOUNT" }) });
      await client.auth.signOut({ scope: "local" });
      router.replace("/login?notice=deleted");
    } catch (err) {
      setBusy(null);
      if (err instanceof ApiError && err.code === "reauth_failed") setDelMsg({ tone: "bad", text: err.message });
      else handle(err, setDelMsg);
    }
  }

  return (
    <Page title="Security" subtitle="Sign-in methods, password, two-factor authentication and sessions.">
      <Card title="Sign-in methods" description="Every method listed here signs in to this same account.">
        <div className="space-y-4">
          {idMsg && <Alert tone={idMsg.tone}>{idMsg.text}</Alert>}
          {identities === null ? (
            <Spinner className="text-faint" />
          ) : (
            <ul className="divide-y divide-line rounded-xl border border-line">
              {identities.map((i) => (
                <li key={i.identity_id} className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
                  <div className="flex min-w-0 items-center gap-3">
                    {i.provider === "email" ? <I.Mail className="shrink-0 text-muted" /> : (OAUTH[i.provider as Provider]?.icon ?? <I.Shield className="shrink-0 text-muted" />)}
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-ink">{i.provider === "email" ? "Email and password" : label(i.provider)}</p>
                      <p className="truncate text-xs text-muted">
                        {String(i.identity_data?.email ?? i.identity_data?.user_name ?? "")}
                        {i.created_at ? ` · added ${new Date(i.created_at).toLocaleDateString()}` : ""}
                      </p>
                    </div>
                  </div>
                  {i.provider !== "email" && (
                    <Button
                      variant="secondary"
                      busy={busy === `unlink:${i.identity_id}`}
                      disabled={identities.length < 2}
                      title={identities.length < 2 ? "Add another sign-in method first" : undefined}
                      onClick={() => disconnect(i)}
                    >
                      Disconnect
                    </Button>
                  )}
                </li>
              ))}
            </ul>
          )}
          {identities !== null && connectable.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {connectable.map((p) => (
                <Button key={p} variant="secondary" busy={redirecting === p} disabled={redirecting !== null || notEnabled.includes(p)}
                  title={notEnabled.includes(p) ? notEnabledNote([p]) ?? undefined : undefined} onClick={() => connect(p)}>
                  {redirecting !== p && OAUTH[p]?.icon}
                  Connect {label(p)}
                  {notEnabled.includes(p) && <span className="text-xs font-normal text-faint">· not enabled yet</span>}
                </Button>
              ))}
            </div>
          )}
          {identities !== null && notEnabled.length > 0 && <p className="text-xs text-faint">{notEnabledNote(notEnabled)}</p>}
        </div>
      </Card>

      <Card
        title="Password"
        description={
          hasPassword
            ? `Signed in as ${me?.email ?? "…"}.`
            : `You sign in with a connected account. Setting a password also lets you sign in as ${me?.email ?? "…"}.`
        }
      >
        <form onSubmit={changePassword} className="space-y-4">
          {pwMsg && <Alert tone={pwMsg.tone}>{pwMsg.text}</Alert>}
          <Field label="New password" type="password" autoComplete="new-password" value={pw.next} onChange={(e) => setPw({ ...pw, next: e.target.value })} />
          <Field label="Confirm new password" type="password" autoComplete="new-password" value={pw.confirm} onChange={(e) => setPw({ ...pw, confirm: e.target.value })} />
          <Button type="submit" variant="secondary" busy={busy === "password"} disabled={!pw.next}>
            {hasPassword ? "Change password" : "Set password"}
          </Button>
        </form>
      </Card>

      <Card
        title="Two-factor authentication"
        description="Protect your account with time-based codes from an authenticator app such as Google Authenticator, Microsoft Authenticator, Authy or 1Password."
      >
        <div className="space-y-4">
          {mfaMsg && <Alert tone={mfaMsg.tone}>{mfaMsg.text}</Alert>}
          {factors === null ? (
            <Spinner className="text-faint" />
          ) : enrol ? (
            <form onSubmit={confirmEnrol} className="space-y-4">
              <ol className="list-decimal space-y-1 pl-5 text-sm text-ink">
                <li>Scan this QR code with your authenticator app.</li>
                <li>Enter the 6-digit code the app shows.</li>
              </ol>
              <div className="flex flex-wrap items-start gap-5">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={enrol.qr} alt="QR code for your authenticator app" width={176} height={176} className="rounded-lg border border-line bg-white p-2" />
                <div className="min-w-0 flex-1 space-y-2">
                  <p className="text-sm text-muted">Can&apos;t scan? Enter this key manually:</p>
                  <code className="block break-all rounded-lg bg-surface-2 px-3 py-2 font-mono text-sm tracking-wider text-ink">
                    {enrol.secret.match(/.{1,4}/g)?.join(" ")}
                  </code>
                  <button type="button" onClick={() => navigator.clipboard.writeText(enrol.secret)} className="inline-flex items-center gap-1.5 text-xs font-medium text-accent hover:underline">
                    <I.Copy width={13} height={13} /> Copy key
                  </button>
                </div>
              </div>
              <Field
                label="Authentication code"
                inputMode="numeric"
                autoComplete="one-time-code"
                maxLength={6}
                value={code}
                onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                className="max-w-[12rem] font-mono tracking-[0.3em]"
              />
              <div className="flex gap-2">
                <Button type="submit" busy={busy === "verify"} disabled={code.length !== 6}>
                  Verify and turn on
                </Button>
                <Button type="button" variant="ghost" onClick={cancelEnrol}>
                  Cancel
                </Button>
              </div>
            </form>
          ) : verified.length ? (
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-ok/30 bg-ok-soft px-4 py-3">
              <div>
                <p className="flex items-center gap-1.5 text-sm font-medium text-ink">
                  <I.Shield width={16} height={16} className="text-ok" /> Two-factor authentication is on
                </p>
                <p className="text-xs text-muted">Authenticator app · added {new Date(verified[0].created_at).toLocaleDateString()}</p>
              </div>
              <Button variant="secondary" busy={busy === "disable"} onClick={() => disable(verified[0])}>
                Turn off
              </Button>
            </div>
          ) : (
            <Button onClick={startEnrol} busy={busy === "enrol"}>
              <I.Shield width={16} height={16} /> Enable two-factor authentication
            </Button>
          )}
        </div>
      </Card>

      <Card title="Sessions" description={`This session: ${aal === "aal2" ? "verified with your authenticator (AAL2)" : "single-factor sign-in (AAL1)"}.`}>
        <div className="space-y-3">
          {sessionMsg && <Alert tone={sessionMsg.tone}>{sessionMsg.text}</Alert>}
          <div className="flex flex-wrap gap-2">
            <Button variant="secondary" busy={busy === "local"} onClick={() => signOut("local")}>
              Sign out of this device
            </Button>
            <Button variant="secondary" busy={busy === "global"} onClick={() => signOut("global")}>
              Sign out everywhere
            </Button>
          </div>
        </div>
      </Card>

      <Card
        tone="danger"
        title="Delete account"
        description="Permanently deletes your profile, every chat and message, all research runs and traces, cached answers and private documents. The shared ResearchX corpus is not affected. This cannot be undone."
      >
        <form onSubmit={deleteAccount} className="space-y-4">
          {delMsg && <Alert tone={delMsg.tone}>{delMsg.text}</Alert>}
          {reauthed ? (
            <Alert tone="ok">Identity confirmed. Delete the account within the next few minutes or confirm again.</Alert>
          ) : (
            <>
              {hasPassword && (
                <Field label="Password" type="password" autoComplete="current-password" value={del.password} onChange={(e) => setDel({ ...del, password: e.target.value })} />
              )}
              {connected.length > 0 && (
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm text-muted">{hasPassword ? "Or confirm it is you with" : "Confirm it is you by signing in again with"}</span>
                  {connected.map((i) => (
                    <Button
                      key={i.identity_id}
                      type="button"
                      variant="secondary"
                      busy={redirecting === i.provider}
                      disabled={redirecting !== null}
                      onClick={() => confirmWith(i.provider as Provider)}
                    >
                      {redirecting !== i.provider && OAUTH[i.provider as Provider]?.icon}
                      {label(i.provider)}
                    </Button>
                  ))}
                </div>
              )}
            </>
          )}
          {verified.length > 0 && (
            <Field
              label="Authenticator code"
              inputMode="numeric"
              autoComplete="one-time-code"
              maxLength={6}
              value={del.code}
              onChange={(e) => setDel({ ...del, code: e.target.value.replace(/\D/g, "").slice(0, 6) })}
              className="max-w-[12rem] font-mono tracking-[0.3em]"
            />
          )}
          <Field label="Type DELETE MY ACCOUNT to confirm" value={del.confirm} onChange={(e) => setDel({ ...del, confirm: e.target.value })} autoComplete="off" />
          <Button
            type="submit"
            variant="danger"
            busy={busy === "delete"}
            disabled={(!reauthed && (!hasPassword || !del.password)) || del.confirm !== "DELETE MY ACCOUNT" || (verified.length > 0 && del.code.length !== 6)}
          >
            Delete my account
          </Button>
        </form>
      </Card>
    </Page>
  );
}
