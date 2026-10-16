"use client";

import type { Provider } from "@supabase/supabase-js";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { supabase } from "@/lib/supabase/client";
import * as I from "@/components/icons";
import { Alert, AuthShell, Button, Field, TextLink } from "@/components/ui";

const MIN_PASSWORD = 10;

export const NOTICES: Record<string, { tone: "ok" | "warn" | "info" | "bad"; text: string }> = {
  confirmed: { tone: "ok", text: "Email confirmed. Sign in to continue." },
  link: { tone: "warn", text: "That link is invalid or has expired. Request a new one." },
  password: { tone: "ok", text: "Password updated. Sign in with your new password." },
  deleted: { tone: "info", text: "Your account and all of its data have been deleted." },
  signedout: { tone: "info", text: "You have been signed out." },
  oauth: { tone: "warn", text: "Signing in with that account did not complete. Try again, or use your email address." },
  oauth_cancelled: { tone: "info", text: "Sign-in was cancelled." },
  oauth_unverified: { tone: "warn", text: "Verify your email address with that provider first, then try again." },
  identity_exists: { tone: "bad", text: "That account is already connected to a different ResearchX user." },
  linking_disabled: { tone: "warn", text: "Connecting another sign-in method is not enabled on this server." },
  linked: { tone: "ok", text: "Sign-in method connected." },
};

/** A notice key from the URL, or nothing: only keys of NOTICES are shown. */
export const noticeOf = (key?: string | null) => (key && Object.hasOwn(NOTICES, key) ? NOTICES[key] : undefined);

// OAuth providers the UI can present, in display order. One is offered only when Supabase Auth has it switched on
// (authProviders in lib/supabase/server.ts); sign-in, sign-up and connecting in Settings all use this list.
export const OAUTH: Partial<Record<Provider, { label: string; icon?: ReactNode }>> = {
  google: { label: "Google", icon: <I.Google /> },
  github: { label: "GitHub", icon: <I.GitHub /> },
  gitlab: { label: "GitLab" },
  azure: { label: "Microsoft" },
  apple: { label: "Apple" },
  linkedin_oidc: { label: "LinkedIn" },
};
export const oauthOf = (enabled: string[]) => (Object.keys(OAUTH) as Provider[]).filter((p) => enabled.includes(p));
// Always shown, as disabled buttons until switched on in Supabase, so a missing configuration is visible instead of silent.
export const PRIMARY_OAUTH: Provider[] = ["google", "github"];
export const offeredOf = (enabled: string[]) => [...new Set([...PRIMARY_OAUTH, ...oauthOf(enabled)])];
export const notEnabledNote = (off: Provider[]) =>
  off.length ? `${off.map((p) => OAUTH[p]?.label).join(" and ")} sign-in ${off.length > 1 ? "are" : "is"} not enabled for this ResearchX project yet.` : null;

/** Where a provider sends the browser back; `flow` tells the callback where to report a failure. */
export const callbackUrl = (flow: "oauth" | "link" | "reauth", next: string) =>
  `${window.location.origin}/auth/callback?flow=${flow}&next=${encodeURIComponent(next)}`;

/** Busy state for a provider redirect, cleared if the browser restores this page from its back/forward cache. */
export function useRedirectBusy() {
  const [busy, setBusy] = useState<string | null>(null);
  useEffect(() => {
    const reset = (e: PageTransitionEvent) => e.persisted && setBusy(null);
    window.addEventListener("pageshow", reset);
    return () => window.removeEventListener("pageshow", reset);
  }, []);
  return [busy, setBusy] as const;
}

function OAuthButtons({ providers, next }: { providers: string[]; next: string }) {
  const [busy, setBusy] = useRedirectBusy();
  const [error, setError] = useState<string | null>(null);
  const shown = offeredOf(providers);
  const off = shown.filter((p) => !providers.includes(p));

  async function start(provider: Provider) {
    setBusy(provider);
    setError(null);
    const { error } = await supabase().auth.signInWithOAuth({ provider, options: { redirectTo: callbackUrl("oauth", next) } });
    if (error) {
      // otherwise the browser is already on its way to the provider
      setBusy(null);
      setError(`Could not start signing in with ${OAUTH[provider]?.label}. Try again.`);
    }
  }

  return (
    <div className="mb-5 space-y-2">
      {error && <Alert tone="bad">{error}</Alert>}
      {shown.map((p) => (
        <Button key={p} type="button" variant="secondary" className="w-full" busy={busy === p} disabled={busy !== null || off.includes(p)}
          title={off.includes(p) ? notEnabledNote([p]) ?? undefined : undefined} onClick={() => start(p)}>
          {busy !== p && OAUTH[p]?.icon}
          Continue with {OAUTH[p]?.label}
          {off.includes(p) && <span className="text-xs font-normal text-faint">· not enabled yet</span>}
        </Button>
      ))}
      {off.length > 0 && <p className="text-xs text-faint">{notEnabledNote(off)}</p>}
      <p className="flex items-center gap-3 pt-3 text-xs text-faint">
        <span className="h-px flex-1 bg-line" />
        or with your email
        <span className="h-px flex-1 bg-line" />
      </p>
    </div>
  );
}

function passwordProblem(pw: string): string | null {
  if (pw.length < MIN_PASSWORD) return `Use at least ${MIN_PASSWORD} characters.`;
  if (!/[a-z]/i.test(pw) || !/\d/.test(pw)) return "Mix letters and numbers.";
  return null;
}

export function LoginForm({ next, notice, providers }: { next: string; notice?: string; providers: string[] }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const { error } = await supabase().auth.signInWithPassword({ email: email.trim(), password });
    if (error) {
      setBusy(false);
      setError(
        error.code === "email_not_confirmed"
          ? "Confirm your email address first: open the link we sent you."
          : error.status === 429
            ? "Too many attempts. Wait a minute and try again."
            : error.status && error.status >= 500
              ? "The sign-in service is unavailable. Try again shortly."
              : "Incorrect email or password.",
      );
      return;
    }
    // full navigation: the proxy sees the new session cookies once (replace + refresh raced and 404'd an RSC request);
    // it sends MFA users to the challenge first
    window.location.replace(next);
  }

  const n = noticeOf(notice);
  return (
    <AuthShell
      title="Sign in"
      footer={
        <>
          New to ResearchX? <TextLink href="/signup">Create an account</TextLink>
        </>
      }
    >
      {n && (
        <div className="mb-4">
          <Alert tone={n.tone}>{n.text}</Alert>
        </div>
      )}
      <OAuthButtons providers={providers} next={next} />
      <form onSubmit={submit} className="space-y-4" noValidate>
        {error && <Alert tone="bad">{error}</Alert>}
        <Field label="Email" name="email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        <Field
          label="Password"
          name="password"
          type="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <div className="flex items-center justify-between">
          <TextLink href="/forgot-password">Forgot password?</TextLink>
          <Button type="submit" busy={busy} disabled={!email || !password}>
            Sign in
          </Button>
        </div>
      </form>
    </AuthShell>
  );
}

export function SignupForm({ providers }: { providers: string[] }) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const weak = password ? passwordProblem(password) : null;

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (weak || password !== confirm) return;
    setBusy(true);
    setError(null);
    const { data, error } = await supabase().auth.signUp({
      email: email.trim(),
      password,
      options: { emailRedirectTo: `${window.location.origin}/auth/callback?next=/` },
    });
    setBusy(false);
    if (error) {
      setError(
        error.code === "weak_password"
          ? "That password is too weak. Choose a longer, less common one."
          : error.status === 429
            ? "Too many sign-up attempts. Wait a few minutes and try again."
            : error.code === "email_address_invalid"
              ? "Enter a valid email address."
              : "We could not create the account. Try again.",
      );
      return;
    }
    if (data.session) {
      router.replace("/");
      router.refresh();
    } else setSent(true); // same message whether or not the address was already registered
  }

  if (sent)
    return (
      <AuthShell title="Check your inbox" footer={<TextLink href="/login">Back to sign in</TextLink>}>
        <Alert tone="ok">
          If <strong>{email.trim()}</strong> can be registered, a confirmation link is on its way. Open it to activate your account.
        </Alert>
      </AuthShell>
    );

  return (
    <AuthShell
      title="Create your account"
      subtitle="Your chats, research runs and documents stay private to you."
      footer={
        <>
          Already have an account? <TextLink href="/login">Sign in</TextLink>
        </>
      }
    >
      <OAuthButtons providers={providers} next="/" />
      <form onSubmit={submit} className="space-y-4" noValidate>
        {error && <Alert tone="bad">{error}</Alert>}
        <Field label="Email" name="email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        <Field
          label="Password"
          name="password"
          type="password"
          autoComplete="new-password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          hint={`At least ${MIN_PASSWORD} characters, letters and numbers.`}
          error={password && weak ? weak : null}
        />
        <Field
          label="Confirm password"
          name="confirm"
          type="password"
          autoComplete="new-password"
          required
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          error={confirm && confirm !== password ? "Passwords do not match." : null}
        />
        <Button type="submit" className="w-full" busy={busy} disabled={!email || !!weak || !password || password !== confirm}>
          Create account
        </Button>
      </form>
    </AuthShell>
  );
}

export function ForgotForm() {
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const { error } = await supabase().auth.resetPasswordForEmail(email.trim(), {
      redirectTo: `${window.location.origin}/auth/callback?next=/reset-password`,
    });
    setBusy(false);
    if (error && error.status === 429) setError("Too many requests. Wait a few minutes and try again.");
    else setDone(true); // never reveal whether the address has an account
  }

  return (
    <AuthShell title="Reset your password" footer={<TextLink href="/login">Back to sign in</TextLink>}>
      {done ? (
        <Alert tone="ok">If an account exists for that address, we have sent a link to choose a new password. It expires in one hour.</Alert>
      ) : (
        <form onSubmit={submit} className="space-y-4" noValidate>
          {error && <Alert tone="bad">{error}</Alert>}
          <Field label="Email" name="email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
          <Button type="submit" className="w-full" busy={busy} disabled={!email}>
            Send reset link
          </Button>
        </form>
      )}
    </AuthShell>
  );
}

export function ResetForm() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const weak = password ? passwordProblem(password) : null;

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (weak || password !== confirm) return;
    setBusy(true);
    setError(null);
    const client = supabase();
    const { error } = await client.auth.updateUser({ password });
    if (error) {
      setBusy(false);
      setError(
        error.code === "same_password"
          ? "Choose a password you have not used for this account."
          : error.code === "insufficient_aal"
            ? "Complete two-factor authentication before changing the password."
            : error.code === "weak_password"
              ? "That password is too weak."
              : "The reset link has expired. Request a new one.",
      );
      return;
    }
    await client.auth.signOut({ scope: "global" }); // a reset ends every other session too
    router.replace("/login?notice=password");
  }

  return (
    <AuthShell title="Choose a new password">
      <form onSubmit={submit} className="space-y-4" noValidate>
        {error && <Alert tone="bad">{error}</Alert>}
        <Field
          label="New password"
          name="password"
          type="password"
          autoComplete="new-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          error={password && weak ? weak : null}
          hint={`At least ${MIN_PASSWORD} characters, letters and numbers.`}
        />
        <Field
          label="Confirm new password"
          name="confirm"
          type="password"
          autoComplete="new-password"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          error={confirm && confirm !== password ? "Passwords do not match." : null}
        />
        <Button type="submit" className="w-full" busy={busy} disabled={!!weak || !password || password !== confirm}>
          Update password
        </Button>
      </form>
    </AuthShell>
  );
}

export function MfaChallenge({ next }: { next: string }) {
  const router = useRouter();
  const [factorId, setFactorId] = useState<string | null>(null);
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    supabase()
      .auth.mfa.listFactors()
      .then(({ data, error }) => {
        if (error) return setError("We could not load your two-factor settings. Refresh to try again.");
        const totp = data?.totp?.[0];
        if (!totp) return router.replace(next);
        setFactorId(totp.id);
        input.current?.focus();
      });
  }, [next, router]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!factorId || code.length !== 6) return;
    setBusy(true);
    setError(null);
    const { error } = await supabase().auth.mfa.challengeAndVerify({ factorId, code });
    if (error) {
      setBusy(false);
      setCode("");
      setError(
        error.status === 429
          ? "Too many attempts. Wait a minute and try again."
          : error.code === "mfa_challenge_expired"
            ? "That challenge expired. Enter the current code."
            : "That code did not work. Check that your device clock is correct and enter the current code.",
      );
      input.current?.focus();
      return;
    }
    window.location.replace(next); // full navigation with the aal2 session cookies
  }

  async function cancel() {
    await supabase().auth.signOut();
    router.replace("/login?notice=signedout");
  }

  return (
    <AuthShell title="Two-factor authentication" subtitle="Enter the 6-digit code from your authenticator app.">
      <form onSubmit={submit} className="space-y-4" noValidate>
        {error && <Alert tone="bad">{error}</Alert>}
        <Field
          ref={input}
          label="Authentication code"
          name="code"
          inputMode="numeric"
          autoComplete="one-time-code"
          pattern="\d{6}"
          maxLength={6}
          placeholder="123456"
          value={code}
          onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
          className="text-center font-mono text-lg tracking-[0.4em]"
        />
        <Button type="submit" className="w-full" busy={busy} disabled={!factorId || code.length !== 6}>
          Verify
        </Button>
        <Button type="button" variant="ghost" className="w-full" onClick={cancel}>
          Cancel and sign out
        </Button>
      </form>
    </AuthShell>
  );
}
