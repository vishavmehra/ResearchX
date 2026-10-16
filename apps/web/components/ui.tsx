"use client";

import Link from "next/link";
import { forwardRef, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode } from "react";

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cx("font-serif text-[1.35rem] font-semibold tracking-tight text-ink", className)}>
      Research<span className="text-accent">X</span>
    </span>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg className={cx("rx-spin h-4 w-4", className)} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "danger"; busy?: boolean };

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "primary", busy, disabled, className, children, ...rest },
  ref,
) {
  const styles = {
    primary: "bg-accent text-accent-ink hover:opacity-90",
    secondary: "border border-line bg-surface text-ink hover:bg-surface-2",
    ghost: "text-muted hover:bg-surface-2 hover:text-ink",
    danger: "bg-bad text-white hover:opacity-90",
  }[variant];
  return (
    <button
      ref={ref}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-55",
        styles,
        className,
      )}
      {...rest}
    >
      {busy && <Spinner />}
      {children}
    </button>
  );
});

type FieldProps = InputHTMLAttributes<HTMLInputElement> & { label: string; hint?: ReactNode; error?: string | null };

export const Field = forwardRef<HTMLInputElement, FieldProps>(function Field({ label, hint, error, id, className, ...rest }, ref) {
  const fid = id ?? rest.name ?? label.toLowerCase().replace(/\W+/g, "-");
  return (
    <div className="space-y-1.5">
      <label htmlFor={fid} className="block text-sm font-medium text-ink">
        {label}
      </label>
      <input
        ref={ref}
        id={fid}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? `${fid}-error` : hint ? `${fid}-hint` : undefined}
        className={cx(
          "w-full rounded-lg border bg-surface px-3 py-2 text-[0.95rem] text-ink placeholder:text-faint",
          error ? "border-bad" : "border-line focus:border-accent",
          className,
        )}
        {...rest}
      />
      {error ? (
        <p id={`${fid}-error`} className="text-sm text-bad">
          {error}
        </p>
      ) : hint ? (
        <p id={`${fid}-hint`} className="text-sm text-muted">
          {hint}
        </p>
      ) : null}
    </div>
  );
});

export function Alert({ tone = "info", children }: { tone?: "info" | "ok" | "warn" | "bad"; children: ReactNode }) {
  const styles = {
    info: "border-line bg-info-soft text-ink",
    ok: "border-ok/30 bg-ok-soft text-ink",
    warn: "border-warn/30 bg-warn-soft text-ink",
    bad: "border-bad/30 bg-bad-soft text-ink",
  }[tone];
  return (
    <div role={tone === "bad" ? "alert" : "status"} className={cx("rounded-lg border px-3.5 py-2.5 text-sm leading-relaxed", styles)}>
      {children}
    </div>
  );
}

export function AuthShell({ title, subtitle, children, footer }: { title: string; subtitle?: ReactNode; children: ReactNode; footer?: ReactNode }) {
  return (
    <main className="flex min-h-full items-center justify-center px-4 py-12">
      <div className="w-full max-w-[400px]">
        <div className="mb-8 text-center">
          <Link href="/" aria-label="ResearchX home">
            <Wordmark className="text-[1.6rem]" />
          </Link>
          <p className="mt-1 text-sm text-muted">Evidence-first research, traceable to every source.</p>
        </div>
        <section className="rounded-2xl border border-line bg-surface p-6 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
          <h1 className="font-serif text-xl font-semibold text-ink">{title}</h1>
          {subtitle && <p className="mt-1 text-sm text-muted">{subtitle}</p>}
          <div className="mt-5">{children}</div>
        </section>
        {footer && <div className="mt-5 text-center text-sm text-muted">{footer}</div>}
      </div>
    </main>
  );
}

export function TextLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <Link href={href} className="font-medium text-accent underline-offset-4 hover:underline">
      {children}
    </Link>
  );
}
