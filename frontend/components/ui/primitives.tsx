/**
 * Minimal shadcn-style UI primitives: accessible, themeable, dependency-free.
 * (Chosen over full shadcn/ui install to keep the dependency tree light; the
 * class API intentionally mirrors it, so swapping later is mechanical.)
 */

import { forwardRef } from "react";
import Link from "next/link";

const cx = (...parts: Array<string | false | null | undefined>) =>
  parts.filter(Boolean).join(" ");

/* --------------------------------- Button --------------------------------- */

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md" | "lg";

const buttonStyles = {
  base: "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 disabled:pointer-events-none disabled:opacity-50",
  variant: {
    primary: "bg-accent text-white hover:opacity-90",
    secondary:
      "border border-line bg-surface-1 text-ink-1 hover:bg-surface-3",
    ghost: "text-ink-2 hover:bg-surface-3 hover:text-ink-1",
    danger: "bg-verdict-fake text-white hover:opacity-90",
  } as Record<ButtonVariant, string>,
  size: {
    sm: "h-8 px-3 text-sm",
    md: "h-10 px-4 text-sm",
    lg: "h-12 px-6 text-base",
  } as Record<ButtonSize, string>,
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", type = "button", ...props }, ref) => (
    <button
      ref={ref}
      type={type}
      className={cx(buttonStyles.base, buttonStyles.variant[variant], buttonStyles.size[size], className)}
      {...props}
    />
  ),
);
Button.displayName = "Button";

export interface ButtonLinkProps extends React.ComponentProps<typeof Link> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export function ButtonLink({ className, variant = "primary", size = "md", ...props }: ButtonLinkProps) {
  return (
    <Link
      className={cx(buttonStyles.base, buttonStyles.variant[variant], buttonStyles.size[size], className)}
      {...props}
    />
  );
}

/* ---------------------------------- Card ---------------------------------- */

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cx("rounded-xl border border-line bg-surface-1 shadow-sm", className)}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cx("border-b border-line px-5 py-4", className)} {...props} />;
}

export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cx("text-sm font-semibold tracking-wide", className)} {...props} />;
}

export function CardContent({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cx("px-5 py-4", className)} {...props} />;
}

/* --------------------------------- Badge ---------------------------------- */

type BadgeTone = "neutral" | "accent" | "real" | "fake" | "warning";

const badgeTones: Record<BadgeTone, string> = {
  neutral: "bg-surface-3 text-ink-2 border-line",
  accent: "bg-accent/10 text-accent border-accent/30",
  real: "bg-verdict-real/10 text-verdict-real border-verdict-real/30",
  fake: "bg-verdict-fake/10 text-verdict-fake border-verdict-fake/30",
  warning: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30",
};

export function Badge({
  tone = "neutral",
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        badgeTones[tone],
        className,
      )}
      {...props}
    />
  );
}

/* -------------------------------- Skeleton -------------------------------- */

export function Skeleton({ className }: { className?: string }) {
  return (
    <div className={cx("animate-pulse rounded-md bg-surface-3", className)} />
  );
}

/* --------------------------------- Alert ---------------------------------- */

export function Alert({
  tone = "warning",
  title,
  children,
  className,
}: {
  tone?: "warning" | "danger" | "info";
  title?: string;
  children?: React.ReactNode;
  className?: string;
}) {
  const tones = {
    warning: "border-amber-500/40 bg-amber-500/5 text-amber-700 dark:text-amber-400",
    danger: "border-verdict-fake/40 bg-verdict-fake/5 text-verdict-fake",
    info: "border-accent/40 bg-accent/5 text-ink-2",
  } as const;
  return (
    <div className={cx("rounded-lg border px-4 py-3 text-sm", tones[tone], className)} role="alert">
      {title ? <p className="font-semibold">{title}</p> : null}
      {children ? <div className="mt-1 opacity-90">{children}</div> : null}
    </div>
  );
}

/* --------------------------------- Select --------------------------------- */

export function Select({
  className,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cx(
        "h-10 rounded-lg border border-line bg-surface-1 px-3 text-sm text-ink-1 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
        className,
      )}
      {...props}
    />
  );
}

/* --------------------------------- Switch --------------------------------- */

export function Switch({
  checked,
  onChange,
  label,
  hint,
  id,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
  id?: string;
}) {
  return (
    <label htmlFor={id} className="flex cursor-pointer items-start gap-3">
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cx(
          "mt-0.5 h-5 w-9 shrink-0 rounded-full border border-line transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
          checked ? "bg-accent" : "bg-surface-3",
        )}
      >
        <span
          className={cx(
            "block h-4 w-4 rounded-full bg-white shadow transition-transform",
            checked ? "translate-x-4" : "translate-x-0.5",
          )}
        />
      </button>
      <span>
        <span className="block text-sm font-medium text-ink-1">{label}</span>
        {hint ? <span className="block text-xs text-ink-3">{hint}</span> : null}
      </span>
    </label>
  );
}
