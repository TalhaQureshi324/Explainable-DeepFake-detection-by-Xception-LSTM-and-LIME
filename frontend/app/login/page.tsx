"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { LogIn } from "lucide-react";
import { useI18n } from "@/lib/i18n/context";
import { useAuth } from "@/lib/auth/context";
import { ApiError } from "@/lib/api/client";
import { Alert, Button, ButtonLink, Card, CardContent } from "@/components/ui/primitives";

export default function LoginPage() {
  const { t } = useI18n();
  const { signIn } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await signIn(email, password);
      const next = new URLSearchParams(window.location.search).get("next");
      router.push(next || "/dashboard");
    } catch (err) {
      const apiErr = err as ApiError;
      setError(apiErr.status === 401 ? t.invalidCredentials : apiErr.detail);
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-md px-4 py-16">
      <Card className="animate-fade-up">
        <CardContent className="p-6">
          <h1 className="text-xl font-bold tracking-tight">{t.loginTitle}</h1>
          <form onSubmit={submit} className="mt-5 grid gap-4">
            <div>
              <label htmlFor="email" className="mb-1.5 block text-sm font-medium">
                {t.emailLabel}
              </label>
              <input
                id="email" type="email" required autoComplete="email"
                value={email} onChange={(e) => setEmail(e.target.value)}
                dir="ltr"
                className="h-10 w-full rounded-lg border border-line bg-surface-1 px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
              />
            </div>
            <div>
              <label htmlFor="password" className="mb-1.5 block text-sm font-medium">
                {t.passwordLabel}
              </label>
              <input
                id="password" type="password" required autoComplete="current-password"
                value={password} onChange={(e) => setPassword(e.target.value)}
                dir="ltr"
                className="h-10 w-full rounded-lg border border-line bg-surface-1 px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
              />
              <p className="mt-1 text-xs text-ink-3">{t.passwordHint}</p>
            </div>
            {error && <Alert tone="danger">{error}</Alert>}
            <Button type="submit" size="lg" disabled={busy}>
              <LogIn className="h-4 w-4 rtl:rotate-180" aria-hidden />
              {busy ? t.signingIn : t.signIn}
            </Button>
          </form>
          <p className="mt-4 text-center text-sm text-ink-2">
            {t.needAccount}{" "}
            <ButtonLink href="/register" variant="ghost" size="sm" className="text-accent">
              {t.signUp}
            </ButtonLink>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
