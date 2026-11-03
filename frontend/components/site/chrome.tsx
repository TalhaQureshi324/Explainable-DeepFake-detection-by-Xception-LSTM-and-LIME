"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Fingerprint, Languages, LogIn, LogOut, Moon, Sun, UserRound } from "lucide-react";
import { useI18n } from "@/lib/i18n/context";
import { useTheme } from "@/lib/theme/provider";
import { useAuth } from "@/lib/auth/context";

export function SiteHeader() {
  const { t, locale, setLocale } = useI18n();
  const { theme, setTheme } = useTheme();
  const { user, signOut } = useAuth();
  const pathname = usePathname();
  const router = useRouter();

  const linkCls = (active: boolean) =>
    `text-sm transition-colors hover:text-ink-1 ${
      active ? "text-ink-1 font-semibold" : "text-ink-2"
    }`;

  const onSignOut = async () => {
    await signOut();
    router.push("/");
  };

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-surface-1/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4">
        <Link href="/" className="flex items-center gap-2">
          <Fingerprint className="h-5 w-5 text-accent" aria-hidden />
          <span className="font-semibold tracking-tight">{t.siteName}</span>
        </Link>

        <nav className="hidden items-center gap-5 md:flex" aria-label="Main">
          <Link href="/" className={linkCls(pathname === "/")}>{t.navHome}</Link>
          <Link href="/analyze" className={linkCls(pathname.startsWith("/analyze"))}>{t.navAnalyze}</Link>
          <Link href="/dashboard" className={linkCls(pathname.startsWith("/dashboard"))}>{t.navDashboard}</Link>
          <Link href="/history" className={linkCls(pathname.startsWith("/history"))}>{t.navHistory}</Link>
        </nav>

        <div className="flex items-center gap-2">
          {user ? (
            <div className="flex items-center gap-2" title={t.accountMenu}>
              <UserRound className="hidden h-4 w-4 text-ink-3 sm:inline" aria-hidden />
              <span className="hidden max-w-40 truncate text-xs text-ink-2 sm:inline" dir="ltr">
                {user.email}
              </span>
              <button
                type="button"
                onClick={onSignOut}
                className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-line px-2.5 text-sm text-ink-2 transition-colors hover:bg-surface-3 hover:text-ink-1"
                aria-label={t.signOut}
              >
                <LogOut className="h-4 w-4 rtl:rotate-180" aria-hidden />
                <span className="hidden sm:inline">{t.signOut}</span>
              </button>
            </div>
          ) : (
            <Link
              href="/login"
              className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-line px-2.5 text-sm text-ink-2 transition-colors hover:bg-surface-3 hover:text-ink-1"
            >
              <LogIn className="h-4 w-4 rtl:rotate-180" aria-hidden />
              <span className="hidden sm:inline">{t.signIn}</span>
            </Link>
          )}
          <button
            type="button"
            onClick={() => setLocale(locale === "en" ? "ur" : "en")}
            className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-line px-2.5 text-sm text-ink-2 transition-colors hover:bg-surface-3 hover:text-ink-1"
            aria-label={t.languageLabel}
          >
            <Languages className="h-4 w-4" aria-hidden />
            <span>{locale === "en" ? "اردو" : "EN"}</span>
          </button>
          <button
            type="button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-line text-ink-2 transition-colors hover:bg-surface-3 hover:text-ink-1"
            aria-label={t.themeLabel}
          >
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </div>
      {/* mobile nav row */}
      <nav className="flex items-center justify-center gap-5 border-t border-line py-2 md:hidden" aria-label="Mobile">
        <Link href="/" className={linkCls(pathname === "/")}>{t.navHome}</Link>
        <Link href="/analyze" className={linkCls(pathname.startsWith("/analyze"))}>{t.navAnalyze}</Link>
        <Link href="/dashboard" className={linkCls(pathname.startsWith("/dashboard"))}>{t.navDashboard}</Link>
        <Link href="/history" className={linkCls(pathname.startsWith("/history"))}>{t.navHistory}</Link>
      </nav>
    </header>
  );
}

export function SiteFooter() {
  const { t } = useI18n();
  return (
    <footer className="border-t border-line bg-surface-1">
      <div className="mx-auto max-w-6xl px-4 py-6 text-center text-xs text-ink-3">
        {t.footerNote}
      </div>
    </footer>
  );
}
