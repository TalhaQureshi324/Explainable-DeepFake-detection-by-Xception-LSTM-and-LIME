"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Fingerprint, Languages, Moon, Sun } from "lucide-react";
import { useI18n } from "@/lib/i18n/context";
import { useTheme } from "@/lib/theme/provider";

export function SiteHeader() {
  const { t, locale, setLocale } = useI18n();
  const { theme, setTheme } = useTheme();
  const pathname = usePathname();

  const linkCls = (active: boolean) =>
    `text-sm transition-colors hover:text-ink-1 ${
      active ? "text-ink-1 font-semibold" : "text-ink-2"
    }`;

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-surface-1/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4">
        <Link href="/" className="flex items-center gap-2">
          <Fingerprint className="h-5 w-5 text-accent" aria-hidden />
          <span className="font-semibold tracking-tight">{t.siteName}</span>
        </Link>

        <nav className="hidden items-center gap-5 sm:flex" aria-label="Main">
          <Link href="/" className={linkCls(pathname === "/")}>
            {t.navHome}
          </Link>
          <Link href="/analyze" className={linkCls(pathname.startsWith("/analyze"))}>
            {t.navAnalyze}
          </Link>
        </nav>

        <div className="flex items-center gap-2">
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
