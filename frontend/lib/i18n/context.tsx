"use client";

/**
 * Locale provider: EN/UR switching + <html dir> handling.
 * Persisted in localStorage; defaults to English.
 */

import {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
} from "react";
import { dictionaries, type Dictionary, type Locale } from "./dictionaries";

interface LocaleContextValue {
  locale: Locale;
  t: Dictionary;
  setLocale: (l: Locale) => void;
  dir: "ltr" | "rtl";
}

const LocaleContext = createContext<LocaleContextValue | null>(null);

const STORAGE_KEY = "df-locale";

export function LocaleProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en");

  useEffect(() => {
    const qp = new URLSearchParams(window.location.search).get("lang");
    if (qp === "en" || qp === "ur") {
      setLocaleState(qp);
      return;
    }
    const saved = window.localStorage.getItem(STORAGE_KEY) as Locale | null;
    if (saved === "en" || saved === "ur") setLocaleState(saved);
  }, []);

  useEffect(() => {
    const dir = locale === "ur" ? "rtl" : "ltr";
    document.documentElement.setAttribute("dir", dir);
    document.documentElement.setAttribute("lang", locale);
  }, [locale]);

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l);
    window.localStorage.setItem(STORAGE_KEY, l);
  }, []);

  const value = useMemo<LocaleContextValue>(
    () => ({
      locale,
      t: dictionaries[locale],
      setLocale,
      dir: locale === "ur" ? "rtl" : "ltr",
    }),
    [locale, setLocale],
  );

  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>;
}

export function useI18n(): LocaleContextValue {
  const ctx = useContext(LocaleContext);
  if (!ctx) throw new Error("useI18n must be used inside <LocaleProvider>");
  return ctx;
}
