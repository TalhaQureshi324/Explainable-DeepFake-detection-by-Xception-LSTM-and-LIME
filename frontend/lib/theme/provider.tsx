"use client";

/**
 * Theme provider: light/dark via the `dark` class on <html>.
 * Inline script in the layout prevents a flash of the wrong theme.
 */

import {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
} from "react";

type Theme = "light" | "dark";

const ThemeContext = createContext<{ theme: Theme; setTheme: (t: Theme) => void } | null>(null);
const STORAGE_KEY = "df-theme";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>("light");

  useEffect(() => {
    const qp = new URLSearchParams(window.location.search).get("theme");
    if (qp === "light" || qp === "dark") {
      setThemeState(qp);
      return;
    }
    const saved = window.localStorage.getItem(STORAGE_KEY) as Theme | null;
    const initial: Theme =
      saved ?? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    setThemeState(initial);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
    window.localStorage.setItem(STORAGE_KEY, t);
  }, []);

  return (
    <ThemeContext.Provider value={useMemo(() => ({ theme, setTheme }), [theme, setTheme])}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside <ThemeProvider>");
  return ctx;
}
