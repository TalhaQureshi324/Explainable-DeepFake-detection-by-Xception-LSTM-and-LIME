import type { Metadata } from "next";
import localFont from "next/font/local";
import { Inter } from "next/font/google";
import { Noto_Nastaliq_Urdu } from "next/font/google";
import "./globals.css";
import { LocaleProvider } from "@/lib/i18n/context";
import { ThemeProvider } from "@/lib/theme/provider";
import { SiteFooter, SiteHeader } from "@/components/site/chrome";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});
const nastaliq = Noto_Nastaliq_Urdu({
  subsets: ["arabic"],
  weight: ["400", "500"],
  variable: "--font-nastaliq",
});

export const metadata: Metadata = {
  title: {
    default: "DeepFake Forensics — Explainable Deepfake Detection",
    template: "%s · DeepFake Forensics",
  },
  description:
    "Research platform for explainable deepfake detection: Xception-LSTM spatiotemporal analysis with LIME explanations over image and video media.",
};

// Prevents a flash of the wrong theme before React hydrates.
const themeScript = `(function(){try{var q=new URLSearchParams(location.search);var t=q.get('theme')||localStorage.getItem('df-theme');if(!t){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}if(t==='dark')document.documentElement.classList.add('dark');var l=q.get('lang')||localStorage.getItem('df-locale');if(l==='ur'){document.documentElement.setAttribute('dir','rtl');document.documentElement.setAttribute('lang','ur');}}catch(e){}})();`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" dir="ltr" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${inter.variable} ${nastaliq.variable} antialiased`}
        style={{ ["--font-sans" as string]: "var(--font-inter)" }}
      >
        <ThemeProvider>
          <LocaleProvider>
            <div className="flex min-h-screen flex-col">
              <SiteHeader />
              <main className="flex-1">{children}</main>
              <SiteFooter />
            </div>
          </LocaleProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
