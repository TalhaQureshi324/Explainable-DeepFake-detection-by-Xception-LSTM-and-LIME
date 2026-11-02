"use client";

import {
  ArrowRight, BadgeCheck, FlaskConical, ScanFace, Scale, ShieldAlert, Waves,
} from "lucide-react";
import { useI18n } from "@/lib/i18n/context";
import { Alert, Badge, ButtonLink, Card, CardContent } from "@/components/ui/primitives";

const stepIcons = [ScanFace, Waves, Scale, FlaskConical];

export default function Home() {
  const { t } = useI18n();

  return (
    <div className="mx-auto max-w-6xl px-4 pb-16">
      {/* Hero */}
      <section className="animate-fade-up py-14 text-center sm:py-20">
        <Badge tone="accent" className="mx-auto">
          <FlaskConical className="h-3.5 w-3.5" aria-hidden />
          {t.homeBadge}
        </Badge>
        <h1 className="mx-auto mt-5 max-w-3xl text-balance text-4xl font-bold tracking-tight sm:text-5xl">
          {t.homeTitle}
        </h1>
        <p className="mx-auto mt-5 max-w-2xl text-balance text-base leading-relaxed text-ink-2">
          {t.homeSubtitle}
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <ButtonLink href="/analyze" size="lg">
            {t.homeCta}
            <ArrowRight className="h-4 w-4 rtl:rotate-180" aria-hidden />
          </ButtonLink>
          <ButtonLink href="#how-it-works" variant="secondary" size="lg">
            {t.homeSecondaryCta}
          </ButtonLink>
        </div>
      </section>

      {/* Disclaimer */}
      <Alert tone="warning" className="animate-fade-up">
        <p className="flex items-start gap-2">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <span>
            <strong>{t.disclaimerTitle}: </strong>
            {t.disclaimerLong}
          </span>
        </p>
      </Alert>

      {/* Architecture */}
      <section id="how-it-works" className="mt-16 scroll-mt-20">
        <h2 className="text-center text-2xl font-bold tracking-tight">{t.archTitle}</h2>
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {t.archSteps.map((step, i) => {
            const Icon = stepIcons[i % stepIcons.length];
            return (
              <Card key={step.title} className="animate-fade-up">
                <CardContent className="p-5">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10">
                    <Icon className="h-5 w-5 text-accent" aria-hidden />
                  </div>
                  <h3 className="mt-4 font-semibold">
                    <span className="text-ink-3">{i + 1}. </span>
                    {step.title}
                  </h3>
                  <p className="mt-2 text-sm leading-relaxed text-ink-2">{step.body}</p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      {/* Honest limitations */}
      <section className="mt-14">
        <Card className="bg-surface-2">
          <CardContent className="flex flex-col items-start gap-3 p-6 sm:flex-row sm:items-center">
            <BadgeCheck className="h-6 w-6 shrink-0 text-accent" aria-hidden />
            <div>
              <h3 className="font-semibold">{t.pipelineTitle}</h3>
              <p className="mt-1 text-sm leading-relaxed text-ink-2">{t.pipelineBody}</p>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Final CTA */}
      <section className="mt-14 rounded-xl border border-line bg-surface-1 p-8 text-center">
        <h2 className="text-xl font-bold">{t.homeCta}</h2>
        <p className="mx-auto mt-2 max-w-xl text-sm text-ink-2">{t.uploadSubtitle}</p>
        <ButtonLink href="/analyze" size="lg" className="mt-5">
          {t.homeCta}
          <ArrowRight className="h-4 w-4 rtl:rotate-180" aria-hidden />
        </ButtonLink>
      </section>
    </div>
  );
}
