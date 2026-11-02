"use client";

/**
 * Window-score timeline (Recharts). Windows are placed on the REAL video
 * timeline using start/end seconds captured during face extraction; the
 * decision threshold is drawn as a reference line. Tooltips show the original
 * timestamps and the model window label — nothing is invented.
 */

import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useI18n } from "@/lib/i18n/context";
import type { WindowScore } from "@/lib/api/types";

export function WindowsChart({
  windows,
  threshold,
}: {
  windows: WindowScore[];
  threshold: number | null;
}) {
  const { t } = useI18n();

  const data = windows
    .filter((w) => w.start_seconds !== null || w.end_seconds !== null)
    .map((w, i) => ({
      name: w.label,
      mid: ((w.start_seconds ?? 0) + (w.end_seconds ?? w.start_seconds ?? 0)) / 2,
      score: w.score,
      start: w.start_seconds,
      end: w.end_seconds,
      i,
    }));

  if (data.length === 0) {
    return <p className="px-5 py-4 text-sm text-ink-3">{t.timelineEmpty}</p>;
  }

  const fmt = (s: number | null | undefined) =>
    typeof s === "number" ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}` : "—";

  return (
    <div className="h-64 px-2 py-4" dir="ltr">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--line))" />
          <XAxis
            dataKey="mid"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={fmt}
            tick={{ fill: "rgb(var(--ink-3))", fontSize: 11 }}
            stroke="rgb(var(--line))"
            label={{
              value: "video time (mm:ss)",
              position: "insideBottom",
              offset: -4,
              fill: "rgb(var(--ink-3))",
              fontSize: 11,
            }}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tick={{ fill: "rgb(var(--ink-3))", fontSize: 11 }}
            stroke="rgb(var(--line))"
            width={40}
          />
          <Tooltip
            contentStyle={{
              background: "rgb(var(--surface-1))",
              border: "1px solid rgb(var(--line))",
              borderRadius: 8,
              color: "rgb(var(--ink-1))",
              fontSize: 12,
            }}
            formatter={(value) => [Number(value).toFixed(4), "fake score"]}
            labelFormatter={(label, payload) => {
              const p = payload?.[0]?.payload as (typeof data)[number] | undefined;
              return p ? `${p.name} · ${fmt(p.start)} → ${fmt(p.end)}` : String(label);
            }}
          />
          {threshold !== null && (
            <ReferenceLine
              y={threshold}
              stroke="rgb(var(--verdict-fake))"
              strokeDasharray="6 4"
              label={{
                value: `threshold ${threshold}`,
                position: "right",
                fill: "rgb(var(--verdict-fake))",
                fontSize: 11,
              }}
            />
          )}
          <Line
            type="monotone"
            dataKey="score"
            stroke="rgb(var(--accent))"
            strokeWidth={2}
            dot={{ r: 4, fill: "rgb(var(--accent))" }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="mt-1 px-3 text-xs text-ink-3">{t.timelineNote}</p>
    </div>
  );
}
