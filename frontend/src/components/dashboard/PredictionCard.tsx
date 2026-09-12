import { fmtCurrency, fmtPct, signalTone } from "@/lib/format";
import type { PredictionRow } from "@/lib/api";
import { ArrowDownRight, ArrowUpRight, Minus, AlertCircle } from "lucide-react";

const toneStyles: Record<string, string> = {
  bull: "bg-bull/10 text-bull border-bull/30",
  bear: "bg-bear/10 text-bear border-bear/30",
  neutral: "bg-muted text-muted-foreground border-border",
  destructive: "bg-destructive/10 text-destructive border-destructive/30",
};

export function SignalBadge({ signal }: { signal: string | null | undefined }) {
  const tone = signalTone(signal);
  const Icon =
    tone === "bull" ? ArrowUpRight : tone === "bear" ? ArrowDownRight : tone === "destructive" ? AlertCircle : Minus;
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-semibold border ${toneStyles[tone]}`}
    >
      <Icon className="size-3.5" />
      {signal ?? "—"}
    </span>
  );
}

export function PredictionCard({ row, loading }: { row: PredictionRow | null; loading?: boolean }) {
  if (loading) {
    return (
      <div className="rounded-xl border border-border bg-card p-5 animate-pulse h-[180px]" />
    );
  }
  if (!row) {
    return (
      <div className="rounded-xl border border-dashed border-border bg-card/50 p-5 text-sm text-muted-foreground">
        No prediction yet. Run the model to generate a signal.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs uppercase tracking-widest text-muted-foreground">RL Agent Signal</div>
          <div className="mt-2 font-mono text-2xl font-semibold">{row.Ticker}</div>
        </div>
        <SignalBadge signal={row.Signal} />
      </div>

      <div className="mt-5 grid grid-cols-2 gap-4 text-sm">
        <Stat label="RL Action" value={row["RL Action"] ?? "—"} mono />
        <Stat label="Transformer Return" value={fmtPct(row["Transformer Return"])} mono />
        <Stat label="Latest Price" value={fmtCurrency(row["Latest Adj Close"])} mono />
        <Stat label="As of" value={row["Latest Price Date"] ?? "—"} mono />
      </div>

      {row.Error && (
        <div className="mt-4 text-xs text-destructive font-mono">{row.Error}</div>
      )}
    </div>
  );
}

function Stat({ label, value, mono }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className={`mt-1 ${mono ? "font-mono" : ""} text-foreground`}>{value}</div>
    </div>
  );
}
