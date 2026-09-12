export const fmtCurrency = (v: number | null | undefined, d = 2) =>
  v == null || isNaN(v as number) ? "—" : `$${(v as number).toFixed(d)}`;

export const fmtNum = (v: number | null | undefined, d = 2) =>
  v == null || isNaN(v as number) ? "—" : (v as number).toFixed(d);

export const fmtPct = (v: number | null | undefined, d = 2) =>
  v == null || isNaN(v as number) ? "—" : `${((v as number) * 100).toFixed(d)}%`;

export const fmtCompact = (v: number | null | undefined) => {
  if (v == null || isNaN(v as number)) return "—";
  const n = v as number;
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(2)}K`;
  return n.toFixed(0);
};

export const signalTone = (signal: string | null | undefined) => {
  const s = (signal ?? "").toUpperCase();
  if (s.includes("BUY")) return "bull";
  if (s.includes("SELL")) return "bear";
  if (s === "ERROR") return "destructive";
  return "neutral";
};
