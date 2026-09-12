import { createFileRoute, useParams } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { fetchDashboard, fetchMarket, runPredict, type PredictionRow } from "@/lib/api";
import { Brand, TickerNav } from "@/components/dashboard/TickerNav";
import { PredictionCard } from "@/components/dashboard/PredictionCard";
import { IndicatorChart, PriceChart } from "@/components/dashboard/Charts";
import { NewsList } from "@/components/dashboard/NewsList";
import { fmtCompact, fmtCurrency, fmtNum, fmtPct } from "@/lib/format";
import { Activity, AlertTriangle, RefreshCw } from "lucide-react";

export const Route = createFileRoute("/ticker/$ticker")({
  component: TickerPage,
});

function TickerPage() {
  const { ticker } = useParams({ from: "/ticker/$ticker" });
  const tickerUpper = ticker.toUpperCase();
  const [running, setRunning] = useState(false);

  const market = useQuery({
    queryKey: ["market", tickerUpper],
    queryFn: () => fetchMarket(tickerUpper, 180),
    refetchOnWindowFocus: false,
  });

  const dashboard = useQuery({
    queryKey: ["dashboard"],
    queryFn: fetchDashboard,
    refetchOnWindowFocus: false,
  });

  const prediction: PredictionRow | null =
    dashboard.data?.predictions?.find((p) => p.Ticker === tickerUpper) ?? null;

  const handleRun = async () => {
    setRunning(true);
    try {
      await runPredict(tickerUpper);
      await dashboard.refetch();
    } catch (e) {
      console.error(e);
    } finally {
      setRunning(false);
    }
  };

  const latest = market.data?.latest;
  const prevClose = market.data?.history?.at(-2)?.adj_close ?? null;
  const change =
    latest?.adj_close != null && prevClose != null ? latest.adj_close - prevClose : null;
  const changePct =
    change != null && prevClose ? change / prevClose : null;
  const isUp = (change ?? 0) >= 0;

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/60 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex flex-wrap items-center justify-between gap-4">
          <Brand />
          <TickerNav />
          <button
            onClick={handleRun}
            disabled={running}
            className="inline-flex items-center gap-2 rounded-md border border-border bg-card hover:bg-accent text-sm px-3 py-1.5 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`size-3.5 ${running ? "animate-spin" : ""}`} />
            {running ? "Running model…" : "Run RL prediction"}
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {market.isError && (
          <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-4 text-sm flex items-start gap-3">
            <AlertTriangle className="size-4 mt-0.5 text-destructive" />
            <div>
              <div className="font-medium text-destructive">Backend not reachable</div>
              <div className="text-muted-foreground mt-1">
                Make sure the Flask backend is running at <span className="font-mono">http://127.0.0.1:5000</span>.
                Start it with <span className="font-mono">python dashboard_backend.py</span>.
              </div>
            </div>
          </div>
        )}

        {/* Price header */}
        <section className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 rounded-xl border border-border bg-card p-6">
            <div className="flex items-end justify-between gap-4 flex-wrap">
              <div>
                <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-muted-foreground">
                  <Activity className="size-3.5" /> {tickerUpper} · Adjusted Close
                </div>
                <div className="mt-2 flex items-baseline gap-3">
                  <div className="font-mono text-4xl font-semibold tracking-tight">
                    {fmtCurrency(latest?.adj_close)}
                  </div>
                  {change != null && (
                    <div className={`font-mono text-sm ${isUp ? "text-bull" : "text-bear"}`}>
                      {isUp ? "+" : ""}
                      {fmtNum(change)} ({isUp ? "+" : ""}
                      {fmtPct(changePct)})
                    </div>
                  )}
                </div>
                <div className="mt-1 text-xs text-muted-foreground font-mono">
                  {latest?.date ?? "—"}
                </div>
              </div>
              <MiniStats latest={latest} />
            </div>
            <div className="mt-4">
              {market.isLoading ? (
                <div className="h-[340px] grid place-items-center text-sm text-muted-foreground">
                  Loading price history…
                </div>
              ) : (
                <PriceChart data={market.data?.history ?? []} />
              )}
            </div>
          </div>

          <PredictionCard row={prediction} loading={dashboard.isLoading} />
        </section>

        {/* Indicators */}
        <section className="grid md:grid-cols-2 gap-6">
          <IndicatorBlock title="MACD" subtitle="Trend momentum">
            <IndicatorChart data={market.data?.history ?? []} dataKey="macd" color="var(--primary)" />
          </IndicatorBlock>
          <IndicatorBlock title="RSI (14)" subtitle="Overbought / oversold">
            <IndicatorChart
              data={market.data?.history ?? []}
              dataKey="rsi"
              color="var(--bull)"
              refLines={[30, 70]}
            />
          </IndicatorBlock>
          <IndicatorBlock title="CCI" subtitle="Cyclic strength">
            <IndicatorChart data={market.data?.history ?? []} dataKey="cci" color="var(--bear)" refLines={[-100, 100]} />
          </IndicatorBlock>
          <IndicatorBlock title="ADX" subtitle="Trend strength">
            <IndicatorChart data={market.data?.history ?? []} dataKey="adx" color="var(--neutral)" refLines={[25]} />
          </IndicatorBlock>
        </section>

        {/* News */}
        <section className="rounded-xl border border-border bg-card p-6">
          <div className="flex items-center justify-between mb-2">
            <div>
              <div className="text-xs uppercase tracking-widest text-muted-foreground">News</div>
              <h2 className="text-lg font-semibold">Latest headlines for {tickerUpper}</h2>
            </div>
          </div>
          <NewsList items={market.data?.news ?? []} />
        </section>

        <footer className="text-xs text-muted-foreground text-center pt-4 pb-8">
          Data via yfinance · Predictions from your RL agent · {new Date().getFullYear()}
        </footer>
      </main>
    </div>
  );
}

function IndicatorBlock({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="flex items-baseline justify-between mb-1">
        <h3 className="text-sm font-semibold">{title}</h3>
        <span className="text-[10px] uppercase tracking-widest text-muted-foreground">{subtitle}</span>
      </div>
      {children}
    </div>
  );
}

function MiniStats({ latest }: { latest: any }) {
  const items = [
    { label: "Open", value: fmtCurrency(latest?.open) },
    { label: "High", value: fmtCurrency(latest?.high) },
    { label: "Low", value: fmtCurrency(latest?.low) },
    { label: "Volume", value: fmtCompact(latest?.volume) },
    { label: "Vol 15d", value: fmtPct(latest?.volatility_15d) },
  ];
  return (
    <div className="grid grid-cols-3 sm:grid-cols-5 gap-x-5 gap-y-2">
      {items.map((s) => (
        <div key={s.label}>
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground">{s.label}</div>
          <div className="font-mono text-sm">{s.value}</div>
        </div>
      ))}
    </div>
  );
}
