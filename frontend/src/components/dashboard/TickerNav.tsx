import { TICKERS } from "@/lib/api";
import { Link, useParams } from "@tanstack/react-router";
import { LineChart as ChartIcon } from "lucide-react";

export function TickerNav() {
  const params = useParams({ strict: false }) as { ticker?: string };
  const active = (params.ticker ?? "AAPL").toUpperCase();

  return (
    <nav className="flex flex-wrap items-center gap-2">
      {TICKERS.map((t) => (
        <Link
          key={t}
          to="/ticker/$ticker"
          params={{ ticker: t }}
          className={`font-mono text-xs tracking-wider px-3 py-1.5 rounded-md border transition-colors ${
            active === t
              ? "bg-primary text-primary-foreground border-primary"
              : "bg-card text-muted-foreground border-border hover:text-foreground hover:border-foreground/20"
          }`}
        >
          {t}
        </Link>
      ))}
    </nav>
  );
}

export function Brand() {
  return (
    <Link to="/" className="flex items-center gap-2">
      <div className="size-8 rounded-lg bg-primary text-primary-foreground grid place-items-center">
        <ChartIcon className="size-4" />
      </div>
      <div className="leading-tight">
        <div className="text-sm font-semibold">Quant Desk</div>
        <div className="text-[10px] uppercase tracking-widest text-muted-foreground">RL Predictions</div>
      </div>
    </Link>
  );
}
