import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine,
} from "recharts";
import type { MarketHistoryRow } from "@/lib/api";
import { fmtCurrency, fmtNum } from "@/lib/format";

const axis = { stroke: "var(--muted-foreground)", fontSize: 11 };
const grid = "var(--border)";

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: "8px",
  fontSize: "12px",
  color: "var(--popover-foreground)",
};

export function PriceChart({ data }: { data: MarketHistoryRow[] }) {
  return (
    <div className="h-[340px] w-full">
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 10, right: 12, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="px" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--primary)" stopOpacity={0.35} />
              <stop offset="100%" stopColor="var(--primary)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={grid} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="date" {...axis} tickFormatter={(d) => String(d).slice(5)} minTickGap={24} />
          <YAxis {...axis} domain={["auto", "auto"]} tickFormatter={(v) => `$${Number(v).toFixed(0)}`} width={60} />
          <Tooltip
            contentStyle={tooltipStyle}
            labelFormatter={(l) => `Date: ${l}`}
            formatter={(v: any, n) => [fmtCurrency(Number(v)), n]}
          />
          <Area type="monotone" dataKey="adj_close" stroke="var(--primary)" strokeWidth={2} fill="url(#px)" name="Adj Close" />
          <Line type="monotone" dataKey="sma_15" stroke="var(--bull)" strokeWidth={1.2} dot={false} name="SMA 15" />
          <Line type="monotone" dataKey="sma_30" stroke="var(--bear)" strokeWidth={1.2} dot={false} name="SMA 30" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function IndicatorChart({
  data,
  dataKey,
  color,
  refLines,
  formatter = fmtNum,
  height = 140,
}: {
  data: MarketHistoryRow[];
  dataKey: keyof MarketHistoryRow;
  color: string;
  refLines?: number[];
  formatter?: (v: number | null) => string;
  height?: number;
}) {
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={grid} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="date" {...axis} tickFormatter={(d) => String(d).slice(5)} minTickGap={24} />
          <YAxis {...axis} width={48} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => formatter(Number(v))} />
          {refLines?.map((y) => (
            <ReferenceLine key={y} y={y} stroke="var(--border)" strokeDasharray="4 4" />
          ))}
          <Line type="monotone" dataKey={dataKey as string} stroke={color} strokeWidth={1.6} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
