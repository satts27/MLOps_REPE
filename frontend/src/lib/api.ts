export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:5000";

export const TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] as const;
export type Ticker = (typeof TICKERS)[number];

export interface MarketHistoryRow {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  adj_close: number | null;
  volume: number | null;
  macd: number | null;
  rsi: number | null;
  cci: number | null;
  adx: number | null;
  daily_return: number | null;
  volatility_15d: number | null;
  sma_15: number | null;
  sma_30: number | null;
}

export interface MarketResponse {
  ticker: string;
  latest: MarketHistoryRow;
  history: MarketHistoryRow[];
  news: NewsItem[];
}

export interface NewsItem {
  datetime: string | null;
  headline: string | null;
  source: string | null;
  url: string | null;
  summary: string | null;
}

export interface PredictionRow {
  Ticker: string;
  Signal: string;
  "RL Action": number | null;
  "Transformer Return": number | null;
  "Latest Price Date": string | null;
  "Latest Adj Close": number | null;
  Algorithm?: string;
  Error?: string;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const fetchMarket = (ticker: string, lookback = 180) =>
  get<MarketResponse>(`/api/market/${ticker}?lookback_days=${lookback}&include_news=true&news_days=10`);

export const fetchDashboard = () =>
  get<{ predictions: PredictionRow[]; benchmark: any[]; summaries: Record<string, any[]> }>(
    "/api/dashboard"
  );

export const runPredict = async (ticker?: string) => {
  const url = `${API_BASE}/api/predict${ticker ? `?ticker=${ticker}` : ""}`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<{ rows: PredictionRow[]; saved_to: string }>;
};
