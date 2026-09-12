import type { NewsItem } from "@/lib/api";
import { ExternalLink } from "lucide-react";

export function NewsList({ items }: { items: NewsItem[] }) {
  if (!items?.length) {
    return (
      <div className="text-sm text-muted-foreground p-4">
        No recent news. (Requires <span className="font-mono">finnhub</span> on the backend.)
      </div>
    );
  }
  return (
    <ul className="divide-y divide-border">
      {items.slice(0, 12).map((n, i) => (
        <li key={i} className="py-3">
          <a
            href={n.url ?? "#"}
            target="_blank"
            rel="noreferrer"
            className="group flex items-start justify-between gap-3"
          >
            <div className="min-w-0">
              <div className="text-sm text-foreground group-hover:text-primary line-clamp-2">
                {n.headline ?? "Untitled"}
              </div>
              <div className="mt-1 text-xs text-muted-foreground font-mono">
                {n.source ?? "—"} · {n.datetime ?? "—"}
              </div>
            </div>
            <ExternalLink className="size-3.5 shrink-0 mt-1 text-muted-foreground group-hover:text-primary" />
          </a>
        </li>
      ))}
    </ul>
  );
}
