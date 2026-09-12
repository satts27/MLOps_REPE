import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  beforeLoad: () => {
    throw redirect({ to: "/ticker/$ticker", params: { ticker: "AAPL" } });
  },
  component: () => null,
});
