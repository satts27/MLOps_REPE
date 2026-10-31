import { expect, test, type Page } from "@playwright/test";

const history = Array.from({ length: 30 }, (_, index) => ({
  date: `2024-01-${String(index + 1).padStart(2, "0")}`,
  open: 100 + index,
  high: 104 + index,
  low: 99 + index,
  close: 103 + index,
  adj_close: 103 + index,
  volume: 1_000_000 + index,
  macd: 1.2 + index / 100,
  rsi: 55 + index / 10,
  cci: 50 + index,
  adx: 20 + index / 10,
  daily_return: 0.01,
  volatility_15d: 0.12,
  sma_15: 101 + index,
  sma_30: 100 + index,
}));

function mockApis(page: Page, ticker = "AAPL") {
  return page.route("**/*", async (route) => {
      const url = new URL(route.request().url());
      if (!url.pathname.startsWith("/api/")) {
        await route.continue();
        return;
      }
      if (url.pathname.startsWith("/api/market/")) {
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({
            ticker,
            latest: history.at(-1),
            history,
            news: [
              {
                datetime: "2024-01-30",
                headline: `${ticker} reports strong quarterly growth`,
                source: "Market News",
                url: "https://example.com/article",
                summary: "Quarterly growth summary.",
              },
            ],
          }),
        });
      } else if (url.pathname === "/api/dashboard") {
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({
            predictions: [
              {
                Ticker: ticker,
                Signal: "BULLISH",
                "RL Action": 2,
                "Transformer Return": 0.025,
                "Latest Price Date": "2024-01-30",
                "Latest Adj Close": 132,
              },
            ],
            benchmark: [],
            summaries: {},
          }),
        });
      } else if (url.pathname === "/api/predict") {
        await new Promise((resolve) => setTimeout(resolve, 3000));
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({ rows: [], saved_to: "outputs/predictions.csv" }),
        });
      } else {
        await route.continue();
      }
  });
}

test("1 redirects the home page to AAPL", async ({ page }) => {
  await mockApis(page);
  await page.goto("/");
  await expect(page).toHaveURL(/\/ticker\/AAPL$/);
});

test("2 displays the Quant Desk brand", async ({ page }) => {
  await mockApis(page);
  await page.goto("/ticker/AAPL");
  await expect(page.getByText("Quant Desk")).toBeVisible();
  await expect(page.getByText("RL Predictions", { exact: true })).toBeVisible();
});

test("3 shows all ticker navigation options", async ({ page }) => {
  await mockApis(page);
  await page.goto("/ticker/AAPL");
  for (const ticker of ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]) {
    await expect(page.getByRole("link", { name: ticker, exact: true })).toBeVisible();
  }
});

test("4 navigates to a different ticker", async ({ page }) => {
  await mockApis(page, "MSFT");
  await page.goto("/ticker/AAPL");
  await page.getByRole("link", { name: "MSFT", exact: true }).click();
  await expect(page).toHaveURL(/\/ticker\/MSFT$/);
  await expect(page.getByText("MSFT · Adjusted Close")).toBeVisible();
});

test("5 displays the latest adjusted close price", async ({ page }) => {
  await mockApis(page);
  await page.goto("/ticker/AAPL");
  await expect(page.getByText("$132.00").first()).toBeVisible();
});

test("6 displays all market statistic labels", async ({ page }) => {
  await mockApis(page);
  await page.goto("/ticker/AAPL");
  for (const label of ["Open", "High", "Low", "Volume", "Vol 15d"]) {
    await expect(page.getByText(label, { exact: true }).first()).toBeVisible();
  }
});

test("7 displays all four technical indicators", async ({ page }) => {
  await mockApis(page);
  await page.goto("/ticker/AAPL");
  await expect(page.getByText("MACD", { exact: true })).toBeVisible();
  await expect(page.getByText("RSI (14)", { exact: true })).toBeVisible();
  await expect(page.getByText("CCI", { exact: true })).toBeVisible();
  await expect(page.getByText("ADX", { exact: true })).toBeVisible();
});

test("8 displays the generated RL prediction signal", async ({ page }) => {
  await mockApis(page);
  await page.goto("/ticker/AAPL");
  await expect(page.getByText("RL Agent Signal")).toBeVisible();
  await expect(page.getByText("BULLISH", { exact: true })).toBeVisible();
});

test("9 shows the latest news headline and source", async ({ page }) => {
  await mockApis(page);
  await page.goto("/ticker/AAPL");
  await expect(page.getByText("AAPL reports strong quarterly growth")).toBeVisible();
  await expect(page.getByText(/Market News/)).toBeVisible();
});

test("10 the prediction action remains usable", async ({ page }) => {
  await mockApis(page);
  await page.goto("/ticker/AAPL");
  const predictionButton = page.getByRole("button", { name: "Run RL prediction" });
  await expect(predictionButton).toBeEnabled();
  await predictionButton.click();
  await expect(predictionButton).toBeVisible();
});

test("11 shows the empty-news state when news is unavailable", async ({ page }) => {
  await mockApis(page);
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (!url.pathname.startsWith("/api/market/")) {
      await route.continue();
      return;
    }
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ ticker: "AAPL", latest: history.at(-1), history, news: [] }),
    });
  });
  await page.goto("/ticker/AAPL");
  await expect(page.getByText(/No recent news/)).toBeVisible();
});

test("12 keeps the dashboard shell visible when market data fails", async ({ page }) => {
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/api/market/")) {
      await route.fulfill({
        status: 500,
        headers: { "access-control-allow-origin": "*" },
        contentType: "application/json",
        body: JSON.stringify({ error: "unavailable" }),
      });
    } else if (url.pathname === "/api/dashboard") {
      await route.fulfill({ contentType: "application/json", body: JSON.stringify({ predictions: [] }) });
    } else {
      await route.continue();
    }
  });
  await page.goto("/ticker/AAPL");
  await expect(page.getByText("Quant Desk")).toBeVisible();
  await expect(page.getByText("AAPL · Adjusted Close")).toBeVisible();
});
