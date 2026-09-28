import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "AI Content Analyzer",
  description:
    "Summarisation, sentiment, entity extraction and document Q&A, each with a measured baseline.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="site-header">
          <h1>AI Content Analyzer</h1>
          <p className="muted">
            Every analysis has a deterministic local baseline and an optional LLM
            implementation behind the same interface.
          </p>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
