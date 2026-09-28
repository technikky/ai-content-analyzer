import "@testing-library/jest-dom/vitest";

import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// Without this, rendered components leak between tests and a query that should find one
// element finds three.
afterEach(() => {
  cleanup();
});
