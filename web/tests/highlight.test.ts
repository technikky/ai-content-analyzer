import { describe, expect, it } from "vitest";

import { countByType, segmentText, spansAreConsistent } from "@/lib/highlight";
import type { Entity } from "@/lib/types";

function entity(text: string, start: number, type: Entity["type"] = "PERSON"): Entity {
  return { text, type, start, end: start + text.length, confidence: 1 };
}

describe("segmentText", () => {
  it("tiles the input exactly once", () => {
    const text = "Ada met Grace in Paris.";
    const segments = segmentText(text, [entity("Ada", 0), entity("Grace", 8)]);

    expect(segments.map((segment) => segment.text).join("")).toBe(text);
  });

  it("marks entity segments and leaves the gaps plain", () => {
    const text = "Ada met Grace.";
    const segments = segmentText(text, [entity("Ada", 0), entity("Grace", 8)]);

    expect(segments.map((segment) => [segment.text, Boolean(segment.entity)])).toEqual([
      ["Ada", true],
      [" met ", false],
      ["Grace", true],
      [".", false],
    ]);
  });

  it("highlights the occurrence the offsets point at, not the first match", () => {
    // The whole reason offsets are used instead of a text search.
    const text = "Acme sued Acme.";
    const segments = segmentText(text, [entity("Acme", 10, "ORG")]);
    const marked = segments.filter((segment) => segment.entity);

    expect(marked).toHaveLength(1);
    expect(marked[0]?.start).toBe(10);
  });

  it("returns a single plain segment when there are no entities", () => {
    expect(segmentText("plain text", [])).toEqual([
      { text: "plain text", start: 0, end: 10 },
    ]);
  });

  it("skips an entity overlapping one already emitted", () => {
    const text = "Ada Lovelace wrote it.";
    const segments = segmentText(text, [entity("Ada Lovelace", 0), entity("Lovelace", 4)]);
    const marked = segments.filter((segment) => segment.entity);

    expect(marked.map((segment) => segment.text)).toEqual(["Ada Lovelace"]);
    expect(segments.map((segment) => segment.text).join("")).toBe(text);
  });

  it("prefers the longer entity when two start at the same offset", () => {
    const text = "Acme Corp filed.";
    const segments = segmentText(text, [entity("Acme", 0, "ORG"), entity("Acme Corp", 0, "ORG")]);
    const marked = segments.filter((segment) => segment.entity);

    expect(marked.map((segment) => segment.text)).toEqual(["Acme Corp"]);
  });

  it("accepts entities given out of order", () => {
    const text = "Ada met Grace.";
    const segments = segmentText(text, [entity("Grace", 8), entity("Ada", 0)]);

    expect(segments.filter((segment) => segment.entity).map((segment) => segment.text)).toEqual([
      "Ada",
      "Grace",
    ]);
  });

  it("drops a span that runs past the end of the text", () => {
    // Clamping would render a plausible but wrong highlight; the text and the entities
    // clearly came from different requests.
    const segments = segmentText("short", [
      { text: "short text", type: "ORG", start: 0, end: 40, confidence: 1 },
    ]);

    expect(segments).toEqual([{ text: "short", start: 0, end: 5 }]);
  });

  it("drops an empty or inverted span", () => {
    const segments = segmentText("some text", [
      { text: "", type: "ORG", start: 3, end: 3, confidence: 1 },
      { text: "x", type: "ORG", start: 5, end: 2, confidence: 1 },
    ]);

    expect(segments).toEqual([{ text: "some text", start: 0, end: 9 }]);
  });

  it("handles an entity at the very start and end", () => {
    const text = "Ada";
    const segments = segmentText(text, [entity("Ada", 0)]);

    expect(segments).toHaveLength(1);
    expect(segments[0]?.entity).toBeDefined();
  });
});

describe("countByType", () => {
  it("counts by type, most frequent first", () => {
    const counts = countByType([
      entity("Ada", 0),
      entity("Grace", 8),
      entity("Acme", 20, "ORG"),
    ]);

    expect(counts).toEqual([
      ["PERSON", 2],
      ["ORG", 1],
    ]);
  });

  it("breaks ties alphabetically so the order is stable", () => {
    const counts = countByType([entity("Acme", 0, "ORG"), entity("Paris", 10, "LOCATION")]);

    expect(counts.map(([type]) => type)).toEqual(["LOCATION", "ORG"]);
  });

  it("returns nothing for no entities", () => {
    expect(countByType([])).toEqual([]);
  });
});

describe("spansAreConsistent", () => {
  it("accepts spans that match the text", () => {
    expect(spansAreConsistent("Ada met Grace.", [entity("Ada", 0), entity("Grace", 8)])).toBe(true);
  });

  it("rejects a span whose text disagrees with the offsets", () => {
    expect(spansAreConsistent("Ada met Grace.", [entity("Ada", 4)])).toBe(false);
  });

  it("is trivially true with no entities", () => {
    expect(spansAreConsistent("anything", [])).toBe(true);
  });
});
