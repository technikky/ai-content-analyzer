/**
 * Turning entity spans into renderable segments.
 *
 * This is the function that makes the backend's character offsets worth having. Rendering
 * highlights by searching for the entity text would highlight the first occurrence of a
 * repeated name rather than the one the extractor actually found, and would break entirely
 * on an entity whose text appears inside another word.
 *
 * Kept as a pure function, separate from the component, so it can be tested on its inputs
 * rather than through the DOM.
 */

import type { Entity } from "@/lib/types";

export interface TextSegment {
  text: string;
  /** Present only on highlighted segments. */
  entity?: Entity;
  start: number;
  end: number;
}

/**
 * Split `text` into consecutive segments, one per entity and one for each gap between them.
 *
 * Entities are sorted by position and any that overlap a segment already emitted are
 * skipped, so the output always tiles the input exactly once. Spans outside the text are
 * dropped rather than clamped: a span that does not fit means the text and the entities came
 * from different requests, and silently clamping it would render a plausible-looking but
 * wrong highlight.
 */
export function segmentText(text: string, entities: readonly Entity[]): TextSegment[] {
  const usable = entities
    .filter((entity) => entity.start >= 0 && entity.end <= text.length && entity.start < entity.end)
    .slice()
    .sort((left, right) => left.start - right.start || right.end - left.end);

  const segments: TextSegment[] = [];
  let cursor = 0;

  for (const entity of usable) {
    if (entity.start < cursor) {
      continue; // overlaps a segment already emitted
    }
    if (entity.start > cursor) {
      segments.push({
        text: text.slice(cursor, entity.start),
        start: cursor,
        end: entity.start,
      });
    }
    segments.push({
      text: text.slice(entity.start, entity.end),
      entity,
      start: entity.start,
      end: entity.end,
    });
    cursor = entity.end;
  }

  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), start: cursor, end: text.length });
  }

  return segments;
}

/** Count of entities by type, in descending order of frequency then alphabetically. */
export function countByType(entities: readonly Entity[]): Array<[string, number]> {
  const counts = new Map<string, number>();
  for (const entity of entities) {
    counts.set(entity.type, (counts.get(entity.type) ?? 0) + 1);
  }
  return [...counts.entries()].sort(
    (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
  );
}

/**
 * Whether an entity's span actually contains the text the backend said it did.
 *
 * A mismatch means the offsets and the text disagree, which is worth surfacing rather than
 * rendering: it would otherwise look like a highlighting bug in the UI when the cause is
 * upstream.
 */
export function spansAreConsistent(text: string, entities: readonly Entity[]): boolean {
  return entities.every((entity) => text.slice(entity.start, entity.end) === entity.text);
}
