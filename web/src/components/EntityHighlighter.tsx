"use client";

import { useMemo, useState } from "react";

import { countByType, segmentText, spansAreConsistent } from "@/lib/highlight";
import type { Entity, EntityType } from "@/lib/types";

export interface EntityHighlighterProps {
  text: string;
  entities: Entity[];
}

/**
 * Renders the source text with entity spans highlighted, filterable by type.
 *
 * Highlighting is driven by the character offsets the backend returns, not by searching for
 * the entity text, so a name appearing twice highlights the occurrence that was actually
 * extracted.
 */
export function EntityHighlighter({ text, entities }: EntityHighlighterProps) {
  const [hidden, setHidden] = useState<ReadonlySet<EntityType>>(new Set());

  const counts = useMemo(() => countByType(entities), [entities]);
  const visible = useMemo(
    () => entities.filter((entity) => !hidden.has(entity.type)),
    [entities, hidden],
  );
  const segments = useMemo(() => segmentText(text, visible), [text, visible]);
  const consistent = useMemo(() => spansAreConsistent(text, entities), [text, entities]);

  function toggle(type: EntityType) {
    setHidden((current) => {
      const next = new Set(current);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  }

  if (entities.length === 0) {
    return (
      <section aria-labelledby="entities-heading" className="card">
        <h2 id="entities-heading">Entities</h2>
        <p className="muted" data-testid="no-entities">
          No entities found.
        </p>
      </section>
    );
  }

  return (
    <section aria-labelledby="entities-heading" className="card">
      <h2 id="entities-heading">Entities</h2>

      {!consistent && (
        <p className="warning" role="alert" data-testid="span-mismatch">
          Entity offsets do not match the text they were extracted from. Highlighting is
          unreliable for this result.
        </p>
      )}

      <div className="filters">
        {counts.map(([type, count]) => {
          const isHidden = hidden.has(type as EntityType);
          return (
            <button
              key={type}
              type="button"
              className={isHidden ? "chip chip-off" : "chip"}
              aria-pressed={!isHidden}
              onClick={() => toggle(type as EntityType)}
            >
              {type} <span className="count">{count}</span>
            </button>
          );
        })}
      </div>

      <p className="highlighted" data-testid="highlighted-text">
        {segments.map((segment) =>
          segment.entity ? (
            <mark
              key={`${segment.start}-${segment.end}`}
              className={`entity entity-${segment.entity.type.toLowerCase()}`}
              title={`${segment.entity.type} · confidence ${segment.entity.confidence.toFixed(2)}`}
            >
              {segment.text}
            </mark>
          ) : (
            <span key={`${segment.start}-${segment.end}`}>{segment.text}</span>
          ),
        )}
      </p>
    </section>
  );
}
