"use client";
import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef } from "react";

// ─────────────────────────────────────────────
//  TUNE THIS — pixels per second to scroll
//  15  = slow, comfortable read
//  40  = medium
//  60  = fast
// ─────────────────────────────────────────────
const SCROLL_SPEED_PX_PER_SEC = 8;

interface SpeechBubbleProps { text: string | null; side: "left" | "right"; active: boolean; }

function ScrollingText({ text }: { text: string }) {
  const containerRef   = useRef<HTMLDivElement>(null);
  const rafRef         = useRef<number | null>(null);
  const startTimeRef   = useRef<number | null>(null);
  const startScrollRef = useRef(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Reset to top on new text
    el.scrollTop         = 0;
    startTimeRef.current = null;
    startScrollRef.current = 0;

    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    const step = (timestamp: number) => {
      if (!el) return;
      const maxScroll = el.scrollHeight - el.clientHeight;
      if (maxScroll <= 0) return; // fits without scrolling

      if (startTimeRef.current === null) {
        startTimeRef.current   = timestamp;
        startScrollRef.current = el.scrollTop;
      }

      const elapsed  = (timestamp - startTimeRef.current) / 1000; // → seconds
      const target   = startScrollRef.current + elapsed * SCROLL_SPEED_PX_PER_SEC;
      el.scrollTop   = Math.min(target, maxScroll);

      if (el.scrollTop < maxScroll) {
        rafRef.current = requestAnimationFrame(step);
      }
    };

    rafRef.current = requestAnimationFrame(step);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [text]);

  return (
    <div
      ref={containerRef}
      style={{
        maxHeight: 90,        // ~4-5 lines — adjust freely
        overflowY: "hidden",  // scrollbar hidden; JS drives it
        scrollbarWidth: "none",
      }}
    >
      {text}
    </div>
  );
}

export default function SpeechBubble({ text, side, active }: SpeechBubbleProps) {
  const isLeft = side === "left";
  return (
    <AnimatePresence mode="wait">
      {active && text && (
        <motion.div
          key={text}
          initial={{ opacity: 0, y: 10, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -8, scale: 0.95 }}
          transition={{ duration: 0.25 }}
          className="absolute z-10"
          style={{
            top: "10%",
            ...(isLeft ? { left: "calc(50% - 20px)" } : { right: "calc(50% - 20px)" }),
            maxWidth: 200,
            background: "#fff",
            border: "4px solid #000",
            boxShadow: "4px 4px 0 #000",
            borderRadius: 10,
            padding: "10px 14px",
            fontFamily: "'Impact','Arial Black',sans-serif",
            fontStyle: "italic",
            fontSize: 13,
            lineHeight: 1.4,
            color: "#1a0a40",
          }}
        >
          <ScrollingText text={text} />

          {/* Outer tail (border colour) */}
          <div style={{
            position: "absolute",
            top: 14,
            ...(isLeft
              ? { left: -14, borderTop: "9px solid transparent", borderBottom: "9px solid transparent", borderRight: "14px solid #000" }
              : { right: -14, borderTop: "9px solid transparent", borderBottom: "9px solid transparent", borderLeft: "14px solid #000" }),
            width: 0,
            height: 0,
          }} />

          {/* Inner tail (fill colour) */}
          <div style={{
            position: "absolute",
            top: 16,
            ...(isLeft
              ? { left: -9, borderTop: "7px solid transparent", borderBottom: "7px solid transparent", borderRight: "10px solid #fff" }
              : { right: -9, borderTop: "7px solid transparent", borderBottom: "7px solid transparent", borderLeft: "10px solid #fff" }),
            width: 0,
            height: 0,
          }} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}