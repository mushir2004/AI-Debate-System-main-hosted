"use client";
import { useEffect, useRef } from "react";
import { motion, useAnimation, AnimatePresence } from "framer-motion";
import { PERSONAS } from "../../lib/personas";
import type { PersonaKey } from "../../types";

interface DebaterProps {
  id: "A" | "B";
  isSpeaking: boolean;
  personaKey: PersonaKey;
  side: "left" | "right";
}

export default function Debater({ id, isSpeaking, personaKey, side }: DebaterProps) {
  const mouthControls = useAnimation();
  const bodyControls  = useAnimation();
  const intervalRef   = useRef<ReturnType<typeof setInterval> | null>(null);
  const persona       = PERSONAS[personaKey];
  const isLeft        = side === "left";

  useEffect(() => {
    if (isSpeaking) {
      bodyControls.start({ y: [0, -8, 0, -5, 0], rotate: isLeft ? [0, -1.5, 1, -0.5, 0] : [0, 1.5, -1, 0.5, 0], transition: { duration: 0.6, repeat: Infinity, ease: "easeInOut" } });
      intervalRef.current = setInterval(() => {
        const h = Math.random() * 14 + 4;
        mouthControls.start({ scaleY: h / 8, transition: { duration: 0.07 } });
      }, 90);
    } else {
      bodyControls.stop();
      bodyControls.start({ y: 0, rotate: 0, transition: { duration: 0.3 } });
      if (intervalRef.current) clearInterval(intervalRef.current);
      mouthControls.start({ scaleY: 0.2, transition: { duration: 0.15 } });
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isSpeaking, mouthControls, bodyControls, isLeft]);

  return (
    <div className="flex flex-col items-center gap-2 flex-1 min-w-0">
      <div style={{ background: persona.nameBg, border: "3px solid #000", boxShadow: "3px 3px 0 #000", fontFamily: "'Impact','Arial Black',sans-serif", letterSpacing: "0.12em", color: "#fff", fontSize: 11, padding: "3px 10px", fontStyle: "italic", textAlign: "center" }}>
        {persona.emoji} {personaKey.toUpperCase()}
      </div>
      <motion.div animate={bodyControls} style={{ originY: 1 }}>
        <div className="relative flex flex-col items-center" style={{ width: 110, height: 130, background: persona.color, border: "5px solid #000", boxShadow: "6px 6px 0 #000", borderRadius: 12 }}>
          <div className="flex flex-col items-center pt-3 gap-1 w-full px-3">
            <div className="flex justify-between w-full px-3">
              {[0, 1].map((i) => (
                <div key={i} className="rounded-full" style={{ width: 18, height: 18, background: "#1a0a40", border: "2px solid #000" }}>
                  <div className="rounded-full" style={{ width: 6, height: 6, background: "#fff", margin: "3px auto" }} />
                </div>
              ))}
            </div>
            <div style={{ width: 8, height: 6, background: "#000", borderRadius: "0 0 4px 4px", marginTop: 4 }} />
            <div style={{ width: 42, height: 16, background: "#000", borderRadius: 8, marginTop: 4, display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden" }}>
              <motion.div animate={mouthControls} initial={{ scaleY: 0.2 }} style={{ width: 36, height: 12, background: "#cc0000", borderRadius: 4, transformOrigin: "center" }} />
            </div>
          </div>
          <div className="absolute bottom-0 left-0 right-0" style={{ height: 45, background: "#1a0a40", borderTop: "3px solid #000", borderRadius: "0 0 8px 8px", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <span style={{ color: persona.color, fontFamily: "'Impact',sans-serif", fontSize: 9, fontStyle: "italic", letterSpacing: "0.08em", textShadow: "1px 1px 0 #000" }}>
              {id === "A" ? "PRO" : "CON"}
            </span>
          </div>
          <AnimatePresence>
            {isSpeaking && (
              <motion.div initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0, opacity: 0 }} className="absolute -top-3 -right-3 flex gap-1">
                {[0, 1, 2].map((i) => (
                  <motion.div key={i} animate={{ scaleY: [1, 2, 1] }} transition={{ duration: 0.4, repeat: Infinity, delay: i * 0.13 }}
                    style={{ width: 4, height: 10, background: persona.color, border: "1.5px solid #000", borderRadius: 2, transformOrigin: "bottom" }} />
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
      <div className="flex flex-col items-center">
        <div style={{ width: 24, height: 24, background: "#888", borderRadius: "50% 50% 20% 20%", border: "3px solid #000", boxShadow: "2px 2px 0 #000" }} />
        <div style={{ width: 4, height: 20, background: "#555", border: "1px solid #000" }} />
        <div style={{ width: 40, height: 5, background: "#555", border: "1px solid #000", borderRadius: 3 }} />
      </div>
    </div>
  );
}