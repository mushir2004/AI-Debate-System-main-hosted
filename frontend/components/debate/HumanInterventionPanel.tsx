"use client";
import { motion, AnimatePresence } from "framer-motion";
import type { SpeakerSide, FallacyDisplay, PersonaKey } from "../../types";
import type { DebatePhase } from "../../hooks/useDebate";

interface HumanInterventionPanelProps {
  phase: DebatePhase; humanText: string; setHumanText: (t: string) => void;
  targetPersona: SpeakerSide; setTargetPersona: (t: SpeakerSide) => void;
  onSubmit: () => void; isInterrupting: boolean; fallacyResult: FallacyDisplay | null;
  proPersona: PersonaKey | null; conPersona: PersonaKey | null;
}

export default function HumanInterventionPanel({ phase, humanText, setHumanText, targetPersona, setTargetPersona, onSubmit, isInterrupting, fallacyResult, proPersona, conPersona }: HumanInterventionPanelProps) {
  return (
    <AnimatePresence>
      {phase === "human_turn" && (
        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
          style={{ background: "#2d1b5e", border: "6px solid #8b5cf6", boxShadow: "6px 6px 0 #8b5cf6", borderRadius: 4, padding: 16, overflow: "hidden" }}>
          <div className="text-center mb-3" style={{ color: "#8b5cf6", fontStyle: "italic", fontSize: 16, letterSpacing: "0.1em" }}>
            🎙️ YOUR TURN, HUMAN
          </div>
          <div className="flex gap-2 mb-3">
            <span style={{ color: "#c4b5fd", fontFamily: "'Arial',sans-serif", fontSize: 12, alignSelf: "center", flexShrink: 0 }}>Attack:</span>
            {(["Pro", "Con"] as SpeakerSide[]).map((side) => {
              const label = side === "Pro" ? proPersona ?? "Pro" : conPersona ?? "Con";
              const active = targetPersona === side;
              return (
                <motion.button key={side} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }} onClick={() => setTargetPersona(side)}
                  style={{ flex: 1, padding: "5px 8px", background: active ? "#8b5cf6" : "#1a0a40", border: `3px solid ${active ? "#8b5cf6" : "#ffffff30"}`, color: active ? "#fff" : "#c4b5fd", fontFamily: "'Impact',sans-serif", fontStyle: "italic", fontSize: 11, cursor: "pointer", letterSpacing: "0.08em" }}>
                  {side}: {label}
                </motion.button>
              );
            })}
          </div>
          <div className="flex gap-2">
            <input type="text" value={humanText} onChange={(e) => setHumanText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !isInterrupting && onSubmit()}
              placeholder="Type your argument here..."
              className="flex-1 px-3 py-2 text-sm"
              style={{ background: "#1a0a40", border: "3px solid #8b5cf6", color: "#e0d8ff", fontFamily: "'Arial',sans-serif", outline: "none" }} />
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={onSubmit} disabled={isInterrupting}
              style={{ background: isInterrupting ? "#555" : "#8b5cf6", border: "3px solid #000", boxShadow: "3px 3px 0 #000", color: "#fff", padding: "8px 16px", fontFamily: "'Impact',sans-serif", fontStyle: "italic", fontSize: 13, cursor: isInterrupting ? "not-allowed" : "pointer" }}>
              {isInterrupting ? "..." : "SPEAK"}
            </motion.button>
          </div>
          <AnimatePresence>
            {fallacyResult && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mt-3 flex flex-col gap-2">
                <div style={{ background: "#1a0a40", border: "3px solid #FF3B3B", boxShadow: "3px 3px 0 #FF3B3B", padding: "8px 12px", borderRadius: 4 }}>
                  <div style={{ color: "#FF3B3B", fontStyle: "italic", fontSize: 12, letterSpacing: "0.1em", marginBottom: 4 }}>⚠️ FALLACY DETECTED: {fallacyResult.fallacy.toUpperCase()}</div>
                  <p style={{ color: "#e0d8ff", fontFamily: "'Arial',sans-serif", fontSize: 11, lineHeight: 1.5 }}>{fallacyResult.explanation}</p>
                </div>
                <div style={{ background: "#1a0a40", border: "3px solid #00ff88", boxShadow: "3px 3px 0 #007a3d", padding: "8px 12px", borderRadius: 4 }}>
                  <div style={{ color: "#00ff88", fontStyle: "italic", fontSize: 12, letterSpacing: "0.1em", marginBottom: 4 }}>💪 STEELMANNED VERSION:</div>
                  <p style={{ color: "#e0d8ff", fontFamily: "'Arial',sans-serif", fontSize: 11, lineHeight: 1.5 }}>{fallacyResult.steelmanned}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}
    </AnimatePresence>
  );
}