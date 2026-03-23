"use client";
import { motion, AnimatePresence } from "framer-motion";
import { PERSONAS } from "../../lib/personas";
import type { JudgeResponse } from "../../types";

interface JudgeResultPanelProps { judgeResult: JudgeResponse | null; isJudging: boolean; }

export default function JudgeResultPanel({ judgeResult, isJudging }: JudgeResultPanelProps) {
  return (
    <AnimatePresence>
      {(isJudging || judgeResult) && (
        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
          style={{ background: "#1a0a40", border: "6px solid #FFE000", boxShadow: "6px 6px 0 #FFE000", borderRadius: 4, padding: 16, overflow: "hidden" }}>
          {isJudging && !judgeResult && (
            <div className="text-center" style={{ color: "#FFE000", fontStyle: "italic", fontSize: 16, letterSpacing: "0.1em" }}>⚖️ THE JUDGE IS DELIBERATING...</div>
          )}
          {judgeResult && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col gap-3">
              <div className="text-center">
                {(() => {
                  const p = PERSONAS[judgeResult.winning_character];
                  return (
                    <div style={{ background: p.nameBg, border: "4px solid #000", boxShadow: "5px 5px 0 #000", padding: "8px 20px", display: "inline-block" }}>
                      <span style={{ color: "#fff", fontStyle: "italic", fontSize: 20, letterSpacing: "0.08em" }}>🏆 {p.emoji} {judgeResult.winning_character.toUpperCase()} WINS!</span>
                    </div>
                  );
                })()}
              </div>
              <div style={{ background: "#0d0626", border: "3px solid #FFE000", padding: "8px 12px", borderRadius: 4 }}>
                <div style={{ color: "#FFE000", fontSize: 11, fontStyle: "italic", letterSpacing: "0.1em", marginBottom: 4 }}>VERDICT:</div>
                <p style={{ color: "#e0d8ff", fontFamily: "'Arial',sans-serif", fontSize: 12, lineHeight: 1.5 }}>{judgeResult.verdict.reason}</p>
              </div>
              <div style={{ background: "#0d0626", border: "3px solid #8b5cf6", boxShadow: "3px 3px 0 #5b21b6", padding: "8px 12px", borderRadius: 4 }}>
                <div style={{ color: "#8b5cf6", fontSize: 11, fontStyle: "italic", letterSpacing: "0.1em", marginBottom: 4 }}>WINNER&apos;S BRAG:</div>
                <p style={{ color: "#c4b5fd", fontFamily: "'Arial',sans-serif", fontSize: 12, lineHeight: 1.5, fontStyle: "italic" }}>&quot;{judgeResult.verdict.brag}&quot;</p>
              </div>
            </motion.div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}