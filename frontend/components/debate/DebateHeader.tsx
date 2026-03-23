"use client";
import { motion } from "framer-motion";

interface DebateHeaderProps { isRunning: boolean; round: number; topic: string; }

export default function DebateHeader({ isRunning, round, topic }: DebateHeaderProps) {
  return (
    <header className="flex items-center justify-between px-6 py-3" style={{ background: "#FFE000", borderBottom: "6px solid #000", boxShadow: "0 6px 0 #000" }}>
      <div className="flex items-center gap-3">
        <motion.div animate={{ rotate: [0, -5, 5, 0] }} transition={{ duration: 1.5, repeat: Infinity }}
          style={{ background: "#FF3B3B", border: "4px solid #000", boxShadow: "4px 4px 0 #000", padding: "4px 12px", fontStyle: "italic", fontSize: 22, color: "#fff", letterSpacing: "0.05em" }}>
          🎙️ DEBATEFORGE
        </motion.div>
        <div className="flex flex-col">
          <span style={{ fontSize: 11, fontFamily: "'Arial',sans-serif", fontWeight: "bold", color: "#1a0a40", letterSpacing: "0.2em", textTransform: "uppercase" }}>
            ROUND {round > 0 ? round : "—"}
          </span>
          {topic && <span style={{ fontSize: 10, fontFamily: "'Arial',sans-serif", color: "#1a0a40", letterSpacing: "0.1em", maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>📋 {topic}</span>}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <motion.div animate={isRunning ? { opacity: [1, 0, 1] } : { opacity: 0.3 }} transition={{ duration: 0.8, repeat: Infinity }}
          style={{ width: 12, height: 12, background: "#FF3B3B", borderRadius: "50%", border: "2px solid #000" }} />
        <span style={{ fontSize: 13, color: isRunning ? "#FF3B3B" : "#888", fontStyle: "italic", letterSpacing: "0.15em" }}>
          {isRunning ? "● ON AIR" : "○ OFF AIR"}
        </span>
      </div>
    </header>
  );
}