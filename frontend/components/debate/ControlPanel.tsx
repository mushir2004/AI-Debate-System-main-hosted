"use client";
import { motion } from "framer-motion";
import type { DebatePhase } from "../../hooks/useDebate";

interface ControlPanelProps { phase: DebatePhase; isRunning: boolean; isJudging: boolean; onShutUp: () => void; onResume: () => void; onJudge: () => void; }

export default function ControlPanel({ phase, isRunning, isJudging, onShutUp, onResume, onJudge }: ControlPanelProps) {
  const humanMode  = phase === "human_turn";
  const debateOver = phase === "finished";
  const canResume  = humanMode && !debateOver;
  const canShutUp  = isRunning && phase === "debating";
  const canJudge   = !isRunning && !isJudging && phase !== "setup" && phase !== "finished";

  return (
    <div className="grid grid-cols-3 gap-3" style={{ background: "#1a0a40", border: "6px solid #000", boxShadow: "6px 6px 0 #000", padding: 16, borderRadius: 4 }}>
      <motion.button whileHover={{ scale: 1.06, y: -3 }} whileTap={{ scale: 0.94, rotate: -2 }} onClick={onShutUp} disabled={!canShutUp}
        style={{ padding: "12px 6px", background: !canShutUp ? "#555" : "#FF3B3B", border: "4px solid #000", boxShadow: !canShutUp ? "3px 3px 0 #000" : "6px 6px 0 #000", color: "#fff", fontStyle: "italic", fontSize: 15, cursor: !canShutUp ? "not-allowed" : "pointer", fontFamily: "'Impact',sans-serif", opacity: !canShutUp ? 0.5 : 1 }}>
        🤫 SHUT UP!
      </motion.button>
      <motion.button whileHover={{ scale: 1.04, y: -2 }} whileTap={{ scale: 0.96 }} onClick={onResume} disabled={!canResume}
        style={{ padding: "12px 6px", background: canResume ? "#8b5cf6" : "#555", border: "4px solid #000", boxShadow: canResume ? "5px 5px 0 #000" : "3px 3px 0 #000", color: "#fff", fontStyle: "italic", fontSize: 14, cursor: canResume ? "pointer" : "not-allowed", fontFamily: "'Impact',sans-serif", opacity: canResume ? 1 : 0.5 }}>
        ⏩ RESUME AIs
      </motion.button>
      <motion.button whileHover={{ scale: 1.04, y: -2 }} whileTap={{ scale: 0.96 }} onClick={onJudge} disabled={!canJudge || isJudging}
        style={{ padding: "12px 6px", background: isJudging ? "#555" : canJudge ? "#FFE000" : "#555", border: "4px solid #000", boxShadow: canJudge && !isJudging ? "5px 5px 0 #000" : "3px 3px 0 #000", color: "#000", fontStyle: "italic", fontSize: 13, cursor: canJudge && !isJudging ? "pointer" : "not-allowed", fontFamily: "'Impact',sans-serif", opacity: canJudge && !isJudging ? 1 : 0.5 }}>
        {isJudging ? "⏳ JUDGING..." : debateOver ? "✅ JUDGED" : "⚖️ END & JUDGE"}
      </motion.button>
    </div>
  );
}