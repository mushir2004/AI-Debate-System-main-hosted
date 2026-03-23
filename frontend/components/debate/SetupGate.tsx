"use client";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { PERSONAS, PERSONA_KEYS } from "../../lib/personas";
import type { DebateSetupConfig, PersonaKey } from "../../types";

interface SetupGateProps { setupConfig: DebateSetupConfig; setSetupConfig: (c: DebateSetupConfig) => void; onStart: () => void; isStarting: boolean; statusMsg: string; }

export default function SetupGate({ setupConfig, setSetupConfig, onStart, isStarting, statusMsg }: SetupGateProps) {
  const { topic, proPersona, conPersona } = setupConfig;
  const update = (patch: Partial<DebateSetupConfig>) => setSetupConfig({ ...setupConfig, ...patch });

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
      <div className="flex items-center justify-between w-full mb-8" style={{ maxWidth: 700 }}>
        <motion.div animate={{ rotate: [0, -5, 5, 0] }} transition={{ duration: 1.5, repeat: Infinity }}
          style={{ background: "#FF3B3B", border: "4px solid #000", boxShadow: "4px 4px 0 #000", padding: "4px 14px", fontStyle: "italic", fontSize: 20, color: "#fff", letterSpacing: "0.05em" }}>
          🎙️ DEBATEFORGE
        </motion.div>
        <Link href="/">
          <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
            style={{ background: "#1a0a40", border: "3px solid #FFE000", boxShadow: "3px 3px 0 #FFE000", color: "#FFE000", padding: "5px 14px", fontFamily: "'Impact',sans-serif", fontStyle: "italic", fontSize: 12, cursor: "pointer", letterSpacing: "0.08em" }}>
            ← BACK HOME
          </motion.button>
        </Link>
      </div>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}
        style={{ width: "100%", maxWidth: 700, background: "#1a0a40", border: "6px solid #000", boxShadow: "8px 8px 0 #000", padding: "24px 20px", borderRadius: 4, display: "flex", flexDirection: "column", gap: 20 }}>
        <div className="text-center" style={{ color: "#FFE000", fontStyle: "italic", fontSize: 18, letterSpacing: "0.12em" }}>SET UP YOUR DEBATE</div>
        <div className="flex flex-col gap-2">
          <label style={{ color: "#FFE000", fontStyle: "italic", fontSize: 13, letterSpacing: "0.1em" }}>📋 DEBATE TOPIC</label>
          <input type="text" value={topic} onChange={(e) => update({ topic: e.target.value })} onKeyDown={(e) => e.key === "Enter" && !isStarting && onStart()}
            placeholder="e.g. AI will destroy humanity..."
            style={{ background: "#0d0626", border: "4px solid #FFE000", color: "#e0d8ff", fontFamily: "'Arial',sans-serif", fontSize: 14, padding: "10px 14px", outline: "none", width: "100%" }} />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="flex flex-col gap-2">
            <label style={{ color: "#FFE000", fontStyle: "italic", fontSize: 12, letterSpacing: "0.1em" }}>⚡ PRO SIDE</label>
            {PERSONA_KEYS.map((key) => {
              const p = PERSONAS[key]; const active = proPersona === key; const taken = conPersona === key;
              return (
                <motion.button key={key} whileHover={!taken ? { scale: 1.02 } : {}} whileTap={!taken ? { scale: 0.98 } : {}}
                  onClick={() => !taken && update({ proPersona: key as PersonaKey })} disabled={taken}
                  style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", background: active ? p.nameBg : "#0d0626", border: `3px solid ${active ? p.color : taken ? "#333" : "#ffffff20"}`, color: taken ? "#444" : "#fff", fontFamily: "'Impact',sans-serif", fontStyle: "italic", fontSize: 12, cursor: taken ? "not-allowed" : "pointer", opacity: taken ? 0.4 : 1, transition: "all 0.15s" }}>
                  <span style={{ fontSize: 18 }}>{p.emoji}</span><span>{key}</span>
                  {active && <span style={{ marginLeft: "auto", color: p.color, fontSize: 9 }}>✔</span>}
                </motion.button>
              );
            })}
          </div>
          <div className="flex flex-col gap-2">
            <label style={{ color: "#FF3B3B", fontStyle: "italic", fontSize: 12, letterSpacing: "0.1em" }}>🔥 CON SIDE</label>
            {PERSONA_KEYS.map((key) => {
              const p = PERSONAS[key]; const active = conPersona === key; const taken = proPersona === key;
              return (
                <motion.button key={key} whileHover={!taken ? { scale: 1.02 } : {}} whileTap={!taken ? { scale: 0.98 } : {}}
                  onClick={() => !taken && update({ conPersona: key as PersonaKey })} disabled={taken}
                  style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 10px", background: active ? p.nameBg : "#0d0626", border: `3px solid ${active ? p.color : taken ? "#333" : "#ffffff20"}`, color: taken ? "#444" : "#fff", fontFamily: "'Impact',sans-serif", fontStyle: "italic", fontSize: 12, cursor: taken ? "not-allowed" : "pointer", opacity: taken ? 0.4 : 1, transition: "all 0.15s" }}>
                  <span style={{ fontSize: 18 }}>{p.emoji}</span><span>{key}</span>
                  {active && <span style={{ marginLeft: "auto", color: p.color, fontSize: 9 }}>✔</span>}
                </motion.button>
              );
            })}
          </div>
        </div>
        <AnimatePresence>
          {isStarting && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              style={{ color: "#FFE000", fontStyle: "italic", fontSize: 13, textAlign: "center", letterSpacing: "0.08em" }}>
              {statusMsg}
            </motion.div>
          )}
        </AnimatePresence>
        <motion.button whileHover={{ scale: 1.04, y: -3 }} whileTap={{ scale: 0.96 }} onClick={onStart} disabled={isStarting}
          style={{ background: isStarting ? "#555" : "#FFE000", border: "5px solid #000", boxShadow: isStarting ? "3px 3px 0 #000" : "7px 7px 0 #000", padding: "14px 0", fontSize: 20, fontStyle: "italic", letterSpacing: "0.1em", color: "#000", cursor: isStarting ? "not-allowed" : "pointer", fontFamily: "'Impact','Arial Black',sans-serif", width: "100%", opacity: isStarting ? 0.7 : 1 }}>
          {isStarting ? "⏳ WARMING UP THE ARENA..." : "▶ START THE DEBATE"}
        </motion.button>
      </motion.div>
    </div>
  );
}