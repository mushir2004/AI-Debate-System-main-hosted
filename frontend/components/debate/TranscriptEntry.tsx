"use client";
import { motion } from "framer-motion";
import { PERSONAS } from "../../lib/personas";
import type { TranscriptLine, PersonaKey } from "../../types";

interface TranscriptEntryProps { entry: TranscriptLine; isActive: boolean; proPersona: PersonaKey | null; conPersona: PersonaKey | null; }

export default function TranscriptEntry({ entry, isActive, proPersona, conPersona }: TranscriptEntryProps) {
  const isHuman = entry.speaker === "User";
  const isPro   = entry.speaker === "Pro";
  const personaKey = isPro ? proPersona : entry.speaker === "Con" ? conPersona : null;
  const persona    = personaKey ? PERSONAS[personaKey] : null;
  const color      = persona?.color ?? "#8b5cf6";

  if (isHuman) {
    return (
      <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="text-center">
        <div className="inline-block px-3 py-2 text-xs"
          style={{ background: "#2d1b5e", border: "3px solid #8b5cf6", boxShadow: "3px 3px 0 #8b5cf6", color: "#c4b5fd", fontFamily: "'Arial',sans-serif", borderRadius: 4, maxWidth: 260 }}>
          {entry.isSteelmanned
            ? <><span style={{ color: "#00ff88", fontStyle: "italic" }}>💪 Steelmanned: </span>{entry.text}</>
            : <><span style={{ fontStyle: "italic" }}>🗣️ You: </span>{entry.text}</>}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div initial={{ opacity: 0, x: isPro ? -20 : 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.3 }} className={`flex gap-2 ${isPro ? "flex-row" : "flex-row-reverse"}`}>
      <div style={{ width: 28, height: 28, background: color, border: "3px solid #000", borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Impact',sans-serif", fontSize: 11, fontWeight: "bold", color: "#000" }}>
        {persona?.emoji ?? (isPro ? "P" : "C")}
      </div>
      <div className={`max-w-xs px-3 py-2 text-xs leading-relaxed ${isActive ? "animate-pulse" : ""}`}
        style={{ background: isActive ? `${color}33` : "#ffffff15", border: `3px solid ${isActive ? color : "#ffffff30"}`, boxShadow: isActive ? `3px 3px 0 ${color}` : "none", color: "#e0d8ff", fontFamily: "'Arial',sans-serif", borderRadius: 6, transition: "all 0.3s" }}>
        {entry.text}
      </div>
    </motion.div>
  );
}