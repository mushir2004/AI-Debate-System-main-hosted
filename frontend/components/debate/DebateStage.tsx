"use client";
import { motion } from "framer-motion";
import Debater from "./Debater";
import SpeechBubble from "./SpeechBubble";
import type { PersonaKey, SpeakerSide, TranscriptLine } from "../../types";

interface DebateStageProps { speakingID: SpeakerSide | null; proPersona: PersonaKey; conPersona: PersonaKey; transcript: TranscriptLine[]; }

export default function DebateStage({ speakingID, proPersona, conPersona, transcript }: DebateStageProps) {
  const lastProText = [...transcript].reverse().find((t) => t.speaker === "Pro")?.text ?? null;
  const lastConText = [...transcript].reverse().find((t) => t.speaker === "Con")?.text ?? null;
  return (
    <div className="relative flex items-end justify-around pb-8 pt-12"
      style={{ background: "#120436", border: "6px solid #000", boxShadow: "8px 8px 0 #000", borderRadius: 4, minHeight: 300, overflow: "hidden" }}>
      <div className="absolute bottom-0 left-0 right-0" style={{ height: 40, background: "#0a0220", borderTop: "4px solid #FFE000" }} />
      {[...Array(6)].map((_, i) => (
        <div key={i} className="absolute top-0 bottom-0" style={{ left: `${i * 17}%`, width: "8%", background: i % 2 === 0 ? "#ffffff05" : "transparent" }} />
      ))}
      <SpeechBubble text={lastProText} side="left"  active={speakingID === "Pro"} />
      <SpeechBubble text={lastConText} side="right" active={speakingID === "Con"} />
      <Debater id="A" isSpeaking={speakingID === "Pro"} personaKey={proPersona} side="left" />
      <motion.div animate={{ scale: [1, 1.08, 1], rotate: [0, -3, 3, 0] }} transition={{ duration: 2, repeat: Infinity }}
        style={{ background: "#FF3B3B", border: "5px solid #000", boxShadow: "5px 5px 0 #000", borderRadius: "50%", width: 64, height: 64, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginBottom: 48, zIndex: 2 }}>
        <span style={{ color: "#fff", fontSize: 20, fontStyle: "italic" }}>VS</span>
      </motion.div>
      <Debater id="B" isSpeaking={speakingID === "Con"} personaKey={conPersona} side="right" />
    </div>
  );
}