"use client";
import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import TranscriptEntry from "./TranscriptEntry";
import type { TranscriptLine, SpeakerSide, PersonaKey } from "../../types";

interface TranscriptPanelProps { transcript: TranscriptLine[]; speakingID: SpeakerSide | null; proPersona: PersonaKey | null; conPersona: PersonaKey | null; totalTurns: number; }

export default function TranscriptPanel({ transcript, speakingID, proPersona, conPersona, totalTurns }: TranscriptPanelProps) {
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [transcript]);
  const aiLines  = transcript.filter((t) => t.speaker !== "User").length;
  const progress = totalTurns > 0 ? Math.min(100, (aiLines / Math.max(totalTurns, 1)) * 100) : 0;

  return (
    <div className="w-full lg:w-80 flex flex-col" style={{ background: "#0d0626", border: "6px solid #000", boxShadow: "6px 6px 0 #000", borderRadius: 4, overflow: "hidden", maxHeight: "80vh" }}>
      <div className="px-4 py-2 text-center" style={{ background: "#FF3B3B", borderBottom: "4px solid #000", color: "#fff", fontStyle: "italic", fontSize: 14, letterSpacing: "0.12em" }}>
        📝 DEBATE LOG
      </div>
      <div ref={logRef} className="flex-1 overflow-y-auto p-3 flex flex-col gap-3" style={{ scrollBehavior: "smooth" }}>
        {transcript.length === 0 && (
          <p style={{ color: "#ffffff30", fontFamily: "'Arial',sans-serif", fontSize: 12, textAlign: "center", marginTop: 40, fontStyle: "italic" }}>
            No arguments yet.<br />Press START to ignite chaos.
          </p>
        )}
        {transcript.map((entry) => (
          <TranscriptEntry key={entry.id} entry={entry} isActive={speakingID === entry.speaker && !entry.isHuman} proPersona={proPersona} conPersona={conPersona} />
        ))}
      </div>
      <div className="p-3" style={{ borderTop: "3px solid #ffffff15" }}>
        <div className="text-xs mb-1" style={{ color: "#ffffff50", fontFamily: "'Arial',sans-serif", letterSpacing: "0.1em" }}>DEBATE PROGRESS</div>
        <div style={{ height: 8, background: "#ffffff15", border: "2px solid #000", borderRadius: 2, overflow: "hidden" }}>
          <motion.div animate={{ width: `${progress}%` }} transition={{ duration: 0.5 }} style={{ height: "100%", background: "#FFE000" }} />
        </div>
        <div className="text-right text-xs mt-1" style={{ color: "#ffffff40", fontFamily: "'Arial',sans-serif" }}>{aiLines} turns</div>
      </div>
    </div>
  );
}