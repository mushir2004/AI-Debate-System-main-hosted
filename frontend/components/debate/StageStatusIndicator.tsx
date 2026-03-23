"use client";
import { motion, AnimatePresence } from "framer-motion";
import type { PersonaKey, SpeakerSide } from "../../types";
import type { DebatePhase } from "../../hooks/useDebate";

interface StageStatusIndicatorProps {
  phase: DebatePhase;
  speakingID: SpeakerSide | null;
  proPersona: PersonaKey;
  conPersona: PersonaKey;
}

export default function StageStatusIndicator({
  phase,
  speakingID,
  proPersona,
  conPersona,
}: StageStatusIndicatorProps) {
  const show = phase === "judging" || (phase === "debating" && speakingID === null);

  const thinkingLabel = phase === "judging"
    ? "THE JUDGE IS DELIBERATING..."
    : "AI IS THINKING...";
  const thinkingColor = phase === "judging" ? "#FFE000" : "#c4b5fd";
  const thinkingEmoji = phase === "judging" ? "⚖️" : "🧠";

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          transition={{ duration: 0.3 }}
          style={{
            position: "absolute",
            // VS badge sits at marginBottom:48 from floor(40px) so its center
            // is at bottom: 40+48+32 = 120px. We use top:50% to center in stage.
            top: "50%",
            left: "50%",
            transform: "translateX(-50%) translateY(-50%)",
            zIndex: 5,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 8,
            // Push it up so it sits ABOVE the VS badge, not on it
            marginTop: -60,
          }}
        >
          {/* Bouncing dots */}
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                animate={{
                  y: [0, -8, 0],
                  opacity: [0.4, 1, 0.4],
                  scale: [0.8, 1.2, 0.8],
                }}
                transition={{
                  duration: 0.7,
                  repeat: Infinity,
                  delay: i * 0.18,
                  ease: "easeInOut",
                }}
                style={{
                  width: 10,
                  height: 10,
                  background: thinkingColor,
                  borderRadius: "50%",
                  border: "2px solid #000",
                }}
              />
            ))}
          </div>

          {/* Label */}
          <motion.div
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
            style={{
              background: "#0d0626cc",
              border: `3px solid ${thinkingColor}`,
              boxShadow: `3px 3px 0 #000`,
              padding: "4px 14px",
              color: thinkingColor,
              fontFamily: "'Impact','Arial Black',sans-serif",
              fontStyle: "italic",
              fontSize: 12,
              letterSpacing: "0.12em",
              whiteSpace: "nowrap",
            }}
          >
            {thinkingEmoji} {thinkingLabel}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}