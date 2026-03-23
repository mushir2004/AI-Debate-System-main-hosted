"use client";
import { motion } from "framer-motion";
import Link from "next/link";

interface MiniDebaterProps {
  color: "yellow" | "red";
  label: string;
  flip: boolean;
}

function MiniDebater({ color, label, flip }: MiniDebaterProps) {
  const bg     = color === "yellow" ? "#FFE000" : "#FF3B3B";
  const accent = color === "yellow" ? "#FFE000" : "#FF3B3B";
  const nameBg = color === "yellow" ? "#c9a800" : "#c41a1a";
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
      <div style={{ background: nameBg, border: "3px solid #000", boxShadow: "3px 3px 0 #000", padding: "3px 10px", color: "#fff", fontSize: 11, fontStyle: "italic", letterSpacing: "0.15em" }}>{label}</div>
      <motion.div animate={{ y: [0, -6, 0] }} transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut", delay: flip ? 0.9 : 0 }}
        style={{ width: 90, height: 110, background: bg, border: "5px solid #000", boxShadow: "5px 5px 0 #000", borderRadius: 10, display: "flex", flexDirection: "column", alignItems: "center", position: "relative" }}>
        <div style={{ display: "flex", justifyContent: "space-between", width: "80%", marginTop: 12 }}>
          {[0, 1].map((i) => (
            <div key={i} style={{ width: 15, height: 15, background: "#1a0a40", border: "2px solid #000", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <div style={{ width: 5, height: 5, background: "#fff", borderRadius: "50%" }} />
            </div>
          ))}
        </div>
        <div style={{ width: 6, height: 5, background: "#000", borderRadius: "0 0 3px 3px", marginTop: 4 }} />
        <div style={{ width: 36, height: 14, background: "#000", borderRadius: 7, marginTop: 5, display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden" }}>
          <div style={{ width: 30, height: 10, background: "#cc0000", borderRadius: 3 }} />
        </div>
        <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 38, background: "#1a0a40", borderTop: "3px solid #000", borderRadius: "0 0 6px 6px", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span style={{ color: accent, fontSize: 9, fontStyle: "italic" }}>{color === "yellow" ? "TEAM HOT" : "TEAM NOT"}</span>
        </div>
        <div style={{ position: "absolute", top: -10, right: -12, display: "flex", gap: 3 }}>
          {[0, 1, 2].map((i) => (
            <motion.div key={i} animate={{ scaleY: [1, 2.5, 1] }} transition={{ duration: 0.4, repeat: Infinity, delay: i * 0.13 }}
              style={{ width: 4, height: 10, background: accent, border: "1.5px solid #000", borderRadius: 2, transformOrigin: "bottom" }} />
          ))}
        </div>
      </motion.div>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
        <div style={{ width: 18, height: 18, background: "#888", borderRadius: "50% 50% 20% 20%", border: "2px solid #000" }} />
        <div style={{ width: 3, height: 14, background: "#555" }} />
        <div style={{ width: 28, height: 4, background: "#555", borderRadius: 2 }} />
      </div>
    </div>
  );
}

const SAMPLE_LINES = [
  { id: "A" as const, text: "Pineapple on pizza is OBJECTIVELY correct!", color: "#FFE000" },
  { id: "B" as const, text: "That is the most CATASTROPHICALLY wrong take ever uttered.", color: "#FF3B3B" },
];

export default function DebatePreview() {
  return (
    <section className="px-6 pb-20">
      <div style={{ maxWidth: 700, margin: "0 auto" }}>
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.4 }}
          style={{ background: "#120436", border: "6px solid #000", boxShadow: "8px 8px 0 #000", borderRadius: 4, padding: "32px 24px", textAlign: "center" }}>
          <div style={{ background: "#FF3B3B", border: "3px solid #000", boxShadow: "3px 3px 0 #000", display: "inline-block", padding: "4px 16px", color: "#fff", fontSize: 12, fontStyle: "italic", letterSpacing: "0.15em", marginBottom: 24 }}>
            TONIGHT&apos;S DEBATE: PINEAPPLE ON PIZZA
          </div>
          <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "center", gap: 32, position: "relative", paddingBottom: 20, borderBottom: "4px solid #FFE000" }}>
            <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 28, background: "#0a0220" }} />
            <MiniDebater color="yellow" label="DEBATER A" flip={false} />
            <motion.div animate={{ scale: [1, 1.1, 1], rotate: [0, -4, 4, 0] }} transition={{ duration: 2, repeat: Infinity }}
              style={{ width: 52, height: 52, background: "#FF3B3B", border: "4px solid #000", boxShadow: "4px 4px 0 #000", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginBottom: 40, zIndex: 2 }}>
              <span style={{ color: "#fff", fontSize: 16, fontStyle: "italic" }}>VS</span>
            </motion.div>
            <MiniDebater color="red" label="DEBATER B" flip={true} />
          </div>
          <div style={{ marginTop: 20, display: "flex", flexDirection: "column", gap: 10 }}>
            {SAMPLE_LINES.map((line, i) => (
              <motion.div key={line.id} initial={{ opacity: 0, x: i === 0 ? -20 : 20 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: 0.2 + i * 0.15 }}
                style={{ display: "flex", gap: 8, flexDirection: i === 0 ? "row" : "row-reverse", alignItems: "center" }}>
                <div style={{ width: 24, height: 24, background: line.color, border: "2px solid #000", borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Impact',sans-serif", fontSize: 11, fontWeight: "bold", color: "#000" }}>{line.id}</div>
                <div style={{ background: "#ffffff10", border: `2px solid ${line.color}40`, padding: "6px 12px", fontFamily: "'Arial',sans-serif", fontSize: 12, color: "#e0d8ff", borderRadius: 5, textAlign: "left", maxWidth: 280 }}>{line.text}</div>
              </motion.div>
            ))}
            <p style={{ fontFamily: "'Arial',sans-serif", fontSize: 11, color: "#ffffff30", fontStyle: "italic", marginTop: 4 }}>...and it only gets worse from here.</p>
          </div>
          <div style={{ marginTop: 24 }}>
            <Link href="/debate">
              <motion.button whileHover={{ scale: 1.05, y: -3 }} whileTap={{ scale: 0.96 }}
                style={{ background: "#FFE000", border: "4px solid #000", boxShadow: "6px 6px 0 #000", padding: "12px 36px", fontSize: 16, fontStyle: "italic", letterSpacing: "0.08em", color: "#000", cursor: "pointer", fontFamily: "'Impact','Arial Black',sans-serif" }}>
                WATCH THE FULL DEBATE →
              </motion.button>
            </Link>
          </div>
        </motion.div>
      </div>
    </section>
  );
}