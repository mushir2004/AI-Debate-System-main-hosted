"use client";
import { motion } from "framer-motion";

interface Feature {
  icon: string;
  title: string;
  desc: string;
  color: string;
  shadow: string;
}

const FEATURES: Feature[] = [
  { icon: "🤖", title: "TWO AI AGENTS",      desc: "Debater A is a high-pitched firecracker. Debater B is a slow deep-voiced philosopher. Neither backs down.",                                   color: "#FFE000", shadow: "#c9a800" },
  { icon: "🎙️", title: "REAL VOICES",         desc: "Powered by your browser's native Web Speech API. Uses the most natural voices available — no robot monotone, no paid APIs.",              color: "#FF3B3B", shadow: "#a01010" },
  { icon: "🤫", title: "HUMAN INTERVENTION",  desc: 'Hit "SHUT UP!" at any time, grab the mic, say your piece, then hand it back. The AIs will continue as if you never existed.',            color: "#8b5cf6", shadow: "#5b21b6" },
  { icon: "🏆", title: "PING-PONG LOOP",      desc: "Automated turn-taking with realistic pauses. Watch the argument escalate, turn by turn, until someone ends it.",                         color: "#FFE000", shadow: "#c9a800" },
  { icon: "💬", title: "LIVE TRANSCRIPT",     desc: "Every argument is logged in real-time with the active line highlighted as it's being spoken.",                                           color: "#FF3B3B", shadow: "#a01010" },
  { icon: "📺", title: "CARTOON STUDIO",      desc: "Lip-synced characters with mouth jitter animation. South Park-inspired bento-box UI. Designed for maximum chaos energy.",               color: "#8b5cf6", shadow: "#5b21b6" },
];

export default function FeatureCards() {
  return (
    <section className="px-6 pb-16">
      <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.4 }} className="text-center mb-10">
        <div style={{ display: "inline-block", background: "#1a0a40", border: "4px solid #FFE000", boxShadow: "5px 5px 0 #FFE000", padding: "6px 24px", color: "#FFE000", fontSize: 18, fontStyle: "italic", letterSpacing: "0.12em" }}>
          HOW IT WORKS
        </div>
      </motion.div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5" style={{ maxWidth: 1100, margin: "0 auto" }}>
        {FEATURES.map((f, i) => (
          <motion.div key={i} initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.35, delay: i * 0.07 }} whileHover={{ y: -5, transition: { duration: 0.15 } }}
            style={{ background: "#1a0a40", border: `5px solid ${f.color}`, boxShadow: `6px 6px 0 ${f.shadow}`, borderRadius: 4, padding: "20px 22px" }}>
            <div style={{ fontSize: 32, marginBottom: 10 }}>{f.icon}</div>
            <div style={{ color: f.color, fontSize: 15, fontStyle: "italic", letterSpacing: "0.1em", marginBottom: 8 }}>{f.title}</div>
            <p style={{ fontFamily: "'Arial',sans-serif", fontSize: 13, color: "#c4b5fd", lineHeight: 1.6 }}>{f.desc}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}