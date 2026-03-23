"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { PERSONAS, PERSONA_KEYS } from "../../lib/personas";
import { apiGetLeaderboard } from "../../lib/api";
import type { LeaderboardData } from "../../types";

export default function PersonaCards() {
  const [leaderboard, setLeaderboard] = useState<LeaderboardData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiGetLeaderboard().then(setLeaderboard).catch(() => setLeaderboard(null)).finally(() => setLoading(false));
  }, []);

  return (
    <section className="px-6 pb-10">
      <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.4 }} className="text-center mb-8">
        <div style={{ display: "inline-block", background: "#1a0a40", border: "4px solid #FFE000", boxShadow: "5px 5px 0 #FFE000", padding: "6px 24px", color: "#FFE000", fontSize: 18, fontStyle: "italic", letterSpacing: "0.12em" }}>
          THE FIGHTERS
        </div>
      </motion.div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5" style={{ maxWidth: 1100, margin: "0 auto" }}>
        {PERSONA_KEYS.map((key, i) => {
          const persona = PERSONAS[key];
          const stats   = leaderboard?.[key];
          return (
            <motion.div key={key} initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.35, delay: i * 0.08 }} whileHover={{ y: -6, transition: { duration: 0.15 } }}
              style={{ background: "#1a0a40", border: `5px solid ${persona.color}`, boxShadow: `6px 6px 0 ${persona.shadow}`, borderRadius: 4, padding: "18px 16px", display: "flex", flexDirection: "column", gap: 10 }}>
              <div className="flex items-center gap-2">
                <span style={{ fontSize: 32 }}>{persona.emoji}</span>
                <div>
                  <div style={{ color: persona.color, fontSize: 14, fontStyle: "italic", letterSpacing: "0.1em" }}>{key.toUpperCase()}</div>
                  <div style={{ color: "#ffffff60", fontFamily: "'Arial',sans-serif", fontSize: 10, letterSpacing: "0.15em" }}>{persona.tagline}</div>
                </div>
              </div>
              <p style={{ fontFamily: "'Arial',sans-serif", fontSize: 12, color: "#c4b5fd", lineHeight: 1.55, flex: 1 }}>{persona.description}</p>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ background: persona.nameBg, border: "2px solid #000", boxShadow: "2px 2px 0 #000", padding: "2px 10px", color: "#fff", fontStyle: "italic", fontSize: 12, letterSpacing: "0.08em" }}>
                  {loading ? "..." : `🏆 ${stats?.wins ?? 0} WINS`}
                </div>
              </div>
              {!loading && stats?.last_brag && (
                <div style={{ background: "#0d0626", border: `2px solid ${persona.color}40`, padding: "6px 10px", borderRadius: 4 }}>
                  <p style={{ color: "#ffffff60", fontFamily: "'Arial',sans-serif", fontSize: 10, fontStyle: "italic", lineHeight: 1.5 }}>
                    &quot;{stats.last_brag}&quot;
                  </p>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>
    </section>
  );
}