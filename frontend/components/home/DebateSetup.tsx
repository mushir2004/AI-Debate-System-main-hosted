"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { PERSONAS, PERSONA_KEYS } from "../../lib/personas";
import type { PersonaKey, DebateLanguage } from "../../types";

// ── Language options ──────────────────────────────────────────────────────────
const LANGUAGES: { key: DebateLanguage; label: string; flag: string }[] = [
  { key: "english", label: "English",  flag: "🇬🇧" },
  { key: "hindi",   label: "Hindi",    flag: "🇮🇳" },
  { key: "bengali", label: "Bengali",  flag: "🇧🇩" },
  { key: "tamil",   label: "Tamil",    flag: "🌴" },
  { key: "telugu",  label: "Telugu",   flag: "🏺" },
];

export default function DebateSetup() {
  const router = useRouter();
  const [topic,    setTopic]    = useState("");
  const [pro,      setPro]      = useState<PersonaKey>("The Professor");
  const [con,      setCon]      = useState<PersonaKey>("The Aggressor");
  const [language, setLanguage] = useState<DebateLanguage>("english");
  const [error,    setError]    = useState("");

  const handleStart = () => {
    if (!topic.trim()) { setError("⚠️ Enter a debate topic first!"); return; }
    if (pro === con)   { setError("⚠️ Pick two DIFFERENT personas!"); return; }
    setError("");
    sessionStorage.setItem(
      "debateConfig",
      JSON.stringify({ topic: topic.trim(), proPersona: pro, conPersona: con, language })
    );
    router.push("/debate");
  };

  return (
    <section className="px-6 pb-16">
      <div style={{ maxWidth: 800, margin: "0 auto" }}>
        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.4 }} className="text-center mb-8">
          <div style={{ display: "inline-block", background: "#1a0a40", border: "4px solid #FF3B3B", boxShadow: "5px 5px 0 #FF3B3B", padding: "6px 24px", color: "#FF3B3B", fontSize: 18, fontStyle: "italic", letterSpacing: "0.12em" }}>
            SET UP THE DEBATE
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.4, delay: 0.1 }}
          style={{ background: "#1a0a40", border: "6px solid #000", boxShadow: "8px 8px 0 #000", padding: "24px 20px", borderRadius: 4, display: "flex", flexDirection: "column", gap: 20 }}>

          {/* Topic input */}
          <div className="flex flex-col gap-2">
            <label style={{ color: "#FFE000", fontStyle: "italic", fontSize: 14, letterSpacing: "0.12em" }}>📋 DEBATE TOPIC</label>
            <input type="text" value={topic} onChange={(e) => setTopic(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleStart()}
              placeholder="e.g. AI will destroy humanity / Pineapple belongs on pizza..."
              style={{ background: "#0d0626", border: "4px solid #FFE000", color: "#e0d8ff", fontFamily: "'Arial',sans-serif", fontSize: 14, padding: "10px 14px", outline: "none", width: "100%" }} />
          </div>

          {/* Language selector */}
          <div className="flex flex-col gap-2">
            <label style={{ color: "#FFE000", fontStyle: "italic", fontSize: 14, letterSpacing: "0.12em" }}>🌐 DEBATE LANGUAGE</label>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {LANGUAGES.map((lang) => {
                const active = language === lang.key;
                return (
                  <motion.button key={lang.key}
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={() => setLanguage(lang.key)}
                    style={{
                      display:       "flex",
                      alignItems:    "center",
                      gap:           6,
                      padding:       "7px 14px",
                      background:    active ? "#FFE000" : "#0d0626",
                      border:        `3px solid ${active ? "#FFE000" : "#ffffff25"}`,
                      boxShadow:     active ? "3px 3px 0 #c9a800" : "none",
                      color:         active ? "#000" : "#e0d8ff",
                      fontFamily:    "'Impact',sans-serif",
                      fontStyle:     "italic",
                      fontSize:      13,
                      letterSpacing: "0.08em",
                      cursor:        "pointer",
                      transition:    "all 0.15s",
                    }}
                  >
                    <span style={{ fontSize: 16 }}>{lang.flag}</span>
                    <span>{lang.label.toUpperCase()}</span>
                    {active && <span style={{ fontSize: 9, marginLeft: 2 }}>✔</span>}
                  </motion.button>
                );
              })}
            </div>
          </div>

          {/* Persona selectors */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="flex flex-col gap-2">
              <label style={{ color: "#FFE000", fontStyle: "italic", fontSize: 13, letterSpacing: "0.12em" }}>⚡ PRO SIDE</label>
              {PERSONA_KEYS.map((key) => {
                const p = PERSONAS[key]; const active = pro === key; const taken = con === key;
                return (
                  <motion.button key={key} whileHover={!taken ? { scale: 1.02 } : {}} whileTap={!taken ? { scale: 0.98 } : {}}
                    onClick={() => !taken && setPro(key)} disabled={taken}
                    style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 12px", background: active ? p.nameBg : "#0d0626", border: `3px solid ${active ? p.color : taken ? "#333" : "#ffffff20"}`, boxShadow: active ? `3px 3px 0 ${p.shadow}` : "none", color: taken ? "#333" : "#fff", fontFamily: "'Impact',sans-serif", fontStyle: "italic", fontSize: 13, cursor: taken ? "not-allowed" : "pointer", letterSpacing: "0.08em", textAlign: "left", opacity: taken ? 0.4 : 1, transition: "all 0.15s" }}>
                    <span style={{ fontSize: 20 }}>{p.emoji}</span>
                    <span>{key}</span>
                    {active && <span style={{ marginLeft: "auto", color: p.color, fontSize: 10 }}>✔ SELECTED</span>}
                    {taken  && <span style={{ marginLeft: "auto", fontSize: 10 }}>TAKEN</span>}
                  </motion.button>
                );
              })}
            </div>
            <div className="flex flex-col gap-2">
              <label style={{ color: "#FF3B3B", fontStyle: "italic", fontSize: 13, letterSpacing: "0.12em" }}>🔥 CON SIDE</label>
              {PERSONA_KEYS.map((key) => {
                const p = PERSONAS[key]; const active = con === key; const taken = pro === key;
                return (
                  <motion.button key={key} whileHover={!taken ? { scale: 1.02 } : {}} whileTap={!taken ? { scale: 0.98 } : {}}
                    onClick={() => !taken && setCon(key)} disabled={taken}
                    style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 12px", background: active ? p.nameBg : "#0d0626", border: `3px solid ${active ? p.color : taken ? "#333" : "#ffffff20"}`, boxShadow: active ? `3px 3px 0 ${p.shadow}` : "none", color: taken ? "#333" : "#fff", fontFamily: "'Impact',sans-serif", fontStyle: "italic", fontSize: 13, cursor: taken ? "not-allowed" : "pointer", letterSpacing: "0.08em", textAlign: "left", opacity: taken ? 0.4 : 1, transition: "all 0.15s" }}>
                    <span style={{ fontSize: 20 }}>{p.emoji}</span>
                    <span>{key}</span>
                    {active && <span style={{ marginLeft: "auto", color: p.color, fontSize: 10 }}>✔ SELECTED</span>}
                    {taken  && <span style={{ marginLeft: "auto", fontSize: 10 }}>TAKEN</span>}
                  </motion.button>
                );
              })}
            </div>
          </div>

          {/* Matchup preview */}
          <div style={{ background: "#0d0626", border: "3px solid #ffffff15", padding: "10px 14px", borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", gap: 16 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 28 }}>{PERSONAS[pro].emoji}</div>
              <div style={{ color: PERSONAS[pro].color, fontSize: 11, fontStyle: "italic", letterSpacing: "0.1em" }}>{pro}</div>
              <div style={{ color: "#ffffff50", fontFamily: "'Arial',sans-serif", fontSize: 10 }}>PRO</div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <div style={{ color: "#FF3B3B", fontSize: 22, fontStyle: "italic" }}>VS</div>
              <div style={{ color: "#FFE000", fontFamily: "'Arial',sans-serif", fontSize: 10, letterSpacing: "0.1em" }}>
                {LANGUAGES.find(l => l.key === language)?.flag} {LANGUAGES.find(l => l.key === language)?.label.toUpperCase()}
              </div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 28 }}>{PERSONAS[con].emoji}</div>
              <div style={{ color: PERSONAS[con].color, fontSize: 11, fontStyle: "italic", letterSpacing: "0.1em" }}>{con}</div>
              <div style={{ color: "#ffffff50", fontFamily: "'Arial',sans-serif", fontSize: 10 }}>CON</div>
            </div>
          </div>

          <AnimatePresence>
            {error && (
              <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                style={{ color: "#FF3B3B", fontFamily: "'Arial',sans-serif", fontSize: 13, textAlign: "center" }}>
                {error}
              </motion.div>
            )}
          </AnimatePresence>

          <motion.button whileHover={{ scale: 1.04, y: -3 }} whileTap={{ scale: 0.96 }} onClick={handleStart}
            style={{ background: "#FFE000", border: "5px solid #000", boxShadow: "7px 7px 0 #000", padding: "14px 0", fontSize: 20, fontStyle: "italic", letterSpacing: "0.1em", color: "#000", cursor: "pointer", fontFamily: "'Impact','Arial Black',sans-serif", width: "100%" }}>
            ▶ START THE DEBATE
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
}