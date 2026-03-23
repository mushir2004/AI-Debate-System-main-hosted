"use client";
import { motion } from "framer-motion";

export default function StatusBanner({ statusMsg }: { statusMsg: string }) {
  return (
    <motion.div key={statusMsg} initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.2 }}
      className="text-center py-2 px-4"
      style={{ background: "#1a0a40", border: "4px solid #FFE000", boxShadow: "4px 4px 0 #FFE000", color: "#FFE000", fontStyle: "italic", fontSize: 14, letterSpacing: "0.08em" }}>
      {statusMsg}
    </motion.div>
  );
}