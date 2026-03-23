"use client";
import HomeNav      from "../components/home/HomeNav";
import HeroSection  from "../components/home/HeroSection";
import PersonaCards from "../components/home/PersonaCards";
import DebateSetup  from "../components/home/DebateSetup";
import HomeFooter   from "../components/home/HomeFooter";

export default function HomePage() {
  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", background: "linear-gradient(160deg, #0d0626 0%, #1a0a40 55%, #0d0626 100%)", fontFamily: "'Impact','Arial Black',sans-serif" }}>
      <HomeNav />
      <main style={{ flex: 1 }}>
        <HeroSection />
        <PersonaCards />
        <DebateSetup />
      </main>
      <HomeFooter />
    </div>
  );
}