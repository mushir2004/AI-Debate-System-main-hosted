"use client";
import { useEffect } from "react";
import { useDebate } from "../../hooks/useDebate";
import DebateHeader           from "../../components/debate/DebateHeader";
import StatusBanner           from "../../components/debate/StatusBanner";
import DebateStage            from "../../components/debate/DebateStage";
import ControlPanel           from "../../components/debate/ControlPanel";
import HumanInterventionPanel from "../../components/debate/HumanInterventionPanel";
import TranscriptPanel        from "../../components/debate/TranscriptPanel";
import JudgeResultPanel       from "../../components/debate/JudgeResultPanel";
import SetupGate              from "../../components/debate/SetupGate";
import type { PersonaKey, DebateSetupConfig } from "../../types";

export default function DebatePage() {
  const debate = useDebate();

  useEffect(() => {
    const raw = sessionStorage.getItem("debateConfig");
    if (!raw) return;
    try {
      const config: DebateSetupConfig = JSON.parse(raw);
      sessionStorage.removeItem("debateConfig");
      debate.setSetupConfig(config);
      setTimeout(() => { debate.startDebate(); }, 300);
    } catch { /* ignore */ }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const proPersona: PersonaKey = debate.proPersona ?? "The Professor";
  const conPersona: PersonaKey = debate.conPersona ?? "The Aggressor";

  if (debate.phase === "setup") {
    return (
      <div style={{ minHeight: "100vh", background: "linear-gradient(135deg, #0d0626 0%, #1a0a40 50%, #0d0626 100%)", fontFamily: "'Impact','Arial Black',sans-serif" }}>
        <SetupGate
          setupConfig={debate.setupConfig}
          setSetupConfig={debate.setSetupConfig}
          onStart={debate.startDebate}
          isStarting={debate.isStarting}
          statusMsg={debate.statusMsg}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col"
      style={{ background: "linear-gradient(135deg, #0d0626 0%, #1a0a40 50%, #0d0626 100%)", fontFamily: "'Impact','Arial Black',sans-serif" }}>
      <DebateHeader isRunning={debate.isRunning} round={debate.round} topic={debate.topic} />
      <main className="flex-1 flex flex-col lg:flex-row gap-4 p-4">
        <div className="flex-1 flex flex-col gap-4">
          <StatusBanner statusMsg={debate.statusMsg} />
          <DebateStage speakingID={debate.speakingID} proPersona={proPersona} conPersona={conPersona} transcript={debate.transcript} />
          <ControlPanel phase={debate.phase} isRunning={debate.isRunning} isJudging={debate.isJudging} onShutUp={debate.shutUp} onResume={debate.resumeDebate} onJudge={debate.requestJudge} />
          <HumanInterventionPanel phase={debate.phase} humanText={debate.humanText} setHumanText={debate.setHumanText} targetPersona={debate.targetPersona} setTargetPersona={debate.setTargetPersona} onSubmit={debate.submitHumanInput} isInterrupting={debate.isInterrupting} fallacyResult={debate.fallacyResult} proPersona={debate.proPersona} conPersona={debate.conPersona} />
          <JudgeResultPanel judgeResult={debate.judgeResult} isJudging={debate.isJudging} />
        </div>
        <TranscriptPanel transcript={debate.transcript} speakingID={debate.speakingID} proPersona={debate.proPersona} conPersona={debate.conPersona} totalTurns={debate.round} />
      </main>
      <footer className="text-center py-2 px-4" style={{ borderTop: "4px solid #000", background: "#0a0220", color: "#ffffff30", fontFamily: "'Arial',sans-serif", fontSize: 10, letterSpacing: "0.15em", textTransform: "uppercase" }}>
        DebateForge — Powered by Ollama + Web Speech API — No AIs were harmed
      </footer>
    </div>
  );
}