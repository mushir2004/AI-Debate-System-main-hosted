export type PersonaKey =
  | "The Professor"
  | "The Aggressor"
  | "The Philosopher"
  | "The Troll";

export type SpeakerSide = "Pro" | "Con" | "User";

export type DebateLanguage = "english" | "hindi" | "bengali" | "tamil" | "telugu";

export interface PersonaMeta {
  key: PersonaKey;
  emoji: string;
  tagline: string;
  color: string;
  shadow: string;
  nameBg: string;
  voicePitch: number;
  voiceRate: number;
  preferHighVoice: boolean;
  description: string;
}

export interface HistoryEntry {
  speaker: SpeakerSide;
  text: string;
}

export interface FirstTurn {
  speaker: SpeakerSide;
  text: string;
  history: HistoryEntry[];
  audio_url?: string;
}

export interface StartDebateResponse {
  session_id: string;
  topic: string;
  pro_character: PersonaKey;
  con_character: PersonaKey;
  language: DebateLanguage;
  first_turn: FirstTurn;
}

export interface NextTurnResponse {
  speaker: SpeakerSide;
  text: string;
  history: HistoryEntry[];
  audio_url?: string;
  status?: string;
}

export interface FallacyResult {
  fallacy: string;
  explanation: string;
}

export interface InterruptResponse {
  status: string;
  fallacies_detected: FallacyResult;
  steelmanned_version: string;
  next_speaker: SpeakerSide;
}

export interface JudgeVerdict {
  winner: "Pro" | "Con";
  reason: string;
  brag: string;
}

export interface JudgeResponse {
  verdict: JudgeVerdict;
  winning_character: PersonaKey;
}

export interface LeaderboardEntry {
  wins: number;
  last_brag: string;
}

export type LeaderboardData = Record<PersonaKey, LeaderboardEntry>;

export interface TranscriptLine {
  id: string;
  speaker: SpeakerSide;
  personaName: string;
  text: string;
  isHuman?: boolean;
  isSteelmanned?: boolean;
}

export interface FallacyDisplay {
  fallacy: string;
  explanation: string;
  steelmanned: string;
}

export interface DebateSetupConfig {
  topic: string;
  proPersona: PersonaKey;
  conPersona: PersonaKey;
  language: DebateLanguage;  // ← added
}