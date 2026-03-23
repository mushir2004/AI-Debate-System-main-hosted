import { SignUp } from "@clerk/nextjs";

export default function SignUpPage() {
  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "linear-gradient(160deg, #0d0626 0%, #1a0a40 55%, #0d0626 100%)" }}>
      <SignUp />
    </div>
  );
}