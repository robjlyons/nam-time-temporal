import React from "react";

export function LossChart({ log }: { log: string }) {
  return (
    <div className="card">
      <h3>Live Logs / Loss</h3>
      <pre className="log">{log}</pre>
    </div>
  );
}
