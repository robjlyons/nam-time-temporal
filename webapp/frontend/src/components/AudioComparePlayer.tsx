import React from "react";

export function AudioComparePlayer({ baseUrl }: { baseUrl: string }) {
  return (
    <div className="card">
      <h3>Audio Preview</h3>
      <p>Use generated preview WAV files from training output directory.</p>
      <audio controls src={`${baseUrl}/input.wav`} />
      <audio controls src={`${baseUrl}/target.wav`} />
      <audio controls src={`${baseUrl}/model_output.wav`} />
    </div>
  );
}
