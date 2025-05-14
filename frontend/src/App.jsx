import React, { useState } from "react";

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    setResult(data);
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h1>Sentiment Classifier</h1>
      <textarea
        rows={4}
        cols={50}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter your review here"
      />
      <br />
      <button onClick={handleSubmit}>Check Sentiment</button>
      {result && (
        <p>
          Sentiment: <strong>{result.sentiment}</strong>
        </p>
      )}
    </div>
  );
}