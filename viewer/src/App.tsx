import { useState } from "react";
import { Id } from "../convex/_generated/dataModel";
import { SessionList } from "./components/SessionList";
import { ResearchGraph } from "./components/ResearchGraph";

function App() {
  const [selectedSessionId, setSelectedSessionId] = useState<Id<"sessions"> | null>(
    null
  );

  if (selectedSessionId) {
    return (
      <ResearchGraph
        sessionId={selectedSessionId}
        onBack={() => setSelectedSessionId(null)}
      />
    );
  }

  return <SessionList onSelectSession={setSelectedSessionId} />;
}

export default App;
