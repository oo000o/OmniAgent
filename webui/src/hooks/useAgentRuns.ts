import { useEffect, useState } from "react";

import { fetchAgentRuns } from "@/lib/api";
import type { AgentRun } from "@/lib/types";

export function useAgentRuns(open: boolean, token: string, sessionKey: string) {
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setFailed(false);
    void fetchAgentRuns(token, sessionKey)
      .then((payload) => {
        if (!cancelled) setRuns(payload.runs);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, sessionKey, token]);

  return { runs, loading, failed };
}
