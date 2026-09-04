import { Activity, CircleAlert, Clock3, Cpu, RefreshCcw } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { useAgentRuns } from "@/hooks/useAgentRuns";
import type { AgentRun } from "@/lib/types";

export const UNIFIED_RUN_SESSION_KEY = "unified:default";

export function resolveRunSessionKey(sessionKey: string, unifiedSession: boolean): string {
  return unifiedSession ? UNIFIED_RUN_SESSION_KEY : sessionKey;
}

export function RunDetailsPopover({
  sessionKey,
  token,
}: {
  sessionKey: string;
  token: string;
}) {
  const [open, setOpen] = useState(false);
  const { runs, loading, failed } = useAgentRuns(open, token, sessionKey);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Run details"
          className="host-no-drag h-8 w-8 rounded-full text-muted-foreground/85 hover:bg-accent/40 hover:text-foreground"
        >
          <Activity className="h-4 w-4 stroke-[1.75]" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" sideOffset={8} className="w-[min(26rem,calc(100vw-1.5rem))] p-0">
        <div className="space-y-3 px-4 py-3.5">
          <div>
            <div className="text-[12px] text-muted-foreground/75">OmniAgent</div>
            <div className="mt-0.5 text-[14px] font-medium">Recent runs</div>
          </div>
          <div className="h-px bg-border/45" />
          {loading ? (
            <div className="flex items-center gap-2 py-3 text-xs text-muted-foreground">
              <RefreshCcw className="h-3.5 w-3.5 animate-spin" /> Loading runs…
            </div>
          ) : failed ? (
            <div className="flex items-center gap-2 py-3 text-xs text-destructive">
              <CircleAlert className="h-3.5 w-3.5" /> Could not load run details.
            </div>
          ) : runs.length ? (
            <div className="max-h-80 space-y-1.5 overflow-y-auto">
              {runs.map((run) => <RunRow key={run.run_id} run={run} />)}
            </div>
          ) : (
            <div className="py-3 text-xs text-muted-foreground">No recorded runs yet.</div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}

function RunRow({ run }: { run: AgentRun }) {
  const duration = run.latency_ms === null ? "running" : `${run.latency_ms} ms`;
  const tokens = run.total_tokens === null ? "—" : run.total_tokens.toLocaleString();
  const failed = run.status === "error" || Boolean(run.error);
  return (
    <div className="rounded-floating bg-muted/35 px-3 py-2.5 text-xs">
      <div className="flex items-center justify-between gap-3">
        <span className="truncate font-medium">{run.model || "Model pending"}</span>
        <span className={failed ? "text-destructive" : "text-emerald-600 dark:text-emerald-400"}>
          {run.status}
        </span>
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-muted-foreground">
        <span>{run.channel}</span>
        <span className="flex items-center gap-1"><Clock3 className="h-3 w-3" />{duration}</span>
        <span className="flex items-center gap-1"><Cpu className="h-3 w-3" />{tokens} tokens</span>
        <span>{run.tool_calls} tools · {run.retries} retries</span>
      </div>
      {run.error ? <div className="mt-1.5 line-clamp-2 text-destructive">{run.error}</div> : null}
    </div>
  );
}
