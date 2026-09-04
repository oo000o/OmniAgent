import { describe, expect, it } from "vitest";

import {
  resolveRunSessionKey,
  UNIFIED_RUN_SESSION_KEY,
} from "@/components/thread/RunDetailsPopover";

describe("resolveRunSessionKey", () => {
  it("uses the canonical key when unified sessions are enabled", () => {
    expect(resolveRunSessionKey("websocket:temporary-id", true)).toBe(
      UNIFIED_RUN_SESSION_KEY,
    );
  });

  it("keeps the channel session key when unified sessions are disabled", () => {
    expect(resolveRunSessionKey("websocket:temporary-id", false)).toBe(
      "websocket:temporary-id",
    );
  });
});
