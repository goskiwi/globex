import { useEffect, useState, useSyncExternalStore } from "react";
import { CommerceClient } from "../lib/commerceClient";
import type {AuthSession} from "../lib/auth";

export function useCommerceAgent(session:AuthSession,onUnauthorized:()=>void) {
  const [client] = useState(() => {
    let storage: Storage | undefined;
    try {
      storage = window.localStorage;
    } catch {
      /* 存储受限时使用内存。 */
    }
    const base = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
    return new CommerceClient({ url: `${base}/commerce/ag-ui/run`, storage,
      buyerId: session.buyerId, accessToken:session.accessToken,onUnauthorized });
  });
  const snapshot = useSyncExternalStore(client.subscribe, client.getSnapshot);
  useEffect(() => { void client.initialize(); return () => client.detach(); }, [client]);
  useEffect(() => {
    void client.refreshConfirmations();
    void client.refreshSkills();
  }, [client, snapshot.sessionId]);
  return {
    ...snapshot,
    submit: client.submit,
    submitForm: client.submitForm,
    refreshShoppingForms:client.refreshShoppingForms,
    resolveToolApproval: client.resolveToolApproval,
    stop: client.stop,
    reset: client.reset,
    setSession: client.setSession,
    renameSession:client.renameSession,
    deleteSession:client.deleteSession,
    prepareOrder: client.prepareOrder,
    prepareCancel: client.prepareCancel,
    resolveConfirmation: client.resolveConfirmation,
    refreshConfirmations: client.refreshConfirmations,
    refreshSkills: client.refreshSkills,
    workspaceRequest: client.workspaceRequest,
    resume: client.resume,
  };
}
