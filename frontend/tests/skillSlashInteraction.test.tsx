// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./WorkspaceFixture";
import type { PublishedSkill } from "../src/types";

const plans: PublishedSkill[] = [
  { id: "weekend-travel", version: "v1", title: "周末轻装出游", description: "按行程、重量和预算比较旅行装备。", scope: "shopping", content_hash: "a".repeat(64), expires_at: null },
  { id: "daily-audio", version: "v2", title: "通勤声音方案", description: "比较通勤耳机的佩戴和续航。", scope: "shopping", content_hash: "b".repeat(64), expires_at: null },
];
let host: HTMLDivElement, root: Root;
let catalog: PublishedSkill[], requests: Array<Record<string, any>>, skillsStatus: number;
let clock: number | undefined;
const originalNow = Date.now;
const query = () => host.querySelector<HTMLTextAreaElement>("#query")!;
const options = () => [...host.querySelectorAll<HTMLButtonElement>('[role="option"]')];

beforeEach(() => {
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
  localStorage.clear();
  requests = []; catalog = plans; skillsStatus = 200; clock = undefined;
  vi.spyOn(Date, "now").mockImplementation(() => clock ?? originalNow());
  vi.stubGlobal("scrollTo", vi.fn());
  Element.prototype.scrollIntoView = vi.fn();
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    if (String(url).includes("/skills?")) return Response.json({ capability_digest: "c".repeat(64), skills: catalog }, { status: skillsStatus });
    if (String(url).includes("/sessions?")) return Response.json({ sessions: [] });
    if (String(url).includes("/confirmations?")) return Response.json({ confirmations: [] });
    if (String(url).endsWith("/ag-ui/run") && init?.method === "POST") {
      const body = JSON.parse(String(init.body)); requests.push(body);
      const events = [{ type: "RUN_STARTED", threadId: body.threadId, runId: body.runId },
        { type: "STATE_SNAPSHOT", snapshot: { products: [], skillUsages: [] } },
        { type: "RUN_FINISHED", threadId: body.threadId, runId: body.runId }];
      return new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), { headers: { "Content-Type": "text/event-stream" } });
    }
    throw new Error(`未预期的测试请求：${url}`);
  }));
  host = document.createElement("div"); document.body.append(host); root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals();
});
async function mount() { await act(async () => { root.render(<App />); }); }
async function type(value: string, caret = value.length) {
  await act(async () => {
    const input = query(); input.focus();
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(input, value);
    input.setSelectionRange(caret, caret);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
async function key(key: string, properties: KeyboardEventInit = {}) {
  await act(async () => { query().dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...properties })); });
}
async function click(element: Element) {
  await act(async () => { element.dispatchEvent(new MouseEvent("click", { bubbles: true })); });
}

describe("普通购物输入不要求用户选择 Skill", () => {
  it("直接发送需求，不存在选择入口或 selectedSkill 参数", async () => {
    await mount(); await type("通勤背包预算300元"); await key("Enter");
    expect(requests).toHaveLength(1);
    expect(requests[0].forwardedProps).not.toHaveProperty("selectedSkill");
    expect(requests[0].messages.at(-1).content).toBe("通勤背包预算300元");
    expect(host.querySelector(".plan-picker-toggle")).toBeNull();
    expect(options()).toHaveLength(0);
  });
  it("斜线是普通输入，不唤起 Skill 菜单", async () => {
    await mount(); await type("用途：通勤/旅行");
    expect(options()).toHaveLength(0);
    await click(host.querySelector('[aria-label="发送选购需求"]')!);
    expect(requests[0].messages.at(-1).content).toBe("用途：通勤/旅行");
  });
  it("输入法确认和 Shift Enter 不提交", async () => {
    await mount(); await type("背包");
    await key("Enter", {isComposing:true}); await key("Enter", {shiftKey:true});
    expect(requests).toHaveLength(0);
    await key("Enter"); expect(requests).toHaveLength(1);
  });
  it("目录加载失败不阻止普通购物请求", async () => {
    skillsStatus=503;await mount();await type("查一下耳机");await key("Enter");
    expect(requests).toHaveLength(1);
    expect(requests[0].forwardedProps).not.toHaveProperty("selectedSkill");
  });
});
