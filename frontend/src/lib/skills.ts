import type { PublishedSkill, SkillUsage } from "../types";

const record = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);
const text = (value: unknown, max = 300): value is string =>
  typeof value === "string" && !!value.trim() && value.length <= max;
const hash = (value: unknown): value is string =>
  typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const identifier = (value: unknown): value is string =>
  typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(value);

/** 只展示真实发布清单；无效条目剔除，错误响应不能伪装成空库。 */
export function readPublishedSkills(value: unknown, now = Date.now()): PublishedSkill[] {
  if (!record(value) || !hash(value.capability_digest) || !Array.isArray(value.skills))
    throw new Error("选购方案清单格式无效");
  const seen = new Set<string>();
  let malformed = false;
  const skills = value.skills.flatMap((entry): PublishedSkill[] => {
    if (!record(entry) || !identifier(entry.id) || !identifier(entry.version) || !text(entry.title, 160)
      || typeof entry.description !== "string" || entry.description.length > 2000 || !text(entry.scope, 160)
      || !hash(entry.content_hash)) { malformed = true; return []; }
    if (entry.expires_at !== null && (typeof entry.expires_at !== "string"
      || !Number.isFinite(Date.parse(entry.expires_at)))) { malformed = true; return []; }
    if (entry.expires_at !== null && Date.parse(entry.expires_at) <= now) return [];
    const key = `${entry.id}@${entry.version}`;
    if (seen.has(key)) return [];
    seen.add(key);
    return [{ id: entry.id, version: entry.version, title: entry.title, description: entry.description,
      scope: entry.scope, content_hash: entry.content_hash, expires_at: entry.expires_at }];
  });
  if (malformed && !skills.length) throw new Error("选购方案条目格式无效");
  return skills;
}

export function readSkillUsages(value: unknown): SkillUsage[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry): SkillUsage[] => {
    if (!record(entry) || !text(entry.toolCallId, 200) || !["reading", "used", "error"].includes(String(entry.status))) return [];
    // 只有后端的有效工具成功投影可标记已读取；选择或工具开始都不是成功。
    if (entry.status === "used" && (!text(entry.id, 160) || !text(entry.version, 80) || !hash(entry.contentHash))) return [];
    return [{ toolCallId: entry.toolCallId, status: entry.status as SkillUsage["status"],
      ...(text(entry.id, 160) ? { id: entry.id } : {}), ...(text(entry.version, 80) ? { version: entry.version } : {}),
      ...(text(entry.title, 160) ? { title: entry.title } : {}), ...(hash(entry.contentHash) ? { contentHash: entry.contentHash } : {}) }];
  });
}

export function skillUsageLabel(usage: SkillUsage): string {
  return `${usage.status === "used" ? "已读取选购方案" : usage.status === "reading" ? "正在读取选购方案" : "选购方案未能读取"}：${usage.title || "选购方案"}${usage.version ? ` · ${usage.version}` : ""}`;
}
