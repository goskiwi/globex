import type { PublishedSkill } from "../types";
import Icon from "./Icon";
import "./shoppingPlans.css";

interface Props {
  skills: PublishedSkill[];
  status: "loading" | "ready" | "error";
  error: string | null;
  compact?: boolean;
  onRefresh: () => void;
}

export default function ShoppingPlans({ skills, status, error, compact, onRefresh }: Props) {
  return <section className={`shopping-plans ${compact ? "shopping-plans--compact" : ""}`} aria-label="选购方案">
    <div className="shopping-plans-heading">
      <div><span className="plan-eyebrow"><Icon name="leaf" />从场景出发</span><h2>选购方案</h2></div>
      <button type="button" className="plan-refresh" onClick={onRefresh} disabled={status === "loading"}>刷新方案</button>
    </div>
    <p className="plan-introduction">这些流程由 Agent 根据需求按需读取。直接描述用途、预算和偏好即可，无需选择方案。</p>
    {status === "loading" ? <p className="plan-empty" role="status">正在查看可用的选购方案…</p>
      : status === "error" ? <p className="plan-empty plan-load-error" role="status">{error || "选购方案暂时无法加载，仍可直接描述需求。"}</p>
      : skills.length === 0 ? <p className="plan-empty" role="status"><Icon name="leaf" />选购方案筹备中，可直接描述需求</p>
      : <div className="plan-grid">{skills.map((skill) => {
        return <article className="plan-card" key={`${skill.id}@${skill.version}`}>
          <span className="plan-card-top"><strong>{skill.title}</strong><span className="plan-version">{skill.version}</span></span>
          <span className="plan-description">{skill.description || "Agent 按任务需要读取。"}</span>
        </article>;
      })}</div>}
  </section>;
}
