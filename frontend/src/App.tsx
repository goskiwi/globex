import ContextWorkspace, { useContextWorkspace } from "./components/ContextWorkspace";
import ShoppingForm, { displayShoppingMessage } from "./components/ShoppingForm";
import { ToolApprovalCards } from "./components/ToolApprovalCards";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useCommerceAgent } from "./hooks/useCommerceAgent";
import { readProducts } from "./lib/commerceClient";
import type { ProductCard } from "./types";
import Icon from "./components/Icon";
import Markdown from "./components/Markdown";
import EventTimeline from "./components/EventTimeline";
import ProductCards, { ProductImage } from "./components/ProductCards";
import ProductHistory from "./components/ProductHistory";
import ProductDetail from "./components/ProductDetail";
import ConfirmationCards from "./components/ConfirmationCards";
import OrderIntentForm from "./components/OrderIntentForm";
import Modal from "./components/Modal";
import {comparisonKey, upsertComparison} from "./lib/comparison";
import ProductComparison from "./components/ProductComparison";
import ExecutionProcess from './components/ExecutionProcess';
import MyOrders from "./components/MyOrders";
import BuyerWorkspace from "./components/BuyerWorkspace";
import type {AuthSession} from "./lib/auth";
import SessionEntry from "./components/SessionEntry";
import {useFrontendUpdate} from "./hooks/useFrontendUpdate";
import type {ProductGroup} from "./lib/productGroups";

type View = "shopping" | "history" | "favorites" | "skills" | "preferences" | "orders";
const STARTERS = [
  "预算300元以内，找一个轻便的周末旅行背包，寄到中国。",
  "想买日常通勤耳机，帮我理一理选购思路。",
  "预算100元以内，找适合短途出行的背包。",
];
const DELIVERY_COUNTRIES=[['CN','中国'],['US','美国'],['JP','日本'],['SG','新加坡'],['EU','欧盟']] as const;
const viewKey = (buyer:string)=>`globex.workspace.view.${encodeURIComponent(buyer)}`;
function readView(buyer:string): View {
  try {
    const saved = sessionStorage.getItem(viewKey(buyer));
    if (saved && ["shopping", "history", "favorites", "skills", "preferences", "orders"].includes(saved))
      return saved as View;
  } catch { /* 存储受限时使用首页。 */ }
  return "shopping";
}
export default function App({session,onLogout}:{session:AuthSession;onLogout:()=>void}) {
  const agent = useCommerceAgent(session,onLogout);
  const hasPendingForm=agent.shoppingForms.some(form=>form.origin_run_id&&form.origin_message_id&&!form.submission);
  const applyForm=async(query:string,runId:string)=>{
    await agent.refreshShoppingForms();
    await agent.submitForm(query,runId);
  };
  const [view, setView] = useState<View>(()=>readView(session.buyerId)),
    [input, setInput] = useState("");
  const [favorites, setFavorites] = useState<ProductCard[]>([]),
    [compared, setCompared] = useState<ProductCard[]>([]);
  const [detail, setDetail] = useState<ProductGroup | null>(null),
    [showCompare, setShowCompare] = useState(false),
    [toast, setToast] = useState("");
  const [quoteOpen,setQuoteOpen]=useState(false),[quoteCountry,setQuoteCountry]=useState('');
  const [orderIntent, setOrderIntent] = useState<{
    product: ProductCard;
    skuId: string;
    currency: string;
  } | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null),
    bottomRef = useRef<HTMLDivElement>(null),
    autoScroll = useRef(true),
    programmaticScroll = useRef(false);
  const composerRef=useRef<HTMLDivElement>(null);
  useEffect(()=>{
    const node=composerRef.current;if(!node||typeof ResizeObserver==='undefined')return;
    const update=()=>document.documentElement.style.setProperty('--composer-height',`${Math.ceil(node.getBoundingClientRect().height)}px`);
    const observer=new ResizeObserver(update);observer.observe(node);update();
    return()=>{observer.disconnect();document.documentElement.style.removeProperty('--composer-height');};
  },[view]);
  useEffect(() => {
    try { sessionStorage.setItem(viewKey(session.buyerId), view); } catch {}
  }, [view,session.buyerId]);
  const contextWorkspace = useContextWorkspace({
    sessionId:agent.sessionId,
    busy:agent.status==="running",
    pending:!!agent.toolApprovals?.length||agent.confirmations.some(c=>c.status==="pending"&&!c.expired),
    hasMessages:agent.messages.length>0,
    request:agent.workspaceRequest,
  });
  const replyRunning = agent.status === "running";
  const busy = replyRunning || contextWorkspace.running;
  const updateAvailable=useFrontendUpdate((view==='shopping'||view==='history')&&!busy&&!input.trim()&&!detail&&!showCompare&&!orderIntent&&!quoteOpen&&!hasPendingForm
    &&!agent.toolApprovals?.length&&!agent.confirmations.some(c=>c.status==='pending'&&!c.expired));
  const favoriteIds = useMemo(
    () => new Set(favorites.map((p) => p.product_id)),
    [favorites],
  );
  const comparedIds = useMemo(
    () => new Set(compared.map(comparisonKey)),
    [compared],
  );
  const lastUser = [...agent.messages]
    .reverse()
    .find((message) => message.role === "user");
  const decision = agent.comparison ?? agent.recommendation;
  const shownProducts = decision?.hits ?? [];
  const comparisonLimit = decision?.max_items;
  const comparisonUsesDelivery = compared.every(p => shownProducts.some(hit => comparisonKey(hit) === comparisonKey(p)));
  useEffect(() => {
    if (!decision) return;
    // 当前交付更新了同一规格的资料时，比较也读取这份资料，不保留第二份旧报价。
    setCompared(previous => {
      const next = previous.map(p => decision.hits.find(hit => comparisonKey(hit) === comparisonKey(p)) ?? p);
      return next.some((p, index) => p !== previous[index]) ? next : previous;
    });
  }, [decision]);
  const landedQuote = shownProducts.map(p => p.landed_price).find(q => q && "ship_to" in q);
  const landedDestination = landedQuote && "ship_to" in landedQuote ? landedQuote.ship_to : undefined;
  const currentDestination='ship_to' in agent.shoppingFilters
    ? (typeof agent.shoppingFilters.ship_to==='string'?agent.shoppingFilters.ship_to:undefined):landedDestination;
  const requestQuote=(country:string)=>{
    const scope=agent.recommendation?.mode==='bundle'?'核对整套组合一起购买的总价':'分别核对每个备选，按已交付的规格和数量独立报价';
    submit(`请${scope}，配送至${country}；保留当前预算和偏好，不下单。`);
  };

  const favoriteBusy = useRef(false);
  useEffect(() => {
    let disposed=false;
    void agent.workspaceRequest("/favorites").then(data => {
      if (!disposed) setFavorites(readProducts(data.products));
    }).catch(() => { if (!disposed) setToast("收藏暂时读取失败，请稍后刷新；数据库中的收藏仍保留。"); });
    return () => {disposed=true;};
  }, [agent.workspaceRequest]);
  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 2600);
    return () => window.clearTimeout(timer);
  }, [toast]);
  useEffect(() => {
    setCompared([]);
    setInput("");
    setDetail(null);
    setOrderIntent(null);
    setShowCompare(false);
  }, [agent.sessionId]);
  useEffect(() => {
    let previousY = window.scrollY;
    const onScroll = () => {
      const currentY = window.scrollY;
      const distance =
        document.documentElement.scrollHeight - currentY - window.innerHeight;
      if (!programmaticScroll.current) {
        // 用户向上查看历史即暂停跟随，不要求先滚出某个距离。
        if (currentY < previousY - 1 || distance > 240)
          autoScroll.current = false;
        else if (currentY > previousY && distance < 28)
          autoScroll.current = true;
      }
      previousY = currentY;
    };
    const onWheel = (event: WheelEvent) => {
      if (event.deltaY < 0) autoScroll.current = false;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("wheel", onWheel, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("wheel", onWheel);
    };
  }, []);
  useEffect(() => {
    if (
      view !== "shopping" ||
      detail ||
      showCompare ||
      !autoScroll.current ||
      !agent.messages.length
    )
      return;
    let releaseFrame = 0;
    const frame = requestAnimationFrame(() => {
      if (!autoScroll.current) return;
      programmaticScroll.current = true;
      bottomRef.current?.scrollIntoView({ block: "end", behavior: "auto" });
      releaseFrame = requestAnimationFrame(() => {
        programmaticScroll.current = false;
      });
    });
    return () => {
      cancelAnimationFrame(frame);
      cancelAnimationFrame(releaseFrame);
      programmaticScroll.current = false;
    };
  }, [agent.messages, agent.recommendation, agent.comparison, view, detail, showCompare]);

  const submit = useCallback(
    (query: string) => {
      if (!query.trim() || busy) return;
      setView("shopping");
      setInput("");
      setCompared([]);
      setShowCompare(false);
      autoScroll.current = true;
      void agent.submit(query.trim());
    },
    [agent.submit, busy],
  );

  const newShopping = () => {
    agent.reset();
    setView("shopping");
    setInput("");
    autoScroll.current = true;
    window.scrollTo({ top: 0 });
    inputRef.current?.focus();
  };
  const openSession = (id: string) => {
    agent.setSession(id);
    setView("shopping");
    window.scrollTo({ top: 0 });
  };
  const switchView = (next: View) => {
    setView(next);
    window.scrollTo({ top: 0 });
  };
  const toggleFavorite = useCallback(
    (product: ProductCard) => {
      if (favoriteBusy.current) return;
      favoriteBusy.current=true;
      const removing=favoriteIds.has(product.product_id);
      void agent.workspaceRequest("/favorites/"+encodeURIComponent(product.product_id),removing ? "DELETE" : "PUT",removing ? undefined : {product})
        .then(data => {setFavorites(readProducts(data.products));setToast(removing ? "已从心选收藏移除。" : "已保存到心选收藏。");})
        .catch(() => setToast("收藏未能保存，请重试。"))
        .finally(() => {favoriteBusy.current=false;});
    },
    [favoriteIds,agent.workspaceRequest],
  );
  const toggleCompare = useCallback(
    (product: ProductCard) => {
      if (comparedIds.has(comparisonKey(product)))
        setCompared((current) =>
          current.filter((item) => comparisonKey(item) !== comparisonKey(product)),
        );
      else if (comparisonLimit !== undefined && compared.length >= comparisonLimit)
        setToast(`一次可以比较 ${comparisonLimit} 件商品，先移出一件再试试。`);
      else setCompared((current) => [...current, product]);
    },
    [comparedIds, compared.length, comparisonLimit],
  );
  const upsertCompare = useCallback(
    (product: ProductCard) => {
      if (!comparedIds.has(comparisonKey(product)) && comparisonLimit !== undefined && compared.length >= comparisonLimit) {
        setToast(`一次可以比较 ${comparisonLimit} 件商品，先移出一件再试试。`);
        return;
      }
      // 详情选择规格是新增或更新；只有商品卡复选框负责移除比较。
      setCompared((current) => upsertComparison(current, product));
      setToast("已按当前所选规格更新比较信息。 ");
    },
    [comparedIds, compared.length, comparisonLimit],
  );
  const renderCards = (products: ProductCard[]) => (
    <ProductCards
      mode="recommendation"
      products={products}
      preferredSkuId={decision?.preferred_sku_id}
      favoriteIds={favoriteIds}
      comparedIds={comparedIds}
      onFavorite={toggleFavorite}
      onCompare={toggleCompare}
      onDetail={setDetail}
    />
  );
  const deliveredMessageId = agent.deliveredRunId ? `${agent.deliveredRunId}:final:answer` : null;
  const deliveredProducts = shownProducts.length > 0 && (
    <section className="search-results" aria-label="商品结果">
      <div className="results-heading">
        <div className="results-label">
          <strong>{agent.comparison ? "商品比较" : "为你推荐"}</strong> · {shownProducts.length} 款
        </div>
        <button className="results-action" onClick={() => setToast("勾选商品卡下方的“加入比较”，可并排比较本次推荐的商品。") }>
          <Icon name="compare" />勾选商品，轻松对比
        </button>
      </div>
      {agent.recommendation?.quote && <p>组合到手总价：{(agent.recommendation.quote.total_amount_minor / 100).toFixed(2)} {agent.recommendation.quote.currency}</p>}
      {!!decision?.unverified_requirements.length && <p>{decision.unverified_requirements.join("；")}</p>}
      {decision&&shownProducts.some(product=>!product.landed_price)&&<button className="quote-action" disabled={busy}
        onClick={()=>{
          if(currentDestination){requestQuote(currentDestination);}
          else{setQuoteOpen(true);setQuoteCountry('');}
        }}>计算到手价</button>}
      {agent.comparison&&<button className="quote-action" onClick={()=>{setCompared([...shownProducts]);setShowCompare(true);}}>查看对比</button>}
      <>
        {!!decision?.dimensions.length && <p className="comparison-focus">本次关注：{decision.dimensions.join("、")}</p>}
        {renderCards(shownProducts)}
      </>
    </section>
  );
  const navItems: { id: View; label: string; icon: string }[] = [
    { id: "shopping", label: "我的选购", icon: "bag" },
    { id: "orders", label: "我的订单", icon: "bag" },
    { id: "history", label: "对话历史", icon: "chat" },
    { id: "favorites", label: "心选收藏", icon: "heart" },
    { id: "skills", label: "我的 Skill", icon: "leaf" },
    { id: "preferences", label: "长期偏好", icon: "spark" },
  ];
  async function deleteConversation(id:string){
    const current=id===agent.sessionId;
    await agent.deleteSession(id);
    if(current){setView('shopping');setInput('');setCompared([]);setDetail(null);setShowCompare(false);setOrderIntent(null);}
    setToast('已删除对话');
  }

  return (
    <>
      {updateAvailable&&<div className="frontend-update" role="status">页面有更新；当前输入和操作会保留，空闲后自动加载新版。</div>}
      <aside className="sidebar" aria-label="主导航">
        <button
          className="brand"
          onClick={() => switchView("shopping")}
          aria-label="Globex 环球好物首页"
        >
          <Icon name="globe" className="brand-mark" />
          <span>
            <span className="brand-name">Globex</span>
            <span className="brand-subtitle">环球好物</span>
          </span>
        </button>
        <button className="new-chat" onClick={newShopping} disabled={busy}>
          <Icon name="plus" />
          开启一次新选购
        </button>
        <nav className="nav">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${view === item.id ? "active" : ""}`}
              onClick={() => switchView(item.id)}
              aria-current={view === item.id ? "page" : undefined}
            >
              <Icon name={item.icon} />
              {item.label}
              {item.id === "favorites" && (
                <span className="nav-count">{favorites.length}</span>
              )}
            </button>
          ))}
        </nav>
        <section className="sidebar-history" aria-labelledby="recent-shopping-title">
          <div className="nav-label" id="recent-shopping-title">最近选购</div>
          <div className="sidebar-history-list">
            {agent.history.slice(0, 4).map((item) => (
              <SessionEntry key={item.id} item={item} compact current={item.id===agent.sessionId} deleteDisabled={busy&&item.id===agent.sessionId}
                onOpen={openSession} onRename={agent.renameSession} onDelete={deleteConversation}/>
            ))}
            {!agent.history.length && (
              <p className="sidebar-empty">第一段选购，等你开启。</p>
            )}
          </div>
        </section>
        <div className="sidebar-bottom">
          <div className="profile">
            <span className="avatar">旅</span>
            <span>
              <span className="profile-name">{session.buyerId}</span>
              <span className="profile-caption">每一次选择，都有新发现</span>
            </span>
            <button className="logout-button" onClick={onLogout}>退出</button>
          </div>
        </div>
      </aside>
      <main>
        <div className="content">
          <header className="topbar">
            <div className="breadcrumb">
              <span>环球好物</span>
              <span>／</span>
              <span>
                {view === "orders" ? "我的订单" : view === "skills" ? "我的 Skill" : view === "preferences" ? "长期偏好" : view === "history"
                  ? "选购对话历史"
                  : view === "favorites"
                    ? "心选收藏"
                    : "为你挑选"}
              </span>
            </div>
            <button
              className="mobile-brand"
              onClick={() => switchView("shopping")}
            >
              <Icon name="globe" />
              Globex
            </button>
            <div className="location">
              <Icon name="pin" />
              {landedDestination
                ? `配送至 ${DELIVERY_COUNTRIES.find(([code])=>code===landedDestination)?.[1]??landedDestination}`
                : "好物，跨越距离"}
            </div>
          </header>
          <nav className="mobile-nav" aria-label="移动导航">
            <button onClick={newShopping} disabled={busy}>
              新选购
            </button>
            {navItems.map((item) => (
              <button
                key={item.id}
                className={view === item.id ? "active" : ""}
                onClick={() => switchView(item.id)}
              >
                {item.label}
              </button>
            ))}
          </nav>
          {view === "orders" && <MyOrders request={agent.workspaceRequest} confirmations={agent.confirmations} busy={agent.confirmationBusy || busy} error={agent.confirmationError} onPrepare={agent.prepareCancel} onResolve={agent.resolveConfirmation} onRefresh={agent.refreshConfirmations} />}
          {(view === "skills" || view === "preferences") && <BuyerWorkspace key={view} mode={view} busy={busy}
            request={agent.workspaceRequest} onSkillsChanged={agent.refreshSkills} />}
          {view === "shopping" && (
            <>
              {!agent.messages.length&&<section className="hero">
                <div className="eyebrow">A LITTLE LESS, A LITTLE BETTER</div>
                <h1>
                  为下一次出发，<em>选得刚刚好。</em>
                </h1>
                <p>说说你的期待。世界各地的好物，我陪你慢慢选。</p>
              </section>}
              {!agent.messages.length && !hasPendingForm ? (
                <section className="welcome-panel">
                  <Icon name="globe" className="welcome-orbit" />
                  <h2>下一件好物，你想找什么？</h2>
                  <p>从一个用途、一段旅程，或一个小偏好聊起。</p>
                  <div className="welcome-ideas">
                    <button onClick={() => submit(STARTERS[0])}>
                      周末出游，轻便背包
                      <Icon name="arrow" />
                    </button>
                    <button onClick={() => submit(STARTERS[1])}>
                      通勤路上的好声音
                      <Icon name="arrow" />
                    </button>
                    <button onClick={() => submit(STARTERS[2])}>
                      预算 100 元以内
                      <Icon name="arrow" />
                    </button>
                  </div>
                  <span className="welcome-caption">
                    每一份推荐，都从你的实际需求开始。
                  </span>
                </section>
              ) : (
                <section className="conversation" aria-label="选购对话">
                  {agent.messages.map((message) =>
                    message.role === "user" ? (
                      <div key={message.id} className="user-turn">
                      <div className="query-row">
                        <div className="query-bubble">{displayShoppingMessage(message,agent.shoppingForms)}</div>
                      </div>
                      {agent.processes.filter(p=>p.userMessageId===message.id).map(process=><ExecutionProcess key={process.runId}
                        process={process} inputResolved={agent.shoppingForms.some(f=>f.origin_run_id===process.runId&&!!f.submission)}
                        stopRequested={agent.stopRequested&&agent.recoverableRunId===process.runId}
                        connection={agent.recoverableRunId===process.runId?agent.connectionState:'idle'}/>)}
                      </div>
                    ) : (
                      <div className="assistant-turn" key={message.id} data-message-id={message.id}>
                        <div className="assistant-row">
                        <div className="assistant-mark">
                          <Icon name="spark" />
                        </div>
                        <div className="assistant-text">
                          <Markdown content={message.content} collapsible={message.content !== decision?.guidance} />
                        </div>
                        </div>
                        {message.id === deliveredMessageId && deliveredProducts}
                        {agent.shoppingForms.filter(form=>message.id===`${form.origin_run_id}:final:answer`).map(form=><ShoppingForm
                          key={form.form_id} form={form} busy={busy||!!agent.toolApprovals?.length||agent.confirmations.some(c=>c.status==='pending'&&!c.expired)}
                          request={agent.workspaceRequest} onApplied={applyForm}
                          continuationStatus={agent.processes.find(p=>p.runId===form.submission?.run_id)?.status}/>)}
                        {agent.productViews?.filter(view=>message.id===`${view.runId}:final:answer`)
                          .map(view=><section key={view.runId} className="product-views" aria-label="商品详情">
                            <ProductCards mode="view" products={view.hits} onDetail={setDetail}/>
                          </section>)}
                        <ProductHistory batches={agent.productHistory?.filter(batch => message.id === `${batch.runId}:final:answer`)}/>
                      </div>
                    ),
                  )}
                </section>
              )}
              {agent.error && (
                <div className="error-panel" role="alert">
                  <Icon name="info" />
                  <div>
                    <strong>暂时没能完成这次选购</strong>
                    <p>{agent.error}</p>
                  </div>
                  {agent.recoverableRunId ? (
                    <>
                      <button disabled={busy} onClick={() => void agent.resume()}>恢复本轮</button>
                      <button disabled={busy} onClick={agent.stop}>停止本轮</button>
                    </>
                  ) : lastUser && (
                    <button
                      disabled={busy}
                      onClick={() => submit(lastUser.content)}
                    >
                      再试一次
                    </button>
                  )}
                </div>
              )}
              <ToolApprovalCards items={agent.toolApprovals ?? []} busy={busy} onResolve={agent.resolveToolApproval} />
              {(agent.confirmations.length > 0 || agent.confirmationError) && (
                <ConfirmationCards
                  confirmations={agent.confirmations}
                  busy={agent.confirmationBusy || busy}
                  error={agent.confirmationError}
                  onResolve={agent.resolveConfirmation}
                  onCancelOrder={agent.prepareCancel}
                  onRefresh={agent.refreshConfirmations}
                />
              )}
              {agent.messages.length > 0 && (
                <div className="suggestions">
                  <button
                    className="suggestion"
                    disabled={busy}
                    onClick={() => {
                      setInput("预算想再少一点，");
                      inputRef.current?.focus();
                    }}
                  >
                    调整预算
                    <Icon name="arrow" />
                  </button>
                  {shownProducts.length > 1 && (
                    <button
                      className="suggestion"
                      onClick={() => {
                        setCompared(shownProducts);
                        setShowCompare(true);
                      }}
                    >
                      一起比较看看
                      <Icon name="compare" />
                    </button>
                  )}
                  <button
                    className="suggestion"
                    onClick={() => switchView("favorites")}
                  >
                    看看我的收藏
                    <Icon name="heart" />
                  </button>
                </div>
              )}
              <EventTimeline events={agent.events} skillUsages={agent.skillUsages} running={busy}>
                <ContextWorkspace state={contextWorkspace}/>
              </EventTimeline>
              <div ref={bottomRef} className="scroll-anchor" />
            </>
          )}
          {view === "favorites" && (
            <>
              <h1 className="library-title">心动的，先留在这里。</h1>
              <p className="library-description">
                收藏已保存到当前用户，刷新或更换浏览器后仍可查看。以下是上次查看的商品信息，价格与库存请重新查询确认。
              </p>
              {favorites.length ? (
                renderCards(favorites)
              ) : (
                <section className="empty-state">
                  <Icon name="heart" />
                  <h2>等待第一份心动</h2>
                  <p>点击商品右上角的爱心，就能把喜欢的留在这里。</p>
                  <button onClick={() => switchView("shopping")}>
                    去发现好物
                    <Icon name="arrow" />
                  </button>
                </section>
              )}
            </>
          )}
          {view === "history" && (
            <>
              <h1 className="library-title">每一次期待，都有迹可循。</h1>
              <p className="library-description">
                选购记录按当前用户保存在服务端。刷新后会重新读取，浏览器缓存仅用于加快展示。
              </p>
              {agent.historyError && <p role="status">{agent.historyError}</p>}
              <div className="history-list">
                {agent.history.length ? (
                  agent.history.map((item) => (
                    <SessionEntry key={item.id} item={item} current={item.id===agent.sessionId} deleteDisabled={busy&&item.id===agent.sessionId}
                      onOpen={openSession} onRename={agent.renameSession} onDelete={deleteConversation}/>
                  ))
                ) : (
                  <section className="empty-state">
                    <Icon name="chat" />
                    <h2>从第一次选购开始</h2>
                    <p>你和 Globex 的每次交流，会为下一次选择留下一点线索。</p>
                    <button onClick={newShopping}>
                      开启新的选购
                      <Icon name="arrow" />
                    </button>
                  </section>
                )}
              </div>
            </>
          )}
        </div>
      </main>
      {view !== "skills" && view !== "preferences" && view !== "orders" && <div className="composer-dock" ref={composerRef}>
        <div className="composer-wrap">
          {compared.length > 0 && (
            <div className="compare-bar">
              <div className="compare-mini">
                {compared.map((p) => (
                  <ProductImage product={p} key={comparisonKey(p)} />
                ))}
              </div>
              <span>
                已选 {compared.length} 件
                {compared.length === 1 ? "，再选一件比比看" : ""}
              </span>
              <button
                className="compare-go"
                onClick={() => setShowCompare(true)}
                disabled={compared.length < 2}
              >
                开始比较
                <Icon name="arrow" />
              </button>
              <button className="compare-clear" onClick={() => setCompared([])}>
                清空
              </button>
            </div>
          )}
          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault();
              if (replyRunning) agent.stop();
              else if (!busy) submit(input);
            }}
          >
            <label htmlFor="query" className="sr-only">
              告诉 Globex 你想寻找的好物
            </label>
            <textarea
              id="query"
              key={agent.sessionId}
              ref={inputRef}
              value={input}
              placeholder="告诉我用途、预算，或你在意的小细节"
              maxLength={4000}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault(); submit(input);
                }
              }}
            />
            <div className="composer-bottom">
              <span className="composer-hint">Enter 发送 · Shift+Enter 换行</span>
              <button
                type="submit"
                className={`send-button ${replyRunning ? "stop" : ""}`}
                disabled={!replyRunning && (busy || !input.trim())}
                aria-label={replyRunning ? "停止生成" : "发送选购需求"}
              >
                <Icon name={replyRunning ? "stop" : "up"} />
                {replyRunning && <span>停止</span>}
              </button>
            </div>
          </form>
        </div>
      </div>}
      {detail && (
        <ProductDetail
          key={detail.key}
          group={detail}
          busy={busy}
          onClose={() => setDetail(null)}
          onCompare={upsertCompare}
          onAsk={submit}
          onPrepare={(product, skuId, currency) => {
            setDetail(null);
            setOrderIntent({ product, skuId, currency });
          }}
        />
      )}
      {quoteOpen&&<Modal title="计算到手价" onClose={()=>setQuoteOpen(false)}>
        <form className="quote-country" onSubmit={event=>{
          event.preventDefault();if(!quoteCountry||busy)return;
          const country=DELIVERY_COUNTRIES.find(([code])=>code===quoteCountry)?.[1];
          setQuoteOpen(false);requestQuote(`${country}（${quoteCountry}）`);
        }}>
          <label>配送国家或地区<select value={quoteCountry} required onChange={event=>setQuoteCountry(event.target.value)}>
            <option value="">请选择配送国家</option>{DELIVERY_COUNTRIES.map(([code,name])=><option key={code} value={code}>{name}</option>)}
          </select></label>
          <button className="primary-button" type="submit" disabled={busy||!quoteCountry}>核对到手价</button>
        </form>
      </Modal>}
      {orderIntent && (
        <OrderIntentForm
          product={orderIntent.product}
          skuId={orderIntent.skuId}
          currency={orderIntent.currency}
          shipTo={currentDestination}
          busy={agent.confirmationBusy}
          error={agent.confirmationError}
          onClose={() => setOrderIntent(null)}
          onPrepare={async (input) => {
            const success = await agent.prepareOrder(input);
            if (success) {
              setView("shopping");
              setToast("确认单已准备好，请核对后决定。");
            }
            return success;
          }}
        />
      )}
      {showCompare && compared.length >= 2 && (
        <Modal title="商品比较" onClose={() => setShowCompare(false)}>
          <ProductComparison products={compared}
            preferredSkuId={comparisonUsesDelivery ? decision?.preferred_sku_id : null}
            dimensions={comparisonUsesDelivery ? decision?.dimensions : []}/>
        </Modal>
      )}
      <div className={`toast ${toast ? "visible" : ""}`} role="status">
        {toast}
      </div>
    </>
  );
}
