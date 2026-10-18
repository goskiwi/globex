# 商品展示媒体

`wanderlite.png`（P1003）、`drypack.png`（P1018）、`budgetpack.png`（P1049）复制自
`docs/previews/assets/`，由内置图像生成工具于 2026-09-09 生成。完整生成记录在
`docs/previews/assets/图像生成记录.md`。

这些图片仅展示虚构商品的概念外观，**不是平台商品实拍，不证明某个 SKU 的颜色、尺寸或材质**。
接口始终标记 `image_kind: "illustration"`，页面应保留“商品示意图”标识；切换 SKU 不改变该说明。
没有媒体的商品返回 `image_url: null`、`image_kind: "placeholder"`，由前端展示品类图标或占位内容。

目录内评分与评论数来自版本化评测数据，`rating_is_live` 为 `false`，不得表述为实时评分。

## 2026-10-18 复用提供的 Globex 展示素材

`catalog-atlas-a-v1.png`、`catalog-atlas-b-v1.png`及对应精确商品ID注册表来自用户提供的`globex-agent-main`本地项目。本次直接复用既有文件，没有重新生成图片或按标题猜测图片。它们是品类示意图，页面保留“商品示意图 · 非实物照片”说明；有目录原图时优先展示原图。注册表仅用于展示，不回写商品事实、历史快照或交易记录。
