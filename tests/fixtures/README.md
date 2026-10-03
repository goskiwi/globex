# 回归输入来源

`context-stop-search-results.json` 提取自
`eval/verification/advice-page-20261001/browser-context-failure.json` 的第二条记录。
只保留商品搜索回执中的 `result_ref`、`product_id` 和 `sku_id`，保持原事件顺序；
不包含买家输入、身份、交易、地址或请求参数。

用途：复现读取超过100件商品后因上下文容量停止时，只交回已有证据，不能生成推荐候选；
断言仍由真实 TaskEvidence 执行。原始日志不修改，测试不再依赖日志目录挂载。
