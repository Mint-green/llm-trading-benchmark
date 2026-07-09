# LLM 多市场投资 Benchmark / 回测系统：当前完整技术方案

## 0. 文档目的

本文定义当前版本的 LLM 多市场投资 Benchmark / 回测系统技术方案。系统目标是评估不同 LLM 在历史市场环境中的投资与交易决策能力，核心指标仍然以净值收益率为主，同时完整记录风险控制、行为特征、工具使用、交易纪律、换手率、拒单、约束触发、记忆状态变化、计划执行与收益归因。

本方案不是自动实盘交易系统，也不是高频撮合模拟器，而是一个历史回测 Benchmark。它试图模拟一个散户/轻量投资者在交易软件、行情筛选器、持仓计划、风险提示和历史交易笔记辅助下进行决策的过程。

系统总体原则：

```text
Profit First：收益率是主目标，但不承诺正收益。
No Future Leakage：任何输入必须在 decision_timestamp 当时可获得。
Context Limited：模型不能看到无限全市场原始数据。
Realistic Friction：手续费、滑点、市场规则、FX、lot、T+1、涨跌停、保证金等由系统处理。
Reproducible：数据、prompt、工具、模型版本、状态、执行、总结都可回放。
Auditable：每一次模型输入、输出、工具调用、执行结果、状态变化都可审计。
```

------

# 1. 总体架构

## 1.1 核心思想

当前版本不再依赖“上下文连续累加”来维持模型记忆，而是通过系统维护结构化状态：

```text
模型不是每 5 分钟从零开始。
模型每次看到：
- 当前市场环境
- 当前持仓
- 当前有效计划
- 当前观察名单
- 当前禁买/冷却名单
- 最近重要操作
- 前一日总结
- 相关 focused 事件
```

系统通过 `Agent State / Plan State` 让模型表现得像一个连续盯盘的投资者。

## 1.2 分层架构

```text
Historical Data Layer
  - 行情数据
  - FX 数据
  - 市场规则数据
  - 资产状态数据
  - 合约映射数据
  - 新闻数据预留

Feature Layer
  - 收益率
  - 波动率
  - RSI
  - ATR
  - trend
  - liquidity
  - cost
  - tradability
  - regime

Candidate Builder
  - 分桶候选
  - 风险过滤
  - 成本过滤
  - 市场内标准化
  - blocked/warning 标记

Portfolio & Accounting
  - 多币种现金
  - 持仓
  - USD NAV
  - PnL
  - FX translation
  - fees/slippage

Agent State Manager
  - active plans
  - plan triggers
  - watchlist
  - avoid/cooldown
  - daily thesis
  - recent activity
  - session summary
  - daily summary

Decision Scheduler
  - 5min scanner
  - full decision trigger
  - focused decision trigger
  - auto hold
  - session close summary
  - global daily summary

Prompt Builder
  - full decision prompt
  - focused position prompt
  - focused market/risk prompt
  - summary prompt

LLM Agent
  - 原生查询工具
  - 最终 JSON 决策输出

Tool Runner
  - screen_universe
  - query_asset
  - query_position
  - query_history
  - query_market_overview
  - query_fx
  - query_futures_contract
  - query_news 预留

Decision Parser & Validator
  - schema 校验
  - action 校验
  - target 校验
  - trigger 校验
  - memory_update 校验

Constraint Engine
  - 仓位限制
  - 市场暴露限制
  - crypto/futures 限制
  - cash reserve
  - 日交易频率限制
  - BUY/SELL 数量限制

Market Rule Engine
  - 市场开闭
  - T+1
  - 涨跌停
  - 停牌
  - 退市
  - lot / board lot
  - 期货合约状态

Execution Engine
  - target_pct_nav → 真实单位
  - 期货 notional/margin → contracts
  - 成本/滑点
  - 拒单/调整/成交

Settlement Engine
  - 股票结算
  - A股 T+1 冻结
  - 多币种现金
  - FX
  - futures daily mark-to-market

Metrics & Attribution
  - Total Return
  - Sharpe
  - MDD
  - turnover
  - rejected orders
  - constraint hits
  - behavior metrics
  - PnL attribution

Audit & Replay Store
  - prompt snapshot
  - raw response
  - parsed decision
  - tool calls
  - execution result
  - state diff
  - replay capability
```

------

# 2. 时间体系与决策节奏

## 2.1 数据频率

系统仍然以 5min bar 作为主要行情更新节奏：

```text
每 5min：
- 更新行情
- 更新 FX
- 更新持仓市值
- 更新 NAV
- 更新候选分数
- 更新风险状态
- 更新 active plan 的 peak/trough
- 检查 plan triggers
- 判断是否需要 LLM focused decision
```

但并不是每 5min 都调用 LLM 做完整投资决策。

## 2.2 决策类型

系统支持四种决策类型：

```text
auto_hold
full_decision
focused_position_decision
focused_market_or_risk_decision
```

### auto_hold

没有重要事件，也未到 full decision 时间点时，不调用 LLM，系统直接记录 auto_hold。

### full_decision

常规完整决策。用于重新审视市场、持仓、候选、计划和记忆。

触发条件：

```text
- 每 30min 一个常规 full decision
- 某市场开盘后前 30min 内，可以每 15min 一次
- 每个 benchmark day 第一次决策
- 重大市场状态变化后需要全局重评
```

### focused_position_decision

针对单个或少数持仓的事件型决策。

触发条件：

```text
- 持仓触发止损复查
- 持仓触发止盈复查
- trailing stop 触发
- bars_elapsed 到复查时间
- T+1 解冻
- 持仓从可交易变不可交易
- 单个持仓风险状态变化
```

### focused_market_or_risk_decision

针对市场或风险事件的短决策。

触发条件：

```text
- 市场 regime 从 GREEN/YELLOW 变 RED
- crypto 风险急剧上升
- futures margin warning/danger
- 市场即将收盘
- 市场开闭切换
- 总风险暴露接近限制
```

------

# 3. 每 5 分钟 Scanner 逻辑

## 3.1 scanner 主要职责

```python
def on_5min_bar(timestamp):
    update_market_data(timestamp)
    update_fx(timestamp)
    update_asset_status(timestamp)
    update_portfolio_valuation(timestamp)
    update_nav(timestamp)
    update_feature_store(timestamp)
    update_candidate_scores(timestamp)
    update_risk_state(timestamp)
    update_plan_runtime_fields(timestamp)
    expire_memory_items(timestamp)
    events = detect_events(timestamp)
    decision_request = schedule_decision(timestamp, events)
    process_decision_request(decision_request)
```

## 3.2 事件检测

系统每 5min 检查：

```text
Portfolio Events:
- position PnL below threshold
- position PnL above take-profit review threshold
- trailing drawdown triggered
- bars_elapsed trigger
- position becomes sellable
- position violates plan

Market Events:
- market open / close
- market close guard
- volatility spike
- regime change
- breadth collapse

Risk Events:
- cash below reserve
- market exposure near max
- crypto exposure near max
- futures margin warning/danger/breach

Memory Events:
- watchlist condition reached
- avoid expired
- plan expired
```

## 3.3 事件优先级

```text
P0：系统强制处理
- margin breach
- hard constraint breach
- forced liquidation
- delisted asset handling

P1：必须 focused decision
- stop review
- trailing stop
- severe adverse move
- regime turns RED

P2：可合并 focused decision
- take-profit review
- scheduled review
- watchlist condition reached

P3：仅记录
- low impact event
- minor fluctuation
```

P0 不问 LLM。P1/P2 进入 focused decision 队列。

------

# 4. Candidate Layer 设计

## 4.1 不在首屏展开完整 universe

完整 universe 仍然存在，且理论可交易，但首屏 prompt 不再完整列出所有 ticker。模型如需发现候选外标的，应调用 `screen_universe` 工具。

首屏只提供：

```text
- universe 统计摘要
- 分桶候选
- 当前持仓
- watchlist/avoid
- blocked/warning
```

## 4.2 Candidate Buckets

候选池不再是单一总榜，而是分桶展示：

```text
held_positions
exit_watch
trend_leaders
pullback_continuation
oversold_reversal
low_vol_defensive
crypto_candidates
futures_macro
blocked_or_warning
```

### held_positions

当前持仓。必须展示，不参与筛选淘汰。

字段：

```text
symbol|mkt|pct_nav|pnl_pct|hold_bars|trend|rsi|sellable|plan_status|risk_note
```

### exit_watch

当前持仓中可能需要减仓/平仓的标的。

进入条件：

```text
is_held = true
AND sellable_now = true
AND (
  pnl_pct <= stop_threshold
  OR trend_score < 0.35
  OR current_score drops significantly
  OR risk_budget requests exposure reduction
)
```

字段：

```text
symbol|mkt|pct_nav|pnl_pct|trend|rsi|reason|allowed_action
```

### trend_leaders

趋势强、成交活跃、不过度过热的标的。

字段：

```text
symbol|mkt|price|score|1h_pct|1d_pct|5d_pct|rsi|trend|cost|tradable
```

### pullback_continuation

中期趋势仍好、短线回调、适合等待或小仓入场的标的。

字段：

```text
symbol|mkt|price|score|1d_pct|5d_pct|rsi|trend|pullback_note|tradable
```

### oversold_reversal

超跌反弹候选，数量应少，风险提示要明显。

字段：

```text
symbol|mkt|price|score|1d_pct|5d_pct|rsi|stabilization|risk_note|tradable
```

### low_vol_defensive

低波动、防御型、风险降低时的候选。

字段：

```text
symbol|mkt|price|score|20d_pct|atr_pct|drawdown|cost|tradable
```

### crypto_candidates

crypto 单独成桶，不与股票抢总榜排名。

字段：

```text
symbol|price|score|1h_pct|1d_pct|5d_pct|rsi|volatility|liquidity|risk_note
```

### futures_macro

期货单独成桶，展示 notional、margin、multiplier、risk budget。

字段：

```text
symbol|actual_contract|price|trend|notional_per_contract|margin_per_contract|suggested_max_notional_pct_nav|risk_note
```

### blocked_or_warning

不是推荐买入，而是避免误操作。

字段：

```text
symbol|mkt|reason|allowed_action
```

------

# 5. 工具设计

## 5.1 总原则

```text
查询信息 → 原生 tool/function calling
交易执行 → 不给模型工具，由系统执行
状态更新 → 不给模型工具，由模型 final JSON 字段表达，系统落库
每日总结 → SummaryEngine 生成，不由交易模型自由写入
```

## 5.2 查询工具完整定义

### 5.2.1 screen_universe

用途：从完整 universe 中按条件筛选一批候选。

```json
{
  "name": "screen_universe",
  "description": "Screen the point-in-time investable universe using market, bucket, filters, and sorting rules. Read-only. Does not change portfolio or memory.",
  "parameters": {
    "type": "object",
    "properties": {
      "market": {
        "type": "string",
        "enum": ["US", "CN", "HK", "CRYPTO", "GOLD_OIL", "FUTURES", "ALL"]
      },
      "bucket": {
        "type": "string",
        "enum": [
          "trend_leaders",
          "pullback_continuation",
          "oversold_reversal",
          "low_vol_defensive",
          "volume_breakout",
          "news_movers",
          "held_weakness",
          "custom"
        ]
      },
      "filters": {
        "type": "object",
        "properties": {
          "rsi_min": {"type": "number"},
          "rsi_max": {"type": "number"},
          "min_1d_return": {"type": "number"},
          "min_5d_return": {"type": "number"},
          "max_atr_pct": {"type": "number"},
          "min_liquidity_rank": {"type": "number"},
          "tradable_now": {"type": "boolean"},
          "exclude_current_positions": {"type": "boolean"},
          "exclude_recent_failed": {"type": "boolean"},
          "exclude_blocked": {"type": "boolean"}
        },
        "additionalProperties": false
      },
      "sort_by": {
        "type": "string",
        "enum": [
          "score_total",
          "trend_score",
          "risk_adjusted_momentum",
          "entry_quality",
          "volume_rank",
          "cost_efficiency",
          "portfolio_fit"
        ]
      },
      "limit": {
        "type": "integer",
        "minimum": 1,
        "maximum": 30
      }
    },
    "required": ["market", "bucket", "limit"],
    "additionalProperties": false
  }
}
```

### 5.2.2 query_asset

用途：查询具体资产的详情。

```json
{
  "name": "query_asset",
  "description": "Query point-in-time details for a specific tradable asset. Read-only.",
  "parameters": {
    "type": "object",
    "properties": {
      "symbol": {"type": "string"},
      "fields": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": [
            "quote",
            "recent_bars",
            "indicators",
            "tradability",
            "cost",
            "lot_info",
            "corporate_action",
            "risk_flags",
            "news_flags"
          ]
        }
      },
      "recent_bar_count": {
        "type": "integer",
        "minimum": 0,
        "maximum": 48
      }
    },
    "required": ["symbol", "fields"],
    "additionalProperties": false
  }
}
```

### 5.2.3 query_position

用途：查询当前持仓细节。

```json
{
  "name": "query_position",
  "description": "Query current position, plan, PnL, tradability, and execution history for a symbol. Read-only.",
  "parameters": {
    "type": "object",
    "properties": {
      "symbol": {"type": "string"},
      "include_plan": {"type": "boolean"},
      "include_recent_executions": {"type": "boolean"}
    },
    "required": ["symbol"],
    "additionalProperties": false
  }
}
```

### 5.2.4 query_history

用途：查询历史行情，不允许返回 decision_timestamp 之后的数据。

```json
{
  "name": "query_history",
  "description": "Query historical price and indicator data available at the decision timestamp. Read-only.",
  "parameters": {
    "type": "object",
    "properties": {
      "symbol": {"type": "string"},
      "lookback_bars": {
        "type": "integer",
        "minimum": 1,
        "maximum": 500
      },
      "bar_size": {
        "type": "string",
        "enum": ["5m", "15m", "1h", "1d"]
      },
      "fields": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["ohlcv", "returns", "rsi", "atr", "ema", "volume_rank"]
        }
      }
    },
    "required": ["symbol", "lookback_bars", "bar_size"],
    "additionalProperties": false
  }
}
```

### 5.2.5 query_market_overview

用途：查询市场级概览。

```json
{
  "name": "query_market_overview",
  "description": "Query point-in-time market overview, regime, breadth, volatility, and open/close status. Read-only.",
  "parameters": {
    "type": "object",
    "properties": {
      "markets": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["US", "CN", "HK", "CRYPTO", "GOLD_OIL", "FUTURES", "FX"]
        }
      },
      "include_regime": {"type": "boolean"},
      "include_breadth": {"type": "boolean"},
      "include_risk_flags": {"type": "boolean"}
    },
    "required": ["markets"],
    "additionalProperties": false
  }
}
```

### 5.2.6 query_fx

用途：查询 FX 与换汇成本。

```json
{
  "name": "query_fx",
  "description": "Query point-in-time FX rates, cash balances, and conversion costs. Read-only.",
  "parameters": {
    "type": "object",
    "properties": {
      "base_currency": {"type": "string"},
      "quote_currency": {"type": "string"},
      "include_cash_balances": {"type": "boolean"},
      "include_conversion_cost": {"type": "boolean"}
    },
    "required": [],
    "additionalProperties": false
  }
}
```

### 5.2.7 query_futures_contract

用途：查询期货连续合约映射、actual_contract、multiplier、margin、roll 状态。

```json
{
  "name": "query_futures_contract",
  "description": "Query current actual futures contract mapped from a continuous symbol, including multiplier, margin, liquidity, and roll metadata. Read-only.",
  "parameters": {
    "type": "object",
    "properties": {
      "continuous_symbol": {"type": "string"},
      "fields": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": [
            "actual_contract",
            "price",
            "multiplier",
            "notional",
            "initial_margin",
            "maintenance_margin",
            "volume",
            "open_interest",
            "roll_status",
            "last_trade_date"
          ]
        }
      }
    },
    "required": ["continuous_symbol"],
    "additionalProperties": false
  }
}
```

### 5.2.8 query_news

当前版本默认不注册到主榜交易模型。新闻模块先不上主流程，保留接口用于后续 ablation。

若开启新闻实验，定义如下：

```json
{
  "name": "query_news",
  "description": "Query point-in-time news available at decision timestamp. Disabled in primary benchmark unless news ablation is enabled.",
  "parameters": {
    "type": "object",
    "properties": {
      "symbol": {"type": "string"},
      "keyword": {"type": "string"},
      "markets": {
        "type": "array",
        "items": {"type": "string"}
      },
      "lookback_hours": {
        "type": "integer",
        "minimum": 1,
        "maximum": 168
      },
      "max_items": {
        "type": "integer",
        "minimum": 1,
        "maximum": 10
      },
      "min_impact_score": {"type": "number"}
    },
    "required": ["lookback_hours", "max_items"],
    "additionalProperties": false
  }
}
```

------

# 6. LLM 输出设计

## 6.1 最终输出总 schema

最终决策不用工具调用，而是返回 JSON：

```json
{
  "action": "hold | rebalance",
  "portfolio_targets": [],
  "plan_updates": [],
  "memory_updates": {},
  "reason": "brief"
}
```

如果模型需要查询信息，则调用原生工具；最终轮不允许再查询。

## 6.2 portfolio_targets

```json
{
  "symbol": "2318.HK",
  "asset_type": "equity | crypto | gold_spot | oil_proxy | futures | cash",
  "target_pct_nav": 0.03,
  "target_notional_pct_nav": null,
  "side": null,
  "max_margin_pct_nav": null,
  "risk_budget_pct_nav": null,
  "priority": 1,
  "max_total_cost_bps": 35,
  "unit_hint": {
    "quantity": 1000,
    "unit": "shares | lots | coins | contracts"
  },
  "reason": "brief"
}
```

### 解释

股票、crypto、spot proxy 使用 `target_pct_nav`。

期货使用：

```json
{
  "target_notional_pct_nav": 0.25,
  "max_margin_pct_nav": 0.03,
  "risk_budget_pct_nav": 0.006
}
```

执行层根据 target 换算单位。

## 6.3 plan_updates

```json
{
  "symbol": "2318.HK",
  "plan_action": "create | update | close | no_change",
  "intended_horizon_bars": 36,
  "plan_note": "brief",
  "triggers": [
    {
      "type": "price_move_pct",
      "direction": "down",
      "anchor": "last_review_price",
      "threshold_pct": 0.01
    },
    {
      "type": "bars_elapsed",
      "since": "last_review",
      "bars": 6
    }
  ]
}
```

## 6.4 memory_updates

```json
{
  "daily_thesis": {
    "text": "HK remains preferable; add only small pullback positions.",
    "confidence": 0.60,
    "expires_in_bars": 18
  },
  "add_watch": [
    {
      "symbol": "0968.HK",
      "reason": "good trend but wait for pullback",
      "desired_condition": {
        "type": "rsi_range",
        "min": 45,
        "max": 65
      },
      "expires_in_bars": 24
    }
  ],
  "add_avoid": [
    {
      "symbol": "sh.600183",
      "reason": "recent weak exit",
      "expires_in_bars": 12
    }
  ],
  "remove_watch": [],
  "remove_avoid": []
}
```

MVP 当前版本允许 `remove_watch/remove_avoid`，但不要求模型主动使用。系统会自动处理到期、买入后 watch 转 plan、平仓后 plan 归档。

------

# 7. Plan / Trigger 设计

## 7.1 Active Plan

每个持仓最多一个 active plan。

字段：

```json
{
  "plan_id": "run_model_symbol_yyyymmdd_seq",
  "symbol": "2318.HK",
  "status": "active",
  "position_id": "position_123",
  "side": "long",
  "entry_time": "2026-01-07T01:30:00Z",
  "entry_price": 72.50,
  "current_pct_nav": 0.03,
  "entry_reason": "HK pullback continuation",
  "plan_version": 1,
  "last_review_time": "2026-01-07T01:30:00Z",
  "last_review_price": 72.50,
  "atr_at_review": 0.42,
  "peak_since_entry": 72.50,
  "peak_since_last_review": 72.50,
  "intended_horizon_bars": 36,
  "created_at": "...",
  "updated_at": "..."
}
```

## 7.2 支持的 trigger 类型

当前版本支持：

```text
price_move_pct
atr_move
pnl_pct
trailing_drawdown_pct
trailing_atr
bars_elapsed
regime_change
asset_status_change
margin_risk_change
```

### price_move_pct

```json
{
  "type": "price_move_pct",
  "direction": "down",
  "anchor": "last_review_price",
  "threshold_pct": 0.01
}
```

### atr_move

```json
{
  "type": "atr_move",
  "direction": "down",
  "anchor": "last_review_price",
  "atr_multiple": 1.0,
  "atr_source": "atr_at_review"
}
```

### pnl_pct

```json
{
  "type": "pnl_pct",
  "operator": "<=",
  "threshold_pct": -0.025
}
```

### trailing_drawdown_pct

```json
{
  "type": "trailing_drawdown_pct",
  "peak_anchor": "peak_since_entry",
  "threshold_pct": 0.015
}
```

### bars_elapsed

```json
{
  "type": "bars_elapsed",
  "since": "last_review",
  "bars": 6
}
```

## 7.3 Plan 更新规则

```text
买入成交 → 自动创建 active_plan，若模型未给 plan_update，系统用默认 plan。
full_decision 提到某持仓 → 覆盖该 symbol 的 plan_version。
focused_decision 提到某持仓 → 更新该 symbol 的 plan_version。
清仓成交 → plan status = closed，归档。
持仓消失但 plan 未关闭 → 系统强制归档并记录 state_repair。
同 symbol 多个 plan_update → 只保留 priority 最高或最后一个，记录 validation warning。
```

------

# 8. Memory 设计

## 8.1 Memory 包含内容

```text
daily_thesis
active_plans
watchlist
avoid_list
recent_activity
execution_feedback
session_summary
daily_summary
rolling_behavior_notes
```

## 8.2 daily_thesis

当天市场主线判断。只保留一个 active 版本，但所有历史版本进入日志。

更新规则：

```text
模型输出新的 daily_thesis → version + 1
prompt 只展示 latest active version
旧版本保存在 daily_thesis_versions
过期后标记 stale，不进入 prompt
```

## 8.3 watchlist

观察名单。

```text
add_watch：新增或更新同 symbol watch。
remove_watch：模型可显式移除。
expire_watch：到期自动归档。
watch_to_plan：买入成交后自动从 watchlist 转为 active_plan。
```

## 8.4 avoid_list

禁买/冷却名单。

```text
add_avoid：新增或更新同 symbol avoid。
remove_avoid：模型可显式解除。
expire_avoid：到期自动归档。
system_avoid：拒单、市场关闭、硬风险事件可自动加入。
```

## 8.5 recent_activity

进入 prompt 的 recent_activity 不等于完整日志，只是重要摘要。

选择规则：

```text
- 最近 3 条非 HOLD 决策
- 最近 2 条 focused decision
- 最近 3 条 execution feedback
- 最近 1 条 risk state change
```

## 8.6 session_summary

市场 session 收盘后生成，例如 HK close、CN close、US close。

用途：

```text
- 当前日后续决策可引用已收盘市场状态
- 下一交易日该市场开盘时引用前一 session summary
```

## 8.7 daily_summary

在 benchmark day boundary 生成，默认 UTC21:00。

用途：

```text
- 次日第一次 full decision 提供 previous_daily_summary
- 后续日内 full decision 不重复完整提供，只提供 active state
```

## 8.8 rolling_behavior_notes

保留最近 3~5 天模型行为倾向，限制最多 3 条。

例如：

```text
- Model tends to hold weak pullback trades too long.
- Avoid chasing RSI>75 names.
- Recent HK pullback setups worked better than CN overheated leaders.
```

------

# 9. Prompt 设计

## 9.1 System Prompt 完整内容

```text
You are an LLM portfolio decision module in a historical multi-market investment benchmark.

Objective:
Maximize net USD NAV over the backtest while respecting hard constraints and realistic market frictions.

You are not a broker and you do not execute orders directly.
You output portfolio targets, plan updates, and memory update proposals.
The system handles unit conversion, lot rounding, execution, settlement, market rules, FX, fees, slippage, margin, and state persistence.

Core rules:
- Never use future information.
- Only use data available at or before decision_timestamp.
- Trade only when expected edge exceeds costs.
- HOLD is valid.
- Avoid unnecessary turnover.
- Do not retry recently rejected actions without new information.
- Do not trade closed or non-tradable markets.
- Use target_pct_nav for cash equities, crypto, and spot proxies.
- Use target_notional_pct_nav, max_margin_pct_nav, and risk_budget_pct_nav for futures.
- Use tools only to query information.
- Final decision must be JSON only.
- Do not include markdown in final decision.

Tool policy:
- Use screen_universe to discover candidates outside visible buckets.
- Use query_asset before trading a symbol not shown in candidate buckets.
- Use query_position when position details or plan state are unclear.
- Use query_history only when recent context is insufficient.
- Use query_fx when currency exposure or conversion cost matters.
- Use query_futures_contract before opening or increasing futures exposure.
- query_news is disabled in the primary benchmark unless news ablation is enabled.
- Do not call tools in the final answer.

Memory policy:
- You may propose plan_updates and memory_updates in the final JSON.
- The system validates and stores memory updates.
- Natural language reasons are logs only; triggers must use structured trigger schema.
- If you continue holding a position, provide review triggers or keep existing triggers.
```

## 9.2 Full Decision Prompt 模板

```text
[OBJECTIVE]
Maximize net USD NAV in a historical backtest. Trade only when edge exceeds cost. Output final JSON only.

[DECISION_CONTEXT]
{
  "decision_type": "full_decision",
  "timestamp_utc": "{{decision_timestamp}}",
  "latest_data_utc": "{{latest_data_timestamp}}",
  "benchmark_day": "{{benchmark_day}}",
  "bar_index": {{bar_index}},
  "open_markets": {{open_markets}},
  "closed_markets": {{closed_markets}},
  "full_universe_available_by_tools": true
}

[RISK_BUDGET]
{
  "risk_mode": "{{GREEN|YELLOW|RED}}",
  "max_new_buys": {{max_new_buys}},
  "max_sells": {{max_sells}},
  "max_single_new_position_pct_nav": {{max_single_new_position_pct_nav}},
  "max_total_new_buy_pct_nav": {{max_total_new_buy_pct_nav}},
  "min_cash_pct_nav": {{min_cash_pct_nav}},
  "crypto_max_pct_nav": {{crypto_max_pct_nav}},
  "futures_margin_max_pct_nav": {{futures_margin_max_pct_nav}},
  "allow_query": true,
  "tail_session_guard": {{true|false}}
}

[MARKET_SCOREBOARD]
market|open|regime|1h_pct|1d_pct|5d_pct|breadth|volatility|warning
{{rows}}

[PORTFOLIO]
NAV_USD={{nav_usd}}
cash_pct_nav={{cash_pct_nav}}

symbol|mkt|pct_nav|pnl_pct|hold_bars|trend|rsi|sellable|plan_status|risk_note
{{position_rows}}

[MEMORY_STATE]
{
  "previous_daily_summary": {{previous_daily_summary_if_first_decision_else_null}},
  "daily_thesis": {{latest_daily_thesis}},
  "recent_activity": {{recent_activity}},
  "watchlist": {{active_watchlist}},
  "avoid_list": {{active_avoid_list}},
  "recent_feedback": {{recent_execution_feedback}},
  "rolling_behavior_notes": {{rolling_behavior_notes}}
}

[CANDIDATE_BUCKETS]

# held_positions
symbol|mkt|price|pnl_pct|trend|rsi|risk_note|suggested_action
{{held_rows}}

# exit_watch
symbol|mkt|price|pct_nav|pnl_pct|trend|rsi|reason|allowed_action
{{exit_watch_rows}}

# trend_leaders
symbol|mkt|price|score|1h_pct|1d_pct|5d_pct|rsi|trend|cost|tradable
{{trend_leader_rows}}

# pullback_continuation
symbol|mkt|price|score|1d_pct|5d_pct|rsi|trend|pullback_note|tradable
{{pullback_rows}}

# oversold_reversal
symbol|mkt|price|score|1d_pct|5d_pct|rsi|stabilization|risk_note|tradable
{{oversold_rows}}

# low_vol_defensive
symbol|mkt|price|score|20d_pct|atr_pct|drawdown|cost|tradable
{{defensive_rows}}

# crypto_candidates
symbol|price|score|1h_pct|1d_pct|5d_pct|rsi|volatility|liquidity|risk_note
{{crypto_rows}}

# futures_macro
symbol|actual_contract|price|trend|notional_per_contract|margin_per_contract|suggested_max_notional_pct_nav|risk_note
{{futures_rows}}

# blocked_or_warning
symbol|mkt|reason|allowed_action
{{blocked_rows}}

[AVAILABLE_TOOLS]
Native tools are available for querying only. Do not use tools in final decision.

[FINAL_OUTPUT_SCHEMA]
Return JSON:
{
  "action": "hold | rebalance",
  "portfolio_targets": [],
  "plan_updates": [],
  "memory_updates": {
    "daily_thesis": null,
    "add_watch": [],
    "add_avoid": [],
    "remove_watch": [],
    "remove_avoid": []
  },
  "reason": "brief"
}
```

## 9.3 Focused Position Prompt 模板

```text
[OBJECTIVE]
Handle one focused position event. Do not re-evaluate the whole market. Output JSON only.

[DECISION_CONTEXT]
{
  "decision_type": "focused_position_decision",
  "timestamp_utc": "{{decision_timestamp}}",
  "latest_data_utc": "{{latest_data_timestamp}}",
  "scope": {{symbols}},
  "open_markets": {{open_markets}},
  "closed_markets": {{closed_markets}}
}

[TRIGGER]
{
  "triggered_events": [
    {
      "symbol": "{{symbol}}",
      "priority": "{{P1|P2}}",
      "trigger_type": "{{trigger_type}}",
      "trigger_detail": {{trigger_detail}},
      "actual_value": {{actual_value}},
      "threshold": {{threshold}}
    }
  ]
}

[POSITION]
symbol|mkt|pct_nav|entry_price|current_price|pnl_pct|hold_bars|trend|rsi|sellable|tradable
{{position_rows}}

[PREVIOUS_PLAN]
{
  "symbol": "{{symbol}}",
  "entry_reason": "{{entry_reason}}",
  "last_review_time": "{{last_review_time}}",
  "last_review_price": {{last_review_price}},
  "last_review_note": "{{last_review_note}}",
  "active_triggers": {{active_triggers}},
  "intended_horizon_bars": {{intended_horizon_bars}}
}

[LOCAL_CONTEXT]
{
  "recent_bars_summary": "{{recent_bars_summary}}",
  "risk_note": "{{risk_note}}",
  "execution_feedback": {{recent_feedback_for_symbol}},
  "related_avoid_or_watch": {{related_memory_items}}
}

[ALLOWED_ACTIONS]
{{allowed_actions}}

Rules:
- Only handle the scoped symbol(s).
- Do not open unrelated new positions.
- If holding, provide updated structured triggers or keep existing triggers.
- Natural language is not a trigger.

[FINAL_OUTPUT_SCHEMA]
Return JSON:
{
  "action": "hold | rebalance",
  "portfolio_targets": [],
  "plan_updates": [],
  "memory_updates": {
    "daily_thesis": null,
    "add_watch": [],
    "add_avoid": [],
    "remove_watch": [],
    "remove_avoid": []
  },
  "reason": "brief"
}
```

## 9.4 Focused Market / Risk Prompt 模板

```text
[OBJECTIVE]
Handle a market/risk event. Do not do general stock picking unless explicitly allowed. Output JSON only.

[DECISION_CONTEXT]
{
  "decision_type": "focused_market_or_risk_decision",
  "timestamp_utc": "{{decision_timestamp}}",
  "event_type": "{{regime_change|market_close_guard|margin_warning|crypto_risk|exposure_limit}}",
  "scope_market": "{{market}}"
}

[EVENT]
{{event_json}}

[EXPOSURE]
market|exposure_pct_nav|limit_pct_nav|risk_note
{{exposure_rows}}

[RELEVANT_POSITIONS]
symbol|mkt|pct_nav|pnl_pct|trend|sellable|plan_status
{{relevant_position_rows}}

[RELEVANT_MEMORY]
{{relevant_memory_json}}

[ALLOWED_ACTIONS]
{{allowed_actions}}

[FINAL_OUTPUT_SCHEMA]
Return JSON:
{
  "action": "hold | rebalance",
  "portfolio_targets": [],
  "plan_updates": [],
  "memory_updates": {},
  "reason": "brief"
}
```

------

# 10. Execution 与状态写回

## 10.1 执行流程

```text
LLM final JSON
→ parse
→ schema validate
→ semantic validate
→ constraint check
→ market rule check
→ unit conversion
→ execution simulation
→ settlement
→ portfolio update
→ execution_feedback
→ state manager update
→ metrics update
→ append audit log
```

## 10.2 target_pct_nav 转真实单位

股票：

```text
target_value_usd = NAV * target_pct_nav
local_value = target_value_usd * USD_to_local_fx
shares_raw = local_value / price
shares_final = round_to_lot(shares_raw)
```

crypto：

```text
coin_amount = target_value_usd / price
```

期货：

```text
target_notional_usd = NAV * target_notional_pct_nav
contracts_raw = target_notional_usd / notional_per_contract
contracts_final = floor_to_integer(contracts_raw)
margin_check
risk_budget_check
```

## 10.3 执行反馈

```json
{
  "symbol": "2318.HK",
  "requested_target_pct_nav": 0.03,
  "filled_target_pct_nav": 0.028,
  "status": "ADJUSTED",
  "reason": "HK board lot rounding",
  "fees_usd": 12.3,
  "slippage_usd": 8.1
}
```

状态写回必须使用真实 filled 结果。

------

# 11. 数据库与审计设计

## 11.1 基本原则

```text
所有过程完整记录。
current state 用于运行。
append-only log 用于回放与分析。
任何状态覆盖都必须保留历史版本。
```

## 11.2 核心表

### benchmark_runs

```sql
CREATE TABLE benchmark_runs (
  run_id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL,
  model_version TEXT NOT NULL,
  benchmark_id TEXT NOT NULL,
  dataset_version TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  tool_version TEXT NOT NULL,
  initial_nav_usd NUMERIC NOT NULL,
  start_time TIMESTAMP NOT NULL,
  end_time TIMESTAMP,
  config_json JSONB NOT NULL,
  created_at TIMESTAMP NOT NULL
);
```

### decision_events

```sql
CREATE TABLE decision_events (
  event_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  decision_timestamp TIMESTAMP NOT NULL,
  decision_type TEXT NOT NULL,
  prompt_hash TEXT NOT NULL,
  prompt_snapshot TEXT NOT NULL,
  tool_schema_hash TEXT,
  raw_model_output TEXT,
  parsed_output JSONB,
  validation_result JSONB,
  execution_result JSONB,
  state_diff JSONB,
  token_usage JSONB,
  latency_ms INTEGER,
  created_at TIMESTAMP NOT NULL
);
```

### tool_calls

```sql
CREATE TABLE tool_calls (
  tool_call_id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  decision_timestamp TIMESTAMP NOT NULL,
  tool_name TEXT NOT NULL,
  tool_args JSONB NOT NULL,
  tool_result JSONB NOT NULL,
  result_hash TEXT NOT NULL,
  latency_ms INTEGER,
  created_at TIMESTAMP NOT NULL
);
```

### portfolio_snapshots

```sql
CREATE TABLE portfolio_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  nav_usd NUMERIC NOT NULL,
  cash_usd_equiv NUMERIC NOT NULL,
  cash_by_currency JSONB NOT NULL,
  exposure_by_market JSONB NOT NULL,
  exposure_by_asset_type JSONB NOT NULL,
  gross_exposure_pct_nav NUMERIC,
  net_exposure_pct_nav NUMERIC,
  drawdown_pct NUMERIC,
  created_at TIMESTAMP NOT NULL
);
```

### positions

```sql
CREATE TABLE positions (
  position_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  asset_type TEXT NOT NULL,
  market TEXT NOT NULL,
  quantity NUMERIC NOT NULL,
  avg_entry_price NUMERIC NOT NULL,
  currency TEXT NOT NULL,
  current_price NUMERIC,
  market_value_usd NUMERIC,
  pct_nav NUMERIC,
  unrealized_pnl_usd NUMERIC,
  unrealized_pnl_pct NUMERIC,
  sellable_quantity NUMERIC,
  frozen_quantity NUMERIC,
  status TEXT NOT NULL,
  opened_at TIMESTAMP,
  updated_at TIMESTAMP
);
```

### orders

```sql
CREATE TABLE orders (
  order_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  decision_timestamp TIMESTAMP NOT NULL,
  symbol TEXT NOT NULL,
  asset_type TEXT NOT NULL,
  target_pct_nav NUMERIC,
  target_notional_pct_nav NUMERIC,
  requested_quantity NUMERIC,
  final_quantity NUMERIC,
  side TEXT,
  status TEXT NOT NULL,
  reject_reason TEXT,
  adjust_reason TEXT,
  created_at TIMESTAMP NOT NULL
);
```

### fills

```sql
CREATE TABLE fills (
  fill_id TEXT PRIMARY KEY,
  order_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  fill_timestamp TIMESTAMP NOT NULL,
  quantity NUMERIC NOT NULL,
  fill_price NUMERIC NOT NULL,
  currency TEXT NOT NULL,
  fill_value_usd NUMERIC NOT NULL,
  commission_usd NUMERIC,
  slippage_usd NUMERIC,
  market_fee_usd NUMERIC,
  fx_fee_usd NUMERIC,
  created_at TIMESTAMP NOT NULL
);
```

### active_plans

```sql
CREATE TABLE active_plans (
  plan_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  position_id TEXT,
  status TEXT NOT NULL,
  side TEXT,
  entry_time TIMESTAMP,
  entry_price NUMERIC,
  current_pct_nav NUMERIC,
  entry_reason TEXT,
  plan_version INTEGER NOT NULL,
  last_review_time TIMESTAMP,
  last_review_price NUMERIC,
  atr_at_review NUMERIC,
  peak_since_entry NUMERIC,
  peak_since_last_review NUMERIC,
  intended_horizon_bars INTEGER,
  plan_note TEXT,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
);
```

### plan_versions

```sql
CREATE TABLE plan_versions (
  plan_version_id TEXT PRIMARY KEY,
  plan_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  source_event_id TEXT NOT NULL,
  plan_snapshot JSONB NOT NULL,
  created_at TIMESTAMP NOT NULL
);
```

### plan_triggers

```sql
CREATE TABLE plan_triggers (
  trigger_id TEXT PRIMARY KEY,
  plan_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  status TEXT NOT NULL,
  trigger_type TEXT NOT NULL,
  trigger_json JSONB NOT NULL,
  created_at TIMESTAMP NOT NULL,
  expires_at TIMESTAMP,
  triggered_at TIMESTAMP,
  archived_at TIMESTAMP
);
```

### watchlist_items

```sql
CREATE TABLE watchlist_items (
  watch_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  status TEXT NOT NULL,
  reason TEXT,
  desired_condition_json JSONB,
  source_event_id TEXT,
  created_at TIMESTAMP NOT NULL,
  expires_at TIMESTAMP,
  archived_at TIMESTAMP
);
```

### avoid_items

```sql
CREATE TABLE avoid_items (
  avoid_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  status TEXT NOT NULL,
  reason TEXT,
  source_event_id TEXT,
  created_at TIMESTAMP NOT NULL,
  expires_at TIMESTAMP,
  archived_at TIMESTAMP
);
```

### daily_thesis_versions

```sql
CREATE TABLE daily_thesis_versions (
  thesis_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  benchmark_day DATE NOT NULL,
  version INTEGER NOT NULL,
  text TEXT NOT NULL,
  confidence NUMERIC,
  source_event_id TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL,
  expires_at TIMESTAMP
);
```

### summaries

```sql
CREATE TABLE summaries (
  summary_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  summary_type TEXT NOT NULL,
  market TEXT,
  benchmark_day DATE,
  source_start TIMESTAMP NOT NULL,
  source_end TIMESTAMP NOT NULL,
  summary_json JSONB NOT NULL,
  summarizer_model TEXT,
  prompt_hash TEXT,
  created_at TIMESTAMP NOT NULL
);
```

### metrics_daily

```sql
CREATE TABLE metrics_daily (
  metrics_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  benchmark_day DATE NOT NULL,
  nav_start_usd NUMERIC,
  nav_end_usd NUMERIC,
  daily_return_pct NUMERIC,
  max_drawdown_pct NUMERIC,
  turnover_pct NUMERIC,
  fees_usd NUMERIC,
  slippage_usd NUMERIC,
  rejected_orders INTEGER,
  adjusted_orders INTEGER,
  constraint_hits INTEGER,
  tool_calls INTEGER,
  pnl_by_market JSONB,
  pnl_by_asset_type JSONB,
  pnl_by_symbol JSONB,
  attribution_json JSONB,
  created_at TIMESTAMP NOT NULL
);
```

------

# 12. SummaryEngine

## 12.1 Session Summary

触发：

```text
某市场 session close + settlement update completed
```

输入：

```text
该市场 session 行情摘要
该市场相关 portfolio changes
该市场相关 decision_events
该市场相关 execution_feedback
该市场仍持有 active_plans
```

输出：

```json
{
  "summary_type": "session",
  "market": "HK",
  "session_date": "2026-01-07",
  "market_read": "HK opened strong but faded near close.",
  "model_actions": [
    "held 1378.HK",
    "opened 2318.HK"
  ],
  "open_positions": [
    {"symbol": "1378.HK", "plan": "profit carry with trailing stop"}
  ],
  "risk_notes": [
    "Avoid new HK buys near close."
  ]
}
```

## 12.2 Daily Global Summary

触发：

```text
benchmark_boundary = UTC21:00
```

crypto 不停市，因此按 benchmark boundary 截取快照。

输入：

```text
全天所有 decision_events
全天所有 tool_calls
全天 execution logs
全天 portfolio snapshots
全天 metrics
当日 session summaries
未平仓 active_plans
```

输出：

```json
{
  "summary_type": "daily_global",
  "date": "2026-01-07",
  "nav_start": 1000000,
  "nav_end": 1003200,
  "daily_return_pct": 0.0032,
  "market_read": "HK contributed most gains; crypto weak; US not traded.",
  "major_decisions": [
    "Closed weak CN position.",
    "Opened HK pullback candidate.",
    "Trimmed profit on 1378.HK."
  ],
  "what_worked": [
    "Exit-watch reduced weak CN exposure."
  ],
  "what_failed_or_uncertain": [
    "0968.HK required focused review after entry."
  ],
  "carryover_positions": [
    {
      "symbol": "1378.HK",
      "plan": "profit carry with trailing stop"
    }
  ],
  "avoid_next_day": [
    {
      "symbol": "sh.600183",
      "reason": "recent weak exit"
    }
  ],
  "behavior": {
    "queries": 2,
    "trades": 3,
    "rejected_orders": 0,
    "adjusted_orders": 1,
    "turnover_level": "moderate"
  }
}
```

## 12.3 总结模型

主 benchmark 使用统一 summarizer：

```text
summarizer_model 固定
temperature = 0
固定 prompt
固定 schema
所有被测模型共用同一个 summarizer
```

不让交易模型自己总结主榜记忆。交易模型自我总结可作为单独消融实验。

------

# 13. 新闻模块分析

## 13.1 当前版本结论

当前主 benchmark 不纳入新闻模块。保留 query_news 接口，但默认 disabled。

## 13.2 为什么暂不上新闻

新闻可能有价值，但有较高实现风险：

```text
1. 发布时间与可用时间容易泄露未来。
2. 新闻摘要可能隐含未来价格反应。
3. 数据源覆盖不同市场不均。
4. 中英文新闻对不同模型不公平。
5. 新闻噪音会显著增加交易频率。
6. 新闻模块贡献难以归因。
```

## 13.3 新闻可能有益的场景

```text
财报
业绩预告
监管政策
并购
回购
分红
重大诉讼
产品事件
宏观数据
商品供需事件
crypto 监管事件
```

## 13.4 后续新闻实验设计

如果后续开启新闻，必须做消融：

```text
A. price-only
B. price + news digest
C. price + query_news only
D. price + news digest + query_news
E. delayed_news_24h
F. shuffled_news
```

只有当 B/C/D 明显优于 A，并且 E/F 收益消失，才说明新闻有真实贡献。

------

# 14. 指标与归因

## 14.1 主榜指标

```text
Profit Leaderboard:
- Total Return
```

## 14.2 综合榜指标

```text
Composite:
- Return
- Sharpe
- Max Drawdown
- Stability
- Efficiency
- Discipline
```

## 14.3 行为指标

```text
constraint_hits
rejected_orders
adjusted_orders
tool_usage
turnover
unit_hint_mismatch
margin_warning_count
forced_liquidation_count
watchlist_hit_rate
avoid_violation_count
plan_follow_rate
focused_decision_count
```

## 14.4 PnL Attribution

```json
{
  "total_pnl_usd": 12345,
  "by_market": {
    "US": 1000,
    "CN": -300,
    "HK": 2200,
    "CRYPTO": 500,
    "FUTURES": 0,
    "FX_TRANSLATION": 50
  },
  "fees_slippage_fx": -120,
  "cash_drag": -80,
  "rejected_order_effect": -30,
  "top_symbol_pnl_share": 0.35,
  "top_market_pnl_share": 0.62
}
```

## 14.5 记忆归因

新增分析：

```text
某笔交易是否来自：
- candidate bucket
- screen_universe
- watchlist
- focused decision
- previous daily summary
- active plan continuation
```

这样可以分析“记忆模块”是否真的改善了稳定性。

------

# 15. Replay 与可复现

每次运行必须保存：

```text
dataset_version
prompt_version
tool_version
model_id
model_version
temperature
seed / sampling config
decision_timestamp
latest_data_timestamp
prompt_snapshot
tool_schema
tool_results
raw_model_output
parsed_decision
execution_result
state_diff
portfolio_snapshot
metrics
```

Replay 方式：

```text
按 decision_events 时间排序
重放 tool_results
重放 parsed_decision
重放 execution_result
重建 state
对比重建 state 与 current tables
```

如果 replay state 与 current state 不一致，标记 replay_error。

------

# 16. 当前版本的完整运行链路

```text
1. 读取 historical data。
2. 每 5min 更新 market / portfolio / features。
3. Candidate Builder 生成分桶候选。
4. StateManager 过期 watch/avoid，更新 active plan runtime fields。
5. Scanner 检查 triggers。
6. Scheduler 判断：
   - 无事件且未到 full decision → auto_hold
   - 到 full decision → build full prompt
   - 有 focused event → build focused prompt
7. LLM 可调用查询工具。
8. LLM 最终输出 JSON。
9. Parser/Validator 校验。
10. Execution Engine 执行 portfolio_targets。
11. StateManager 应用 plan_updates / memory_updates / execution_feedback。
12. MetricsEngine 更新收益、行为、归因。
13. AuditStore 写入所有输入输出和 state_diff。
14. Session close 生成 session_summary。
15. Benchmark boundary 生成 daily_global_summary。
16. 次日第一次 full decision 注入 previous_daily_summary + carryover active plans。
```

------

# 17. 关键设计边界

```text
模型可以：
- 查询信息
- 提出交易目标
- 提出计划更新
- 提出观察/禁买/每日观点更新

模型不可以：
- 直接执行交易
- 直接改数据库
- 直接删除审计日志
- 用自然语言触发条件
- 绕过市场规则
- 查看未来数据

系统负责：
- 查询工具执行
- 交易执行
- 规则校验
- 记忆落库
- 状态过期
- 总结生成
- 指标归因
- replay 审计
```