# Cross-Market LLM Trading Benchmark — 系统说明

## 1. 系统概述

本系统是一个**可复现、可扩展的多市场 LLM 投资决策 Benchmark**，用于评估不同大语言模型在真实历史市场环境下的投资决策能力。

系统模拟真实交易环境，支持 6 个市场（US/HK/CN/Crypto/Gold/Futures），包含完整的交易引擎、风控系统、记忆机制和决策调度。

### 核心特性

- **多市场覆盖**：美股、港股、A股、加密货币、黄金现货、期货
- **真实摩擦模拟**：佣金、滑点、印花税、外汇费用、T+1 结算
- **完整决策链路**：数据 → 候选筛选 → LLM 决策 → 交易执行 → 风控 → 评估
- **记忆系统**：持仓计划、触发器、观察列表、回避列表、执行反馈
- **断点续跑**：checkpoint 机制支持中断恢复
- **配置驱动**：所有阈值、参数均通过 TOML 配置

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    ExperimentRunner                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Scheduler │  │ EventDet │  │ Context  │  │ Agent   │ │
│  │ 决策调度  │→│ 事件检测  │→│ Prompt   │→│ LLM调用 │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│       ↑              ↑             ↑            ↓       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Memory   │  │ Trigger  │  │ Features │  │Protocol │ │
│  │ 记忆管理  │  │ 触发引擎  │  │ 指标计算  │  │ 协议解析│ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│       ↑              ↑             ↑            ↓       │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Portfolio + Execution                │   │
│  │    组合管理 + 交易执行 + 结算 + 约束 + 市场规则    │   │
│  └──────────────────────────────────────────────────┘   │
│       ↑                                                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │              MarketDataProvider                   │   │
│  │         历史行情 + 特征缓存 + 汇率                  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 决策流程

### 3.1 决策类型

| 类型 | 触发条件 | LLM 轮次 | 说明 |
|------|---------|---------|------|
| `auto_hold` | 无开放市场 | 0 | 系统自动记录 hold |
| `light_decision` | 整点 + 仅24h市场 | 1 | 只看加密/黄金，不买新股 |
| `full_decision` | 半整点/开盘窗口/收盘窗口 | 3 | 全市场扫描 + 候选筛选 |
| `focused_position` | 持仓触发器触发 | 2 | 单个持仓风险审查 |

### 3.2 调度优先级

```
1. 收盘窗口（闭市前30分钟，每15分钟）→ full_decision
2. 开盘窗口（开盘后30分钟，每15分钟）→ full_decision
3. 常规调度（每30分钟）→ full_decision
4. P1 触发器 → focused_position（仅非半整点时间）
5. P2 触发器 → focused_position
6. auto_hold
```

**关键设计**：半整点时间优先执行 full_decision，触发器信息合并到 full decision 的 `[TRIGGER_ALERTS]` 中，避免 focused 替代 full 导致全市场 review 缺失。

### 3.3 决策输出格式

```json
{
  "action": "rebalance",
  "portfolio_targets": [
    {"symbol": "AAPL.US", "target_pct_nav": 0.05, "side": "long"}
  ],
  "plan_updates": [
    {
      "symbol": "AAPL.US",
      "plan_action": "update",
      "triggers": [{"trigger_type": "pnl_pct", "operator": "<=", "threshold_pct": -0.03}]
    }
  ],
  "memory_updates": {
    "daily_thesis": "市场整体偏多",
    "add_watch": [{"symbol": "TSLA.US", "reason": "RSI recovering"}]
  },
  "reason": "AAPL setup strong, RSI 45, trend UU"
}
```

---

## 4. 记忆系统

### 4.1 组件总览

| 组件 | 用途 | 写入方式 | 触发机制 |
|------|------|---------|---------|
| `active_plans` | 持仓交易计划 + 触发器 | 系统自动 + 模型 | 9种 trigger |
| `daily_thesis` | 每日市场观点 | 模型 | 无 |
| `watchlist` | 观察列表（未持仓） | 模型 | 无（仅上下文） |
| `avoid_list` | 回避/冷却列表 | 模型 + 止损 | 冷却期自动过期 |
| `execution_feedback` | 交易执行结果 | 系统自动 | 无 |
| `recent_activity` | 最近决策摘要 | 系统自动 | 无 |

### 4.2 Plan Trigger 机制

每个持仓在 BUY 成功后自动创建 `ActivePlan`，带一个 `pnl_pct` 触发器（默认 -3%）。

**9 种触发器类型**：

| 触发器 | 说明 | 冷却 |
|--------|------|------|
| `pnl_pct` | 持仓盈亏百分比 | 6 bars（30分钟） |
| `trailing_drawdown_pct` | 从最高点回落 | 6 bars |
| `price_move_pct` | 价格从锚点变动 | 6 bars |
| `atr_move` | ATR 倍数变动 | 6 bars |
| `trailing_atr` | ATR 倍数回撤 | 6 bars |
| `bars_elapsed` | 距上次 review 的 bar 数 | 无（本身就是冷却机制） |
| `regime_change` | 市场 regime 变化 | 无 |
| `asset_status_change` | 资产变为不可交易 | 无 |
| `margin_risk_change` | 期货保证金风险 | 无 |

**冷却机制**：
- 触发器触发后，30 分钟内不重复触发
- 如果模型在 focused decision 中修改了触发器阈值，跳过冷却（视为新决策）

**阈值验证**：
- `trailing_drawdown_pct` 强制为正数（模型设负数自动修正）
- 所有阈值 clamp 到配置范围（可配置）

### 4.3 Plan 生命周期

```
BUY 成功
  → 系统自动创建 plan（pnl_pct trigger -3%）
  → plan 存入 memory._plans

PnL ≤ -3%（6 bars 后）
  → trigger 触发 → focused decision
  → 模型决定 hold / reduce / close
  → 如果修改阈值 → 跳过冷却
  → 如果只 hold → 冷却 30 分钟

SELL 全部清仓
  → 系统自动 close plan

合约换月（期货）
  → 更新 plan entry_price
```

---

## 5. 风控系统

### 5.1 双层止损

| 层级 | 阈值（股票/期货） | 阈值（Crypto） | 机制 |
|------|-----------------|---------------|------|
| Plan trigger（预警） | -3% | -5% | 触发 focused decision，模型决定 |
| Hard stop（强制） | -5% | -8% | 系统直接卖出，不经过 LLM |

**流程**：
```
PnL -3% → plan trigger → focused decision → 模型可能 hold
PnL -5% → hard stop → 系统强制卖出 → 买入冷却期
```

### 5.2 止损后行为

- 自动加入 `avoid_list`（冷却期）
- 同市场暂停新买入（60-180分钟）
- 下次 full decision 可以看到止损信息（`trade_feedback`）

### 5.3 所有阈值均可配置

```toml
[trigger]
pnl_pct_threshold = -0.03
trailing_drawdown_pct = 0.02
cooldown_bars = 6

[trigger.limits]
pnl_pct_min = -0.10
pnl_pct_max = 0.20
trailing_drawdown_min = 0.001
trailing_drawdown_max = 0.20

[stop_loss]
hard_stop_pct = -0.05
crypto_hard_stop_pct = -0.08
```

---

## 6. 数据层

### 6.1 数据源配置

数据来自 `getStockData` 项目（5分钟K线，SQLite 格式）。在 TOML 中配置：

```toml
[data]
base_dir = "D:/Projects/claw/getStockData"
```

系统自动拼接路径：`{base_dir}/data/{market}_stock.db`。

### 6.2 行情数据库

| 市场 | 数据库 | 表名 | 频率 |
|------|--------|------|------|
| US | US_stock.db | us_5min | 5分钟 |
| HK | HK_stock.db | hk_5min | 5分钟 |
| CN | A_stock.db | stock_5min | 5分钟 |
| CRYPTO | CRYPTO_stock.db | crypto_5min | 5分钟 |
| GOLD | GOLD_stock.db | gold_5min | 5分钟 |
| FUTURES | FUTURES_stock.db | futures_5min | 5分钟 |

### 6.3 Derived Cache（期货需要）

期货回测需要预构建 derived cache，包含合约解析和特征数据：

```bash
# 构建缓存（指定时间范围）
python scripts/build_derived_cache.py --start 2026-01-20 --end 2026-02-28
```

- 输出：`artifacts/cache/derived_<dataset_version>.db`
- 回测时自动读取，只读不影响结果
- 缓存按 `(dataset_version, namespace, symbol, timestamp)` 键存储
- 构建过程中会预加载历史K线到内存，加速后续解析

### 6.3 特征计算

每个 5 分钟 bar 计算：
- RSI（14期）
- ATR（14期）
- EMA（9/21期）
- 布林带位置
- 相对成交量
- 1h/1d 变动
- 趋势模式（6 bar）
- Setup 分类

---

## 7. 交易引擎

### 7.1 支持的市场

| 市场 | 交易时间（UTC） | 结算 | 特殊规则 |
|------|---------------|------|---------|
| US | 14:30-21:00 | T+1 | - |
| HK | 01:30-08:00 | T+1 | - |
| CN | 01:30-07:00 | T+1 | 涨跌停、印花税 |
| CRYPTO | 24/7 | T+0 | - |
| GOLD | 24/7 | T+0 | 点差 |
| FUTURES | 数据驱动 | T+0 | 保证金、换月 |

### 7.2 成本模型

- 佣金：按市场配置（bps）
- 滑点：按市场配置（bps）
- A股印花税：卖出 5bps
- 外汇费用：5bps
- 期货：按合约收费

### 7.3 约束系统

- 单个持仓上限：25% NAV
- 单个市场暴露上限：50% NAV
- 加密货币暴露上限：25% NAV
- 最低现金储备：5% NAV
- 期货保证金上限：10% NAV

---

## 8. 期货支持

### 8.1 支持的品种

| Family | 标准合约 | 微型合约 | 角色 |
|--------|---------|---------|------|
| GOLD_FUT | GC | MGC | 黄金/避险 |
| OIL_FUT | CL | MCL | 原油/通胀 |
| SP500_FUT | ES | MES | 美股大盘 |
| NASDAQ_FUT | NQ | MNQ | 美股科技 |
| RUSSELL_FUT | RTY | M2K | 美股小盘 |
| UST10Y_FUT | ZN | - | 美债 |
| EUR_FX_FUT | 6E | - | 欧元汇率 |
| JPY_FX_FUT | 6J | - | 日元汇率 |
| BTC_FUT | BTC | - | 比特币 |

### 8.2 期货 Plan 管理

- Plan key 使用 family symbol（如 `GOLD_FUT`）
- Position 使用 contract symbol（如 `GCQ5.CM`）
- 通过 `futures_family_variants` 映射 root symbol
- 合约换月自动更新 plan entry_price
- Checkpoint 恢复时自动补建缺失的 plan

---

## 9. Prompt 结构

### 9.1 Full Decision（4层结构）

```
[MARKET_RULES] — 交易规则（稳定）
[MARKET_SUMMARY] — 市场概览
[OPEN MARKETS] — 开放市场
[PORTFOLIO] — 持仓 + NAV + 风险状态
[MEMORY_STATE] — 记忆状态（thesis/watchlist/avoid/feedback/active_plans）
[TRIGGER_ALERTS] — 触发器告警（如有）
[CANDIDATES] — 候选股票（分桶排序）
[DECISION_CONTEXT] — 决策上下文
[ROUND] — 轮次指示
```

### 9.2 Focused Decision

```
[OBJECTIVE] — 处理一个/多个持仓事件
[DECISION_CONTEXT] — 决策类型 + 时间
[POSITION N] — 持仓信息 + RSI/ATR/趋势
[PREVIOUS_PLAN] — 之前的计划
[ALLOWED_POSITION_ACTIONS] — 允许的操作
[OUTPUT_ACTION] — 输出格式说明
```

---

## 10. 评估指标

| 指标 | 说明 |
|------|------|
| Total Return | 总收益率 |
| Sharpe Ratio | 夏普比率 |
| Max Drawdown | 最大回撤 |
| Win Rate | 胜率 |
| Total Trades | 交易次数 |
| Rejection Rate | 订单拒绝率 |
| LLM Calls | LLM 调用次数 |
| Latency | 决策延迟 |

---

## 11. 配置文件结构

```
config/
├── config.toml        # 主配置（含 API key，不提交）
├── deepseek.toml      # DeepSeek 配置
├── mimo.toml          # Mimo 配置
└── template.toml      # 配置模板（提交）
```

### 配置段落

| 段落 | 说明 |
|------|------|
| `[data]` | 数据路径 |
| `[backtest]` | 回测参数（日期、间隔） |
| `[model]` | 模型参数（名称、温度、token） |
| `[model.api]` | API 配置（key、url） |
| `[portfolio]` | 资金 + 持仓限制 |
| `[costs]` | 成本模型 |
| `[futures]` | 期货配置 |
| `[gold]` | 黄金配置 |
| `[trigger]` | 触发器阈值 |
| `[trigger.limits]` | 触发器范围限制 |
| `[stop_loss]` | 止损阈值 |
| `[decision_schedule]` | 决策调度 |
| `[tail_guard]` | 收盘保护 |

---

## 12. 使用方法

### 12.1 完整流程

```bash
# 1. 准备数据（需要 getStockData 项目的 SQLite 数据库）
#    在 config/*.toml 中配置 [data] base_dir

# 2. 构建期货缓存（如果需要回测期货）
python scripts/build_derived_cache.py --start 2026-01-20 --end 2026-02-28

# 3. 配置 API key
cp config/template.toml config/config.toml
# 编辑 config.toml，填入 API key

# 4. 运行回测
python runners/run_backtest.py --config config/mimo.toml --start 2026-02-03 --end 2026-02-09

# 5. 查看结果
sqlite3 artifacts/runs/xxx.db "SELECT * FROM benchmark_runs;"
```

### 12.2 运行回测

```bash
# 使用 Mimo 配置，2月3-9日
python runners/run_backtest.py --config config/mimo.toml --start 2026-02-03 --end 2026-02-09

# 使用 DeepSeek 配置
python runners/run_backtest.py --config config/deepseek.toml --start 2026-02-03 --end 2026-02-09

# 自定义参数
python runners/run_backtest.py --model mimo-v2.5-pro --start 2026-02-03 --end 2026-02-09 --initial-cash 500000

# 中断恢复
python runners/run_backtest.py --resume --output artifacts/runs/xxx.db
```

### 12.2 查看结果

```bash
# 查看回测结果
sqlite3 artifacts/runs/xxx.db "SELECT * FROM benchmark_runs;"

# 查看决策分布
sqlite3 artifacts/runs/xxx.db "SELECT decision_type, COUNT(*) FROM decision_events GROUP BY decision_type;"

# 查看交易记录
sqlite3 artifacts/runs/xxx.db "SELECT * FROM trades WHERE success=1;"

# 查看止损触发
sqlite3 artifacts/runs/xxx.db "SELECT decision_timestamp, execution_result FROM decision_events WHERE execution_result LIKE '%AUTO STOP-LOSS%';"
```

### 12.3 运行测试

```bash
# 全部测试
python -m pytest tests/ -q

# 指定测试
python -m pytest tests/test_memory_plan_lifecycle.py -q
```

---

## 13. 技术栈

- **语言**：Python 3.11+
- **LLM API**：OpenAI 兼容（Mimo / DeepSeek）
- **数据库**：SQLite（行情 + 回测输出 + 缓存）
- **配置**：TOML
- **测试**：pytest
