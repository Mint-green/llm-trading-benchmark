# Cross-Market LLM Trading Benchmark

多市场 LLM 投资决策 Benchmark 系统。用于评估不同 LLM 在真实历史市场环境下的投资决策能力。

---

## 快速开始

### 1. 环境准备

```bash
# 依赖
pip install openai

# 数据源（只读）
# 需要 getStockData 项目的数据，路径在 config 中配置
# 如: D:/Projects/getStockData/data
```

### 2. 配置

```bash
# 复制模板
cp config/template.toml config/config.toml

# 编辑配置
vim config/config.toml
```

配置文件说明：
- `config/template.toml` — 配置模板（提交到 git）
- `config/config.toml` — 用户配置（不提交，含 API key）
- `config/deepseek.toml` — DeepSeek 专用配置（可选）
- `config/mimo.toml` — MIMO 专用配置（可选）

### 3. 运行回测

```bash
# 使用默认配置
python runners/run_backtest.py

# 使用指定配置
python runners/run_backtest.py --config config/deepseek.toml

# 自定义参数
python runners/run_backtest.py --model deepseek-v4-pro --start 2026-02-03 --end 2026-02-09 --interval 60

# 从上次中断处继续
python runners/run_backtest.py --resume
```

### 4. 查看结果

```bash
# 查看数据库
sqlite3 output/results/benchmark.db "SELECT * FROM benchmark_runs;"
sqlite3 output/results/benchmark.db "SELECT * FROM decisions;"
sqlite3 output/results/benchmark.db "SELECT * FROM trades;"
```

---

## 项目结构

```
llm-trading-benchmark/
├── .gitignore
├── config/
│   ├── template.toml          # 配置模板（提交）
│   ├── config.toml            # 用户配置（不提交）
│   ├── deepseek.toml          # DeepSeek 配置（不提交）
│   └── mimo.toml              # MIMO 配置（不提交）
├── runners/
│   └── run_backtest.py        # 主入口（CLI）
├── src/
│   ├── core/                  # 类型、配置、接口
│   │   ├── config.py
│   │   ├── interfaces.py
│   │   └── types.py
│   ├── data/                  # 数据层
│   │   ├── provider.py
│   │   ├── universe.py
│   │   ├── features.py
│   │   ├── screener.py
│   │   ├── fx_provider.py
│   │   └── index_provider.py
│   ├── portfolio/             # 交易引擎
│   │   ├── portfolio.py
│   │   ├── nav.py
│   │   ├── constraints.py
│   │   ├── market_rules.py
│   │   ├── execution.py
│   │   └── settlement.py
│   ├── agent/                 # LLM Agent
│   │   ├── context.py
│   │   ├── tools.py
│   │   ├── protocol.py
│   │   └── runner.py
│   ├── evaluation/            # 评估
│   │   ├── metrics.py
│   │   └── behavior.py
│   ├── platform/              # 平台
│   │   ├── experiment.py
│   │   └── logging.py
│   └── prompts/               # Prompt 配置
│       └── active/
│           └── prompts.py
├── output/                    # 回测输出（不提交）
│   └── results/
│       └── benchmark.db       # 统一数据库
├── docs/
├── requirements.txt
└── README.md
```

---

## 配置说明

### 模型配置

```toml
[model]
name = "deepseek-v4-pro"  # 模型名称：mimo-v2.5-pro / deepseek-v4-pro
temperature = 0.3          # 温度（0.0-1.0）
thinking_enabled = false   # 是否开启思考模式
max_tokens = 4096          # 最大 token 数
timeout = 180              # 超时时间（秒）
```

### 回测配置

```toml
[backtest]
start = "2026-02-03"       # 开始日期
end = "2026-02-09"         # 结束日期
decision_interval = 60     # 决策间隔（分钟）
max_decisions = 0          # 最大决策数（0=不限制）
```

### 资金配置

```toml
[portfolio]
initial_cash = 100000      # 初始资金（USD）

[portfolio.position_limits]
max_single = 0.25          # 单个持仓上限（25% NAV）
max_market = 0.50          # 单个市场暴露上限（50% NAV）
max_crypto = 0.25          # 加密货币暴露上限（25% NAV）
min_cash = 0.05            # 最低现金储备（5% NAV）
```

### 成本配置

```toml
[costs]
cn_sell_tax_bps = 5        # A股卖出印花税（bps）
fx_fee_bps = 5             # 外汇兑换费用（bps）

[costs.commission_bps]
US = 3                     # 美股佣金（bps）
HK = 5                     # 港股佣金（bps）
CN = 3                     # A股佣金（bps）
CRYPTO = 10                # 加密货币佣金（bps）

[costs.slippage_bps]
US = 5                     # 美股滑点（bps）
HK = 5                     # 港股滑点（bps）
CN = 5                     # A股滑点（bps）
CRYPTO = 10                # 加密货币滑点（bps）
```

---

## CLI 接口

```bash
python runners/run_backtest.py [OPTIONS]

Options:
  --model TEXT          模型名称 (mimo-v2.5-pro | deepseek-v4-pro)
  --start TEXT          开始日期 (YYYY-MM-DD)
  --end TEXT            结束日期 (YYYY-MM-DD)
  --interval INT        决策间隔（分钟）
  --initial-cash FLOAT  初始资金
  --max-decisions INT   最大决策数（0=不限制）
  --thinking            开启思考模式
  --config PATH         配置文件路径（默认 config/config.toml）
  --output PATH         输出数据库路径（默认 output/results/benchmark.db）
  --resume              从上次中断处继续
```

---

## 数据库结构

每个回测生成记录到统一的 SQLite 数据库 `output/results/benchmark.db`：

```sql
benchmark_runs       -- 实验元数据（run_id, model, status, progress）
decisions            -- 决策记录 (timestamp, action, trades, reason, nav)
trades               -- 交易记录 (symbol, market, side, quantity, price, cost, fees, success, error)
portfolio_snapshots  -- 组合快照 (cash, nav, positions, exposure)
agent_rounds         -- Agent 轮次
llm_calls            -- LLM 调用详情 (prompt_tokens, completion_tokens, latency, reasoning, response)
```

---

## 进度输出

每个决策点输出一行结构化日志：

```
[2026-02-03 05:00] decision=5/77 | calls=2 | latency=8.3s | action=trade | trades=BUY 0700.HK(5%), SELL 0981.HK | NAV=$102,345
[2026-02-03 06:00] decision=6/77 | calls=1 | latency=4.2s | action=hold | NAV=$102,345
```

格式：`[{timestamp}] decision={n}/{total} | calls={n} | latency={s}s | action={type} | trades={details} | NAV=${value}`

---

## 测试管理

### 查看测试状态

```bash
sqlite3 output/results/benchmark.db "
SELECT run_id, model, status, decisions_made, current_nav, total_trades
FROM benchmark_runs
ORDER BY created_at DESC;
"
```

### 继续未完成的测试

```bash
python runners/run_backtest.py --resume
```

---

## 历史测试结果

### 2026-02-03 ~ 02-09（无思考，1h 粒度）

| 模型 | 收益率 | 交易笔数 | LLM调用 | 费用 |
|------|--------|----------|---------|------|
| deepseek-v4-pro | +1.48% | 118 | 130 | $1,441 |
| mimo-v2.5-pro | -0.46% | 45 | 88 | $230 |

---

## 常见问题

### Q: 如何添加新的 LLM 模型？

在 `config/template.toml` 中添加新的 API 配置，然后在 `src/agent/runner.py` 中添加模型选择逻辑。

### Q: 如何修改候选数量？

在 `src/data/screener.py` 中修改 `market_quotas` 配置。

### Q: 如何查看模型的 reasoning？

```bash
sqlite3 output/results/benchmark.db "
SELECT reasoning FROM llm_calls 
WHERE run_id = 'xxx' AND decision_timestamp = '2026-02-03 05:00';
"
```

### Q: 为什么有些决策点被跳过？

当没有任何股市开放时（如凌晨只有 Crypto），决策点会被跳过。见 `src/platform/experiment.py` 的 `_any_stock_market_open` 方法。


