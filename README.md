# Cross-Market LLM Trading Benchmark

多市场 LLM 投资决策 Benchmark 系统。评估不同大语言模型在真实历史市场环境下的投资决策能力。

支持 6 个市场：美股、港股、A股、加密货币、黄金现货、期货。

---

## 快速开始

### 1. 安装依赖

```bash
pip install openai
```

### 2. 准备数据

行情数据来自 `getStockData` 项目（5分钟K线，SQLite 格式）。在配置中指定数据目录：

```toml
[data]
base_dir = "D:/Projects/claw/getStockData"
```

目录下需包含：`US_stock.db`、`HK_stock.db`、`A_stock.db`、`CRYPTO_stock.db`、`GOLD_stock.db`、`FUTURES_stock.db`。

### 3. 构建缓存（期货需要）

期货回测需要先构建 derived cache（合约解析 + 特征预计算）：

```bash
python scripts/build_derived_cache.py --start 2026-01-20 --end 2026-02-28
```

缓存文件生成在 `artifacts/cache/derived_<dataset_version>.db`，回测时自动读取。

### 4. 配置

```bash
# 复制模板
cp config/template.toml config/config.toml

# 编辑配置，填入 API key
vim config/config.toml
```

### 5. 运行回测

```bash
# Mimo 7天回测
python runners/run_backtest.py --config config/mimo.toml --start 2026-02-03 --end 2026-02-09

# DeepSeek 7天回测
python runners/run_backtest.py --config config/deepseek.toml --start 2026-02-03 --end 2026-02-09
```

### 6. 中断恢复与续跑

支持两种续跑模式：

**`--resume`：从中断处继续**

程序异常退出后，从最后一个 checkpoint 恢复，继续跑完剩余时间段。

```bash
python runners/run_backtest.py --config config/mimo.toml --output artifacts/runs/xxx.db --resume
```

**`--extend-end`：延长回测时间**

回测完成后，延长结束日期继续跑。所有数据写入同一个数据库，checkpoint 自动衔接。

```bash
# 原来跑 1/5-1/9，延长到 1/15
python runners/run_backtest.py --config config/mimo.toml --start 2026-01-05 --output artifacts/runs/xxx.db --extend-end 2026-01-15
```

注意事项：
- `--resume` 要求代码版本一致（code_version 匹配）
- `--extend-end` 允许代码版本不同（只延续 end_date）
- 旧版本代码生成的 checkpoint 可以用新版本代码续跑（自动处理 enum 兼容）
- 必须指定 `--output` 指向已有的数据库文件
- `--start` 需要和原始运行一致

### 7. 查看结果

```bash
sqlite3 artifacts/runs/xxx.db "SELECT * FROM benchmark_runs;"
```

---

## CLI 参数

```
python runners/run_backtest.py [OPTIONS]

--config PATH              配置文件路径（默认 config/config.toml）
--model TEXT               模型名称（覆盖配置文件）
--start TEXT               开始日期 (YYYY-MM-DD)
--end TEXT                 结束日期 (YYYY-MM-DD)
--interval INT             决策间隔（分钟）
--initial-cash FLOAT       初始资金
--max-decisions INT        最大决策数（0=不限制）
--max-rounds INT           每决策最大 LLM 轮次
--thinking                 开启思考模式
--output PATH              输出数据库路径
--resume                   从上次中断处继续（需代码版本一致）
--extend-end TEXT          延长回测到新结束日期（允许代码版本不同）
--fork-from-checkpoint DB  从另一个运行的 checkpoint 分叉
```

---

## 项目结构

```
llm-trading-benchmark/
├── config/                  # 配置文件
│   ├── config.toml          # 主配置（含 API key）
│   ├── deepseek.toml        # DeepSeek 配置
│   ├── mimo.toml            # Mimo 配置
│   └── template.toml        # 配置模板
├── runners/
│   └── run_backtest.py      # 主入口
├── src/
│   ├── core/                # 类型、配置、接口
│   ├── data/                # 数据层（行情、特征、缓存）
│   ├── portfolio/           # 交易引擎（组合、执行、风控）
│   ├── agent/               # LLM Agent（上下文、协议、工具）
│   ├── platform/            # 平台（实验、调度、日志、checkpoint）
│   └── evaluation/          # 评估指标
├── prompts/                 # Prompt 配置
├── tests/                   # 测试
├── docs/                    # 文档
│   ├── SYSTEM_OVERVIEW.md   # 系统详细说明
├── artifacts/               # 回测输出（不提交）
│   ├── runs/                # 回测数据库
│   └── cache/               # Derived cache
└── scripts/                 # 工具脚本
```

---

## 配置说明

所有阈值均可通过 TOML 配置：

```toml
# 回测参数
[backtest]
start = "2026-02-03"
end = "2026-02-09"
decision_interval = 5

# 资金
[portfolio]
initial_cash = 1000000

# 触发器阈值
[trigger]
pnl_pct_threshold = -0.03       # Plan trigger: -3%
cooldown_bars = 6               # 冷却 30 分钟

# 止损阈值
[stop_loss]
hard_stop_pct = -0.05           # Hard stop: -5%
crypto_hard_stop_pct = -0.08    # Crypto hard stop: -8%
```

详见 `config/template.toml` 和 `docs/SYSTEM_OVERVIEW.md`。

---

## 运行测试

```bash
# 全部测试
python -m pytest tests/ -q

# 指定模块
python -m pytest tests/test_memory_plan_lifecycle.py -q
```

---

## 文档

- `docs/SYSTEM_OVERVIEW.md` — 系统架构、决策流程、记忆系统、风控机制
- `docs/v8_runtime_cache_resume_progress.md` — 开发进度记录
