# spliceosome_agent

可安装的英文终端科研程序：对话与本地数据探查、自动计划与用户确认、七角色多智能体循环、确定性RNA/蛋白统计、证据审查、恢复和报告。支持两组配对/非配对 RNA/蛋白普通差异分析，以及可选的剪接体候选筛选。

本仓库只包含可安装的软件代码、内置资源与测试；原始数据、研究脚本、历史报告和设计图不随仓库分发。

## 安装与启动

要求Python≥3.11，支持macOS/Linux及WSL。使用独立环境：

```bash
uv tool install .
# 或在虚拟环境内：
pip install .

spliceosome-agent
spliceosome_agent --help
spliceosome-agent --workspace ./my-screening doctor
```

开发安装：`uv venv --python 3.11 .venv`，然后 `uv pip install --python .venv/bin/python -e '.[test]'`。本项目可直接运行 `.venv/bin/spliceosome-agent`。

全局参数 `--workspace/-w` 和 `--config` 放在子命令之前。默认工作区为当前目录下可见的 `result/`，新任务直接存放于 `result/RUN_ID/`，管理记录位于 `result/state/`；模型配置使用系统用户配置目录，可通过 `--config` 或 `SPLICEOSOME_AGENT_CONFIG` 指定。安装目录不存储用户数据。

```text
result/
├── RUN_ID/          # Report, figures, results, code, logs and checkpoints
├── ANOTHER_RUN_ID/
└── state/           # Sessions, plans, probes and cache
```

`-w DIR` 直接指定工作区，不再追加 `result/`；新建的自定义工作区同样直接保存 `DIR/RUN_ID/`。布局记录在 `state/workspace.json`，会话数据库在 `state/sessions.sqlite3`。探查、模型调用记录与缓存位于可见的 `state/` 子目录，报告缓存为任务内 `plot-cache/`。保留内部锁文件和系统模型配置／密钥管理机制。

已有 `.spliceosome-agent/` 或其他旧工作区保留原位，不自动迁移或合并；使用 `spliceosome-agent -w .spliceosome-agent` 打开原会话，任务仍位于该目录下的 `runs/RUN_ID/`。新旧布局自动识别；布局标记损坏或结构冲突时明确报错，不覆盖或自动修复。新布局中的 `state`、`runs`、`plans` 和 `sessions.sqlite3` 为管理保留名称，不能作为自定义任务ID。


## 模型配置

首次启动会要求配置模型；未完成配置前无法进入工作区。TUI 全部内置文案为英文，包括设置、编辑器、帮助、状态、错误与数据识别／尺度确认消息。模型提示也要求用英文回复；中文请求仍可输入，用户标签、路径及已冻结的历史内容保留原文。准确填写服务端模型 ID 和 base URL。

API key 保存在操作系统钥匙串（macOS Keychain / Linux Secret Service），不写入配置文件；配置目录权限为0600。运行时优先读取同名环境变量，其次读取钥匙串。keyring 不可用时明确报错并提示改用环境变量，不回退明文保存。

```bash
spliceosome-agent models add cloud --preset openai --model YOUR_MODEL_ID --key-env OPENAI_API_KEY --default
spliceosome-agent models add local --preset ollama --model YOUR_INSTALLED_MODEL_ID --default
spliceosome-agent models add lm --preset lmstudio --model YOUR_LOADED_MODEL_ID
spliceosome-agent models add gpu --preset vllm --model YOUR_SERVED_MODEL_ID
spliceosome-agent models add custom --base-url https://YOUR_HOST/v1 --model YOUR_MODEL_ID --key-env YOUR_API_KEY_ENV
spliceosome-agent models role Critic cloud
spliceosome-agent models list
spliceosome-agent models test --role Orchestrator
```

`models add --key` 将值写入钥匙串；不提供 `--key` 时该 profile 依赖 `--key-env` 指定的环境变量或钥匙串中已有条目。Ollama、LM Studio、vLLM预设分别使用loopback的11434、1234和8000端口及 `/v1`。程序只连接已有服务，不下载或加载权重。默认尝试JSON Schema，不兼容时降级JSON对象/普通JSON并严格本地校验；可用 `--format plain` 显式指定。

远程模型仅接收脱敏概要和特征级证据，不接收患者ID、样本矩阵或临床明细；外部tracing禁用。不要在自由文本中粘贴患者资料。无模型时可使用完整StudySpec建立确定性计划、审批和执行注册工具，不声称完成自然语言理解或模型审核。

## R统计环境

Python安装不修改系统R。锁定安装器使用R 4.5.x / Bioconductor 3.22，写入指定的独立库并记录实际版本：

```bash
spliceosome-agent deps --install-r --library ./r-library
spliceosome-agent doctor --study ./study.yaml
```

StudySpec的 `r_library` 指向该目录。自备环境也可用 `rscript`、`r_library` 指定；每run记录实际R和包版本。不要继承旧Linux路径。

## 输入契约

参见 [研究配置模板](docs/study_template.yaml)。输入路径相对study文件解析，之后冻结为绝对路径和SHA256。

- 矩阵首列为唯一feature ID，其余为唯一sample ID；CSV/TSV及RDS matrix/data.frame均支持。metadata也支持这三种格式。
- 候选筛选必须声明物种、ID体系；普通差异可保留 unknown。测量尺度必须有明确声明或用户确认；上游处理未知时如实记录。分组及配对先依据实际结构提出建议并集中确认，不猜测 log2 或归一化历史。
- 标准metadata为 `sample_id,group,subject_id`；现有 `ID,Type,Pair` 通过字段映射支持。配对时每个subject恰有一份参考和一份病例样本；非配对时subject不能重复。按ID连接，不按行序连接。
- 候选筛选模式的候选表必有 `gene_id,member_type,source,version,species`；类型为 `core`、`regulator` 或 `pathway_member`。`candidate_id_type` 声明候选ID体系。
- ID转换使用显式 `feature_id,gene_id` 映射表；歧义、多特征对应同一基因和未映射保留，不按p值选择映射。
- Knowledge支持导入冻结GO GAF/OBO、Reactome六列表或有来源的候选表。必须指定物种、版本和选择范围；宽泛Reactome成员保持 `pathway_member`。
- 当前研究RDS尺度仍需确认，不能由旧脚本或文件名自动继承。单组学可以运行，报告保留跨组学覆盖限制。

RNA整数counts使用DESeq2 Wald；已声明的log2 RNA/蛋白使用limma。配对模型含subject固定效应，配对内不变化的协变量由subject吸收并登记；其他协变量进入设计矩阵。秩亏、混杂及缺失协变量阻塞执行。

默认95% CI、双向检验、q≤0.05、不附加效应阈值，counts总计数≥10。连续矩阵不自动插补；缺失默认整特征排除并登记。各组学在所有有限p值范围BH后再提取候选；关闭DESeq2自适应独立过滤。`min_effect` 是报告筛选条件，不表示正式最小效应检验。

## 对话、探查、确认与执行

主流程：**打开 agent → 配置 LLM → 对话说明目的并指定数据 → 本地探查并澄清 → 自动展示计划 → Confirm and run → 生成运行脚本、执行分析 → 报告**。

在对话中给出具体文件路径或带引号的数据目录路径（含空格时必须加引号）。只扫描用户指定范围，跳过隐藏目录和符号链接，最多列出 2,000 个文件；不会扫描当前项目寻找历史队列。`/inputs` 按需显示本地文件引用与真实路径，模型仅接收 `file_0000` 等引用和脱敏结构摘要。多套候选输入需要用户说明选择与角色，不能按文件名猜测尺度、配对或处理历史；也可用 `/study /path/to/study.yaml` 显式加载现成配置。

Agent 先探查文件再组织确认内容；RDS 的首次结构检查使用 base R，不再要求先填写完整 StudySpec。数值型 data.frame 和 matrix 都能作为矩阵识别，通过样本 ID 的实际对应关系识别 metadata、分组列和配对列；只在结构支持时提出建议，不能从文件名或对话历史推断不存在的“癌栓”等标签。仅在存在多套可能输入时显示数据集选择，技术细节通过 `/plan`、`/inputs` 按需查看。

对“差异分析、画火山图”等普通请求，使用 StudySpec 2.0 的 `analysis_mode: differential`：不要求候选列表、物种或基因映射，使用原始 feature ID，默认全特征 BH q≤0.05 和95% CI。QC → RNA/Protein → Critic 后输出完整差异表、独立 PDF/SVG 火山图、source data、脚本和报告；未知物种/ID/上游处理历史如实保留，不声称完成注释。显式候选筛选继续使用原有知识/映射/整合流程和相应数据要求，旧 StudySpec 1.0/计划哈希读取保持兼容。

已核实的信息直接填入确认卡，不再重复问列名、已有分组或研究名称。数值范围不能证明 log2；缺少明确尺度声明时，只需选择“Already log2 abundance”“Positive abundance; apply log2 (no pseudocount)”或“Unknown; do not run”（RNA 还可选择“Raw integer counts”）。选择后自动刷新计划并收起选择框，也支持 `/scale log2`、`/scale raw_log2`、`/scale counts`、`/scale unknown` 修改尺度。原丰度只有显式选择后才会在隔离副本中做 log2，遇零/负值阻塞，不加伪计数；不确定时不执行。`/swap` 反转建议方向并使旧审批失效；`/dataset <序号>` 等价于数据集下拉框选择。

部分草稿与“已核实／建议确认／用户确认／待确认”来源记录跨会话保存。信息齐备后预检输入可读性、ID/metadata/配对、缺失/数值、设计秩与残差自由度、必要 R 包和字体；候选模式才检查候选来源/映射。预检不拟合模型、不计算显著性、不生成分析脚本。RDS 临时导出检查后移除，可从原输入重建；摘要和哈希写入工作区并绑定计划。R 工具链优先使用本地 StudySpec/`SPLICEOSOME_R_LIBRARY`，否则使用当前项目已有的 `.r-library`；不自动安装依赖或修改系统 R。

TUI 只显示当前回复、必要澄清、确认卡或运行结果；无 History、日志或技术详情常驻面板，无阶段条、发送按钮及重复提示。Enter 发送，输入框固定在底部，长内容在正文区完整滚动。确认卡保留文件名、方向、样本／配对数、尺度与方法，以及关键阻塞和警告。

空闲欢迎页用实心字符 `█` 拼出上下两行大字 `SPLICEOSOME`、`AGENT`，整体居中。标准窗口使用5×7字模，空间不足时切换为3×5紧凑字模；极窄或极矮时仅横幅区域滚动，不拆分单词，底部输入框保持固定。每次启动默认显示欢迎页，即使已有草稿、计划或运行；保留原会话与全部记录，后台任务继续运行。开始请求或查看任务／帮助时隐藏横幅。

核对确认卡后，点击 **Confirm and run** 才批准确切计划并启动 worker。仅在计划可确认时显示该按钮；运行中仅显示 **Stop**，点击后请求停止并保留部分结果，无额外确认面板。普通聊天不构成审批。模型配置用 `/models`，研究配置编辑器用 `/edit`（无参数）；`/edit <study.yaml>` 的原有命令保持兼容。

`/help`或F1显示全部可用命令，每个命令独立列出；`/tasks`位于顶部，`/runs`为同义命令，不支持单数`/task`。PgUp／PgDn滚动完整帮助，输入框与草稿保持可用。`/tasks`（或 `/runs`）查看运行记录，`/plans` 查看计划；`/status [run-id]`、`/report [run-id]`、`/logs [run-id]` 查看详情。TUI 专用 `/home` 返回欢迎页，`/current` 只读显示保留的当前任务，不启动分析。查询不会切换当前任务或修改审批，不显示确认／停止按钮，也不会被定时刷新覆盖。F1 查看帮助；Esc 返回查询前的欢迎页或任务页。欢迎页始终停留，首次模型设置完成后也不会自动显示旧任务。历史记录仍保存于 SQLite 和原运行目录；CLI／REPL 命令保持不变。

0.1.2包含完整英文帮助、启动欢迎页和可见result工作区。可运行已更新的全局`spliceosome-agent`／`spliceosome_agent`，或项目`.venv/bin/spliceosome-agent`；已有进程须退出后重启。后续源码修改仍需重新构建／安装才会更新全局版本，旧dist安装包继续保留。历史运行仍受代码指纹校验约束，安装更新不绕过恢复检查。

```bash
# 显式配置的 CLI 路径（也可不用 LLM）
spliceosome-agent -w ./my-screening init ./study.yaml
spliceosome-agent -w ./my-screening plan --goal '比较病例与参考的剪接体候选证据'
# 查看计划和 SHA256 后，一次确认并启动：
spliceosome-agent -w ./my-screening confirm PLAN_HASH_SHOWN_ABOVE
spliceosome-agent -w ./my-screening runs
spliceosome-agent -w ./my-screening verify RUN_ID
```

REPL 等价命令为 `/confirm <完整计划哈希>`。原有 CLI `approve PLAN_HASH` / `run PLAN_HASH` 和 REPL `/approve` / `/run` 保持各自语义，供现有脚本使用。新 UI 主流程只使用 Confirm and run。旧版计划可原样读取并保持原哈希；通过新确认入口使用旧计划时，先重新探查为新版本，必须再次确认新哈希。

输入或科学配置变更会使审批失效；启动前复核输入和环境，漂移后重新探查并要求确认。重复确认同一计划不会创建第二个任务，worker 启动失败可显式重试同一任务；执行中断使用 `resume RUN_ID`。重启仅恢复显示状态，不自动执行。`stop RUN_ID` 保留部分产物；后台 worker 在退出界面后继续运行。

每次获批后创建 `result/RUN_ID/run_analysis.py`（自定义工作区为 `DIR/RUN_ID/`）、冻结 `plan.json`、`preflight.json` 和 `runner.json`。worker 实际通过该入口调用既有 Engine 和注册统计工具。主脚本由受控模板生成，核心 DESeq2/limma 逻辑不交给自由生成的代码替换。输入、脚本、包代码或环境变化都会阻止执行/恢复。安装相同包和依赖后，也可在原工作区用 `python result/RUN_ID/run_analysis.py` 显式恢复；它仍需有效审批和指纹匹配，不复制原始数据。

七角色流程仍为 Orchestrator规划 → QC/Knowledge → RNA/Protein并行 → Integration → Critic。技术修复有预算、保留历史；科学设计变化必须重新规划审批。默认每任务最多3次修复、30次角色调用、100,000 tokens，不能为显著性自动改变方法或阈值。

历史命令保留：`/plans [哈希前缀]`、`/runs`（别名 `/tasks`）、`/status [run-id]`、`/report [run-id]`、`/logs [run-id] [n]`（默认40行）。停止与恢复仍校验原有身份、审批和指纹。

## 生成代码的执行与约束

注册工具按已批准计划连续执行；模型提出的新Python/R代码在**批准的计划内自动执行**，不再逐文件人工审阅：

```bash
# 运行目录查看可选补充代码与执行结果（无 approve/reject 命令）
ls result/RUN_ID/code/*/
```

执行在本机受限子进程中：显式白名单环境，不传入任何模型密钥或token；输入路径经 `SPLICEOSOME_INPUT_000...` 环境变量提供，写入限该提案输出目录，临时文件限 `SPLICEOSOME_WORK_DIR`；600秒超时及CPU/内存资源上限。提案绑定源码SHA256、执行工具链（Python/R解释器版本）和计划输入哈希；源码、工具链或输入变化都会使执行失效。执行前的静态检查会拒绝网络、子进程、删除、安装等明显模式并校验Python/R依赖，检查失败时代码直接阻断、绝不执行（记入run warnings）。

**安全边界说明**：生成代码的授权来自"用户已批准计划"，静态检查与受限环境是防误用过滤层，不构成强安全边界：本机执行无法强制断网，刻意混淆的代码可能绕过文本检查。请只在你信任的模型与端点下使用生成代码功能。补充产物标为未验证，不替换注册统计结果。

## 产物与科学边界

`result/RUN_ID/`（或新自定义工作区的 `DIR/RUN_ID/`）保存主分析入口、计划、探查快照、批准、指纹、各角色结果、完整差异表、映射审计、候选证据与覆盖、Critic意见、SQLite checkpoint、JSONL事件、模型调用审计、报告及manifest。

整合分层为双组学同向、双组学反向、单组学、未获支持、不可评价；层内按稳定ID排序，各队列分开。阴性和零候选是有效结果。图输出PDF/SVG，保存代码/source data/图注，不生成PNG。默认Arial 8 pt；只有计划显式允许时使用记录在案的替代字体。

表达/丰度不能证明PSI、剪接活性、装配、因果或临床效用。同一患者多组学是互补证据，不是独立验证。技术完成、模型审核、合成演示和科学结论分开登记。

## 合成演示与验证

```bash
pip install -e '.[test]'
spliceosome-agent examples ./synthetic-demo --r-library ./r-library
spliceosome-agent -w ./demo-workspace plan --study ./synthetic-demo/study.yaml --goal 'Synthetic software demonstration'
# 查看计划后，按上方 approve / run 流程执行
pytest -q
python -m build
```

数值测试真实调用DESeq2/limma并对照独立R拟合；模型协议和循环测试使用标注的模拟响应，不代替真实模型联调。

测试分两类：协议、契约、布局与命令测试无需R，在任何环境都可运行；依赖锁定R库的数值与预检测试在缺少 `jsonlite`/`limma`/`DESeq2` 时自动跳过并注明原因，不会误报为代码失败。安装R环境后可用 `SPLICEOSOME_R_LIBRARY=/path/to/r-library pytest -q` 启用全部测试。

官方参考：[OpenAI结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat)、[Ollama兼容接口](https://docs.ollama.com/api/openai-compatibility)、[Textual交互测试](https://textual.textualize.io/guide/testing/)。
