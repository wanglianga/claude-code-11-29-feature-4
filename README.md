# 湿地多台站观测记录处理服务

把不同台站听到的同一次声响归并为**同一事件**，给出发生位置、发生时间与仍不确定的候选范围，
避免"一只动物被四台站听到"被统计成四次活动。

- 技术栈：Python 3.12 / FastAPI / PostgreSQL 18 / SQLAlchemy 2 / NumPy
- 交付形态：**后端服务**（无前端页面），因此提供 OpenAPI 3.1 文档与 Swagger UI
- 全部数据为**虚构平面坐标与模拟台站**，不依赖真实硬件，结果**不用于预警**

> 本服务不做音频识别。台站端识别器已经给出「本机时刻 / 声响类别 / 相似性特征向量 / 误差范围」，
> 服务负责去重、校准时钟、归并同一声响、反演候选区域、定稿与修订。

---

## 1. 关键设计（与需求的对应关系）

| 需求 | 实现位置 | 说明 |
| --- | --- | --- |
| 每条观测保存站点、启动批次、源序号、本机采集时间、上传时间、类别、特征向量、误差范围 | `app/models.py::Observation`、`POST /stations/{id}/observations` | `local_capture_us` 是**台站时钟读数**，`uploaded_at_us`/`ingest_at_us` 是参考时间；两者从不混用 |
| 同批次同序号内容相同只收一次；内容不同单列冲突，不覆盖 | `app/ingest.py` | 身份 = `(station_id, boot_id, source_seq)`；相同内容 → duplicate；不同内容 → `observation_conflicts`，原文保留；整批内容指纹相同的重放才是 no-op，`batch_key` 复用但内容不同时按记录逐条判定而非整批丢弃 |
| 用已知位置的校准声源建立本机时间 → 参考时间的分段线性映射 | `app/calibration.py` | 先减去 `距离 / 配置传播速度` 得到参考到达时刻，再拟合 `local = a·ref + b`；可按残差增益自动分段（漂移） |
| 重启前后不得共用参数 | `calibration_models` 唯一键 `(station_id, boot_id, version_hash)` | 每个 boot 独立建模；斜率不合理时判定为"把重启跳时当成漂移"，拒绝该模型 |
| 锚点太少 / 残差超限时保留未校准观测，不拿上传时间替代 | `fit_calibration` / `apply_calibration` | 低于最少锚点数、跨度不足、RMS/最大残差超限、超出锚点覆盖并被计入外推，都返回 `uncalibrated` 并写明理由 |
| 以校正后的到达区间筛选候选组合，同站同次事件至多一条 | `app/association.py` | 候选组合是**团（clique）**，逐对校验 |
| 按声类、特征距离、传播时差检查相容性 | `_compatible` | `|Δt| ≤ d(站,站)/v + u_i + u_j`（三角不等式必要条件）+ 特征欧氏距离上限 |
| 不允许仅靠两两邻近传递合并相距很远的声源 | 团枚举 + 整组可定位性 | 每个候选还必须整体通过网格搜索（存在共同位置与时刻），链式传递无法通过 |
| 有限区域内由粗到细网格搜索位置与发声时刻 | `app/localize.py` | 粗网格 → 逐级细化（默认 100 m → 12.5 m），以各站 `到达 − 传播时间` 的离散程度为拟合误差 |
| 满足全部误差约束的候选区域；多个不连通区域全部返回 | `Localization.regions` | 8 邻域连通分量；每个区域给出边界、面积、代表点、单元格 |
| 几何布局不足不能报一个确定坐标 | `position_certain` / `reason` / `warnings` | 共线站、区域过大、单区域但站数不足 → 不声称确定坐标 |
| 多个事件争用同一观测时枚举互斥组合 | `app/solver.py` | 精确集合划分（整数字典序目标）：先最大化可靠解释记录数，再最大化解释总数，最后最小化总残差 |
| 出现同分解需等待人工选择 | `status = awaiting_selection` + `GET /runs/{id}/options` + `POST /runs/{id}/select` | 完全并列时**不自动选**，每套分解各自独立、不共享观测 |
| 按台站已完整上报的采集进度判断可否定稿 | `pipeline._finalization_state` | 仅当"可能听到该事件的每个台站"都以**已校准的**进度声明覆盖窗口才 final；断联站不因时间流逝被视为已报齐 |
| 允许调查员明确排除某站并保存理由 | `POST /events/{id}/exclusions`、`POST /stations/{id}/exclusions` | 记录排除理由与时间范围，只有排除掉最后一个未决台站时才推动定稿 |
| 定稿后迟报/校准修订生成新事件修订及新旧对应 | 每次运行自动 `_link_revisions`；`mode="revision"` 递增 `revision` | 旧事件置 `superseded` 并记 `superseded_by_event_id`；新事件记 `supersedes_events`，并写 `event_revision_links` |
| 处理中断从同一输入截止点继续，输出不混用不同校准参数 | `checkpoint_json` + `cutoff_us` + `calibration_json` | 暂停/恢复复用同一 cutoff 与校准快照；`collect` 阶段把输入快照整体冻结 |
| 能从事件回查原观测、所用锚点、排除理由及候选区域 | `GET /events/{id}`、`/events/{id}/anchors`、`/events/{id}/observations` | 事件详情含观测、残差、校准模型、锚点 ID、候选区域单元格、排除理由、修订链接 |
| 独立按给定声源反算到达时间检验计算 | `verifier/verify.py` | 不 import 任何服务代码，按文档化前向模型重算每条本地读数并与发布结果对比 |

### 时间模型（为什么不用到达时间归并）

```
台站本机读数 local = a_boot · 参考到达时刻 + b_boot        （每个 boot 一组参数）
参考到达时刻       = 发声时刻 + 距离(声源, 台站) / 传播速度
```

校准声源的位置和发声时刻已知，所以先扣掉传播时间，再用最小二乘拟合 `a_boot, b_boot`。
重启产生新的 `boot_id`，`b` 可以任意跳变；同 boot 内允许缓慢漂移（分段线性）。
上传时间只做记录，从不参与归并——这正是断电重启后补传记录会错位的原因。

---

## 2. 快速启动

### 2.1 Docker Compose（推荐）

```bash
docker compose up --build -d
curl -s http://127.0.0.1:8000/health
# {"status":"ok","database":"ok","version":"1.0.0"}
```

- API: <http://127.0.0.1:8000>
- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>（仓库内亦存有 `openapi.json`）
- 数据库 PGDATA 在命名卷 `dbdata` 中，`docker compose down -v` 可彻底清空
- 本次交付环境未安装 Docker CLI，因此 `docker compose` 未在本机实际执行；镜像内容是
  `pip install -r requirements.txt` + 复制源码 + `uvicorn app.main:app`，与本地验证过的启动方式一致

### 2.2 本地开发（本机 PostgreSQL）

容器内已预装 PostgreSQL 18 与 Python 3.12。默认连接串为
`postgresql+psycopg://ccdev:ccdev@127.0.0.1:5432/wetland_dev`（用环境变量 `DATABASE_URL` 覆盖）。

```bash
/opt/python-3.12/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt          # 已预置镜像源
PGPASSWORD=ccdev /opt/postgresql-18/bin/psql -h 127.0.0.1 -U ccdev -d postgres \
  -c 'CREATE DATABASE wetland_dev;'
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

表结构在应用启动时自动创建（`app/db.py::init_db`）。所有受保护接口需要请求头
`X-API-Key: dev-api-key`（`WETLAND_API_KEY` 可改，`WETLAND_AUTH_DISABLED=true` 可关闭）。
`/health`、`/docs`、`/openapi.json` 不需要鉴权。

### 2.3 一键演示 + 自检

```bash
BASE=http://127.0.0.1:8000 ./scripts/demo.sh
```

该脚本对七个内置场景逐个执行：重置 → 模拟台站分批上传 → 运行处理 → 独立校验，
最后打印每条检查的通过情况。也可以手动分步执行（见第 5 节）。

---

## 3. 测试账号 / 密钥

本服务是机器对机器的数据接口，没有用户账号体系。唯一的凭据是 API Key：

| 项 | 开发默认值 | 覆盖方式 |
| --- | --- | --- |
| API Key | `dev-api-key` | 环境变量 `WETLAND_API_KEY` / 请求头 `X-API-Key` |
| 数据库 | `ccdev/ccdev@127.0.0.1:5432/wetland_dev` | `DATABASE_URL` |
| Compose 数据库 | `wetland/wetland@db:5432/wetland` | `docker-compose.yml` |
| 关闭鉴权 | 默认开启 | `WETLAND_AUTH_DISABLED=true`（仅限隔离环境） |
| 危险操作 | `/admin/reset?confirm=true` | `WETLAND_ALLOW_RESET=false` 可禁用 |

测试库与开发库分离：`tests/conftest.py` 默认使用 `wetland_test`，测试前自动建表、
结束后删除。创建测试库：

```bash
PGPASSWORD=ccdev /opt/postgresql-18/bin/psql -h 127.0.0.1 -U ccdev -d postgres \
  -c 'CREATE DATABASE wetland_test;'
```

---

## 4. 验证方式（实际执行过的）

```bash
# 1) 单元 + 集成 + 接口契约 + 独立校验器，共 82 个用例
.venv/bin/python -m pytest
# 82 passed

# 2) 七个样例场景的端到端自检（真实 HTTP 服务）
BASE=http://127.0.0.1:8000 bash scripts/demo.sh

# 说明：上面的 pytest 含 8 个由独立校验器驱动的端到端用例、5 个接口契约用例，
# 以及 14 个诊断纯算法/集成用例与 7 个认证簿用例；demo.sh 使用真实 uvicorn 进程
# + 真实 PostgreSQL，逐个场景输出校验结论。

# 3) 单独校验某个已加载场景
.venv/bin/python -m verifier.verify --scenario reboot \
  --base-url http://127.0.0.1:8000 --api-key dev-api-key

# 4) 重新生成样例真值文件
.venv/bin/python scripts/generate_samples.py

# 5) 重新导出 OpenAPI 文档
.venv/bin/python scripts/export_openapi.py
```

最近一次完整执行的结果（`scripts/demo.sh` 写出 `samples/<name>/verification.json`，该文件是运行产物、已加入 `.gitignore`）：

| 场景 | 校验项 | 结果 | 关键观测 |
| --- | --- | --- | --- |
| `overlapping` 双声源重叠 | 8 | 全部通过 | 6 事件 × 4 台站；位置误差 7.3 m；发声时刻点估计误差 5.2 ms，真值均落在发布区间内 |
| `reboot` 重启跳时 + 迟报 | 9 | 全部通过 | S2 的 `b1`/`b2` 各有独立有效模型（斜率 1.0000596 / 0.9999546）；S3 迟到 20 分钟后补传仍参与全部事件 |
| `collinear` 共线站点 | 6 | 全部通过 | 每个事件返回 2 个镜像候选区域，`position.certain=false`，不报单一坐标 |
| `duplicates` 重复补传 | 10 | 全部通过 | 整批重放 `accepted=0`；冲突记录 1 条且原文未被覆盖；事件数与记录数不变 |
| `revision` 迟报 + 校准修订 | 11 | 全部通过 | 第二次运行校准哈希不同；旧事件 `superseded` 且由新事件 `supersedes_events` 指名；实际活动数 5 条而非叠加 |
| `diagnosis` 时钟阶跃/坐标误记/区域速度 | 7 | 全部通过 | S1 时钟阶跃拟合 181.4 ms（真值 180）；S2 坐标位移恢复 93.6 m；S3 异常锚点排除并留前后对比；区域共享速度 323.5 m/s（真值 328） |
| `sparse_anchors` 锚点不足 | 3 | 全部通过 | 无法形成留出分区，全部假设被认证门禁拒绝，`/certified-parameters` 为 0 |

独立校验器（`verifier/verify.py`）不引用 `app/` 下任何模块，按
`arrival = emission + 距离/速度`、`local = slope_boot·arrival + intercept_boot` 从真值反算，
再逐条与实际上传的本地时间戳对比（最差 12.1 ms，来自校准声源的模拟噪声）。

---

## 5. 手动走查（curl）

```bash
KEY='X-API-Key: dev-api-key'
BASE=http://127.0.0.1:8000

curl -s -X POST "$BASE/admin/reset?confirm=true"
curl -s -X POST "$BASE/stations" -H "$KEY" -H 'Content-Type: application/json' \
  -d '{"id":"S1","name":"North","x":300,"y":100,"hearing_radius_m":1700}'
curl -s -X POST "$BASE/calibration-sources" -H "$KEY" -H 'Content-Type: application/json' \
  -d '{"id":"C1","x":100,"y":-150,"emissions":["2026-11-13T23:20:00.000000Z"]}'
curl -s -X POST "$BASE/sound-classes" -H "$KEY" -H 'Content-Type: application/json' \
  -d '{"id":"chirp","feature_dim":4,"feature_distance_limit":0.05}'

# 台站上传（local_capture_at 不带时区 —— 它是台站时钟读数）
curl -s -X POST "$BASE/stations/S1/observations" -H "$KEY" -H 'Content-Type: application/json' -d '{
  "station_id":"S1","boot_id":"b1","batch_key":"S1-b1-1",
  "records":[
    {"source_seq":1,"kind":"calibration","class_label":"calibration",
     "local_capture_at":"2026-11-14T09:40:12.345678","features":[0.0,0.1,0.0,0.0],
     "uncertainty_ms":12,
     "calibration":{"source_id":"C1","emission_reference_at":"2026-11-13T23:20:00.000000Z"}},
    {"source_seq":2,"kind":"sound","class_label":"chirp",
     "local_capture_at":"2026-11-14T09:41:00.000000","features":[1.0,0.30,0.20,0.90],
     "uncertainty_ms":25}
  ],
  "progress":{"complete_through_local_at":"2026-11-14T09:50:00.000000","complete":true}
}'

curl -s -X POST "$BASE/runs" -H "$KEY" -H 'Content-Type: application/json' -d '{"wait":true}'
curl -s "$BASE/observations" -H "$KEY" | python -m json.tool
```

上面这条最小样例只有 1 个校准锚点，因此**预期**结果是该观测被保留为
`calibration_status = "uncalibrated"`、`calibration_reason` 说明锚点不足，且不参与归并
——这本身就是一条要验证的约束。完整的可定位流程用模拟器一步到位：

```bash
.venv/bin/python -m simulator.run_simulator --scenario reboot --reset
curl -s -X POST "$BASE/runs" -H "$KEY" -H 'Content-Type: application/json' -d '{"wait":true}'
curl -s "$BASE/events" -H "$KEY" | python -m json.tool         # 事件、位置、候选区域
curl -s "$BASE/events/1" -H "$KEY" | python -m json.tool       # 观测、锚点、排除理由、修订
curl -s "$BASE/events/1/anchors" -H "$KEY" | python -m json.tool
```

其它常用入口：

| 接口 | 作用 |
| --- | --- |
| `GET /observations?calibration_status=uncalibrated` | 查看被保留但未校准的观测及其理由 |
| `GET /conflicts` | 查看同序号异内容的补传 |
| `GET /calibrations?station_id=S2` | 查看每个 boot 的分段线性模型、残差、锚点 |
| `GET /runs/{id}/options` | 求解并列时各套分解的对比 |
| `POST /runs/{id}/select {"option_id":"OPT-02"}` | 人工选定分解 |
| `POST /runs/{id}/resume` | 从原 cutoff 续算 |
| `GET /stats` | 总体计数 |
| `POST /propagation-regions` | 定义区域传播模型的适用范围与基线速度 |
| `POST /diagnostic-runs` | 跑一次故障归因诊断（station 或 region 范围），冻结证据并生成假设册 |
| `GET /diagnostic-runs/{id}/evidence` | 查看冻结证据库，可按 `partition=fit|holdout`、`excluded` 过滤 |
| `GET /hypotheses?diagnostic_run_id=...` | 查看时钟/坐标/速度假设的拟合、留出指标、可识别条件与排名理由 |
| `POST /hypotheses/{id}/certify` | 走认证三道门禁，批准或拒绝一个假设 |
| `GET /certifications` / `/{id}/impacts` | 认证簿与每个认证影响的事件版本 |
| `GET /certified-parameters` | 已交给定位数据集的生效参数 |
| `POST /events/{id}/confirmations` | 人工确认事件（之后参数修订只生成待复核稿） |
| `GET /review-drafts` / `POST /review-drafts/{id}/resolve` | 待复核定位稿的查询与人工受理/驳回 |

---

## 6. 样例与模拟器

```
samples/
  index.json            七个场景的概要
  <scenario>/truth.json 台站位置（含注册坐标偏移）、每 boot 真值时钟模型、校准声源时刻、上传波次、真实事件
```

`truth.json` 中的 `waves[].records` 就是模拟器实际 POST 的载荷；`waves[].truth` 记录每条
记录的生成来源（校准源 / 声源+叫次、本地读数、到达时刻），供校验器独立反算。

```bash
# 仅生成真值文件，不访问服务
.venv/bin/python -m simulator.run_simulator --scenario collinear --ground-truth-only

# 只上传前 4 波（模拟处理中断）
.venv/bin/python -m simulator.run_simulator --scenario overlapping --reset --pause-after-wave 4

# 只上传 early 或 late 阶段
.venv/bin/python -m simulator.run_simulator --scenario revision --phases early
.venv/bin/python -m simulator.run_simulator --scenario revision --phases late

# 重放首批并制造一条内容冲突
.venv/bin/python -m simulator.run_simulator --scenario duplicates --reset --retransmit
```

五个内置场景：

1. **overlapping** — 两个相距约 200 m 的同类声源在数秒内各自鸣叫，S2/S3 同时听到两者；
   要求"四站一声 = 1 事件"且"两兽重叠 = 2 事件"。
2. **reboot** — S2 在 t=900 s 断电重启，时钟跳变 1500 s 且漂移反号；S3 断联 20 分钟后补传；
   要求按 boot 分别校准、迟报记录仍参与，未报齐时事件保持 provisional。
3. **collinear** — 四站严格共线，跨线镜像在传播时延上不可区分；要求返回两个不连通区域，
   且不声称单点坐标，但仍正确计数。
4. **duplicates** — 整批重放（应无效）+ 同序号异内容（应入冲突表且不覆盖原文）；
   要求事件与计数不变。
5. **revision** — 先无 S3 数据跑一次，再补传 S3 与迟到校准后跑修订；要求旧事件被取代、
   新事件指名被取代者，而不是一次活动悄悄裂成两条。

两个故障归因场景（第 7.5 节的服务对象）：

6. **diagnosis** — 三种故障并存：S1 时钟在 boot 晚期阶跃 +180 ms（各方向同向残差）、
   S2 测点向西误记 80 m（残差随方向反转）、区域真实声速 328 m/s 而基线仍为 343（残差随
   距离增长），并向 S3 注入一个 900 ms 异常锚点；要求正确归因、异常排除留前后证据、
   区域共享速度可恢复。
7. **sparse_anchors** — 每站仅 4 个校准锚点，无法形成留出分区；要求所有假设都被认证
   门禁拒绝，定位数据集收不到任何认证参数。

---

## 7. 算法要点与已知边界

**候选组合**：按声类分组 → 时间与特征索引构图 → Bron–Kerbosch 枚举团（限制最大站数）→
每个团做整组网格定位，不可行则尝试去掉 1–2 个成员（覆盖双声源争用同一台站记录的情形）。

**定位**：在配置观测区域内做两级以上网格。对一个候选格心，计算每站
`校正到达 − 距离/速度`，以不确定度加权求均值，要求所有站都落在
`u_i + 全局松弛 + 半对角线时差` 之内（保守扩张，保证不漏掉边界区域）。可行格逐级四等分细化，
最后取 8 邻域连通分量。输出的精度就是实际到达的最细格距；若因格数上限提前停止，
事件上的 `warnings` 会说明"精度低于请求值"。

**求解**：候选 = 观测集合（含权重量）。目标为整数字典序：可靠解释数（其窗口已可定稿的观测）
> 解释总数 > 总残差平方和。中间这一层是必要的实现补充：需求只规定"可靠解释数"与"总残差"两项，
但若台站尚未报齐（可靠数为 0），仅按残差最小会退化为"少报事件"，把已收到的记录无故丢弃；
因此显式加入"解释总数"作为第二优先级，并且仅在可靠数相同时起作用。先按"共享观测"把候选图拆成互不影响的连通分量（分量最优解相加即全局最优），
每个分量做分支定界；精确同分时置 `awaiting_selection`。求解器与暴力枚举在 400 组随机小实例上逐一对拍一致。

**定稿**：`final` 需要"所有可能听到该事件的台站"满足：已校准进度覆盖窗口末尾 + 余量、
或被调查员显式排除（带理由）、或几何上不在听力半径内。断联站与未校准站都会让事件保持
`provisional`，运行状态记为 `awaiting_review`。

**修订**：任何一次重新运行只要重新解释了仍在生效的旧事件记录（或有同声类重叠窗口且共享台站），
就把旧事件置 `superseded`、新事件记录 `supersedes_events`；旧行保留（`include_superseded=true`
可查），因此不会"拆成两条又保留旧计数"。`mode="revision"` 额外把 `revision` 递增。

**边界说明（有意为之的保守行为）**

- 只承诺**既定网格精度内的候选区域**，不做精确反演，也不做统计置信区间。
- 两个站点的记录永远不构成"确定坐标"；共线布局永远返回镜像区域。
- 常规定位的传播速度取配置常数；只有**已认证**的区域速度参数才会改变定位速度，
  诊断本身不修改任何定位输入。
- 校准不合格的观测被**保留但不参与**归并，接口可按 `calibration_status` 回查原因。
- 迟报记录不会追溯修改已定稿事件本身，而是通过下一次运行的修订关系表达。
- 诊断层在管线校准被拒时自建的时钟趋势仅用于产生证据残差，带
  `mapping_source=diagnostic_clock_trend` 标注，永不冒充已认证时钟模型。
- 留不出 holdout 分区、留出 RMSE 恶化、参数不可识别或 boot 版本冲突时，认证簿一律
  拒绝发布，宁可不签发参数也不把残差强行修到单一参数上。
- 区域式组织中方向覆盖不足的站点暂时拿不到局部修正；单站式组织的跨站速度离散度会
  显式提示"区域变化被吸收进设备参数"的风险，两种组织始终并列对比、不互相覆盖。
- 已人工确认事件受参数修订影响时只生成待复核稿，原始事件行不改写；本服务不派发工单、
  不改动现场设备。
- `/runs` 支持 `wait=false` 后台执行以便观察中断（`debug_stop_after_phase`），
  生产使用建议 `wait=true` 或外部任务队列。

---

## 7.5 校准假设诊断与参数认证（故障归因）

常规时钟校准把每个残差都归因于台站时钟。维护人员需要进一步判断残差究竟来自
**时钟偏移/漂移**、**测点坐标误记**还是**区域传播速度变化**，避免把所有误差强行修到
一个参数上而让后续定位系统性偏移。本服务为此提供独立的四层结构：

1. **校准证据库**（`diagnostic_evidence`）：每次诊断把声源位置、发声时刻、站点记录
   测点、参考/实测到达、传播时延、方向角与 boot 身份冻结成独立快照；管线校准被拒的
   boot 会在诊断层内拟合带标注的稳健时钟趋势（`mapping_source` 区分两种来源），不会
   因为没有有效校准而丢失证据。
2. **假设册**（`hypotheses`）：同一批锚点上并行维护三类可识别假设——
   - 时钟 `clock`：仿射偏移/漂移，并在锚点时间分位上搜索**时钟阶跃**（连续仿射模型
     无法表达的跳变，是时钟独有的可识别信号）；
   - 坐标 `coordinate`：`r·v = u·shift` 的测点位移，要求多方向覆盖与设计矩阵良态；
   - 区域速度 `propagation_speed`：`r_seconds = k·distance` 的共享速度，可联合估计
     各站时钟截距；另有 `clock_coordinate` 联合模型作为对照。
3. **留出验证器**：按 (station, boot) 确定性分层把锚点划分为 fit/holdout，所有假设只
   在 fit 上拟合、在同一批未参与拟合的 holdout 上评分。方向规则（任务原文要求）参与
   排名：各方向同向优先评估时钟、残差随方向反转则判定单一时钟修正不足而优先坐标、
   残差随距离显著相关则指向传播速度，其余情形按留出 RMSE 排序。
4. **认证簿**（`certifications` / `certified_parameters`）：只有通过三道机械门禁的假设
   才能发布给定位数据集——
   - 可识别门禁：锚点跨度、方向覆盖、设计条件数、物理取值范围；
   - 留出门禁：新拟合在未触碰锚点上不得比基线相对恶化超过 15%，且绝对 RMSE 不超过
     150 ms；留出锚点不足时直接拒绝；
   - boot 版本门禁：设备参数按 boot 隔离，旧 boot 的迟到拟合不得覆盖新 boot 已认证
     参数（boot 顺序以证据库参考到达时刻为准，因为本地时钟读数在重启时会跳变）。

**两种参数组织方式都在同一留出集和同一批定位事件上对比**：

- 单站式（`per_station`）：每站独立估计时钟与坐标，局部残差易定位、锚点稀疏也能先行
  处理；风险是区域传播变化会被吸收为各站的伪设备参数（诊断结果中的跨站速度离散度会
  暴露这一点）。
- 区域式（`regional`）：先估计一个共享传播速度，再仅对方向覆盖充分的站点追加局部坐标
  修正；跨站口径一致，但覆盖不足的站点暂时**不签发**局部参数。

**异常锚点**仅在 fit 分区做稳健 σ 截断，holdout 永不参与剔除；排除前后的 fit/holdout
表现与逐条排除理由都写进诊断结果和证据行。**人工确认过的事件**若受参数修订影响，只
生成 `localization_review_drafts` 待复核稿（含前后位置/发声/残差对比），原始事件行
绝不改写；未确认事件只记录受影响版本号。本服务不派发工单、不改动现场设备。

核心模块：`app/diagnostics.py`（证据、划分、拟合、方向分析、两组织对比）、
`app/certification.py`（门禁、版本、事件影响、待复核稿）、`app/certified.py`（向定位
数据集的受控交付）。

---

## 8. 目录结构

```
app/
  main.py          FastAPI 应用、鉴权中间件、OpenAPI 定制
  api.py           全部路由
  schemas.py       Pydantic 请求/响应
  models.py        SQLAlchemy 表结构（含证据库、假设册、认证簿、待复核稿）
  db.py            引擎与会话
  config.py        全部可调阈值（会写入每次运行的快照）
  timeutil.py      整数微秒时间工具
  ingest.py        上传、去重、冲突、进度
  calibration.py   锚点解析与分段线性时钟校准
  diagnostics.py   假设诊断：证据冻结、留出划分、时钟/坐标/速度拟合、方向分析
  certification.py 认证门禁、boot 版本控制、事件影响与待复核稿
  certified.py     认证参数向定位数据集的受控交付
  localize.py      粗到细网格搜索与候选区域
  physics.py       传播时延与相容性原语
  association.py   候选团枚举
  solver.py        互斥集合划分求解与并列检测
  pipeline.py      阶段化处理、检查点、定稿、修订
simulator/
  scenarios.py     七个虚构场景与每个 boot 的时钟模型
  run_simulator.py 真值生成、可暂停的分波上传、重放/冲突注入
verifier/
  verify.py        独立前向模型校验器（不引用 app/）
scripts/
  demo.sh          七场景一键演示 + 自检
  generate_samples.py
  export_openapi.py
tests/             82 个用例（校准、定位、关联、求解、流水线、校验器、接口契约、诊断、认证）
samples/           七个场景的真值文件
openapi.json       导出的 OpenAPI 3.1 文档
Dockerfile / docker-compose.yml
```

## 9. 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://ccdev:ccdev@127.0.0.1:5432/wetland_dev` | 数据库连接 |
| `WETLAND_API_KEY` | `dev-api-key` | 接口密钥 |
| `WETLAND_AUTH_DISABLED` | `false` | 关闭密钥校验（仅隔离环境） |
| `WETLAND_ALLOW_RESET` | `true` | 是否允许 `/admin/reset` |
| `PROPAGATION_SPEED_MPS` | `343.0` | 传播速度 |
| `AREA_X_MIN/X_MAX/Y_MIN/Y_MAX` | `0/2100/-1100/1100` | 观测区域 |
| `CAL_MIN_ANCHORS` / `CAL_MIN_SEGMENT_ANCHORS` | `4` / `3` | 最少锚点数 |
| `CAL_MAX_RESIDUAL_MS` / `CAL_MAX_RMS_RESIDUAL_MS` | `250` / `120` | 残差上限 |
| `CAL_MIN_ANCHOR_SPAN_S` | `60` | 锚点最短跨度 |
| `CAL_COVERAGE_MARGIN_S` | `3600` | 允许的外推余量 |
| `FEATURE_DISTANCE_LIMIT` | `0.25` | 未配置声类时的特征距离上限 |
| `ASSOCIATION_TIME_SLACK_MS` | `150` | 关联松弛 |
| `GRID_COARSE_M` / `GRID_REFINE_LEVELS` | `100` / `3` | 网格精度 |
| `MIN_EVENT_STATIONS` | `2` | 事件最少台站数 |
| `MAX_SOLUTION_OPTIONS` | `5` | 并列/备选分解上限 |
| `HEARING_RADIUS_M` / `PROGRESS_MARGIN_S` | `1600` / `120` | 定稿判定 |
| `DIAG_HOLDOUT_FRACTION` / `DIAG_MIN_HOLDOUT` | `0.25` / `2` | 留出锚点比例与最少数量 |
| `DIAG_MIN_FIT_ANCHORS` / `DIAG_MIN_ANCHOR_SPAN_S` | `4` / `60` | 假设拟合的最少锚点与最短跨度 |
| `DIAG_MIN_DIRECTIONS` / `DIAG_SAME_SIGN_FRACTION` | `3` / `0.75` | 方向扇区数与时钟偏移同向阈值 |
| `DIAG_REVERSAL_RATIO` / `DIAG_DISTANCE_CORRELATION` | `0.6` / `0.7` | 方向反转与距离相关判据 |
| `DIAG_OUTLIER_SIGMA` / `DIAG_MIN_STEP_MS` | `3.5` / `30` | 异常锚点截断与时钟阶跃最小幅度 |
| `DIAG_SPEED_MIN_MPS` / `DIAG_SPEED_MAX_MPS` | `300` / `400` | 区域速度可识别的合理区间 |
| `DIAG_HOLDOUT_ABS_LIMIT_MS` / `DIAG_HOLDOUT_IMPROVEMENT_RATIO` | `150` / `0.1` | 留出绝对上限与实质性改善比例 |
| `CERT_HOLDOUT_WORSEN_RATIO` | `1.15` | 认证允许的留出相对恶化倍数 |
| `CERT_STALE_MAX_AGE_S` | `2592000`（30 天） | 拟合证据相对 cutoff 的过期上限 |
