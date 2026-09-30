# FlagOS 竞赛官方资料归档

本目录保存 2026-09-29 初次获取并于 2026-09-30 补充的公开官方资料，用于本项目的离线阅读和来源追踪。它不是 SDK、驱动、模型或容器镜像，不包含凭据，也没有执行任何项目代码或 GPU 工作负载。

## 目录内容

- `repos/flagos-skills`：`flagos-ai/skills` 的浅克隆，稀疏检出仓库根文件和完整的 `skills/kernelgen-flagos` 目录。
- `repos/flagos-docs`：`flagos-ai/docs` 的浅克隆，稀疏检出仓库根文件及 FlagGems、vLLM Plugin-FL、FlagTree 中文文档。
- `repos/KernelGen`：`flagos-ai/KernelGen` 的浅克隆，稀疏检出仓库根文件和完整 `docs` 目录。
- `snapshots`：FlagOS、KernelGen、天数和沐曦公开入口的响应快照，以及 2026-09-30 访问的赛题中文原始 JSON 和 MiniCPM5-2B `config.json`。
- `manifest.tsv`：来源 URL、最终 URL、本地路径、状态、类型、大小、摘要及 Git 提交信息。
- `SHA256SUMS`：所有已检出资料文件和网页快照的 SHA-256；不包含 `.git` 内部文件，也不递归包含本清单文件。

## 固定的仓库版本

| 本地目录 | 分支 | 提交 |
|---|---|---|
| `repos/flagos-skills` | `main` | `e470f91f3f6e55efe108b822a7b6bbf818f6a99c` |
| `repos/flagos-docs` | `main` | `6fac9b4c2842e983bdfc855ed3ce4b69e77ad81c` |
| `repos/KernelGen` | `main` | `e0cdde1dd0572ec012c41be51099e1c593532e5c` |

三个工作树在归档完成时均为 clean。`skills/kernelgen-flagos` 在提交树和本地工作树中均为 14 个文件，文件集合完整。

## 快照状态说明

- `full-html`：服务端返回了可直接阅读且具有标题的文档 HTML。
- `shell-only`：只获得动态前端入口壳，正文仍依赖浏览器 JavaScript 或后续 API；不得把该文件当作完整文档正文。
- `public-json`：未登录公开 GET 接口返回的结构化索引结果。

FlagOS SkillHub、KernelGen Skill 详情、天数开发者首页和沐曦文档入口均属于动态站点，因此相应 HTML 标记为 `shell-only`。KernelGen、FlagGems、vLLM Plugin-FL、FlagTree 和 FlagOS 文档中心快照可直接离线阅读。

沐曦 C500 条目通过公开的 `file/search` GET 接口按 `file_name=C500` 获取。响应头为 `application/json`，本地文件魔数检测也为 `application/json`。本次没有调用下载接口，没有把预览响应命名为 PDF，也没有下载任何大型或旧版文档。

赛题原始响应使用 `Lang: CN` 请求头从公开 API 获取，模型配置从 ModelScope 模型仓库的 `resolve/master/config.json` 获取；两者均以 `curl --compressed` 保存解压后的原始 JSON 响应，没有重新排版或抽取字段。它们的 URL、抓取 UTC 时间、本地访问日期、字节数和 SHA-256 均记录在 `manifest.tsv`，对应文件分别为 `snapshots/race-539vlt2p-cn-20260930.json` 与 `snapshots/minicpm5-2b-config-20260930.json`。

## 使用和更新边界

这些快照只代表清单中的抓取时间。实施竞赛修改前，仍应以赛题页面、组委会书面答复和对应仓库固定提交为准；需要刷新资料时，应重新记录最终 URL、HTTP 类型、摘要和提交号。动态站点若要求登录，应停止并由用户提供授权，不得绕过认证。

校验可在本目录运行：

```bash
sha256sum -c SHA256SUMS
```

该命令只做文件完整性检查，不导入项目、不加载模型，也不运行 GPU 代码。
