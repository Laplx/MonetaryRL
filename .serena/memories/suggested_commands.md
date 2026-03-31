# MonetaryRL 常用命令

## Python 入口脚本
- `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts\phase2_estimation.py`
- `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts\phase3_build_benchmark.py`
- `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts\phase4_solve_lq.py`
- `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts\phase5_train_ppo.py`

## Windows / PowerShell 常用命令
- 列目录：`Get-ChildItem`
- 递归列目录：`Get-ChildItem -Recurse`
- 搜索文本（优先 `rg`，若可用）：`rg pattern path`
- 读文件：`Get-Content path`
- Git 状态：`git status`

## 说明
- 当前仓库未发现 `pyproject.toml`、`requirements*.txt`、`environment*.yml` 等统一项目配置文件。
- 当前也未发现明确的测试、lint、format 命令文档；主工作流以 phase 脚本为主。