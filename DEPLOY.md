# 🚀 SentiPort Streamlit Cloud 部署指南

完整的 Streamlit Cloud 部署方案，帮助你将 AlphaQuest NLP 交易系统部署到云端。

## 📋 目录

- [部署前准备](#部署前准备)
- [Streamlit Cloud 部署步骤](#streamlit-cloud-部署步骤)
- [环境变量配置](#环境变量配置)
- [数据持久化方案](#数据持久化方案)
- [故障排查](#故障排查)
- [性能优化建议](#性能优化建议)

---

## 🔧 部署前准备

### 1. GitHub 仓库准备

确保你的项目已经推送到 GitHub：

```bash
# 如果还没有初始化 Git
git init
git add .
git commit -m "feat: 准备部署到 Streamlit Cloud"

# 创建远程仓库并推送
git remote add origin https://github.com/你的用户名/sentiport.git
git branch -M main
git push -u origin main
```

### 2. 必需文件检查

确保以下文件存在于项目根目录：

- ✅ `demo_streamlit.py` - 主应用文件
- ✅ `requirements.txt` - Python 依赖
- ✅ `.streamlit/config.toml` - Streamlit 配置
- ✅ `financial_data.db` - 数据库文件（可选，会自动生成示例数据）
- ✅ `core_alpha_system.py` - 核心系统

### 3. 账号准备

- **GitHub 账号**：用于托管代码
- **Streamlit Cloud 账号**：访问 [share.streamlit.io](https://share.streamlit.io) 使用 GitHub 登录

---

## 🌐 Streamlit Cloud 部署步骤

### 步骤 1: 登录 Streamlit Cloud

1. 访问 [https://share.streamlit.io](https://share.streamlit.io)
2. 点击 **"Sign in with GitHub"**
3. 授权 Streamlit 访问你的 GitHub 仓库

### 步骤 2: 创建新应用

1. 点击 **"New app"** 按钮
2. 选择你的仓库：`你的用户名/sentiport`
3. 配置部署参数：
   - **Branch**: `main`
   - **Main file path**: `demo_streamlit.py`
   - **App URL**: 自定义你的应用地址（如 `sentiport-demo`）

### 步骤 3: 高级设置（可选）

点击 **"Advanced settings"** 进行配置：

#### Python 版本
```
Python version: 3.11
```

#### 环境变量（如果需要）
```
ALPHA_VANTAGE_API_KEY=你的API密钥
```

### 步骤 4: 部署

1. 点击 **"Deploy!"** 按钮
2. 等待构建完成（首次部署约 3-5 分钟）
3. 构建日志会实时显示，检查是否有错误

### 步骤 5: 验证部署

部署成功后：

1. 自动跳转到应用界面
2. 检查所有功能是否正常
3. 测试数据加载和图表展示

---

## 🔐 环境变量配置

### Alpha Vantage API 密钥（可选）

如果需要获取真实新闻数据，配置 API 密钥：

1. 访问 [Alpha Vantage](https://www.alphavantage.co/support/#api-key) 获取免费 API 密钥
2. 在 Streamlit Cloud 应用设置中添加环境变量：
   ```
   ALPHA_VANTAGE_API_KEY=你的密钥
   ```

### 配置方式

#### 方法 1: Streamlit Cloud Dashboard

1. 进入你的应用页面
2. 点击右上角 **"Settings"**
3. 选择 **"Secrets"**
4. 添加配置（TOML 格式）：

```toml
[api]
alpha_vantage_key = "你的API密钥"
```

#### 方法 2: 本地 .streamlit/secrets.toml（不推荐提交到 Git）

创建 `.streamlit/secrets.toml` 文件：

```toml
[api]
alpha_vantage_key = "你的API密钥"
```

**⚠️ 重要**: 确保 `.gitignore` 包含此文件！

---

## 💾 数据持久化方案

### 问题说明

Streamlit Cloud 的文件系统是**临时的**，每次重启应用时会重置。

### 解决方案

#### 方案 1: 使用 GitHub 提交的数据库（推荐用于演示）

✅ **当前方案 - 已配置**

```python
# demo_streamlit.py 中已实现
@st.cache_resource
def get_database_connection():
    db_path = Path("financial_data.db")
    if not db_path.exists():
        return None  # 会自动生成示例数据
    return sqlite3.connect(db_path, check_same_thread=False)
```

**优点**:
- 简单直接，无需额外配置
- 适合演示和展示用途
- 数据库文件 (2.1MB) 可以直接提交到 Git

**限制**:
- 数据不会自动更新
- 每次更新数据需要重新提交代码

#### 方案 2: 使用云数据库（生产环境推荐）

如果需要持久化和实时更新，可以使用：

**选项 A: SQLite 云存储**
- 使用 Deta Base 或 Turso
- 修改 `core_alpha_system.py` 中的连接字符串

**选项 B: PostgreSQL**
```bash
# requirements.txt 中添加
psycopg2-binary>=2.9.0

# 修改连接代码
import psycopg2
conn = psycopg2.connect(
    host=st.secrets["db"]["host"],
    database=st.secrets["db"]["name"],
    user=st.secrets["db"]["user"],
    password=st.secrets["db"]["password"]
)
```

#### 方案 3: API 模式（最灵活）

将数据获取改为 API 调用：

```python
# 示例代码
@st.cache_data(ttl=600)
def fetch_data_from_api():
    response = requests.get("https://your-api.com/data")
    return response.json()
```

### 当前配置

✅ 项目已配置为**自动降级模式**：

1. 优先使用 `financial_data.db` 中的数据
2. 如果数据库不存在或为空，自动生成演示数据
3. 演示数据逻辑完善，展示效果良好

---

## 🛠️ 故障排查

### 常见问题及解决方案

#### 1. 部署失败：依赖安装错误

**错误信息**:
```
ERROR: Could not find a version that satisfies the requirement torch
```

**解决方案**:
已在 `requirements.txt` 中移除大型依赖（torch, transformers）。如果仍有问题，检查版本约束：

```txt
# 确保版本范围合理
streamlit>=1.28.0,<2.0.0
pandas>=2.0.0,<3.0.0
```

#### 2. 应用启动超时

**原因**: 初始数据加载过慢

**解决方案**:
```python
# 使用 @st.cache_data 和 @st.cache_resource
@st.cache_data(ttl=600)
def load_data():
    # 你的数据加载逻辑
    pass
```

#### 3. 数据库连接错误

**错误信息**:
```
sqlite3.OperationalError: unable to open database file
```

**解决方案**:
- 确保 `financial_data.db` 已提交到 Git
- 或者让应用自动生成示例数据：

```python
if not db_path.exists():
    return generate_sample_data()
```

#### 4. NLTK 数据下载失败

**错误信息**:
```
LookupError: Resource vader_lexicon not found
```

**解决方案**:
在 `core_alpha_system.py` 中已添加自动下载：

```python
nltk.download('vader_lexicon', quiet=True)
```

如果仍有问题，可以预下载并提交到仓库：

```bash
# 本地执行
python -c "import nltk; nltk.download('vader_lexicon')"
```

#### 5. 内存不足

**错误信息**:
```
MemoryError: Unable to allocate array
```

**解决方案**:
- 减少数据加载量
- 优化缓存策略
- 使用 `ttl` 参数自动清理缓存：

```python
@st.cache_data(ttl=3600)  # 1小时后自动清理
def load_large_data():
    pass
```

---

## ⚡ 性能优化建议

### 1. 缓存策略

```python
# 静态数据 - 永久缓存
@st.cache_resource
def load_model():
    return expensive_model_loading()

# 动态数据 - 带TTL缓存
@st.cache_data(ttl=600)  # 10分钟
def load_market_data():
    return fetch_latest_data()
```

### 2. 异步加载

```python
# 使用 spinner 提升用户体验
with st.spinner('正在加载数据...'):
    data = load_data()
```

### 3. 分页加载

```python
# 避免一次性加载大量数据
@st.cache_data
def load_paginated_data(page, page_size=100):
    offset = page * page_size
    return query_with_limit(offset, page_size)
```

### 4. 压缩图表数据

```python
# 对大型时间序列进行采样
if len(df) > 1000:
    df = df.sample(1000).sort_index()
```

### 5. 懒加载

```python
# 只在需要时加载
if st.sidebar.button('加载详细数据'):
    detailed_data = load_detailed_data()
```

---

## 📊 部署后检查清单

部署完成后，确保以下功能正常：

- [ ] 主页正常显示
- [ ] 所有导航页面可访问
- [ ] 图表正确渲染
- [ ] 数据加载正常
- [ ] Alpha 因子计算正确
- [ ] 交易信号生成正常
- [ ] 新闻分析页面工作
- [ ] 移动端适配良好

---

## 🔗 有用的链接

- [Streamlit Cloud 官方文档](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit 论坛](https://discuss.streamlit.io/)
- [常见部署问题](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app#common-deployment-issues)
- [Alpha Vantage API](https://www.alphavantage.co/documentation/)

---

## 📧 支持

如果遇到问题：

1. 检查 Streamlit Cloud 的构建日志
2. 查看本文档的故障排查部分
3. 访问 [Streamlit 论坛](https://discuss.streamlit.io/) 寻求帮助

---

## 🎉 完成！

恭喜！你的 AlphaQuest 系统现在已部署到云端。

**你的应用地址**: `https://你的应用名.streamlit.app`

分享给你的朋友和同事，展示你的 NLP 驱动的量化交易系统！

---

*最后更新: 2025-01-12*
