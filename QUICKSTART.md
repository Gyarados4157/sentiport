# 🚀 快速部署指南

5 分钟将 SentiPort 部署到 Streamlit Cloud！

## 📝 前置条件

- ✅ GitHub 账号
- ✅ Git 已安装

## 🎯 部署步骤

### 1️⃣ 运行部署检查

```bash
python check_deployment.py
```

确保所有检查通过。

### 2️⃣ 推送到 GitHub

```bash
# 初始化 Git（如果还没有）
git init

# 添加所有文件
git add .

# 提交
git commit -m "Ready for Streamlit Cloud deployment"

# 创建并推送到 GitHub
git remote add origin https://github.com/你的用户名/sentiport.git
git branch -M main
git push -u origin main
```

### 3️⃣ 部署到 Streamlit Cloud

1. 访问 [https://share.streamlit.io](https://share.streamlit.io)
2. 使用 GitHub 登录
3. 点击 **"New app"**
4. 选择你的仓库和分支
5. 设置参数：
   - **Main file**: `demo_streamlit.py`
   - **Python version**: `3.11`
6. 点击 **"Deploy"**

### 4️⃣ 等待部署完成

⏱️ 首次部署约 3-5 分钟

✅ 部署成功后，你将获得一个公开 URL：
```
https://你的应用名.streamlit.app
```

## 🎉 完成！

你的 NLP 驱动的量化交易系统现已在线！

---

## 📚 更多信息

- **详细部署指南**: 查看 [DEPLOY.md](DEPLOY.md)
- **故障排查**: [DEPLOY.md#故障排查](DEPLOY.md#故障排查)
- **API 配置**: [DEPLOY.md#环境变量配置](DEPLOY.md#环境变量配置)

## 🆘 需要帮助？

- [Streamlit 社区论坛](https://discuss.streamlit.io/)
- [项目 Issues](https://github.com/你的用户名/sentiport/issues)

---

**祝部署顺利！** 🎊
