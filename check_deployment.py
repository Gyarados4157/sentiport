#!/usr/bin/env python3
"""
部署前检查脚本
检查所有必需文件和配置是否就绪
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, required=True):
    """检查文件是否存在"""
    exists = Path(filepath).exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = "必需" if required else "可选"
    print(f"{status} {filepath} ({req_text})")
    return exists if required else True

def check_file_size(filepath, max_mb=10):
    """检查文件大小"""
    if not Path(filepath).exists():
        return True

    size_mb = Path(filepath).stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        print(f"⚠️  {filepath} 文件较大 ({size_mb:.1f} MB)，可能影响部署速度")
        return False
    return True

def check_requirements():
    """检查 requirements.txt 内容"""
    if not Path("requirements.txt").exists():
        return False

    with open("requirements.txt", "r") as f:
        lines = f.readlines()

    # 只检查非注释行
    package_lines = [
        line.strip().lower()
        for line in lines
        if line.strip() and not line.strip().startswith('#')
    ]

    # 检查是否包含大型依赖
    large_packages = ["torch", "tensorflow", "transformers"]
    found_large = []

    for pkg in large_packages:
        for line in package_lines:
            if line.startswith(pkg):
                found_large.append(pkg)
                break

    if found_large:
        print(f"⚠️  requirements.txt 包含大型依赖: {', '.join(found_large)}")
        print("   这可能导致部署失败或超时")
        return False

    return True

def check_gitignore():
    """检查 .gitignore 配置"""
    if not Path(".gitignore").exists():
        print("⚠️  .gitignore 不存在")
        return False

    with open(".gitignore", "r") as f:
        content = f.read()

    # 检查是否忽略敏感文件
    sensitive_patterns = [".env", "secrets.toml", "*.log"]
    missing = [p for p in sensitive_patterns if p not in content]

    if missing:
        print(f"⚠️  .gitignore 缺少模式: {', '.join(missing)}")
        return False

    return True

def main():
    """主检查流程"""
    print("=" * 60)
    print("🔍 Streamlit Cloud 部署前检查")
    print("=" * 60)
    print()

    all_ok = True

    # 1. 检查必需文件
    print("📁 检查必需文件:")
    required_files = [
        "demo_streamlit.py",
        "requirements.txt",
        ".streamlit/config.toml",
        "core_alpha_system.py"
    ]

    for filepath in required_files:
        if not check_file_exists(filepath, required=True):
            all_ok = False
    print()

    # 2. 检查可选文件
    print("📄 检查可选文件:")
    optional_files = [
        "financial_data.db",
        "DEPLOY.md",
        ".gitignore",
        ".env.example"
    ]

    for filepath in optional_files:
        check_file_exists(filepath, required=False)
    print()

    # 3. 检查文件大小
    print("📊 检查文件大小:")
    if Path("financial_data.db").exists():
        check_file_size("financial_data.db", max_mb=10)
    print()

    # 4. 检查 requirements.txt
    print("📦 检查依赖配置:")
    if not check_requirements():
        all_ok = False
    else:
        print("✅ requirements.txt 配置正常")
    print()

    # 5. 检查 .gitignore
    print("🔒 检查 Git 配置:")
    if not check_gitignore():
        all_ok = False
    else:
        print("✅ .gitignore 配置正常")
    print()

    # 6. 检查 Git 状态
    print("🔄 检查 Git 状态:")
    if os.path.exists(".git"):
        print("✅ Git 仓库已初始化")

        # 检查是否有 remote
        import subprocess
        try:
            result = subprocess.run(
                ["git", "remote", "-v"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout.strip():
                print("✅ Git remote 已配置")
            else:
                print("⚠️  Git remote 未配置")
                print("   运行: git remote add origin <你的仓库URL>")
        except:
            pass
    else:
        print("❌ Git 仓库未初始化")
        print("   运行: git init")
        all_ok = False
    print()

    # 总结
    print("=" * 60)
    if all_ok:
        print("✅ 所有检查通过！可以开始部署。")
        print()
        print("📚 下一步:")
        print("1. 提交更改: git add . && git commit -m 'Ready for deployment'")
        print("2. 推送到 GitHub: git push origin main")
        print("3. 访问 https://share.streamlit.io 部署")
        print("4. 参考 DEPLOY.md 获取详细步骤")
    else:
        print("❌ 检查失败，请修复上述问题后重试。")
        print()
        print("📚 参考文档: DEPLOY.md")
    print("=" * 60)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
