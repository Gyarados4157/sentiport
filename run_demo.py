#!/usr/bin/env python3
"""
SentiPort V2 Demo运行脚本
快速演示Alpha因子系统的核心功能
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo_run.log')
        ]
    )

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'pandas', 'numpy', 'streamlit', 'yfinance', 
        'nltk', 'transformers', 'torch', 'plotly',
        'sklearn', 'requests'  # scikit-learn导入时使用sklearn
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def run_core_demo():
    """运行核心系统Demo"""
    try:
        print("\n🚀 启动SentiPort V2 核心系统Demo...")
        
        from core_alpha_system import main_demo
        
        # 运行Demo
        results, performance = main_demo()
        
        print("\n📊 Demo结果摘要:")
        print(f"- 生成交易信号: {len(results)} 个")
        print(f"- 平均IC: {performance['ic_mean']:.4f}")
        print(f"- 信息比率: {performance['ir']:.4f}")
        print(f"- 胜率: {performance['hit_rate']:.2%}")
        
        print("\n💡 交易信号详情:")
        for _, row in results.iterrows():
            signal_emoji = "🟢" if row['combined_alpha'] > 0.1 else "🔴" if row['combined_alpha'] < -0.1 else "🟡"
            print(f"{signal_emoji} {row['ticker']}: Alpha={row['combined_alpha']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 核心系统Demo运行失败: {e}")
        logging.error(f"Core demo failed: {e}", exc_info=True)
        return False

def run_streamlit_app():
    """启动Streamlit应用"""
    try:
        print("\n🌐 启动Streamlit Web应用...")
        print("浏览器将自动打开: http://localhost:8501")
        print("按 Ctrl+C 停止服务")
        
        os.system("streamlit run app.py")
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ Streamlit应用启动失败: {e}")

def main():
    """主函数"""
    setup_logging()
    
    print("=" * 60)
    print("🎯 SentiPort - NLP驱动的量化交易系统")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 选择运行模式
    print("\n选择运行模式:")
    print("1. 🖥️  只运行核心Demo (命令行)")
    print("2. 🌐 只启动Web界面")
    print("3. 🚀 先运行Demo，再启动Web界面")
    print("4. ❌ 退出")
    
    try:
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == "1":
            success = run_core_demo()
            if success:
                print("\n✅ 核心Demo运行完成")
                print("💡 提示: 运行选项2可查看Web界面的详细可视化")
        
        elif choice == "2":
            run_streamlit_app()
        
        elif choice == "3":
            success = run_core_demo()
            if success:
                print("\n✅ 核心Demo完成，即将启动Web界面...")
                input("按Enter键继续...")
                run_streamlit_app()
        
        elif choice == "4":
            print("👋 再见!")
            return
        
        else:
            print("❌ 无效选择，请重新运行")
    
    except KeyboardInterrupt:
        print("\n👋 用户取消操作")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        logging.error(f"Main execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()