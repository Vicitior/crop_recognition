# -*- coding: utf-8 -*-
"""
启动 API 服务

用法:
    python run_api.py
    python run_api.py --port 8000 --host 0.0.0.0
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="启动农作物识别 API 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式（热重载）")
    args = parser.parse_args()

    print("=" * 50)
    print("  农作物生长阶段识别 API")
    print("=" * 50)
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  文档: http://{args.host}:{args.port}/docs")
    print("=" * 50)

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
