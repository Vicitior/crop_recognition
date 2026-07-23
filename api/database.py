# -*- coding: utf-8 -*-
"""
数据库模块 - SQLite + DAO 模式
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

# 数据库路径
DB_PATH = Path("data/crop_recognition.db")


def get_db_path():
    """获取数据库路径，确保目录存在"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return str(DB_PATH)


@contextmanager
def get_connection():
    """获取数据库连接（上下文管理器）"""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row  # 返回字典格式
    conn.execute("PRAGMA journal_mode=WAL")  # 提高并发性能
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """初始化数据库表"""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                -- 图片信息
                image_path TEXT NOT NULL,
                image_filename TEXT,

                -- 模型识别结果
                model_crop TEXT NOT NULL,
                model_stage TEXT NOT NULL,
                model_confidence REAL NOT NULL,
                model_top3 TEXT,  -- JSON格式的top3结果

                -- 用户修正结果（可为空，表示用户未修改）
                user_crop TEXT,
                user_stage TEXT,
                user_note TEXT,

                -- 状态
                is_correct INTEGER DEFAULT 1,  -- 1=正确, 0=已修改
                is_exported INTEGER DEFAULT 0,  -- 是否已导出

                -- 时间
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)

        # 创建索引
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_created ON records(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_crop ON records(model_crop)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_exported ON records(is_exported)")


class RecordDAO:
    """记录数据访问对象"""

    @staticmethod
    def create(data: Dict[str, Any]) -> int:
        """创建记录，返回ID"""
        with get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO records (
                    image_path, image_filename,
                    model_crop, model_stage, model_confidence, model_top3,
                    user_crop, user_stage, user_note,
                    is_correct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["image_path"],
                data.get("image_filename"),
                data["model_crop"],
                data["model_stage"],
                data["model_confidence"],
                json.dumps(data.get("model_top3", []), ensure_ascii=False),
                data.get("user_crop"),
                data.get("user_stage"),
                data.get("user_note"),
                data.get("is_correct", 1),
            ))
            return cursor.lastrowid

    @staticmethod
    def get_by_id(record_id: int) -> Optional[Dict[str, Any]]:
        """根据ID查询记录"""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM records WHERE id = ?", (record_id,)
            ).fetchone()
            if row:
                return dict(row)
            return None

    @staticmethod
    def get_list(
        page: int = 1,
        page_size: int = 20,
        crop: Optional[str] = None,
        stage: Optional[str] = None,
        is_correct: Optional[int] = None,
        is_exported: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """分页查询记录列表"""
        conditions = []
        params = []

        if crop:
            conditions.append("(model_crop = ? OR user_crop = ?)")
            params.extend([crop, crop])
        if stage:
            conditions.append("(model_stage = ? OR user_stage = ?)")
            params.extend([stage, stage])
        if is_correct is not None:
            conditions.append("is_correct = ?")
            params.append(is_correct)
        if is_exported is not None:
            conditions.append("is_exported = ?")
            params.append(is_exported)
        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with get_connection() as conn:
            # 查询总数
            count = conn.execute(
                f"SELECT COUNT(*) FROM records WHERE {where_clause}", params
            ).fetchone()[0]

            # 查询分页数据
            offset = (page - 1) * page_size
            rows = conn.execute(
                f"SELECT * FROM records WHERE {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [page_size, offset]
            ).fetchall()

            return {
                "total": count,
                "page": page,
                "page_size": page_size,
                "records": [dict(row) for row in rows],
            }

    @staticmethod
    def update(record_id: int, data: Dict[str, Any]) -> bool:
        """更新记录"""
        fields = []
        params = []

        for key in ["user_crop", "user_stage", "user_note", "is_correct", "is_exported"]:
            if key in data:
                fields.append(f"{key} = ?")
                params.append(data[key])

        if not fields:
            return False

        fields.append("updated_at = datetime('now', 'localtime')")
        params.append(record_id)

        with get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE records SET {', '.join(fields)} WHERE id = ?", params
            )
            return cursor.rowcount > 0

    @staticmethod
    def delete(record_id: int) -> bool:
        """删除记录"""
        with get_connection() as conn:
            cursor = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
            return cursor.rowcount > 0

    @staticmethod
    def get_unexported() -> List[Dict[str, Any]]:
        """获取未导出的记录"""
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM records WHERE is_exported = 0 ORDER BY created_at"
            ).fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def mark_exported(record_ids: List[int]) -> int:
        """标记记录为已导出"""
        if not record_ids:
            return 0
        placeholders = ",".join(["?"] * len(record_ids))
        with get_connection() as conn:
            cursor = conn.execute(
                f"UPDATE records SET is_exported = 1, updated_at = datetime('now', 'localtime') WHERE id IN ({placeholders})",
                record_ids
            )
            return cursor.rowcount

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """获取统计信息"""
        with get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            correct = conn.execute("SELECT COUNT(*) FROM records WHERE is_correct = 1").fetchone()[0]
            modified = conn.execute("SELECT COUNT(*) FROM records WHERE is_correct = 0").fetchone()[0]
            unexported = conn.execute("SELECT COUNT(*) FROM records WHERE is_exported = 0").fetchone()[0]

            # 按作物统计
            crop_stats = conn.execute("""
                SELECT
                    COALESCE(user_crop, model_crop) as crop,
                    COUNT(*) as count
                FROM records
                GROUP BY crop
                ORDER BY count DESC
            """).fetchall()

            return {
                "total": total,
                "correct": correct,
                "modified": modified,
                "unexported": unexported,
                "by_crop": [dict(row) for row in crop_stats],
            }
