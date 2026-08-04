# -*- coding: utf-8 -*-
"""
API 接口服务 - FastAPI
"""

import io
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image

from api.models import (
    RecognizeResponse, FeedbackRequest, FeedbackResponse,
    RecordListResponse, RecordItem,
    StatsResponse, ExportResponse,
    MessageResponse, ErrorResponse,
)
from api.service import model_service, record_service

from fastapi.middleware.cors import CORSMiddleware

# 创建 FastAPI 应用
app = FastAPI(
    title="农作物生长阶段识别 API",
    description="基于 CLIP 的农作物生长阶段识别系统，支持图片识别、结果修正、数据管理",
    version="1.0.0",
)

# 配置 CORS 允许跨域请求（支持微信小程序 / Web UI）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 识别接口 ====================

@app.post("/api/recognize", response_model=RecognizeResponse, tags=["识别"])
async def recognize(
    file: UploadFile = File(..., description="农作物图片"),
):
    """
    上传图片识别农作物生长阶段

    - 支持 JPG、PNG 等常见格式
    - 返回 Top3 识别结果和记录ID
    - 记录ID用于后续用户修正
    """
    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")

    try:
        # 读取图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 识别
        result = record_service.recognize(image, file.filename)
        return result

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")


# ==================== 反馈接口 ====================

@app.post("/api/feedback/upload", tags=["反馈"])
async def upload_feedback_sample(
    file: UploadFile = File(..., description="纠错后的样本图片"),
    crop: str = Form(..., description="正确的作物英文名，如 corn, wheat, cotton"),
    stage: str = Form(..., description="正确的生育期英文名，如 seedling, jointing"),
    user_note: str = Form("", description="用户备注或纠错说明"),
    is_correct: int = Form(0, description="是否已修正，0=已修改, 1=确认"),
):
    """
    用户（如 Android 移动端）直接上传纠错/确认后的样本图片与分类标签，
    存入项目数据集 `dataset/user_feedback/<crop>_<stage>/` 目录以扩充训练样本量。
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        res = record_service.save_feedback_sample(
            image=image,
            filename=file.filename or "uploaded_sample.jpg",
            crop=crop,
            stage=stage,
            user_note=user_note,
            is_correct=is_correct
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存样本失败: {str(e)}")


@app.put("/api/records/{record_id}", response_model=FeedbackResponse, tags=["反馈"])
async def update_record(
    record_id: int,
    feedback: FeedbackRequest,
):
    """
    用户修正识别结果

    - 传入用户认为正确的作物类型和生长阶段
    - is_correct=0 表示原结果错误，已修改
    - is_correct=1 表示原结果正确，确认
    """
    try:
        result = record_service.feedback(record_id, feedback)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ==================== 记录查询接口 ====================

@app.get("/api/records", response_model=RecordListResponse, tags=["记录"])
async def get_records(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    crop: Optional[str] = Query(None, description="按作物类型筛选，如 corn"),
    stage: Optional[str] = Query(None, description="按生长阶段筛选，如 seedling"),
    is_correct: Optional[int] = Query(None, description="是否已修正，0=已修改, 1=正确"),
):
    """
    查询历史记录列表

    - 支持分页
    - 支持按作物类型、生长阶段筛选
    - 支持按是否修正筛选
    """
    return record_service.get_records(
        page=page,
        page_size=page_size,
        crop=crop,
        stage=stage,
        is_correct=is_correct,
    )


@app.get("/api/records/{record_id}", response_model=RecordItem, tags=["记录"])
async def get_record(record_id: int):
    """
    查询单条记录详情
    """
    record = record_service.get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")
    return record


@app.delete("/api/records/{record_id}", response_model=MessageResponse, tags=["记录"])
async def delete_record(record_id: int):
    """
    删除记录
    """
    success = record_service.delete_record(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="记录不存在")
    return MessageResponse(message="删除成功")


# ==================== 统计接口 ====================

@app.get("/api/stats", response_model=StatsResponse, tags=["统计"])
async def get_stats():
    """
    获取统计信息

    - 总记录数
    - 正确/修改数量
    - 按作物分类统计
    """
    return record_service.get_stats()


# ==================== 导出接口 ====================

@app.post("/api/export", response_model=ExportResponse, tags=["导出"])
async def export_data(
    mark_exported: bool = Query(True, description="导出后是否标记为已导出"),
):
    """
    导出未导出的数据为 JSON 文件

    - 导出后记录标记为已导出
    - 返回导出文件路径
    """
    return record_service.export_data(mark_exported=mark_exported)


@app.get("/api/export/download", tags=["导出"])
async def download_export(file_path: str = Query(..., description="导出文件路径")):
    """
    下载导出的文件
    """
    import os
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path, filename=os.path.basename(file_path))


# ==================== 模型管理接口 ====================

@app.post("/api/model/reload", response_model=MessageResponse, tags=["模型"])
async def reload_model():
    """
    重新加载模型

    - 用于热更新模型后刷新
    """
    try:
        model_service.reload()
        return MessageResponse(message="模型重新加载成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


# ==================== 健康检查 ====================

@app.get("/health", tags=["系统"])
async def health_check():
    """
    健康检查
    """
    return {
        "status": "ok",
        "model_loaded": model_service.model is not None,
        "model_path": model_service.model_path,
    }


@app.get("/", tags=["系统"])
async def root():
    """
    API 根路径
    """
    return {
        "name": "农作物生长阶段识别 API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
