# -*- coding: utf-8 -*-
"""
数据模型 - Pydantic schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ==================== 识别结果 ====================

class RecognitionResult(BaseModel):
    """单个识别结果"""
    class_name: str = Field(..., description="类别名称，如 corn_seedling")
    crop: str = Field(..., description="作物类型，如 corn")
    stage: str = Field(..., description="生长阶段，如 seedling")
    crop_cn: str = Field(..., description="作物中文名，如 玉米")
    stage_cn: str = Field(..., description="阶段中文名，如 出苗期")
    confidence: float = Field(..., description="置信度 0-1")


class RecognizeResponse(BaseModel):
    """识别接口响应"""
    success: bool = True
    record_id: int = Field(..., description="记录ID，用于后续修改")
    image_path: str = Field(..., description="图片保存路径")
    top1: RecognitionResult = Field(..., description="最可能的结果")
    top3: List[RecognitionResult] = Field(..., description="Top3结果")


# ==================== 用户反馈 ====================

class FeedbackRequest(BaseModel):
    """用户反馈（修改识别结果）"""
    user_crop: str = Field(..., description="用户选择的作物类型，如 corn")
    user_stage: str = Field(..., description="用户选择的生长阶段，如 seedling")
    user_note: Optional[str] = Field(None, description="用户备注")
    is_correct: int = Field(0, description="0=已修改, 1=原结果正确")


class FeedbackResponse(BaseModel):
    """反馈接口响应"""
    success: bool = True
    record_id: int
    message: str = "修改成功"


# ==================== 记录查询 ====================

class RecordItem(BaseModel):
    """单条记录"""
    id: int
    image_path: str
    image_filename: Optional[str] = None

    # 模型识别
    model_crop: str
    model_stage: str
    model_confidence: float
    model_top3: Optional[str] = None  # JSON字符串

    # 用户修正
    user_crop: Optional[str] = None
    user_stage: Optional[str] = None
    user_note: Optional[str] = None

    # 状态
    is_correct: int = 1
    is_exported: int = 0

    # 时间
    created_at: str
    updated_at: str

    # 计算属性
    @property
    def final_crop(self) -> str:
        """最终作物类型（用户修改优先）"""
        return self.user_crop if self.user_crop else self.model_crop

    @property
    def final_stage(self) -> str:
        """最终生长阶段（用户修改优先）"""
        return self.user_stage if self.user_stage else self.model_stage


class RecordListResponse(BaseModel):
    """记录列表响应"""
    total: int
    page: int
    page_size: int
    records: List[RecordItem]


# ==================== 统计 ====================

class CropStat(BaseModel):
    """作物统计"""
    crop: str
    count: int


class StatsResponse(BaseModel):
    """统计响应"""
    total: int
    correct: int
    modified: int
    unexported: int
    by_crop: List[CropStat]


# ==================== 导出 ====================

class ExportResponse(BaseModel):
    """导出响应"""
    success: bool = True
    count: int = Field(..., description="导出记录数")
    file_path: str = Field(..., description="导出文件路径")
    message: str = "导出成功"


# ==================== 通用响应 ====================

class MessageResponse(BaseModel):
    """通用消息响应"""
    success: bool = True
    message: str


class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = False
    error: str
    detail: Optional[str] = None
