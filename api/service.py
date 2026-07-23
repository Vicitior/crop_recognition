# -*- coding: utf-8 -*-
"""
服务层 - 业务逻辑
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from PIL import Image
from torchvision import transforms

from api.database import RecordDAO, init_db
from api.models import (
    RecognizeResponse, RecognitionResult,
    FeedbackRequest, FeedbackResponse,
    RecordListResponse, RecordItem,
    StatsResponse, ExportResponse,
)
from models.growth_stages import CLASS_MAP, CROP_INFO


# ==================== 模型管理 ====================

class ModelService:
    """模型服务 - 管理模型加载和推理"""

    def __init__(self):
        self.model = None
        self.class_names = None
        self.device = None
        self.model_path = None
        self._load_model()

    def _find_model_path(self) -> Optional[str]:
        """查找可用的模型文件"""
        candidates = [
            "saved_models/clip/hot_reload/best.pth",
            "saved_models/clip/clip-vit-large-patch14-336-v2/best.pth",
            "saved_models/clip/clip-large-336/best.pth",
            "saved_models/clip/clip-large/best.pth",
            "saved_models/clip/best.pth",
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _load_model(self):
        """加载模型"""
        model_path = self._find_model_path()
        if not model_path:
            print("[WARN] 未找到模型文件，识别功能不可用")
            return

        try:
            from scripts.train_clip_v2 import CLIPWithClassifier, apply_lora
            from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            model_name = checkpoint.get("model_name", "openai/clip-vit-large-patch14-336")
            self.class_names = checkpoint["class_names"]
            num_classes = len(self.class_names)
            method = checkpoint.get("method", "lora")

            # 推断 LoRA rank
            lora_rank = 8
            for key, val in checkpoint["model_state_dict"].items():
                if "lora_A" in key:
                    lora_rank = val.shape[0]
                    break

            # 加载 CLIP 模型
            if "siglip" in model_name.lower():
                clip_model = AutoModel.from_pretrained(model_name)
            else:
                clip_model = CLIPModel.from_pretrained(model_name)

            if method == "lora":
                clip_model = apply_lora(clip_model, rank=lora_rank)

            img_size = 336 if "336" in model_name else 224
            self.model = CLIPWithClassifier(clip_model, num_classes, img_size=img_size)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()

            self.model_path = model_path
            print(f"[OK] 模型加载成功: {model_path}")
            print(f"     设备: {self.device}, 类别数: {num_classes}")

        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            self.model = None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """识别图片"""
        if self.model is None:
            raise RuntimeError("模型未加载")

        # 图像预处理
        model_name = ""
        if self.model_path:
            model_name = self.model_path

        img_size = 336 if "336" in model_name else 224
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)

        top_probs, top_indices = probs[0].topk(min(3, len(self.class_names)))

        results = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.class_names[idx.item()]
            info = CLASS_MAP.get(class_name, {})

            results.append(RecognitionResult(
                class_name=class_name,
                crop=info.get("crop_en", class_name.split("_")[0]),
                stage=info.get("stage_en", class_name.split("_")[1] if "_" in class_name else "unknown"),
                crop_cn=info.get("crop_cn", class_name.split("_")[0]),
                stage_cn=info.get("stage_cn", class_name.split("_")[1] if "_" in class_name else "unknown"),
                confidence=round(prob.item(), 4),
            ))

        return {
            "top1": results[0],
            "top3": results,
        }

    def reload(self):
        """重新加载模型"""
        self.model = None
        self.class_names = None
        self._load_model()


# ==================== 记录服务 ====================

class RecordService:
    """记录服务 - 处理识别和反馈的业务逻辑"""

    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        init_db()

    def _save_image(self, image: Image.Image, filename: str) -> str:
        """保存图片到本地"""
        # 按日期分目录
        date_dir = self.upload_dir / datetime.now().strftime("%Y%m%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        # 生成唯一文件名
        timestamp = datetime.now().strftime("%H%M%S_%f")
        ext = Path(filename).suffix if filename else ".jpg"
        save_name = f"{timestamp}{ext}"
        save_path = date_dir / save_name

        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(save_path, "JPEG", quality=95)

        return str(save_path)

    def recognize(self, image: Image.Image, filename: str = "upload.jpg") -> RecognizeResponse:
        """识别图片并保存记录"""
        # 保存图片
        image_path = self._save_image(image, filename)

        # 模型识别
        result = self.model_service.predict(image)
        top1 = result["top1"]
        top3 = result["top3"]

        # 保存到数据库
        record_id = RecordDAO.create({
            "image_path": image_path,
            "image_filename": filename,
            "model_crop": top1.crop,
            "model_stage": top1.stage,
            "model_confidence": top1.confidence,
            "model_top3": [r.dict() for r in top3],
            "is_correct": 1,  # 默认认为正确
        })

        return RecognizeResponse(
            record_id=record_id,
            image_path=image_path,
            top1=top1,
            top3=top3,
        )

    def feedback(self, record_id: int, feedback: FeedbackRequest) -> FeedbackResponse:
        """用户反馈（修改识别结果）"""
        # 检查记录是否存在
        record = RecordDAO.get_by_id(record_id)
        if not record:
            raise ValueError(f"记录不存在: {record_id}")

        # 更新记录
        success = RecordDAO.update(record_id, {
            "user_crop": feedback.user_crop,
            "user_stage": feedback.user_stage,
            "user_note": feedback.user_note,
            "is_correct": feedback.is_correct,
        })

        if not success:
            raise RuntimeError("更新失败")

        return FeedbackResponse(
            record_id=record_id,
            message="修改成功" if feedback.is_correct == 0 else "已确认正确",
        )

    def get_record(self, record_id: int) -> Optional[RecordItem]:
        """查询单条记录"""
        record = RecordDAO.get_by_id(record_id)
        if record:
            return RecordItem(**record)
        return None

    def get_records(
        self,
        page: int = 1,
        page_size: int = 20,
        crop: Optional[str] = None,
        stage: Optional[str] = None,
        is_correct: Optional[int] = None,
    ) -> RecordListResponse:
        """查询记录列表"""
        result = RecordDAO.get_list(
            page=page,
            page_size=page_size,
            crop=crop,
            stage=stage,
            is_correct=is_correct,
        )
        return RecordListResponse(
            total=result["total"],
            page=result["page"],
            page_size=result["page_size"],
            records=[RecordItem(**r) for r in result["records"]],
        )

    def delete_record(self, record_id: int) -> bool:
        """删除记录"""
        return RecordDAO.delete(record_id)

    def get_stats(self) -> StatsResponse:
        """获取统计信息"""
        stats = RecordDAO.get_stats()
        return StatsResponse(**stats)

    def export_data(self, mark_exported: bool = True) -> ExportResponse:
        """导出数据为 JSON 文件"""
        records = RecordDAO.get_unexported()

        if not records:
            return ExportResponse(
                success=False,
                count=0,
                file_path="",
                message="没有可导出的数据",
            )

        # 保存为 JSON
        export_dir = Path("exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = export_dir / f"records_{timestamp}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        # 标记已导出
        if mark_exported:
            ids = [r["id"] for r in records]
            RecordDAO.mark_exported(ids)

        return ExportResponse(
            count=len(records),
            file_path=str(file_path),
            message=f"成功导出 {len(records)} 条记录",
        )


# ==================== 全局实例 ====================

# 模型服务（单例）
model_service = ModelService()

# 记录服务
record_service = RecordService(model_service)
