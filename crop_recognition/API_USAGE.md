# API 接口调用指南

## 基础信息

- **Base URL**: `http://your-server:8000`
- **API 文档**: `http://your-server:8000/docs` (Swagger UI)
- **数据格式**: JSON

---

## 1. 识别图片

### cURL
```bash
curl -X POST "http://localhost:8000/api/recognize" \
  -F "file=@/path/to/crop_image.jpg"
```

### Python
```python
import requests

url = "http://localhost:8000/api/recognize"

with open("crop_image.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

data = response.json()
print(data)
```

### JavaScript (fetch)
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/api/recognize', {
    method: 'POST',
    body: formData
});
const data = await response.json();
```

### 响应示例
```json
{
    "success": true,
    "record_id": 1,
    "image_path": "uploads/20260609/120000_123456.jpg",
    "top1": {
        "class_name": "corn_jointing",
        "crop": "corn",
        "stage": "jointing",
        "crop_cn": "玉米",
        "stage_cn": "拔节期",
        "confidence": 0.8562
    },
    "top3": [
        {
            "class_name": "corn_jointing",
            "crop": "corn",
            "stage": "jointing",
            "crop_cn": "玉米",
            "stage_cn": "拔节期",
            "confidence": 0.8562
        },
        {
            "class_name": "corn_tasseling",
            "crop": "corn",
            "stage": "tasseling",
            "crop_cn": "玉米",
            "stage_cn": "抽穗期",
            "confidence": 0.1023
        },
        {
            "class_name": "corn_seedling",
            "crop": "corn",
            "stage": "seedling",
            "crop_cn": "玉米",
            "stage_cn": "出苗期",
            "confidence": 0.0415
        }
    ]
}
```

---

## 2. 用户修改识别结果

当识别结果不正确时，用户可以修正：

### cURL
```bash
curl -X PUT "http://localhost:8000/api/records/1" \
  -H "Content-Type: application/json" \
  -d '{
    "user_crop": "wheat",
    "user_stage": "heading",
    "user_note": "这是小麦抽穗期",
    "is_correct": 0
  }'
```

### Python
```python
import requests

url = "http://localhost:8000/api/records/1"  # record_id = 1

data = {
    "user_crop": "wheat",        # 作物类型
    "user_stage": "heading",     # 生长阶段
    "user_note": "这是小麦抽穗期", # 备注（可选）
    "is_correct": 0              # 0=已修改, 1=确认原结果正确
}

response = requests.put(url, json=data)
print(response.json())
```

### 可选的作物和阶段值

| 作物 (user_crop) | 阶段 (user_stage) |
|------------------|-------------------|
| corn | seedling, jointing, tasseling, filling, maturity |
| wheat | seedling, tillering, jointing, heading, maturity |
| cotton | seedling, squaring, flowering, boll_setting, boll_opening |

### 响应示例
```json
{
    "success": true,
    "record_id": 1,
    "message": "修改成功"
}
```

---

## 3. 查询历史记录

### cURL
```bash
# 查询所有记录（分页）
curl "http://localhost:8000/api/records?page=1&page_size=20"

# 按作物筛选
curl "http://localhost:8000/api/records?crop=corn"

# 按是否修改筛选
curl "http://localhost:8000/api/records?is_correct=0"
```

### Python
```python
import requests

# 基础查询
response = requests.get("http://localhost:8000/api/records", params={
    "page": 1,
    "page_size": 20,
    "crop": "corn",        # 可选：按作物筛选
    "stage": "seedling",   # 可选：按阶段筛选
    "is_correct": 0        # 可选：0=已修改, 1=正确
})

data = response.json()
print(f"总数: {data['total']}")
for record in data['records']:
    print(f"  [{record['id']}] {record['model_crop']} - {record['model_stage']}")
```

### 响应示例
```json
{
    "total": 100,
    "page": 1,
    "page_size": 20,
    "records": [
        {
            "id": 1,
            "image_path": "uploads/20260609/120000_123456.jpg",
            "image_filename": "crop.jpg",
            "model_crop": "corn",
            "model_stage": "jointing",
            "model_confidence": 0.8562,
            "model_top3": "[...]",
            "user_crop": "wheat",
            "user_stage": "heading",
            "user_note": "修正为小麦",
            "is_correct": 0,
            "is_exported": 0,
            "created_at": "2026-06-09 12:00:00",
            "updated_at": "2026-06-09 12:05:00"
        }
    ]
}
```

---

## 4. 查询单条记录

### cURL
```bash
curl "http://localhost:8000/api/records/1"
```

### Python
```python
response = requests.get("http://localhost:8000/api/records/1")
record = response.json()
```

---

## 5. 删除记录

### cURL
```bash
curl -X DELETE "http://localhost:8000/api/records/1"
```

### Python
```python
response = requests.delete("http://localhost:8000/api/records/1")
print(response.json())  # {"success": true, "message": "删除成功"}
```

---

## 6. 获取统计信息

### cURL
```bash
curl "http://localhost:8000/api/stats"
```

### Python
```python
response = requests.get("http://localhost:8000/api/stats")
stats = response.json()
print(f"总记录: {stats['total']}")
print(f"正确: {stats['correct']}")
print(f"已修改: {stats['modified']}")
print(f"未导出: {stats['unexported']}")
```

### 响应示例
```json
{
    "total": 100,
    "correct": 85,
    "modified": 15,
    "unexported": 50,
    "by_crop": [
        {"crop": "corn", "count": 45},
        {"crop": "wheat", "count": 35},
        {"crop": "cotton", "count": 20}
    ]
}
```

---

## 7. 导出数据

### cURL
```bash
# 导出并标记为已导出
curl -X POST "http://localhost:8000/api/export?mark_exported=true"

# 导出但不标记
curl -X POST "http://localhost:8000/api/export?mark_exported=false"
```

### Python
```python
response = requests.post("http://localhost:8000/api/export", params={
    "mark_exported": True
})
result = response.json()
print(f"导出了 {result['count']} 条记录")
print(f"文件路径: {result['file_path']}")
```

### 响应示例
```json
{
    "success": true,
    "count": 50,
    "file_path": "exports/records_20260609_120000.json",
    "message": "成功导出 50 条记录"
}
```

---

## 8. 健康检查

### cURL
```bash
curl "http://localhost:8000/health"
```

### 响应示例
```json
{
    "status": "ok",
    "model_loaded": true,
    "model_path": "saved_models/clip/clip-vit-large-patch14-336-v2/best.pth"
}
```

---

## 完整调用流程示例

### Python 完整示例

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. 识别图片
def recognize_image(image_path):
    with open(image_path, "rb") as f:
        response = requests.post(f"{BASE_URL}/api/recognize", files={"file": f})
    return response.json()

# 2. 修改识别结果
def update_record(record_id, crop, stage, note=""):
    response = requests.put(f"{BASE_URL}/api/records/{record_id}", json={
        "user_crop": crop,
        "user_stage": stage,
        "user_note": note,
        "is_correct": 0
    })
    return response.json()

# 3. 查询记录
def get_records(page=1, page_size=20, crop=None):
    params = {"page": page, "page_size": page_size}
    if crop:
        params["crop"] = crop
    response = requests.get(f"{BASE_URL}/api/records", params=params)
    return response.json()

# 4. 获取统计
def get_stats():
    response = requests.get(f"{BASE_URL}/api/stats")
    return response.json()

# 5. 导出数据
def export_data():
    response = requests.post(f"{BASE_URL}/api/export")
    return response.json()


# 使用示例
if __name__ == "__main__":
    # 识别
    result = recognize_image("test.jpg")
    print(f"识别结果: {result['top1']['crop_cn']} - {result['top1']['stage_cn']}")
    print(f"置信度: {result['top1']['confidence']:.2%}")
    
    record_id = result['record_id']
    
    # 如果识别错误，修改
    if result['top1']['confidence'] < 0.5:
        update_record(record_id, "wheat", "heading", "低置信度，手动修正")
    
    # 查看统计
    stats = get_stats()
    print(f"\n总记录: {stats['total']}")
```

### JavaScript 完整示例

```javascript
const BASE_URL = 'http://localhost:8000';

// 1. 识别图片
async function recognizeImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${BASE_URL}/api/recognize`, {
        method: 'POST',
        body: formData
    });
    return response.json();
}

// 2. 修改识别结果
async function updateRecord(recordId, crop, stage, note = '') {
    const response = await fetch(`${BASE_URL}/api/records/${recordId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            user_crop: crop,
            user_stage: stage,
            user_note: note,
            is_correct: 0
        })
    });
    return response.json();
}

// 3. 查询记录
async function getRecords(page = 1, pageSize = 20, crop = null) {
    let url = `${BASE_URL}/api/records?page=${page}&page_size=${pageSize}`;
    if (crop) url += `&crop=${crop}`;
    
    const response = await fetch(url);
    return response.json();
}

// 4. 获取统计
async function getStats() {
    const response = await fetch(`${BASE_URL}/api/stats`);
    return response.json();
}

// 使用示例
document.getElementById('upload-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('file-input');
    const result = await recognizeImage(fileInput.files[0]);
    
    document.getElementById('result').innerText = 
        `${result.top1.crop_cn} - ${result.top1.stage_cn} (${(result.top1.confidence * 100).toFixed(1)}%)`;
});
```

---

## 错误处理

### 常见错误码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 记录不存在 |
| 500 | 服务器内部错误 |
| 503 | 模型未加载 |

### 错误响应格式
```json
{
    "success": false,
    "error": "错误信息",
    "detail": "详细错误描述"
}
```

### Python 错误处理
```python
try:
    response = requests.post(f"{BASE_URL}/api/recognize", files={"file": f})
    response.raise_for_status()  # 如果状态码不是 2xx，抛出异常
    result = response.json()
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
except Exception as e:
    print(f"处理失败: {e}")
```
