package com.example.croprecognition

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.EditText
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.croprecognition.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var recognitionEngine: CropRecognitionEngine? = null
    private var currentResults: List<RecognitionResult>? = null
    private var currentBitmap: Bitmap? = null
    private lateinit var dbHelper: CropDatabaseHelper
    private var isEnglish: Boolean = false

    // 默认后端服务器地址 (可在 App 内点击“⚙️ 服务器”按钮任意修改)
    private var serverBaseUrl: String = "http://10.0.2.2:8000"
    // 1. 从相册选择图片回调
    private val pickGalleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { handleSelectedImage(it) }
    }

    // 2. 拍照识别回调
    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap: Bitmap? ->
        bitmap?.let { processAndRecognize(it) }
    }

    // 3. 相机运行时权限申请回调
    private val requestCameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            takePictureLauncher.launch(null)
        } else {
            Toast.makeText(
                this,
                if (isEnglish) "Camera permission denied" else "相机权限被拒绝，无法拍照",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    // 当前选中的模型文件名 ("crop_model_int8.onnx" 或 "crop_model_fp32.onnx")
    private var selectedModelFileName: String = "crop_model_int8.onnx"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 初始化本地数据库与配置
        dbHelper = CropDatabaseHelper(this)
        loadServerConfig()
        loadModelConfig()

        // 异步初始化选中的本地 ONNX AI 模型
        initRecognitionEngine()

        // 按钮点击事件绑定
        binding.btnGallery.setOnClickListener {
            pickGalleryLauncher.launch("image/*")
        }

        binding.btnCamera.setOnClickListener {
            checkCameraPermissionAndLaunch()
        }

        // 结果纠错按钮绑定
        binding.btnFeedback.setOnClickListener {
            showCorrectionDialog(isUploadDirect = false)
        }

        // 上传样本库按钮绑定
        binding.btnUploadDataset.setOnClickListener {
            showCorrectionDialog(isUploadDirect = true)
        }

        // 🤖 模型选择按钮绑定
        binding.btnModelConfig.setOnClickListener {
            showModelSelectDialog()
        }

        // 服务器配置按钮绑定
        binding.btnServerConfig.setOnClickListener {
            showServerConfigDialog()
        }

        // 语言切换按钮绑定
        binding.btnLangToggle.setOnClickListener {
            isEnglish = !isEnglish
            updateLanguageUI()
        }

        updateLanguageUI()
    }

    private fun loadModelConfig() {
        val sp = getSharedPreferences("crop_app_config", Context.MODE_PRIVATE)
        selectedModelFileName = sp.getString("selected_model", "crop_model_int8.onnx") ?: "crop_model_int8.onnx"
    }

    private fun initRecognitionEngine() {
        lifecycleScope.launch {
            try {
                recognitionEngine?.close()
                recognitionEngine = CropRecognitionEngine(this@MainActivity, selectedModelFileName)
            } catch (e: Exception) {
                Toast.makeText(
                    this@MainActivity,
                    if (isEnglish) "Offline model init failed: ${e.message}" else "❌ 离线模型初始化失败: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
                e.printStackTrace()
            }
        }
    }

    private fun showModelSelectDialog() {
        val options = arrayOf(
            if (isEnglish) "⚡ INT8 Quantized (~293MB - Recommended for mobile)" else "⚡ INT8 量化轻量版 (~293MB·推荐手机使用)",
            if (isEnglish) "💎 FP32 Full Precision (~1.2GB - For flagship/PC)" else "💎 FP32 原生高精度版 (~1.2GB·适合高性能设备)"
        )

        val currentIdx = if (selectedModelFileName == "crop_model_fp32.onnx") 1 else 0

        AlertDialog.Builder(this)
            .setTitle(if (isEnglish) "🤖 Select AI Model Version" else "🤖 选择 AI 模型版本")
            .setSingleChoiceItems(options, currentIdx) { dialog, which ->
                val newModel = if (which == 1) "crop_model_fp32.onnx" else "crop_model_int8.onnx"
                if (newModel != selectedModelFileName) {
                    selectedModelFileName = newModel
                    getSharedPreferences("crop_app_config", Context.MODE_PRIVATE)
                        .edit().putString("selected_model", newModel).apply()

                    initRecognitionEngine()

                    Toast.makeText(
                        this,
                        if (isEnglish) "Model switched to: ${if (which == 1) "FP32 Native" else "INT8 Quantized"}"
                        else "✅ AI 模型已切换为: ${if (which == 1) "FP32 原生高精度版" else "INT8 量化轻量版"}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
                dialog.dismiss()
            }
            .setNegativeButton(if (isEnglish) "Cancel" else "取消", null)
            .show()
    }

    private fun loadServerConfig() {
        val sp = getSharedPreferences("crop_app_config", Context.MODE_PRIVATE)
        serverBaseUrl = sp.getString("server_url", "http://10.0.2.2:8000") ?: "http://10.0.2.2:8000"
        updateServerStatusBadge()
    }

    private fun updateServerStatusBadge() {
        binding.tvCurrentServerStatus.text = "🌐 API: $serverBaseUrl"
    }

    private fun showServerConfigDialog() {
        val etUrl = EditText(this).apply {
            setText(serverBaseUrl)
            hint = "http://192.168.1.100:8000 or http://your-cloud-ip:8000"
            setPadding(40, 30, 40, 30)
        }

        val title = if (isEnglish) "⚙️ Configure Backend Server URL" else "⚙️ 配置公网 / 局域网服务器地址"
        val message = if (isEnglish)
            "Enter your Python API URL for remote dataset collection:\n\n• Local WiFi: http://192.168.x.x:8000\n• Cloud Server: http://x.x.x.x:8000 or domain\n• Tunneling (cpolar/ngrok): https://xxx.cpolar.cn"
        else
            "请输入你的 Python API 服务器地址，支持全球远程联网收集样本：\n\n• 局域网/WiFi收集: http://192.168.x.x:8000\n• 云服务器/公网收集: http://云公网IP:8000 或域名\n• 内网穿透映射: https://xxx.cpolar.cn"

        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setView(etUrl)
            .setPositiveButton(if (isEnglish) "Save" else "保存配置") { _, _ ->
                val newUrl = etUrl.text.toString().trim()
                if (newUrl.startsWith("http://") || newUrl.startsWith("https://")) {
                    serverBaseUrl = newUrl
                    getSharedPreferences("crop_app_config", Context.MODE_PRIVATE)
                        .edit().putString("server_url", newUrl).apply()
                    updateServerStatusBadge()
                    Toast.makeText(
                        this,
                        if (isEnglish) "Server URL saved: $newUrl" else "✅ 服务器地址已保存: $newUrl",
                        Toast.LENGTH_LONG
                    ).show()
                } else {
                    Toast.makeText(
                        this,
                        if (isEnglish) "Invalid URL format! Must start with http:// or https://" else "❌ 无效格式！地址必须以 http:// 或 https:// 开头",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
            .setNegativeButton(if (isEnglish) "Cancel" else "取消", null)
            .show()
    }


    private fun checkCameraPermissionAndLaunch() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            takePictureLauncher.launch(null)
        } else {
            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun handleSelectedImage(uri: Uri) {
        try {
            val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { decoder, _, _ ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    decoder.isMutableRequired = true
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }
            processAndRecognize(bitmap)
        } catch (e: Exception) {
            Toast.makeText(
                this,
                if (isEnglish) "Failed to load image: ${e.message}" else "图片读取失败: ${e.message}",
                Toast.LENGTH_SHORT
            ).show()
            e.printStackTrace()
        }
    }

    private fun processAndRecognize(bitmap: Bitmap) {
        currentBitmap = bitmap
        binding.ivCropPreview.setImageBitmap(bitmap)
        binding.tvImagePlaceholder.visibility = View.GONE

        binding.layoutLoading.visibility = View.VISIBLE
        binding.cardResult.visibility = View.GONE
        binding.cardTopCandidates.visibility = View.GONE

        lifecycleScope.launch {
            try {
                val engine = recognitionEngine ?: run {
                    Toast.makeText(
                        this@MainActivity,
                        if (isEnglish) "Model loading, please wait..." else "⏳ 模型正在加载中，请稍后...",
                        Toast.LENGTH_SHORT
                    ).show()
                    return@launch
                }

                val results = engine.predictAsync(bitmap, topK = 3)
                if (results.isNotEmpty()) {
                    currentResults = results
                    displayResults(results)
                }
            } catch (e: Exception) {
                Toast.makeText(
                    this@MainActivity,
                    if (isEnglish) "Inference error: ${e.message}" else "识别异常: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
                e.printStackTrace()
            } finally {
                binding.layoutLoading.visibility = View.GONE
            }
        }
    }

    private fun displayResults(results: List<RecognitionResult>) {
        val top1 = results.first()
        val advice = top1.advice

        // 1. 渲染主卡片
        binding.tvCropName.text = "${advice.getCropName(isEnglish)} · ${advice.getStageName(isEnglish)}"
        val confPercent = (top1.confidence * 100).toInt()
        binding.tvConfidence.text = if (isEnglish) "Confidence: $confPercent%" else "置信度: $confPercent%"
        binding.tvDays.text = advice.getDaysInfo(isEnglish)

        binding.tvDescription.text = advice.getDescription(isEnglish)
        binding.tvWater.text = if (isEnglish) "💦 Water: ${advice.getWater(true)}" else "💦 水分管理: ${advice.getWater(false)}"
        binding.tvFertilizer.text = if (isEnglish) "🌱 Fertilizer: ${advice.getFertilizer(true)}" else "🌱 施肥调控: ${advice.getFertilizer(false)}"
        binding.tvOps.text = if (isEnglish) "🌾 Key Operations: ${advice.getOps(true)}" else "🌾 关键农事: ${advice.getOps(false)}"

        binding.cardResult.visibility = View.VISIBLE

        // 2. 渲染 Top-3 候选匹配列表
        if (results.size >= 1) {
            val r = results[0]
            binding.tvCandidate1.text = "🥇 1. ${r.advice.getCropName(isEnglish)}·${r.advice.getStageName(isEnglish)} (${(r.confidence * 100).toInt()}%)"
            binding.progressCandidate1.progress = (r.confidence * 100).toInt()
        }
        if (results.size >= 2) {
            val r = results[1]
            binding.tvCandidate2.text = "🥈 2. ${r.advice.getCropName(isEnglish)}·${r.advice.getStageName(isEnglish)} (${(r.confidence * 100).toInt()}%)"
            binding.progressCandidate2.progress = (r.confidence * 100).toInt()
        }
        if (results.size >= 3) {
            val r = results[2]
            binding.tvCandidate3.text = "🥉 3. ${r.advice.getCropName(isEnglish)}·${r.advice.getStageName(isEnglish)} (${(r.confidence * 100).toInt()}%)"
            binding.progressCandidate3.progress = (r.confidence * 100).toInt()
        }

        binding.cardTopCandidates.visibility = View.VISIBLE
    }

    /**
     * 弹出纠错与样本标注对话框
     */
    private fun showCorrectionDialog(isUploadDirect: Boolean) {
        val results = currentResults
        val bitmap = currentBitmap
        if (results.isNullOrEmpty() || bitmap == null) {
            Toast.makeText(
                this,
                if (isEnglish) "Please recognize an image first" else "请先拍照或选择图片进行识别",
                Toast.LENGTH_SHORT
            ).show()
            return
        }

        val top1 = results.first()
        val origCropCn = top1.advice.getCropName(false)
        val origStageCn = top1.advice.getStageName(false)

        val dialog = CorrectionDialog(
            context = this,
            currentCropCn = origCropCn,
            currentStageCn = origStageCn,
            isEnglish = isEnglish,
            onSaveLocal = { cropEn, stageEn, cropCn, stageCn, note, isCorrect ->
                saveFeedbackToLocal(bitmap, origCropCn, origStageCn, cropEn, stageEn, cropCn, stageCn, note, isCorrect)
            },
            onSaveAndUpload = { cropEn, stageEn, cropCn, stageCn, note, isCorrect ->
                saveFeedbackAndUpload(bitmap, origCropCn, origStageCn, cropEn, stageEn, cropCn, stageCn, note, isCorrect)
            }
        )
        dialog.show()
    }

    /**
     * 保存纠错记录至 Android 本地 SQLite 数据库
     */
    private fun saveFeedbackToLocal(
        bitmap: Bitmap,
        origCropCn: String,
        origStageCn: String,
        cropEn: String,
        stageEn: String,
        cropCn: String,
        stageCn: String,
        note: String,
        isCorrect: Int
    ) {
        try {
            val imageFile = DatasetUploader.saveBitmapToLocalFile(cacheDir, bitmap)
            val rowId = dbHelper.insertRecord(
                imagePath = imageFile.absolutePath,
                originalCrop = origCropCn,
                originalStage = origStageCn,
                correctedCrop = "$cropCn ($cropEn)",
                correctedStage = "$stageCn ($stageEn)",
                isCorrected = isCorrect,
                userNote = note,
                isUploaded = 0
            )

            val msg = if (isEnglish)
                "✅ Correction saved to local database (ID: $rowId)!"
            else
                "✅ 纠错结果已成功存入本地数据库 (记录 ID: $rowId)！"
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

        } catch (e: Exception) {
            val msg = if (isEnglish) "Failed to save local database: ${e.message}" else "❌ 本地保存失败: ${e.message}"
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }
    }

    /**
     * 保存至本地 SQLite 数据库并上传回 Python 后端样本库
     */
    private fun saveFeedbackAndUpload(
        bitmap: Bitmap,
        origCropCn: String,
        origStageCn: String,
        cropEn: String,
        stageEn: String,
        cropCn: String,
        stageCn: String,
        note: String,
        isCorrect: Int
    ) {
        val imageFile = DatasetUploader.saveBitmapToLocalFile(cacheDir, bitmap)
        val rowId = dbHelper.insertRecord(
            imagePath = imageFile.absolutePath,
            originalCrop = origCropCn,
            originalStage = origStageCn,
            correctedCrop = "$cropCn ($cropEn)",
            correctedStage = "$stageCn ($stageEn)",
            isCorrected = isCorrect,
            userNote = note,
            isUploaded = 0
        )

        Toast.makeText(
            this,
            if (isEnglish) "⚡ Saved locally, uploading to server ($serverBaseUrl)..." else "⚡ 本地已保存，正在远程上传至服务器 ($serverBaseUrl)...",
            Toast.LENGTH_SHORT
        ).show()

        // 异步协程联网上传回后端
        lifecycleScope.launch {
            val result = DatasetUploader.uploadSampleToBackend(
                serverBaseUrl = serverBaseUrl,
                imageFile = imageFile,
                cropEn = cropEn,
                stageEn = stageEn,
                userNote = note,
                isCorrect = isCorrect
            )

            result.onSuccess { respText ->
                dbHelper.updateUploadStatus(rowId, 1)
                val msg = if (isEnglish)
                    "🎉 Upload successful! Sample added to dataset dataset/user_feedback/${cropEn}_${stageEn}"
                else
                    "🎉 远程上传成功！样本已扩充至服务器数据集 dataset/user_feedback/${cropEn}_${stageEn}"
                Toast.makeText(this@MainActivity, msg, Toast.LENGTH_LONG).show()
            }.onFailure { err ->
                val msg = if (isEnglish)
                    "⚠️ Local DB saved, but upload failed: ${err.message}\n(Check server URL at $serverBaseUrl)"
                else
                    "⚠️ 本地已保存，但联网上传失败: ${err.message}\n(请在顶部点击“⚙️ 服务器”确认地址是否连通: $serverBaseUrl)"
                Toast.makeText(this@MainActivity, msg, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun updateLanguageUI() {
        if (isEnglish) {
            binding.btnLangToggle.text = "🌐 中文"
            binding.btnServerConfig.text = "⚙️ Server"
            binding.tvAppTitle.text = "🌾 Crop Growth Stage AI"
            binding.tvAppSubtitle.text = "CLIP+LoRA Offline Engine · Corn / Wheat / Cotton"
            binding.tvImagePlaceholder.text = "📷 Tap below to capture or pick a crop image"
            binding.btnCamera.text = "📷 Camera"
            binding.btnGallery.text = "🖼️ Gallery"
            binding.tvLoadingText.text = "⚡ Executing offline ONNX AI inference..."
            binding.tvHeaderDesc.text = "💡 Stage Characteristics"
            binding.tvHeaderAdvice.text = "🌿 Agronomic Management Advice"
            binding.tvHeaderCandidates.text = "📈 Top-3 Candidate Matching"
            binding.btnFeedback.text = "✏️ Feedback"
            binding.btnUploadDataset.text = "☁️ Upload Dataset"
        } else {
            binding.btnLangToggle.text = "🌐 English"
            binding.btnServerConfig.text = "⚙️ 服务器"
            binding.tvAppTitle.text = "🌾 农作物生育期智能诊断"
            binding.tvAppSubtitle.text = "CLIP+LoRA 离线 NPU 引擎 · 玉米 / 小麦 / 棉花"
            binding.tvImagePlaceholder.text = "📷 点击下方按钮拍摄或选择农作物图片"
            binding.btnCamera.text = "📷 拍照识别"
            binding.btnGallery.text = "🖼️ 相册选择"
            binding.tvLoadingText.text = "⚡ 正在执行离线 ONNX AI 推理..."
            binding.tvHeaderDesc.text = "💡 阶段形态特征描述"
            binding.tvHeaderAdvice.text = "🌿 智能农艺养护建议"
            binding.tvHeaderCandidates.text = "📈 Top-3 候选匹配分析"
            binding.btnFeedback.text = "✏️ 结果纠错"
            binding.btnUploadDataset.text = "☁️ 上传样本库"
        }

        currentResults?.let { displayResults(it) }
    }

    override fun onDestroy() {
        super.onDestroy()
        recognitionEngine?.close()
    }
}
