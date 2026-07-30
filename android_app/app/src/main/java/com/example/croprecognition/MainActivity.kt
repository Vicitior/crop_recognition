package com.example.croprecognition

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.croprecognition.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var recognitionEngine: CropRecognitionEngine? = null
    private var currentResults: List<RecognitionResult>? = null
    private var isEnglish: Boolean = false // 语言状态控制 (默认中文)

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

    // 3. 相机运行时权限申请回调 (解决拍照闪退)
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 异步初始化本地 ONNX 推理引擎
        lifecycleScope.launch {
            try {
                recognitionEngine = CropRecognitionEngine(this@MainActivity, "crop_model_int8.onnx")
            } catch (e: Exception) {
                Toast.makeText(
                    this@MainActivity,
                    if (isEnglish) "Offline model init failed: ${e.message}" else "❌ 离线模型初始化失败: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
                e.printStackTrace()
            }
        }

        // 按钮点击事件绑定
        binding.btnGallery.setOnClickListener {
            pickGalleryLauncher.launch("image/*")
        }

        binding.btnCamera.setOnClickListener {
            checkCameraPermissionAndLaunch()
        }

        // 语言切换按钮绑定
        binding.btnLangToggle.setOnClickListener {
            isEnglish = !isEnglish
            updateLanguageUI()
        }

        updateLanguageUI()
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

    private fun updateLanguageUI() {
        if (isEnglish) {
            binding.btnLangToggle.text = "🌐 中文"
            binding.tvAppTitle.text = "🌾 Crop Growth Stage AI"
            binding.tvAppSubtitle.text = "CLIP+LoRA Offline Engine · Corn / Wheat / Cotton"
            binding.tvImagePlaceholder.text = "📷 Tap below to capture or pick a crop image"
            binding.btnCamera.text = "📷 Camera"
            binding.btnGallery.text = "🖼️ Gallery"
            binding.tvLoadingText.text = "⚡ Executing offline ONNX AI inference..."
            binding.tvHeaderDesc.text = "💡 Stage Characteristics"
            binding.tvHeaderAdvice.text = "🌿 Agronomic Management Advice"
            binding.tvHeaderCandidates.text = "📈 Top-3 Candidate Matching"
        } else {
            binding.btnLangToggle.text = "🌐 English"
            binding.tvAppTitle.text = "🌾 农作物生育期智能诊断"
            binding.tvAppSubtitle.text = "CLIP+LoRA 离线 NPU 引擎 · 玉米 / 小麦 / 棉花"
            binding.tvImagePlaceholder.text = "📷 点击下方按钮拍摄或选择农作物图片"
            binding.btnCamera.text = "📷 拍照识别"
            binding.btnGallery.text = "🖼️ 相册选择"
            binding.tvLoadingText.text = "⚡ 正在执行离线 ONNX AI 推理..."
            binding.tvHeaderDesc.text = "💡 阶段形态特征描述"
            binding.tvHeaderAdvice.text = "🌿 智能农艺养护建议"
            binding.tvHeaderCandidates.text = "📈 Top-3 候选匹配分析"
        }

        // 如果已有结果，重新刷新文字
        currentResults?.let { displayResults(it) }
    }

    override fun onDestroy() {
        super.onDestroy()
        recognitionEngine?.close()
    }
}
