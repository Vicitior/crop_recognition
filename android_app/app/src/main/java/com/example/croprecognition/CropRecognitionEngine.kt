package com.example.croprecognition

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

data class RecognitionResult(
    val className: String,
    val confidence: Float,
    val advice: AgronomicAdvice
)

class CropRecognitionEngine(context: Context, modelFileName: String = "crop_model_int8.onnx") {
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession

    // 15 类阶段名称（顺序必须与 PyTorch 模型训练的 index 完全对齐）
    private val classList = listOf(
        "corn_seedling", "corn_jointing", "corn_tasseling", "corn_filling", "corn_maturity",
        "wheat_seedling", "wheat_tillering", "wheat_jointing", "wheat_heading", "wheat_maturity",
        "cotton_seedling", "cotton_squaring", "cotton_flowering", "cotton_boll_setting", "cotton_boll_opening"
    )

    init {
        // 1. 将 293MB 大模型文件从小块 Buffer 拷贝到应用内部存储，防止 Java 堆内存溢出 (OOM)
        val modelFile = File(context.filesDir, modelFileName)
        if (!modelFile.exists() || modelFile.length() < 100 * 1024 * 1024) {
            context.assets.open(modelFileName).use { inputStream ->
                FileOutputStream(modelFile).use { outputStream ->
                    val buffer = ByteArray(64 * 1024) // 64KB 小缓冲区
                    var bytesRead: Int
                    while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                        outputStream.write(buffer, 0, bytesRead)
                    }
                    outputStream.flush()
                }
            }
        }

        // 2. 使用直接文件路径创建 Session，ONNX Runtime 在 C++ 层通过 mmap 映射文件，0 Java 堆内存占用
        val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setInterOpNumThreads(2)
            setIntraOpNumThreads(4)
        }
        ortSession = ortEnv.createSession(modelFile.absolutePath, sessionOptions)
    }

    /**
     * 在后台协程线程中异步运行离线 AI 推理，防止卡顿 UI
     */
    suspend fun predictAsync(bitmap: Bitmap, topK: Int = 3): List<RecognitionResult> = withContext(Dispatchers.Default) {
        val floatBuffer = ImageUtils.bitmapToFloatBuffer(bitmap)
        val shape = longArrayOf(1, 3, 336, 336)

        val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, shape)
        val inputs = mapOf("pixel_values" to inputTensor)

        val results = ortSession.run(inputs)
        val outputTensor = results.get(0).value as Array<FloatArray>
        val probs = outputTensor[0]

        val sortedIndices = probs.indices.sortedByDescending { probs[it] }.take(topK)

        val finalResults = mutableListOf<RecognitionResult>()
        for (idx in sortedIndices) {
            val className = classList[idx]
            val advice = AgronomicKnowledge.getAdvice(className)
            finalResults.add(
                RecognitionResult(
                    className = className,
                    confidence = probs[idx],
                    advice = advice
                )
            )
        }

        inputTensor.close()
        results.close()
        return@withContext finalResults
    }

    fun close() {
        ortSession.close()
        ortEnv.close()
    }
}
