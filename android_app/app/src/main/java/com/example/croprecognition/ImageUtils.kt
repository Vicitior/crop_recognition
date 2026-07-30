package com.example.croprecognition

import android.graphics.Bitmap
import java.nio.FloatBuffer

object ImageUtils {
    private const val INPUT_SIZE = 336
    private val MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
    private val STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

    /**
     * 将 Bitmap 图片调整为 336x336 尺寸，并进行 ImageNet 标准归一化，
     * 转换为 NCHW [1, 3, 336, 336] 格式的 FloatBuffer 给 ONNX Runtime 使用
     */
    fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        // 防止硬件加速位图 (HARDWARE Bitmap) 导致 getPixels 抛出 IllegalArgumentException 闪退
        val softwareBitmap = if (bitmap.config == Bitmap.Config.HARDWARE) {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            bitmap
        }

        val scaledBitmap = Bitmap.createScaledBitmap(softwareBitmap, INPUT_SIZE, INPUT_SIZE, true)
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaledBitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val buffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)
        buffer.rewind()

        // NCHW (Channel R -> G -> B)
        for (c in 0..2) {
            for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
                val pixel = intValues[i]
                val channelVal = when (c) {
                    0 -> (pixel shr 16 and 0xFF) / 255.0f // R
                    1 -> (pixel shr 8 and 0xFF) / 255.0f  // G
                    else -> (pixel and 0xFF) / 255.0f     // B
                }
                val normalized = (channelVal - MEAN[c]) / STD[c]
                buffer.put(normalized)
            }
        }
        buffer.rewind()
        return buffer
    }
}
