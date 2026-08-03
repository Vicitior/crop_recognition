package com.example.croprecognition

import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

object DatasetUploader {

    /**
     * 将 Bitmap 图片保存至 Android 应用局部缓存沙盒
     */
    fun saveBitmapToLocalFile(cacheDir: File, bitmap: Bitmap): File {
        val feedbackDir = File(cacheDir, "crop_feedback")
        if (!feedbackDir.exists()) {
            feedbackDir.mkdirs()
        }
        val file = File(feedbackDir, "sample_${System.currentTimeMillis()}_${UUID.randomUUID().toString().take(6)}.jpg")
        FileOutputStream(file).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 92, out)
        }
        return file
    }

    /**
     * 通过 HTTP Multipart 表单上传纠错样本图片与标注元数据至后端 FastAPI 服务器
     */
    suspend fun uploadSampleToBackend(
        serverBaseUrl: String,
        imageFile: File,
        cropEn: String,
        stageEn: String,
        userNote: String,
        isCorrect: Int
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val urlString = if (serverBaseUrl.endsWith("/")) {
                "${serverBaseUrl}api/feedback/upload"
            } else {
                "${serverBaseUrl}/api/feedback/upload"
            }

            val boundary = "---AndroidCropBoundary${System.currentTimeMillis()}"
            val lineEnd = "\r\n"
            val twoHyphens = "--"

            val url = URL(urlString)
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.connectTimeout = 10000
            conn.readTimeout = 15000
            conn.doInput = true
            conn.doOutput = true
            conn.useCaches = false
            conn.setRequestProperty("Connection", "Keep-Alive")
            conn.setRequestProperty("Content-Type", "multipart/form-boundary;boundary=$boundary")

            DataOutputStream(conn.outputStream).use { dos ->
                // 1. 发送 crop 文本字段
                dos.writeBytes("$twoHyphens$boundary$lineEnd")
                dos.writeBytes("Content-Disposition: form-data; name=\"crop\"$lineEnd$lineEnd")
                dos.writeBytes(cropEn)
                dos.writeBytes(lineEnd)

                // 2. 发送 stage 文本字段
                dos.writeBytes("$twoHyphens$boundary$lineEnd")
                dos.writeBytes("Content-Disposition: form-data; name=\"stage\"$lineEnd$lineEnd")
                dos.writeBytes(stageEn)
                dos.writeBytes(lineEnd)

                // 3. 发送 user_note 文本字段
                dos.writeBytes("$twoHyphens$boundary$lineEnd")
                dos.writeBytes("Content-Disposition: form-data; name=\"user_note\"$lineEnd$lineEnd")
                dos.write(userNote.toByteArray(Charsets.UTF_8))
                dos.writeBytes(lineEnd)

                // 4. 发送 is_correct 文本字段
                dos.writeBytes("$twoHyphens$boundary$lineEnd")
                dos.writeBytes("Content-Disposition: form-data; name=\"is_correct\"$lineEnd$lineEnd")
                dos.writeBytes(isCorrect.toString())
                dos.writeBytes(lineEnd)

                // 5. 发送 file 图片二进制
                dos.writeBytes("$twoHyphens$boundary$lineEnd")
                dos.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"${imageFile.name}\"$lineEnd")
                dos.writeBytes("Content-Type: image/jpeg$lineEnd$lineEnd")

                imageFile.inputStream().use { input ->
                    val buffer = ByteArray(4096)
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        dos.write(buffer, 0, bytesRead)
                    }
                }
                dos.writeBytes(lineEnd)

                // 结束 boundary
                dos.writeBytes("$twoHyphens$boundary$twoHyphens$lineEnd")
                dos.flush()
            }

            val responseCode = conn.responseCode
            if (responseCode == HttpURLConnection.HTTP_OK) {
                val responseText = conn.inputStream.bufferedReader().use { it.readText() }
                Result.success(responseText)
            } else {
                val errorText = conn.errorStream?.bufferedReader()?.use { it.readText() } ?: "HTTP $responseCode"
                Result.failure(Exception("上传失败 ($responseCode): $errorText"))
            }

        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
