package com.example.croprecognition

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class FeedbackRecord(
    val id: Long = 0,
    val imagePath: String,
    val originalCrop: String,
    val originalStage: String,
    val correctedCrop: String,
    val correctedStage: String,
    val isCorrected: Int, // 0 = 已纠错修改, 1 = 确认正确
    val userNote: String,
    val timestamp: String,
    val isUploaded: Int // 0 = 未上传, 1 = 已上传
)

class CropDatabaseHelper(context: Context) : SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {

    companion object {
        private const val DATABASE_NAME = "crop_feedback.db"
        private const val DATABASE_VERSION = 1

        const val TABLE_FEEDBACK = "feedback_records"
        const val COLUMN_ID = "id"
        const val COLUMN_IMAGE_PATH = "image_path"
        const val COLUMN_ORIGINAL_CROP = "original_crop"
        const val COLUMN_ORIGINAL_STAGE = "original_stage"
        const val COLUMN_CORRECTED_CROP = "corrected_crop"
        const val COLUMN_CORRECTED_STAGE = "corrected_stage"
        const val COLUMN_IS_CORRECTED = "is_corrected"
        const val COLUMN_USER_NOTE = "user_note"
        const val COLUMN_TIMESTAMP = "timestamp"
        const val COLUMN_IS_UPLOADED = "is_uploaded"
    }

    override fun onCreate(db: SQLiteDatabase) {
        val createTableQuery = """
            CREATE TABLE $TABLE_FEEDBACK (
                $COLUMN_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                $COLUMN_IMAGE_PATH TEXT NOT NULL,
                $COLUMN_ORIGINAL_CROP TEXT NOT NULL,
                $COLUMN_ORIGINAL_STAGE TEXT NOT NULL,
                $COLUMN_CORRECTED_CROP TEXT NOT NULL,
                $COLUMN_CORRECTED_STAGE TEXT NOT NULL,
                $COLUMN_IS_CORRECTED INTEGER NOT NULL DEFAULT 0,
                $COLUMN_USER_NOTE TEXT,
                $COLUMN_TIMESTAMP TEXT NOT NULL,
                $COLUMN_IS_UPLOADED INTEGER NOT NULL DEFAULT 0
            )
        """.trimIndent()
        db.execSQL(createTableQuery)
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS $TABLE_FEEDBACK")
        onCreate(db)
    }

    /**
     * 插入一条用户纠错/确认记录
     */
    fun insertRecord(
        imagePath: String,
        originalCrop: String,
        originalStage: String,
        correctedCrop: String,
        correctedStage: String,
        isCorrected: Int,
        userNote: String,
        isUploaded: Int = 0
    ): Long {
        val db = writableDatabase
        val timeStr = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())

        val values = ContentValues().apply {
            put(COLUMN_IMAGE_PATH, imagePath)
            put(COLUMN_ORIGINAL_CROP, originalCrop)
            put(COLUMN_ORIGINAL_STAGE, originalStage)
            put(COLUMN_CORRECTED_CROP, correctedCrop)
            put(COLUMN_CORRECTED_STAGE, correctedStage)
            put(COLUMN_IS_CORRECTED, isCorrected)
            put(COLUMN_USER_NOTE, userNote)
            put(COLUMN_TIMESTAMP, timeStr)
            put(COLUMN_IS_UPLOADED, isUploaded)
        }
        return db.insert(TABLE_FEEDBACK, null, values)
    }

    /**
     * 更新上传状态
     */
    fun updateUploadStatus(recordId: Long, isUploaded: Int) {
        val db = writableDatabase
        val values = ContentValues().apply {
            put(COLUMN_IS_UPLOADED, isUploaded)
        }
        db.update(TABLE_FEEDBACK, values, "$COLUMN_ID = ?", arrayOf(recordId.toString()))
    }

    /**
     * 查询所有本地纠错记录
     */
    fun getAllRecords(): List<FeedbackRecord> {
        val list = mutableListOf<FeedbackRecord>()
        val db = readableDatabase
        val cursor = db.query(
            TABLE_FEEDBACK,
            null, null, null, null, null,
            "$COLUMN_ID DESC"
        )
        cursor.use { c ->
            while (c.moveToNext()) {
                val record = FeedbackRecord(
                    id = c.getLong(c.getColumnIndexOrThrow(COLUMN_ID)),
                    imagePath = c.getString(c.getColumnIndexOrThrow(COLUMN_IMAGE_PATH)),
                    originalCrop = c.getString(c.getColumnIndexOrThrow(COLUMN_ORIGINAL_CROP)),
                    originalStage = c.getString(c.getColumnIndexOrThrow(COLUMN_ORIGINAL_STAGE)),
                    correctedCrop = c.getString(c.getColumnIndexOrThrow(COLUMN_CORRECTED_CROP)),
                    correctedStage = c.getString(c.getColumnIndexOrThrow(COLUMN_CORRECTED_STAGE)),
                    isCorrected = c.getInt(c.getColumnIndexOrThrow(COLUMN_IS_CORRECTED)),
                    userNote = c.getString(c.getColumnIndexOrThrow(COLUMN_USER_NOTE)) ?: "",
                    timestamp = c.getString(c.getColumnIndexOrThrow(COLUMN_TIMESTAMP)),
                    isUploaded = c.getInt(c.getColumnIndexOrThrow(COLUMN_IS_UPLOADED))
                )
                list.add(record)
            }
        }
        return list
    }
}
