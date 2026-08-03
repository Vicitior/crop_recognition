package com.example.croprecognition

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.view.View
import android.view.Window
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast

class CorrectionDialog(
    context: Context,
    private val currentCropCn: String,
    private val currentStageCn: String,
    private val isEnglish: Boolean,
    private val onSaveLocal: (cropEn: String, stageEn: String, cropCn: String, stageCn: String, note: String, isCorrect: Int) -> Unit,
    private val onSaveAndUpload: (cropEn: String, stageEn: String, cropCn: String, stageCn: String, note: String, isCorrect: Int) -> Unit
) : Dialog(context) {

    private val cropOptions = listOf(
        Pair("玉米 (Corn)", "corn"),
        Pair("小麦 (Wheat)", "wheat"),
        Pair("棉花 (Cotton)", "cotton")
    )

    private val stageOptionsMap = mapOf(
        "corn" to listOf(
            Pair("出苗期 (Seedling)", "seedling"),
            Pair("拔节期 (Jointing)", "jointing"),
            Pair("抽穗期 (Tasseling)", "tasseling"),
            Pair("灌浆期 (Filling)", "filling"),
            Pair("成熟期 (Maturity)", "maturity")
        ),
        "wheat" to listOf(
            Pair("出苗期 (Seedling)", "seedling"),
            Pair("分蘖期 (Tillering)", "tillering"),
            Pair("拔节期 (Jointing)", "jointing"),
            Pair("抽穗期 (Heading)", "heading"),
            Pair("成熟期 (Maturity)", "maturity")
        ),
        "cotton" to listOf(
            Pair("苗期 (Seedling)", "seedling"),
            Pair("蕾期 (Squaring)", "squaring"),
            Pair("开花期 (Flowering)", "flowering"),
            Pair("结铃期 (Boll Setting)", "boll_setting"),
            Pair("吐絮期 (Boll Opening)", "boll_opening")
        )
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        
        // 创建动态 Dialog 布局
        val view = createDialogView()
        setContentView(view)

        window?.apply {
            setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
            setLayout(
                (context.resources.displayMetrics.widthPixels * 0.92).toInt(),
                WindowManager.LayoutParams.WRAP_CONTENT
            )
        }
    }

    private fun createDialogView(): View {
        val layout = android.widget.LinearLayout(context).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
            background = android.graphics.drawable.GradientDrawable().apply {
                setColor(Color.parseColor("#FFFFFF"))
                cornerRadius = 32f
            }
        }

        // 标题
        val tvTitle = TextView(context).apply {
            text = if (isEnglish) "✏️ Correct Crop & Stage Result" else "✏️ 识别结果纠错与样本标注"
            textSize = 18f
            setTextColor(Color.parseColor("#064E3B"))
            setTypeface(null, android.graphics.Typeface.BOLD)
        }
        layout.addView(tvTitle)

        // 提示
        val tvSub = TextView(context).apply {
            text = if (isEnglish) 
                "Original Result: $currentCropCn · $currentStageCn\nSelect correct crop and stage below:"
            else 
                "原识别结果: $currentCropCn · $currentStageCn\n请在下方校准正确的作物与生育期:"
            textSize = 13f
            setTextColor(Color.parseColor("#64748B"))
            setPadding(0, 12, 0, 20)
        }
        layout.addView(tvSub)

        // 1. 作物选择 Label + Spinner
        val tvCropLabel = TextView(context).apply {
            text = if (isEnglish) "🌾 Select Correct Crop:" else "🌾 正确的农作物类型:"
            textSize = 14f
            setTextColor(Color.parseColor("#047857"))
            setTypeface(null, android.graphics.Typeface.BOLD)
        }
        layout.addView(tvCropLabel)

        val spinnerCrop = Spinner(context).apply {
            setPadding(12, 16, 12, 16)
        }
        val cropAdapter = ArrayAdapter(
            context,
            android.R.layout.simple_spinner_dropdown_item,
            cropOptions.map { it.first }
        )
        spinnerCrop.adapter = cropAdapter
        layout.addView(spinnerCrop)

        // 2. 生育期选择 Label + Spinner
        val tvStageLabel = TextView(context).apply {
            text = if (isEnglish) "🌱 Select Correct Growth Stage:" else "🌱 正确的生育期阶段:"
            textSize = 14f
            setTextColor(Color.parseColor("#047857"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(0, 20, 0, 0)
        }
        layout.addView(tvStageLabel)

        val spinnerStage = Spinner(context).apply {
            setPadding(12, 16, 12, 16)
        }
        layout.addView(spinnerStage)

        // 联动更新生育期 Spinner 选项
        var currentStageList = stageOptionsMap["corn"]!!
        spinnerCrop.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, v: View?, position: Int, id: Long) {
                val cropKey = cropOptions[position].second
                currentStageList = stageOptionsMap[cropKey] ?: stageOptionsMap["corn"]!!
                val stageAdapter = ArrayAdapter(
                    context,
                    android.R.layout.simple_spinner_dropdown_item,
                    currentStageList.map { it.first }
                )
                spinnerStage.adapter = stageAdapter
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // 3. 备注输入框
        val etNote = EditText(context).apply {
            hint = if (isEnglish) "Optional Note (e.g. Backlight / Insect damage)" else "可选备注 (如: 暗光侧面照/光照强)"
            textSize = 13f
            setPadding(24, 20, 24, 20)
            background = android.graphics.drawable.GradientDrawable().apply {
                setColor(Color.parseColor("#F8FAFC"))
                setStroke(2, Color.parseColor("#CBD5E1"))
                cornerRadius = 16f
            }
        }
        val paramsNote = android.widget.LinearLayout.LayoutParams(
            android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
            android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { setMargins(0, 24, 0, 24) }
        layout.addView(etNote, paramsNote)

        // 4. 确认无误 Checkbox
        val cbVerified = CheckBox(context).apply {
            text = if (isEnglish) "Verify Result is Correct (Confirm Top-1)" else "确认 AI 预测无误 (确认为准)"
            textSize = 13f
            setTextColor(Color.parseColor("#334155"))
        }
        layout.addView(cbVerified)

        // 5. 按钮操作区 (仅保存本地 vs 保存并上传)
        val layoutButtons = android.widget.LinearLayout(context).apply {
            orientation = android.widget.LinearLayout.HORIZONTAL
            setPadding(0, 24, 0, 0)
        }

        val btnLocal = Button(context).apply {
            text = if (isEnglish) "💾 Save Local" else "💾 保存本地"
            textSize = 13f
            setTextColor(Color.parseColor("#059669"))
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#ECFDF5"))
        }
        val btnParams1 = android.widget.LinearLayout.LayoutParams(0, 110, 1f).apply {
            marginEnd = 12
        }
        layoutButtons.addView(btnLocal, btnParams1)

        val btnUpload = Button(context).apply {
            text = if (isEnglish) "☁️ Save & Upload" else "☁️ 保存并上传样本库"
            textSize = 13f
            setTextColor(Color.WHITE)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.parseColor("#059669"))
        }
        val btnParams2 = android.widget.LinearLayout.LayoutParams(0, 110, 1f).apply {
            marginStart = 12
        }
        layoutButtons.addView(btnUpload, btnParams2)

        layout.addView(layoutButtons)

        // 事件监听
        btnLocal.setOnClickListener {
            val cropIdx = spinnerCrop.selectedItemPosition
            val stageIdx = spinnerStage.selectedItemPosition
            val cropPair = cropOptions.getOrElse(cropIdx) { cropOptions[0] }
            val stagePair = currentStageList.getOrElse(stageIdx) { currentStageList[0] }

            val cropEn = cropPair.second
            val stageEn = stagePair.second
            val cropCn = cropPair.first.split(" ")[0]
            val stageCn = stagePair.first.split(" ")[0]
            val note = etNote.text.toString().trim()
            val isCorrect = if (cbVerified.isChecked) 1 else 0

            onSaveLocal(cropEn, stageEn, cropCn, stageCn, note, isCorrect)
            dismiss()
        }

        btnUpload.setOnClickListener {
            val cropIdx = spinnerCrop.selectedItemPosition
            val stageIdx = spinnerStage.selectedItemPosition
            val cropPair = cropOptions.getOrElse(cropIdx) { cropOptions[0] }
            val stagePair = currentStageList.getOrElse(stageIdx) { currentStageList[0] }

            val cropEn = cropPair.second
            val stageEn = stagePair.second
            val cropCn = cropPair.first.split(" ")[0]
            val stageCn = stagePair.first.split(" ")[0]
            val note = etNote.text.toString().trim()
            val isCorrect = if (cbVerified.isChecked) 1 else 0

            onSaveAndUpload(cropEn, stageEn, cropCn, stageCn, note, isCorrect)
            dismiss()
        }

        return layout
    }
}
