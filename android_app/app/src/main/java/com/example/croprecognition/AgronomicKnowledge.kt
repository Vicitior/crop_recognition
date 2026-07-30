package com.example.croprecognition

data class AgronomicAdvice(
    val cropCn: String,
    val cropEn: String,
    val stageCn: String,
    val stageEn: String,
    val stageDays: String,
    val totalDays: String,
    val descCn: String,
    val descEn: String,
    val waterCn: String,
    val waterEn: String,
    val fertilizerCn: String,
    val fertilizerEn: String,
    val pestCn: String,
    val pestEn: String
) {
    fun getCropName(isEn: Boolean) = if (isEn) cropEn else cropCn
    fun getStageName(isEn: Boolean) = if (isEn) stageEn else stageCn
    fun getDescription(isEn: Boolean) = if (isEn) descEn else descCn
    fun getWater(isEn: Boolean) = if (isEn) waterEn else waterCn
    fun getFertilizer(isEn: Boolean) = if (isEn) fertilizerEn else fertilizerCn
    fun getPest(isEn: Boolean) = if (isEn) pestEn else pestCn
    fun getDaysInfo(isEn: Boolean) = if (isEn) {
        val sDays = stageDays.replace("天", " days")
        val tDays = totalDays.replace("天", " days")
        "Est. Stage: $sDays | Full Life Cycle: $tDays"
    } else {
        val sDays = stageDays.replace(" days", "天")
        val tDays = totalDays.replace(" days", "天")
        "估算阶段周期: $sDays | 全生育期: $tDays"
    }
}

object AgronomicKnowledge {
    private val adviceMap = mapOf(
        "corn_seedling" to AgronomicAdvice(
            "玉米", "Corn", "出苗期", "Seedling Stage", "10-15天", "110-130天",
            "幼苗矮小（<30cm），单叶出土，无显性茎节，土壤裸露。",
            "Very short seedlings (<30cm), single thin upright leaves, no visible stem nodes, bare soil between plants.",
            "控水促根，幼苗期忌大水漫灌，适度蹲苗。",
            "Control water to promote root depth; avoid flood irrigation during early seedling stage.",
            "轻施提苗肥，以稀薄速效氮肥为主。",
            "Apply light seedling fertilizer with diluted quick-acting nitrogen.",
            "重点防范地老虎、蝼蛄及苗期蚜虫。",
            "Focus on preventing cutworms, mole crickets, and early aphids."
        ),
        "corn_jointing" to AgronomicAdvice(
            "玉米", "Corn", "拔节期", "Jointing Stage", "15-20天", "110-130天",
            "植株拔高齐腰，绿茎粗壮，有明显膨大节间，无穗无花。",
            "Waist-high robust green stems with 3-5 distinct swollen nodes, purely vegetative growth without tassels or ears.",
            "水分临界期前夕，保持土壤湿度 60-70%。",
            "Keep soil moisture at 60-70% ahead of the moisture-critical period.",
            "重施拔节壮秆肥，追施尿素 15-20kg/亩。",
            "Apply jointing fertilizer; top-dress with 15-20kg/acre urea.",
            "重点防治玉米螟、粘虫及大斑病。",
            "Prevent corn borers, armyworms, and large spot disease."
        ),
        "corn_tasseling" to AgronomicAdvice(
            "玉米", "Corn", "抽穗期", "Tasseling Stage", "10-14天", "110-130天",
            "顶端雄穗羽状抽出，中部雌穗吐丝，花粉散落。",
            "Feathery tassels emerging at plant apex, female silk strands protruding from mid-stalk ear shoots.",
            "水分极敏感期！确保花粉活力，切忌干旱缺水。",
            "Moisture-critical stage! Ensure pollen viability; avoid water stress.",
            "补施攻粒肥，防早衰与脱肥。",
            "Apply kernel-filling fertilizer to prevent premature senescence.",
            "人工辅助授粉或防控穗腐病。",
            "Assist pollination if needed and control ear rot disease."
        ),
        "corn_filling" to AgronomicAdvice(
            "玉米", "Corn", "灌浆期", "Grain-Filling Stage", "25-30天", "110-130天",
            "苞叶包裹饱满果穗，籽粒呈黄色乳浊状，茎叶保持绿色。",
            "Bulging ear cobs with plump yellow kernels in dough stage; foliage and stalks remain green.",
            "保持干干湿湿，利于养分向籽粒转运。",
            "Maintain alternating dry-wet soil conditions to promote nutrient translocation to kernels.",
            "叶面喷施 0.2% 磷酸二氢钾溶液。",
            "Foliar spray with 0.2% potassium dihydrogen phosphate.",
            "注意防范玉米锈病及粉芽害虫。",
            "Watch out for corn rust disease and ear aphids."
        ),
        "corn_maturity" to AgronomicAdvice(
            "玉米", "Corn", "成熟期", "Maturity Stage", "10-15天", "110-130天",
            "全株枯黄，苞叶张开干枯，籽粒硬化变亮，黑层形成。",
            "All foliage brown and dried, husks wide open exposing hard glossy dry yellow cobs.",
            "断水落干，便于机械化收割。",
            "Cut off irrigation to dry out the field for mechanical harvesting.",
            "停止施肥，促进完熟。",
            "Cease all fertilization to encourage full maturity.",
            "及时抢晴收割，防范霉变风险。",
            "Harvest promptly on dry clear days to prevent mold."
        ),
        "wheat_seedling" to AgronomicAdvice(
            "小麦", "Wheat", "出苗期", "Seedling Stage", "15-20天", "220-240天",
            "极矮小草状细叶，单芽出土，未分蘖，株高不足15cm。",
            "Very short grass-like shoots (<15cm), single thin blades, no tillers formed yet.",
            "浇好越冬水，确保分蘖幼苗安全过冬。",
            "Irrigate wintering water properly to help young plants survive winter.",
            "基肥充足时少施氮肥，防旺长。",
            "Apply minimal nitrogen if basal fertilizer is sufficient to avoid leggy growth.",
            "防治种蝇及根腐病。",
            "Prevent seed flies and root rot disease."
        ),
        "wheat_tillering" to AgronomicAdvice(
            "小麦", "Wheat", "分蘖期", "Tillering Stage", "60-90天", "220-240天",
            "基部丛生多条侧芽分蘖，呈矮分生绒毯状，无垂直茎秆。",
            "Multiple side shoots branching from base, forming dense low bushy clumps.",
            "适度蹲苗，促进有效分蘖成穗。",
            "Control water moderately to encourage effective tillers.",
            "追施分蘖肥，壮苗增蘖。",
            "Top-dress tiller fertilizer to strengthen shoots.",
            "防控麦黄吸浆虫、红蜘蛛。",
            "Control wheat midge and spider mites."
        ),
        "wheat_jointing" to AgronomicAdvice(
            "小麦", "Wheat", "拔节期", "Jointing Stage", "15-20天", "220-240天",
            "茎秆垂直伸长，出现2-3个膨大节间，顶部无麦穗抽出。",
            "Stems elongating with 2-3 visible swollen joints; no heads emerging from leaf sheaths.",
            "水肥齐攻，促进大穗多粒。",
            "Apply water and fertilizer together to promote large heads and grain count.",
            "重施拔节孕穗肥（氮钾结合）。",
            "Apply heavy jointing-booting fertilizer combining nitrogen and potassium.",
            "防治纹枯病、白粉病。",
            "Control sheath blight and powdery mildew."
        ),
        "wheat_heading" to AgronomicAdvice(
            "小麦", "Wheat", "抽穗期", "Heading Stage", "10-15天", "220-240天",
            "直立绿穗抽出旗叶，麦芒清晰，小花悬挂黄褐色花药。",
            "Compact green spikes emerged from flag leaf sheath with visible dangling yellow anthers.",
            "开花期忌大水漫灌，防倒伏与病害。",
            "Avoid flood irrigation during flowering to prevent lodging and diseases.",
            "喷施微量元素与磷酸二氢钾。",
            "Spray micro-nutrients and KH2PO4.",
            "重点防控赤霉病（见花打药）。",
            "Target head blight (Gibberella) at early bloom."
        ),
        "wheat_maturity" to AgronomicAdvice(
            "小麦", "Wheat", "成熟期", "Maturity Stage", "10-15天", "220-240天",
            "全株金黄琥珀色，麦穗重重下垂弯曲，茎秆干燥直立。",
            "Uniformly golden-amber plants, heavy grain heads drooping downward on dried stalks.",
            "收获前 7-10 天停水。",
            "Stop irrigation 7-10 days before harvest.",
            "停止施肥。",
            "Cease all fertilization.",
            "防范干热风与雨后烂麦穗。",
            "Protect against dry hot winds and rain-induced head sprouting."
        ),
        "cotton_seedling" to AgronomicAdvice(
            "棉花", "Cotton", "苗期", "Seedling Stage", "25-30天", "180-200天",
            "贴地生长两片圆形子叶，茎细弱，未分枝，株高低于15cm。",
            "Tiny plants close to ground with two round cotyledon leaves, thin green stems (<15cm).",
            "中耕保墒，提高地温促早发。",
            "Cultivate soil to conserve moisture and raise temperature.",
            "苗期少量齐苗肥。",
            "Apply a small amount of seedling fertilizer.",
            "防治立枯病、棉蚜。",
            "Control damping-off and cotton aphids."
        ),
        "cotton_squaring" to AgronomicAdvice(
            "棉花", "Cotton", "蕾期", "Squaring Stage", "25-30天", "180-200天",
            "分枝显现，叶腋长出三角形小方蕾，未见开放花朵。",
            "Branching structure with small square-shaped green triangular buds; no open blooms.",
            "稳水控氮，防止盲目旺长。",
            "Regulate water and control nitrogen to prevent excessive vegetative growth.",
            "适量施配方肥，控氮增磷钾。",
            "Apply formula fertilizer increasing phosphorus and potassium.",
            "防治棉铃虫、盲蝽蟌。",
            "Control bollworms and plant bugs."
        ),
        "cotton_flowering" to AgronomicAdvice(
            "棉花", "Cotton", "开花期", "Flowering Stage", "25-30天", "180-200天",
            "大朵白/粉色花瓣开放，黄色雄蕊柱明显，无绿色圆球棉铃。",
            "Open flowers with creamy white or pink petals and yellow stamens; no round green bolls.",
            "盛花期水肥关键期，不能缺水。",
            "Critical flowering period; ensure adequate water and nutrients.",
            "重施花铃肥，防止落花落铃。",
            "Apply flower-boll fertilizer to reduce shed buds and flowers.",
            "防治三代棉铃虫及红蜘蛛。",
            "Control 3rd-generation bollworms and spider mites."
        ),
        "cotton_boll_setting" to AgronomicAdvice(
            "棉花", "Cotton", "结铃期", "Boll Setting Stage", "30-40天", "180-200天",
            "枝头挂满深绿硬质圆球形棉铃果实，无开花，叶片仍绿。",
            "Firm green spherical fruit capsules hanging from branches; leaves remain green.",
            "见干见湿，防早衰防脱落。",
            "Maintain dry-wet cycle to prevent premature aging and boll drop.",
            "补施盖顶肥或叶面肥。",
            "Top-dress or foliar spray top fertilizer.",
            "防烂铃与红叶茎枯病。",
            "Prevent boll rot and red leaf disease."
        ),
        "cotton_boll_opening" to AgronomicAdvice(
            "棉花", "Cotton", "吐絮期", "Boll Opening Stage", "30-40天", "180-200天",
            "棉铃爆裂开缝，洁白絮状棉花外露，苞叶干枯卷曲。",
            "Capsules splitting open exposing fluffy white cotton fiber; dried brown bracts.",
            "推株并行，通风透光促吐絮。",
            "Open canopy for light and air ventilation to encourage boll opening.",
            "停止施肥。",
            "Cease all fertilization.",
            "分批采收，防泥污絮。",
            "Pick cotton in batches to avoid fiber contamination."
        )
    )

    fun getAdvice(className: String): AgronomicAdvice {
        return adviceMap[className] ?: AgronomicAdvice(
            "未知作物", "Unknown Crop", "未知阶段", "Unknown Stage", "--", "--",
            "暂无当前阶段农艺特征描述。", "No description available.",
            "保持适宜土壤湿度。", "Keep adequate soil moisture.",
            "根据长势合理补充养分。", "Apply nutrients based on crop growth.",
            "定期巡田，早发现早防控。", "Inspect field regularly for pest control."
        )
    }
}
