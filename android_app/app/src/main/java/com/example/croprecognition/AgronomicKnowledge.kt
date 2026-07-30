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
    val pestEn: String,
    val opsCn: String,
    val opsEn: String
) {
    fun getCropName(isEn: Boolean) = if (isEn) cropEn else cropCn
    fun getStageName(isEn: Boolean) = if (isEn) stageEn else stageCn
    fun getDescription(isEn: Boolean) = if (isEn) descEn else descCn
    fun getWater(isEn: Boolean) = if (isEn) waterEn else waterCn
    fun getFertilizer(isEn: Boolean) = if (isEn) fertilizerEn else fertilizerCn
    fun getPest(isEn: Boolean) = if (isEn) pestEn else pestCn
    fun getOps(isEn: Boolean) = if (isEn) opsEn else opsCn
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
            "控水促根蹲苗：保持土壤相对持水量 55%-60%。苗期抗旱怕涝，忌大水漫灌，适度蹲苗促进根系下扎 20-30 cm。",
            "Water control for deep roots: Maintain soil moisture at 55%-60%. Avoid flood irrigation; moderate drought stress encourages roots to penetrate 20-30 cm deep.",
            "轻施提苗肥：3-4 叶期每亩追施速效氮肥（尿素 5-8 kg）+ 腐殖酸水溶肥 2-3 kg，促弱苗转壮苗。",
            "Light seedling fertilizer: Top-dress 5-8 kg/acre urea + 2-3 kg/acre humic acid water-soluble fertilizer at 3-4 leaf stage.",
            "地下害虫与苗蚜防控：用 3% 辛硫磷颗粒剂拌细土撒施防地老虎、蛴螬；用 20% 啶虫脒 1500 倍液喷雾防治苗期蚜虫。",
            "Pest control: Apply 3% phoxim granules with dry soil for cutworms and grubs; spray 20% acetamiprid 1500x for early aphids.",
            "查苗补苗与定苗：及时进行间苗定苗（单株留苗，拔除弱残苗），结合中耕松土 3-5 cm 提高地温。",
            "Thinning & Soil hoeing: Thin to single strong seedling per hill; hoe 3-5 cm deep to loosen soil and raise temperature."
        ),
        "corn_jointing" to AgronomicAdvice(
            "玉米", "Corn", "拔节期", "Jointing Stage", "15-20天", "110-130天",
            "植株拔高齐腰，绿茎粗壮，有明显膨大节间，无穗无花。",
            "Waist-high robust green stems with 3-5 distinct swollen nodes, purely vegetative growth without tassels or ears.",
            "临界期水分保障：保持土壤相对持水量 65%-70%。临界期前夕若连续 7 天无雨且叶片中午卷曲，及时灌水 30-40 m³/亩。",
            "Critical period watering: Keep soil moisture at 65%-70%. Irrigate 30-40 m³/acre if no rain for 7 days and leaves curl at midday.",
            "重施拔节壮秆肥：大喇叭口期前 10 天追施尿素 15-20 kg/亩 + 氯化钾 5-8 kg/亩，开沟条施深施 8-10 cm 并覆土。",
            "Heavy jointing fertilizer: Top-dress 15-20 kg/acre urea + 5-8 kg/acre KCl 10 days before big trumpet stage, applying 8-10 cm deep.",
            "玉米螟与大斑病预警：用 0.5% 氟氯氰菊酯颗粒剂心叶丢芯防玉米螟；用 50% 多菌灵可湿性粉剂 500 倍液喷雾防大斑病。",
            "Borer & Blight control: Drop 0.5% cyfluthrin granules into leaf whorls for corn borer; spray 50% carbendazim 500x for leaf spot.",
            "培土防倒伏：结合追肥进行中耕培土 5-8 cm 促进次生根发生，增强抗倒伏能力，拔除弱苗分蘖。",
            "Ridge hilling: Hill soil 5-8 cm high around stem bases during fertilization to promote brace roots and lodging resistance."
        ),
        "corn_tasseling" to AgronomicAdvice(
            "玉米", "Corn", "Tasseling Stage", "10-14天", "110-130天",
            "顶端雄穗羽状抽出，中部雌穗吐丝，花粉散落。",
            "Feathery tassels emerging at plant apex, female silk strands protruding from mid-stalk ear shoots.",
            "水分敏感高峰保障：保持土壤持水量 75%-80%，遇旱必须及时浇水 40-50 m³/亩，确保花粉活力与吐丝顺畅。",
            "Peak water sensitivity: Maintain 75%-80% soil moisture; irrigate 40-50 m³/acre during drought to preserve pollen vitality.",
            "补施攻粒肥：叶面喷施 0.2% 磷酸二氢钾 + 0.1% 硼砂溶液，提高结实率与千粒重，防早衰脱肥。",
            "Ear fertility spray: Foliar spray 0.2% KH2PO4 + 0.1% borax solution to increase kernel setting rate and 1000-grain weight.",
            "穗腐病与叶甲防控：喷施 25% 丙环唑乳油 2000 倍液防玉米锈病与穗腐病；用 2.5% 高效氯氟氰菊酯防双斑萤叶甲。",
            "Ear rot & Flea beetle control: Spray 25% propiconazole 2000x for rust and ear rot; use 2.5% lambda-cyhalothrin for leaf beetles.",
            "辅助授粉与去雄：无风晴天上午 9-11 时人工拉绳辅助授粉；隔行去雄 1/2 降低养分消耗。",
            "Assisted pollination & Detasseling: Pull ropes manually at 9-11 AM on calm dry days for pollination; detassel alternate rows 50%."
        ),
        "corn_filling" to AgronomicAdvice(
            "玉米", "Corn", "Grain-Filling Stage", "25-30天", "110-130天",
            "苞叶包裹饱满果穗，籽粒呈黄色乳浊状，茎叶保持绿色。",
            "Bulging ear cobs with plump yellow kernels in dough stage; foliage and stalks remain green.",
            "干湿交替灌溉：保持土壤湿度 70% 左右，采取“干干湿湿”交替灌溉，促进光合产物向籽粒转运，切忌过早停水。",
            "Alternating wet-dry irrigation: Keep soil moisture ~70% with dry-wet cycles to drive photosynthate to kernels; avoid early water cutoff.",
            "保叶防早衰：禁施大氮肥防贪青晚熟；每亩叶面喷施 0.3% 磷酸二氢钾 + 1% 尿素混合液 50 kg 延长绿叶功能期。",
            "Foliar green retention: Avoid heavy nitrogen; foliar spray 50 kg/acre 0.3% KH2PO4 + 1% urea to extend leaf functional lifespan.",
            "顶腐病与红蜘蛛防控：喷施 1.8% 阿维菌素乳油 3000 倍液防红蜘蛛；注意防范茎基腐病及顶腐病。",
            "Mite & Stem rot control: Spray 1.8% abamectin 3000x for spider mites; watch for top rot and stalk rot.",
            "通风透光管理：站秆打剥下部 2-3 片老黄叶改善田间通风透光，清理雨后田间积水防烂根。",
            "Canopy light management: Strip off bottom 2-3 yellow senescent leaves to improve light penetration and drainage."
        ),
        "corn_maturity" to AgronomicAdvice(
            "玉米", "Corn", "Maturity Stage", "10-15天", "110-130天",
            "全株枯黄，苞叶张开干枯，籽粒硬化变亮，黑层形成。",
            "All foliage brown and dried, husks wide open exposing hard glossy dry yellow cobs.",
            "适时断水促干：收割前 7-10 天停止灌水，促使籽粒自然失水干燥与黑层形成，便于机械化直接采收。",
            "Water cutoff: Stop irrigation 7-10 days before harvest to promote natural grain drying and black layer formation.",
            "停止施肥：完全停止一切土壤与叶面施肥，促进养分向籽粒彻底转移完熟。",
            "Cease fertilization: Completely stop soil and foliar fertilization to facilitate full maturity.",
            "防范霉菌污染：清理田间枯枝落叶，机械收获后及时晾晒烘干，防止黄曲霉毒素侵染。",
            "Mold prevention: Clear field debris; dry harvested grain immediately to prevent aflatoxin contamination.",
            "抢晴适时收获：苞叶变黄枯干、籽粒基部出现黑色层、乳线消失时，抢晴进行机械联合收获。",
            "Timely combine harvest: Harvest when husks turn brown, milk line disappears, and black layer forms."
        ),
        "wheat_seedling" to AgronomicAdvice(
            "小麦", "Wheat", "Seedling Stage", "15-20天", "220-240天",
            "极矮小草状细叶，单芽出土，未分蘖，株高不足15cm。",
            "Very short grass-like shoots (<15cm), single thin blades, no tillers formed yet.",
            "浇好越冬水：土壤相对持水量保持 60%-70%。播后缺水及时微灌，日平均气温降至 3-5℃ 时浇透越冬水。",
            "Wintering irrigation: Maintain 60%-70% soil moisture; apply thorough winter irrigation when daily temperature drops to 3-5°C.",
            "控氮促弱转壮：基肥充足时少施速效氮防旺长；若苗弱叶黄，结合灌水亩追尿素 5-7 kg + 磷酸二铵 3-5 kg。",
            "Nitrogen control: Limit fast-acting nitrogen if basal fertilizer is high; if weak, top-dress 5-7 kg/acre urea + 3-5 kg DAP.",
            "药剂拌种与苗病防治：用 5% 咪鲜胺喷雾防治纹枯病；药剂拌种防根腐病、全蚀病，防范种蝇及地下害虫。",
            "Seed dressing & Blight control: Spray 5% prochloraz for sheath blight; use dressed seeds against root rot and seed flies.",
            "查苗补种与镇压：缺苗断垄处及时补种；日暖夜冻时及时适度镇压保墒，促进根系发育。",
            "Compaction & Re-seeding: Fill bare spots promptly; roll soil lightly during freeze-thaw days to retain moisture and press roots."
        ),
        "wheat_tillering" to AgronomicAdvice(
            "小麦", "Wheat", "Tillering Stage", "60-90天", "220-240天",
            "基部丛生多条侧芽分蘖，呈矮分生绒毯状，无垂直茎秆。",
            "Multiple side shoots branching from base, forming dense low bushy clumps.",
            "适度蹲苗控蘖：土壤湿度控制在 60% 左右。适度蹲苗控制无效分蘖，促壮苗下扎，冬前分蘖达到 5-6 个为宜。",
            "Drought hardening for tillers: Keep soil moisture ~60%; suppress non-effective tillers to ensure 5-6 strong pre-winter tillers per plant.",
            "追施分蘖壮苗肥：每亩追施尿素 8-10 kg，促壮苗增蘖，提高冬前大蘖成穗率。",
            "Tiller fertilizer: Top-dress 8-10 kg/acre urea to strengthen shoots and elevate large tiller spike-bearing percentage.",
            "红蜘蛛与吸浆虫防控：用 15% 哒螨灵乳油 1500 倍液喷雾防控麦上红蜘蛛及麦黄吸浆虫。",
            "Spider mite & Midge control: Spray 15% pyridaben 1500x for wheat spider mites and orange wheat blossom midge.",
            "划锄松土防旺长：深划锄 3-4 cm 破除土壤板结、提高地温；旺长田块喷施多效唑控制株高防冬前抽穗。",
            "Inter-row hoeing: Hoe 3-4 cm deep to break crusting; spray paclobutrazol on over-vigorous fields to prevent premature heading."
        ),
        "wheat_jointing" to AgronomicAdvice(
            "小麦", "Wheat", "Jointing Stage", "15-20天", "220-240天",
            "茎秆垂直伸长，出现2-3个膨大节间，顶部无麦穗抽出。",
            "Stems elongating with 2-3 visible swollen joints; no heads emerging from leaf sheaths.",
            "水肥齐攻促大穗：结合追肥及时灌水（水肥齐攻），保持土壤持水量 70%-75%，促小花发育减少退化。",
            "Jointing water & fertilizer: Apply irrigation and fertilizer together (70%-75% moisture) to reduce floret abortion.",
            "重施拔节孕穗肥：亩追尿素 10-12 kg + 高钾复合肥 5 kg，开沟深施，增强秆强与穗粒数。",
            "Heavy booting fertilizer: Top-dress 10-12 kg/acre urea + 5 kg/acre high-K compound fertilizer in furrows.",
            "纹枯病与白粉病防控：用 20% 井冈霉素 1000 倍液防纹枯病；用 10% 吡虫啉 2000 倍液防治麦蚜。",
            "Sheath blight & Aphid control: Spray 20% jinggangmycin 1000x for sheath blight; apply 10% imidacloprid 2000x for aphids.",
            "化控防倒伏：结合划锄清除杂草；对旺长倒伏隐患田块喷施矮壮素（CCC）控制基部节间伸长。",
            "Chemical height control: Spray chlormequat chloride (CCC) on overly tall fields to shorten lower internodes and prevent lodging."
        ),
        "wheat_heading" to AgronomicAdvice(
            "小麦", "Wheat", "Heading Stage", "10-15天", "220-240天",
            "直立绿穗抽出旗叶，麦芒清晰，小花悬挂黄褐色花药。",
            "Compact green spikes emerged from flag leaf sheath with visible dangling yellow anthers.",
            "微灌防倒伏：开花期忌大水漫灌（易引起倒伏与赤霉病爆发），维持持水量 70% 左右，微灌为宜。",
            "Micro-irrigation: Avoid flood irrigation during bloom (prevents lodging and Head Blight epidemics); maintain 70% moisture.",
            "一喷三防全覆盖：亩喷 0.3% 磷酸二氢钾 + 0.1% 芸苔素内酯 + 杀虫杀菌剂，防早衰、防病虫、防干热风。",
            "One spray three protections: Tank-mix 0.3% KH2PO4 + 0.1% brassinolide + fungicides/insecticides.",
            "赤霉病见花打药：重点防控赤霉病（见花打药！用 43% 戊唑醇 3000 倍液），兼治锈病与穗蚜。",
            "Head blight bloom timing: Apply 43% tebuconazole 3000x AT EARLY BLOOM (10-20% flowering) for Gibberella Head Blight.",
            "避风避雨管理：关注天气预报，强降雨或大风到来前切忌灌水，谨防后期大面积倒伏。",
            "Weather precautions: Do not irrigate immediately before strong winds or heavy rain to prevent severe field lodging."
        ),
        "wheat_maturity" to AgronomicAdvice(
            "小麦", "Wheat", "Maturity Stage", "10-15天", "220-240天",
            "全株金黄琥珀色，麦穗重重下垂弯曲，茎秆干燥直立。",
            "Uniformly golden-amber plants, heavy grain heads drooping downward on dried stalks.",
            "收获前断水：收获前 7-10 天全面断水，促进籽粒脱水、硬化与琥珀色形成。",
            "Water cutoff: Stop irrigation 7-10 days prior to harvest for rapid grain moisture drop and amber hardness.",
            "停止一切施肥：完全停止土壤与叶面施肥，避免贪青晚熟。",
            "Cease all inputs: No soil or foliar fertilization to prevent delayed maturity.",
            "防干热风与穗发芽：喷施 0.2% 磷酸二氢钾抗干热风；雨后及时排水防穗发芽烂麦。",
            "Hot dry wind protection: Spray 0.2% KH2PO4 against dry hot winds; drain fields quickly after rain to prevent pre-harvest sprouting.",
            "抢晴完熟期收获：蜡熟末期至完熟期（麦穗弯曲下垂、籽粒硬化、千粒重最高）抢晴机械联合收割。",
            "Combine harvest: Harvest during late dough to full maturity stage on dry sunny days for maximum 1000-grain weight."
        ),
        "cotton_seedling" to AgronomicAdvice(
            "棉花", "Cotton", "Seedling Stage", "25-30天", "180-200天",
            "贴地生长两片圆形子叶，茎细弱，未分枝，株高低于15cm。",
            "Tiny plants close to ground with two round cotyledon leaves, thin green stems (<15cm).",
            "中耕保墒促根：主攻中耕保墒提高地温，苗期一般不灌水，避免降低地温导致立枯病与烂根。",
            "Inter-tillage conservation: Hoe soil to retain moisture and increase ground temperature; refrain from early irrigation.",
            "轻施齐苗肥：苗弱时亩追尿素 3-4 kg 或喷施氨基酸叶面肥，切忌偏施氮肥防旺长。",
            "Light seedling dressing: Apply 3-4 kg/acre urea or amino acid foliar fertilizer on weak seedlings; avoid excessive nitrogen.",
            "立枯病与棉蚜防控：多菌灵拌种防立枯病、炭疽病；用 10% 吡虫啉 1500 倍液防治苗期棉蚜。",
            "Damping-off & Aphids: Dress seeds with carbendazim; spray 10% imidacloprid 1500x for seedling aphids.",
            "间苗定苗深中耕：及时打膜孔引苗封土；间苗定苗（一穴一株），深中耕 6-8 cm 破除板结。",
            "Thinning & Deep hoeing: Uncover mulch holes for seedlings; thin to 1 plant per hill and deep hoe 6-8 cm."
        ),
        "cotton_squaring" to AgronomicAdvice(
            "棉花", "Cotton", "Squaring Stage", "25-30天", "180-200天",
            "分枝显现，叶腋长出三角形小方蕾，未见开放花朵。",
            "Branching structure with small square-shaped green triangular buds; no open blooms.",
            "稳水控氮防旺长：土壤持水量保持 60%-65%，遇旱小水轻灌，切忌大水漫灌促使主茎狂长。",
            "Regulate water & N: Maintain 60%-65% moisture; light irrigation during drought to prevent leggy main stem growth.",
            "稳施蕾肥增磷钾：过磷酸钙 15 kg + 硫酸钾 8 kg/亩开沟施入，控施氮肥防止落蕾落花。",
            "Square stage P-K boost: Furrow-apply 15 kg/acre superphosphate + 8 kg/acre potassium sulfate; control nitrogen.",
            "棉铃虫与盲蝽蟌防控：用 20% 氯虫苯甲酰胺 3000 倍液防治二代棉铃虫、盲蝽蟌及棉红蜘蛛。",
            "Bollworm & Bug control: Apply 20% chlorantraniliprole 3000x for 2nd-gen bollworms, plant bugs, and spider mites.",
            "整枝抹芽化控：及时打边心（抹去叶枝）；喷施缩节胺（DPC 1.5-2.0 g/亩）塑造紧凑株型。",
            "Pruning & Pix control: Prune vegetative branches; spray Pix (mepiquat chloride 1.5-2.0 g/acre) for compact structure."
        ),
        "cotton_flowering" to AgronomicAdvice(
            "棉花", "Cotton", "Flowering Stage", "25-30天", "180-200天",
            "大朵白/粉色花瓣开放，黄色雄蕊柱明显，无绿色圆球棉铃。",
            "Open flowers with creamy white or pink petals and yellow stamens; no round green bolls.",
            "盛花期足水保障：盛花期为水肥敏感高峰！持水量保持 75%-80%，5-7 天灌一次水，严禁缺水防落铃。",
            "Peak bloom watering: Keep soil moisture 75%-80%; irrigate every 5-7 days; water stress causes massive boll drop.",
            "重施花铃肥：亩追尿素 15-20 kg + 钾肥 8-10 kg；叶面补充 0.2% 硼砂防止“花而不实”。",
            "Heavy flower-boll fertilizer: Top-dress 15-20 kg/acre urea + 8-10 kg K fertilizer; foliar spray 0.2% borax.",
            "伏蚜与枯黄萎病防控：用氨基 oligosaccharin 喷雾防枯黄萎病；用 25% 噻虫嗪防治伏蚜与三代棉铃虫。",
            "Wilt & Aphid protection: Spray amino oligosaccharin for Verticillium wilt; use 25% thiamethoxam for summer aphids.",
            "适时打顶心：当主茎果枝达到 12-14 个时（7 月中下旬）及时打顶心，集中养分供应果铃。",
            "Main stem topping: Pinch off main stem top when 12-14 fruiting branches form (mid-to-late July) to direct nutrients to bolls."
        ),
        "cotton_boll_setting" to AgronomicAdvice(
            "棉花", "Cotton", "Boll Setting Stage", "30-40天", "180-200天",
            "枝头挂满深绿硬质圆球形棉铃果实，无开花，叶片仍绿。",
            "Firm green spherical fruit capsules hanging from branches; leaves remain green.",
            "见干见湿保铃：持水量保持 70% 左右，小水勤灌，防止高温干旱造成早衰或大水漫灌导致烂铃。",
            "Alternating moist-dry: Maintain ~70% moisture with light frequent watering; prevent premature aging or boll rot.",
            "补施盖顶肥：亩施尿素 4-5 kg 或每周叶面喷施 0.3% 磷酸二氢钾，增加单铃重与吐絮品质。",
            "Top capping fertilizer: Apply 4-5 kg/acre urea or weekly foliar 0.3% KH2PO4 to boost boll weight and lint quality.",
            "疫病与四代棉铃虫防控：喷施 70% 代森锰锌 800 倍液防治棉铃疫病、红腐病及四代棉铃虫。",
            "Boll rot & Pest defense: Spray 70% mancozeb 800x for boll rot/phytophthora and 4th-gen bollworms.",
            "老叶空枝清理：剪去下部老空枝、黄叶，改善田间通风透光条件，降低湿气减烂铃。",
            "Lower leaf pruning: Prune lower fruitless branches and yellow senescent leaves to enhance ventilation."
        ),
        "cotton_boll_opening" to AgronomicAdvice(
            "棉花", "Cotton", "Boll Opening Stage", "30-40天", "180-200天",
            "棉铃爆裂开缝，洁白絮状棉花外露，苞叶干枯卷曲。",
            "Capsules splitting open exposing fluffy white cotton fiber; dried brown bracts.",
            "吐絮断水防泥污：吐絮后逐步停水，清沟排水防渍，促进棉铃自然开裂吐絮，防止泥水污染棉絮。",
            "Defoliation water cutoff: Stop watering gradually; drain field channels to keep cotton fiber clean.",
            "停止施肥：完全停止土壤施肥，促棉铃开裂自然成熟。",
            "Stop fertilization: Completely stop soil fertilization to allow natural boll maturation.",
            "后季棉蚜与防烂铃：控制后季棉蚜与烂铃，保持棉絮洁白无杂质。",
            "Late aphid control: Keep late aphids off open bolls to prevent sticky fiber contamination.",
            "推株并行与脱叶催熟：推株并行加强透光；机采棉在采收前 15-20 天喷施脱叶催熟剂（噻苯隆 + 乙烯利）。",
            "Defoliant spray: Spray thidiazuron + ethephon 15-20 days before machine picking for synchronous leaf drop and boll opening."
        )
    )

    fun getAdvice(className: String): AgronomicAdvice {
        return adviceMap[className] ?: AgronomicAdvice(
            "未知作物", "Unknown Crop", "未知阶段", "Unknown Stage", "--", "--",
            "暂无当前阶段农艺特征描述。", "No description available.",
            "保持土壤持水量 65%-70%，因地制宜按需灌溉。", "Maintain 65%-70% soil moisture.",
            "根据作物长势氮磷钾配比施肥，补充微量元素。", "Apply N-P-K fertilizer based on growth.",
            "定期巡田防范病虫害，早发现早对症防治。", "Inspect field regularly for pest control.",
            "做好田间松土与除草管理，保障光合效率。", "Maintain soil cultivation and weed control."
        )
    }
}
