// pages/knowledge/knowledge.js
const KNOWLEDGE_DATA = {
  corn: [
    {
      id: "corn_seedling",
      nameZh: "出苗期",
      nameEn: "Seedling",
      desc: "从幼苗出土至 3 叶期，植株较小，主要是根系发育与建苗阶段。",
      fertilizer: "适量施用种肥，保证齐苗壮苗；注意补施磷锌肥。",
      irrigation: "保持土壤湿润但忌积水，促进根系下扎。",
      pest: "重点防治地老虎、蛴螬等地下害虫及黏虫。"
    },
    {
      id: "corn_jointing",
      nameZh: "拔节期",
      nameEn: "Jointing",
      desc: "茎节开始伸长，植株生长加快，进入营养生长与生殖生长并进期。",
      fertilizer: "重施拔节肥（攻杆肥），以氮肥为主，促茎粗叶茂。",
      irrigation: "水分需求适中，遇旱及时浇水。",
      pest: "防治玉米螟、红蜘蛛及大斑病。"
    },
    {
      id: "corn_tasseling",
      nameZh: "抽穗期",
      nameEn: "Tasseling",
      desc: "雄穗抽出，开花散粉，雌穗吐丝受粉，是决定穗粒数的关键期。",
      fertilizer: "巧施攻穗肥，注意补施钾肥提高抗逆性。",
      irrigation: "临界水时期，土壤水分必须保持在持水量的 70-80%。",
      pest: "重点防治玉米螟、玉米蚜虫及茎腐病。"
    },
    {
      id: "corn_filling",
      nameZh: "灌浆期",
      nameEn: "Filling",
      desc: "籽粒形成与干物质积累期，决定百粒重和最终产量。",
      fertilizer: "叶面喷施磷酸二氢钾及微肥，防止脱肥早衰。",
      irrigation: "保证水份供给，遇干旱及时小水勤灌。",
      pest: "防治玉米锈病、穗腐病及蚜虫。"
    },
    {
      id: "corn_maturity",
      nameZh: "成熟期",
      nameEn: "Maturity",
      desc: "籽粒乳线消失，基部黑层出现，籽粒变硬显现品种特征色泽。",
      fertilizer: "停止施肥，保持通风透光。",
      irrigation: "适时控水，促进成熟与脱水干燥。",
      pest: "注意防鼠防倒伏，适时机械收割。"
    }
  ],
  wheat: [
    {
      id: "wheat_seedling",
      nameZh: "出苗期",
      nameEn: "Seedling",
      desc: "麦苗出土显绿至分蘖前，根系下扎与叶片分化阶段。",
      fertilizer: "基肥充足时少施；弱苗可适量追施促苗肥。",
      irrigation: "浇好分蘖水，保持土壤湿润。",
      pest: "防治地下害虫、地下蛴螬及网蝇。"
    },
    {
      id: "wheat_tillering",
      nameZh: "分蘖期",
      nameEn: "Tillering",
      desc: "分蘖节分生分蘖，冬小麦经历越冬潜伏与分蘖累积阶段。",
      fertilizer: "追施壮苗肥与越冬肥，促多成穗大穗。",
      irrigation: "冬前浇足越冬水，保苗安全越冬。",
      pest: "防治红蜘蛛、纹枯病。"
    },
    {
      id: "wheat_jointing",
      nameZh: "拔节期",
      nameEn: "Jointing",
      desc: "基部第一节间伸长，小花分化加速，生长最为旺盛。",
      fertilizer: "重施拔节孕穗肥，氮钾配合，巩固分蘖。",
      irrigation: "浇好拔节水，满足旺盛生长水分需求。",
      pest: "防治白粉病、锈病及吸浆虫。"
    },
    {
      id: "wheat_heading",
      nameZh: "抽穗期",
      nameEn: "Heading",
      desc: "麦穗顶端抽出叶鞘并开花受粉，决定每穗粒数。",
      fertilizer: "叶面喷肥，结合‘一喷三防’补充微量元素。",
      irrigation: "抽穗开花水不可少，但忌大水漫灌。",
      pest: "全面防治赤霉病、穗蚜及锈病。"
    },
    {
      id: "wheat_maturity",
      nameZh: "成熟期",
      nameEn: "Maturity",
      desc: "籽粒灌浆完成，金黄硬化，达到收割标准。",
      fertilizer: "停止施肥。",
      irrigation: "收获前 7-10 天停止灌溉。",
      pest: "防倒伏、防干热风，适时抢收。"
    }
  ],
  cotton: [
    {
      id: "cotton_seedling",
      nameZh: "苗期",
      nameEn: "Seedling",
      desc: "出苗至显蕾，以根系扩展和叶片生长为主。",
      fertilizer: "控制氮肥，防止旺长；适量施锌肥。",
      irrigation: "中耕保墒，控水蹲苗。",
      pest: "防治棉蚜、立枯病及猝倒病。"
    },
    {
      id: "cotton_squaring",
      nameZh: "蕾期",
      nameEn: "Squaring",
      desc: "出现果枝与三角形棉蕾，生长加快。",
      fertilizer: "稳施蕾肥，增施钾肥，适度调控。",
      irrigation: "适度灌水，促稳长搭好丰产架子。",
      pest: "防治棉铃虫、盲蝽蟓。"
    },
    {
      id: "cotton_flowering",
      nameZh: "开花期",
      nameEn: "Flowering",
      desc: "开花授粉并开始受精成铃，营养与生殖生长高峰。",
      fertilizer: "重施花铃肥，补充硼与有效钾。",
      irrigation: "盛花期需水高峰，保持土壤透气湿润。",
      pest: "防治红蜘蛛、伏蚜及枯黄萎病。"
    },
    {
      id: "cotton_boll_setting",
      nameZh: "结铃期",
      nameEn: "Boll Setting",
      desc: "棉铃快速膨大，干物质充实积累。",
      fertilizer: "补施盖顶肥，防止早衰；叶面喷磷钾肥。",
      irrigation: "后期小水勤灌，防止脱水落铃。",
      pest: "防治棉铃虫及红铃虫。"
    },
    {
      id: "cotton_boll_opening",
      nameZh: "吐絮期",
      nameEn: "Boll Opening",
      desc: "棉铃成熟开裂，吐出洁白棉纤。",
      fertilizer: "停止土壤施肥。",
      irrigation: "及时排水防渍，保持通风透光。",
      pest: "防烂铃病，分批采收保障品质。"
    }
  ]
};

Page({
  data: {
    currentCrop: 'corn',
    activeStages: []
  },

  onLoad: function () {
    this.setData({
      activeStages: KNOWLEDGE_DATA['corn']
    });
  },

  switchCrop: function (e) {
    const crop = e.currentTarget.dataset.crop;
    this.setData({
      currentCrop: crop,
      activeStages: KNOWLEDGE_DATA[crop] || []
    });
  }
});
