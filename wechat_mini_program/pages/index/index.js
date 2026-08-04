// pages/index/index.js
const app = getApp();

// 作物及其 5 个生长阶段列表选项
const CROP_STAGE_MAP = {
  corn: {
    label: "🌽 玉米 (Corn)",
    stages: [
      { id: "seedling", label: "🌱 出苗期 (Seedling)" },
      { id: "jointing", label: "🌿 拔节期 (Jointing)" },
      { id: "tasseling", label: "🌾 抽穗期 (Tasseling)" },
      { id: "filling", label: "🌽 灌浆期 (Filling)" },
      { id: "maturity", label: "🍂 成熟期 (Maturity)" }
    ]
  },
  wheat: {
    label: "🌾 小麦 (Wheat)",
    stages: [
      { id: "seedling", label: "🌱 出苗期 (Seedling)" },
      { id: "tillering", label: "🌿 分蘖期 (Tillering)" },
      { id: "jointing", label: "🍀 拔节期 (Jointing)" },
      { id: "heading", label: "🌾 抽穗期 (Heading)" },
      { id: "maturity", label: "🍂 成熟期 (Maturity)" }
    ]
  },
  cotton: {
    label: "☁️ 棉花 (Cotton)",
    stages: [
      { id: "seedling", label: "🌱 苗期 (Seedling)" },
      { id: "squaring", label: "🌿 蕾期 (Squaring)" },
      { id: "flowering", label: "🌸 开花期 (Flowering)" },
      { id: "boll_setting", label: "🍈 结铃期 (Boll Setting)" },
      { id: "boll_opening", label: "☁️ 吐絮期 (Boll Opening)" }
    ]
  }
};

Page({
  data: {
    serverUrl: 'http://127.0.0.1:8000',
    imagePath: '',
    isAnalyzing: false,
    resultData: null,

    cropClassMap: {
      corn: 'tag-corn',
      wheat: 'tag-wheat',
      cotton: 'tag-cotton'
    },

    // 弹窗状态
    showServerModal: false,
    inputServerUrl: '',

    showFeedbackModal: false,
    cropOptions: [
      { id: 'corn', label: '🌽 玉米 (Corn)' },
      { id: 'wheat', label: '🌾 小麦 (Wheat)' },
      { id: 'cotton', label: '☁️ 棉花 (Cotton)' }
    ],
    selectedCropIndex: 0,
    currentStageOptions: [],
    selectedStageIndex: 0,
    feedbackNote: ''
  },

  onLoad: function () {
    this.setData({
      serverUrl: app.globalData.serverUrl || 'http://127.0.0.1:8000'
    });
    this.updateStageOptions('corn');
  },

  onShow: function () {
    this.setData({
      serverUrl: app.globalData.serverUrl
    });
  },

  // 1. 从相册选择照片或拍照
  chooseImage: function () {
    const that = this;
    if (wx.chooseMedia) {
      wx.chooseMedia({
        count: 1,
        mediaType: ['image'],
        sourceType: ['album', 'camera'],
        success(res) {
          if (res.tempFiles && res.tempFiles.length > 0) {
            that.setData({
              imagePath: res.tempFiles[0].tempFilePath,
              resultData: null
            });
          }
        }
      });
    } else {
      wx.chooseImage({
        count: 1,
        sourceType: ['album', 'camera'],
        success(res) {
          that.setData({
            imagePath: res.tempFilePaths[0],
            resultData: null
          });
        }
      });
    }
  },

  clearImage: function () {
    this.setData({
      imagePath: '',
      resultData: null
    });
  },

  // 2. 核心 AI 智能诊断
  runDiagnosis: function () {
    if (!this.data.imagePath) {
      wx.showToast({
        title: '请先拍摄或选择一张图片',
        icon: 'none'
      });
      return;
    }

    const that = this;
    const apiUrl = app.getApiUrl('/api/recognize');

    this.setData({ isAnalyzing: true });
    wx.showLoading({ title: 'AI 识别分析中...' });

    wx.uploadFile({
      url: apiUrl,
      filePath: that.data.imagePath,
      name: 'file',
      success(res) {
        wx.hideLoading();
        that.setData({ isAnalyzing: false });

        if (res.statusCode === 200) {
          try {
            const data = JSON.parse(res.data);
            console.log("识别成功数据:", data);

            // 解析主要识别结果
            const primary = data.primary_result || (data.top3 ? data.top3[0] : null);
            const cropKey = data.crop || (primary ? primary.crop : 'corn');
            const cropZh = data.crop_zh || (cropKey === 'corn' ? '玉米' : cropKey === 'wheat' ? '小麦' : '棉花');

            const stageZh = primary ? (primary.stage_zh || primary.stage) : '未知阶段';
            const stageName = primary ? primary.stage : '';
            const confidence = primary ? primary.confidence : 0.95;
            const confidencePercent = (confidence * 100).toFixed(1);

            const processedTop3 = (data.top3 || []).map(item => ({
              ...item,
              stage_zh: item.stage_zh || item.stage,
              confidencePercent: (item.confidence * 100).toFixed(1)
            }));

            const processedResult = {
              primaryCrop: cropKey,
              cropZh: cropZh,
              stageZh: stageZh,
              stageName: stageName,
              confidence: confidence,
              confidencePercent: confidencePercent,
              top3: processedTop3,
              advice: data.advice || null,
              recordId: data.record_id || Date.now()
            };

            that.setData({ resultData: processedResult });

            // 保存到本地历史记录
            app.addHistoryRecord({
              id: processedResult.recordId,
              timestamp: new Date().toLocaleString(),
              imagePath: that.data.imagePath,
              cropZh: cropZh,
              stageZh: stageZh,
              confidence: (confidence * 100).toFixed(1) + '%'
            });

            wx.showToast({
              title: '识别完成！',
              icon: 'success'
            });

          } catch (e) {
            wx.showToast({
              title: '数据解析失败: ' + e.message,
              icon: 'none'
            });
          }
        } else {
          wx.showToast({
            title: '后端错误 (' + res.statusCode + ')',
            icon: 'none'
          });
        }
      },
      fail(err) {
        wx.hideLoading();
        that.setData({ isAnalyzing: false });
        console.error("上传失败详情:", err);
        wx.showModal({
          title: '网络连接失败',
          content: '无法连接至 API 服务器 (' + apiUrl + ')。\n请检查后端 run_api.py 是否已启动，并确保手机/模拟器与服务器处于同一网络。',
          showCancel: false
        });
      }
    });
  },

  // 3. 服务器配置弹窗
  openServerModal: function () {
    this.setData({
      showServerModal: true,
      inputServerUrl: this.data.serverUrl
    });
  },

  closeServerModal: function () {
    this.setData({ showServerModal: false });
  },

  onServerInput: function (e) {
    this.setData({ inputServerUrl: e.detail.value });
  },

  saveServerConfig: function () {
    let url = this.data.inputServerUrl.trim();
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      wx.showToast({
        title: '请输入以 http:// 或 https:// 开头的有效地址',
        icon: 'none'
      });
      return;
    }
    app.saveServerUrl(url);
    this.setData({
      serverUrl: url,
      showServerModal: false
    });
    wx.showToast({
      title: '服务器配置已保存',
      icon: 'success'
    });
  },

  // 4. 纠错反馈弹窗
  openFeedbackModal: function () {
    this.setData({
      showFeedbackModal: true,
      feedbackNote: ''
    });
  },

  closeFeedbackModal: function () {
    this.setData({ showFeedbackModal: false });
  },

  updateStageOptions: function (cropKey) {
    const cropData = CROP_STAGE_MAP[cropKey];
    if (cropData) {
      this.setData({
        currentStageOptions: cropData.stages,
        selectedStageIndex: 0
      });
    }
  },

  onCropChange: function (e) {
    const idx = parseInt(e.detail.value);
    const cropKey = this.data.cropOptions[idx].id;
    this.setData({ selectedCropIndex: idx });
    this.updateStageOptions(cropKey);
  },

  onStageChange: function (e) {
    this.setData({ selectedStageIndex: parseInt(e.detail.value) });
  },

  onNoteInput: function (e) {
    this.setData({ feedbackNote: e.detail.value });
  },

  submitFeedback: function () {
    if (!this.data.imagePath) {
      wx.showToast({ title: '缺失纠错图片', icon: 'none' });
      return;
    }

    const cropKey = this.data.cropOptions[this.data.selectedCropIndex].id;
    const stageKey = this.data.currentStageOptions[this.data.selectedStageIndex].id;
    const note = this.data.feedbackNote;
    const that = this;

    const apiUrl = app.getApiUrl('/api/feedback/upload');

    wx.showLoading({ title: '正在上传样本库...' });

    wx.uploadFile({
      url: apiUrl,
      filePath: that.data.imagePath,
      name: 'file',
      formData: {
        crop: cropKey,
        stage: stageKey,
        user_note: note,
        is_correct: 0
      },
      success(res) {
        wx.hideLoading();
        if (res.statusCode === 200) {
          that.setData({ showFeedbackModal: false });
          wx.showModal({
            title: '🎉 反馈成功',
            content: '已将修正后的图片与标签存入后端 dataset/user_feedback/ 样本库，感谢协助扩充训练集！',
            showCancel: false
          });
        } else {
          wx.showToast({ title: '提交失败: ' + res.statusCode, icon: 'none' });
        }
      },
      fail(err) {
        wx.hideLoading();
        wx.showToast({ title: '网络通信失败', icon: 'none' });
      }
    });
  }
});
