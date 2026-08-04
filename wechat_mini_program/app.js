// app.js
App({
  globalData: {
    // 默认后端服务器地址（可在小程序主页右上方点击“⚙️ 服务器”动态配置）
    serverUrl: 'http://127.0.0.1:8000',
    userInfo: null,
    // 历史诊断记录
    historyRecords: []
  },

  onLaunch: function () {
    console.log("🌾 农作物生育期智能诊断微信小程序初始化");
    // 加载持久化的服务器地址配置
    const savedUrl = wx.getStorageSync('server_url');
    if (savedUrl) {
      this.globalData.serverUrl = savedUrl;
    }
    // 加载历史记录
    const savedRecords = wx.getStorageSync('crop_history_records');
    if (savedRecords && Array.isArray(savedRecords)) {
      this.globalData.historyRecords = savedRecords;
    }
  },

  // 获取标准的全接口地址
  getApiUrl: function (endpoint) {
    let baseUrl = this.globalData.serverUrl || 'http://127.0.0.1:8000';
    if (baseUrl.endsWith('/')) {
      baseUrl = baseUrl.substring(0, baseUrl.length - 1);
    }
    return baseUrl + endpoint;
  },

  // 保存服务器地址
  saveServerUrl: function (url) {
    this.globalData.serverUrl = url;
    wx.setStorageSync('server_url', url);
  },

  // 保存新诊断历史
  addHistoryRecord: function (record) {
    this.globalData.historyRecords.unshift(record);
    // 只保留最近 50 条
    if (this.globalData.historyRecords.length > 50) {
      this.globalData.historyRecords = this.globalData.historyRecords.slice(0, 50);
    }
    wx.setStorageSync('crop_history_records', this.globalData.historyRecords);
  }
});
