// app.js
App({
  globalData: {
    // 运行模式：'cloud' (微信云托管免IP模式) 或 'custom' (自定义 API 地址模式)
    apiMode: 'cloud',

    // 自定义后端服务器地址（HTTP 模式下使用）
    serverUrl: 'http://127.0.0.1:8000',

    // 微信云托管环境 ID 与服务名称
    cloudEnvId: 'prod-crop-recognition',
    cloudServiceName: 'crop-api',

    userInfo: null,
    historyRecords: []
  },

  onLaunch: function () {
    console.log("🌾 农作物生育期智能诊断微信小程序初始化");

    // 初始化微信云开发环境
    if (wx.cloud) {
      wx.cloud.init({
        traceUser: true
      });
    }

    // 加载持久化的服务器配置
    const savedMode = wx.getStorageSync('api_mode');
    if (savedMode) {
      this.globalData.apiMode = savedMode;
    }
    const savedUrl = wx.getStorageSync('server_url');
    if (savedUrl) {
      this.globalData.serverUrl = savedUrl;
    }
    const savedEnv = wx.getStorageSync('cloud_env_id');
    if (savedEnv) {
      this.globalData.cloudEnvId = savedEnv;
    }

    // 加载历史记录
    const savedRecords = wx.getStorageSync('crop_history_records');
    if (savedRecords && Array.isArray(savedRecords)) {
      this.globalData.historyRecords = savedRecords;
    }
  },

  // 获取 API 完整 URL
  getApiUrl: function (endpoint) {
    let baseUrl = this.globalData.serverUrl || 'http://127.0.0.1:8000';
    if (baseUrl.endsWith('/')) {
      baseUrl = baseUrl.substring(0, baseUrl.length - 1);
    }
    return baseUrl + endpoint;
  },

  // 保存 API 配置
  saveApiConfig: function (mode, url, envId) {
    this.globalData.apiMode = mode;
    this.globalData.serverUrl = url;
    this.globalData.cloudEnvId = envId;

    wx.setStorageSync('api_mode', mode);
    wx.setStorageSync('server_url', url);
    wx.setStorageSync('cloud_env_id', envId);
  },

  // 微信云托管底层免 IP 容器调用
  callCloudContainer: function (options) {
    const that = this;
    return wx.cloud.callContainer({
      config: {
        env: that.globalData.cloudEnvId || 'prod-crop-recognition'
      },
      path: options.path,
      header: Object.assign({
        'X-WX-SERVICE': that.globalData.cloudServiceName || 'crop-api'
      }, options.header || {}),
      method: options.method || 'GET',
      data: options.data
    });
  },

  // 保存历史记录
  addHistoryRecord: function (record) {
    this.globalData.historyRecords.unshift(record);
    if (this.globalData.historyRecords.length > 50) {
      this.globalData.historyRecords = this.globalData.historyRecords.slice(0, 50);
    }
    wx.setStorageSync('crop_history_records', this.globalData.historyRecords);
  }
});
