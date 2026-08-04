// pages/history/history.js
const app = getApp();

Page({
  data: {
    records: []
  },

  onShow: function () {
    this.loadHistory();
  },

  onPullDownRefresh: function () {
    this.loadHistory();
    wx.stopPullDownRefresh();
  },

  loadHistory: function () {
    const saved = wx.getStorageSync('crop_history_records') || [];
    this.setData({ records: saved });
  },

  clearAllHistory: function () {
    const that = this;
    wx.showModal({
      title: '确认清空记录',
      content: '确定要清空微信小程序上的所有本地诊断历史吗？',
      success(res) {
        if (res.confirm) {
          app.globalData.historyRecords = [];
          wx.removeStorageSync('crop_history_records');
          that.setData({ records: [] });
          wx.showToast({ title: '已清空历史', icon: 'success' });
        }
      }
    });
  }
});
