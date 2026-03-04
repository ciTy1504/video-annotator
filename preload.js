const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
    openFolderDialog: () => ipcRenderer.invoke('open-folder-dialog'),
    listVideos: (folderPath) => ipcRenderer.invoke('list-videos', folderPath),
    annotationExists: (jsonPath) => ipcRenderer.invoke('annotation-exists', jsonPath),
    loadJson: (jsonPath) => ipcRenderer.invoke('load-json', jsonPath),
    saveJson: (jsonPath, data) => ipcRenderer.invoke('save-json', jsonPath, data),
    getVideoInfo: (videoPath) => ipcRenderer.invoke('get-video-info', videoPath),
    suggestSplits: (videoPath) => ipcRenderer.invoke('suggest-splits', videoPath),
    onAIProgress: (callback) => ipcRenderer.on('ai-progress', (event, percent) => callback(percent)),
});
