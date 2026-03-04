const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { execFile, exec, spawn } = require('child_process');

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 650,
    backgroundColor: '#0f0f1a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    titleBarStyle: 'default',
    icon: path.join(__dirname, 'assets', 'icon.png'),
  });

  win.loadFile('index.html');
  win.setTitle('Video Annotator — Sign Language Segmentation');
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// ─── IPC Handlers ──────────────────────────────────────────────────────────────

// Open dialog to choose a folder of videos
ipcMain.handle('open-folder-dialog', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
    title: 'Select folder containing videos',
  });
  if (result.canceled || !result.filePaths.length) return null;
  return result.filePaths[0];
});

// List video files in a folder
ipcMain.handle('list-videos', async (_event, folderPath) => {
  const exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
  try {
    const files = fs.readdirSync(folderPath);
    return files
      .filter(f => exts.includes(path.extname(f).toLowerCase()))
      .sort()
      .map(f => ({
        name: f,
        fullPath: path.join(folderPath, f),
        jsonPath: path.join(folderPath, f.replace(/\.[^.]+$/, '') + '.json'),
      }));
  } catch {
    return [];
  }
});

// Check if a JSON annotation file exists
ipcMain.handle('annotation-exists', async (_event, jsonPath) => {
  return fs.existsSync(jsonPath);
});

// Load existing annotation JSON
ipcMain.handle('load-json', async (_event, jsonPath) => {
  try {
    const raw = fs.readFileSync(jsonPath, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return null;
  }
});

// Save annotation JSON
ipcMain.handle('save-json', async (_event, jsonPath, data) => {
  try {
    fs.writeFileSync(jsonPath, JSON.stringify(data, null, 2), 'utf-8');
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});

// Get video duration via ffprobe
ipcMain.handle('get-video-info', async (_event, videoPath) => {
  return new Promise((resolve) => {
    exec(
      `ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,nb_frames -show_entries format=duration -of json "${videoPath}"`,
      (err, stdout) => {
        if (err) {
          resolve({ duration: null, fps: 30 });
          return;
        }
        try {
          const info = JSON.parse(stdout);
          const duration = parseFloat(info.format?.duration || 0);
          const fpsStr = info.streams?.[0]?.r_frame_rate || '30/1';
          const [num, den] = fpsStr.split('/').map(Number);
          const fps = Math.round(num / den);
          resolve({ duration, fps });
        } catch {
          resolve({ duration: null, fps: 30 });
        }
      }
    );
  });
});

ipcMain.handle('suggest-splits', async (event, videoPath) => {
  return new Promise((resolve) => {
    const pythonPath = 'D:\\PythonProject\\NNKH\\.venv\\Scripts\\python.exe';
    const scriptPath = path.join(__dirname, 'suggest_splits.py');

    console.log(`--- Khởi chạy AI cho: ${videoPath} ---`);

    // Dùng spawn để nhận dữ liệu liên tục (real-time)
    const pyProcess = spawn(pythonPath, [scriptPath, videoPath]);

    let stdoutData = '';
    let stderrData = '';

    pyProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });

    pyProcess.stderr.on('data', (data) => {
      const msg = data.toString();
      stderrData += msg;

      // Bắt dòng PROGRESS:XX từ Python
      if (msg.includes("PROGRESS:")) {
        const parts = msg.split("PROGRESS:");
        const percent = parts[parts.length - 1].trim().split('\n')[0];
        // Gửi phần trăm về giao diện
        event.sender.send('ai-progress', percent);
      }
      console.log(`[Python]: ${msg.trim()}`);
    });

    pyProcess.on('close', (code) => {
      if (code !== 0) {
        console.error("Lỗi script Python:", stderrData);
        resolve({ ok: false, error: stderrData });
        return;
      }

      try {
        // Trích xuất JSON từ stdout
        const jsonMatch = stdoutData.match(/\{.*\}/s);
        if (!jsonMatch) throw new Error("Không tìm thấy JSON trong kết quả");

        const result = JSON.parse(jsonMatch[0]);
        resolve({ ok: true, data: result });
      } catch (e) {
        resolve({ ok: false, error: "Lỗi xử lý kết quả AI" });
      }
    });
  });
});