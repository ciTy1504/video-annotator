/* renderer.js — all UI logic for Video Annotator */
'use strict';

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
    folderPath: null,
    videos: [],          // [{name, fullPath, jsonPath}]
    currentIndex: -1,
    splits: [],          // sorted array of timestamps (seconds)
    duration: 0,
    fps: 30,
    speed: 1,
    saved: true,         // Ban đầu true, nhưng sẽ set false cho video mới
    // videoInfo: null,  // Không dùng, có thể remove
};

const SPEEDS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];

// ─── DOM Refs ─────────────────────────────────────────────────────────────────
const video = document.getElementById('video-player');
const placeholder = document.getElementById('video-placeholder');
const btnOpenFol = document.getElementById('btn-open-folder');
const videoList = document.getElementById('video-list');
const videoListEmpty = document.getElementById('video-list-empty');
const currentName = document.getElementById('current-video-name');
const speedInd = document.getElementById('speed-indicator');
const saveStatus = document.getElementById('save-status');
const btnSave = document.getElementById('btn-save');
const btnNext = document.getElementById('btn-next');
const btnSplit = document.getElementById('btn-split');
const btnUndo = document.getElementById('btn-undo');
const btnPlayPause = document.getElementById('btn-play-pause');
const btnSeekBwd = document.getElementById('btn-seek-bwd');
const btnSeekFwd = document.getElementById('btn-seek-fwd');
const btnSeekBwdFine = document.getElementById('btn-seek-bwd-fine');
const btnSeekFwdFine = document.getElementById('btn-seek-fwd-fine');
const timeCurrent = document.getElementById('time-current');
const timeTotal = document.getElementById('time-total');
const timelineFill = document.getElementById('timeline-fill');
const playhead = document.getElementById('playhead');
const timelineTrack = document.getElementById('timeline-track');
const timelineCanvas = document.getElementById('timeline-canvas');
const splitsList = document.getElementById('splits-list');
const splitsCount = document.getElementById('splits-count');
const segmentsCount = document.getElementById('segments-count');
const statTotal = document.getElementById('stat-total');
const statDone = document.getElementById('stat-done');

// Tạo button Suggest AI
const btnSuggestAI = document.createElement('button');
btnSuggestAI.id = 'btn-suggest-ai';
btnSuggestAI.textContent = 'Suggest by AI';
btnSuggestAI.className = 'btn';  // Style theo CSS, e.g., green button
window.api.onAIProgress((percent) => {
    btnSuggestAI.textContent = `Analyzing... ${percent}%`;
});
// ctx for canvas
let ctx = null;

// ─── Helpers ──────────────────────────────────────────────────────────────────
function formatTime(sec) {
    if (!isFinite(sec)) return '0:00.0';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const d = Math.floor((sec % 1) * 10);
    return `${m}:${String(s).padStart(2, '0')}.${d}`;
}

function markUnsaved() {
    state.saved = false;
    saveStatus.textContent = 'unsaved';
    saveStatus.className = 'badge badge-unsaved';
}

function markSaved() {
    state.saved = true;
    saveStatus.textContent = 'saved';
    saveStatus.className = 'badge badge-saved';
}

// ─── Timeline ─────────────────────────────────────────────────────────────────
function resizeCanvas() {
    timelineCanvas.width = timelineTrack.clientWidth;
    timelineCanvas.height = timelineTrack.clientHeight;
    ctx = timelineCanvas.getContext('2d');
    drawTimeline();
}

function drawTimeline() {
    if (!ctx) return;
    const W = timelineCanvas.width;
    const H = timelineCanvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!state.duration) return;

    ctx.strokeStyle = '#4ade80';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#4ade80';
    ctx.shadowBlur = 4;

    for (const ts of state.splits) {
        const x = Math.round((ts / state.duration) * W);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
    }
    ctx.shadowBlur = 0;
}

function updatePlayhead() {
    if (!state.duration) return;
    const pct = (video.currentTime / state.duration) * 100;
    playhead.style.left = pct + '%';
    timelineFill.style.width = pct + '%';
    timeCurrent.textContent = formatTime(video.currentTime);
}

// Seek by clicking on timeline
timelineTrack.addEventListener('click', (e) => {
    if (!state.duration) return;
    const rect = timelineTrack.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    video.currentTime = pct * state.duration;
});

// ─── Split Management ─────────────────────────────────────────────────────────
function addSplit(ts) {
    const t = parseFloat(ts.toFixed(3));
    if (state.splits.includes(t)) return; // deduplicate
    state.splits.push(t);
    state.splits.sort((a, b) => a - b);
    markUnsaved();
    renderSplitsList();
    drawTimeline();
    // Sau khi add split thủ công, ẩn button AI nếu đang hiển thị
    if (state.splits.length > 0) btnSuggestAI.style.display = 'none';
}

function removeSplit(ts) {
    state.splits = state.splits.filter(s => s !== ts);
    markUnsaved();
    renderSplitsList();
    drawTimeline();
    // Nếu remove hết splits, show button lại nếu chưa saved
    if (state.splits.length === 0 && !state.saved) btnSuggestAI.style.display = 'block';
}

function undoLastSplit() {
    if (!state.splits.length) return;
    state.splits.pop();
    markUnsaved();
    renderSplitsList();
    drawTimeline();
    flashStatus('Undone');
    // Tương tự remove, check để show button nếu cần
    if (state.splits.length === 0 && !state.saved) btnSuggestAI.style.display = 'block';
}

function renderSplitsList() {
    splitsList.innerHTML = '';
    splitsCount.textContent = state.splits.length;
    segmentsCount.textContent = (state.splits.length + 1) + ' segments';

    state.splits.forEach((ts, i) => {
        const li = document.createElement('li');
        li.dataset.ts = ts;

        const idx = document.createElement('span');
        idx.className = 'split-index';
        idx.textContent = '#' + (i + 1);

        const time = document.createElement('span');
        time.className = 'split-time';
        time.textContent = formatTime(ts);

        const del = document.createElement('button');
        del.className = 'btn-delete-split';
        del.textContent = '×';
        del.title = 'Delete this split';
        del.addEventListener('click', (e) => {
            e.stopPropagation();
            removeSplit(ts);
        });

        li.appendChild(idx);
        li.appendChild(time);
        li.appendChild(del);

        // Click to seek to that timestamp
        li.addEventListener('click', () => {
            video.currentTime = ts;
        });

        splitsList.appendChild(li);
        // Animate newest split
        if (i === state.splits.length - 1) {
            li.classList.add('new-split');
        }
    });
}

// ─── Status flash ─────────────────────────────────────────────────────────────
let _flashTimer = null;
function flashStatus(msg) {
    const prev = saveStatus.textContent;
    const prevClass = saveStatus.className;
    saveStatus.textContent = msg;
    saveStatus.className = 'badge';
    if (_flashTimer) clearTimeout(_flashTimer);
    _flashTimer = setTimeout(() => {
        saveStatus.textContent = prev;
        saveStatus.className = prevClass;
    }, 800);
}

// ─── Save / Load ──────────────────────────────────────────────────────────────
async function saveAnnotation() {
    if (state.currentIndex < 0) return;
    const v = state.videos[state.currentIndex];
    const data = {
        video: v.name,
        fps: state.fps,
        duration: state.duration,
        splits: state.splits,
    };
    const res = await window.api.saveJson(v.jsonPath, data);
    if (res.ok) {
        markSaved();
        // Mark video as annotated in list
        const li = videoList.querySelector(`li[data-index="${state.currentIndex}"]`);
        if (li) li.classList.add('annotated');
        updateStats();
        // Sau save, ẩn button AI
        btnSuggestAI.style.display = 'none';
    } else {
        alert('Save failed: ' + res.error);
    }
}

async function loadAnnotation(videoEntry) {
    const existing = await window.api.loadJson(videoEntry.jsonPath);
    if (existing && Array.isArray(existing.splits)) {
        state.splits = existing.splits;
        flashStatus('Resumed');
        return true;  // Trả về true nếu resumed
    } else {
        state.splits = [];
        return false;  // False nếu mới
    }
}

// ─── Video Loading ────────────────────────────────────────────────────────────
async function loadVideo(index) {
    if (index < 0 || index >= state.videos.length) return;

    // Auto-save previous
    if (state.currentIndex >= 0 && !state.saved) {
        await saveAnnotation();
    }

    state.currentIndex = index;
    const v = state.videos[index];
    state.splits = [];
    state.duration = 0;
    state.fps = 30;

    // Highlight in sidebar
    videoList.querySelectorAll('li').forEach(li => li.classList.remove('active'));
    const activeLi = videoList.querySelector(`li[data-index="${index}"]`);
    if (activeLi) {
        activeLi.classList.add('active');
        activeLi.scrollIntoView({ block: 'nearest' });
    }

    currentName.textContent = v.name;
    document.title = `Video Annotator — ${v.name}`;

    // Show video
    video.src = v.fullPath.replace(/\\/g, '/');
    video.style.display = 'block';
    placeholder.style.display = 'none';

    // Load annotation first (before metadata)
    const isResumed = await loadAnnotation(v);

    // Chỉ markSaved nếu resumed (có existing splits), иначе markUnsaved cho video mới
    if (isResumed) {
        markSaved();
    } else {
        markUnsaved();  // Set saved=false cho video mới để show button
    }

    // Get info via ffprobe (fallback to video element metadata)
    const info = await window.api.getVideoInfo(v.fullPath);
    state.fps = info.fps || 30;
    if (info.duration) {
        state.duration = info.duration;
    }

    // Toggle button Suggest AI: Chỉ show nếu chưa có splits và chưa saved (video mới)
    if (state.splits.length === 0 && !state.saved) {
        btnSuggestAI.style.display = 'block';
        console.log('Showing Suggest AI button');  // Debug: Check console
    } else {
        btnSuggestAI.style.display = 'none';
        console.log('Hiding Suggest AI button');  // Debug
    }
}

// After video metadata is ready, use it if ffprobe didn't work
video.addEventListener('loadedmetadata', () => {
    if (!state.duration) {
        state.duration = video.duration;
    }
    timeTotal.textContent = formatTime(state.duration);
    drawTimeline();
    video.play();
    btnPlayPause.textContent = 'Pause';
});

video.addEventListener('timeupdate', updatePlayhead);
video.addEventListener('ended', () => {
    btnPlayPause.textContent = 'Play';
});
video.addEventListener('play', () => { btnPlayPause.textContent = 'Pause'; });
video.addEventListener('pause', () => { btnPlayPause.textContent = 'Play'; });

// ─── Video List ───────────────────────────────────────────────────────────────
async function openFolder() {
    const folderPath = await window.api.openFolderDialog();
    if (!folderPath) return;
    state.folderPath = folderPath;
    const videos = await window.api.listVideos(folderPath);
    state.videos = videos;

    videoList.innerHTML = '';
    if (!videos.length) {
        videoListEmpty.style.display = 'block';
        videoList.style.display = 'none';
        return;
    }
    videoListEmpty.style.display = 'none';
    videoList.style.display = 'block';

    // Build list items, check for existing annotations
    const existChecks = await Promise.all(videos.map(v => window.api.annotationExists(v.jsonPath)));

    videos.forEach((v, i) => {
        const li = document.createElement('li');
        li.dataset.index = i;
        if (existChecks[i]) li.classList.add('annotated');

        const dot = document.createElement('span');
        dot.className = 'status-dot';

        const name = document.createElement('span');
        name.className = 'video-name';
        name.textContent = v.name;
        name.title = v.name;

        li.appendChild(dot);
        li.appendChild(name);
        li.addEventListener('click', () => loadVideo(i));
        videoList.appendChild(li);
    });

    updateStats();

    // Auto-load first unannotated or first video
    const firstUnannotated = existChecks.findIndex(e => !e);
    loadVideo(firstUnannotated >= 0 ? firstUnannotated : 0);
}

function updateStats() {
    statTotal.textContent = state.videos.length + ' videos';
    const doneCount = videoList.querySelectorAll('li.annotated').length;
    statDone.textContent = doneCount + ' done';
}

// ─── Speed ────────────────────────────────────────────────────────────────────
function setSpeed(newSpeed) {
    state.speed = newSpeed;
    video.playbackRate = newSpeed;
    speedInd.textContent = newSpeed + '×';
    flashStatus(newSpeed + '× speed');
}

function cycleSpeedUp() {
    const cur = SPEEDS.indexOf(state.speed);
    if (cur < SPEEDS.length - 1) setSpeed(SPEEDS[cur + 1]);
}

function cycleSpeedDown() {
    const cur = SPEEDS.indexOf(state.speed);
    if (cur > 0) setSpeed(SPEEDS[cur - 1]);
}

// ─── Keyboard Shortcuts ───────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    // Don't hijack if focus is on an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch (true) {
        case e.code === 'Space':
            e.preventDefault();
            addSplit(video.currentTime);
            break;

        case e.code === 'ArrowLeft' && !e.shiftKey && !e.ctrlKey && !e.metaKey:
            e.preventDefault();
            video.currentTime = Math.max(0, video.currentTime - 1);
            break;

        case e.code === 'ArrowRight' && !e.shiftKey && !e.ctrlKey && !e.metaKey:
            e.preventDefault();
            video.currentTime = Math.min(state.duration, video.currentTime + 1);
            break;

        case e.code === 'ArrowLeft' && e.shiftKey:
            e.preventDefault();
            video.currentTime = Math.max(0, video.currentTime - 0.2);
            break;

        case e.code === 'ArrowRight' && e.shiftKey:
            e.preventDefault();
            video.currentTime = Math.min(state.duration, video.currentTime + 0.2);
            break;

        case e.code === 'ArrowUp':
            e.preventDefault();
            cycleSpeedUp();
            break;

        case e.code === 'ArrowDown':
            e.preventDefault();
            cycleSpeedDown();
            break;

        case e.code === 'KeyZ' && (e.ctrlKey || e.metaKey):
            e.preventDefault();
            undoLastSplit();
            break;

        case e.code === 'KeyS' && !e.ctrlKey && !e.metaKey:
            e.preventDefault();
            saveAnnotation();
            break;

        case e.code === 'KeyS' && (e.ctrlKey || e.metaKey):
            e.preventDefault();
            saveAnnotation();
            break;

        case e.code === 'KeyN':
            e.preventDefault();
            goNext();
            break;

        case e.code === 'KeyP' || e.code === 'Period':
            e.preventDefault();
            togglePlayPause();
            break;
    }
});

function togglePlayPause() {
    if (video.paused) { video.play(); } else { video.pause(); }
}

async function goNext() {
    await saveAnnotation();
    if (state.currentIndex < state.videos.length - 1) {
        loadVideo(state.currentIndex + 1);
    } else {
        flashStatus('Last video');
    }
}

// ─── Button listeners ─────────────────────────────────────────────────────────
btnOpenFol.addEventListener('click', openFolder);
btnSave.addEventListener('click', saveAnnotation);
btnNext.addEventListener('click', goNext);
btnSplit.addEventListener('click', () => addSplit(video.currentTime));
btnUndo.addEventListener('click', undoLastSplit);
btnPlayPause.addEventListener('click', togglePlayPause);
btnSeekBwd.addEventListener('click', () => { video.currentTime = Math.max(0, video.currentTime - 1); });
btnSeekFwd.addEventListener('click', () => { video.currentTime = Math.min(state.duration, video.currentTime + 1); });
btnSeekBwdFine.addEventListener('click', () => { video.currentTime = Math.max(0, video.currentTime - 0.2); });
btnSeekFwdFine.addEventListener('click', () => { video.currentTime = Math.min(state.duration, video.currentTime + 0.2); });

// Listener cho Suggest AI
btnSuggestAI.addEventListener('click', async () => {
    if (state.currentIndex < 0) return;

    // Khóa nút và đổi trạng thái
    btnSuggestAI.disabled = true;
    btnSuggestAI.textContent = 'Analyzing... 0%';
    saveStatus.textContent = 'AI is processing...';
    saveStatus.className = 'badge badge-unsaved';

    const v = state.videos[state.currentIndex];

    try {
        const res = await window.api.suggestSplits(v.fullPath);

        if (res.ok) {
            state.splits = res.data.splits || [];
            renderSplitsList();
            drawTimeline();
            markUnsaved();
            flashStatus('AI Suggestions Loaded');
            btnSuggestAI.style.display = 'none'; // Ẩn nút khi xong
        } else {
            alert('AI Error: ' + res.error);
        }
    } catch (err) {
        console.error(err);
        alert('System Error occurred');
    } finally {
        btnSuggestAI.disabled = false;
        btnSuggestAI.textContent = 'Suggest by AI';
    }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
    // Append button Suggest AI một lần khi load app - append vào #transport để nằm cùng controls
    const controlsContainer = document.getElementById('transport') || document.body;  // #transport tồn tại trong index.html
    controlsContainer.appendChild(btnSuggestAI);
    // Ban đầu ẩn button đến khi load video phù hợp
    btnSuggestAI.style.display = 'none';

    resizeCanvas();
});

window.addEventListener('resize', () => {
    resizeCanvas();
});

// ResizeObserver for timeline track
if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(resizeCanvas).observe(timelineTrack);
}

window.api.onAIProgress((percent) => {
    btnSuggestAI.textContent = `Analyzing... ${percent}%`;
});