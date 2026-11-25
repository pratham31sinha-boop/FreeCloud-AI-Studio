/**
 * FreeCloud AI Studio — 24/7 Cloud Serverless + Optional FastAPI Controller
 * Works seamlessly offline-from-PC via GitHub Actions + GitHub Releases CDN,
 * or connected to any FastAPI/Docker backend instance.
 */

const GITHUB_REPO = 'pratham31sinha-boop/FreeCloud-AI-Studio';
const WORKFLOW_FILE = 'autonomous_studio.yml';
const RELEASE_TAG = 'cloud-gallery';

const STORAGE_KEYS = {
  apiUrl: 'fc_cloud_api_url',
  ghToken: 'fc_github_pat',
};

const PROMPT_THEMES = {
  hypercar: 'Sleek silver futuristic hypercar speeding along a golden sunset Pacific coastal cliffside highway, drifting aggressively with glowing tire smoke toward twilight city skyscrapers, 8k photorealistic',
  cyberpunk: 'Cyberpunk anime samurai warrior sprinting across rain-slicked neon Tokyo rooftops under a crimson eclipse, deflecting combat drones with glowing plasma katana slashes, dynamic camera, 8k',
  cosmos: 'Voyager camera gliding smoothly into a luminous spiral nebula galaxy surrounded by swirling cosmic dust pillars and glowing blue supergiant stars, IMAX deep space cinematography, 8k',
  dragon: 'Emerald celestial dragon soaring powerfully between floating emerald sky sanctuaries and cascading waterfalls bathed in golden clouds, dynamic aerial tracking camera, 8k',
  drift: 'Widebody neon coupe drifting sideways through rain-soaked Tokyo elevated expressway bends with glowing blue underglow and sparks reflecting on asphalt, 8k',
  alpine: 'Cinema drone swooping through dramatic sunlit Dolomites granite mountain peaks above swirling morning clouds at golden sunrise, photorealistic RED camera 8k',
};

const RANDOM_PROMPTS = Object.values(PROMPT_THEMES);

document.addEventListener('DOMContentLoaded', () => {
  const state = {
    selectedModel: 'auto_relay',
    durationSeconds: 24,
    durationMinutes: 0.4,
    uploadedVideoId: null,
    apiBase: null, // '' = same-origin FastAPI, 'https://...' = remote API, null = Serverless GitHub mode
    ghToken: localStorage.getItem(STORAGE_KEYS.ghToken) || '',
    galleryItems: [],
    pollTimer: null,
    elapsedTimer: null,
    startTimeMs: 0,
  };

  // DOM References
  const el = {
    navTabs: document.querySelectorAll('.nav-tab'),
    tabPanels: document.querySelectorAll('.tab-panel'),
    modePill: document.getElementById('connectionModePill'),
    modeDot: document.getElementById('connectionDot'),
    modeText: document.getElementById('connectionModeText'),
    cloudBtnLabel: document.getElementById('cloudBtnLabel'),
    openCloudModalBtn: document.getElementById('openCloudModalBtn'),
    quickConnectTokenBtn: document.getElementById('quickConnectTokenBtn'),
    cloudModal: document.getElementById('cloudModal'),
    closeCloudModalBtn: document.getElementById('closeCloudModalBtn'),
    saveCloudSettingsBtn: document.getElementById('saveCloudSettingsBtn'),
    clearCloudSettingsBtn: document.getElementById('clearCloudSettingsBtn'),
    inputCloudApiUrl: document.getElementById('inputCloudApiUrl'),
    inputGhToken: document.getElementById('inputGhToken'),
    modelCards: document.querySelectorAll('.model-card'),
    themeChips: document.querySelectorAll('.chip'),
    durPills: document.querySelectorAll('.dur-pill'),
    promptInput: document.getElementById('storyPromptInput'),
    resSelect: document.getElementById('resolutionSelect'),
    styleSelect: document.getElementById('animeStyleSelect'),
    scalingBadge: document.getElementById('scalingBadge'),
    v2vDrawer: document.getElementById('v2vDrawer'),
    uploadDropzone: document.getElementById('uploadDropzone'),
    videoFileInput: document.getElementById('videoFileInput'),
    uploadedFileName: document.getElementById('uploadedFileName'),
    form: document.getElementById('generationForm'),
    btnRandom: document.getElementById('btnRandomPrompt'),
    btnSurprise: document.getElementById('btnSurpriseMe'),
    genBtn: document.getElementById('generateBtn'),
    genBtnText: document.getElementById('generateBtnText'),
    theaterTitle: document.getElementById('theaterTitle'),
    theaterSubtitle: document.getElementById('theaterSubtitle'),
    jobStateTag: document.getElementById('jobStateTag'),
    videoPlayer: document.getElementById('finalVideoPlayer'),
    videoSource: document.getElementById('videoSource'),
    theaterOverlay: document.getElementById('theaterOverlayPill'),
    specJobChip: document.getElementById('specJobChip'),
    downloadBtn: document.getElementById('downloadVideoBtn'),
    remixBtn: document.getElementById('remixCurrentBtn'),
    openRunLinkBtn: document.getElementById('openRunLinkBtn'),
    progressStage: document.getElementById('progressStage'),
    progressPercent: document.getElementById('progressPercent'),
    progressBarFill: document.getElementById('progressBarFill'),
    teleMode: document.getElementById('teleMode'),
    teleElapsed: document.getElementById('telemetryElapsed'),
    teleFleet: document.getElementById('telemetryFleet'),
    workerGrid: document.getElementById('workerGrid'),
    serverlessHint: document.getElementById('serverlessHintBanner'),
    galleryGrid: document.getElementById('cloudGalleryGrid'),
    galleryBadge: document.getElementById('galleryCountBadge'),
    refreshGalleryBtn: document.getElementById('refreshGalleryBtn'),
    diagEndpoint: document.getElementById('diagBackendEndpoint'),
  };

  initTabs();
  initControls();
  initCloudSettingsModal();
  bootApplication();

  function initTabs() {
    el.navTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.tab;
        el.navTabs.forEach((t) => { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
        el.tabPanels.forEach((p) => p.classList.remove('active'));
        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');
        document.getElementById(target)?.classList.add('active');
      });
    });
  }

  function initControls() {
    el.modelCards.forEach((card) => {
      card.addEventListener('click', () => {
        el.modelCards.forEach((c) => c.classList.remove('active'));
        card.classList.add('active');
        state.selectedModel = card.dataset.model || 'auto_relay';
      });
    });

    el.durPills.forEach((pill) => {
      pill.addEventListener('click', () => {
        el.durPills.forEach((p) => p.classList.remove('active'));
        pill.classList.add('active');
        state.durationSeconds = parseInt(pill.dataset.seconds || '24', 10);
        state.durationMinutes = parseFloat(pill.dataset.minutes || '0.4');
        const vms = parseInt(pill.dataset.vms || '1', 10);
        const acts = Math.max(1, Math.min(4, Math.round(state.durationSeconds / 6))) * vms;
        el.scalingBadge.textContent = `${vms} Cloud VM${vms > 1 ? 's' : ''} • ${acts} Extend-Clip Hops`;
      });
    });

    el.themeChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        const t = chip.dataset.theme;
        if (PROMPT_THEMES[t]) el.promptInput.value = PROMPT_THEMES[t];
      });
    });

    el.btnRandom?.addEventListener('click', () => {
      el.promptInput.value = RANDOM_PROMPTS[Math.floor(Math.random() * RANDOM_PROMPTS.length)];
    });

    el.btnSurprise?.addEventListener('click', () => {
      el.promptInput.value = RANDOM_PROMPTS[Math.floor(Math.random() * RANDOM_PROMPTS.length)];
      el.form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    });

    el.remixBtn?.addEventListener('click', () => {
      const current = state.galleryItems[0];
      if (current?.prompt) el.promptInput.value = current.prompt;
      document.querySelector('[data-tab="tab-studio"]')?.click();
      el.promptInput.focus();
    });

    // Optional Video-to-Video Upload (Active when API backend connected)
    el.uploadDropzone?.addEventListener('click', () => el.videoFileInput.click());
    el.videoFileInput?.addEventListener('change', async () => {
      const file = el.videoFileInput.files?.[0];
      if (!file || state.apiBase === null) return;
      el.uploadedFileName.textContent = `Uploading ${file.name}...`;
      const fd = new FormData();
      fd.append('file', file);
      try {
        const res = await fetch(`${state.apiBase}/api/v1/upload/video`, { method: 'POST', body: fd });
        if (!res.ok) throw new Error('Upload rejected');
        const data = await res.json();
        state.uploadedVideoId = data.file_id;
        el.uploadedFileName.textContent = `Ready: ${file.name} (${data.file_id})`;
      } catch (err) {
        el.uploadedFileName.textContent = `Upload unavailable (${err.message})`;
      }
    });

    el.refreshGalleryBtn?.addEventListener('click', () => loadCloudGallery(true));
    el.form.addEventListener('submit', handleGenerateSubmit);
  }

  function initCloudSettingsModal() {
    const openModal = () => {
      el.inputCloudApiUrl.value = localStorage.getItem(STORAGE_KEYS.apiUrl) || '';
      el.inputGhToken.value = state.ghToken || '';
      el.cloudModal.hidden = false;
    };
    const closeModal = () => { el.cloudModal.hidden = true; };

    el.openCloudModalBtn?.addEventListener('click', openModal);
    el.quickConnectTokenBtn?.addEventListener('click', openModal);
    el.closeCloudModalBtn?.addEventListener('click', closeModal);
    el.cloudModal?.addEventListener('click', (e) => { if (e.target === el.cloudModal) closeModal(); });

    el.saveCloudSettingsBtn?.addEventListener('click', async () => {
      const url = el.inputCloudApiUrl.value.trim().replace(/\/+$/, '');
      const tok = el.inputGhToken.value.trim();
      if (url) localStorage.setItem(STORAGE_KEYS.apiUrl, url);
      else localStorage.removeItem(STORAGE_KEYS.apiUrl);
      if (tok) localStorage.setItem(STORAGE_KEYS.ghToken, tok);
      else localStorage.removeItem(STORAGE_KEYS.ghToken);
      state.ghToken = tok;
      closeModal();
      await detectExecutionPlane();
    });

    el.clearCloudSettingsBtn?.addEventListener('click', async () => {
      localStorage.removeItem(STORAGE_KEYS.apiUrl);
      localStorage.removeItem(STORAGE_KEYS.ghToken);
      el.inputCloudApiUrl.value = '';
      el.inputGhToken.value = '';
      state.ghToken = '';
      closeModal();
      await detectExecutionPlane();
    });
  }

  async function bootApplication() {
    await detectExecutionPlane();
    await loadCloudGallery(false);
  }

  async function detectExecutionPlane() {
    const savedApi = localStorage.getItem(STORAGE_KEYS.apiUrl);
    const candidates = savedApi ? [savedApi] : (location.hostname.endsWith('github.io') ? [] : ['']);

    state.apiBase = null;
    for (const base of candidates) {
      try {
        const controller = new AbortController();
        const tid = setTimeout(() => controller.abort(), 2200);
        const res = await fetch(`${base}/health`, { signal: controller.signal });
        clearTimeout(tid);
        if (res.ok) {
          state.apiBase = base;
          break;
        }
      } catch (_) {}
    }

    if (state.apiBase !== null) {
      const label = state.apiBase ? new URL(state.apiBase).host : location.host;
      el.modeText.textContent = `Connected API (${label})`;
      el.modeDot.className = 'status-dot pulse-green';
      el.cloudBtnLabel.textContent = 'Cloud Settings';
      el.teleMode.textContent = 'FastAPI + GHA';
      el.diagEndpoint.textContent = state.apiBase || location.origin;
      el.serverlessHint.style.display = 'none';
      el.v2vDrawer.style.display = 'block';
    } else {
      el.modeText.textContent = '24/7 Serverless Cloud';
      el.modeDot.className = 'status-dot pulse-cyan';
      el.cloudBtnLabel.textContent = state.ghToken ? 'Token Connected' : 'Connect Cloud';
      el.teleMode.textContent = 'Serverless GitHub';
      el.diagEndpoint.textContent = `https://api.github.com/repos/${GITHUB_REPO}`;
      el.serverlessHint.style.display = state.ghToken ? 'none' : 'flex';
      el.v2vDrawer.style.display = 'none';
    }
  }

  function parseReleaseBody(body) {
    if (!body) return [];
    try {
      const cleaned = body.trim().includes('```json')
        ? body.split('```json')[1].split('```')[0].trim()
        : body.trim();
      const parsed = JSON.parse(cleaned);
      if (parsed && Array.isArray(parsed.items)) return parsed.items;
    } catch (_) {}
    return [];
  }

  async function loadCloudGallery(autoplayFirst = false) {
    try {
      const items = [];
      // 1. Always fetch 24/7 public GitHub Releases CDN catalog (no token needed!)
      const relRes = await fetch(`https://api.github.com/repos/${GITHUB_REPO}/releases/tags/${RELEASE_TAG}`);
      if (relRes.ok) {
        const rel = await relRes.json();
        const catalog = parseReleaseBody(rel.body);
        if (catalog.length > 0) {
          items.push(...catalog);
        } else if (Array.isArray(rel.assets)) {
          rel.assets.filter((a) => a.name.endsWith('.mp4')).forEach((a) => {
            items.push({
              job_id: a.name.replace('.mp4', ''),
              prompt: `Published 1080p Cloud Feature (${a.name})`,
              video_url: a.browser_download_url,
              poster_url: '',
              resolution: '1920x1080',
              duration_seconds: 24,
              engines_used: ['ZeroGPU Diffusion'],
              size_mb: (a.size / (1024 * 1024)).toFixed(2),
            });
          });
        }
      }

      // 2. Also include local/remote FastAPI jobs when backend is connected
      if (state.apiBase !== null) {
        try {
          const jRes = await fetch(`${state.apiBase}/api/v1/jobs`);
          if (jRes.ok) {
            const jobs = await jRes.json();
            jobs.filter((j) => j.status === 'completed' && j.output_video_path).forEach((j) => {
              if (!items.some((x) => x.job_id === j.job_id)) {
                items.push({
                  job_id: j.job_id,
                  prompt: j.request?.story_prompt || 'Generated Studio Feature',
                  video_url: `${state.apiBase}/api/v1/jobs/${j.job_id}/video`,
                  poster_url: '',
                  resolution: `${j.request?.width || 1920}x${j.request?.height || 1080}`,
                  duration_seconds: Math.round((j.request?.duration_minutes || 0.4) * 60),
                  engines_used: [j.request?.model_name || 'auto_relay'],
                  size_mb: 'HD',
                });
              }
            });
          }
        } catch (_) {}
      }

      state.galleryItems = items;
      el.galleryBadge.textContent = String(items.length);
      renderGalleryGrid(items);

      // Load top featured 1080p master into the Theater immediately so screen is never blank!
      if (items.length > 0 && (!el.videoSource.getAttribute('src') || autoplayFirst)) {
        loadIntoTheater(items[0], autoplayFirst);
      }
    } catch (err) {
      el.galleryGrid.innerHTML = `<div class="gallery-loading">Could not reach GitHub Releases CDN (${err.message}).</div>`;
    }
  }

  function loadIntoTheater(item, unmutePlay = true) {
    el.videoSource.src = item.video_url;
    el.videoPlayer.poster = item.poster_url || '';
    el.videoPlayer.load();
    if (unmutePlay) {
      el.videoPlayer.muted = false;
      el.videoPlayer.play().catch(() => {});
    }
    el.theaterTitle.textContent = `1080p Cinema Master — ${item.job_id}`;
    el.theaterSubtitle.textContent = item.prompt || 'Cloud ZeroGPU Diffusion Feature';
    const engText = Array.isArray(item.engines_used) ? item.engines_used.slice(0, 3).join(' → ') : 'ZeroGPU Diffusion';
    el.theaterOverlay.textContent = `${item.resolution || '1920x1080'} • ${item.duration_seconds || 24}s • ${engText}`;
    el.specJobChip.textContent = `Job: ${item.job_id}`;
    el.downloadBtn.href = item.video_url;
    el.downloadBtn.download = `${item.job_id}_1080p.mp4`;
    if (item.run_url) {
      el.openRunLinkBtn.href = item.run_url;
      el.openRunLinkBtn.style.display = 'inline-flex';
    }
  }

  function renderGalleryGrid(items) {
    if (!items.length) {
      el.galleryGrid.innerHTML = '<div class="gallery-loading">No cloud renders published yet. Submit a job to publish your first 1080p master!</div>';
      return;
    }
    el.galleryGrid.innerHTML = '';
    items.forEach((item) => {
      const card = document.createElement('div');
      card.className = 'gallery-card';
      const thumbHtml = item.poster_url
        ? `<img src="${item.poster_url}" alt="${item.job_id}" loading="lazy">`
        : `<video src="${item.video_url}" preload="metadata" muted></video>`;
      const eng = Array.isArray(item.engines_used) ? item.engines_used.join(' → ') : 'Cloud Diffusion';

      card.innerHTML = `
        <div class="gc-thumb">
          ${thumbHtml}
          <span class="gc-badge">${item.resolution || '1080p'} &bull; ${item.duration_seconds || 24}s</span>
        </div>
        <div class="gc-body">
          <div class="gc-prompt">${escapeHtml(item.prompt || 'Cloud AI Video Master')}</div>
          <div class="gc-meta">${item.job_id} &bull; ${item.size_mb || 'HD'} MB &bull; ${escapeHtml(eng)}</div>
          <div class="gc-actions">
            <button type="button" class="mini-btn mini-btn-glow js-watch">▶ Watch 1080p</button>
            <button type="button" class="mini-btn js-remix">♻️ Remix</button>
            <a class="mini-btn" href="${item.video_url}" target="_blank" rel="noopener">⬇️ MP4</a>
          </div>
        </div>
      `;

      card.querySelector('.js-watch').addEventListener('click', () => {
        document.querySelector('[data-tab="tab-studio"]')?.click();
        loadIntoTheater(item, true);
      });
      card.querySelector('.js-remix').addEventListener('click', () => {
        el.promptInput.value = item.prompt || '';
        document.querySelector('[data-tab="tab-studio"]')?.click();
        el.promptInput.focus();
      });
      el.galleryGrid.appendChild(card);
    });
  }

  async function handleGenerateSubmit(e) {
    e.preventDefault();
    const prompt = el.promptInput.value.trim();
    if (!prompt) return;

    const resolution = el.resSelect.value || '1920x1080';
    const [w, h] = resolution.split('x').map((n) => parseInt(n, 10));
    const animeStyle = el.styleSelect.value || 'wan_2_1_realistic';

    if (state.apiBase !== null) {
      await submitViaFastApi(prompt, w, h, animeStyle);
      return;
    }

    if (!state.ghToken) {
      el.serverlessHint.style.display = 'flex';
      el.cloudModal.hidden = false;
      return;
    }

    await submitViaGitHubServerless(prompt, resolution, animeStyle);
  }

  async function submitViaFastApi(prompt, width, height, animeStyle) {
    setBusy(true, 'Dispatching Cloud Workers...');
    try {
      const res = await fetch(`${state.apiBase}/api/v1/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          original_video_id: state.uploadedVideoId,
          story_prompt: prompt,
          duration_minutes: state.durationMinutes,
          anime_style: animeStyle,
          model_name: state.selectedModel,
          model_relay: state.selectedModel === 'auto_relay',
          width,
          height,
          fps: 24,
          execution_mode: 'cloud_github',
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || 'Job failed');
      const job = await res.json();
      pollFastApiJob(job.job_id);
    } catch (err) {
      setBusy(false, `Error: ${err.message}`);
    }
  }

  function pollFastApiJob(jobId) {
    clearTimers();
    state.startTimeMs = Date.now();
    state.elapsedTimer = setInterval(updateClock, 1000);
    el.jobStateTag.textContent = `Rendering ${jobId}`;
    el.jobStateTag.className = 'state-badge running';

    state.pollTimer = setInterval(async () => {
      try {
        const res = await fetch(`${state.apiBase}/api/v1/jobs/${jobId}`);
        if (!res.ok) return;
        const job = await res.json();
        el.progressPercent.textContent = `${Math.round(job.progress_percent || 0)}%`;
        el.progressBarFill.style.width = `${job.progress_percent || 0}%`;
        el.progressStage.textContent = job.stage_message || 'Rendering Cloud Diffusion chunks';
        el.teleFleet.textContent = `${job.completed_chunks}/${job.total_chunks} Chunks`;
        renderChunkChips(job.chunks || []);

        if (job.status === 'completed') {
          clearTimers();
          setBusy(false, 'Completed 1080p Master');
          el.jobStateTag.textContent = 'Completed';
          el.jobStateTag.className = 'state-badge idle';
          loadIntoTheater({
            job_id: jobId,
            prompt: job.request?.story_prompt,
            video_url: `${state.apiBase}/api/v1/jobs/${jobId}/video`,
            resolution: `${job.request?.width || 1920}x${job.request?.height || 1080}`,
            duration_seconds: Math.round((job.request?.duration_minutes || 0.4) * 60),
            engines_used: [job.request?.model_name || 'auto_relay'],
          }, true);
        } else if (job.status === 'failed' || job.status === 'cancelled') {
          clearTimers();
          setBusy(false, job.error || 'Terminated');
          el.jobStateTag.textContent = job.status;
        }
      } catch (_) {}
    }, 2000);
  }

  async function submitViaGitHubServerless(prompt, resolution, animeStyle) {
    const customJobId = `fc_cloud_${Math.random().toString(16).slice(2, 10)}`;
    setBusy(true, 'Dispatching 24/7 GitHub Cloud VM...');
    el.jobStateTag.textContent = `Cloud Run ${customJobId}`;
    el.jobStateTag.className = 'state-badge running';

    const hdrs = {
      Authorization: `Bearer ${state.ghToken}`,
      Accept: 'application/vnd.github+json',
      'Content-Type': 'application/json',
    };

    try {
      const dispatchAt = Date.now() - 3000;
      const dRes = await fetch(`https://api.github.com/repos/${GITHUB_REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches`, {
        method: 'POST',
        headers: hdrs,
        body: JSON.stringify({
          ref: 'main',
          inputs: {
            job_id: customJobId,
            story_prompt: prompt,
            duration_seconds: String(state.durationSeconds),
            model_name: state.selectedModel,
            anime_style: animeStyle,
            resolution,
          },
        }),
      });
      if (!dRes.ok) {
        const txt = await dRes.text();
        throw new Error(`GitHub Dispatch (${dRes.status}): ${txt.slice(0, 120)}`);
      }

      pollGitHubWorkflowRun(customJobId, dispatchAt, hdrs);
    } catch (err) {
      setBusy(false, err.message);
      el.jobStateTag.textContent = 'Dispatch Failed';
      el.jobStateTag.className = 'state-badge idle';
    }
  }

  function pollGitHubWorkflowRun(jobId, dispatchMs, hdrs) {
    clearTimers();
    state.startTimeMs = Date.now();
    state.elapsedTimer = setInterval(updateClock, 1000);
    let trackedRunId = null;

    state.pollTimer = setInterval(async () => {
      try {
        const rRes = await fetch(`https://api.github.com/repos/${GITHUB_REPO}/actions/workflows/${WORKFLOW_FILE}/runs?per_page=5`, { headers: hdrs });
        if (!rRes.ok) return;
        const { workflow_runs = [] } = await rRes.json();

        const run = trackedRunId
          ? workflow_runs.find((r) => r.id === trackedRunId)
          : workflow_runs.find((r) => new Date(r.created_at).getTime() >= dispatchMs);
        if (!run) {
          el.progressStage.textContent = 'Waiting for GitHub Cloud runner allocation...';
          el.progressBarFill.style.width = '12%';
          return;
        }
        trackedRunId = run.id;
        el.openRunLinkBtn.href = run.html_url;
        el.openRunLinkBtn.style.display = 'inline-flex';

        const jRes = await fetch(`https://api.github.com/repos/${GITHUB_REPO}/actions/runs/${run.id}/jobs`, { headers: hdrs });
        if (jRes.ok) {
          const { jobs = [] } = await jRes.json();
          const done = jobs.filter((j) => j.status === 'completed').length;
          const pct = Math.min(95, Math.max(15, Math.round((done / Math.max(1, jobs.length)) * 90)));
          el.progressPercent.textContent = `${pct}%`;
          el.progressBarFill.style.width = `${pct}%`;
          el.teleFleet.textContent = `${jobs.length} Cloud Job${jobs.length > 1 ? 's' : ''}`;
          el.progressStage.textContent = `Cloud ${run.status}: ${jobs.map((j) => `${j.name} (${j.status})`).join(' • ')}`;
          el.workerGrid.innerHTML = jobs.map((j) => `<span class="node-chip">${escapeHtml(j.name)}: ${j.status}</span>`).join('');
        }

        if (run.status === 'completed') {
          clearTimers();
          if (run.conclusion === 'success') {
            el.progressPercent.textContent = '100%';
            el.progressBarFill.style.width = '100%';
            setBusy(false, 'Published to 24/7 Cloud Gallery!');
            el.jobStateTag.textContent = 'Completed';
            el.jobStateTag.className = 'state-badge idle';
            await loadCloudGallery(true);
          } else {
            setBusy(false, `Cloud Run finished: ${run.conclusion}`);
            el.jobStateTag.textContent = run.conclusion || 'Failed';
            el.jobStateTag.className = 'state-badge idle';
          }
        }
      } catch (_) {}
    }, 4500);
  }

  function renderChunkChips(chunks) {
    el.workerGrid.innerHTML = chunks
      .map((c) => `<span class="node-chip">Act ${c.sequence_index + 1} (${c.model_name}): ${c.status}</span>`)
      .join('');
  }

  function setBusy(busy, msg) {
    el.genBtn.disabled = busy;
    el.genBtnText.textContent = busy ? 'Cloud Rendering in Progress...' : 'Generate 1080p Video in 24/7 Cloud';
    if (msg) el.progressStage.textContent = msg;
  }

  function updateClock() {
    const s = Math.floor((Date.now() - state.startTimeMs) / 1000);
    el.teleElapsed.textContent = s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
  }

  function clearTimers() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    if (state.elapsedTimer) clearInterval(state.elapsedTimer);
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
  }
});
