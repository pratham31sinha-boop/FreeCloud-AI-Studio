/**
 * FreeCloud AI Studio - Frontend Application Controller
 * Handles user interactions, tab switching, ComfyUI monitoring, video analysis,
 * Jagdhatri showcase, job submissions, progress tracking, and worker orchestration.
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- Application State ---
  let selectedDuration = 1.0;
  let targetWorkers = 4;
  let currentJobId = null;
  let pollInterval = null;
  let elapsedTimerInterval = null;
  let jobStartTime = null;
  let uploadedVideoId = null;

  // --- DOM Elements: Navigation ---
  const navTabs = document.querySelectorAll('.nav-tab');
  const tabPanels = document.querySelectorAll('.tab-panel');

  // --- DOM Elements: Header Status ---
  const backendStatusText = document.getElementById('backendStatusText');
  const backendStatusPill = document.getElementById('backendStatusPill');
  const comfyStatusText = document.getElementById('comfyStatusText');
  const comfyStatusPill = document.getElementById('comfyStatusPill');
  const fleetStatusText = document.getElementById('fleetStatusText');

  // --- DOM Elements: Studio Tab ---
  const allocationBadge = document.getElementById('allocationBadge');
  const durationButtons = document.querySelectorAll('.duration-btn');
  const jagdhatriPresetBtn = document.getElementById('jagdhatriPresetBtn');
  const btnRandomPrompt = document.getElementById('btnRandomPrompt');
  const btnSurpriseMe = document.getElementById('btnSurpriseMe');
  const promptChips = document.querySelectorAll('.prompt-chip');
  const storyPromptInput = document.getElementById('storyPromptInput');
  const animeStyleSelect = document.getElementById('animeStyleSelect');
  const resolutionSelect = document.getElementById('resolutionSelect');
  const uploadDropzone = document.getElementById('uploadDropzone');
  const videoFileInput = document.getElementById('videoFileInput');
  const uploadedFileName = document.getElementById('uploadedFileName');

  const generationForm = document.getElementById('generationForm');
  const generateBtn = document.getElementById('generateBtn');
  const generateBtnText = document.getElementById('generateBtnText');
  const cancelJobBtn = document.getElementById('cancelJobBtn');

  const jobStateTag = document.getElementById('jobStateTag');
  const progressStage = document.getElementById('progressStage');
  const progressPercent = document.getElementById('progressPercent');
  const progressBarFill = document.getElementById('progressBarFill');
  const telemetryEta = document.getElementById('telemetryEta');
  const telemetryElapsed = document.getElementById('telemetryElapsed');
  const telemetryFleet = document.getElementById('telemetryFleet');
  const fleetCountLabel = document.getElementById('fleetCountLabel');
  const workerGrid = document.getElementById('workerGrid');
  const simulateWorkerBtn = document.getElementById('simulateWorkerBtn');

  const metricIdentity = document.getElementById('metricIdentity');
  const metricMotion = document.getElementById('metricMotion');
  const metricAudio = document.getElementById('metricAudio');
  const metricFlicker = document.getElementById('metricFlicker');

  const videoPlayer = document.getElementById('finalVideoPlayer');
  const videoSource = document.getElementById('videoSource');
  const videoPlaceholder = document.getElementById('videoPlaceholder');
  const downloadVideoBtn = document.getElementById('downloadVideoBtn');

  // --- DOM Elements: Jagdhatri Showcase ---
  const btnAnalyzeJagdhatri = document.getElementById('btnAnalyzeJagdhatri');
  const btnGenerateJagdhatri = document.getElementById('btnGenerateJagdhatri');
  const jagdhatriShotsContainer = document.getElementById('jagdhatriShotsContainer');
  const shotCountBadge = document.getElementById('shotCountBadge');

  // --- DOM Elements: Video Analysis Tab ---
  const analysisVideoPathInput = document.getElementById('analysisVideoPathInput');
  const btnRunCustomAnalysis = document.getElementById('btnRunCustomAnalysis');
  const analysisResultsBox = document.getElementById('analysisResultsBox');
  const statResolution = document.getElementById('statResolution');
  const statFps = document.getElementById('statFps');
  const statDuration = document.getElementById('statDuration');
  const statCuts = document.getElementById('statCuts');
  const statCameraStyle = document.getElementById('statCameraStyle');
  const analysisShotsList = document.getElementById('analysisShotsList');

  // --- DOM Elements: Continuity Tab ---
  const continuityCharactersList = document.getElementById('continuityCharactersList');

  // --- DOM Elements: Monitor Tab ---
  const comfyNodeStatus = document.getElementById('comfyNodeStatus');
  const comfyNodeCount = document.getElementById('comfyNodeCount');

  // ==================== INITIALIZATION ====================
  initTabs();
  checkSystemHealth();
  checkComfyUIHealth();
  loadContinuityCharacters();

  // ==================== TAB SWITCHING ====================
  function initTabs() {
    navTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const targetTabId = tab.dataset.tab;
        navTabs.forEach((t) => {
          t.classList.remove('active');
          t.setAttribute('aria-selected', 'false');
        });
        tabPanels.forEach((p) => p.classList.remove('active'));

        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');

        const activePanel = document.getElementById(targetTabId);
        if (activePanel) {
          activePanel.classList.add('active');
        }
      });
    });
  }

  function switchToTab(tabId) {
    const tabBtn = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
    if (tabBtn) tabBtn.click();
  }

  // ==================== HEALTH & DIAGNOSTICS ====================
  async function checkSystemHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        backendStatusText.textContent = 'Backend Ready';
        backendStatusPill.querySelector('.status-dot').style.backgroundColor = '#10b981';
      } else {
        backendStatusText.textContent = 'Backend Degraded';
      }
    } catch (err) {
      backendStatusText.textContent = 'Offline';
      backendStatusPill.querySelector('.status-dot').style.backgroundColor = '#ef4444';
    }
  }

  async function checkComfyUIHealth() {
    try {
      const res = await fetch('/api/v1/comfyui/health');
      if (res.ok) {
        const data = await res.json();
        if (data.online) {
          const portNum = data.base_url ? data.base_url.split(':').pop().replace('/', '') : '8188';
          comfyStatusText.textContent = `ComfyUI Port ${portNum}`;
          if (comfyNodeStatus) {
            comfyNodeStatus.textContent = `Online (Port ${portNum})`;
            comfyNodeStatus.className = 'badge badge-emerald';
          }
        } else {
          comfyStatusText.textContent = 'ComfyUI Offline';
          if (comfyNodeStatus) {
            comfyNodeStatus.textContent = 'Offline';
            comfyNodeStatus.className = 'badge badge-amber';
          }
        }
      }

      // Check registered nodes
      const nodesRes = await fetch('/api/v1/comfyui/nodes');
      if (nodesRes.ok) {
        const nodesData = await nodesRes.json();
        if (comfyNodeCount && nodesData.count) {
          comfyNodeCount.textContent = `${nodesData.count} Registered Nodes`;
        }
      }
    } catch (err) {
      comfyStatusText.textContent = 'ComfyUI Standby';
    }
  }

  // ==================== DURATION SELECTOR ====================
  durationButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      durationButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      selectedDuration = parseFloat(btn.dataset.duration);
      targetWorkers = parseInt(btn.dataset.workers, 10) || 4;

      allocationBadge.textContent = `${targetWorkers} Cloud Servers (16GB RAM Each)`;
      fleetStatusText.textContent = `${targetWorkers} Cloud Servers (16GB Cluster)`;
    });
  });

  // ==================== JAGDHATRI PRESET ====================
  const JAGDHATRI_STORY_PROMPT =
    'Jagdhatri (Codename: JAS) enters the warehouse to recover classified intelligence. ' +
    'She detects an ambush and executes a defensive martial arts spin-kick disarming the first operative, maintaining razor-sharp focus. ' +
    'Jash arrives at the perimeter, securing the exits with unwavering composure. ' +
    'Dynamic anime action sequence, sharp linework, high-contrast lighting, preserving authentic character identities.';

  jagdhatriPresetBtn.addEventListener('click', () => {
    storyPromptInput.value = JAGDHATRI_STORY_PROMPT;
    animeStyleSelect.value = 'minimax_h3_cinematic';

    jagdhatriPresetBtn.style.borderColor = '#10b981';
    setTimeout(() => {
      jagdhatriPresetBtn.style.borderColor = 'rgba(99, 102, 241, 0.3)';
    }, 800);
  });

  // ==================== UNIVERSAL PROMPT LIBRARY & RANDOMIZER ====================
  const UNIVERSAL_PROMPT_LIBRARY = {
    cyberpunk: {
      style: 'ufotable_digital',
      prompts: [
        'Neon-lit Neo-Tokyo rain-drenched streets at night, holographic advertisements flickering. A cybernetic katana warrior in glowing trenchcoat dashes across towering rooftops under digital storm clouds.',
        'Futuristic underground hacking den with cascading green phosphor terminals. A cyborg operative with illuminated neural implants interfaces with an AI core amidst dense steam pipes.',
        'High-speed anti-gravity hoverbike pursuit through illuminated subterranean mega-tunnels. Cyberpunk drone pursuit firing plasma bursts through neon billboards.'
      ]
    },
    fantasy: {
      style: 'ufotable_digital',
      prompts: [
        'Ancient dragon soaring above misty floating islands and crystal waterfalls at sunrise. A hooded dragon-rider with glowing runic staff navigates turbulent cloud canyons.',
        'Enchanted elven forest cathedral bathed in celestial bioluminescence. Ancient sacred guardian spirit materializing with translucent wings of ethereal starlight.',
        'Spellcaster unleashing a spiral vortex of elemental azure flames against an encroaching shadow colossus at the edge of an ancient crumbling fortress.'
      ]
    },
    shinkai: {
      style: 'shinkai_cinematic',
      prompts: [
        'Makoto Shinkai style emotional golden-hour sunset over rural railway crossing. Cherry blossom petals drifting in soft summer breeze, dramatic towering cumulonimbus clouds.',
        'Glistening rain puddles reflecting twilight Tokyo skyline. Two high school students sharing a clear umbrella under gentle pastel rain showers with emotional atmospheric lighting.',
        'Starry night sky over a tranquil mountain lake with a vibrant comet splitting into two glowing trails, reflecting across gentle ripples in breathtaking detail.'
      ]
    },
    scifi: {
      style: 'ufotable_digital',
      prompts: [
        'Epic deep-space orbital defense platform firing particle beams into a celestial nebula. Giant tactical mecha squadron launching into hyperspace with blazing thrusters.',
        'Colossal interstellar colony ship arriving at a binary star system. Swarms of automated drones repairing planetary ring shields against cosmic solar flares.',
        'High-tech humanoid scout exploring crystalline alien caverns on a distant moon, illuminating iridescent alien flora that pulse to harmonic acoustic resonance.'
      ]
    },
    samurai: {
      style: 'mappa_action',
      prompts: [
        'Autumn maple leaves falling in slow motion during a tense bamboo grove katana duel. Two master ronin standing motionless in the mist before an explosive split-second clash.',
        'Moonlit castle rooftop confrontation in feudal Japan. Black-clad shinobi dodging shuriken under dramatic moonlight shadows with dynamic calligraphy-style motion lines.',
        'Lone wandering swordswoman walking across wind-swept pampas grass under a stormy crimson dusk, hand resting calmly on the wrapped hilt of her ancestral nodachi.'
      ]
    },
    ghibli: {
      style: 'ghibli_classic',
      prompts: [
        'Studio Ghibli style whimsical flying machine soaring over emerald rolling hills and sleepy seaside villages. Fluffy white clouds and warm nostalgic afternoon sunlight.',
        'Cozy woodland witch cottage surrounded by giant glowing mushrooms, moss-covered stone walls, and mischievous forest spirits gathering around a steaming brass cauldron.',
        'Quaint vintage steam train traveling along submerged tracks over calm shallow turquoise ocean water, passengers looking out at distant lighthouse islands.'
      ]
    },
    jagdhatri: {
      style: 'ufotable_digital',
      prompts: [
        'Jagdhatri (Codename: JAS) enters the warehouse to recover classified intelligence. She detects an ambush and executes a defensive martial arts spin-kick disarming the first operative with razor-sharp focus.',
        'Undercover agent JAS sprinting through crowded dockyards at dusk, executing acrobatic tactical maneuvers to evade hostile mercenaries while Jash provides long-range surveillance.',
        'High-stakes intelligence exchange atop an abandoned grain elevator. JAS confronts syndicate leader in a fluid high-speed martial arts combat choreography.'
      ]
    },
    shonen: {
      style: 'mappa_action',
      prompts: [
        'High-octane anime tournament arena. Young warrior powering up glowing golden aura as shockwaves crack the bedrock, preparing an ultimate finishing strike with electric sparks.',
        'Intense mountain cliff confrontation between rival fighters. Elemental energy spheres clashing in mid-air, causing shockwaves that shatter surrounding boulders.'
      ]
    }
  };

  function selectRandomThemePrompt(theme = null) {
    const themes = Object.keys(UNIVERSAL_PROMPT_LIBRARY);
    const chosenTheme = (theme && UNIVERSAL_PROMPT_LIBRARY[theme])
      ? theme
      : themes[Math.floor(Math.random() * themes.length)];

    const themeData = UNIVERSAL_PROMPT_LIBRARY[chosenTheme];
    const promptList = themeData.prompts;
    const chosenPrompt = promptList[Math.floor(Math.random() * promptList.length)];

    storyPromptInput.value = chosenPrompt;
    if (themeData.style && animeStyleSelect) {
      animeStyleSelect.value = themeData.style;
    }

    // Visual feedback
    storyPromptInput.style.borderColor = '#10b981';
    storyPromptInput.style.boxShadow = '0 0 12px rgba(16, 185, 129, 0.35)';
    setTimeout(() => {
      storyPromptInput.style.borderColor = '';
      storyPromptInput.style.boxShadow = '';
    }, 700);

    return { theme: chosenTheme, prompt: chosenPrompt };
  }

  if (btnRandomPrompt) {
    btnRandomPrompt.addEventListener('click', () => {
      selectRandomThemePrompt();
    });
  }

  if (btnSurpriseMe) {
    btnSurpriseMe.addEventListener('click', () => {
      selectRandomThemePrompt();
      // Ensure at least 30s or 1 min is selected
      const defaultDurationBtn = document.querySelector('.duration-btn[data-duration="0.5"]') || document.querySelector('.duration-btn[data-duration="1.0"]');
      if (defaultDurationBtn) {
        defaultDurationBtn.click();
      }
      setTimeout(() => {
        generationForm.requestSubmit();
      }, 250);
    });
  }

  promptChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const theme = chip.dataset.theme;
      selectRandomThemePrompt(theme);
      promptChips.forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      setTimeout(() => chip.classList.remove('active'), 1200);
    });
  });

  function formatDuration(sec) {
    if (sec === null || sec === undefined || isNaN(sec)) return '—';
    sec = Math.max(0, Math.round(sec));
    if (sec < 60) return `${sec}s`;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
  }

  // ==================== VIDEO UPLOAD ====================
  uploadDropzone.addEventListener('click', () => videoFileInput.click());

  uploadDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadDropzone.classList.add('dragover');
  });

  uploadDropzone.addEventListener('dragleave', () => {
    uploadDropzone.classList.remove('dragover');
  });

  uploadDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  videoFileInput.addEventListener('change', () => {
    if (videoFileInput.files.length > 0) {
      handleFileUpload(videoFileInput.files[0]);
    }
  });

  async function handleFileUpload(file) {
    uploadedFileName.textContent = `Uploading: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)...`;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/v1/upload/video', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      uploadedVideoId = data.file_id;
      uploadedFileName.textContent = `✓ Uploaded: ${file.name}`;
      uploadedFileName.style.color = '#34d399';
    } catch (err) {
      uploadedFileName.textContent = `✗ Upload error: ${err.message}`;
      uploadedFileName.style.color = '#fb7185';
    }
  }

  // ==================== JAGDHATRI SHOWCASE ACTIONS ====================
  if (btnAnalyzeJagdhatri) {
    btnAnalyzeJagdhatri.addEventListener('click', async () => {
      btnAnalyzeJagdhatri.disabled = true;
      btnAnalyzeJagdhatri.innerHTML = `
        <span class="btn-icon">⏳</span>
        <span>Analyzing Shots & Optical Flow...</span>
      `;
      if (shotCountBadge) shotCountBadge.textContent = 'Computing Optical Flow...';

      try {
        const res = await fetch('/api/v1/analysis/jagdhatri-sample?max_duration_sec=15.0');
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Analysis request failed');
        }

        const report = await res.json();
        renderShotsTimeline(report.scenes, jagdhatriShotsContainer);
        if (shotCountBadge) {
          shotCountBadge.textContent = `${report.detected_cuts_count} Shots Detected (${report.duration_sec}s scanned)`;
        }
      } catch (err) {
        alert(`Analysis Error: ${err.message}`);
        if (shotCountBadge) shotCountBadge.textContent = 'Analysis Failed';
      } finally {
        btnAnalyzeJagdhatri.disabled = false;
        btnAnalyzeJagdhatri.innerHTML = `
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          <span>Analyze Jagdhatri Scene Shots & Motion</span>
        `;
      }
    });
  }

  if (btnGenerateJagdhatri) {
    btnGenerateJagdhatri.addEventListener('click', () => {
      // 1. Switch to Studio Tab
      switchToTab('tab-studio');

      // 2. Preload Jagdhatri details
      storyPromptInput.value = JAGDHATRI_STORY_PROMPT;
      animeStyleSelect.value = 'ufotable_digital';
      uploadedFileName.textContent = '✓ Using: jagdhatri_ep224_reel.mp4 (Sample Reference)';
      uploadedFileName.style.color = '#34d399';

      // 3. Trigger Form submission
      setTimeout(() => {
        generationForm.requestSubmit();
      }, 300);
    });
  }

  function renderShotsTimeline(scenes, container) {
    if (!scenes || scenes.length === 0) {
      container.innerHTML = '<p class="timeline-empty">No scene cuts detected.</p>';
      return;
    }

    container.innerHTML = '';
    scenes.forEach((shot) => {
      const card = document.createElement('div');
      card.className = 'shot-item-card';

      const intensityPct = Math.round(shot.motion_intensity * 100);
      const kfSrc = shot.keyframe_url || `/data/analysis/jagdhatri_ep224/${shot.keyframe_relative_path}`;

      card.innerHTML = `
        <div class="shot-thumb-box">
          <img src="${kfSrc}" alt="Shot ${shot.shot_index + 1}" loading="lazy" onerror="this.style.display='none'">
          <span class="shot-time-tag">${shot.start_sec.toFixed(1)}s - ${shot.end_sec.toFixed(1)}s (${shot.duration_sec.toFixed(1)}s)</span>
        </div>
        <div class="shot-meta-body">
          <div class="shot-title-row">
            <span class="shot-idx">Shot #${shot.shot_index + 1}</span>
            <span class="shot-motion-tag">${shot.camera_movement.replace('_', ' ')}</span>
          </div>
          <div class="motion-meter-box">
            <span class="meter-label">Motion:</span>
            <div class="meter-track">
              <div class="meter-fill" style="width: ${intensityPct}%;"></div>
            </div>
            <span class="meter-label">${intensityPct}%</span>
          </div>
        </div>
      `;
      container.appendChild(card);
    });
  }

  // ==================== VIDEO ANALYSIS STUDIO ====================
  if (btnRunCustomAnalysis) {
    btnRunCustomAnalysis.addEventListener('click', async () => {
      const pathVal = analysisVideoPathInput.value.trim() || 'jagdhatri_ep224_reel.mp4';
      btnRunCustomAnalysis.disabled = true;
      btnRunCustomAnalysis.textContent = 'Processing Analysis...';

      try {
        const res = await fetch('/api/v1/analysis/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            file_id_or_path: pathVal,
            max_duration_sec: 15.0,
            sample_fps: 6.0,
          }),
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Analysis failed');
        }

        const report = await res.json();
        analysisResultsBox.style.display = 'block';

        statResolution.textContent = `${report.width}×${report.height} (${report.aspect_ratio})`;
        statFps.textContent = `${report.fps} FPS`;
        statDuration.textContent = `${report.duration_sec}s`;
        statCuts.textContent = `${report.detected_cuts_count}`;
        statCameraStyle.textContent = (report.motion_summary?.dominant_camera_style || 'Dynamic').replace('_', ' ').toUpperCase();

        renderShotsTimeline(report.scenes, analysisShotsList);
      } catch (err) {
        alert(`Analysis Error: ${err.message}`);
      } finally {
        btnRunCustomAnalysis.disabled = false;
        btnRunCustomAnalysis.textContent = 'Analyze Video File';
      }
    });
  }

  // ==================== CONTINUITY CHARACTERS ====================
  async function loadContinuityCharacters() {
    try {
      const res = await fetch('/api/v1/continuity/characters');
      if (!res.ok) return;

      const characters = await res.json();
      if (!continuityCharactersList) return;

      continuityCharactersList.innerHTML = '';
      characters.forEach((char) => {
        const card = document.createElement('div');
        card.className = 'char-card';

        const isJas = char.character_id.includes('jagdhatri');
        const avatarClass = isJas ? 'jas-avatar' : 'jash-avatar';
        const avatarText = isJas ? 'JAS' : 'JASH';

        const clothingDesc = char.clothing_styles
          ? char.clothing_styles.map((s) => `${s.style_name}: ${s.palette}`).join(' | ')
          : 'Standard Tactical Attire';

        card.innerHTML = `
          <div class="char-avatar ${avatarClass}">${avatarText}</div>
          <div class="char-info" style="flex: 1;">
            <h5>${char.name}</h5>
            <p class="char-role">${char.role} • <em>${char.anime_archetype}</em></p>
            <ul class="char-traits">
              <li><strong>Hair & Eyes:</strong> ${char.physical_traits?.hair || 'Dark'}, ${char.physical_traits?.eyes || 'Brown'}</li>
              <li><strong>Facial Features:</strong> ${char.physical_traits?.facial_features || 'Resolute expression'}</li>
              <li><strong>Attire Styles:</strong> ${clothingDesc}</li>
              <li><strong>LoRA Trigger:</strong> <code>${char.lora_trigger || 'anime_character_v1'}</code></li>
            </ul>
            <div class="meta-tags-list" style="margin-top: 8px;">
              <span class="tag">Identity Drift Cutoff: 0.25</span>
              <span class="tag">Embedding Dimension: 8</span>
              <span class="tag">Pose Tracking: DWPose</span>
            </div>
          </div>
        `;
        continuityCharactersList.appendChild(card);
      });
    } catch (err) {
      console.warn('Failed to load continuity characters:', err);
    }
  }

  // ==================== GENERATION FORM SUBMISSION ====================
  generationForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const promptText = storyPromptInput.value.trim();
    if (!promptText) return;

    const payload = {
      original_video_id: uploadedVideoId,
      story_prompt: promptText,
      duration_minutes: selectedDuration,
      anime_style: animeStyleSelect.value,
      character_ids: ['jagdhatri_main', 'jash_lead'],
      fps: 24,
      continuity_identity_strict: document.getElementById('checkIdentity').checked,
      continuity_motion_tracking: document.getElementById('checkMotion').checked,
      continuity_audio_sync: document.getElementById('checkAudioSync').checked,
    };

    setGeneratingState(true);

    try {
      const res = await fetch('/api/v1/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to submit generation job');
      }

      const jobRecord = await res.json();
      currentJobId = jobRecord.job_id;
      jobStateTag.textContent = `Job: ${currentJobId.slice(0, 8)}...`;
      jobStateTag.className = 'job-state-tag running';

      updateDashboard(jobRecord);
      startJobPolling(currentJobId);
    } catch (err) {
      alert(`Submission Error: ${err.message}`);
      setGeneratingState(false);
    }
  });

  // ==================== POLLING JOB STATUS ====================
  function startJobPolling(jobId) {
    if (pollInterval) clearInterval(pollInterval);
    if (elapsedTimerInterval) clearInterval(elapsedTimerInterval);

    jobStartTime = Date.now();
    elapsedTimerInterval = setInterval(() => {
      if (jobStartTime && telemetryElapsed) {
        const elapsedSec = Math.floor((Date.now() - jobStartTime) / 1000);
        telemetryElapsed.textContent = formatDuration(elapsedSec);
      }
    }, 1000);

    pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/jobs/${jobId}`);
        if (!res.ok) return;

        const data = await res.json();
        updateDashboard(data);

        if (data.status === 'completed') {
          clearInterval(pollInterval);
          if (elapsedTimerInterval) clearInterval(elapsedTimerInterval);
          handleJobCompleted(data);
        } else if (data.status === 'failed' || data.status === 'cancelled') {
          clearInterval(pollInterval);
          if (elapsedTimerInterval) clearInterval(elapsedTimerInterval);
          handleJobTerminated(data);
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 1500);
  }

  function updateDashboard(data) {
    progressPercent.textContent = `${Math.round(data.progress_percent)}%`;
    progressBarFill.style.width = `${data.progress_percent}%`;
    progressStage.textContent = data.stage_message || 'Processing scene chunks';

    fleetCountLabel.textContent = `${data.target_workers} Target Workers (${data.completed_chunks}/${data.total_chunks} Chunks Done)`;

    // Update telemetry bar in real time
    if (telemetryEta) {
      if (data.status === 'completed') {
        telemetryEta.textContent = 'Completed';
      } else if (data.estimated_remaining_sec !== null && data.estimated_remaining_sec !== undefined) {
        telemetryEta.textContent = `~${formatDuration(data.estimated_remaining_sec)}`;
      } else {
        telemetryEta.textContent = 'Calculating...';
      }
    }

    if (telemetryElapsed && data.elapsed_sec !== undefined && data.elapsed_sec !== null) {
      if (!jobStartTime || Math.abs((Date.now() - jobStartTime) / 1000 - data.elapsed_sec) > 3) {
        jobStartTime = Date.now() - (data.elapsed_sec * 1000);
      }
      telemetryElapsed.textContent = formatDuration(data.elapsed_sec);
    }

    if (telemetryFleet) {
      const activeCount = data.active_workers_count || 0;
      telemetryFleet.textContent = activeCount > 0 
        ? `${activeCount} Active / ${data.target_workers} Nodes`
        : `${data.target_workers} Server Nodes`;
    }

    renderWorkerGrid(data.chunks);

    if (data.quality_report) {
      const qr = data.quality_report;
      metricIdentity.textContent = `${Math.round(qr.identity_consistency_score * 100)}%`;
      metricMotion.textContent = `${Math.round(qr.motion_fidelity_score * 100)}%`;
      metricAudio.textContent = `${Math.round(qr.audio_sync_confidence * 100)}%`;
      metricFlicker.textContent = `${(qr.temporal_flicker_score * 100).toFixed(1)}%`;
    }
  }

  function renderWorkerGrid(chunks) {
    if (!chunks || chunks.length === 0) return;

    workerGrid.innerHTML = '';
    chunks.forEach((chunk, idx) => {
      const card = document.createElement('div');
      card.className = 'worker-card';

      const statusClass = chunk.status.toLowerCase();
      const workerLabel = chunk.assigned_worker_id
        ? chunk.assigned_worker_id.replace('gha-runner-', 'Cloud Server ')
        : `Cloud Server Slot ${idx + 1} (16GB RAM)`;

      card.innerHTML = `
        <span class="worker-name" title="${workerLabel}">${workerLabel}</span>
        <span class="worker-task">Shot ${idx + 1} (${chunk.start_sec.toFixed(1)}s - ${chunk.end_sec.toFixed(1)}s)</span>
        <span class="worker-status ${statusClass}">${chunk.status.toUpperCase()}</span>
      `;
      workerGrid.appendChild(card);
    });

    simulateWorkerBtn.style.display = 'inline-block';
  }

  function handleJobCompleted(data) {
    if (elapsedTimerInterval) clearInterval(elapsedTimerInterval);
    setGeneratingState(false);
    jobStateTag.textContent = 'Completed';
    jobStateTag.className = 'job-state-tag completed';
    progressStage.textContent = 'Final coherent video assembled!';
    progressPercent.textContent = '100%';
    progressBarFill.style.width = '100%';
    if (telemetryEta) telemetryEta.textContent = 'Done';

    const videoUrl = `/api/v1/jobs/${data.job_id}/video`;
    videoSource.src = videoUrl;
    videoPlayer.load();
    videoPlayer.classList.add('active');
    videoPlaceholder.style.display = 'none';

    downloadVideoBtn.href = videoUrl;
    downloadVideoBtn.classList.remove('disabled');
  }

  function handleJobTerminated(data) {
    if (elapsedTimerInterval) clearInterval(elapsedTimerInterval);
    setGeneratingState(false);
    jobStateTag.textContent = data.status.toUpperCase();
    jobStateTag.className = 'job-state-tag';
    progressStage.textContent = data.stage_message || `Job ${data.status}`;
  }

  // ==================== CANCEL JOB ====================
  cancelJobBtn.addEventListener('click', async () => {
    if (!currentJobId) return;
    try {
      await fetch(`/api/v1/jobs/${currentJobId}/cancel`, { method: 'POST' });
      jobStateTag.textContent = 'Cancelled';
      if (pollInterval) clearInterval(pollInterval);
      if (elapsedTimerInterval) clearInterval(elapsedTimerInterval);
      setGeneratingState(false);
    } catch (err) {
      console.error('Cancel failed:', err);
    }
  });

  // ==================== WORKER SIMULATION / EXECUTION ====================
  simulateWorkerBtn.addEventListener('click', async () => {
    if (!currentJobId) return;

    const simWorkerId = `sim_worker_${Math.floor(Math.random() * 1000)}`;

    try {
      await fetch('/api/v1/workers/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ worker_id: simWorkerId, runner_type: 'local-simulator' }),
      });

      const claimRes = await fetch(`/api/v1/workers/claim-task?worker_id=${simWorkerId}`, {
        method: 'POST',
      });
      const lease = await claimRes.json();

      if (lease.has_task) {
        await fetch('/api/v1/workers/complete-chunk', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            worker_id: simWorkerId,
            chunk_id: lease.chunk_id,
            job_id: lease.job_id,
            sha256_checksum: 'sha256_verified_render_chunk',
            render_duration_sec: 2.5,
            output_frames_count: 24,
            continuity_metrics: {
              identity_score: 0.93,
              motion_score: 0.90,
              flicker_score: 0.08,
              audio_sync_score: 0.94,
            },
          }),
        });
      }
    } catch (err) {
      console.error('Simulated worker failed:', err);
    }
  });

  function setGeneratingState(isGenerating) {
    generateBtn.disabled = isGenerating;
    generateBtnText.textContent = isGenerating ? 'Orchestrating Workers...' : 'Generate Anime Video';
    cancelJobBtn.style.display = isGenerating ? 'inline-block' : 'none';
  }
});
