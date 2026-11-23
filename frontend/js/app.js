/**
 * FreeCloud AI Studio - Frontend Application Controller
 * Handles user interactions, job submissions, progress tracking, and worker matrix visualization.
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- State ---
  let selectedDuration = 1.0;
  let targetWorkers = 4;
  let currentJobId = null;
  let pollInterval = null;
  let uploadedVideoId = null;

  // --- DOM Elements ---
  const backendStatusText = document.getElementById('backendStatusText');
  const backendStatusPill = document.getElementById('backendStatusPill');
  const fleetStatusText = document.getElementById('fleetStatusText');
  const allocationBadge = document.getElementById('allocationBadge');
  const durationButtons = document.querySelectorAll('.duration-btn');

  const jagdhatriPresetBtn = document.getElementById('jagdhatriPresetBtn');
  const storyPromptInput = document.getElementById('storyPromptInput');
  const animeStyleSelect = document.getElementById('animeStyleSelect');
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

  // --- Initial Setup & Health Check ---
  checkSystemHealth();

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
      backendStatusText.textContent = 'Offline (Check Server)';
      backendStatusPill.querySelector('.status-dot').style.backgroundColor = '#ef4444';
    }
  }

  // --- Duration Selector ---
  durationButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      durationButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      selectedDuration = parseFloat(btn.dataset.duration);
      targetWorkers = parseInt(btn.dataset.workers, 10);

      allocationBadge.textContent = `${targetWorkers} Workers Target`;
      fleetStatusText.textContent = `${targetWorkers} Target Workers`;
    });
  });

  // --- Jagdhatri Serial Preset Loader ---
  jagdhatriPresetBtn.addEventListener('click', () => {
    const jagdhatriPrompt =
      'Jagdhatri (Codename: JAS) enters the abandoned warehouse to recover classified intelligence. ' +
      'She detects an ambush and executes a defensive spin-kick disarming the first operative, maintaining razor-sharp focus. ' +
      'Jash arrives at the perimeter, securing the exits with unwavering composure. ' +
      'Dynamic anime action sequence, sharp linework, high-contrast lighting, preserving authentic character identities.';

    storyPromptInput.value = jagdhatriPrompt;
    animeStyleSelect.value = 'ufotable_digital';

    // Highlight button briefly
    jagdhatriPresetBtn.style.borderColor = '#10b981';
    setTimeout(() => {
      jagdhatriPresetBtn.style.borderColor = 'rgba(99, 102, 241, 0.3)';
    }, 800);
  });

  // --- Video Upload Handling ---
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

  // --- Generation Form Submission ---
  generationForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const promptText = storyPromptInput.value.trim();
    if (!promptText) return;

    // Build payload
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
        body: jsonSafeStringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to submit generation job');
      }

      const jobRecord = await res.json();
      currentJobId = jobRecord.job_id;
      jobStateTag.textContent = `Job: ${currentJobId.slice(0, 8)}...`;
      jobStateTag.className = 'job-state-tag running';

      startJobPolling(currentJobId);
    } catch (err) {
      alert(`Submission Error: ${err.message}`);
      setGeneratingState(false);
    }
  });

  // --- Polling Job Status ---
  function startJobPolling(jobId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/jobs/${jobId}`);
        if (!res.ok) return;

        const data = await res.json();
        updateDashboard(data);

        if (data.status === 'completed') {
          clearInterval(pollInterval);
          handleJobCompleted(data);
        } else if (data.status === 'failed' || data.status === 'cancelled') {
          clearInterval(pollInterval);
          handleJobTerminated(data);
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 1500);
  }

  function updateDashboard(data) {
    // Update progress
    progressPercent.textContent = `${Math.round(data.progress_percent)}%`;
    progressBarFill.style.width = `${data.progress_percent}%`;
    progressStage.textContent = data.stage_message || 'Processing scene chunks';

    // Update fleet count
    fleetCountLabel.textContent = `${data.target_workers} Target Workers (${data.completed_chunks}/${data.total_chunks} Chunks Done)`;

    // Render worker cards
    renderWorkerGrid(data.chunks);

    // Update continuity metrics
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
        ? chunk.assigned_worker_id.replace('gha-runner-', 'Runner-')
        : `Worker Slot ${idx + 1}`;

      card.innerHTML = `
        <span class="worker-name" title="${workerLabel}">${workerLabel}</span>
        <span class="worker-task">Shot ${idx + 1} (${chunk.start_sec}s - ${chunk.end_sec}s)</span>
        <span class="worker-status ${statusClass}">${chunk.status.toUpperCase()}</span>
      `;
      workerGrid.appendChild(card);
    });

    simulateWorkerBtn.style.display = 'inline-block';
  }

  function handleJobCompleted(data) {
    setGeneratingState(false);
    jobStateTag.textContent = 'Completed';
    jobStateTag.className = 'job-state-tag completed';
    progressStage.textContent = 'Final coherent video assembled!';

    // Enable Video Player
    const videoUrl = `/api/v1/jobs/${data.job_id}/video`;
    videoSource.src = videoUrl;
    videoPlayer.load();
    videoPlayer.classList.add('active');
    videoPlaceholder.style.display = 'none';

    // Enable Download button
    downloadVideoBtn.href = videoUrl;
    downloadVideoBtn.classList.remove('disabled');
  }

  function handleJobTerminated(data) {
    setGeneratingState(false);
    jobStateTag.textContent = data.status.toUpperCase();
    jobStateTag.className = 'job-state-tag';
    progressStage.textContent = data.stage_message || `Job ${data.status}`;
  }

  // --- Cancel Job ---
  cancelJobBtn.addEventListener('click', async () => {
    if (!currentJobId) return;
    try {
      await fetch(`/api/v1/jobs/${currentJobId}/cancel`, { method: 'POST' });
      jobStateTag.textContent = 'Cancelled';
      clearInterval(pollInterval);
      setGeneratingState(false);
    } catch (err) {
      console.error('Cancel failed:', err);
    }
  });

  // --- Simulate Worker Execution (Demo Tool) ---
  simulateWorkerBtn.addEventListener('click', async () => {
    if (!currentJobId) return;

    // Simulate registration, lease, and completion of one chunk
    const simWorkerId = `sim_worker_${Math.floor(Math.random() * 1000)}`;

    try {
      // 1. Register
      await fetch('/api/v1/workers/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ worker_id: simWorkerId, runner_type: 'local-simulator' }),
      });

      // 2. Claim
      const claimRes = await fetch(`/api/v1/workers/claim-task?worker_id=${simWorkerId}`, {
        method: 'POST',
      });
      const lease = await claimRes.json();

      if (lease.has_task) {
        // 3. Complete chunk
        const compRes = await fetch('/api/v1/workers/complete-chunk', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            worker_id: simWorkerId,
            chunk_id: lease.chunk_id,
            job_id: lease.job_id,
            sha256_checksum: 'simulated_sha256_hash_abc123',
            render_duration_sec: 3.5,
            output_frames_count: 24,
            continuity_metrics: {
              identity_score: 0.91,
              motion_score: 0.88,
              flicker_score: 0.09,
              audio_sync_score: 0.93,
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

  function jsonSafeStringify(obj) {
    return JSON.stringify(obj);
  }
});
