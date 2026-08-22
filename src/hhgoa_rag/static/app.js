/**
 * HH Goa 2026 Multilingual Voice RAG — Client Application
 * Features:
 * - Real-time Web Audio API frequency waveform visualizer
 * - MediaRecorder audio recording & audio file upload
 * - Sarvam STT & Bulbul TTS integration
 * - Sub-200ms latency & stage telemetry dashboard
 * - Dynamic multilingual script routing & citations inspector
 */

(function () {
  'use strict';

  // DOM Elements
  const micBtn = document.getElementById('mic-btn');
  const micLabel = document.getElementById('mic-label');
  const audioFileInput = document.getElementById('audio-file-input');
  const langSelect = document.getElementById('lang-select');
  const textForm = document.getElementById('text-query-form');
  const queryTextInput = document.getElementById('query-text-input');
  const submitBtn = document.getElementById('submit-query-btn');
  
  const canvas = document.getElementById('audio-visualizer');
  const canvasCtx = canvas.getContext('2d');
  const timerDisplay = document.getElementById('recording-time');
  const statusText = document.getElementById('backend-status-text');
  
  const transcriptBox = document.getElementById('transcript-container');
  const transcriptText = document.getElementById('transcript-text');
  const sttLatencyBadge = document.getElementById('stt-latency-badge');
  const decisionBadge = document.getElementById('decision-badge');
  const answerContent = document.getElementById('answer-content');
  const ttsPlayBtn = document.getElementById('tts-play-btn');
  const audioPlayer = document.getElementById('audio-player');
  
  const totalLatencyVal = document.getElementById('total-latency-value');
  const metricEmbed = document.getElementById('metric-embed');
  const metricRetrieve = document.getElementById('metric-retrieve');
  const metricRerank = document.getElementById('metric-rerank');
  const metricGrounding = document.getElementById('metric-grounding');
  
  const timelineEmbed = document.getElementById('timeline-embed');
  const timelineRetrieve = document.getElementById('timeline-retrieve');
  const timelineRerank = document.getElementById('timeline-rerank');
  const timelineGround = document.getElementById('timeline-ground');
  
  const citationsList = document.getElementById('citations-list');
  const citationsCount = document.getElementById('citations-count');
  const detectedLangTag = document.getElementById('detected-lang-tag');

  // State
  let isRecording = false;
  let mediaRecorder = null;
  let audioChunks = [];
  let audioContext = null;
  let analyser = null;
  let visualizerAnimId = null;
  let recordingStartTime = 0;
  let timerInterval = null;
  let currentAudioBase64 = null;

  // Initialize Canvas Visualizer
  function initCanvas() {
    canvas.width = canvas.parentElement.clientWidth || 500;
    canvas.height = canvas.parentElement.clientHeight || 90;
    drawIdleVisualizer();
  }

  function drawIdleVisualizer() {
    const w = canvas.width;
    const h = canvas.height;
    canvasCtx.clearRect(0, 0, w, h);
    
    // Draw a subtle resting wave
    canvasCtx.beginPath();
    canvasCtx.strokeStyle = 'rgba(56, 189, 248, 0.25)';
    canvasCtx.lineWidth = 2;
    canvasCtx.moveTo(0, h / 2);
    for (let x = 0; x < w; x++) {
      const y = h / 2 + Math.sin(x * 0.05) * 3;
      canvasCtx.lineTo(x, y);
    }
    canvasCtx.stroke();
  }

  // Live Audio Frequency Visualizer
  function drawLiveVisualizer() {
    if (!isRecording || !analyser) return;

    visualizerAnimId = requestAnimationFrame(drawLiveVisualizer);
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);

    const w = canvas.width;
    const h = canvas.height;
    canvasCtx.clearRect(0, 0, w, h);

    const barWidth = (w / bufferLength) * 2.5;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * h * 0.85;

      const gradient = canvasCtx.createLinearGradient(0, h, 0, h - barHeight);
      gradient.addColorStop(0, '#38bdf8');
      gradient.addColorStop(1, '#818cf8');

      canvasCtx.fillStyle = gradient;
      canvasCtx.fillRect(x, h - barHeight, barWidth - 1, barHeight);

      x += barWidth + 1;
    }
  }

  // Audio Recording Handlers
  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser);

      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunks.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        await handleVoiceSubmit(audioBlob);
        stream.getTracks().forEach((track) => track.stop());
        if (audioContext && audioContext.state !== 'closed') {
          audioContext.close();
        }
      };

      mediaRecorder.start(100);
      isRecording = true;
      micBtn.classList.add('recording');
      micLabel.textContent = 'Recording… (Click to Stop)';
      timerDisplay.classList.add('active');

      recordingStartTime = Date.now();
      timerInterval = setInterval(updateTimer, 1000);
      drawLiveVisualizer();
    } catch (err) {
      console.error('Microphone access denied:', err);
      alert('Could not access microphone. Please enable audio permissions or upload an audio file.');
    }
  }

  function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    isRecording = false;
    micBtn.classList.remove('recording');
    micLabel.textContent = 'Processing Voice…';
    timerDisplay.classList.remove('active');
    clearInterval(timerInterval);
    if (visualizerAnimId) cancelAnimationFrame(visualizerAnimId);
    drawIdleVisualizer();
    mediaRecorder.stop();
  }

  function updateTimer() {
    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    timerDisplay.textContent = `${mins}:${secs}`;
  }

  // Voice Query Submission
  async function handleVoiceSubmit(audioBlob) {
    setLoadingState(true, 'Transcribing & running multilingual RAG…');
    const formData = new FormData();
    formData.append('file', audioBlob, 'query.wav');
    
    const selectedLang = langSelect.value;
    if (selectedLang !== 'auto') {
      formData.append('language_hint', selectedLang);
    }
    formData.append('generate_audio', 'true');

    try {
      const response = await fetch('/v1/voice/query', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      renderVoiceResponse(data);
    } catch (err) {
      console.error('Voice query failed:', err);
      renderError(err.message);
    } finally {
      setLoadingState(false);
      micLabel.textContent = 'Press to Speak';
    }
  }

  // Text Query Submission
  async function handleTextSubmit(text, lang) {
    if (!text || !text.trim()) return;
    setLoadingState(true, 'Retrieving from MSMARCO-XI & verifying…');
    transcriptBox.style.display = 'none';

    const payload = {
      question: text.trim(),
      language_hint: lang && lang !== 'auto' ? lang : null,
    };

    try {
      const response = await fetch('/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      renderTextResponse(data);
    } catch (err) {
      console.error('Text query failed:', err);
      renderError(err.message);
    } finally {
      setLoadingState(false);
    }
  }

  // Render Responses
  function renderVoiceResponse(data) {
    // Transcript
    transcriptBox.style.display = 'block';
    transcriptText.textContent = `"${data.transcript || 'No transcript'}"`;
    sttLatencyBadge.textContent = `STT: ${(data.stt_latency_ms || 0).toFixed(1)}ms`;

    renderCommonData(data);

    // Audio Playback
    if (data.audio_base64) {
      currentAudioBase64 = data.audio_base64;
      ttsPlayBtn.style.display = 'inline-flex';
      playAudio(data.audio_base64);
    } else {
      ttsPlayBtn.style.display = 'none';
    }
  }

  function renderTextResponse(data) {
    renderCommonData(data);
    currentAudioBase64 = null;
    if (data.answer && data.decision === 'allow') {
      ttsPlayBtn.style.display = 'inline-flex';
    } else {
      ttsPlayBtn.style.display = 'none';
    }
  }

  function renderCommonData(data) {
    // Decision Badge
    decisionBadge.className = 'decision-badge';
    if (data.decision === 'allow') {
      decisionBadge.classList.add('badge-allow');
      decisionBadge.textContent = 'Allowed (Answered)';
    } else if (data.decision === 'abstain') {
      decisionBadge.classList.add('badge-abstain');
      decisionBadge.textContent = `Abstained (${data.reason_code || 'Ungrounded'})`;
    } else if (data.decision === 'refuse') {
      decisionBadge.classList.add('badge-refuse');
      decisionBadge.textContent = `Refused (${data.reason_code || 'Unsafe'})`;
    } else {
      decisionBadge.classList.add('badge-refuse');
      decisionBadge.textContent = 'Error';
    }

    // Answer Content
    if (data.answer) {
      answerContent.innerHTML = `<p>${escapeHtml(data.answer)}</p>`;
    } else if (data.decision === 'abstain') {
      answerContent.innerHTML = `<p class="placeholder-text" style="color:var(--accent-amber)">System abstained from answering: <b>${escapeHtml(data.reason_code)}</b>. No sufficiently relevant passage was found in the indexed MSMARCO-XI corpus.</p>`;
    } else if (data.decision === 'refuse') {
      answerContent.innerHTML = `<p class="placeholder-text" style="color:var(--accent-rose)">Request refused by guardrail: <b>${escapeHtml(data.reason_code)}</b>.</p>`;
    } else {
      answerContent.innerHTML = `<p class="placeholder-text" style="color:var(--accent-rose)">${escapeHtml((data.error && data.error.message) || 'Query failed')}</p>`;
    }

    // Telemetry & Latencies
    const stages = data.timings_ms || {};
    const total = data.total_backend_ms || stages.total_backend || 0;
    
    totalLatencyVal.textContent = `${total.toFixed(1)} ms`;
    metricEmbed.textContent = `${(stages.query_embed || 0).toFixed(1)} ms`;
    metricRetrieve.textContent = `${(stages.local_hybrid_retrieve || 0).toFixed(1)} ms`;
    metricRerank.textContent = `${(stages.answer_extract || 0).toFixed(1)} ms`;
    metricGrounding.textContent = `${(stages.grounding_verify || 0).toFixed(1)} ms`;

    // Timeline Bar Widths
    const safeTotal = Math.max(total, 1);
    timelineEmbed.style.width = `${((stages.query_embed || 0) / safeTotal) * 100}%`;
    timelineRetrieve.style.width = `${((stages.local_hybrid_retrieve || 0) / safeTotal) * 100}%`;
    timelineRerank.style.width = `${((stages.answer_extract || 0) / safeTotal) * 100}%`;
    timelineGround.style.width = `${((stages.grounding_verify || 0) / safeTotal) * 100}%`;

    // Citations
    detectedLangTag.textContent = `Lang: ${(data.detected_language || '--').toUpperCase()}`;
    const citations = data.citations || [];
    citationsCount.textContent = citations.length;

    if (citations.length === 0) {
      citationsList.innerHTML = '<p class="empty-citations">No citations returned for this query.</p>';
    } else {
      citationsList.innerHTML = citations
        .map(
          (c, idx) => `
          <div class="citation-card">
            <div class="citation-meta">
              <span class="citation-passage-id">#${idx + 1} &bull; ${escapeHtml(c.passage_id || 'passage')}</span>
              <span class="citation-score">RRF Score: ${(c.score || 0).toFixed(4)} &bull; ${c.language.toUpperCase()}</span>
            </div>
            <p class="citation-text">${escapeHtml(c.text || '')}</p>
          </div>
        `
        )
        .join('');
    }
  }

  function playAudio(base64Data) {
    if (!base64Data) return;
    audioPlayer.src = `data:audio/wav;base64,${base64Data}`;
    audioPlayer.play().catch((e) => console.log('Autoplay prevented:', e));
  }

  function speakBrowser(text, lang) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    if (lang === 'hi') utterance.lang = 'hi-IN';
    else if (lang === 'bn') utterance.lang = 'bn-IN';
    else if (lang === 'ta') utterance.lang = 'ta-IN';
    else if (lang === 'mr') utterance.lang = 'mr-IN';
    else if (lang === 'gu') utterance.lang = 'gu-IN';
    else if (lang === 'ur') utterance.lang = 'ur-PK';
    else utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
  }

  function setLoadingState(loading, message) {
    if (loading) {
      submitBtn.disabled = true;
      submitBtn.innerHTML = `<span>Running…</span>`;
      decisionBadge.className = 'decision-badge badge-idle';
      decisionBadge.textContent = 'Processing';
      answerContent.innerHTML = `<p class="placeholder-text">${escapeHtml(message || 'Processing…')}</p>`;
    } else {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
          <line x1="22" y1="2" x2="11" y2="13"/>
          <polygon points="22 2 15 22 11 13 2 9 22 2"/>
        </svg>
        Run RAG
      `;
    }
  }

  function renderError(message) {
    decisionBadge.className = 'decision-badge badge-refuse';
    decisionBadge.textContent = 'Error';
    answerContent.innerHTML = `<p class="placeholder-text" style="color:var(--accent-rose)">Request failed: ${escapeHtml(message)}</p>`;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>"']/g, function (m) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[m];
    });
  }

  // Health Check at Startup
  async function checkHealth() {
    try {
      const res = await fetch('/health/ready');
      if (res.ok) {
        statusText.textContent = 'Backend Online (<200ms)';
      } else {
        statusText.textContent = 'Backend Warming Up';
      }
    } catch {
      statusText.textContent = 'Backend Offline';
    }
  }

  // Event Listeners
  micBtn.addEventListener('click', () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  audioFileInput.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      handleVoiceSubmit(file);
    }
  });

  textForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = queryTextInput.value;
    const lang = langSelect.value;
    handleTextSubmit(q, lang);
  });

  document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const q = chip.dataset.query;
      const lang = chip.dataset.lang;
      queryTextInput.value = q;
      if (lang) langSelect.value = lang;
      handleTextSubmit(q, lang);
    });
  });

  ttsPlayBtn.addEventListener('click', () => {
    if (currentAudioBase64) {
      playAudio(currentAudioBase64);
    } else {
      const answer = answerContent.textContent;
      const lang = langSelect.value !== 'auto' ? langSelect.value : 'en';
      speakBrowser(answer, lang);
    }
  });

  window.addEventListener('resize', initCanvas);

  // Initialize
  initCanvas();
  checkHealth();
})();
