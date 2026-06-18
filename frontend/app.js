/* Looma — single-page UI logic (AC-8).
 *
 * No build step, no framework. Pure browser JS that talks to the
 * FastAPI backend at /api/extract/async and renders the response
 * into the DOM nodes defined in index.html.
 *
 * The flow is the async polling pattern (fix for the 524 timeout
 * bug — long Whisper runs would otherwise hit Cloudflare's edge
 * proxy timeout):
 *
 *   1. POST /api/extract/async          → 202 + {job_id, status_url, result_url}
 *   2. GET  /api/jobs/{job_id}          → {status, progress, stage_msg, ...}
 *      ... poll every 2s until status ∈ {done, failed, timeout}
 *   3. GET  /api/jobs/{job_id}/result   → 200 LoomaResult (or 409/5xx on failure)
 *
 * The legacy synchronous POST /api/extract is still served by the
 * backend for backward compat but the UI no longer uses it.
 *
 * Pure helpers (formatTimestamp, buildMarkdown, escapeHtml) are
 * exposed on the global Looma namespace so they can be unit-tested
 * with Node by running:
 *
 *   node -e "globalThis.window = {}; const s = require('fs').readFileSync('app.js','utf8'); eval(s); console.log(Looma.formatTimestamp(125));"
 *
 * In production they are just called as Looma.formatTimestamp(...).
 */

(function () {
  "use strict";

  // --- Pure helpers (testable) -------------------------------------

  function formatTimestamp(seconds) {
    if (typeof seconds !== "number" || !isFinite(seconds) || seconds < 0) {
      return "0:00";
    }
    var s = Math.floor(seconds);
    var h = Math.floor(s / 3600);
    var m = Math.floor((s % 3600) / 60);
    var sec = s % 60;
    var pad2 = function (n) { return n < 10 ? "0" + n : "" + n; };
    if (h > 0) {
      return h + ":" + pad2(m) + ":" + pad2(sec);
    }
    return m + ":" + pad2(sec);
  }

  function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatTimestampForMarkdown(seconds) {
    // Markdown links: the chapter link uses "[mm:ss]" so it stays
    // copyable as plain text but also reads as a clickable
    // timestamp in editors that support `[mm:ss]()` syntax.
    return "[" + formatTimestamp(seconds) + "]";
  }

  function buildMarkdown(result) {
    if (!result || typeof result !== "object") return "";
    var k = result.knowledge || {};
    var lines = [];
    lines.push("# " + (k.title || result.title || "Untitled"));
    lines.push("");
    if (k.summary) {
      lines.push("## Summary");
      lines.push("");
      lines.push(k.summary);
      lines.push("");
    }
    if (Array.isArray(k.insights) && k.insights.length) {
      lines.push("## Key insights");
      lines.push("");
      for (var i = 0; i < k.insights.length; i++) {
        lines.push("- " + k.insights[i]);
      }
      lines.push("");
    }
    if (Array.isArray(k.chapters) && k.chapters.length) {
      lines.push("## Chapters");
      lines.push("");
      for (var c = 0; c < k.chapters.length; c++) {
        var ch = k.chapters[c];
        var start = formatTimestampForMarkdown(ch.start_seconds);
        lines.push("- " + start + " " + (ch.title || ""));
      }
      lines.push("");
    }
    if (k.narrative) {
      lines.push("## Narration");
      lines.push("");
      lines.push(k.narrative);
      lines.push("");
    }
    if (result.audio_url) {
      lines.push("## Audio");
      lines.push("");
      lines.push("[Listen](" + result.audio_url + ")");
      lines.push("");
    }
    return lines.join("\n");
  }

  // --- DOM helpers -------------------------------------------------

  function $(sel) { return document.querySelector(sel); }
  function $$(sel) { return Array.prototype.slice.call(document.querySelectorAll(sel)); }

  function setText(el, text) {
    if (el) el.textContent = text == null ? "" : String(text);
  }

  // --- UI state -----------------------------------------------------

  // Stage order matches the async backend's reported lifecycle:
  // downloading → transcribing → done. The backend collapses the
  // "extract" and "narrate" sub-pipeline stages into the
  // "transcribing" beat at the API surface (the per-stage
  // ``stage_msg`` from the backend reflects finer-grained
  // progress for the curious).
  var STAGE_ORDER = ["downloading", "transcribing", "done"];

  var ui = {
    audioPlayer: null,
    progressEl: null,
    progressFill: null,
    progressLabel: null,
    progressMsg: null,
    stagesEl: null,
    errorEl: null,
    errorMessageEl: null,
    errorCodeEl: null,
    resultsEl: null,
    titleEl: null,
    summaryEl: null,
    insightsEl: null,
    chaptersEl: null,
    narrativeEl: null,
    copyButton: null,
    analyzeButton: null,
    summaryHint: null,
    settingsHint: null,
    insightsHint: null,
    chaptersHint: null,
    submitButtons: [],
    formYoutube: null,
    formUpload: null,
    lastResult: null,
    lastTranscriptionData: null,  // raw transcription for re-analysis
    progressTimer: null,
  };

  function cacheElements() {
    ui.audioPlayer = $("#audio-player");
    ui.progressEl = $("#progress");
    ui.progressFill = $("#progress-fill");
    ui.progressLabel = $("#progress-label");
    ui.progressMsg = $("#progress-msg");
    ui.stagesEl = $("#stages");
    ui.errorEl = $("#error");
    ui.errorMessageEl = $("#error-message");
    ui.errorCodeEl = $("#error-code");
    ui.resultsEl = $("#results");
    ui.titleEl = $("#result-title");
    ui.summaryEl = $("#result-summary");
    ui.insightsEl = $("#result-insights");
    ui.chaptersEl = $("#result-chapters");
    ui.narrativeEl = $("#result-narrative");
    ui.copyButton = $("#copy-md-button");
    ui.analyzeButton = $("#analyze-button");
    ui.summaryHint = $("#summary-empty-hint");
    ui.settingsHint = $("#settings-hint-text");
    ui.insightsHint = $("#insights-empty-hint");
    ui.chaptersHint = $("#chapters-empty-hint");
    ui.submitButtons = [$("#submit-youtube"), $("#submit-upload")];
    ui.formYoutube = $("#form-youtube");
    ui.formUpload = $("#form-upload");
  }

  // --- Tab switching -----------------------------------------------

  function initTabs() {
    var tabs = $$(".tab");
    var panels = {
      youtube: $("#panel-youtube"),
      upload: $("#panel-upload"),
    };
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        var target = tab.getAttribute("data-tab");
        tabs.forEach(function (other) {
          var isActive = other === tab;
          other.classList.toggle("tab--active", isActive);
          other.setAttribute("aria-selected", isActive ? "true" : "false");
          other.setAttribute("tabindex", isActive ? "0" : "-1");
        });
        Object.keys(panels).forEach(function (key) {
          var panel = panels[key];
          var active = key === target;
          panel.hidden = !active;
          panel.classList.toggle("panel--active", active);
        });
      });
    });
  }

  // --- Progress indicator ------------------------------------------
  //
  // Driven by the real status from ``GET /api/jobs/{id}`` (every
  // 2 s during a poll). Each tick the polling loop calls
  // :func:`setStagesFromStatus` with the latest ``status`` string
  // and the indicator updates to match.

  function resetProgress() {
    $$(".stage").forEach(function (s) {
      s.classList.remove("stage--active", "stage--done");
      s.classList.add("stage--pending");
      var icon = s.querySelector(".stage__icon");
      if (icon) icon.textContent = "";
    });
  }

  function startProgress() {
    resetProgress();
    ui.progressEl.hidden = false;
    setStagesFromStatus("queued");
  }

  // Map the backend's status to the visual stage indicator.
  // ``status`` is one of: ``queued``, ``downloading``,
  // ``transcribing``, ``done``, ``failed``, ``timeout``.
  function setStagesFromStatus(status) {
    if (!ui.stagesEl) return;
    var activeIdx = STAGE_ORDER.indexOf(status);
    if (status === "done") {
      activeIdx = STAGE_ORDER.length; // mark every stage as done
    } else if (status === "failed" || status === "timeout") {
      // Don't move the active pointer; just leave the indicator
      // on whatever stage was last "active" so the user can see
      // where the failure happened.
      return;
    } else if (activeIdx < 0) {
      // Unknown status (``queued``) — show first stage as active.
      activeIdx = 0;
    }
    STAGE_ORDER.forEach(function (stageName, i) {
      var stageEl = ui.stagesEl.querySelector('[data-stage="' + stageName + '"]');
      if (!stageEl) return;
      stageEl.classList.remove("stage--active", "stage--done", "stage--pending");
      if (i < activeIdx) {
        stageEl.classList.add("stage--done");
      } else if (i === activeIdx) {
        stageEl.classList.add("stage--active");
      } else {
        stageEl.classList.add("stage--pending");
      }
    });
  }

  function updateProgressBar(pct) {
    if (ui.progressFill) ui.progressFill.style.width = Math.min(100, Math.max(0, pct)) + "%";
    if (ui.progressLabel) ui.progressLabel.textContent = Math.min(100, Math.max(0, pct)) + "%";
  }

  function finishProgress() {
    updateProgressBar(100);
    setStagesFromStatus("done");
    // Hide after a brief pause so the user sees the "all done" state.
    setTimeout(function () {
      ui.progressEl.hidden = true;
    }, 600);
  }

  // --- Error display -----------------------------------------------

  function showError(message, code) {
    ui.errorEl.hidden = false;
    setText(ui.errorMessageEl, message || "Unknown error.");
    setText(ui.errorCodeEl, code || "INTERNAL_ERROR");
  }

  function clearError() {
    ui.errorEl.hidden = true;
  }

  // --- Result rendering --------------------------------------------

  function renderResult(result) {
    ui.lastResult = result;
    clearError();
    ui.resultsEl.hidden = false;

    var k = result.knowledge || {};
    setText(ui.titleEl, k.title || result.title || "Untitled");
    setText(ui.summaryEl, k.summary || "");
    setText(ui.narrativeEl, k.narrative || "");

    // Insights
    ui.insightsEl.innerHTML = "";
    var insights = Array.isArray(k.insights) ? k.insights : [];
    insights.forEach(function (insight) {
      var li = document.createElement("li");
      li.textContent = insight;
      ui.insightsEl.appendChild(li);
    });

    // Chapters — each is a clickable button that seeks the player.
    ui.chaptersEl.innerHTML = "";
    var chapters = Array.isArray(k.chapters) ? k.chapters : [];
    chapters.forEach(function (ch) {
      var li = document.createElement("li");
      li.className = "chapter";

      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chapter__timestamp";
      btn.textContent = formatTimestamp(ch.start_seconds);
      btn.setAttribute(
        "aria-label",
        "Jump to " + (ch.title || "chapter") + " at " + formatTimestamp(ch.start_seconds)
      );
      btn.addEventListener("click", function () {
        if (ui.audioPlayer) {
          try {
            ui.audioPlayer.currentTime = ch.start_seconds;
            var p = ui.audioPlayer.play();
            if (p && typeof p.catch === "function") p.catch(function () {});
          } catch (e) {
            // Autoplay or seek can fail in some browsers; ignore.
          }
        }
      });

      var title = document.createElement("span");
      title.className = "chapter__title";
      title.textContent = ch.title || "";

      li.appendChild(btn);
      li.appendChild(title);
      ui.chaptersEl.appendChild(li);
    });

    // Audio
    if (ui.audioPlayer && result.audio_url) {
      ui.audioPlayer.src = result.audio_url;
      ui.audioPlayer.load();
    }
  }

  // --- Copy as Markdown --------------------------------------------

  function initCopyButton() {
    if (!ui.copyButton) return;
    ui.copyButton.addEventListener("click", function () {
      if (!ui.lastResult) return;
      var md = buildMarkdown(ui.lastResult);
      copyToClipboard(md).then(function (ok) {
        showToast(ok ? "Copied as Markdown" : "Copy failed");
      });
    });
  }

  // --- Analyze with AI button ---------------------------------------

  function initAnalyzeButton() {
    if (!ui.analyzeButton) return;
    ui.analyzeButton.addEventListener("click", function () {
      // If no API key, open Settings instead
      if (!hasValidSettings()) {
        var modal = document.getElementById("settings-modal");
        if (modal) {
          // Populate form with saved values before showing
          var provEl = document.getElementById("settings-provider");
          var keyEl = document.getElementById("settings-apikey");
          var domainEl = document.getElementById("settings-domain");
          var modelEl = document.getElementById("settings-model");
          var s = loadSettings();
          if (provEl) provEl.value = s.provider;
          if (keyEl) keyEl.value = s.apiKey;
          if (domainEl) domainEl.value = s.domain;
          if (modelEl) modelEl.value = s.model;
          var testResult = document.getElementById("settings-test-result");
          if (testResult) testResult.hidden = true;
          modal.hidden = false;
        }
        return;
      }

      var data = ui.lastTranscriptionData;
      if (!data) {
        showToast("No transcription data available.");
        return;
      }
      runLLMExtraction(data);
    });
  }

  function runLLMExtraction(data) {
    ui.analyzeButton.disabled = true;
    ui.analyzeButton.textContent = "⏳ Analyzing…";
    setBusy(true);
    ui.progressEl.hidden = false;
    if (ui.progressMsg) ui.progressMsg.textContent = "Extracting knowledge via LLM…";
    updateProgressBar(90);
    setStagesFromStatus("transcribing");

    extractWithLLM(data).then(function (knowledge) {
      var renderable = {
        title: knowledge.title,
        audio_url: "",
        knowledge: knowledge,
      };
      ui.lastResult = renderable;
      clearError();
      ui.resultsEl.hidden = false;
      renderResult(renderable);
      finishProgress();
      setBusy(false);
      ui.analyzeButton.disabled = false;
      ui.analyzeButton.textContent = "✨ Analyze with AI";
      ui.analyzeButton.hidden = true;
    }).catch(function (err) {
      setBusy(false);
      showError(err.message || "LLM extraction failed.", "LLM_ERROR");
      ui.progressEl.hidden = true;
      ui.analyzeButton.disabled = false;
      ui.analyzeButton.textContent = "✨ Retry Analysis";
    });
  }

  function copyToClipboard(text) {
    if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(
        function () { return true; },
        function () { return fallbackCopy(text); }
      );
    }
    return Promise.resolve(fallbackCopy(text));
  }

  function fallbackCopy(text) {
    try {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "absolute";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch (e) {
      return false;
    }
  }

  function showToast(message) {
    var existing = document.querySelector(".toast");
    if (existing) existing.remove();
    var toast = document.createElement("div");
    toast.className = "toast";
    toast.setAttribute("role", "status");
    toast.textContent = message;
    document.body.appendChild(toast);
    // Trigger a reflow so the transition runs.
    void toast.offsetWidth;
    toast.classList.add("toast--visible");
    setTimeout(function () {
      toast.classList.remove("toast--visible");
      setTimeout(function () {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
      }, 250);
    }, 1800);
  }

  // --- Form submission ---------------------------------------------

  function setBusy(busy) {
    ui.submitButtons.forEach(function (b) {
      if (b) b.disabled = busy;
    });
  }

  function apiExtract(body, isMultipart) {
    var url = "/api/extract/async";
    var init = { method: "POST" };
    if (isMultipart) {
      init.body = body;
    } else {
      init.headers = { "Content-Type": "application/json" };
      init.body = JSON.stringify(body);
    }
    return fetch(url, init).then(function (resp) {
      return resp.json().then(function (data) {
        return { status: resp.status, data: data };
      });
    });
  }

  // Poll ``GET /api/jobs/{id}`` every 10 s. Resolves with the final
  // status payload when the job reaches a terminal state
  // (``done`` / ``failed`` / ``timeout``); rejects on network error
  // or non-200. The caller is responsible for actually fetching the
  // LoomaResult via ``GET /api/jobs/{id}/result`` once ``status ===
  // "done"``.
  function pollJobStatus(jobId) {
    return new Promise(function (resolve, reject) {
      var interval = setInterval(function () {
        fetch("/api/jobs/" + encodeURIComponent(jobId))
          .then(function (resp) {
            return resp.json().then(function (data) {
              if (resp.status !== 200) {
                clearInterval(interval);
                reject(new Error(
                  (data && data.error) || "Job status check failed (" + resp.status + ")."
                ));
                return;
              }
              if (data.status === "done") {
                clearInterval(interval);
                resolve(data);
              } else if (data.status === "failed" || data.status === "timeout") {
                clearInterval(interval);
                var msg = (data.error && data.error.msg) ||
                  "Job " + data.status + ".";
                var code = (data.error && data.error.code) ||
                  (data.status === "timeout" ? "TIMEOUT" : "INTERNAL_ERROR");
                var err = new Error(msg);
                err.code = code;
                err.jobStatus = data.status;
                reject(err);
              } else {
                // Still running — surface the current stage and
                // progress so the UI can update. ``stage_msg`` is
                // a free string from the backend; ``status`` is one
                // of ``queued`` / ``downloading`` / ``transcribing``
                // and drives the visual stage indicator.
                // ``progress`` (0‑100) powers the progress bar.
                setStagesFromStatus(data.status);
                if (typeof data.progress === "number") {
                  updateProgressBar(data.progress);
                }
                if (ui.progressMsg) {
                  ui.progressMsg.textContent = data.stage_msg || "";
                }
              }
            });
          })
          .catch(function (err) {
            clearInterval(interval);
            reject(err);
          });
      }, 10000);
    });
  }

  function fetchJobResult(jobId) {
    return fetch("/api/jobs/" + encodeURIComponent(jobId) + "/result")
      .then(function (resp) {
        return resp.json().then(function (data) {
          return { status: resp.status, data: data };
        });
      });
  }

  function handleResult(resp) {
    if (resp.status === 200) {
      renderResult(resp.data);
      finishProgress();
    } else {
      var msg = (resp.data && resp.data.error) || "Request failed.";
      var code = (resp.data && resp.data.code) || "INTERNAL_ERROR";
      showError(msg, code);
      ui.progressEl.hidden = true;
    }
  }

  function initForms() {
    if (ui.formYoutube) {
      ui.formYoutube.addEventListener("submit", function (event) {
        event.preventDefault();
        var input = $("#youtube-url-input");
        var url = (input && input.value || "").trim();
        if (!url) return;
        runAsyncJob(
          { youtube_url: url },
          false,
          "youtube-url-input",
        );
      });
    }
    if (ui.formUpload) {
      ui.formUpload.addEventListener("submit", function (event) {
        event.preventDefault();
        var input = $("#file-input");
        var file = input && input.files && input.files[0];
        if (!file) return;
        var fd = new FormData();
        fd.append("file", file, file.name);
        runAsyncJob(fd, true, "file-input");
      });
    }
  }

  // The async submission flow:
  //   POST /api/extract/async  → job_id
  //   poll /api/jobs/{id}      → terminal status
  //   GET  /api/jobs/{id}/result → LoomaResult
  // On any failure the input is preserved so the user can retry
  // without re-typing.
  // On any failure the input is preserved so the user can retry
  // without re-typing.

  // --- Mock transcription (Debug Mode) -------------------------------------
  //
  // When Debug Mode is enabled in Settings, we skip the real backend
  // pipeline and simulate a 10-second transcription flow.  This lets
  // you test the LLM extraction UI without waiting for Whisper.

  function runMockTranscription() {
    var mockPhases = [
      { delay: 1500, status: "downloading", msg: "Downloading audio", pct: 5 },
      { delay: 3000, status: "downloading", msg: "Downloading audio", pct: 10 },
      { delay: 5000, status: "transcribing", msg: "Transcribing audio (33%)", pct: 35 },
      { delay: 7000, status: "transcribing", msg: "Transcribing audio (66%)", pct: 55 },
      { delay: 9000, status: "transcribing", msg: "Transcribing audio (100%)", pct: 85 },
    ];

    var mockResult = {
      status: 200,
      data: {
        transcription: {
          transcript: "So today I want to talk about building better products. " +
            "The key insight is that you need to understand your users deeply before writing any code. " +
            "Many teams jump straight into implementation without proper research. " +
            "That's a mistake that costs time and money. Instead, start with user interviews. " +
            "Talk to at least ten potential users. Understand their pain points. " +
            "Map out their workflow. Then and only then should you think about solutions. " +
            "Another important lesson is to build iteratively. Don't try to ship everything at once. " +
            "Launch with a minimal viable product and gather feedback. " +
            "Use that feedback to improve your product. " +
            "This approach has been proven to work at companies like Dropbox and Airbnb. " +
            "Finally, remember that great products are built by great teams. " +
            "Invest in your team's culture and processes. " +
            "The best technology in the world won't save a product if the team isn't aligned.",
          segments: [
            { start: 0.0, end: 5.0, text: "So today I want to talk about building better products." },
            { start: 5.0, end: 12.0, text: "The key insight is that you need to understand your users deeply before writing any code." },
            { start: 12.0, end: 16.0, text: "Many teams jump straight into implementation without proper research." },
            { start: 16.0, end: 19.0, text: "That's a mistake that costs time and money." },
            { start: 19.0, end: 24.0, text: "Instead, start with user interviews." },
            { start: 24.0, end: 29.0, text: "Talk to at least ten potential users." },
            { start: 29.0, end: 33.0, text: "Understand their pain points." },
            { start: 33.0, end: 38.0, text: "Map out their workflow." },
            { start: 38.0, end: 43.0, text: "Then and only then should you think about solutions." },
            { start: 43.0, end: 50.0, text: "Another important lesson is to build iteratively." },
            { start: 50.0, end: 56.0, text: "Don't try to ship everything at once." },
            { start: 56.0, end: 62.0, text: "Launch with a minimal viable product and gather feedback." },
            { start: 62.0, end: 68.0, text: "Use that feedback to improve your product." },
            { start: 68.0, end: 74.0, text: "This approach has been proven to work at companies like Dropbox and Airbnb." },
            { start: 74.0, end: 81.0, text: "Finally, remember that great products are built by great teams." },
            { start: 81.0, end: 87.0, text: "Invest in your team's culture and processes." },
            { start: 87.0, end: 90.0, text: "The best technology in the world won't save a product if the team isn't aligned." },
          ],
          language: "en",
          duration_seconds: 90.0,
        },
        segments: [
          { start: 0.0, end: 5.0, text: "So today I want to talk about building better products." },
          { start: 5.0, end: 12.0, text: "The key insight is that you need to understand your users deeply before writing any code." },
          { start: 12.0, end: 16.0, text: "Many teams jump straight into implementation without proper research." },
          { start: 16.0, end: 19.0, text: "That's a mistake that costs time and money." },
          { start: 19.0, end: 24.0, text: "Instead, start with user interviews." },
          { start: 24.0, end: 29.0, text: "Talk to at least ten potential users." },
          { start: 29.0, end: 33.0, text: "Understand their pain points." },
          { start: 33.0, end: 38.0, text: "Map out their workflow." },
          { start: 38.0, end: 43.0, text: "Then and only then should you think about solutions." },
          { start: 43.0, end: 50.0, text: "Another important lesson is to build iteratively." },
          { start: 50.0, end: 56.0, text: "Don't try to ship everything at once." },
          { start: 56.0, end: 62.0, text: "Launch with a minimal viable product and gather feedback." },
          { start: 62.0, end: 68.0, text: "Use that feedback to improve your product." },
          { start: 68.0, end: 74.0, text: "This approach has been proven to work at companies like Dropbox and Airbnb." },
          { start: 74.0, end: 81.0, text: "Finally, remember that great products are built by great teams." },
          { start: 81.0, end: 87.0, text: "Invest in your team's culture and processes." },
          { start: 87.0, end: 90.0, text: "The best technology in the world won't save a product if the team isn't aligned." },
        ],
        language: "en",
        duration_seconds: 90.0,
      },
    };

    mockPhases.forEach(function (phase) {
      setTimeout(function () {
        setStagesFromStatus(phase.status);
        updateProgressBar(phase.pct);
        if (ui.progressMsg) ui.progressMsg.textContent = phase.msg;
      }, phase.delay);
    });

    // After 10s, deliver the mock result
    setTimeout(function () {
      handleResult(mockResult);
    }, 10000);
  }

  function runAsyncJob(body, isMultipart, inputId) {
    ui.resultsEl.hidden = true;
    clearError();
    setBusy(true);
    startProgress();

    // === DEBUG MODE: skip the real backend entirely ===
    if (hasDebugMode()) {
      runMockTranscription();
      return;
    }

    apiExtract(body, isMultipart).then(function (submitResp) {
      if (submitResp.status !== 202) {
        setBusy(false);
        handleResult(submitResp);
        return;
      }
      var jobId = submitResp.data && submitResp.data.job_id;
      if (!jobId) {
        setBusy(false);
        showError("Server returned 202 without a job_id.", "INTERNAL_ERROR");
        ui.progressEl.hidden = true;
        return;
      }
      // Persist the job ID in the URL hash so a page refresh
      // doesn't lose it.  The ``init`` function reads it back
      // on load and tries to restore the polling state.
      window.location.hash = jobId;
      // Poll until terminal, then fetch the result.
      pollJobStatus(jobId).then(function () {
        return fetchJobResult(jobId);
      }).then(function (resultResp) {
        setBusy(false);
        handleResult(resultResp);
      }).catch(function (err) {
        setBusy(false);
        showError(err.message || "Request failed.", err.code || "INTERNAL_ERROR");
        ui.progressEl.hidden = true;
      });
    }).catch(function (err) {
      setBusy(false);
      showError(err.message || "Request failed.", "INTERNAL_ERROR");
      ui.progressEl.hidden = true;
    });
  }

  // --- URL hash restore --------------------------------------------
  //
  // When the page is refreshed, the job ID stored in
  // ``window.location.hash`` by ``runAsyncJob`` survives.  This
  // function checks the hash on load, fetches the job status, and
  // resumes the polling / result UI if the job is still valid.
  // If the job has expired or the backend was restarted (404) we
  // just clear the hash and show the blank form.

  function restoreJobFromHash() {
    var hash = window.location.hash.replace(/^#/, "");
    if (!hash) return;

    var jobId = hash;
    fetch("/api/jobs/" + encodeURIComponent(jobId))
      .then(function (resp) {
        if (resp.status !== 200) {
          // Job not found (evicted or backend restarted).
          window.location.hash = "";
          return;
        }
        return resp.json().then(function (data) {
          if (data.status === "done") {
            // Job already finished — fetch and render result.
            setBusy(true);
            startProgress();
            fetchJobResult(jobId).then(function (resultResp) {
              setBusy(false);
              handleResult(resultResp);
            });
          } else if (data.status === "failed" || data.status === "timeout") {
            // Show the error state.
            var msg = (data.error && data.error.msg) ||
              "Job " + data.status + ".";
            var code = (data.error && data.error.code) ||
              (data.status === "timeout" ? "TIMEOUT" : "INTERNAL_ERROR");
            showError(msg, code);
          } else {
            // Still running — resume polling.
            setBusy(true);
            startProgress();
            pollJobStatus(jobId).then(function () {
              return fetchJobResult(jobId);
            }).then(function (resultResp) {
              setBusy(false);
              handleResult(resultResp);
            }).catch(function (err) {
              setBusy(false);
              showError(err.message || "Request failed.", err.code || "INTERNAL_ERROR");
              ui.progressEl.hidden = true;
            });
          }
        });
      })
      .catch(function () {
        // Network error — just clear the hash silently.
        window.location.hash = "";
      });
  }

  // --- BYOK settings (localStorage) ----------------------------------------
  //
  // LLM configuration is stored in the browser. The backend never sees it.
  // Keys are prefixed with ``looma_`` to avoid collisions with other apps.

  var SETTINGS_KEYS = {
    provider: "looma_llm_provider",
    apiKey: "looma_llm_api_key",
    domain: "looma_llm_domain",
    model: "looma_llm_model",
    debug: "looma_debug_mode",
  };

  function loadSettings() {
    return {
      provider: localStorage.getItem(SETTINGS_KEYS.provider) || "openai",
      apiKey: localStorage.getItem(SETTINGS_KEYS.apiKey) || "",
      domain: localStorage.getItem(SETTINGS_KEYS.domain) || "",
      model: localStorage.getItem(SETTINGS_KEYS.model) || "",
      debug: localStorage.getItem(SETTINGS_KEYS.debug) === "true",
    };
  }

  function saveSettings(s) {
    localStorage.setItem(SETTINGS_KEYS.provider, s.provider);
    localStorage.setItem(SETTINGS_KEYS.apiKey, s.apiKey);
    localStorage.setItem(SETTINGS_KEYS.domain, s.domain);
    localStorage.setItem(SETTINGS_KEYS.model, s.model);
    localStorage.setItem(SETTINGS_KEYS.debug, s.debug ? "true" : "false");
  }

  function hasValidSettings() {
    var s = loadSettings();
    return !!s.apiKey;
  }

  function hasDebugMode() {
    return loadSettings().debug;
  }

  // --- LLM provider calls ---------------------------------------------------

  var DEFAULT_DOMAINS = {
    openai: "https://api.openai.com/v1",
    anthropic: "https://api.anthropic.com",
  };

  var DEFAULT_MODELS = {
    openai: "gpt-4o-mini",
    anthropic: "claude-3-5-sonnet-20241022",
  };

  function callLLM(systemPrompt, userPrompt) {
    var settings = loadSettings();
    var provider = settings.provider;
    var apiKey = settings.apiKey;
    if (!apiKey) {
      return Promise.reject(new Error("No API key configured. Open Settings to add one."));
    }
    var domain = settings.domain || DEFAULT_DOMAINS[provider] || DEFAULT_DOMAINS.openai;
    var model = settings.model || DEFAULT_MODELS[provider] || DEFAULT_MODELS.openai;

    if (provider === "anthropic") {
      return callAnthropic(systemPrompt, userPrompt, apiKey, domain, model);
    }
    // Default: OpenAI API format
    return callOpenAI(systemPrompt, userPrompt, apiKey, domain, model);
  }

  function callOpenAI(systemPrompt, userPrompt, apiKey, domain, model) {
    var url = domain.replace(/\/+$/, "") + "/chat/completions";
    return fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + apiKey,
      },
      body: JSON.stringify({
        model: model,
        max_tokens: 4096,
        temperature: 0.2,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
      }),
    }).then(function (resp) {
      if (!resp.ok) {
        return resp.json().then(function (data) {
          var msg = (data && data.error && data.error.message) || resp.statusText;
          throw new Error("OpenAI API error (" + resp.status + "): " + msg);
        });
      }
      return resp.json();
    }).then(function (data) {
      return (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) || "";
    });
  }

  function callAnthropic(systemPrompt, userPrompt, apiKey, domain, model) {
    var url = domain.replace(/\/+$/, "") + "/v1/messages";
    return fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: model,
        max_tokens: 4096,
        temperature: 0.2,
        system: systemPrompt,
        messages: [{ role: "user", content: userPrompt }],
      }),
    }).then(function (resp) {
      if (!resp.ok) {
        return resp.json().then(function (data) {
          var msg = (data && data.error && data.error.message) || resp.statusText;
          throw new Error("Anthropic API error (" + resp.status + "): " + msg);
        });
      }
      return resp.json();
    }).then(function (data) {
      // Anthropic returns content blocks
      if (!data.content) return "";
      var texts = [];
      for (var i = 0; i < data.content.length; i++) {
        if (data.content[i].type === "text") {
          texts.push(data.content[i].text);
        }
      }
      return texts.join("\n").trim();
    });
  }

  // --- JSON repair (ported from backend/app/pipeline/extract.py) -----------

  function stripThinkBlocks(raw) {
    return raw.replace(/<think>[\s\S]*?<\/think>/g, "");
  }

  function stripFences(raw) {
    return raw.replace(/^\s*```(?:json)?\s*|```\s*$/gmi, "").trim();
  }

  function trimAfterLastBrace(s) {
    var last = s.lastIndexOf("}");
    if (last < 0) return s;
    return s.substring(0, last + 1).trim();
  }

  function fixTrailingCommas(s) {
    return s.replace(/,\s*([}\]])/g, "$1");
  }

  function parseLLMJson(raw) {
    var cleaned = stripThinkBlocks(raw);
    cleaned = stripFences(cleaned);

    // Try direct parse first
    try { return JSON.parse(cleaned); } catch (e) {}

    // Repair pass 1: trim after last brace
    var repaired = trimAfterLastBrace(cleaned);
    if (repaired !== cleaned) {
      try { return JSON.parse(repaired); } catch (e) {}
    }

    // Repair pass 2: fix trailing commas
    repaired = fixTrailingCommas(cleaned);
    try { return JSON.parse(repaired); } catch (e) {}

    // All repairs failed
    throw new Error("LLM response is not valid JSON even after repair: " + cleaned.substring(0, 200));
  }

  // --- Post-validation helpers (ported from Python) -------------------------

  function countSentences(text) {
    if (!text || !text.trim()) return 0;
    var cleaned = text.replace(/\b(?:e\.g|i\.e|etc|mr|mrs|ms|dr)\./gi, "");
    var parts = cleaned.split(/[.!?]+(?:\s|$)|[。！？]+/);
    var count = 0;
    for (var i = 0; i < parts.length; i++) {
      if (parts[i].trim()) count++;
    }
    return count;
  }

  function narrativeWordCount(text) {
    if (!text) return 0;
    // CJK characters count as one word each
    var cjk = text.match(/[　-鿿가-힯]/g);
    var cjkLen = cjk ? cjk.length : 0;
    // Latin words
    var latin = text.split(/\s+/).filter(function (w) { return w.trim(); }).length;
    return cjkLen + latin;
  }

  // --- Chapter snapping (ported from Python) --------------------------------

  function snapChapters(chapters, segments, duration) {
    if (!chapters || !chapters.length) return chapters;

    // Collect boundaries from segments
    var boundaries = [0, duration];
    if (segments && segments.length) {
      for (var i = 0; i < segments.length; i++) {
        boundaries.push(segments[i].start, segments[i].end);
      }
    }
    boundaries.sort(function (a, b) { return a - b; });

    function snap(value) {
      var closest = boundaries[0];
      var minDiff = Math.abs(value - closest);
      for (var j = 1; j < boundaries.length; j++) {
        var diff = Math.abs(value - boundaries[j]);
        if (diff < minDiff) { minDiff = diff; closest = boundaries[j]; }
      }
      return closest;
    }

    var snapped = chapters.map(function (ch) {
      return { start_seconds: snap(ch.start_seconds), end_seconds: snap(ch.end_seconds), title: ch.title || "" };
    });

    // Force bookends
    if (snapped.length > 0) {
      snapped[0].start_seconds = 0;
      snapped[snapped.length - 1].end_seconds = duration;
    }

    // Remove duplicates
    var deduped = [snapped[0]];
    for (var k = 1; k < snapped.length; k++) {
      if (snapped[k].start_seconds > deduped[deduped.length - 1].start_seconds) {
        deduped.push(snapped[k]);
      }
    }

    // Force contiguous
    for (var m = 0; m < deduped.length; m++) {
      if (m < deduped.length - 1) {
        deduped[m].end_seconds = deduped[m + 1].start_seconds;
      } else {
        deduped[m].end_seconds = duration;
      }
      deduped[m].start_seconds = m === 0 ? 0 : deduped[m].start_seconds;
    }

    return deduped;
  }

  // --- Build prompts (ported from Python prompt templates) -------------------

  var SYSTEM_PROMPT =
    "You are Looma's knowledge-extraction engine. You receive a raw\n" +
    "auto-generated transcript (with timestamps and occasional filler words)\n" +
    "and you must return a single, strict JSON object that captures the\n" +
    "*value* of the content — never a verbatim transcript.\n\n" +
    "Strict output contract\n" +
    "======================\n" +
    "You MUST return exactly one JSON object with these keys and no others:\n\n" +
    '{"title":"...","summary":"...","insights":["..."],"chapters":[{"start_seconds":0,"end_seconds":10,"title":"..."}],"narrative":"...","filler_removed":0}\n\n' +
    "Rules\n" +
    "=====\n" +
    "1. NO prose, NO markdown fences, NO comments outside the JSON object.\n" +
    "2. Remove filler: um, uh, like, you know, basically, actually, kind of, sort of.\n" +
    "3. Reframe sentences in clear declarative form.\n" +
    "4. Insights must be imperative (\"Build a backlog before scaling\").\n" +
    "5. Chapter timestamps MUST be on segment boundaries. First chapter start=0, last chapter end=duration.\n" +
    "6. Narrative is TTS-friendly flowing prose, 150-400 words.\n" +
    "7. filler_removed is your honest count of filler tokens removed.\n\n" +
    "Return ONLY the JSON object. No prose, no markdown, no commentary.";

  function buildSegmentAnchors(segments) {
    if (!segments || !segments.length) return "(no segments available)";
    var lines = [];
    for (var i = 0; i < segments.length; i++) {
      var s = segments[i];
      var pad = (i < 10 ? "00" : i < 100 ? "0" : "");
      lines.push("[" + pad + i + "] " + s.start.toFixed(1) + "-" + s.end.toFixed(1) + ": " + s.text);
    }
    return lines.join("\n");
  }

  function buildUserPrompt(transcription, retryNotice) {
    var transcript = transcription.transcription && transcription.transcription.transcript
      ? transcription.transcription.transcript
      : "";
    var segments = transcription.segments || [];
    var duration = transcription.duration_seconds || 0;
    var language = transcription.language || "en";

    // Truncate to 30k chars if needed
    var truncated = transcript;
    var truncMarker = "";
    if (transcript.length > 30000) {
      truncated = transcript.substring(0, 30000) + "\n\n[... transcript truncated ...]";
      truncMarker = " [TRUNCATED]";
    }

    var anchors = buildSegmentAnchors(segments);
    var segCount = segments.length;

    var retry = retryNotice || "";

    return (
      "VIDEO METADATA\n" +
      "==============\n" +
      "language:        " + language + "\n" +
      "duration:        " + duration.toFixed(1) + " seconds\n" +
      "transcript_chars:" + transcript.length + truncMarker + "\n\n" +
      "SEGMENT ANCHORS\n" +
      "===============\n" +
      "Use these " + segCount + " segment boundaries to align chapter timestamps.\n" +
      "The first chapter MUST start at start_seconds=0.0 and the last chapter\n" +
      "MUST end at end_seconds=" + duration.toFixed(1) + ".\n\n" +
      anchors + "\n\n" +
      "TRANSCRIPT\n" +
      "==========\n" +
      truncated + "\n\n" +
      retry +
      "\n\nReturn ONLY the JSON object. No prose, no markdown, no commentary."
    );
  }

  // --- LLM extraction step --------------------------------------------------

  function extractWithLLM(transcription, retries) {
    if (retries === undefined) retries = 3;

    function attempt(remaining) {
      var retryNotice = "";
      if (remaining < retries) {
        retryNotice = "\n\n[RETRY NOTICE — your previous response failed validation. Respond ONLY with the corrected JSON object. Make sure ALL strings are properly closed, all special characters are escaped, and the JSON is complete and well-formed.]\n";
      }

      var userPrompt = buildUserPrompt(transcription, retryNotice);

      return callLLM(SYSTEM_PROMPT, userPrompt).then(function (raw) {
        // Parse
        var payload;
        try {
          payload = parseLLMJson(raw);
        } catch (e) {
          if (remaining <= 1) throw new Error("LLM returned invalid JSON after " + retries + " attempts: " + e.message);
          return attempt(remaining - 1);
        }

        // Validate required fields
        if (!payload.title || !payload.summary || !payload.narrative) {
          if (remaining <= 1) throw new Error("LLM response missing required fields.");
          return attempt(remaining - 1);
        }

        // Apply post-validation
        var issues = [];
        if (payload.title.length > 120) issues.push("Title too long");
        var sentCount = countSentences(payload.summary);
        if (sentCount < 3 || sentCount > 5) issues.push("Summary has " + sentCount + " sentences; need 3-5");
        if (!payload.insights || payload.insights.length < 5 || payload.insights.length > 10) {
          issues.push("Insights count: " + (payload.insights ? payload.insights.length : 0) + "; need 5-10");
        }
        var wc = narrativeWordCount(payload.narrative);
        if (wc < 150 || wc > 500) issues.push("Narrative word count: " + wc + "; need 150-500");

        if (issues.length > 0) {
          if (remaining <= 1) throw new Error("LLM failed validation: " + issues.join("; "));
          return attempt(remaining - 1);
        }

        // Snap chapters
        var chapters = payload.chapters || [];
        var snappedChapters = snapChapters(
          chapters,
          transcription.segments || [],
          transcription.duration_seconds || 0
        );

        return {
          title: payload.title,
          summary: payload.summary,
          insights: payload.insights || [],
          chapters: snappedChapters,
          narrative: payload.narrative,
          filler_removed: typeof payload.filler_removed === "number" ? payload.filler_removed : 0,
        };
      });
    }

    return attempt(retries);
  }

  // --- Settings UI ----------------------------------------------------------

  function initSettings() {
    var modal = document.getElementById("settings-modal");
    var openBtn = document.getElementById("settings-button");
    var closeBtn = document.getElementById("settings-close");
    var form = document.getElementById("settings-form");
    var provEl = document.getElementById("settings-provider");
    var keyEl = document.getElementById("settings-apikey");
    var domainEl = document.getElementById("settings-domain");
    var modelEl = document.getElementById("settings-model");
    var debugEl = document.getElementById("settings-debug");
    var testBtn = document.getElementById("settings-test");
    var testResult = document.getElementById("settings-test-result");

    if (!modal || !openBtn) return;

    // Load existing settings into form
    function populateForm() {
      var s = loadSettings();
      provEl.value = s.provider;
      keyEl.value = s.apiKey;
      domainEl.value = s.domain;
      modelEl.value = s.model;
      if (debugEl) debugEl.checked = s.debug;
    }

    // Open
    openBtn.addEventListener("click", function () {
      populateForm();
      modal.hidden = false;
      if (testResult) testResult.hidden = true;
    });

    // Debug toggle auto-saves immediately (no need to click Save)
    if (debugEl) {
      debugEl.addEventListener("change", function () {
        var s = loadSettings();
        s.debug = debugEl.checked;
        saveSettings(s);
      });
    }

    // Close
    function closeModal() {
      modal.hidden = true;
    }
    if (closeBtn) closeBtn.addEventListener("click", closeModal);
    modal.addEventListener("click", function (e) {
      if (e.target === modal) closeModal();
    });

    // Save
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      saveSettings({
        provider: provEl.value,
        apiKey: keyEl.value,
        domain: domainEl.value,
        model: modelEl.value,
        debug: debugEl ? debugEl.checked : false,
      });
      if (!keyEl.value) {
        showToast("Settings saved — add an API key later to use AI features");
      } else {
        showToast("Settings saved");
      }
      closeModal();
    });

    // Test connection
    if (testBtn) {
      testBtn.addEventListener("click", function () {
        var settings = {
          provider: provEl.value,
          apiKey: keyEl.value,
          domain: domainEl.value,
          model: modelEl.value,
          debug: debugEl ? debugEl.checked : false,
        };
        // Temporarily save for the test
        var prev = loadSettings();
        saveSettings(settings);

        if (testResult) {
          testResult.hidden = false;
          testResult.textContent = "Testing…";
          testResult.className = "hint";
        }

        callLLM("Respond with exactly: {\"status\":\"ok\"}", "Say ok")
          .then(function (raw) {
            if (testResult) {
              testResult.textContent = "✓ Connection successful!";
              testResult.className = "hint success";
            }
          })
          .catch(function (err) {
            if (testResult) {
              testResult.textContent = "✗ Connection failed: " + (err.message || "unknown error");
              testResult.className = "hint error";
            }
          })
          .then(function () {
            // Restore previous settings
            saveSettings(prev);
          });
      });
    }
  }

  // --- Update the async job flow to add LLM extraction step ------------------

  // Override / wrap the handleResult function to add LLM extraction
  // when the backend returns transcription data.

  var _originalHandleResult = handleResult;

  function handleResult(resp) {
    if (resp.status === 200) {
      var data = resp.data;

      // Store raw transcription so the "Analyze with AI" button can use it
      if (data && data.transcription) {
        ui.lastTranscriptionData = data;
      }

      // Check if we have transcription data and LLM is configured — auto-run
      if (data && data.transcription && hasValidSettings()) {
        runLLMExtraction(data);
        return;
      }

      // No LLM configured — show raw transcription + Analyze button
      if (data && data.transcription) {
        var t = data.transcription;

        // Build basic chapters from Whisper segments (use segment text as title)
        var basicChapters = [];
        if (data.segments && data.segments.length > 0) {
          var segs = data.segments;
          for (var si = 0; si < segs.length; si++) {
            var segText = (segs[si].text || "").trim();
            // Truncate long segment text for the chapter display
            var title = segText.length > 100
              ? segText.substring(0, 97) + "…"
              : segText;
            basicChapters.push({
              start_seconds: segs[si].start,
              end_seconds: segs[si].end,
              title: title,
            });
          }
        }

        var transcriptText = t.transcript || "";
        ui.lastResult = {
          title: "Transcription complete",
          audio_url: "",
          knowledge: {
            summary: transcriptText,
            insights: [],
            chapters: basicChapters,
            narrative: "",
          },
        };
        renderResult(ui.lastResult);
        finishProgress();

        // Show the Analyze button and empty-state hints
        ui.analyzeButton.hidden = false;
        if (hasValidSettings()) {
          ui.analyzeButton.textContent = "✨ Analyze with AI";
          ui.analyzeButton.disabled = false;
          if (ui.summaryHint) {
            ui.summaryHint.hidden = false;
            ui.settingsHint.hidden = true;
            document.getElementById("summary-hint-text").textContent =
              'Click "✨ Analyze with AI" above to generate a structured summary.';
          }
        } else {
          ui.analyzeButton.textContent = "⚙️ Add API Key";
          ui.analyzeButton.disabled = false;
          if (ui.summaryHint) {
            ui.summaryHint.hidden = false;
            ui.settingsHint.hidden = false;
          }
        }
        if (ui.insightsHint) ui.insightsHint.hidden = false;
        // Hide chapters hint if we rendered basic segments as chapters
        if (ui.chaptersHint) {
          ui.chaptersHint.hidden = basicChapters.length > 0;
        }
        return;
      }

      // Full knowledge result (after LLM extraction) — hide button & hints
      if (data && data.knowledge) {
        if (data.knowledge.insights && data.knowledge.insights.length) {
          if (ui.insightsHint) ui.insightsHint.hidden = true;
        }
        if (data.knowledge.chapters && data.knowledge.chapters.length) {
          if (ui.chaptersHint) ui.chaptersHint.hidden = true;
        }
      }
      ui.analyzeButton.hidden = true;
      if (ui.summaryHint) ui.summaryHint.hidden = true;
      renderResult(data);
      finishProgress();
    } else {
      var msg = (resp.data && resp.data.error) || "Request failed.";
      var code = (resp.data && resp.data.code) || "INTERNAL_ERROR";
      showError(msg, code);
      ui.progressEl.hidden = true;
    }
  }

  // --- Init ---------------------------------------------------------

  function init() {
    cacheElements();
    initTabs();
    initForms();
    initCopyButton();
    initAnalyzeButton();
    initSettings();
    restoreJobFromHash();
  }

  // Expose pure helpers for testing.
  globalThis.Loomafmt = { formatTimestamp: formatTimestamp, buildMarkdown: buildMarkdown, escapeHtml: escapeHtml };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
