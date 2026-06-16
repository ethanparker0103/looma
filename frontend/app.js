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
    submitButtons: [],
    formYoutube: null,
    formUpload: null,
    lastResult: null,
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
  function runAsyncJob(body, isMultipart, inputId) {
    ui.resultsEl.hidden = true;
    clearError();
    setBusy(true);
    startProgress();

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

  // --- Init --------------------------------------------------------

  function init() {
    cacheElements();
    initTabs();
    initForms();
    initCopyButton();
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
