/**
 * Jovi Voice Presenter — Frontend Controller
 * Handles slide navigation, TTS audio playback, speaking animation, and live Q&A.
 */
(function () {
    "use strict";

    // ─── DOM References ──────────────────────────────────────────
    const slideLabel    = document.getElementById("slideLabel");
    const slideTitle    = document.getElementById("slideTitle");
    const slideVisual   = document.getElementById("slideVisual");
    const narrationBox  = document.getElementById("narrationBox");
    const avatarRing    = document.getElementById("avatarRing");
    const btnPrev       = document.getElementById("btnPrev");
    const btnNext       = document.getElementById("btnNext");
    const qaForm        = document.getElementById("qaForm");
    const qaInput       = document.getElementById("qaInput");
    const qaHistory     = document.getElementById("qaHistory");
    const audioPlayer   = document.getElementById("audioPlayer");

    const TOTAL_SLIDES = 4;
    let currentSlide = 0; // 0 = not started yet
    let isBusy = false;

    // ─── Slide Visuals (inline SVG placeholders) ─────────────────
    const slideVisuals = {
        1: `<div class="slide-svg-container">
                <svg width="420" height="300" viewBox="0 0 420 300" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="10" y="10" width="400" height="280" rx="16" stroke="#00d4aa" stroke-width="1.5" stroke-dasharray="6 4" fill="#1a1a26"/>
                    <text x="210" y="60" text-anchor="middle" fill="#e8e8f0" font-size="18" font-weight="700" font-family="Inter">◈  J O V I</text>
                    <text x="210" y="85" text-anchor="middle" fill="#9090a8" font-size="12" font-family="Inter">GeoAI Engine · BuildSight v1.1</text>
                    <line x1="60" y1="110" x2="360" y2="110" stroke="#2a2a3d" stroke-width="1"/>
                    <rect x="80" y="135" width="100" height="50" rx="8" fill="#222233" stroke="#00d4aa" stroke-width="1"/>
                    <text x="130" y="165" text-anchor="middle" fill="#00d4aa" font-size="11" font-family="JetBrains Mono">YOLOv11</text>
                    <rect x="240" y="135" width="100" height="50" rx="8" fill="#222233" stroke="#00d4aa" stroke-width="1"/>
                    <text x="290" y="165" text-anchor="middle" fill="#00d4aa" font-size="11" font-family="JetBrains Mono">AdaFace</text>
                    <rect x="160" y="215" width="100" height="50" rx="8" fill="#222233" stroke="#00d4aa" stroke-width="1"/>
                    <text x="210" y="245" text-anchor="middle" fill="#00d4aa" font-size="11" font-family="JetBrains Mono">QGIS</text>
                    <line x1="130" y1="185" x2="210" y2="215" stroke="#2a2a3d" stroke-width="1.5"/>
                    <line x1="290" y1="185" x2="210" y2="215" stroke="#2a2a3d" stroke-width="1.5"/>
                </svg>
            </div>`,
        2: `<div class="slide-svg-container">
                <svg width="420" height="300" viewBox="0 0 420 300" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="10" y="10" width="400" height="280" rx="16" stroke="#2a2a3d" stroke-width="1" fill="#1a1a26"/>
                    <text x="210" y="40" text-anchor="middle" fill="#9090a8" font-size="11" font-family="JetBrains Mono" letter-spacing="2">ARCHITECTURE PIPELINE</text>
                    <rect x="30" y="60" width="120" height="40" rx="6" fill="#222233" stroke="#00d4aa" stroke-width="1"/>
                    <text x="90" y="85" text-anchor="middle" fill="#e8e8f0" font-size="11" font-family="Inter">📷 Camera Feed</text>
                    <path d="M150 80 L170 80" stroke="#00d4aa" stroke-width="1.5" marker-end="url(#arrow)"/>
                    <rect x="175" y="60" width="120" height="40" rx="6" fill="#222233" stroke="#ffaa33" stroke-width="1"/>
                    <text x="235" y="85" text-anchor="middle" fill="#e8e8f0" font-size="11" font-family="Inter">🔍 YOLOv11</text>
                    <path d="M295 80 L315 80" stroke="#ffaa33" stroke-width="1.5"/>
                    <rect x="320" y="60" width="80" height="40" rx="6" fill="#222233" stroke="#ff4466" stroke-width="1"/>
                    <text x="360" y="85" text-anchor="middle" fill="#e8e8f0" font-size="10" font-family="Inter">PPE Tags</text>
                    <rect x="30" y="130" width="120" height="40" rx="6" fill="#222233" stroke="#00d4aa" stroke-width="1"/>
                    <text x="90" y="155" text-anchor="middle" fill="#e8e8f0" font-size="11" font-family="Inter">👤 Face Crop</text>
                    <path d="M150 150 L170 150" stroke="#00d4aa" stroke-width="1.5"/>
                    <rect x="175" y="130" width="120" height="40" rx="6" fill="#222233" stroke="#ffaa33" stroke-width="1"/>
                    <text x="235" y="155" text-anchor="middle" fill="#e8e8f0" font-size="11" font-family="Inter">🧬 AdaFace</text>
                    <path d="M295 150 L315 150" stroke="#ffaa33" stroke-width="1.5"/>
                    <rect x="320" y="130" width="80" height="40" rx="6" fill="#222233" stroke="#ff4466" stroke-width="1"/>
                    <text x="360" y="155" text-anchor="middle" fill="#e8e8f0" font-size="10" font-family="Inter">Worker ID</text>
                    <rect x="130" y="210" width="160" height="50" rx="10" fill="rgba(0,212,170,0.08)" stroke="#00d4aa" stroke-width="1.5"/>
                    <text x="210" y="240" text-anchor="middle" fill="#00d4aa" font-size="13" font-weight="600" font-family="Inter">🌍 QGIS Heatmap</text>
                    <line x1="235" y1="100" x2="210" y2="210" stroke="#2a2a3d" stroke-width="1" stroke-dasharray="4 3"/>
                    <line x1="235" y1="170" x2="210" y2="210" stroke="#2a2a3d" stroke-width="1" stroke-dasharray="4 3"/>
                </svg>
            </div>`,
        3: `<div class="slide-svg-container">
                <svg width="420" height="300" viewBox="0 0 420 300" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="10" y="10" width="400" height="280" rx="16" stroke="#2a2a3d" stroke-width="1" fill="#1a1a26"/>
                    <text x="210" y="40" text-anchor="middle" fill="#9090a8" font-size="11" font-family="JetBrains Mono" letter-spacing="2">SITE RISK HEATMAP</text>
                    <rect x="40" y="60" width="80" height="80" rx="6" fill="rgba(0,212,170,0.15)" stroke="#00d4aa" stroke-width="1"/>
                    <text x="80" y="105" text-anchor="middle" fill="#00d4aa" font-size="11" font-family="Inter" font-weight="600">Zone A</text>
                    <text x="80" y="120" text-anchor="middle" fill="#9090a8" font-size="9" font-family="JetBrains Mono">LOW RISK</text>
                    <rect x="140" y="60" width="80" height="80" rx="6" fill="rgba(255,170,51,0.2)" stroke="#ffaa33" stroke-width="1"/>
                    <text x="180" y="105" text-anchor="middle" fill="#ffaa33" font-size="11" font-family="Inter" font-weight="600">Zone B</text>
                    <text x="180" y="120" text-anchor="middle" fill="#9090a8" font-size="9" font-family="JetBrains Mono">MED RISK</text>
                    <rect x="240" y="60" width="80" height="80" rx="6" fill="rgba(255,68,102,0.2)" stroke="#ff4466" stroke-width="1.5"/>
                    <text x="280" y="105" text-anchor="middle" fill="#ff4466" font-size="11" font-family="Inter" font-weight="600">Zone C</text>
                    <text x="280" y="120" text-anchor="middle" fill="#9090a8" font-size="9" font-family="JetBrains Mono">HIGH RISK</text>
                    <rect x="340" y="60" width="52" height="80" rx="6" fill="rgba(255,68,102,0.35)" stroke="#ff4466" stroke-width="2"/>
                    <text x="366" y="105" text-anchor="middle" fill="#ff4466" font-size="10" font-family="Inter" font-weight="700">C+</text>
                    <text x="366" y="120" text-anchor="middle" fill="#ff4466" font-size="8" font-family="JetBrains Mono">CRITICAL</text>
                    <rect x="40" y="165" width="350" height="30" rx="4" fill="#222233"/>
                    <rect x="40" y="165" width="120" height="30" rx="4" fill="rgba(0,212,170,0.2)"/>
                    <rect x="160" y="165" width="90" height="30" rx="4" fill="rgba(255,170,51,0.25)"/>
                    <rect x="250" y="165" width="140" height="30" rx="4" fill="rgba(255,68,102,0.25)"/>
                    <text x="100" y="185" text-anchor="middle" fill="#00d4aa" font-size="10" font-family="JetBrains Mono">32%</text>
                    <text x="205" y="185" text-anchor="middle" fill="#ffaa33" font-size="10" font-family="JetBrains Mono">24%</text>
                    <text x="320" y="185" text-anchor="middle" fill="#ff4466" font-size="10" font-family="JetBrains Mono">67% violations</text>
                    <text x="210" y="230" text-anchor="middle" fill="#5c5c72" font-size="10" font-family="Inter">PPE Violations by Zone · 9 AM – 11 AM Shift</text>
                    <text x="210" y="270" text-anchor="middle" fill="#9090a8" font-size="10" font-family="Inter">Zone C (Scaffolding Area) flagged for immediate intervention</text>
                </svg>
            </div>`,
        4: `<div class="slide-svg-container">
                <svg width="420" height="300" viewBox="0 0 420 300" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="10" y="10" width="400" height="280" rx="16" stroke="#2a2a3d" stroke-width="1" fill="#1a1a26"/>
                    <text x="210" y="40" text-anchor="middle" fill="#9090a8" font-size="11" font-family="JetBrains Mono" letter-spacing="2">IS-CODE COMPLIANCE REPORT</text>
                    <rect x="30" y="60" width="360" height="36" rx="6" fill="#222233"/>
                    <text x="45" y="83" fill="#00d4aa" font-size="11" font-family="JetBrains Mono">✓ IS 2925</text>
                    <text x="150" y="83" fill="#e8e8f0" font-size="11" font-family="Inter">Industrial Safety Helmets</text>
                    <text x="370" y="83" text-anchor="end" fill="#00d4aa" font-size="11" font-family="JetBrains Mono">94%</text>
                    <rect x="30" y="104" width="360" height="36" rx="6" fill="#222233"/>
                    <text x="45" y="127" fill="#ffaa33" font-size="11" font-family="JetBrains Mono">⚠ IS 3696</text>
                    <text x="150" y="127" fill="#e8e8f0" font-size="11" font-family="Inter">Scaffolds & Ladders</text>
                    <text x="370" y="127" text-anchor="end" fill="#ffaa33" font-size="11" font-family="JetBrains Mono">78%</text>
                    <rect x="30" y="148" width="360" height="36" rx="6" fill="#222233"/>
                    <text x="45" y="171" fill="#00d4aa" font-size="11" font-family="JetBrains Mono">✓ IS 3764</text>
                    <text x="150" y="171" fill="#e8e8f0" font-size="11" font-family="Inter">Excavation Work Safety</text>
                    <text x="370" y="171" text-anchor="end" fill="#00d4aa" font-size="11" font-family="JetBrains Mono">91%</text>
                    <rect x="30" y="192" width="360" height="36" rx="6" fill="#222233"/>
                    <text x="45" y="215" fill="#ff4466" font-size="11" font-family="JetBrains Mono">✗ BOCW 96</text>
                    <text x="150" y="215" fill="#e8e8f0" font-size="11" font-family="Inter">Worker Protection Act</text>
                    <text x="370" y="215" text-anchor="end" fill="#ff4466" font-size="11" font-family="JetBrains Mono">62%</text>
                    <rect x="30" y="246" width="360" height="36" rx="6" fill="rgba(0,212,170,0.06)" stroke="#00d4aa" stroke-width="1"/>
                    <text x="210" y="269" text-anchor="middle" fill="#00d4aa" font-size="12" font-family="Inter" font-weight="600">Overall Compliance Score: 81.25%</text>
                </svg>
            </div>`
    };

    // ─── Helpers ─────────────────────────────────────────────────
    function setLoading(state) {
        isBusy = state;
        btnNext.disabled = state;
        btnPrev.disabled = state || currentSlide <= 1;
    }

    function setSpeaking(state) {
        if (state) {
            avatarRing.classList.add("speaking");
        } else {
            avatarRing.classList.remove("speaking");
        }
    }

    function updateSlideUI(slideNum, title) {
        slideLabel.textContent = `SLIDE ${slideNum} OF ${TOTAL_SLIDES}`;
        slideTitle.textContent = title;
        slideVisual.innerHTML = slideVisuals[slideNum] || "";
    }

    function setNarration(text) {
        narrationBox.innerHTML = `<p class="narration-text narration-active">${text}</p>`;
    }

    function playAudio(url) {
        return new Promise((resolve) => {
            audioPlayer.src = url + "?t=" + Date.now(); // cache-bust
            audioPlayer.onended = () => {
                setSpeaking(false);
                resolve();
            };
            audioPlayer.onerror = () => {
                setSpeaking(false);
                resolve();
            };
            setSpeaking(true);
            audioPlayer.play().catch(() => {
                setSpeaking(false);
                resolve();
            });
        });
    }

    function addQABubble(text, type) {
        const div = document.createElement("div");
        div.className = `qa-bubble ${type}`;
        div.textContent = text;
        qaHistory.appendChild(div);
        qaHistory.scrollTop = qaHistory.scrollHeight;
    }

    function addLoadingBubble() {
        const div = document.createElement("div");
        div.className = "qa-bubble answer";
        div.id = "loadingBubble";
        div.innerHTML = `<div class="loading-dots"><span></span><span></span><span></span></div>`;
        qaHistory.appendChild(div);
        qaHistory.scrollTop = qaHistory.scrollHeight;
    }

    function removeLoadingBubble() {
        const el = document.getElementById("loadingBubble");
        if (el) el.remove();
    }

    // ─── Slide Navigation ────────────────────────────────────────
    async function goToSlide(slideNum) {
        if (slideNum < 1 || slideNum > TOTAL_SLIDES || isBusy) return;
        setLoading(true);
        currentSlide = slideNum;

        // Update button labels
        btnNext.textContent = slideNum >= TOTAL_SLIDES ? "Presentation Complete ✓" : "Next Slide →";
        if (slideNum >= TOTAL_SLIDES) btnNext.disabled = true;

        // Fetch narration + audio
        try {
            const res = await fetch(`/api/narrate/${slideNum}`);
            const data = await res.json();
            updateSlideUI(data.slide_id, data.title);
            setNarration(data.narration);
            await playAudio(data.audio_url);
        } catch (err) {
            setNarration("Connection issue. Please retry.");
        }
        setLoading(false);
    }

    // ─── Event Listeners ─────────────────────────────────────────
    btnNext.addEventListener("click", () => {
        if (currentSlide === 0) {
            // First click starts the presentation
            btnNext.textContent = "Next Slide →";
            goToSlide(1);
        } else if (currentSlide < TOTAL_SLIDES) {
            goToSlide(currentSlide + 1);
        }
    });

    btnPrev.addEventListener("click", () => {
        if (currentSlide > 1) goToSlide(currentSlide - 1);
    });

    // ─── Q&A Form ────────────────────────────────────────────────
    qaForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const question = qaInput.value.trim();
        if (!question || isBusy) return;

        qaInput.value = "";
        addQABubble(question, "question");
        addLoadingBubble();
        setLoading(true);

        try {
            const res = await fetch("/api/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });
            const data = await res.json();
            removeLoadingBubble();
            addQABubble(data.answer, "answer");
            await playAudio(data.audio_url);
        } catch (err) {
            removeLoadingBubble();
            addQABubble("Connection error. Please try again.", "answer");
        }
        setLoading(false);
    });
})();
