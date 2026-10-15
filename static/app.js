/* ═════════════════════════════════════════════════════════════════════
   CreditLens AI — Application Controller
   Real-Time Inference, Radial Gauge, Persona Presets & Diagnostics
   Author: Muhammed Sahal KK
   ═════════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    const API_URL = '/predict';
    const GAUGE_CIRCUMFERENCE = 251.3; // Half-circle radius 80: pi * 80 ≈ 251.3

    // ── DOM References ─────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => [...document.querySelectorAll(sel)];

    const els = {
        // Form inputs
        sliderLimitBal: $('#slider-limit-bal'),
        inputLimitBal:  $('#input-limit-bal'),
        limitBalDisplay:$('#limit-bal-display'),

        inputAge:       $('#input-age'),
        inputSex:       $('#input-sex'),
        inputEducation: $('#input-education'),
        inputMarriage:  $('#input-marriage'),

        inputPay0: $('#input-pay0'),
        inputPay2: $('#input-pay2'),
        inputPay3: $('#input-pay3'),
        inputPay4: $('#input-pay4'),
        inputPay5: $('#input-pay5'),
        inputPay6: $('#input-pay6'),

        inputBill1: $('#input-bill1'),
        inputBill2: $('#input-bill2'),
        inputBill3: $('#input-bill3'),
        inputBill4: $('#input-bill4'),
        inputBill5: $('#input-bill5'),
        inputBill6: $('#input-bill6'),

        inputPayAmt1: $('#input-payamt1'),
        inputPayAmt2: $('#input-payamt2'),
        inputPayAmt3: $('#input-payamt3'),
        inputPayAmt4: $('#input-payamt4'),
        inputPayAmt5: $('#input-payamt5'),
        inputPayAmt6: $('#input-payamt6'),

        syncMonths: $('#sync-months'),
        runBtn:     $('#run-btn'),

        // Diagnostic Outputs
        apiStatus:      $('#api-status'),
        badge:          $('#output-badge'),
        probVal:        $('#output-prob-value'),
        probPct:        $('#output-prob-pct'),
        statusText:     $('#output-status'),
        statusCard:     $('#output-status-card'),
        statusIcon:     $('#status-icon'),
        gaugeArc:       $('#gauge-arc-fill'),
        diagRepay:      $('#diag-repay'),
        diagUtil:       $('#diag-util'),

        // Persona cards
        personaCards:   $$('.persona-card'),
    };

    // ── Preset Profiles from Real UCI Test Rows ────────────────────
    const PRESETS = {
        low: {
            LIMIT_BAL: 500000, SEX: 1, EDUCATION: 1, MARRIAGE: 2, AGE: 38,
            PAY_0: -1, PAY_2: -1, PAY_3: -1, PAY_4: -1, PAY_5: -1, PAY_6: -1,
            BILL_AMT1: 15000, BILL_AMT2: 14200, BILL_AMT3: 13800,
            BILL_AMT4: 12500, BILL_AMT5: 11000, BILL_AMT6: 10500,
            PAY_AMT1: 15000, PAY_AMT2: 14200, PAY_AMT3: 13800,
            PAY_AMT4: 12500, PAY_AMT5: 11000, PAY_AMT6: 10500,
        },
        mid: {
            LIMIT_BAL: 120000, SEX: 2, EDUCATION: 2, MARRIAGE: 2, AGE: 26,
            PAY_0: 0, PAY_2: 0, PAY_3: 0, PAY_4: 0, PAY_5: 0, PAY_6: 0,
            BILL_AMT1: 46990, BILL_AMT2: 48233, BILL_AMT3: 49291,
            BILL_AMT4: 28314, BILL_AMT5: 28959, BILL_AMT6: 29547,
            PAY_AMT1: 2000, PAY_AMT2: 2019, PAY_AMT3: 1200,
            PAY_AMT4: 1100, PAY_AMT5: 1069, PAY_AMT6: 1000,
        },
        high: {
            LIMIT_BAL: 20000, SEX: 2, EDUCATION: 2, MARRIAGE: 1, AGE: 24,
            PAY_0: 2, PAY_2: 2, PAY_3: 2, PAY_4: 2, PAY_5: 2, PAY_6: 2,
            BILL_AMT1: 19000, BILL_AMT2: 19500, BILL_AMT3: 20000,
            BILL_AMT4: 19000, BILL_AMT5: 18500, BILL_AMT6: 18000,
            PAY_AMT1: 0, PAY_AMT2: 0, PAY_AMT3: 0,
            PAY_AMT4: 0, PAY_AMT5: 0, PAY_AMT6: 0,
        },
    };

    // ── Feature Mapping ────────────────────────────────────────────
    const FIELD_MAP = {
        LIMIT_BAL:  els.inputLimitBal,
        SEX:        els.inputSex,
        EDUCATION:  els.inputEducation,
        MARRIAGE:   els.inputMarriage,
        AGE:        els.inputAge,
        PAY_0:      els.inputPay0,
        PAY_2:      els.inputPay2,
        PAY_3:      els.inputPay3,
        PAY_4:      els.inputPay4,
        PAY_5:      els.inputPay5,
        PAY_6:      els.inputPay6,
        BILL_AMT1:  els.inputBill1,
        BILL_AMT2:  els.inputBill2,
        BILL_AMT3:  els.inputBill3,
        BILL_AMT4:  els.inputBill4,
        BILL_AMT5:  els.inputBill5,
        BILL_AMT6:  els.inputBill6,
        PAY_AMT1:   els.inputPayAmt1,
        PAY_AMT2:   els.inputPayAmt2,
        PAY_AMT3:   els.inputPayAmt3,
        PAY_AMT4:   els.inputPayAmt4,
        PAY_AMT5:   els.inputPayAmt5,
        PAY_AMT6:   els.inputPayAmt6,
    };

    const PAY_FIELDS = [
        els.inputPay0, els.inputPay2, els.inputPay3,
        els.inputPay4, els.inputPay5, els.inputPay6
    ];

    // Format currency
    function formatCurrency(num) {
        return 'NT$ ' + Number(num).toLocaleString('en-US');
    }

    // ── Collect Features ───────────────────────────────────────────
    function collectFeatures() {
        const payload = {};
        for (const [key, el] of Object.entries(FIELD_MAP)) {
            payload[key] = parseFloat(el.value) || 0;
        }
        return payload;
    }

    // ── Apply Persona Preset ───────────────────────────────────────
    function applyPreset(name) {
        const data = PRESETS[name];
        if (!data) return;

        // Fill all form fields
        for (const [key, val] of Object.entries(data)) {
            if (FIELD_MAP[key]) {
                FIELD_MAP[key].value = val;
            }
        }

        // Sync slider & display
        els.sliderLimitBal.value = data.LIMIT_BAL;
        els.limitBalDisplay.textContent = formatCurrency(data.LIMIT_BAL);

        // Active persona card styling
        els.personaCards.forEach(card => {
            if (card.dataset.preset === name) {
                card.classList.add('active');
            } else {
                card.classList.remove('active');
            }
        });

        // Trigger immediate live prediction
        requestPrediction();
    }

    // ── Debounced Prediction ───────────────────────────────────────
    let debounceTimer = null;
    function debouncedPredict() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(requestPrediction, 250);
    }

    // ── Request Prediction ─────────────────────────────────────────
    let isLoading = false;

    async function requestPrediction() {
        if (isLoading) return;
        isLoading = true;
        if (els.runBtn) els.runBtn.classList.add('btn-primary--loading');

        try {
            const features = collectFeatures();
            const res = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(features),
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const data = await res.json();
            renderOutput(data, features);
            setApiOnline(true);
        } catch (err) {
            console.error('Inference error:', err);
            if (els.statusText) {
                els.statusText.textContent = `Backend error: ${err.message}. Ensure FastAPI is running.`;
            }
            setApiOnline(false);
        } finally {
            isLoading = false;
            if (els.runBtn) els.runBtn.classList.remove('btn-primary--loading');
        }
    }

    // ── Render Output ──────────────────────────────────────────────
    function renderOutput(data, features) {
        const prob = data.default_probability;
        const pct = prob * 100;
        const isDefault = data.prediction === 1;

        // Numerical Probability Display
        if (els.probVal) els.probVal.textContent = pct.toFixed(2) + '%';
        if (els.probPct) els.probPct.textContent = `Raw Sigmoid Activation: ${prob.toFixed(6)}`;

        // SVG Radial Gauge Arc Calculation
        if (els.gaugeArc) {
            const offset = GAUGE_CIRCUMFERENCE * (1 - Math.min(Math.max(prob, 0), 1));
            els.gaugeArc.style.strokeDashoffset = offset.toFixed(1);
        }

        // Decision Badge
        if (els.badge) {
            if (isDefault) {
                els.badge.textContent = 'HIGH RISK';
                els.badge.className = 'decision-badge decision-badge--danger';
            } else {
                els.badge.textContent = 'SAFE / LOW RISK';
                els.badge.className = 'decision-badge decision-badge--safe';
            }
        }

        // Diagnosis Banner & Icon
        if (els.statusText) els.statusText.textContent = data.status;
        if (els.statusIcon) {
            if (pct >= 50) {
                els.statusIcon.textContent = '🚨';
                els.statusCard.style.borderColor = 'rgba(239, 68, 68, 0.4)';
            } else if (pct >= 30) {
                els.statusIcon.textContent = '⚠️';
                els.statusCard.style.borderColor = 'rgba(245, 158, 11, 0.4)';
            } else {
                els.statusIcon.textContent = '✅';
                els.statusCard.style.borderColor = 'rgba(16, 185, 129, 0.4)';
            }
        }

        // Diagnostics Breakdown
        if (features) {
            // Utilization
            const maxBill = Math.max(
                features.BILL_AMT1, features.BILL_AMT2, features.BILL_AMT3,
                features.BILL_AMT4, features.BILL_AMT5, features.BILL_AMT6
            );
            const utilRatio = (maxBill / (features.LIMIT_BAL || 1)) * 100;
            if (els.diagUtil) {
                els.diagUtil.textContent = `${utilRatio.toFixed(1)}% of Limit`;
                if (utilRatio > 85) {
                    els.diagUtil.className = 'metric-val metric-val--bad';
                } else if (utilRatio > 50) {
                    els.diagUtil.className = 'metric-val metric-val--warning';
                } else {
                    els.diagUtil.className = 'metric-val metric-val--good';
                }
            }

            // Max delay across 6 months
            const maxDelay = Math.max(
                features.PAY_0, features.PAY_2, features.PAY_3,
                features.PAY_4, features.PAY_5, features.PAY_6
            );
            if (els.diagRepay) {
                if (maxDelay > 1) {
                    els.diagRepay.textContent = `${maxDelay} Months Overdue`;
                    els.diagRepay.className = 'metric-val metric-val--bad';
                } else if (maxDelay === 1) {
                    els.diagRepay.textContent = '1 Month Delay';
                    els.diagRepay.className = 'metric-val metric-val--warning';
                } else if (maxDelay === 0) {
                    els.diagRepay.textContent = 'Revolving Credit Active';
                    els.diagRepay.className = 'metric-val metric-val--warning';
                } else {
                    els.diagRepay.textContent = 'Paid Duly / On-Time';
                    els.diagRepay.className = 'metric-val metric-val--good';
                }
            }
        }
    }

    // ── API Status Check ───────────────────────────────────────────
    function setApiOnline(online) {
        if (!els.apiStatus) return;
        const pulse = els.apiStatus.querySelector('.status-pulse');
        const text = els.apiStatus.querySelector('.status-text');

        if (pulse) {
            pulse.className = 'status-pulse ' + (online ? 'status-pulse--online' : 'status-pulse--offline');
        }
        if (text) {
            text.textContent = online ? 'Live Model Connected' : 'Model Offline';
        }
    }

    async function checkHealth() {
        try {
            const res = await fetch('/');
            setApiOnline(res.ok);
        } catch {
            setApiOnline(false);
        }
    }

    // ── Event Handlers ─────────────────────────────────────────────
    function initEvents() {
        // Range slider & limit number input sync
        if (els.sliderLimitBal && els.inputLimitBal) {
            els.sliderLimitBal.addEventListener('input', () => {
                els.inputLimitBal.value = els.sliderLimitBal.value;
                if (els.limitBalDisplay) {
                    els.limitBalDisplay.textContent = formatCurrency(els.sliderLimitBal.value);
                }
                debouncedPredict();
            });

            els.inputLimitBal.addEventListener('input', () => {
                els.sliderLimitBal.value = els.inputLimitBal.value;
                if (els.limitBalDisplay) {
                    els.limitBalDisplay.textContent = formatCurrency(els.inputLimitBal.value);
                }
                debouncedPredict();
            });
        }

        // Persona card clicks
        els.personaCards.forEach(card => {
            card.addEventListener('click', () => {
                const presetKey = card.dataset.preset;
                applyPreset(presetKey);
            });
            card.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    card.click();
                }
            });
        });

        // Sync months toggle
        if (els.syncMonths && els.inputPay0) {
            els.inputPay0.addEventListener('change', () => {
                if (els.syncMonths.checked) {
                    const val = els.inputPay0.value;
                    PAY_FIELDS.slice(1).forEach(sel => sel.value = val);
                }
                debouncedPredict();
            });

            els.syncMonths.addEventListener('change', () => {
                if (els.syncMonths.checked) {
                    const val = els.inputPay0.value;
                    PAY_FIELDS.slice(1).forEach(sel => sel.value = val);
                    debouncedPredict();
                }
            });
        }

        // All form inputs debounce trigger
        const allInputs = $$('.input-control, .select-control, .table-num-input');
        allInputs.forEach(input => {
            if (input === els.inputLimitBal) return;
            input.addEventListener('input', debouncedPredict);
            input.addEventListener('change', debouncedPredict);
        });

        // Run button click
        if (els.runBtn) {
            els.runBtn.addEventListener('click', (e) => {
                e.preventDefault();
                requestPrediction();
            });
        }
    }

    // ── Initialize on Page Load ────────────────────────────────────
    function init() {
        initEvents();
        checkHealth();
        // Start with the high-risk preset loaded as demonstration
        applyPreset('high');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
