// API Base URL
const API_URL = 'http://127.0.0.1:8000';

// Global Chart layout configuration (dark theme)
const plotLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#c9d1d9', family: 'Inter' },
    margin: { t: 40, r: 20, b: 40, l: 50 },
    xaxis: { gridcolor: '#30363d', zerolinecolor: '#484f58' },
    yaxis: { gridcolor: '#30363d', zerolinecolor: '#484f58' }
};

// ==========================================
// Initialization
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
    setupNavigation();
    checkApiHealth();
    setupPricingLab();
    setupGreeksExplorer();
    setupRiskEngine();
});

// ==========================================
// Navigation & UI Helpers
// ==========================================
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-links a');
    const sections = document.querySelectorAll('.page');
    const navBtns = document.querySelectorAll('.nav-btn');

    function switchPage(pageId) {
        sections.forEach(s => s.classList.remove('active'));
        navLinks.forEach(l => l.classList.remove('active'));
        
        document.getElementById(`page-${pageId}`).classList.add('active');
        const link = document.querySelector(`.nav-links a[data-page="${pageId}"]`);
        if(link) link.classList.add('active');
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            switchPage(e.target.dataset.page);
        });
    });

    navBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            switchPage(e.target.dataset.target);
        });
    });
}

function showSpinner() { document.getElementById('global-spinner').style.display = 'flex'; }
function hideSpinner() { document.getElementById('global-spinner').style.display = 'none'; }

async function checkApiHealth() {
    const ind = document.querySelector('.status-indicator');
    const text = document.getElementById('api-status');
    try {
        const res = await fetch(`${API_URL}/`);
        if (res.ok) {
            ind.classList.remove('offline');
            ind.classList.add('online');
            text.innerText = "Connected to API";
        }
    } catch (e) {
        ind.classList.remove('online');
        ind.classList.add('offline');
        text.innerText = "API Offline";
    }
}

// ==========================================
// Pricing Lab
// ==========================================
function setupPricingLab() {
    const modelSelect = document.getElementById('price-model');
    const styleGroup = document.getElementById('binomial-style-group');
    
    // Toggle style select for Binomial
    modelSelect.addEventListener('change', (e) => {
        styleGroup.style.display = e.target.value === 'binomial' ? 'block' : 'none';
        document.getElementById('res-mc-extra').style.display = e.target.value === 'monte-carlo' ? 'block' : 'none';
    });

    document.getElementById('pricing-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Build payload
        const payload = {
            S: parseFloat(document.getElementById('price-s').value),
            K: parseFloat(document.getElementById('price-k').value),
            T: parseFloat(document.getElementById('price-t').value),
            sigma: parseFloat(document.getElementById('price-v').value),
            r: parseFloat(document.getElementById('price-r').value),
            option_type: document.getElementById('price-type').value
        };

        const model = document.getElementById('price-model').value;
        let endpoint = '/price/black-scholes';

        if (model === 'monte-carlo') {
            endpoint = '/price/monte-carlo';
            payload.n_sims = 10000;
        } else if (model === 'binomial') {
            endpoint = '/price/binomial';
            payload.style = document.getElementById('price-style').value;
            payload.N = 200;
        }

        showSpinner();
        try {
            const res = await fetch(`${API_URL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (res.ok) {
                // Update UI
                document.getElementById('res-price-value').innerText = `$${data.price.toFixed(4)}`;
                document.getElementById('res-price-time').innerText = `Computed in ${data.elapsed_ms.toFixed(1)}ms`;
                
                if(model === 'monte-carlo' && data.confidence_interval) {
                     document.getElementById('res-mc-ci').innerText = `[${data.confidence_interval.lower.toFixed(4)}, ${data.confidence_interval.upper.toFixed(4)}]`;
                     document.getElementById('res-mc-extra').style.display = 'block';
                }

                // Render simple payoff line for current params
                renderPayoffChart(payload.S, payload.K, payload.option_type, data.price);
            } else {
                alert(`Error: ${data.detail || data.error}`);
            }
        } catch(e) {
            alert('Failed to connect to API');
        }
        hideSpinner();
    });
}

function renderPayoffChart(S, K, type, premium) {
    const spots = Array.from({length: 100}, (_, i) => K * 0.5 + (K * 1.0)*i/100);
    const payoffs = spots.map(s => {
        let val = type === 'call' ? Math.max(s - K, 0) : Math.max(K - s, 0);
        return val - premium; // profit
    });

    const trace = {
        x: spots,
        y: payoffs,
        type: 'scatter',
        mode: 'lines',
        line: {color: '#58a6ff', width: 2},
        fill: 'tozeroy',
        fillcolor: 'rgba(88, 166, 255, 0.2)'
    };
    
    // Line at zero
    const zeroLine = {
        x: [spots[0], spots[spots.length-1]], y: [0,0],
        type: 'scatter', mode: 'lines', line: {color: '#8b949e', dash: 'dash'}
    };

    const layout = { ...plotLayout, title: 'Option P&L at Expiry', xaxis: {title:'Spot Price'}, yaxis: {title:'Profit / Loss'} };
    Plotly.newPlot('pricing-chart', [trace, zeroLine], layout, {responsive: true});
}

// ==========================================
// Greeks Explorer
// ==========================================
function setupGreeksExplorer() {
    document.getElementById('greeks-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const payload = {
            S: parseFloat(document.getElementById('gr-s').value),
            K: parseFloat(document.getElementById('gr-k').value),
            T: parseFloat(document.getElementById('gr-t').value),
            sigma: parseFloat(document.getElementById('gr-v').value),
            r: parseFloat(document.getElementById('gr-r').value),
            option_type: document.getElementById('greeks-type').value,
            method: 'analytical'
        };

        showSpinner();
        try {
            const res = await fetch(`${API_URL}/greeks/calculate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (res.ok) {
                const g = data.greeks;
                document.getElementById('gr-res-delta').innerText = g.delta.toFixed(4);
                document.getElementById('gr-res-gamma').innerText = g.gamma.toFixed(5);
                document.getElementById('gr-res-vega').innerText = g.vega.toFixed(4);
                document.getElementById('gr-res-theta').innerText = g.theta.toFixed(4);

                // Simulation: Calculate delta across spot prices
                renderDeltaChart(payload);
            }
        } catch(e) { console.error(e); }
        hideSpinner();
    });
}

async function renderDeltaChart(basePayload) {
    // We will generate a crude chart client side or simulate calls.
    // For simplicity, we just show a static placeholder or call BS manually if needed.
    // Ideally, the API would have a surface endpoint. We'll do an approximation for visual flair.
    
    // Quick approx for Delta array
    const spots = Array.from({length: 50}, (_, i) => basePayload.K * 0.5 + (basePayload.K * 1.0)*i/50);
    
    // Helper standard normal CDF
    const cdf = (x) => {
        let t = 1 / (1 + 0.2316419 * Math.abs(x));
        let d = 0.3989423 * Math.exp(-x * x / 2);
        let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return x > 0 ? 1 - p : p;
    };

    const deltas = spots.map(s => {
        let d1 = (Math.log(s/basePayload.K) + (basePayload.r + 0.5 * Math.pow(basePayload.sigma, 2)) * basePayload.T) / (basePayload.sigma * Math.sqrt(basePayload.T));
        return basePayload.option_type === 'call' ? cdf(d1) : cdf(d1) - 1;
    });

    const trace = {
        x: spots, y: deltas, type: 'scatter', mode: 'lines',
        line: {color: '#f85149', width: 3}
    };
    const layout = { ...plotLayout, title: 'Delta Profile vs Spot', xaxis: {title:'Spot'}, yaxis: {title:'Delta'} };
    Plotly.newPlot('greeks-chart', [trace], layout, {responsive: true});
}

// ==========================================
// Risk Engine
// ==========================================
let globalPortfolio = [];

function setupRiskEngine() {
    renderPortfolio();

    // Add position
    document.getElementById('btn-add-pos').addEventListener('click', () => {
        const type = document.getElementById('pos-type').value;
        const qtyStr = document.getElementById('pos-qty').value;
        const kStr = document.getElementById('pos-k').value;
        
        const qty = qtyStr === "" ? 10 : parseInt(qtyStr);
        const k = kStr === "" ? 100 : parseFloat(kStr);
        
        globalPortfolio.push({
            type: type, S: 100, K: k, T: 0.5, r: 0.05, sigma: 0.2, qty: qty, asset: 'default'
        });
        renderPortfolio();
    });

    // Run risk
    document.getElementById('risk-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if(globalPortfolio.length === 0) {
            alert('Add positions to portfolio first.'); return;
        }

        const payload = {
            portfolio: globalPortfolio,
            method: document.getElementById('risk-method').value,
            confidence: parseFloat(document.getElementById('risk-conf').value),
            n_sims: parseInt(document.getElementById('risk-sims').value),
            horizon_days: 1, seed: 42
        };

        showSpinner();
        try {
            const res = await fetch(`${API_URL}/risk/portfolio`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (res.ok) {
                document.getElementById('res-var').innerText = `$${data.VaR.toLocaleString()}`;
                document.getElementById('res-cvar').innerText = `$${data.CVaR.toLocaleString()}`;
                document.getElementById('res-pnl').innerText = `$${data.pnl_statistics.mean.toLocaleString()}`;
                
                renderRiskChart(data.VaR, data.CVaR, data.pnl_statistics.mean, data.pnl_statistics.std, payload.confidence);
            } else {
                alert(`Error: ${data.detail || data.error}`);
            }
        } catch(e) { console.error(e); }
        hideSpinner();
    });
}

function renderPortfolio() {
    const list = document.getElementById('portfolio-list');
    list.innerHTML = '';
    
    if(globalPortfolio.length === 0) {
        list.innerHTML = '<span style="color:var(--text-muted); font-size:0.85rem;">Portfolio is empty.</span>';
        return;
    }

    globalPortfolio.forEach((pos, i) => {
        const div = document.createElement('div');
        div.className = 'pos-item';
        div.innerHTML = `
            <span>${pos.qty > 0 ? 'Long' : 'Short'} ${Math.abs(pos.qty)}x ${pos.type.toUpperCase()} (K=${pos.K})</span>
            <button onclick="removePos(${i})">✕</button>
        `;
        list.appendChild(div);
    });
}

window.removePos = function(index) {
    globalPortfolio.splice(index, 1);
    renderPortfolio();
}

function renderRiskChart(varVal, cvarVal, mean, std, conf) {
    // Generate synthetic normal P&L distribution centered at mean with true std
    // strictly for display purpose since API doesn't return full 100k array to save bandwidth.
    let pnl = [];
    for(let i=0; i<10000; i++) {
        // Box-Muller approx
        let u1=Math.random(), u2=Math.random();
        let z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        pnl.push(mean + z * std);
    }

    const trace = {
        x: pnl,
        type: 'histogram',
        nbinsx: 50,
        marker: { color: 'rgba(88, 166, 255, 0.7)' }
    };
    
    // VaR line
    const varLine = {
        type: 'line',
        x0: -varVal, x1: -varVal,
        y0: 0, y1: 1, yref: 'paper',
        line: { color: '#f85149', width: 2, dash: 'dash' }
    };

    const layout = { 
        ...plotLayout, 
        title: 'P&L Distribution & Tail Risk',
        xaxis: {title: 'P&L'}, yaxis: {title: 'Frequency'},
        shapes: [varLine],
        annotations: [{
            x: -varVal, y: 1, xref: 'x', yref: 'paper',
            text: `VaR (${(conf*100).toFixed(0)}%)`, showarrow: true, font: {color: '#f85149'}
        }]
    };
    
    Plotly.newPlot('risk-chart', [trace], layout, {responsive: true});
}
