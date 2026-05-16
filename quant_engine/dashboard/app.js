// API Base URL (dynamically uses local or production host)
const API_URL = window.location.origin;

// Global Chart layout configuration (Sensibull flat theme)
const plotLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#8b9298', family: 'Inter', size: 12 },
    margin: { t: 40, r: 20, b: 40, l: 50 },
    xaxis: { gridcolor: '#272a31', zerolinecolor: '#3f434a', zerolinewidth: 2 },
    yaxis: { gridcolor: '#272a31', zerolinecolor: '#3f434a', zerolinewidth: 2 }
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
    setupStrategySimulator();
    setupScenarioAnalysis();
    setupMarketData();
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
        const res = await fetch(`${API_URL}/health`);
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
        
        let t_val = parseFloat(document.getElementById('price-t').value);
        const t_unit = document.getElementById('price-t-unit').value;
        if(t_unit === 'days') t_val = t_val / 365.0;
        if(t_unit === 'weeks') t_val = t_val / 52.0;

        // Build payload
        const payload = {
            S: parseFloat(document.getElementById('price-s').value),
            K: parseFloat(document.getElementById('price-k').value),
            T: t_val,
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
        return val - premium;
    });

    const profit = payoffs.map(p => p >= 0 ? p : null);
    const loss = payoffs.map(p => p <= 0 ? p : null);

    const traceProfit = {
        x: spots, y: profit, type: 'scatter', mode: 'none',
        fill: 'tozeroy', fillcolor: 'rgba(0, 204, 102, 0.25)', name: 'Profit',
        hoverinfo: 'skip'
    };
    
    const traceLoss = {
        x: spots, y: loss, type: 'scatter', mode: 'none',
        fill: 'tozeroy', fillcolor: 'rgba(255, 59, 48, 0.25)', name: 'Loss',
        hoverinfo: 'skip'
    };

    const traceLine = {
        x: spots, y: payoffs, type: 'scatter', mode: 'lines',
        line: {color: '#2962ff', width: 2}, name: 'P&L'
    };

    const layout = { ...plotLayout, title: 'Option P&L at Expiry', showlegend: false, xaxis: {title:'Spot Price'}, yaxis: {title:'Profit / Loss'} };
    Plotly.newPlot('pricing-chart', [traceProfit, traceLoss, traceLine], layout, {responsive: true});
}

// ==========================================
// Greeks Explorer
// ==========================================
function setupGreeksExplorer() {
    document.getElementById('greeks-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        let t_val = parseFloat(document.getElementById('gr-t').value);
        const t_unit = document.getElementById('gr-t-unit').value;
        if(t_unit === 'days') t_val = t_val / 365.0;
        if(t_unit === 'weeks') t_val = t_val / 52.0;

        const payload = {
            S: parseFloat(document.getElementById('gr-s').value),
            K: parseFloat(document.getElementById('gr-k').value),
            T: t_val,
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
                if(g.rho !== undefined) document.getElementById('gr-res-rho').innerText = g.rho.toFixed(4);

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
        line: {color: '#2962ff', width: 2},
        fill: 'tozeroy',
        fillcolor: 'rgba(41, 98, 255, 0.1)'
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
        const sStr = document.getElementById('pos-s').value;
        const kStr = document.getElementById('pos-k').value;
        
        const qty = qtyStr === "" ? 1 : parseInt(qtyStr);
        const s = sStr === "" ? 24000 : parseFloat(sStr);
        const k = kStr === "" ? 24000 : parseFloat(kStr);
        
        globalPortfolio.push({
            type: type, S: s, K: k, T: 0.5, r: 0.05, sigma: 0.2, qty: qty, asset: 'default'
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
            <span>${pos.qty > 0 ? 'Long' : 'Short'} ${Math.abs(pos.qty)}x ${pos.type.toUpperCase()} (S=${pos.S}, K=${pos.K})</span>
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
        marker: { color: 'rgba(0, 204, 102, 0.6)' }
    };
    
    // VaR line
    const varLine = {
        type: 'line',
        x0: -varVal, x1: -varVal,
        y0: 0, y1: 1, yref: 'paper',
        line: { color: '#ff3b30', width: 2, dash: 'dash' }
    };

    const layout = { 
        ...plotLayout, 
        title: 'P&L Distribution & Tail Risk',
        xaxis: {title: 'P&L'}, yaxis: {title: 'Frequency'},
        shapes: [varLine],
        annotations: [{
            x: -varVal, y: 1, xref: 'x', yref: 'paper',
            text: `VaR (${(conf*100).toFixed(0)}%)`, showarrow: true, font: {color: '#ff3b30', size: 10}
        }]
    };
    
    Plotly.newPlot('risk-chart', [trace], layout, {responsive: true});
}

// ==========================================
// Strategy Simulator
// ==========================================
let selectedStrategy = 'straddle';
const STRATEGIES = {
    long_call: {name:'Long Call', legs:[{type:'call',off:0,qty:1}]},
    long_put: {name:'Long Put', legs:[{type:'put',off:0,qty:1}]},
    bull_call_spread: {name:'Bull Call Spread', legs:[{type:'call',off:-50,qty:1},{type:'call',off:50,qty:-1}]},
    bear_put_spread: {name:'Bear Put Spread', legs:[{type:'put',off:50,qty:1},{type:'put',off:-50,qty:-1}]},
    straddle: {name:'Straddle', legs:[{type:'call',off:0,qty:1},{type:'put',off:0,qty:1}]},
    strangle: {name:'Strangle', legs:[{type:'call',off:100,qty:1},{type:'put',off:-100,qty:1}]},
    iron_condor: {name:'Iron Condor', legs:[{type:'put',off:-200,qty:1},{type:'put',off:-100,qty:-1},{type:'call',off:100,qty:-1},{type:'call',off:200,qty:1}]},
    butterfly: {name:'Butterfly', legs:[{type:'call',off:-100,qty:1},{type:'call',off:0,qty:-2},{type:'call',off:100,qty:1}]},
};

function setupStrategySimulator() {
    const sel = document.getElementById('strategy-selector');
    Object.keys(STRATEGIES).forEach(id => {
        const btn = document.createElement('button');
        btn.className = 'strategy-btn' + (id === selectedStrategy ? ' active' : '');
        btn.innerText = STRATEGIES[id].name;
        btn.addEventListener('click', () => {
            selectedStrategy = id;
            sel.querySelectorAll('.strategy-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderStrategyLegs();
        });
        sel.appendChild(btn);
    });
    renderStrategyLegs();

    document.getElementById('strategy-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        let t = parseFloat(document.getElementById('strat-t').value);
        const unit = document.getElementById('strat-t-unit').value;
        if(unit==='days') t /= 365; if(unit==='weeks') t /= 52;

        const payload = {
            strategy_id: selectedStrategy,
            S: parseFloat(document.getElementById('strat-s').value),
            K: parseFloat(document.getElementById('strat-k').value),
            T: t,
            r: parseFloat(document.getElementById('strat-r').value),
            sigma: parseFloat(document.getElementById('strat-v').value),
            lot_size: 1,
        };
        showSpinner();
        try {
            const res = await fetch(`${API_URL}/strategies/simulate`, {
                method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)
            });
            const data = await res.json();
            if(res.ok) renderStrategyResults(data);
            else alert(data.detail || data.error);
        } catch(e) { console.error(e); }
        hideSpinner();
    });
}

function renderStrategyLegs() {
    const K = parseFloat(document.getElementById('strat-k').value) || 24000;
    const legs = STRATEGIES[selectedStrategy].legs;
    const el = document.getElementById('strategy-legs');
    el.innerHTML = '';
    legs.forEach(l => {
        const div = document.createElement('div');
        div.className = 'leg-item';
        const cls = l.qty > 0 ? 'leg-long' : 'leg-short';
        const dir = l.qty > 0 ? 'BUY' : 'SELL';
        div.innerHTML = `<span class="${cls}">${dir} ${Math.abs(l.qty)}x ${l.type.toUpperCase()}</span><span>K=${K + l.off}</span>`;
        el.appendChild(div);
    });
}

function renderStrategyResults(data) {
    // Stats
    const stats = document.getElementById('strategy-stats');
    stats.innerHTML = `
        <div class="mini-stat"><div class="label">Entry Cost</div><div class="value">₹${data.entry_premium.toFixed(2)}</div></div>
        <div class="mini-stat"><div class="label">Max Profit</div><div class="value profit">${data.max_profit >= 9999 ? '∞' : '₹'+data.max_profit.toFixed(2)}</div></div>
        <div class="mini-stat"><div class="label">Max Loss</div><div class="value loss">₹${data.max_loss.toFixed(2)}</div></div>
        <div class="mini-stat"><div class="label">Breakeven</div><div class="value">${data.breakevens.length ? data.breakevens.map(b=>'₹'+b).join(', ') : 'N/A'}</div></div>
    `;
    // Greeks
    const gEl = document.getElementById('strategy-greeks');
    const g = data.greeks;
    gEl.innerHTML = `
        <div class="stat-card glass"><h4>Net Δ</h4><div class="stat-value">${g.delta.toFixed(4)}</div></div>
        <div class="stat-card glass"><h4>Net Γ</h4><div class="stat-value">${g.gamma.toFixed(5)}</div></div>
        <div class="stat-card glass"><h4>Net V</h4><div class="stat-value">${g.vega.toFixed(4)}</div></div>
        <div class="stat-card glass"><h4>Net Θ</h4><div class="stat-value">${g.theta.toFixed(4)}</div></div>
        <div class="stat-card glass"><h4>Net ρ</h4><div class="stat-value">${g.rho.toFixed(4)}</div></div>
    `;
    // Chart
    const profit = data.pnl.map(p => p >= 0 ? p : null);
    const loss = data.pnl.map(p => p <= 0 ? p : null);
    const traces = [
        {x:data.spots, y:profit, type:'scatter', mode:'none', fill:'tozeroy', fillcolor:'rgba(0,204,102,0.25)', name:'Profit', hoverinfo:'skip'},
        {x:data.spots, y:loss, type:'scatter', mode:'none', fill:'tozeroy', fillcolor:'rgba(255,59,48,0.25)', name:'Loss', hoverinfo:'skip'},
        {x:data.spots, y:data.pnl, type:'scatter', mode:'lines', line:{color:'#2962ff',width:2}, name:'P&L'},
    ];
    const layout = {...plotLayout, title:`${data.strategy.name} — P&L at Expiry`, xaxis:{title:'Spot Price'}, yaxis:{title:'P&L'}};
    Plotly.newPlot('strategy-chart', traces, layout, {responsive:true});
}

// ==========================================
// Scenario Analysis
// ==========================================
let scenarioPortfolio = [];

function setupScenarioAnalysis() {
    renderScenarioPortfolio();
    document.getElementById('btn-sc-add').addEventListener('click', () => {
        scenarioPortfolio.push({
            type: document.getElementById('sc-pos-type').value,
            S: parseFloat(document.getElementById('sc-pos-s').value) || 24000,
            K: parseFloat(document.getElementById('sc-pos-k').value) || 24000,
            T: 0.08, r: 0.069, sigma: 0.14,
            qty: parseInt(document.getElementById('sc-pos-qty').value) || 10,
        });
        renderScenarioPortfolio();
    });

    document.getElementById('scenario-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        if(!scenarioPortfolio.length) { alert('Add positions first'); return; }
        const payload = {
            positions: scenarioPortfolio,
            x_axis: document.getElementById('sc-x-axis').value,
            y_axis: document.getElementById('sc-y-axis').value,
            n_points: parseInt(document.getElementById('sc-res').value) || 15,
        };
        showSpinner();
        try {
            const res = await fetch(`${API_URL}/scenario/heatmap`, {
                method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)
            });
            const data = await res.json();
            if(res.ok) renderScenarioHeatmap(data);
            else alert(data.detail || data.error);
        } catch(e) { console.error(e); }
        hideSpinner();
    });
}

function renderScenarioPortfolio() {
    const el = document.getElementById('scenario-portfolio');
    el.innerHTML = '';
    if(!scenarioPortfolio.length) { el.innerHTML = '<span style="color:var(--text-muted);font-size:0.8rem;">No positions</span>'; return; }
    scenarioPortfolio.forEach((p,i) => {
        const div = document.createElement('div');
        div.className = 'pos-item';
        div.innerHTML = `<span>${p.qty>0?'Long':'Short'} ${Math.abs(p.qty)}x ${p.type.toUpperCase()} K=${p.K}</span><button onclick="removeScPos(${i})">✕</button>`;
        el.appendChild(div);
    });
}
window.removeScPos = function(i) { scenarioPortfolio.splice(i,1); renderScenarioPortfolio(); }

function renderScenarioHeatmap(data) {
    const stats = document.getElementById('scenario-stats');
    stats.innerHTML = `
        <div class="stat-card glass"><h4>Base Value</h4><div class="stat-value">₹${data.base_value.toFixed(2)}</div></div>
        <div class="stat-card glass"><h4>Computed In</h4><div class="stat-value">${data.elapsed_ms.toFixed(0)}ms</div></div>
    `;
    const trace = {
        z: data.z_matrix, x: data.x_labels, y: data.y_labels,
        type: 'heatmap',
        colorscale: [
            [0, '#ff3b30'], [0.35, '#ff6b5a'], [0.5, '#1a1c20'],
            [0.65, '#4caf50'], [1, '#00cc66']
        ],
        zmid: 0,
        colorbar: {title:'P&L', titlefont:{color:'#8b9298'}, tickfont:{color:'#8b9298'}},
    };
    const layout = {
        ...plotLayout,
        title: `P&L Heatmap (${data.x_axis} × ${data.y_axis})`,
        xaxis: {title: data.x_axis === 'spot' ? 'Spot Shift' : 'Days Forward', tickfont:{size:10}},
        yaxis: {title: data.y_axis === 'vol' ? 'Vol Shift' : 'Days Forward', tickfont:{size:10}},
    };
    Plotly.newPlot('scenario-chart', [trace], layout, {responsive:true});
}

// ==========================================
// Market Data
// ==========================================
let activeSymbol = 'NIFTY';

function setupMarketData() {
    document.querySelectorAll('.symbol-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.symbol-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            activeSymbol = tab.dataset.symbol;
            loadMarketData(activeSymbol);
        });
    });
    loadMarketData('NIFTY');
}

async function loadMarketData(symbol) {
    showSpinner();
    try {
        const [qRes, cRes] = await Promise.all([
            fetch(`${API_URL}/market/quote/${symbol}`),
            fetch(`${API_URL}/market/option-chain/${symbol}`),
        ]);
        const quote = await qRes.json();
        const chain = await cRes.json();
        if(qRes.ok) renderQuote(quote);
        if(cRes.ok) renderOptionChain(chain);
    } catch(e) { console.error(e); }
    hideSpinner();
}

function renderQuote(q) {
    const el = document.getElementById('market-quote');
    const cls = q.change >= 0 ? 'up' : 'down';
    const arrow = q.change >= 0 ? '▲' : '▼';
    el.innerHTML = `
        <div class="symbol-name">${q.name} (${q.symbol})</div>
        <div class="quote-price" style="color:var(--text-dark)">₹${q.last_price.toLocaleString()}</div>
        <div class="quote-change ${cls}">${arrow} ₹${Math.abs(q.change).toFixed(2)} (${q.change_pct >= 0 ? '+' : ''}${q.change_pct.toFixed(2)}%)</div>
        <div class="quote-meta">
            <span>O: ₹${q.open.toLocaleString()}</span>
            <span>H: ₹${q.high.toLocaleString()}</span>
            <span>L: ₹${q.low.toLocaleString()}</span>
            <span>Vol: ${(q.volume/1e6).toFixed(1)}M</span>
            <span>Lot: ${q.lot_size}</span>
            <span style="margin-left:auto;font-size:0.7rem;">${q.provider.toUpperCase()}</span>
        </div>
    `;
}

function renderOptionChain(data) {
    const wrap = document.getElementById('chain-table-wrap');
    const spot = data.spot;
    let html = `<table class="chain-table">
        <thead><tr>
            <th>OI</th><th>Vol</th><th>IV%</th><th>Δ</th><th>Price</th>
            <th class="strike-col">Strike</th>
            <th>Price</th><th>Δ</th><th>IV%</th><th>Vol</th><th>OI</th>
        </tr></thead><tbody>`;

    data.chain.forEach(row => {
        const isATM = Math.abs(row.strike - spot) < data.chain[1].strike - data.chain[0].strike;
        const cls = isATM ? ' class="atm-row"' : '';
        const cITM = row.strike < spot ? 'itm' : 'otm';
        const pITM = row.strike > spot ? 'itm' : 'otm';
        html += `<tr${cls}>
            <td class="${cITM}">${(row.call.oi/1000).toFixed(0)}K</td>
            <td class="${cITM}">${(row.call.volume/1000).toFixed(0)}K</td>
            <td class="${cITM}">${row.call.iv.toFixed(1)}</td>
            <td class="${cITM}">${row.call.delta.toFixed(2)}</td>
            <td class="${cITM}" style="font-weight:600">₹${row.call.price.toFixed(2)}</td>
            <td class="strike-col">${row.strike.toFixed(0)}</td>
            <td class="${pITM}" style="font-weight:600">₹${row.put.price.toFixed(2)}</td>
            <td class="${pITM}">${row.put.delta.toFixed(2)}</td>
            <td class="${pITM}">${row.put.iv.toFixed(1)}</td>
            <td class="${pITM}">${(row.put.volume/1000).toFixed(0)}K</td>
            <td class="${pITM}">${(row.put.oi/1000).toFixed(0)}K</td>
        </tr>`;
    });
    html += '</tbody></table>';
    wrap.innerHTML = html;
}
