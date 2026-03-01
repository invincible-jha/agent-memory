/* agent-memory dashboard — vanilla JS + Chart.js */
'use strict';

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + target).classList.add('active');
    refreshActiveTab(target);
  });
});

function refreshActiveTab(tab) {
  if (tab === 'browser') loadMemories();
  else if (tab === 'graph') loadGraph();
  else if (tab === 'stats') loadStats();
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function fmtTime(ts) {
  if (!ts) return '-';
  return new Date(ts * 1000).toLocaleString();
}
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
function layerClass(layer) {
  return 'layer-badge layer-' + (layer || 'working');
}
function importanceBar(val) {
  const pct = Math.round((Number(val) || 0.5) * 100);
  return `<span class="importance-bar" style="width:${pct}px" title="${pct}%"></span> ${pct}%`;
}

async function apiFetch(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return res.json();
}

function showToast(msg) {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

function setLastRefresh() {
  document.getElementById('last-refresh').textContent =
    'Refreshed ' + new Date().toLocaleTimeString();
}

// ---------------------------------------------------------------------------
// Memory Browser
// ---------------------------------------------------------------------------
async function loadMemories() {
  try {
    const layer = document.getElementById('layer-filter').value;
    let url = '/api/memories?limit=200';
    if (layer) url += '&layer=' + encodeURIComponent(layer);

    const [memoriesData, statsData] = await Promise.all([
      apiFetch(url),
      apiFetch('/api/stats'),
    ]);

    // Quick stats
    document.getElementById('qs-total').textContent = statsData.total || 0;
    const byLayer = statsData.by_layer || {};
    document.getElementById('qs-working').textContent = byLayer.working || 0;
    document.getElementById('qs-episodic').textContent = byLayer.episodic || 0;
    document.getElementById('qs-semantic').textContent = byLayer.semantic || 0;
    document.getElementById('qs-procedural').textContent = byLayer.procedural || 0;

    const tbody = document.getElementById('memories-tbody');
    const memories = memoriesData.memories || [];
    if (memories.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No memory entries found.</td></tr>';
      setLastRefresh();
      return;
    }
    tbody.innerHTML = memories.slice().reverse().map(m => `
      <tr>
        <td style="font-family:monospace;font-size:11px">${escHtml(String(m.id || '-').slice(0, 12))}...</td>
        <td><span class="${layerClass(m.layer)}">${escHtml(m.layer || 'working')}</span></td>
        <td class="content-preview" title="${escHtml(m.content || '')}">${escHtml(String(m.content || '-').slice(0, 80))}${String(m.content || '').length > 80 ? '…' : ''}</td>
        <td>${importanceBar(m.importance)}</td>
        <td>${escHtml(m.source || '-')}</td>
        <td>${fmtTime(m.timestamp)}</td>
      </tr>
    `).join('');
    setLastRefresh();
  } catch (err) {
    showToast('Error loading memories: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Knowledge Graph (force-directed — canvas-based, no external lib)
// ---------------------------------------------------------------------------
let graphAnimationId = null;

async function loadGraph() {
  try {
    const data = await apiFetch('/api/graph');
    renderGraph(data.nodes || [], data.edges || []);
    setLastRefresh();
  } catch (err) {
    showToast('Error loading graph: ' + err.message);
  }
}

function renderGraph(nodes, edges) {
  const canvas = document.getElementById('graph-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight;
  canvas.width = W;
  canvas.height = H;

  if (graphAnimationId) cancelAnimationFrame(graphAnimationId);

  if (nodes.length === 0) {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim();
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No knowledge graph nodes yet.', W / 2, H / 2);
    return;
  }

  // Initialize positions randomly
  const positions = nodes.map(() => ({
    x: W * 0.1 + Math.random() * W * 0.8,
    y: H * 0.1 + Math.random() * H * 0.8,
    vx: 0,
    vy: 0,
  }));

  const nodeIndex = {};
  nodes.forEach((n, i) => { nodeIndex[n.id] = i; });

  const nodeRadius = 18;
  const accentColor = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
  const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text').trim();
  const borderColor = getComputedStyle(document.documentElement).getPropertyValue('--border').trim();

  let frame = 0;

  function tick() {
    // Force-directed layout (simplified)
    const k = 60;
    const repulse = k * k;

    positions.forEach((p, i) => {
      positions.forEach((q, j) => {
        if (i === j) return;
        const dx = p.x - q.x;
        const dy = p.y - q.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = repulse / (dist * dist);
        p.vx += force * (dx / dist);
        p.vy += force * (dy / dist);
      });
    });

    edges.forEach(edge => {
      const si = nodeIndex[edge.source];
      const ti = nodeIndex[edge.target];
      if (si == null || ti == null) return;
      const dx = positions[si].x - positions[ti].x;
      const dy = positions[si].y - positions[ti].y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const force = (dist - k) * 0.05;
      positions[si].vx -= force * (dx / dist);
      positions[si].vy -= force * (dy / dist);
      positions[ti].vx += force * (dx / dist);
      positions[ti].vy += force * (dy / dist);
    });

    positions.forEach(p => {
      p.vx *= 0.85;
      p.vy *= 0.85;
      p.x = Math.max(nodeRadius, Math.min(W - nodeRadius, p.x + p.vx));
      p.y = Math.max(nodeRadius, Math.min(H - nodeRadius, p.y + p.vy));
    });

    // Draw
    ctx.clearRect(0, 0, W, H);

    // Edges
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1.5;
    edges.forEach(edge => {
      const si = nodeIndex[edge.source];
      const ti = nodeIndex[edge.target];
      if (si == null || ti == null) return;
      ctx.beginPath();
      ctx.moveTo(positions[si].x, positions[si].y);
      ctx.lineTo(positions[ti].x, positions[ti].y);
      ctx.stroke();
    });

    // Nodes
    nodes.forEach((node, i) => {
      const p = positions[i];
      ctx.beginPath();
      ctx.arc(p.x, p.y, nodeRadius, 0, Math.PI * 2);
      ctx.fillStyle = accentColor;
      ctx.globalAlpha = 0.8;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = textColor;
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const label = String(node.label || node.id || '').slice(0, 8);
      ctx.fillText(label, p.x, p.y);
    });

    frame++;
    if (frame < 200) {
      graphAnimationId = requestAnimationFrame(tick);
    }
  }

  tick();
}

// ---------------------------------------------------------------------------
// Stats Tab
// ---------------------------------------------------------------------------
let layerChart = null;

async function loadStats() {
  try {
    const data = await apiFetch('/api/stats');
    const byLayer = data.by_layer || {};
    const layers = ['working', 'episodic', 'semantic', 'procedural'];
    const counts = layers.map(l => byLayer[l] || 0);
    const colors = ['#58a6ff', '#f78166', '#3fb950', '#d2a8ff'];

    const ctx = document.getElementById('layer-chart').getContext('2d');
    if (layerChart) layerChart.destroy();
    layerChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: layers.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
        datasets: [{
          data: counts,
          backgroundColor: colors,
          borderColor: getComputedStyle(document.documentElement).getPropertyValue('--surface').trim(),
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: getComputedStyle(document.documentElement).getPropertyValue('--text').trim(),
            },
          },
        },
      },
    });

    const tbody = document.getElementById('stats-tbody');
    tbody.innerHTML = [
      ['Total Entries', data.total || 0],
      ['Working Layer', byLayer.working || 0],
      ['Episodic Layer', byLayer.episodic || 0],
      ['Semantic Layer', byLayer.semantic || 0],
      ['Procedural Layer', byLayer.procedural || 0],
      ['Graph Nodes', data.graph_nodes || 0],
      ['Graph Edges', data.graph_edges || 0],
      ['Avg Importance', data.avg_importance || '0.000'],
    ].map(([k, v]) => `<tr><td>${k}</td><td style="font-weight:600">${v}</td></tr>`).join('');

    setLastRefresh();
  } catch (err) {
    showToast('Error loading stats: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Search Tab
// ---------------------------------------------------------------------------
document.getElementById('search-query').addEventListener('keydown', e => {
  if (e.key === 'Enter') runSearch();
});

async function runSearch() {
  const query = document.getElementById('search-query').value.trim();
  if (!query) { showToast('Please enter a search query.'); return; }

  const container = document.getElementById('search-results-container');
  container.innerHTML = '<div class="empty-state">Searching...</div>';

  try {
    const data = await apiFetch('/api/search?q=' + encodeURIComponent(query) + '&limit=100');
    const results = data.results || [];

    if (results.length === 0) {
      container.innerHTML = `<div class="empty-state"><div class="icon">&#128269;</div>No results for "${escHtml(query)}".</div>`;
      return;
    }

    container.innerHTML = `
      <div style="margin-bottom:12px;color:var(--text-muted);font-size:13px">
        Found <strong>${results.length}</strong> result(s) for "<strong>${escHtml(query)}</strong>"
      </div>
      <table>
        <thead>
          <tr><th>ID</th><th>Layer</th><th>Content</th><th>Importance</th><th>Timestamp</th></tr>
        </thead>
        <tbody>
          ${results.map(m => `
            <tr>
              <td style="font-family:monospace;font-size:11px">${escHtml(String(m.id || '-').slice(0, 12))}...</td>
              <td><span class="${layerClass(m.layer)}">${escHtml(m.layer || 'working')}</span></td>
              <td class="content-preview">${escHtml(String(m.content || '-').slice(0, 120))}</td>
              <td>${importanceBar(m.importance)}</td>
              <td>${fmtTime(m.timestamp)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } catch (err) {
    showToast('Search error: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
loadMemories();
setInterval(() => {
  const active = document.querySelector('.tab-btn.active');
  if (active) refreshActiveTab(active.dataset.tab);
}, 30000);
