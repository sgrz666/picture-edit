/**
 * SG-Node01 显存与内存实时监管仪表盘 — 前端核心控制逻辑 v2.0
 * Zero-CDN / 100% 局域网友好 / 原生 Vanilla JS + Inline SVG
 */

(() => {
  'use strict';

  // =========================================================================
  // 全局响应式状态管理 (State Management)
  // =========================================================================
  const state = {
    pollingInterval: 2000,
    isPaused: false,
    timerId: null,
    isFetching: false,
    isOffline: false,
    retryCount: 0,
    currentData: null,

    // 视图模式: 'all' | 'memory' | 'gpu'
    viewMode: 'all',

    // 多 GPU 筛选状态: 'all' | '0' | '1' | '2'
    selectedGpu: 'all',

    // GPU 进程表格状态
    searchQuery: '',
    sortColumn: 'vram_used_mb',
    sortDirection: 'desc',

    // 内存进程表格状态
    memSearchQuery: '',
    memQuickFilter: 'all',    // 'all' | 'swap' | 'python' | 'oom'
    memUserFilter: '',        // '' 表示全部用户
    memSortColumn: 'rss_mb',
    memSortDirection: 'desc',
  };

  // =========================================================================
  // DOM 元素缓存 (Element Selectors)
  // =========================================================================
  const el = {
    offlineBanner: document.getElementById('offline-banner'),
    retryCount: document.getElementById('retry-count'),

    // 头部与视图切换
    hostName: document.getElementById('host-name'),
    gpuModelHeader: document.getElementById('gpu-model-header'),
    connectionBadge: document.getElementById('connection-badge'),
    connectionStatusText: document.getElementById('connection-status-text'),
    lastUpdated: document.getElementById('last-updated'),
    viewTabs: document.querySelectorAll('.view-tab'),
    quickAnonGb: document.getElementById('quick-anon-gb'),
    quickShmemGb: document.getElementById('quick-shmem-gb'),
    quickSwapGb: document.getElementById('quick-swap-gb'),
    quickPsi: document.getElementById('quick-psi'),
    quickOomPill: document.getElementById('quick-oom-pill'),
    quickOomKills: document.getElementById('quick-oom-kills'),

    // 控制栏
    intervalSelect: document.getElementById('interval-select'),
    btnPause: document.getElementById('btn-pause'),
    btnPauseText: document.getElementById('btn-pause-text'),
    btnRefresh: document.getElementById('btn-refresh'),

    // 多 GPU 设备选择栏与筛选
    gpuDevicesRibbon: document.getElementById('gpu-devices-ribbon'),
    gpuQuickFilters: document.getElementById('gpu-quick-filters'),
    cardVramTitle: document.getElementById('card-vram-title'),
    cardUtilTitle: document.getElementById('card-util-title'),
    cardTempTitle: document.getElementById('card-temp-title'),

    // 顶部 4 张核心概览卡片
    cardVram: document.getElementById('card-vram'),
    cardRam: document.getElementById('card-ram'),
    cardUtil: document.getElementById('card-util'),
    cardTemp: document.getElementById('card-temp'),

    vramUsedGb: document.getElementById('vram-used-gb'),
    vramTotalGb: document.getElementById('vram-total-gb'),
    vramUsedMb: document.getElementById('vram-used-mb'),
    vramFreeGb: document.getElementById('vram-free-gb'),
    vramPctBadge: document.getElementById('vram-pct-badge'),
    vramProgressBar: document.getElementById('vram-progress-bar'),

    ramUsedGb: document.getElementById('ram-used-gb'),
    ramTotalGb: document.getElementById('ram-total-gb'),
    ramAnonShmemStat: document.getElementById('ram-anon-shmem-stat'),
    ramAvailGb: document.getElementById('ram-avail-gb'),
    swapUsedStat: document.getElementById('swap-used-stat'),
    ramPctBadge: document.getElementById('ram-pct-badge'),
    ramProgressBar: document.getElementById('ram-progress-bar'),

    gpuUtilPct: document.getElementById('gpu-util-pct'),
    gpuUtilBadge: document.getElementById('gpu-util-badge'),
    utilProgressBar: document.getElementById('util-progress-bar'),
    gpuDriverVersion: document.getElementById('gpu-driver-version'),
    sysLoad: document.getElementById('sys-load'),

    gpuTempC: document.getElementById('gpu-temp-c'),
    gpuTempBadge: document.getElementById('gpu-temp-badge'),
    gpuTempDesc: document.getElementById('gpu-temp-desc'),
    tempProgressBar: document.getElementById('temp-progress-bar'),
    sysUptime: document.getElementById('sys-uptime'),

    // 内存用户配额与内存进程表
    userQuotaBar: document.getElementById('user-quota-bar'),
    memProcessCountBadge: document.getElementById('mem-process-count-badge'),
    memQuickFilters: document.querySelectorAll('#mem-quick-filters .filter-chip'),
    memProcessSearch: document.getElementById('mem-process-search'),
    memSearchClearBtn: document.getElementById('mem-search-clear-btn'),
    memProcessTable: document.getElementById('mem-process-table'),
    memProcessTableBody: document.getElementById('mem-process-table-body'),
    memEmptyState: document.getElementById('mem-process-empty-state'),
    memEmptyMessage: document.getElementById('mem-empty-state-message'),
    memEmptySubmessage: document.getElementById('mem-empty-state-submessage'),
    memFooterPrefix: document.getElementById('mem-footer-prefix'),
    memFooterHint: document.getElementById('mem-footer-hint'),
    totalTrackedRss: document.getElementById('total-tracked-rss'),
    totalTrackedSwap: document.getElementById('total-tracked-swap'),
    activeUserFilterLabel: document.getElementById('active-user-filter-label'),
    memTableHeaders: document.querySelectorAll('.th-sortable-mem'),

    // GPU 进程表
    processTable: document.getElementById('process-table'),
    processTableBody: document.getElementById('process-table-body'),
    processCountBadge: document.getElementById('process-count-badge'),
    processSearch: document.getElementById('process-search'),
    searchClearBtn: document.getElementById('search-clear-btn'),
    emptyState: document.getElementById('process-empty-state'),
    emptyStateMessage: document.getElementById('empty-state-message'),
    emptyStateSubmessage: document.getElementById('empty-state-submessage'),
    totalTrackedVram: document.getElementById('total-tracked-vram'),
    totalGpuCapacity: document.getElementById('total-gpu-capacity'),
    tableHeaders: document.querySelectorAll('.th-sortable'),
  };

  // =========================================================================
  // 工具函数 (Utility Functions)
  // =========================================================================

  function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function formatMemMb(mb, zeroLabel) {
    if (mb == null || isNaN(mb)) return '--';
    if (zeroLabel && mb <= 0) return zeroLabel;
    if (mb >= 1024) {
      return (mb / 1024).toFixed(2) + ' GB';
    }
    return Math.round(mb) + ' MB';
  }

  function formatGbFromMb(mb, decimals = 1) {
    if (mb == null || isNaN(mb)) return '--';
    return (mb / 1024).toFixed(decimals) + ' GB';
  }

  function formatUptime(seconds) {
    if (!seconds || isNaN(seconds)) return '--';
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (days > 0) return `${days}天 ${hours}小时`;
    if (hours > 0) return `${hours}小时 ${minutes}分`;
    return `${minutes}分钟`;
  }

  function getUsageBadgeMeta(pct) {
    const rounded = pct.toFixed(1);
    if (pct >= 90) return { text: `${rounded}% 告警`, badgeClass: 'badge-crit' };
    if (pct >= 75) return { text: `${rounded}% 预警`, badgeClass: 'badge-warn' };
    return { text: `${rounded}% 正常`, badgeClass: 'badge-safe' };
  }

  function getStatusFillClass(pct) {
    if (pct >= 90) return 'fill-crit';
    if (pct >= 75) return 'fill-warn';
    return 'fill-safe';
  }

  function filterByTokens(processes, query) {
    if (!query || !query.trim()) return processes;
    const tokens = query.toLowerCase().trim().split(/\s+/);
    return processes.filter(p => {
      const pidStr = String(p.pid || '');
      const userStr = String(p.user || '').toLowerCase();
      const nameStr = String(p.name || '').toLowerCase();
      const cmdStr = String(p.cmdline || '').toLowerCase();
      return tokens.every(token =>
        pidStr.includes(token) ||
        userStr.includes(token) ||
        nameStr.includes(token) ||
        cmdStr.includes(token)
      );
    });
  }

  function sortProcessList(processes, column, direction, numericCols) {
    const list = [...processes];
    const isAsc = direction === 'asc';
    const isNum = numericCols.has(column);

    list.sort((a, b) => {
      const valA = a[column];
      const valB = b[column];
      if (isNum) {
        const numA = Number(valA) || 0;
        const numB = Number(valB) || 0;
        return isAsc ? numA - numB : numB - numA;
      }
      const strA = String(valA || '').toLowerCase();
      const strB = String(valB || '').toLowerCase();
      if (strA < strB) return isAsc ? -1 : 1;
      if (strA > strB) return isAsc ? 1 : -1;
      return 0;
    });
    return list;
  }

  function updateSortHeaders(headerNodes, activeCol, activeDir) {
    headerNodes.forEach(th => {
      const col = th.getAttribute('data-sort');
      th.classList.remove('sorted-asc', 'sorted-desc');
      if (col === activeCol) {
        th.classList.add(activeDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
      }
    });
  }

  // =========================================================================
  // 视图切换与哈希路由 (View Mode Switcher)
  // =========================================================================
  function setViewMode(mode, updateHash = true) {
    const valid = ['all', 'memory', 'gpu'];
    const nextMode = valid.includes(mode) ? mode : 'all';
    state.viewMode = nextMode;

    document.body.classList.remove('view-all', 'view-memory', 'view-gpu');
    document.body.classList.add('view-' + nextMode);

    el.viewTabs.forEach(btn => {
      const isActive = btn.getAttribute('data-view') === nextMode;
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    if (updateHash) {
      const pathLower = (window.location.pathname || '/').toLowerCase();
      if (pathLower === '/memory' || pathLower === '/gpu') {
        const nextPath = nextMode === 'all' ? '/' : '/' + nextMode;
        try {
          history.replaceState(null, '', nextPath);
        } catch (_) {
          window.location.hash = nextMode === 'all' ? '' : '#' + nextMode;
        }
      } else {
        const targetHash = nextMode === 'all' ? '' : '#' + nextMode;
        if (window.location.hash !== targetHash) {
          try {
            history.replaceState(null, '', targetHash || window.location.pathname);
          } catch (_) {
            window.location.hash = targetHash;
          }
        }
      }
    }
  }

  function initViewModeFromHash() {
    const h = (window.location.hash || '').replace('#', '').toLowerCase();
    const p = (window.location.pathname || '/').replace(/\/+$/, '').toLowerCase();
    if (h === 'memory' || h === 'gpu' || h === 'all') {
      setViewMode(h, false);
    } else if (p === '/memory') {
      setViewMode('memory', false);
    } else if (p === '/gpu') {
      setViewMode('gpu', false);
    } else {
      setViewMode('all', false);
    }
  }

  // =========================================================================
  // 核心数据拉取与轮询调度器 (Fetcher & Poller)
  // =========================================================================

  async function fetchMetrics() {
    if (state.isFetching) return;
    state.isFetching = true;

    try {
      const response = await fetch('/api/metrics', {
        method: 'GET',
        cache: 'no-store',
        headers: { 'Accept': 'application/json' },
      });

      if (!response.ok) {
        throw new Error(`HTTP Error ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      handleFetchSuccess(data);
    } catch (error) {
      handleFetchError(error);
    } finally {
      state.isFetching = false;
      scheduleNextFetch();
    }
  }

  function handleFetchSuccess(data) {
    state.currentData = data;
    state.retryCount = 0;

    if (state.isOffline) {
      state.isOffline = false;
      if (el.offlineBanner) el.offlineBanner.classList.add('hidden');
      if (el.connectionBadge) {
        el.connectionBadge.classList.remove('offline');
        el.connectionBadge.classList.add('online');
      }
      if (el.connectionStatusText) el.connectionStatusText.textContent = '实时在线';

      [el.cardVram, el.cardRam, el.cardUtil, el.cardTemp].forEach(card => {
        if (card) card.classList.remove('offline');
      });
    }

    renderDashboard(data);
  }

  function handleFetchError() {
    state.retryCount++;
    state.isOffline = true;

    if (el.offlineBanner) el.offlineBanner.classList.remove('hidden');
    if (el.retryCount) el.retryCount.textContent = state.retryCount;

    if (el.connectionBadge) {
      el.connectionBadge.classList.remove('online');
      el.connectionBadge.classList.add('offline');
    }
    if (el.connectionStatusText) {
      el.connectionStatusText.textContent = `连接中断 (${state.retryCount} 次重试)`;
    }

    [el.cardVram, el.cardRam, el.cardUtil, el.cardTemp].forEach(card => {
      if (card) card.classList.add('offline');
    });
  }

  function scheduleNextFetch() {
    clearTimeout(state.timerId);
    if (state.isPaused) return;

    let delay = state.pollingInterval;
    if (state.isOffline) {
      delay = Math.min(1000 * Math.pow(1.5, Math.min(state.retryCount, 6)), 12000);
    }
    state.timerId = setTimeout(fetchMetrics, delay);
  }

  // =========================================================================
  // 视图渲染引擎 (Dashboard Renderers)
  // =========================================================================

  function renderDashboard(data) {
    if (!data) return;

    renderHeader(data);
    renderGpuRibbon(data.gpu);
    renderVramCard(data.gpu);
    renderRamCard(data.memory, data.swap);
    renderGpuUtilCard(data.gpu, data.system);
    renderGpuTempCard(data.gpu, data.system);

    renderUserQuotaBar(data.memory);
    renderMemoryProcessTable(data.memory, data.swap);

    renderProcessTable(data.gpu);
  }

  function renderHeader(data) {
    if (data.system?.hostname && el.hostName) {
      el.hostName.textContent = data.system.hostname;
    }

    if (el.gpuModelHeader) {
      const gpu = data.gpu || {};
      const devCount = gpu.device_count || gpu.count || (gpu.devices ? gpu.devices.length : 1);
      if (devCount > 1) {
        const totalCapGb = gpu.all_vram?.total_gb ? Math.round(gpu.all_vram.total_gb) : Math.round(devCount * 83.64);
        const cardCapGb = gpu.vram?.total_gb ? Math.round(gpu.vram.total_gb) : 84;
        const modelName = gpu.model || 'NVIDIA RTX 6000D';
        el.gpuModelHeader.textContent = `${devCount}x ${modelName} ${cardCapGb}GB (${totalCapGb}GB)`;
      } else if (gpu.model) {
        const vramTotalGb = gpu.vram?.total_gb ? Math.round(gpu.vram.total_gb) : 84;
        el.gpuModelHeader.textContent = `${gpu.model} ${vramTotalGb}GB`;
      }
    }

    if (el.lastUpdated) {
      if (data.formatted_time) {
        const parts = data.formatted_time.split(' ');
        el.lastUpdated.textContent = parts[1] || data.formatted_time;
      } else {
        el.lastUpdated.textContent = new Date().toTimeString().split(' ')[0];
      }
    }

    // Quick status strip in nav bar
    const mem = data.memory || {};
    const bd = mem.breakdown || {};
    const swap = data.swap || {};
    const psi = mem.pressure || {};
    if (el.quickAnonGb) el.quickAnonGb.textContent = formatGbFromMb(bd.anon_mb, 1);
    if (el.quickShmemGb) el.quickShmemGb.textContent = formatGbFromMb(bd.shmem_mb, 1);
    if (el.quickSwapGb) el.quickSwapGb.textContent = formatGbFromMb(swap.used_mb, 1);
    if (el.quickPsi) el.quickPsi.textContent = (psi.some_avg10 != null ? psi.some_avg10.toFixed(2) : '0.00') + '%';
    if (el.quickOomKills) {
      const oomKills = swap.oom_kills || 0;
      el.quickOomKills.textContent = `${oomKills} 次`;
      if (el.quickOomPill) el.quickOomPill.classList.toggle('warn', oomKills > 0);
    }
  }

  /**
   * 渲染多 GPU 状态切换栏及过滤按钮 (Multi-GPU Devices Ribbon)
   */
  function renderGpuRibbon(gpu) {
    if (!el.gpuDevicesRibbon) return;
    const devices = gpu?.devices || [];
    if (devices.length <= 1) {
      el.gpuDevicesRibbon.classList.add('hidden');
      if (el.gpuQuickFilters) el.gpuQuickFilters.classList.add('hidden');
      return;
    }

    el.gpuDevicesRibbon.classList.remove('hidden');
    if (el.gpuQuickFilters) el.gpuQuickFilters.classList.remove('hidden');

    const allVram = gpu.all_vram || {
      used_gb: devices.reduce((acc, d) => acc + (d.vram?.used_gb || 0), 0),
      total_gb: devices.reduce((acc, d) => acc + (d.vram?.total_gb || 83.64), 0),
      used_pct: 0,
    };
    if (allVram.total_gb > 0 && allVram.used_pct === 0) {
      allVram.used_pct = (allVram.used_gb / allVram.total_gb) * 100;
    }

    const maxUtil = Math.max(...devices.map(d => d.utilization_pct || 0));
    const activeCards = devices.filter(d => (d.utilization_pct || 0) > 0 || (d.processes || []).length > 0).length;

    // 1. Ribbon 选项: 全部 GPU (3卡池) + 每张卡
    const isAllActive = state.selectedGpu === 'all';
    let ribbonHtml = `
      <div class="gpu-device-chip ${isAllActive ? 'active' : ''}" data-gpu="all" role="tab" aria-selected="${isAllActive}">
        <div class="gpu-chip-header">
          <span class="gpu-chip-title">全部 GPU (3卡池)</span>
          <span class="badge badge-gpu-all">${activeCards}/${devices.length} 卡活跃</span>
        </div>
        <div class="gpu-chip-body">
          <span class="gpu-chip-vram">${allVram.used_gb.toFixed(1)} / ${allVram.total_gb.toFixed(1)} GB</span>
          <span>峰值负载 ${maxUtil}%</span>
        </div>
        <div class="gpu-chip-track">
          <div class="gpu-chip-bar ${getStatusFillClass(allVram.used_pct)}" style="width: ${Math.min(100, Math.max(0, allVram.used_pct))}%"></div>
        </div>
      </div>
    `;

    devices.forEach(dev => {
      const isDevActive = state.selectedGpu === String(dev.index);
      const devVram = dev.vram || {};
      const usedGb = devVram.used_gb != null ? devVram.used_gb.toFixed(1) : '--';
      const totalGb = devVram.total_gb != null ? devVram.total_gb.toFixed(1) : '83.6';
      const devPct = devVram.used_pct != null ? devVram.used_pct : 0;
      const utilPct = dev.utilization_pct != null ? dev.utilization_pct : 0;
      const tempC = dev.temperature_c != null ? dev.temperature_c : '--';
      const procCount = (dev.processes || []).length;

      let badgeClass = 'badge-safe';
      let badgeLabel = '空闲';
      if (utilPct >= 90) { badgeClass = 'badge-crit'; badgeLabel = `${utilPct}% 满载`; }
      else if (utilPct >= 50) { badgeClass = 'badge-warn'; badgeLabel = `${utilPct}% 负载`; }
      else if (utilPct > 0 || procCount > 0) { badgeClass = 'badge-info'; badgeLabel = `${utilPct}% (${procCount}进程)`; }

      ribbonHtml += `
        <div class="gpu-device-chip ${isDevActive ? 'active' : ''}" data-gpu="${dev.index}" role="tab" aria-selected="${isDevActive}">
          <div class="gpu-chip-header">
            <span class="gpu-chip-title">GPU ${dev.index}: RTX 6000D</span>
            <span class="badge ${badgeClass}">${badgeLabel}</span>
          </div>
          <div class="gpu-chip-body">
            <span class="gpu-chip-vram">${usedGb} / ${totalGb} GB</span>
            <span>${tempC}℃</span>
          </div>
          <div class="gpu-chip-track">
            <div class="gpu-chip-bar ${getStatusFillClass(devPct)}" style="width: ${Math.min(100, Math.max(0, devPct))}%"></div>
          </div>
        </div>
      `;
    });

    el.gpuDevicesRibbon.innerHTML = ribbonHtml;

    // 2. 进程表格快捷卡槽过滤按钮
    if (el.gpuQuickFilters) {
      let filterBtnsHtml = `
        <button type="button" class="filter-chip ${isAllActive ? 'active' : ''}" data-gpu="all">全部卡槽</button>
      `;
      devices.forEach(dev => {
        const isDevActive = state.selectedGpu === String(dev.index);
        const procCount = (dev.processes || []).length;
        filterBtnsHtml += `
          <button type="button" class="filter-chip ${isDevActive ? 'active' : ''}" data-gpu="${dev.index}">GPU ${dev.index} (${procCount})</button>
        `;
      });
      el.gpuQuickFilters.innerHTML = filterBtnsHtml;
    }
  }

  function renderVramCard(gpu) {
    if (!gpu) return;
    const isAll = state.selectedGpu === 'all';
    let v;
    let cardTitle = 'GPU 显存 (VRAM)';

    if (isAll) {
      v = gpu.all_vram || gpu.vram;
      if (gpu.devices && gpu.devices.length > 1) {
        cardTitle = `GPU 显存池 (共 ${gpu.devices.length} 张卡)`;
      }
    } else {
      const targetDev = (gpu.devices || []).find(d => String(d.index) === String(state.selectedGpu));
      if (targetDev && targetDev.vram) {
        v = targetDev.vram;
        cardTitle = `GPU ${state.selectedGpu} 显存 (VRAM)`;
      } else {
        v = gpu.vram;
      }
    }

    if (!v) return;

    if (el.cardVramTitle) el.cardVramTitle.textContent = cardTitle;

    if (el.vramUsedGb) el.vramUsedGb.textContent = v.used_gb != null ? v.used_gb.toFixed(1) : '--';
    if (el.vramTotalGb) el.vramTotalGb.textContent = v.total_gb != null ? v.total_gb.toFixed(1) : '85.6';
    if (el.vramUsedMb) el.vramUsedMb.textContent = v.used_mb != null ? `${v.used_mb.toLocaleString()} MB` : '--';
    if (el.vramFreeGb) el.vramFreeGb.textContent = v.free_gb != null ? `${v.free_gb.toFixed(1)} GB` : '--';

    const pct = v.used_pct != null ? v.used_pct : 0;
    if (el.vramProgressBar) {
      el.vramProgressBar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
      el.vramProgressBar.className = 'progress-bar-fill ' + getStatusFillClass(pct);
    }

    if (el.vramPctBadge) {
      const { text, badgeClass } = getUsageBadgeMeta(pct);
      el.vramPctBadge.textContent = text;
      el.vramPctBadge.className = 'badge ' + badgeClass;
    }
  }

  function renderRamCard(mem, swap) {
    if (!mem) return;

    if (el.ramUsedGb) el.ramUsedGb.textContent = mem.used_gb != null ? mem.used_gb.toFixed(1) : '--';
    if (el.ramTotalGb) el.ramTotalGb.textContent = mem.total_gb != null ? mem.total_gb.toFixed(1) : '--';
    if (el.ramAvailGb) el.ramAvailGb.textContent = mem.available_gb != null ? `${mem.available_gb.toFixed(1)} GB` : '--';

    const bd = mem.breakdown || {};
    if (el.ramAnonShmemStat) {
      el.ramAnonShmemStat.textContent = `私有 Anon ${formatGbFromMb(bd.anon_mb, 1)} · 共享 Shmem ${formatGbFromMb(bd.shmem_mb, 1)}`;
    }

    if (swap && el.swapUsedStat) {
      if (!swap.total_mb || swap.total_mb <= 0) {
        el.swapUsedStat.textContent = '未启用';
      } else {
        const swapPct = swap.used_pct != null ? `${swap.used_pct.toFixed(1)}%` : '--';
        const swapUsedGb = swap.used_mb != null ? (swap.used_mb / 1024).toFixed(1) : '0.0';
        el.swapUsedStat.textContent = `${swapPct} (${swapUsedGb} GB)`;
      }
    }

    const pct = mem.used_pct != null ? mem.used_pct : 0;
    if (el.ramProgressBar) {
      el.ramProgressBar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
      el.ramProgressBar.className = 'progress-bar-fill ' + getStatusFillClass(pct);
    }

    if (el.ramPctBadge) {
      const { text, badgeClass } = getUsageBadgeMeta(pct);
      el.ramPctBadge.textContent = text;
      el.ramPctBadge.className = 'badge ' + badgeClass;
    }
  }

  function renderGpuUtilCard(gpu, system) {
    const isAll = state.selectedGpu === 'all';
    let utilPct = 0;
    let cardTitle = 'GPU 核心利用率';

    if (isAll) {
      const devices = gpu?.devices || [];
      if (devices.length > 1) {
        cardTitle = 'GPU 核心利用率 (多卡)';
        utilPct = Math.max(...devices.map(d => d.utilization_pct || 0));
      } else {
        utilPct = gpu?.utilization_pct != null ? gpu.utilization_pct : 0;
      }
    } else {
      const targetDev = (gpu?.devices || []).find(d => String(d.index) === String(state.selectedGpu));
      cardTitle = `GPU ${state.selectedGpu} 核心利用率`;
      utilPct = targetDev?.utilization_pct != null ? targetDev.utilization_pct : 0;
    }

    if (el.cardUtilTitle) el.cardUtilTitle.textContent = cardTitle;
    if (el.gpuUtilPct) el.gpuUtilPct.textContent = utilPct;

    if (el.utilProgressBar) {
      el.utilProgressBar.style.width = `${Math.min(100, Math.max(0, utilPct))}%`;
      el.utilProgressBar.className = 'progress-bar-fill ' + getStatusFillClass(utilPct);
    }

    if (el.gpuUtilBadge) {
      let badgeText = '空闲';
      let badgeClass = 'badge-safe';
      if (utilPct >= 90) {
        badgeText = '满载运算';
        badgeClass = 'badge-crit';
      } else if (utilPct >= 70) {
        badgeText = '高负载';
        badgeClass = 'badge-warn';
      } else if (utilPct >= 15) {
        badgeText = '计算中';
        badgeClass = 'badge-info';
      }

      if (isAll && gpu?.devices && gpu.devices.length > 1) {
        const activeCards = gpu.devices.filter(d => (d.utilization_pct || 0) > 0).length;
        el.gpuUtilBadge.textContent = `峰值 ${utilPct}% (${activeCards}/${gpu.devices.length}卡活跃)`;
      } else {
        el.gpuUtilBadge.textContent = `${utilPct}% ${badgeText}`;
      }
      el.gpuUtilBadge.className = 'badge ' + badgeClass;
    }

    if (el.gpuDriverVersion) el.gpuDriverVersion.textContent = gpu?.driver_version || '--';

    if (el.sysLoad) {
      if (system?.load_average && system.load_average.length > 0) {
        el.sysLoad.textContent = system.load_average.map(n => Number(n).toFixed(2)).join(', ');
      } else {
        el.sysLoad.textContent = '--';
      }
    }
  }

  function renderGpuTempCard(gpu, system) {
    const isAll = state.selectedGpu === 'all';
    let tempC = null;
    let cardTitle = 'GPU 核心温度';

    if (isAll) {
      const devices = gpu?.devices || [];
      if (devices.length > 1) {
        cardTitle = 'GPU 核心温度 (最高)';
        tempC = Math.max(...devices.map(d => d.temperature_c || 0));
      } else {
        tempC = gpu?.temperature_c != null ? gpu.temperature_c : null;
      }
    } else {
      const targetDev = (gpu?.devices || []).find(d => String(d.index) === String(state.selectedGpu));
      cardTitle = `GPU ${state.selectedGpu} 核心温度`;
      tempC = targetDev?.temperature_c != null ? targetDev.temperature_c : null;
    }

    if (el.cardTempTitle) el.cardTempTitle.textContent = cardTitle;
    if (el.gpuTempC) el.gpuTempC.textContent = tempC != null ? tempC : '--';

    if (el.tempProgressBar) {
      const pct = tempC != null ? (tempC / 100) * 100 : 0;
      el.tempProgressBar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
      if (tempC >= 80) el.tempProgressBar.className = 'progress-bar-fill fill-crit';
      else if (tempC >= 60) el.tempProgressBar.className = 'progress-bar-fill fill-warn';
      else el.tempProgressBar.className = 'progress-bar-fill fill-safe';
    }

    if (el.gpuTempBadge && tempC != null) {
      if (tempC >= 80) {
        el.gpuTempBadge.textContent = '过热告警 (>=80℃)';
        el.gpuTempBadge.className = 'badge badge-crit';
      } else if (tempC >= 60) {
        el.gpuTempBadge.textContent = '正常 (60-79℃)';
        el.gpuTempBadge.className = 'badge badge-warn';
      } else {
        el.gpuTempBadge.textContent = '优良 (<60℃)';
        el.gpuTempBadge.className = 'badge badge-safe';
      }
    }

    if (el.gpuTempDesc) {
      if (tempC >= 80) el.gpuTempDesc.textContent = '风扇高转速告警';
      else if (tempC >= 60) el.gpuTempDesc.textContent = '温控正常稳定';
      else el.gpuTempDesc.textContent = '状态优良低温';
    }

    if (el.sysUptime) el.sysUptime.textContent = formatUptime(system?.uptime_seconds);
  }

  /**
   * 渲染按用户聚合的内存 & Swap 配额条
   */
  function renderUserQuotaBar(mem) {
    if (!el.userQuotaBar) return;
    const users = mem?.by_user || [];
    if (users.length === 0) {
      el.userQuotaBar.innerHTML = '<span style="color: var(--text-dim); font-size: 12px;">暂无用户态内存聚合数据</span>';
      return;
    }

    const totalHostMb = mem.total_mb || 131072;
    const topUsers = users.slice(0, 8);
    const maxFootprint = Math.max(1, topUsers[0].total_mb || 1);

    const chipsHtml = topUsers.map(u => {
      const isActive = state.memUserFilter === u.user;
      const rssPctBar = Math.min(100, (u.rss_mb / maxFootprint) * 100);
      const swapPctBar = Math.min(100 - rssPctBar, (u.swap_mb / maxFootprint) * 100);
      const sharePct = ((u.rss_mb / totalHostMb) * 100).toFixed(1);
      const swapExact = u.swap_exact !== false;
      const swapLabel = u.swap_mb > 0 && !swapExact
        ? `Swap ~${formatMemMb(u.swap_mb)}`
        : `Swap ${formatMemMb(u.swap_mb, '0')}`;
      const swapDetailHint = swapExact
        ? (u.vm_swap_mb && u.vm_swap_mb > u.swap_mb + 10
          ? `已按 SwapPss 去重 fork 共享页（原 VmSwap ${formatMemMb(u.vm_swap_mb)}）`
          : '已按内核 SwapPss 精确统计')
        : '非本账户进程受内核权限限制使用 VmSwap 累加，可能含 fork 共享页';

      return `
        <button type="button" class="user-quota-chip ${isActive ? 'active' : ''}" data-user="${escapeHtml(u.user)}" title="点击筛选用户 ${escapeHtml(u.user)} 的进程 (占物理内存 ${sharePct}%；${swapDetailHint})">
          <div class="user-chip-top">
            <span class="user-chip-name">${escapeHtml(u.user)}</span>
            <span class="user-chip-count">${u.process_count} 进程</span>
          </div>
          <div class="user-chip-stats">
            <span>RSS ${formatMemMb(u.rss_mb)}</span>
            <span style="color: ${u.swap_mb > 512 ? '#fbbf24' : 'var(--text-dim)'};">${swapLabel}</span>
          </div>
          <div class="user-chip-bar-track">
            <div class="user-chip-rss" style="width: ${rssPctBar.toFixed(1)}%"></div>
            <div class="user-chip-swap" style="width: ${swapPctBar.toFixed(1)}%"></div>
          </div>
        </button>
      `;
    }).join('');

    el.userQuotaBar.innerHTML = chipsHtml;
  }

  /**
   * 渲染系统内存与 Swap 占用进程清单表格
   */
  function renderMemoryProcessTable(mem, swap) {
    if (!mem) return;
    const allProcesses = mem.processes || [];
    const byUserList = mem.by_user || [];
    const summary = mem.process_summary || {};
    const totalHostMb = mem.total_mb || 1;
    const totalSwapMb = swap?.total_mb || 1;

    // 1. 快捷过滤器 + 用户筛选 + 搜索框
    let filtered = allProcesses;
    const hasSearch = Boolean(state.memSearchQuery.trim());
    const hasQuick = state.memQuickFilter !== 'all';
    const hasUser = Boolean(state.memUserFilter);

    if (hasUser) {
      filtered = filtered.filter(p => p.user === state.memUserFilter);
    }

    if (state.memQuickFilter === 'swap') {
      filtered = filtered.filter(p => (p.swap_mb || 0) >= 100);
    } else if (state.memQuickFilter === 'python') {
      filtered = filtered.filter(p => {
        const text = `${p.name || ''} ${p.cmdline || ''}`.toLowerCase();
        return text.includes('python') || text.includes('llama') || text.includes('torch') || text.includes('comfy');
      });
    } else if (state.memQuickFilter === 'oom') {
      filtered = filtered.filter(p => (p.oom_score || 0) >= 600);
    }

    filtered = filterByTokens(filtered, state.memSearchQuery);

    // 2. 排序
    const memNumericCols = new Set(['pid', 'rss_mb', 'swap_mb', 'vsz_mb', 'ram_pct', 'total_mb', 'oom_score']);
    const sorted = sortProcessList(filtered, state.memSortColumn, state.memSortDirection, memNumericCols);

    // 3. 更新汇总与徽章（根据筛选上下文动态切换精确汇总）
    if (el.memProcessCountBadge) {
      const totalActive = summary.active_count || allProcesses.length;
      if (hasSearch || hasUser || hasQuick) {
        el.memProcessCountBadge.textContent = `匹配 ${sorted.length} / Top ${allProcesses.length} (全机 ${totalActive})`;
      } else {
        el.memProcessCountBadge.textContent = `Top ${allProcesses.length} / 全机 ${totalActive} 个进程`;
      }
    }

    let footerRssMb = summary.total_rss_mb || 0;
    let footerSwapMb = summary.total_swap_mb || 0;
    let footerPrefix = '全机用户态进程汇总:';

    if (hasUser && !hasSearch && !hasQuick) {
      const userRec = byUserList.find(u => u.user === state.memUserFilter);
      if (userRec) {
        footerPrefix = `用户 ${state.memUserFilter} 汇总 (${userRec.process_count} 进程):`;
        footerRssMb = userRec.rss_mb || 0;
        footerSwapMb = userRec.swap_mb || 0;
      } else {
        footerPrefix = `用户 ${state.memUserFilter} (Top 匹配 ${sorted.length} 项):`;
        footerRssMb = sorted.reduce((acc, p) => acc + (p.rss_mb || 0), 0);
        footerSwapMb = sorted.reduce((acc, p) => acc + (p.swap_mb || 0), 0);
      }
    } else if (hasSearch || hasQuick) {
      footerPrefix = `当前筛选汇总 (${sorted.length} 项):`;
      footerRssMb = sorted.reduce((acc, p) => acc + (p.rss_mb || 0), 0);
      footerSwapMb = sorted.reduce((acc, p) => acc + (p.swap_mb || 0), 0);
    }

    if (el.memFooterPrefix) {
      el.memFooterPrefix.textContent = footerPrefix;
    }
    if (el.totalTrackedRss) {
      el.totalTrackedRss.textContent = `${formatMemMb(footerRssMb)} RSS`;
    }
    if (el.totalTrackedSwap) {
      el.totalTrackedSwap.textContent = `${formatMemMb(footerSwapMb)} Swap`;
    }
    if (el.activeUserFilterLabel) {
      el.activeUserFilterLabel.textContent = hasUser && (hasSearch || hasQuick)
        ? `· 用户: ${state.memUserFilter}`
        : '';
    }
    if (el.memFooterHint) {
      const fbCount = summary.swap_fallback_procs || 0;
      el.memFooterHint.textContent = fbCount > 0
        ? '提示：物理内存与 Swap 独立统计；本账户 Swap 已按 SwapPss 去重，其他用户含 VmSwap 共享'
        : '提示：物理内存与 Swap 独立统计；Swap 已按 SwapPss 精确去重 fork 共享页';
    }

    // 4. 空状态或表格行
    if (sorted.length === 0) {
      if (el.memProcessTable) el.memProcessTable.classList.add('hidden');
      if (el.memEmptyState) el.memEmptyState.classList.remove('hidden');
      if (el.memEmptyMessage) {
        el.memEmptyMessage.textContent = hasSearch
          ? `未找到匹配 "${state.memSearchQuery.trim()}" 的内存进程`
          : '当前筛选条件下无匹配内存进程';
      }
      if (el.memEmptySubmessage) {
        el.memEmptySubmessage.textContent = hasUser
          ? `可再次点击上方 "${state.memUserFilter}" 胶囊取消用户筛选，或切换快捷过滤标签`
          : '请调整搜索关键词或切换快捷过滤标签';
      }
      return;
    }

    if (el.memEmptyState) el.memEmptyState.classList.add('hidden');
    if (el.memProcessTable) el.memProcessTable.classList.remove('hidden');

    const rowsHtml = sorted.map(proc => {
      const pid = proc.pid != null ? proc.pid : '--';
      const user = proc.user || 'unknown';
      const name = proc.name || 'unknown';
      const procState = proc.state || 'S';
      const threads = proc.threads || 1;
      const cmdline = proc.cmdline || name;
      const rssMb = proc.rss_mb || 0;
      const swapMb = proc.swap_mb || 0;
      const vszMb = proc.vsz_mb || 0;
      const shmemMb = proc.shmem_mb || 0;
      const ramPct = proc.ram_pct != null ? proc.ram_pct : (totalHostMb > 0 ? (rssMb / totalHostMb) * 100 : 0);
      const ramPctFormatted = ramPct.toFixed(1);
      const oomScore = proc.oom_score || 0;

      const isSelf = user === 'shangguanrz';
      const userTagClass = isSelf ? 'user-tag self-user' : 'user-tag';
      const swapClass = swapMb >= 1024 ? 'mem-swap-crit' : swapMb >= 100 ? 'mem-swap-warn' : 'mem-swap-none';
      const oomClass = oomScore >= 700 ? 'oom-high' : oomScore >= 500 ? 'oom-med' : 'oom-low';
      const stateClass = (procState === 'D' || procState === 'Z') ? `proc-meta-sub proc-state-${procState}` : 'proc-meta-sub';

      const rssTitle = proc.pss_mb != null
        ? `私有 Anon: ${formatMemMb(proc.anon_mb)} | 共享 Shmem: ${formatMemMb(shmemMb)} | 均摊 PSS: ${formatMemMb(proc.pss_mb)}`
        : `私有 Anon: ${formatMemMb(proc.anon_mb)} | 共享 Shmem: ${formatMemMb(shmemMb)}`;
      const shmemBadge = shmemMb >= 64
        ? `<span class="proc-meta-sub" title="其中含共享映射/tmpfs 内存 ${formatMemMb(shmemMb)}">共享 ${escapeHtml(formatMemMb(shmemMb))}</span>`
        : '';

      const swapSharePct = totalSwapMb > 0 ? ((swapMb / totalSwapMb) * 100).toFixed(1) : '0.0';
      const swapTitle = proc.swap_is_pss
        ? (proc.vm_swap_mb > swapMb + 10
          ? `精确分摊 SwapPss: ${formatMemMb(swapMb)}（占 Swap ${swapSharePct}%；原映射 VmSwap 含 fork 共享 ${formatMemMb(proc.vm_swap_mb)}）`
          : `精确分摊 SwapPss: ${formatMemMb(swapMb)}（占全局 Swap 容量 ${swapSharePct}%）`)
        : (swapMb > 0
          ? `内核 VmSwap: ${formatMemMb(swapMb)}（占全局 Swap 容量 ${swapSharePct}%；非本账户进程受权限限制未用 SwapPss 去重）`
          : '未使用 Swap');

      return `
        <tr>
          <td class="cell-pid">${escapeHtml(pid)}</td>
          <td class="cell-user">
            <span class="${userTagClass}">${escapeHtml(user)}</span>
          </td>
          <td class="cell-name">
            <span>${escapeHtml(name)}</span>
            <span class="${stateClass}" title="进程状态 ${escapeHtml(procState)}, 线程数 ${threads}">${escapeHtml(procState)}·${threads}t</span>
          </td>
          <td class="cell-cmdline" title="${escapeHtml(cmdline)}">${escapeHtml(cmdline)}</td>
          <td class="cell-num" title="${escapeHtml(rssTitle)}">${escapeHtml(formatMemMb(rssMb))}${shmemBadge}</td>
          <td class="cell-num ${swapClass}" title="${escapeHtml(swapTitle)}">${escapeHtml(formatMemMb(swapMb, '-'))}</td>
          <td class="cell-num" style="color: var(--text-secondary);">${escapeHtml(formatMemMb(vszMb))}</td>
          <td class="cell-bar">
            <div class="micro-bar-wrapper">
              <div class="micro-bar-track" title="纯物理内存 RSS 占整机 RAM 比例: ${ramPctFormatted}%（不含 Swap）">
                <div class="micro-bar-fill fill-mem" style="width: ${Math.min(100, Math.max(0.5, ramPct)).toFixed(1)}%"></div>
              </div>
              <span class="micro-bar-text">${ramPctFormatted}%</span>
            </div>
          </td>
          <td>
            <span class="oom-badge ${oomClass}" title="Linux OOM Killer 评分 (0-1000)">${oomScore}</span>
          </td>
        </tr>
      `;
    }).join('');

    if (el.memProcessTableBody) {
      el.memProcessTableBody.innerHTML = rowsHtml;
    }

    updateSortHeaders(el.memTableHeaders, state.memSortColumn, state.memSortDirection);
  }

  /**
   * 渲染 GPU 进程表格及汇总信息 (支持多卡槽筛选与聚合)
   */
  function renderProcessTable(gpu) {
    let allProcesses = gpu?.processes || [];
    const isAll = state.selectedGpu === 'all';

    // 1. 卡槽过滤
    if (!isAll) {
      allProcesses = allProcesses.filter(p => String(p.gpu_index ?? 0) === String(state.selectedGpu));
    }

    // 2. 计算卡槽基准显存
    let totalGpuMb = 85651;
    if (isAll) {
      totalGpuMb = gpu?.all_vram?.total_mb || (gpu?.vram?.total_mb ? gpu.vram.total_mb * (gpu?.device_count || 3) : 256953);
    } else {
      const targetDev = (gpu?.devices || []).find(d => String(d.index) === String(state.selectedGpu));
      totalGpuMb = targetDev?.vram?.total_mb || gpu?.vram?.total_mb || 85651;
    }

    // 3. 关键词过滤与排序
    const filtered = filterByTokens(allProcesses, state.searchQuery);
    const gpuNumCols = new Set(['pid', 'vram_used_mb', 'vram_pct', 'gpu_index']);
    const sorted = sortProcessList(filtered, state.sortColumn, state.sortDirection, gpuNumCols);

    let sumVramMb = 0;
    filtered.forEach(p => {
      sumVramMb += (p.vram_used_mb || 0);
    });

    if (el.totalTrackedVram) el.totalTrackedVram.textContent = formatMemMb(sumVramMb);

    if (el.totalGpuCapacity) {
      if (isAll) {
        const totalGb = gpu?.all_vram?.total_gb ? gpu.all_vram.total_gb.toFixed(2) : '250.93';
        el.totalGpuCapacity.textContent = `${totalGb} GB (3卡池)`;
      } else {
        const targetDev = (gpu?.devices || []).find(d => String(d.index) === String(state.selectedGpu));
        const totalGb = targetDev?.vram?.total_gb ? targetDev.vram.total_gb.toFixed(2) : '83.64';
        el.totalGpuCapacity.textContent = `${totalGb} GB (GPU ${state.selectedGpu})`;
      }
    }

    if (el.processCountBadge) {
      const scopeLabel = isAll ? '全机' : `GPU ${state.selectedGpu}`;
      if (state.searchQuery.trim()) {
        el.processCountBadge.textContent = `${scopeLabel} 匹配 ${sorted.length} / ${allProcesses.length} 个进程`;
      } else {
        el.processCountBadge.textContent = `${scopeLabel} ${allProcesses.length} 个活跃进程`;
      }
    }

    if (sorted.length === 0) {
      if (el.processTable) el.processTable.classList.add('hidden');
      if (el.emptyState) el.emptyState.classList.remove('hidden');

      if (allProcesses.length === 0) {
        const cardDesc = isAll ? '全机 3 张显卡' : `GPU ${state.selectedGpu}`;
        if (el.emptyStateMessage) el.emptyStateMessage.textContent = `当前 ${cardDesc} 无活跃计算进程`;
        if (el.emptyStateSubmessage) el.emptyStateSubmessage.textContent = `${cardDesc} 处于空闲待命状态，显存均可用于新计算任务`;
      } else {
        if (el.emptyStateMessage) el.emptyStateMessage.textContent = `无匹配进程 (当前范围共 ${allProcesses.length} 个进程)`;
        if (el.emptyStateSubmessage) el.emptyStateSubmessage.textContent = `未找到包含关键词 "${escapeHtml(state.searchQuery)}" 的计算进程，请尝试其他关键词`;
      }
      return;
    }

    if (el.emptyState) el.emptyState.classList.add('hidden');
    if (el.processTable) el.processTable.classList.remove('hidden');

    const rowsHtml = sorted.map(proc => {
      const pid = proc.pid != null ? proc.pid : '--';
      const gpuIdx = proc.gpu_index != null ? proc.gpu_index : 0;
      const user = proc.user || 'unknown';
      const name = proc.name || 'unknown';
      const cmdline = proc.cmdline || name;
      const vramMb = proc.vram_used_mb || 0;
      const vramText = formatMemMb(vramMb);
      const vramPct = proc.vram_pct != null
        ? proc.vram_pct
        : (85651 > 0 ? (vramMb / 85651) * 100 : 0);
      const vramPctFormatted = vramPct.toFixed(1);

      const isSelf = user === 'shangguanrz';
      const userTagClass = isSelf ? 'user-tag self-user' : 'user-tag';

      return `
        <tr>
          <td class="cell-pid">${escapeHtml(pid)}</td>
          <td class="cell-gpu">
            <span class="badge-gpu badge-gpu-${gpuIdx}">GPU ${gpuIdx}</span>
          </td>
          <td class="cell-user">
            <span class="${userTagClass}">${escapeHtml(user)}</span>
          </td>
          <td class="cell-name">${escapeHtml(name)}</td>
          <td class="cell-cmdline" title="${escapeHtml(cmdline)}">${escapeHtml(cmdline)}</td>
          <td class="cell-vram font-mono">${escapeHtml(vramText)}</td>
          <td class="cell-bar">
            <div class="micro-bar-wrapper">
              <div class="micro-bar-track" title="占单卡显存比例 ${vramPctFormatted}%">
                <div class="micro-bar-fill" style="width: ${Math.min(100, Math.max(0, vramPct))}%"></div>
              </div>
              <span class="micro-bar-text">${vramPctFormatted}%</span>
            </div>
          </td>
        </tr>
      `;
    }).join('');

    if (el.processTableBody) el.processTableBody.innerHTML = rowsHtml;
    updateSortHeaders(el.tableHeaders, state.sortColumn, state.sortDirection);
  }

  // =========================================================================
  // 事件交互绑定 (Event Handlers)
  // =========================================================================

  function initEventListeners() {
    // 视图模式切换
    el.viewTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        setViewMode(tab.getAttribute('data-view') || 'all');
      });
    });

    window.addEventListener('hashchange', initViewModeFromHash);

    // 刷新频率
    if (el.intervalSelect) {
      el.intervalSelect.addEventListener('change', (e) => {
        state.pollingInterval = parseInt(e.target.value, 10) || 2000;
        if (!state.isPaused) scheduleNextFetch();
      });
    }

    if (el.btnPause) el.btnPause.addEventListener('click', togglePauseResume);
    if (el.btnRefresh) el.btnRefresh.addEventListener('click', () => fetchMetrics());

    // 内存用户配额点击筛选
    if (el.userQuotaBar) {
      el.userQuotaBar.addEventListener('click', (e) => {
        const chip = e.target.closest('.user-quota-chip');
        if (!chip) return;
        const clickedUser = chip.getAttribute('data-user') || '';
        state.memUserFilter = state.memUserFilter === clickedUser ? '' : clickedUser;
        if (state.currentData?.memory) {
          renderUserQuotaBar(state.currentData.memory);
          renderMemoryProcessTable(state.currentData.memory, state.currentData.swap);
        }
      });
    }

    // 内存快捷过滤器
    el.memQuickFilters.forEach(btn => {
      btn.addEventListener('click', () => {
        const filterVal = btn.getAttribute('data-filter') || 'all';
        state.memQuickFilter = filterVal;
        el.memQuickFilters.forEach(b => b.classList.toggle('active', b === btn));
        if (state.currentData?.memory) {
          renderMemoryProcessTable(state.currentData.memory, state.currentData.swap);
        }
      });
    });

    // 内存搜索输入
    if (el.memProcessSearch) {
      el.memProcessSearch.addEventListener('input', (e) => {
        state.memSearchQuery = e.target.value;
        if (el.memSearchClearBtn) {
          el.memSearchClearBtn.classList.toggle('hidden', state.memSearchQuery.length === 0);
        }
        if (state.currentData?.memory) {
          renderMemoryProcessTable(state.currentData.memory, state.currentData.swap);
        }
      });
    }

    if (el.memSearchClearBtn) {
      el.memSearchClearBtn.addEventListener('click', () => {
        if (el.memProcessSearch) {
          el.memProcessSearch.value = '';
          el.memProcessSearch.focus();
        }
        state.memSearchQuery = '';
        el.memSearchClearBtn.classList.add('hidden');
        if (state.currentData?.memory) {
          renderMemoryProcessTable(state.currentData.memory, state.currentData.swap);
        }
      });
    }

    // 内存表头排序
    el.memTableHeaders.forEach(th => {
      th.addEventListener('click', () => {
        const col = th.getAttribute('data-sort');
        if (!col) return;
        const numCols = new Set(['pid', 'rss_mb', 'swap_mb', 'vsz_mb', 'ram_pct', 'total_mb', 'oom_score']);
        if (state.memSortColumn === col) {
          state.memSortDirection = state.memSortDirection === 'asc' ? 'desc' : 'asc';
        } else {
          state.memSortColumn = col;
          state.memSortDirection = numCols.has(col) ? 'desc' : 'asc';
        }
        if (state.currentData?.memory) {
          renderMemoryProcessTable(state.currentData.memory, state.currentData.swap);
        }
      });
    });

    // GPU 多卡槽切换 (Ribbon 与 快速筛选按钮)
    if (el.gpuDevicesRibbon) {
      el.gpuDevicesRibbon.addEventListener('click', (e) => {
        const chip = e.target.closest('.gpu-device-chip');
        if (!chip) return;
        const gpuIdx = chip.getAttribute('data-gpu') || 'all';
        state.selectedGpu = (state.selectedGpu === gpuIdx && gpuIdx !== 'all') ? 'all' : gpuIdx;
        if (state.currentData) {
          renderGpuRibbon(state.currentData.gpu);
          renderVramCard(state.currentData.gpu);
          renderGpuUtilCard(state.currentData.gpu, state.currentData.system);
          renderGpuTempCard(state.currentData.gpu, state.currentData.system);
          renderProcessTable(state.currentData.gpu);
        }
      });
    }

    if (el.gpuQuickFilters) {
      el.gpuQuickFilters.addEventListener('click', (e) => {
        const btn = e.target.closest('.filter-chip');
        if (!btn) return;
        const gpuIdx = btn.getAttribute('data-gpu') || 'all';
        state.selectedGpu = gpuIdx;
        if (state.currentData) {
          renderGpuRibbon(state.currentData.gpu);
          renderVramCard(state.currentData.gpu);
          renderGpuUtilCard(state.currentData.gpu, state.currentData.system);
          renderGpuTempCard(state.currentData.gpu, state.currentData.system);
          renderProcessTable(state.currentData.gpu);
        }
      });
    }

    // GPU 搜索与排序
    if (el.processSearch) {
      el.processSearch.addEventListener('input', (e) => {
        state.searchQuery = e.target.value;
        if (el.searchClearBtn) {
          el.searchClearBtn.classList.toggle('hidden', state.searchQuery.length === 0);
        }
        if (state.currentData?.gpu) renderProcessTable(state.currentData.gpu);
      });
    }

    if (el.searchClearBtn) {
      el.searchClearBtn.addEventListener('click', () => {
        if (el.processSearch) {
          el.processSearch.value = '';
          el.processSearch.focus();
        }
        state.searchQuery = '';
        el.searchClearBtn.classList.add('hidden');
        if (state.currentData?.gpu) renderProcessTable(state.currentData.gpu);
      });
    }

    el.tableHeaders.forEach(th => {
      th.addEventListener('click', () => {
        const col = th.getAttribute('data-sort');
        if (!col) return;
        const numCols = new Set(['vram_used_mb', 'vram_pct', 'pid', 'gpu_index']);
        if (state.sortColumn === col) {
          state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
          state.sortColumn = col;
          state.sortDirection = numCols.has(col) ? 'desc' : 'asc';
        }
        if (state.currentData?.gpu) renderProcessTable(state.currentData.gpu);
      });
    });

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        clearTimeout(state.timerId);
      } else if (!state.isPaused) {
        fetchMetrics();
      }
    });
  }

  function togglePauseResume() {
    state.isPaused = !state.isPaused;
    if (state.isPaused) {
      clearTimeout(state.timerId);
      if (el.btnPause) el.btnPause.classList.add('paused');
      if (el.btnPauseText) el.btnPauseText.textContent = '继续';
      const pauseIcon = el.btnPause?.querySelector('.btn-icon');
      if (pauseIcon) pauseIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    } else {
      if (el.btnPause) el.btnPause.classList.remove('paused');
      if (el.btnPauseText) el.btnPauseText.textContent = '暂停';
      const pauseIcon = el.btnPause?.querySelector('.btn-icon');
      if (pauseIcon) {
        pauseIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
      }
      fetchMetrics();
    }
  }

  // =========================================================================
  // 应用初始化启动 (Application Bootstrap)
  // =========================================================================
  function init() {
    initViewModeFromHash();
    initEventListeners();
    updateSortHeaders(el.tableHeaders, state.sortColumn, state.sortDirection);
    updateSortHeaders(el.memTableHeaders, state.memSortColumn, state.memSortDirection);
    fetchMetrics();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
