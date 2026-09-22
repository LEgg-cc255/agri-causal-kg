/* ==========================================================================
   农业因果知识图谱系统 - 前端逻辑（原生JS）
   - 标签页切换
   - API 调用（fetch + JSON）
   - vis.js Network 因果图渲染（按 entity_class 着色）
   - 路径搜索结果展示（A → B → C 链式）
   - 端到端问答 4 阶段计时展示
   - 错误处理与加载状态
   ========================================================================== */

/* ---------- 全局配置 ---------- */
const API_BASE = '';  // 同源，无需额外前缀

// 子类中文名 → 颜色映射（前端按3父类着色，后端本体是嵌套细粒度子类）
// 环境因子3父类用红色系深浅区分，生物胁迫4子类用紫色系，防治措施橙色，功能症状5子类蓝色系
const SUB_CLASS_COLOR_MAP = {
    // 环境因子（红色系3父类）
    '天气': '#e74c3c',
    '土壤': '#c0392b',
    '水质': '#922b21',
    // 生物胁迫（紫色系4子类）
    '病害': '#9b59b6',
    '虫害': '#8e44ad',
    '虫害/害虫': '#8e44ad',
    '病原': '#6c3483',
    '杂草竞争': '#5b2c6f',
    // 防治措施（橙色）
    '防治措施': '#f39c12',
    // 功能症状（蓝色系深浅区分5个子类）
    '生理症状': '#3498db',
    '生殖症状': '#2980b9',
    '产量症状': '#1abc9c',
    '品质症状': '#16a085',
    '胁迫响应症状': '#48c9b0',
    // 作物（绿色系，作物主体节点）
    '作物': '#27ae60',
    '解剖症状': '#5dade2',
};

// 环境因子细粒度子类名 → 父类名（着色用父类，tooltip保留细粒度）
// 仅用于 getColorByNode 的查找转换，不影响展示
const ENVF_PARENT_MAP = {
    // 天气7细粒度 → 天气
    '空气温度': '天气', '空气湿度': '天气', '降水': '天气',
    '光照': '天气', '风速': '天气', 'CO₂浓度': '天气', '气压': '天气',
    '天气': '天气',
    // 土壤4细粒度 → 土壤
    '土壤温度': '土壤', '土壤湿度': '土壤', '土壤养分': '土壤', '土壤pH': '土壤',
    '土壤': '土壤',
    // 水质3细粒度 → 水质
    '水质pH': '水质', '水质盐分': '水质', '水质重金属': '水质',
    '水质': '水质',
    // 模糊/顶层节点名也一并映射
    'EnvFactor': '天气',
};

// 大类 → 回退颜色（sub_class_name 缺失时用）
const CLASS_COLOR_MAP = {
    '环境因子': '#e74c3c',
    '功能症状': '#3498db',
    '防治措施': '#f39c12',
    '生物胁迫': '#9b59b6',
    '作物': '#27ae60',
};

// 默认灰色（其他/未分类）
const DEFAULT_COLOR = '#95a5a6';

// 主动学习审核面板状态颜色
const CLASS_STATUS_COLOR = {
    'KNOWN': '#3498db', 'FORBIDDEN': '#e74c3c', 'NEW_UNKNOWN': '#e67e22',
    'SELF_LOOP': '#95a5a6', 'LOW_CONF': '#7f8c8d', 'CONFLICT': '#8e44ad',
};

// vis.js Network 实例
let networkInstance = null;
// 图数据缓存（后端原始数据）
let graphDataCache = { nodes: [], edges: [] };
// 节点大类+子类可见性过滤状态
// graphFilter = { '环境因子': {visible: true, subs: {'天气':true, '土壤':true, '水质':true}}, ... }
let graphFilter = {};
let graphDataSet = null;

/* ==========================================================================
   工具函数
   ========================================================================== */

/** HTML 转义，防止注入 */
function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/** 根据节点的 sub_class_name / group 获取颜色（子类优先） */
function getColorByNode(node) {
    // 1. 优先用 sub_class_name 查颜色（环境因子细粒度→父类名转换）
    let subName = node.sub_class_name || '';
    // 环境因子：细粒度子类名先转父类名再查颜色
    if (subName && ENVF_PARENT_MAP[subName]) {
        subName = ENVF_PARENT_MAP[subName];
    }
    if (subName && SUB_CLASS_COLOR_MAP[subName]) {
        return SUB_CLASS_COLOR_MAP[subName];
    }
    // 2. 回退：用 group（大类）查颜色
    const group = node.group || '';
    if (group && CLASS_COLOR_MAP[group]) {
        return CLASS_COLOR_MAP[group];
    }
    // 3. 模糊匹配（包含关键词）
    if (group) {
        for (const key in CLASS_COLOR_MAP) {
            if (group.includes(key) || key.includes(group)) {
                return CLASS_COLOR_MAP[key];
            }
        }
    }
    return DEFAULT_COLOR;
}

/** 显示 Toast 提示 */
function showToast(message, type = 'error') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = 'toast show ' + (type === 'success' ? 'success' : '');
    setTimeout(() => {
        toast.className = 'toast';
    }, 3000);
}

/** 通用 fetch 封装 */
async function callApi(url, options = {}) {
    const defaultOptions = {
        headers: { 'Content-Type': 'application/json' },
    };
    const merged = { ...defaultOptions, ...options };
    const response = await fetch(API_BASE + url, merged);
    if (!response.ok) {
        let errMsg = `HTTP ${response.status}`;
        try {
            const errData = await response.json();
            errMsg = errData.detail || errData.message || errMsg;
        } catch (e) { /* 忽略解析失败 */ }
        throw new Error(errMsg);
    }
    return response.json();
}

/** 设置加载状态 */
function setLoading(loadingId, isLoading) {
    const el = document.getElementById(loadingId);
    if (el) el.style.display = isLoading ? 'flex' : 'none';
}

/** 设置按钮禁用状态 */
function setBtnDisabled(btnId, disabled) {
    const btn = document.getElementById(btnId);
    if (btn) btn.disabled = disabled;
}

/* ==========================================================================
   标签页切换
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // 绑定标签页切换
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // 初始化：加载统计信息
    loadStats();
    // 渲染图例
    renderLegend();
    // Enter 键触发搜索/问答
    document.getElementById('searchQuestion').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') runSearch();
    });
    document.getElementById('answerQuestion').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') runAnswer();
    });
});

/** 切换标签页 */
function switchTab(tabName) {
    // 更新按钮状态
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.tab-btn[data-tab="${tabName}"]`).classList.add('active');
    // 更新面板显示
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');

    // 切换到因果图标签时自动加载
    if (tabName === 'graph' && !networkInstance) {
        loadGraph();
    }
}

/* ==========================================================================
   顶部统计信息
   ========================================================================== */

async function loadStats() {
    try {
        const data = await callApi('/api/stats');
        if (!data.ok) {
            showToast('Neo4j 连接失败');
            return;
        }
        document.getElementById('statNodes').textContent = data.nodes ?? '-';
        document.getElementById('statEdges').textContent = data.causal_edges ?? '-';
        document.getElementById('statForbid').textContent = data.forbidden_edges ?? '-';
        document.getElementById('statVersion').textContent = data.neo4j_version || '-';

        // 大类分布
        const dist = data.class_distribution || {};
        const distStr = Object.entries(dist)
            .map(([k, v]) => `${k}:${v}`)
            .join(' / ');
        document.getElementById('statClassDist').textContent = distStr || '-';
    } catch (e) {
        showToast('加载统计信息失败: ' + e.message);
    }
}

/* ==========================================================================
   Tab 1: 因果抽取
   ========================================================================== */

async function runExtract() {
    const text = document.getElementById('extractText').value.trim();
    if (!text) {
        showToast('请输入待抽取文本');
        return;
    }
    setLoading('extractLoading', true);
    setBtnDisabled('btnExtract', true);
    const resultDiv = document.getElementById('extractResult');
    resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><span>抽取中...</span></div>';

    try {
        const data = await callApi('/api/extract', {
            method: 'POST',
            body: JSON.stringify({ text }),
        });
        renderExtractResult(data);
        // 立即调用主动学习 review（知识库对比+价值评分）
        runActiveReview(data, text);
    } catch (e) {
        resultDiv.innerHTML = `<div class="error-msg">抽取失败: ${escapeHtml(e.message)}</div>`;
    } finally {
        setLoading('extractLoading', false);
        setBtnDisabled('btnExtract', false);
    }
}

/** 主动学习：知识库对比+价值评分 */
async function runActiveReview(extractData, sourceText) {
    const container = document.getElementById('activeLearning');
    if (!container) {
        // 在抽取结果末尾追加主动学习面板
        const resultDiv = document.getElementById('extractResult');
        const panelHtml = `<div id="activeLearning" class="result-container"></div>`;
        resultDiv.insertAdjacentHTML('beforeend', panelHtml);
    }
    const activeDiv = document.getElementById('activeLearning');
    activeDiv.innerHTML = `<div class="loading"><div class="spinner"></div><span>知识库对比与价值评分中...</span></div>`;

    try {
        const reviewData = await callApi('/api/active/review', {
            method: 'POST',
            body: JSON.stringify({
                causal_pairs: extractData.causal_pairs || [],
                events: extractData.events || [],
                source_text: sourceText || '',
            }),
        });
        renderActiveReview(reviewData, extractData.causal_pairs || []);
    } catch (e) {
        activeDiv.innerHTML = `<div class="error-msg">知识库对比失败: ${escapeHtml(e.message)}</div>`;
    }
}

/** 渲染主动学习审核面板 */
function renderActiveReview(reviewData, rawPairs) {
    const activeDiv = document.getElementById('activeLearning');
    const pairs = reviewData.reviewed_pairs || [];
    const summary = reviewData.summary || {};
    const kbStats = reviewData.kb_stats || {};

    // 保存到全局供 confirm 使用
    window._activeReviewCache = pairs;

    let html = '';
    html += `<div class="section-title">
        🧠 主动学习 — 知识库审核
        <span class="count-badge" style="background:#2ecc7122;color:#2ecc71;">NEW: ${summary.NEW_UNKNOWN || 0}</span>
        <span class="count-badge" style="background:#3498db22;color:#3498db;">KNOWN: ${summary.KNOWN || 0}</span>
        <span class="count-badge" style="background:#95a5a622;color:#95a5a6;">自环: ${summary.SELF_LOOP || 0}</span>
        <span class="count-badge" style="background:#7f8c8d22;color:#7f8c8d;">低置信: ${summary.LOW_CONF || 0}</span>
        ${(summary.FORBIDDEN || 0) > 0 ? `<span class="count-badge" style="background:#e74c3c22;color:#e74c3c;">禁止: ${summary.FORBIDDEN || 0}</span>` : ''}
    </div>`;
    html += `<div class="parse-box" style="margin-bottom:12px;">
        <div class="parse-item"><span class="parse-key">KB当前规模:</span><span class="parse-value">已知 ${kbStats.num_known_pairs || 0} / 禁止 ${kbStats.num_forbidden_pairs || 0}</span></div>
        <div class="parse-item"><span class="parse-key">待审核:</span><span class="parse-value" style="color:#e67e22;">${summary.NEW_UNKNOWN || 0} 条新增候选，仅选 NEW / FORBIDDEN 可操作</span></div>
        <div class="parse-item"><span class="parse-key">操作说明:</span><span class="parse-value">✓确认(y)=加入已知因果　✗否定(n)=加入禁止　↻反向(r)=反方向正确　↷跳过(s)=不处理</span></div>
    </div>`;

    // 审核表格
    html += `<table class="data-table" id="activeTable">
        <thead><tr>
            <th class="idx-col">#</th><th>状态</th><th>原因事件</th><th></th><th>结果事件</th>
            <th>置信度</th><th>价值分</th>
            <th style="width:220px;">操作（仅 NEW/FORBIDDEN 需选）</th>
        </tr></thead><tbody>`;

    pairs.forEach((p, i) => {
        const cls = CLASS_STATUS_COLOR[p.status];
        const badgeColor = {
            'KNOWN': '#3498db', 'FORBIDDEN': '#e74c3c', 'NEW_UNKNOWN': '#e67e22',
            'SELF_LOOP': '#95a5a6', 'LOW_CONF': '#7f8c8d', 'CONFLICT': '#8e44ad',
        }[p.status] || '#95a5a6';
        const disabled = (p.status === 'KNOWN' || p.status === 'SELF_LOOP' || p.status === 'LOW_CONF');
        const defaultDecision = p.decision || 's';
        const totalScore = p.value_score ? p.value_score.total : '-';
        const scoreReason = p.value_score ? p.value_score.reason : '';

        html += `<tr data-index="${p.index}">
            <td class="idx-col">${i + 1}</td>
            <td><span class="mini-badge" style="background:${badgeColor}22;color:${badgeColor};">${escapeHtml(p.status_label || p.status)}</span></td>
            <td>${escapeHtml(p.cause_concept || '-')}
                ${p.cause_class ? `<span class="mini-badge" style="background:${CLASS_COLOR_MAP[p.cause_class]}22;color:${CLASS_COLOR_MAP[p.cause_class]};">${escapeHtml(p.cause_class)}</span>` : ''}
            </td>
            <td style="color:#2ecc71;font-weight:bold;">→</td>
            <td>${escapeHtml(p.effect_concept || '-')}
                ${p.effect_class ? `<span class="mini-badge" style="background:${CLASS_COLOR_MAP[p.effect_class]}22;color:${CLASS_COLOR_MAP[p.effect_class]};">${escapeHtml(p.effect_class)}</span>` : ''}
            </td>
            <td>${Number(p.confidence || 0).toFixed(3)}</td>
            <td>${totalScore !== '-' ? totalScore : ''}
                ${scoreReason ? `<br><small style="color:#7f8c8d;">${escapeHtml(scoreReason)}</small>` : ''}
            </td>
            <td>
                <div class="btn-row" style="flex-wrap:wrap;gap:4px;">
                    <button class="btn btn-mini btn-success" ${disabled ? 'disabled' : ''} onclick="setDecision(${p.index}, 'y', this)">✓ 确认</button>
                    <button class="btn btn-mini btn-danger" ${disabled ? 'disabled' : ''} onclick="setDecision(${p.index}, 'n', this)">✗ 否定</button>
                    <button class="btn btn-mini btn-warning" ${disabled ? 'disabled' : ''} onclick="setDecision(${p.index}, 'r', this)">↻ 反向</button>
                    <button class="btn btn-mini btn-ghost" ${disabled ? 'disabled' : ''} onclick="setDecision(${p.index}, 's', this)">↷ 跳过</button>
                    <input type="hidden" name="decision_${p.index}" id="decision_${p.index}" value="${defaultDecision}">
                </div>
            </td>
        </tr>`;
    });
    html += `</tbody></table>`;

    // 确认按钮
    html += `<div style="margin-top:16px;text-align:right;">
        <button class="btn btn-primary" onclick="confirmActiveLearning()">💾 确认并写入知识库 + 图谱</button>
    </div>`;

    html += `<div id="activeConfirmResult"></div>`;

    activeDiv.innerHTML = html;
}

/** 设置单条审核决策 */
function setDecision(index, decision, btnEl) {
    const input = document.getElementById(`decision_${index}`);
    if (input) input.value = decision;

    // 按钮视觉反馈：高亮选中按钮，其他变暗
    const row = btnEl.closest('tr');
    row.querySelectorAll('.btn-mini').forEach(b => {
        b.classList.remove('active');
    });
    btnEl.classList.add('active');
}

/** 确认所有审核，提交到后端写入 KB 和 Neo4j */
async function confirmActiveLearning() {
    const pairs = window._activeReviewCache || [];
    if (pairs.length === 0) {
        showToast('无待审核数据', 'warning');
        return;
    }
    const decisions = [];
    let actionable = 0;
    pairs.forEach(p => {
        const input = document.getElementById(`decision_${p.index}`);
        const dec = input ? input.value : 's';
        if (dec !== 's') actionable++;
        decisions.push({
            index: p.index,
            cause_concept: p.cause_concept,
            effect_concept: p.effect_concept,
            cause_class: p.cause_class,
            effect_class: p.effect_class,
            cause_subtype: p.cause_subtype,
            effect_subtype: p.effect_subtype,
            decision: dec,
            confidence: p.confidence,
        });
    });
    if (actionable === 0) {
        showToast('没有需要写入的决策（所有均为跳过）', 'warning');
        return;
    }
    const btn = event.target.closest('.btn');
    btn.disabled = true;
    btn.textContent = '写入中...';
    try {
        const resp = await callApi('/api/active/confirm', {
            method: 'POST',
            body: JSON.stringify({ decisions }),
        });
        renderActiveConfirmResult(resp);
    } catch (e) {
        document.getElementById('activeConfirmResult').innerHTML =
            `<div class="error-msg">写入失败: ${escapeHtml(e.message)}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = '💾 确认并写入知识库 + 图谱';
    }
}

/** 渲染 confirm 结果 */
function renderActiveConfirmResult(resp) {
    const div = document.getElementById('activeConfirmResult');
    const kb = resp.kb || {};
    const neo = resp.neo4j || {};
    let html = `<div class="parse-box" style="border-color:#2ecc71;background:#2ecc7111;margin-top:16px;">
        <div class="parse-item"><span class="parse-key">✅ 写入成功</span>
            <span class="parse-value">已处理 ${resp.decisions_processed || 0} 条决策</span></div>
        <div class="parse-item"><span class="parse-key">知识库（JSON）:</span>
            <span class="parse-value">已知 ${kb.before?.known}→${kb.after?.known} (<span style="color:#2ecc71;">+${kb.added?.known}</span>)
            　禁止 ${kb.before?.forbidden}→${kb.after?.forbidden} (<span style="color:#e67e22;">+${kb.added?.forbidden}</span>)</span></div>
        <div class="parse-item"><span class="parse-key">Neo4j 图谱:</span>
            <span class="parse-value">节点 upserted: ${neo.nodes_upserted || 0}
            　因果边: ${neo.causal_edges || 0}　FORBIDDEN 边: ${neo.forbidden_edges || 0}</span></div>
        <div class="parse-item"><span class="parse-key">路径:</span><span class="parse-value" style="font-size:12px;">${escapeHtml(kb.kb_path || '')}</span></div>
    </div>`;
    html += `<p class="hint">💡 切换到「因果图」标签页可刷新查看新节点和边。下一次抽取/搜索将自动使用更新后的知识库。</p>`;
    div.innerHTML = html;
    showToast(`写入成功：+${kb.added?.known || 0} 已知，+${kb.added?.forbidden || 0} 禁止`, 'success');
}

/** 渲染抽取结果 */
function renderExtractResult(data) {
    const resultDiv = document.getElementById('extractResult');
    const events = data.events || [];
    const pairs = data.causal_pairs || [];
    const chains = data.causal_chains || [];
    const stats = data.stats || {};

    let html = '';

    // 统计摘要
    // 从因果对里提取作物主体（去重）
    const crops = [...new Set(pairs.map(p => p.crop).filter(c => c && c.trim()))];
    const cropStr = crops.length > 0 ? crops.join('、') : '通用';
    html += `<div class="section-title">抽取统计</div>`;
    html += `<div class="parse-box">
        <div class="parse-item"><span class="parse-key">事件数:</span><span class="parse-value">${stats.events || events.length}</span></div>
        <div class="parse-item"><span class="parse-key">因果对:</span><span class="parse-value">${stats.causal_pairs || pairs.length}</span></div>
        <div class="parse-item"><span class="parse-key">因果链:</span><span class="parse-value">${stats.causal_chains || chains.length}</span></div>
        <div class="parse-item"><span class="parse-key">作物主体:</span><span class="parse-value" style="color:#27ae60;font-weight:bold;">${escapeHtml(cropStr)}</span></div>
    </div>`;

    // 事件表格
    html += `<div class="section-title">事件表 <span class="count-badge">${events.length}</span></div>`;
    if (events.length === 0) {
        html += '<p class="hint">未抽取到事件</p>';
    } else {
        html += '<table class="data-table"><thead><tr><th class="idx-col">#</th><th>事件</th><th>大类</th><th>子类</th></tr></thead><tbody>';
        events.forEach((ev, i) => {
            html += `<tr>
                <td class="idx-col">${i + 1}</td>
                <td>${escapeHtml(ev.concept || ev.event || ev.name || JSON.stringify(ev))}</td>
                <td>${escapeHtml(ev.entity_class || ev.event_class || ev.class || '-')}</td>
                <td>${escapeHtml(ev.sub_class_name || ev.sub_class_id || ev.subtype_id || ev.subtype || '-')}</td>
            </tr>`;
        });
        html += '</tbody></table>';
    }

    // 因果对表格
    html += `<div class="section-title">因果对 <span class="count-badge">${pairs.length}</span></div>`;
    if (pairs.length === 0) {
        html += '<p class="hint">未抽取到因果对</p>';
    } else {
        html += '<table class="data-table"><thead><tr><th class="idx-col">#</th><th>原因</th><th></th><th>结果</th><th>作物</th><th>置信度</th></tr></thead><tbody>';
        pairs.forEach((p, i) => {
            // p.cause / p.effect 可能是对象 {concept, text, entity_class} 或字符串
            const cause = typeof p.cause === 'string' ? p.cause
                : (p.cause?.concept || p.cause?.text || p.cause?.name || p.from || '-');
            const effect = typeof p.effect === 'string' ? p.effect
                : (p.effect?.concept || p.effect?.text || p.effect?.name || p.to || '-');
            const causeCls = typeof p.cause === 'object' && p.cause ? (p.cause.entity_class || p.cause.event_class || '') : '';
            const effectCls = typeof p.effect === 'object' && p.effect ? (p.effect.entity_class || p.effect.event_class || '') : '';
            const conf = p.confidence !== undefined ? p.confidence : (p.conf ?? '-');
            const causeBadge = causeCls ? `<span class="mini-badge" style="background:${CLASS_COLOR_MAP[causeCls]}22;color:${CLASS_COLOR_MAP[causeCls]};">${escapeHtml(causeCls)}</span>` : '';
            const effectBadge = effectCls ? `<span class="mini-badge" style="background:${CLASS_COLOR_MAP[effectCls]}22;color:${CLASS_COLOR_MAP[effectCls]};">${escapeHtml(effectCls)}</span>` : '';
            // 作物徽章（绿色=有作物主体，灰色=通用）
            const crop = p.crop || '';
            const cropBadge = crop.trim()
                ? `<span class="mini-badge" style="background:#27ae6022;color:#27ae60;">${escapeHtml(crop)}</span>`
                : `<span class="mini-badge" style="background:#95a5a622;color:#95a5a6;">通用</span>`;
            html += `<tr>
                <td class="idx-col">${i + 1}</td>
                <td>${escapeHtml(String(cause))} ${causeBadge}</td>
                <td style="color:#2ecc71;font-weight:bold;">→</td>
                <td>${escapeHtml(String(effect))} ${effectBadge}</td>
                <td>${cropBadge}</td>
                <td>${conf}</td>
            </tr>`;
        });
        html += '</tbody></table>';
    }

    // 因果链
    html += `<div class="section-title">因果链 <span class="count-badge">${chains.length}</span></div>`;
    if (chains.length === 0) {
        html += '<p class="hint">未抽取到因果链</p>';
    } else {
        chains.forEach((chainItem, i) => {
            // chain 可能是 { chain:[...], confidence, events, reasoning } 或直接数组
            let chainArr = Array.isArray(chainItem) ? chainItem : (chainItem?.chain || []);
            const chainConf = !Array.isArray(chainItem) ? (chainItem?.confidence ?? '') : '';
            const chainValid = !Array.isArray(chainItem) ? (chainItem?.is_valid ?? true) : true;

            let chainHtml = '';
            if (Array.isArray(chainArr) && chainArr.length > 0) {
                chainArr.forEach((node, idx) => {
                    const nodeText = typeof node === 'string' ? node
                        : (node?.concept || node?.event || node?.text || JSON.stringify(node));
                    chainHtml += `<span class="chain-node">${escapeHtml(String(nodeText))}</span>`;
                    if (idx < chainArr.length - 1) {
                        chainHtml += `<span class="chain-arrow">→</span>`;
                    }
                });
            } else {
                chainHtml = escapeHtml(String(chainItem));
            }
            const confText = chainConf !== '' ? `<span class="mini-badge" style="background:#2ecc7122;color:#2ecc71;">置信度 ${Number(chainConf).toFixed(3)}</span>` : '';
            const validText = !chainValid ? `<span class="mini-badge" style="background:#e74c3c22;color:#e74c3c;">无效</span>` : '';
            html += `<div class="causal-chain"><strong>链${i + 1}:</strong> ${chainHtml} ${confText} ${validText}</div>`;
        });
    }

    resultDiv.innerHTML = html;
}

/** 载入示例文本 */
async function loadSampleText() {
    try {
        const data = await callApi('/api/sample-texts');
        const samples = data.samples || [];
        if (samples.length === 0) {
            showToast('暂无示例文本');
            return;
        }
        // 随机选一条
        const sample = samples[Math.floor(Math.random() * samples.length)];
        document.getElementById('extractText').value = sample.text;
        showToast('已载入: ' + sample.title, 'success');
    } catch (e) {
        showToast('载入示例失败: ' + e.message);
    }
}

/** 清空抽取 */
function clearExtract() {
    document.getElementById('extractText').value = '';
    document.getElementById('extractResult').innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">📝</div>
            <p>请在左侧输入文本并点击「开始抽取」</p>
        </div>`;
}

/* ==========================================================================
   Tab 2: 因果图可视化（vis.js Network）
   ========================================================================== */

async function loadGraph() {
    const container = document.getElementById('graphNetwork');
    if (!container) return;
    setLoading('graphLoading', true);

    // 前置检查 vis.js 是否已加载
    if (typeof vis === 'undefined' || !vis || !vis.DataSet) {
        const fallbackMsg = window._visLoadError
            ? `<div style="margin-top:8px;color:#e67e22;">CDN加载情况: ${escapeHtml(window._visLoadError)}</div>`
            : '';
        container.innerHTML = `<div class="error-msg">
            ❌ vis.js 图可视化库未加载（vis is not defined）
            <div style="margin-top:8px;font-size:13px;line-height:1.8;">
            <strong>可能原因：</strong>CDN (cdn.jsdelivr.net / unpkg.com) 被网络拦截<br>
            <strong>解决方法：</strong><br>
            ① 先尝试：<button class="btn btn-mini btn-success" onclick="location.reload()">刷新页面重试</button><br>
            ② 关闭浏览器的广告拦截/代理工具（如 uBlock/AdBlock/翻墙分流）对 jsdelivr.net/unpkg.com 的拦截<br>
            ③ 或手动安装本地 vis.js: <code style="background:#f3f4f6;padding:2px 6px;border-radius:4px;">pip install -U nodeenv ; nodeenv -p ; npm install vis-network@9.1.9</code><br>
            ④ 或切换到可访问外网的网络后重试
            </div>
            ${fallbackMsg}
        </div>`;
        setLoading('graphLoading', false);
        return;
    }

    try {
        const data = await callApi('/api/graph?limit=200');
        graphDataCache = data;
        initGraphFilter();   // 重置筛选状态
        renderLegend();       // 重新渲染图例
        renderGraph(data.nodes || [], data.edges || []);
        showToast(`已加载 ${data.nodes.length} 节点 / ${data.edges.length} 边`, 'success');
    } catch (e) {
        container.innerHTML = `<div class="error-msg">加载图谱失败: ${escapeHtml(e.message)}</div>`;
    } finally {
        setLoading('graphLoading', false);
    }
}

/** 渲染 vis.js Network */
function renderGraph(nodes, edges) {
    const container = document.getElementById('graphNetwork');

    // 构造 vis 节点数据集（按 getColorByNode 着色，不用 group 字段避免 vis.js 自动覆盖颜色）
    const visNodes = nodes.map(n => {
        const color = getColorByNode(n);
        return {
            id: n.id,
            label: n.label,
            title: n.title || n.label,
            color: {
                background: color,
                border: color,
                highlight: { background: color, border: color },
                hover: { background: color, border: color },
            },
            font: { color: '#2c3e50', size: 13, face: 'sans-serif' },
            shape: 'dot',
            size: 12,
        };
    });

    // 构造 vis 边数据集
    const visEdges = edges.map(e => ({
        from: e.from,
        to: e.to,
        label: e.label || '',
        title: e.title || '',
        arrows: e.arrows || 'to',
        color: { color: '#bdc3c7', highlight: '#2ecc71', hover: '#3498db' },
        font: { size: 10, color: '#7f8c8d', align: 'middle' },
        smooth: { type: 'curvedCW', roundness: 0.15 },
    }));

    const graphData = {
        nodes: new vis.DataSet(visNodes),
        edges: new vis.DataSet(visEdges),
    };
    graphDataSet = graphData;  // 保存到全局变量供 highlightNode 使用

    // 物理布局：forceAtlasWideBased（力导向）
    const options = {
        nodes: {
            borderWidth: 2,
            shadow: true,
        },
        edges: {
            width: 1.2,
            selectionWidth: 2,
            arrows: { to: { enabled: true, scaleFactor: 0.5 } },
        },
        physics: {
            enabled: true,
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
                gravitationalConstant: -50,
                centralGravity: 0.01,
                springLength: 120,
                springConstant: 0.05,
                damping: 0.4,
                avoidOverlap: 0.5,
            },
            stabilization: {
                enabled: true,
                iterations: 150,
                fit: true,
            },
            timestep: 0.5,
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            zoomView: true,
            dragView: true,
            dragNodes: true,
            navigationButtons: false,
            keyboard: true,
        },
        layout: { improvedLayout: true },
    };

    // 创建/重建网络
    if (networkInstance) {
        networkInstance.destroy();
    }
    networkInstance = new vis.Network(container, graphData, options);

    // 节点点击高亮邻居
    networkInstance.on('click', (params) => {
        if (params.nodes.length > 0) {
            highlightNode(params.nodes[0]);
        } else if (highlightActive) {
            resetHighlight();
        }
    });

    // 双击取消高亮
    networkInstance.on('doubleClick', () => {
        if (highlightActive) resetHighlight();
    });
}

/** 高亮节点及其邻居 */
function highlightNode(nodeId) {
    if (!networkInstance || !graphDataSet) return;
    const allNodes = graphDataCache.nodes || [];
    const allEdges = graphDataCache.edges || [];

    // 找出所有邻居
    const connectedNodes = new Set([nodeId]);
    allEdges.forEach(edge => {
        if (edge.from === nodeId) connectedNodes.add(edge.to);
        if (edge.to === nodeId) connectedNodes.add(edge.from);
    });

    // 更新节点样式：高亮邻居，其他变暗
    const updatedNodes = allNodes.map(n => {
        const color = getColorByNode(n);
        if (connectedNodes.has(n.id)) {
            return {
                id: n.id,
                color: {
                    background: color,
                    border: color,
                },
                font: { color: '#2c3e50', size: 14 },
                opacity: 1.0,
            };
        } else {
            return {
                id: n.id,
                color: { background: color, border: color },
                font: { color: '#bdc3c7', size: 13 },
                opacity: 0.25,
            };
        }
    });

    graphDataSet.nodes.update(updatedNodes);
    highlightActive = true;
}

/** 重置高亮 */
function resetHighlight() {
    if (!networkInstance || !graphDataSet) return;
    const allNodes = graphDataCache.nodes || [];
    const updatedNodes = allNodes.map(n => {
        const color = getColorByNode(n);
        return {
            id: n.id,
            color: { background: color, border: color },
            font: { color: '#2c3e50', size: 13 },
            opacity: 1.0,
        };
    });
    graphDataSet.nodes.update(updatedNodes);
    highlightActive = false;
}

/** 适应屏幕 */
function fitGraph() {
    if (networkInstance) {
        networkInstance.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
    }
}

/** 图例大类配置（两级筛选：大类 → 父类子类 → 后端细粒度 sub_class_name 映射） */
const LEGEND_GROUPS = [
    {
        group: '作物', color: '#27ae60',
        subs: [
            { name: '作物', sub_keys: ['作物', '茄果类', '瓜类', '禾谷类', '油料作物', '纤维作物', '叶菜类', '饮料作物', '果树类'] },
        ]
    },
    {
        group: '环境因子', color: '#e74c3c',
        subs: [
            { name: '天气', sub_keys: ['空气温度', '光照', '降水', '天气'] },
            { name: '土壤', sub_keys: ['土壤湿度', '土壤pH'] },
            { name: '水质', sub_keys: [] },
        ]
    },
    {
        group: '生物胁迫', color: '#9b59b6',
        subs: [
            { name: '病害', sub_keys: ['病害'] },
            { name: '虫害/害虫', sub_keys: ['虫害', '虫害/害虫'] },
            { name: '病原', sub_keys: ['病原'] },
            { name: '杂草竞争', sub_keys: [] },
        ]
    },
    {
        group: '防治措施', color: '#f39c12',
        subs: [
            { name: '防治措施', sub_keys: ['防治措施'] },
        ]
    },
    {
        group: '功能症状', color: '#3498db',
        subs: [
            { name: '生理症状', sub_keys: ['生理症状', '解剖症状'] },
            { name: '生殖症状', sub_keys: ['生殖症状'] },
            { name: '产量症状', sub_keys: ['产量症状'] },
            { name: '品质症状', sub_keys: ['品质症状'] },
            { name: '胁迫响应症状', sub_keys: ['胁迫响应症状'] },
        ]
    },
];

/** 初始化 graphFilter 为两级结构 */
function initGraphFilter() {
    graphFilter = {};
    LEGEND_GROUPS.forEach(g => {
        graphFilter[g.group] = {
            visible: true,
            subs: Object.fromEntries(g.subs.map(s => [s.name, true])),
        };
    });
}

/** 判断某节点是否可见（大类+子类两级判断，sub_keys 映射后端细粒度 sub_class_name） */
function isNodeVisible(node) {
    const g = graphFilter[node.group];
    if (!g) return true;
    if (!g.visible) return false;
    const subKey = node.sub_class_name || '';
    // 查找该 subKey 属于哪个父类子类的 sub_keys
    const legendGroup = LEGEND_GROUPS.find(x => x.group === node.group);
    if (legendGroup) {
        for (const sub of legendGroup.subs) {
            if (sub.sub_keys.includes(subKey)) {
                return g.subs[sub.name] !== false;
            }
        }
    }
    return true;  // 未匹配到，默认可见
}

/** 渲染图例（两级可点击筛选 + 全隐藏/全显示） */
function renderLegend() {
    const legendList = document.getElementById('legendList');
    if (Object.keys(graphFilter).length === 0) initGraphFilter();

    let html = `
        <li class="legend-filter-bar">
            <button class="btn-mini btn-success" onclick="setAllFilter(true)">全显示</button>
            <button class="btn-mini btn-danger" onclick="setAllFilter(false)">全隐藏</button>
        </li>`;
    for (const g of LEGEND_GROUPS) {
        const gState = graphFilter[g.group];
        const gVisible = gState && gState.visible;
        const allSubsVisible = gVisible && g.subs.every(s => gState.subs[s.name]);
        html += `<li class="legend-group-title ${allSubsVisible ? '' : 'hidden-group'}"
                     onclick="toggleFilterGroup('${g.group}')">
                    <span class="filter-toggle">${allSubsVisible ? '◼' : '◻'}</span>
                    ${g.group}
                </li>`;
        for (const sub of g.subs) {
            const subVisible = gVisible && gState.subs[sub.name];
            html += `<li class="legend-sub ${subVisible ? '' : 'hidden-sub'}"
                        onclick="toggleFilterSub('${g.group}', '${sub.name}')">
                        <span class="color-dot" style="background:${g.color}${subVisible ? '' : ';opacity:0.3'}"></span>${sub.name}
                    </li>`;
        }
    }
    legendList.innerHTML = html;
}

/** 切换某大类的可见性（同时重置其下所有子类） */
function toggleFilterGroup(group) {
    const g = graphFilter[group];
    g.visible = !g.visible;
    if (g.visible) {
        // 大类重新可见时，子类默认全可见
        LEGEND_GROUPS.find(x => x.group === group).subs.forEach(s => g.subs[s.name] = true);
    }
    renderLegend();
    applyGraphFilter();
}

/** 切换某子类的可见性 */
function toggleFilterSub(group, subName) {
    const g = graphFilter[group];
    g.subs[subName] = !g.subs[subName];
    // 如果所有子类都隐藏了，大类也标记为隐藏；如果有子类可见，大类保持可见
    g.visible = Object.values(g.subs).some(v => v);
    renderLegend();
    applyGraphFilter();
}

/** 设置全部大类/子类可见性 */
function setAllFilter(visible) {
    LEGEND_GROUPS.forEach(g => {
        graphFilter[g.group].visible = visible;
        g.subs.forEach(s => graphFilter[g.group].subs[s.name] = visible);
    });
    renderLegend();
    applyGraphFilter();
}

/** 应用筛选到 vis-network（隐藏不可见节点 + 关联边） */
function applyGraphFilter() {
    if (!graphDataSet || !graphDataCache.nodes) return;
    // 重新计算每个节点的 hidden 状态，并确保 color 不丢失
    const updatedNodes = graphDataCache.nodes.map(n => {
        const color = getColorByNode(n);
        return {
            id: n.id,
            hidden: !isNodeVisible(n),
            // 显式传入 color 防止 vis-network 用 group 默认色覆盖
            color: {
                background: color,
                border: color,
                highlight: { background: color, border: color },
                hover: { background: color, border: color },
            },
        };
    });
    const visibleIds = new Set(graphDataCache.nodes.filter(isNodeVisible).map(n => n.id));
    const updatedEdges = graphDataCache.edges.map(e => ({
        id: e.id,
        hidden: !(visibleIds.has(e.from) && visibleIds.has(e.to)),
    }));
    graphDataSet.nodes.update(updatedNodes);
    graphDataSet.edges.update(updatedEdges);
}

/* ==========================================================================
   Tab 3: 路径搜索
   ========================================================================== */

async function runSearch() {
    const question = document.getElementById('searchQuestion').value.trim();
    if (!question) {
        showToast('请输入问题');
        return;
    }
    const direction = document.getElementById('searchDirection').value;
    const maxHops = parseInt(document.getElementById('searchMaxHops').value) || 4;
    const topK = parseInt(document.getElementById('searchTopK').value) || 5;

    setLoading('searchLoading', true);
    setBtnDisabled('btnSearch', true);
    const resultDiv = document.getElementById('searchResult');
    resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><span>检索路径中...</span></div>';

    try {
        // 前端 direction 传给后端：auto 由 parser 决定，backward/forward 显式覆盖
        const data = await callApi('/api/search', {
            method: 'POST',
            body: JSON.stringify({ question, direction, max_hops: maxHops, top_k: topK }),
        });
        renderSearchResult(data, direction);
    } catch (e) {
        resultDiv.innerHTML = `<div class="error-msg">搜索失败: ${escapeHtml(e.message)}</div>`;
    } finally {
        setLoading('searchLoading', false);
        setBtnDisabled('btnSearch', false);
    }
}

/** 渲染路径搜索结果 */
function renderSearchResult(data, userDirection) {
    const resultDiv = document.getElementById('searchResult');
    const parse = data.parse || {};
    const paths = data.paths || [];

    if (!data.ok) {
        resultDiv.innerHTML = `
            <div class="error-msg">${escapeHtml(data.message || '未能识别目标事件')}</div>
            <div class="parse-box">
                <div class="parse-item"><span class="parse-key">问题解析:</span><span class="parse-value">${escapeHtml(JSON.stringify(parse))}</span></div>
            </div>`;
        return;
    }

    let html = '';

    // 解析结果
    html += `<div class="section-title">问题解析</div>`;
    html += `<div class="parse-box">
        <div class="parse-item"><span class="parse-key">目标概念:</span><span class="parse-value">${escapeHtml(parse.target_concept || '-')}</span></div>
        <div class="parse-item"><span class="parse-key">检索方向:</span><span class="parse-value">${escapeHtml(parse.direction || userDirection || '-')}</span></div>
        <div class="parse-item"><span class="parse-key">意图:</span><span class="parse-value">${escapeHtml(parse.intent || '-')}</span></div>
        <div class="parse-item"><span class="parse-key">作物主体:</span><span class="parse-value" style="color:#27ae60;font-weight:bold;">${escapeHtml(parse.domain && parse.domain !== '通用' ? parse.domain : '通用')}</span></div>
    </div>`;

    // 路径列表
    const cropDomain = parse.domain && parse.domain !== '通用' ? parse.domain : '';
    html += `<div class="section-title">因果路径 <span class="count-badge">${paths.length}</span></div>`;
    if (paths.length === 0) {
        html += '<p class="hint">未找到路径</p>';
    } else {
        paths.forEach((path, i) => {
            const concepts = path.concepts || [];
            const confidences = path.confidences || [];
            const evidences = path.evidences || [];
            const score = path.score !== undefined ? path.score.toFixed(3) : '-';
            const avgConf = path.avg_confidence !== undefined ? path.avg_confidence.toFixed(3) : '-';
            // 作物匹配层级徽章：target=绿色(目标作物) / class=橙色(同类放宽) / fallback=灰色(全图兜底)
            const cropMatch = path.crop_match || (cropDomain ? 'target' : 'fallback');
            let matchBadge = '';
            if (cropMatch === 'target' && cropDomain) {
                matchBadge = `<span class="mini-badge" style="background:#27ae6022;color:#27ae60;margin-left:8px;">作物: ${escapeHtml(cropDomain)}</span>`;
            } else if (cropMatch === 'class') {
                matchBadge = `<span class="mini-badge" style="background:#f39c1222;color:#f39c12;margin-left:8px;">同类作物放宽</span>`;
            } else if (cropMatch === 'fallback') {
                matchBadge = `<span class="mini-badge" style="background:#95a5a622;color:#95a5a6;margin-left:8px;">全图兜底</span>`;
            }
            const length = path.length || concepts.length;

            // 构造因果链 A → B → C
            let chainHtml = '';
            concepts.forEach((c, idx) => {
                chainHtml += `<span class="chain-node">${escapeHtml(c)}</span>`;
                if (idx < concepts.length - 1) {
                    chainHtml += `<span class="chain-arrow">→</span>`;
                }
            });

            // 置信度标签
            let confHtml = '';
            confidences.forEach((cf, idx) => {
                if (idx < concepts.length - 1) {
                    confHtml += `<span class="conf-tag">${concepts[idx]} → ${concepts[idx + 1]}: ${Number(cf).toFixed(2)}</span>`;
                }
            });

            // 证据列表
            let evHtml = '';
            if (evidences.length > 0) {
                evHtml = '<ul class="path-evidences">';
                evidences.forEach(ev => {
                    evHtml += `<li>${escapeHtml(typeof ev === 'string' ? ev : JSON.stringify(ev))}</li>`;
                });
                evHtml += '</ul>';
            }

            html += `<div class="path-card">
                <div class="path-header">
                    <span class="path-rank">路径 #${i + 1}</span>
                    <div class="path-score">
                        长度: <span>${length}</span> |
                        综合分: <span>${score}</span> |
                        平均置信: <span>${avgConf}</span>
                        ${matchBadge}
                    </div>
                </div>
                <div class="path-chain">${chainHtml}</div>
                ${confHtml ? `<div class="confidence-list">${confHtml}</div>` : ''}
                ${evHtml}
            </div>`;
        });
    }

    resultDiv.innerHTML = html;
}

/** 清空搜索 */
function clearSearch() {
    document.getElementById('searchQuestion').value = '';
    document.getElementById('searchResult').innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <p>输入问题，查找因果路径</p>
        </div>`;
}

/* ==========================================================================
   Tab 4: 端到端问答
   ========================================================================== */

async function runAnswer() {
    const question = document.getElementById('answerQuestion').value.trim();
    if (!question) {
        showToast('请输入问题');
        return;
    }
    const maxHops = parseInt(document.getElementById('answerMaxHops').value) || 4;
    const topK = parseInt(document.getElementById('answerTopK').value) || 5;

    setLoading('answerLoading', true);
    setBtnDisabled('btnAnswer', true);
    const resultDiv = document.getElementById('answerResult');
    resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><span>生成回答中（Step1→2→3）...</span></div>';

    try {
        const data = await callApi('/api/answer', {
            method: 'POST',
            body: JSON.stringify({ question, max_hops: maxHops, top_k: topK }),
        });
        renderAnswerResult(data);
    } catch (e) {
        resultDiv.innerHTML = `<div class="error-msg">问答失败: ${escapeHtml(e.message)}</div>`;
    } finally {
        setLoading('answerLoading', false);
        setBtnDisabled('btnAnswer', false);
    }
}

/** 渲染端到端问答结果 */
function renderAnswerResult(data) {
    const resultDiv = document.getElementById('answerResult');
    const parse = data.parse || {};
    const paths = data.paths || [];
    const explanation = data.explanation || '';

    let html = '';

    // 4 阶段计时展示
    html += `<div class="section-title">Pipeline 计时</div>`;
    html += `<div class="timing-grid">
        <div class="timing-card step1">
            <div class="timing-label">Step 1 问题解析</div>
            <div class="timing-value">${data.step1_parse_ms ?? '-'}<span class="timing-unit">ms</span></div>
            <div class="timing-desc">QuestionParser</div>
        </div>
        <div class="timing-card step2">
            <div class="timing-label">Step 2 路径检索</div>
            <div class="timing-value">${data.step2_search_ms ?? '-'}<span class="timing-unit">ms</span></div>
            <div class="timing-desc">双向检索 + 4维排序</div>
        </div>
        <div class="timing-card step3">
            <div class="timing-label">Step 3 生成解释</div>
            <div class="timing-value">${data.step3_generate_ms ?? '-'}<span class="timing-unit">ms</span></div>
            <div class="timing-desc">CausalRAG LLM</div>
        </div>
        <div class="timing-card total">
            <div class="timing-label">总耗时</div>
            <div class="timing-value">${data.elapsed_ms ?? '-'}<span class="timing-unit">ms</span></div>
            <div class="timing-desc">端到端</div>
        </div>
    </div>`;

    // 问题解析
    html += `<div class="section-title">问题解析</div>`;
    html += `<div class="parse-box">
        <div class="parse-item"><span class="parse-key">目标概念:</span><span class="parse-value">${escapeHtml(parse.target_concept || '-')}</span></div>
        <div class="parse-item"><span class="parse-key">检索方向:</span><span class="parse-value">${escapeHtml(parse.direction || '-')}</span></div>
        <div class="parse-item"><span class="parse-key">意图:</span><span class="parse-value">${escapeHtml(parse.intent || '-')}</span></div>
        <div class="parse-item"><span class="parse-key">作物主体:</span><span class="parse-value" style="color:#27ae60;font-weight:bold;">${escapeHtml(parse.domain && parse.domain !== '通用' ? parse.domain : '通用')}</span></div>
    </div>`;

    // 解释文本
    html += `<div class="section-title">生成解释</div>`;
    html += `<div class="explanation-box">${escapeHtml(explanation)}</div>`;

    // 检索到的路径
    const ansCropDomain = parse.domain && parse.domain !== '通用' ? parse.domain : '';
    html += `<div class="section-title">检索路径 <span class="count-badge">${paths.length}</span></div>`;
    if (paths.length === 0) {
        html += '<p class="hint">未检索到路径</p>';
    } else {
        paths.forEach((path, i) => {
            const concepts = path.concepts || [];
            const score = path.score !== undefined ? path.score.toFixed(3) : '-';
            const avgConf = path.avg_confidence !== undefined ? path.avg_confidence.toFixed(3) : '-';
            // 作物匹配层级徽章
            const cropMatch = path.crop_match || (ansCropDomain ? 'target' : 'fallback');
            let ansMatchBadge = '';
            if (cropMatch === 'target' && ansCropDomain) {
                ansMatchBadge = `<span class="mini-badge" style="background:#27ae6022;color:#27ae60;margin-left:8px;">作物: ${escapeHtml(ansCropDomain)}</span>`;
            } else if (cropMatch === 'class') {
                ansMatchBadge = `<span class="mini-badge" style="background:#f39c1222;color:#f39c12;margin-left:8px;">同类作物放宽</span>`;
            } else if (cropMatch === 'fallback') {
                ansMatchBadge = `<span class="mini-badge" style="background:#95a5a622;color:#95a5a6;margin-left:8px;">全图兜底</span>`;
            }

            let chainHtml = '';
            concepts.forEach((c, idx) => {
                chainHtml += `<span class="chain-node">${escapeHtml(c)}</span>`;
                if (idx < concepts.length - 1) {
                    chainHtml += `<span class="chain-arrow">→</span>`;
                }
            });

            html += `<div class="path-card">
                <div class="path-header">
                    <span class="path-rank">路径 #${i + 1}</span>
                    <div class="path-score">分数: <span>${score}</span> | 平均置信: <span>${avgConf}</span>${ansMatchBadge}</div>
                </div>
                <div class="path-chain">${chainHtml}</div>
            </div>`;
        });
    }

    resultDiv.innerHTML = html;
}

/** 清空问答 */
function clearAnswer() {
    document.getElementById('answerQuestion').value = '';
    document.getElementById('answerResult').innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">💡</div>
            <p>输入问题，生成可解释的因果回答</p>
        </div>`;
}
