/* Offline engineering analysis: data validation, physical identity and slice geometry. */
(function (root) {
  'use strict';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));
  const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, key => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[key]));

  function physicalSignature(design) {
    if (!design || design.preview_only || !Array.isArray(design.rings) || !design.rings.length) return '';
    const rings = [];
    for (const ring of design.rings) {
      if (!ring || !finite(ring.radius_mm) || !finite(ring.z_mm) || !finite(ring.phase_deg) || !Number.isInteger(ring.count) || ring.count < 1 || !['out','in','down','up'].includes(ring.polarity)) return '';
      const period = 360 / ring.count;
      let phase = ((ring.phase_deg % period) + period) % period;
      if (Math.abs(phase - period) < 1e-9 || Math.abs(phase) < 1e-9) phase = 0;
      rings.push([ring.radius_mm, ring.z_mm, ring.count, Number(phase.toFixed(9)), ring.polarity]);
    }
    rings.sort((a,b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
    return JSON.stringify([design.cube_mm ?? 12, design.remanence_T ?? 1.43, design.reference_mur ?? 1.05, rings]);
  }

  function matchingAnalysis(analysis, design, config = {}) {
    if (!analysis || !physicalSignature(design) || physicalSignature(analysis.design) !== physicalSignature(design)) return null;
    const actual = analysis.domain || {}, target = config.target || {};
    if (actual.geometry !== target.geometry || Number(actual.diameter_mm) !== Number(target.diameter_mm) || Number(actual.z_mm ?? 0) !== Number(target.z_mm ?? 0)) return null;
    if (analysis.polarization_model !== (config.polarization_model || 'remanence')) return null;
    return analysis;
  }

  // Shared signed scale: blue below reference, white at zero, red above reference.
  function ppmColor(value, limit) {
    if (!finite(value) || !finite(limit) || limit <= 0) return null;
    const t = clamp(value / limit, -1, 1), white = [247,249,251];
    const end = t < 0 ? [40,104,179] : [190,62,66], amount = Math.abs(t);
    return white.map((channel,index) => Math.round(channel + (end[index] - channel) * amount));
  }

  function sliceCells(analysis, options = {}) {
    const slices = analysis?.slices;
    if (!slices || !Array.isArray(slices.planes) || !finite(slices.color_limit_ppm) || slices.color_limit_ppm <= 0) return [];
    const result = [], domain = analysis.domain || {};
    for (const plane of slices.planes) {
      if (!['XY','XZ','YZ'].includes(plane.name) || (domain.geometry === 'disk' && plane.name !== 'XY')) continue;
      if (options.plane && options.plane !== 'all' && plane.name !== options.plane) continue;
      const n = Number(plane.grid_size), count = n*n;
      if (!Number.isInteger(n) || n < 2 || n > 129 || !Array.isArray(plane.points_mm) || plane.points_mm.length !== count || !Array.isArray(plane.values_ppm) || plane.values_ppm.length !== count || !Array.isArray(plane.valid_mask) || plane.valid_mask.length !== count) continue;
      const valid = index => Boolean(plane.valid_mask[index]) && finite(plane.values_ppm[index]) && Array.isArray(plane.points_mm[index]) && plane.points_mm[index].length === 3 && plane.points_mm[index].every(finite);
      for (let row = 0; row < n-1; row++) for (let column = 0; column < n-1; column++) {
        const indices = [row*n+column, row*n+column+1, (row+1)*n+column+1, (row+1)*n+column];
        if (!indices.every(valid)) continue;
        const points = indices.map(index => plane.points_mm[index]);
        // Points are the engineering coordinates, never camera/presentation coordinates.
        if (options.cutaway && points.some(point => point[1] < -1e-8)) continue;
        const value = indices.reduce((sum,index) => sum + plane.values_ppm[index],0) / 4;
        result.push({plane:plane.name, points, value_ppm:value, color:ppmColor(value,slices.color_limit_ppm), indices});
      }
    }
    return result;
  }

  function requestPayload(source, options = {}) {
    if (!source?.design || !physicalSignature(source.design) || ['illustrative','reference','unavailable'].includes(source.kind)) throw new Error('请先生成并选择真实磁体方案。');
    return {design:source.design, source_label:String(source.label || '当前磁体方案'), analysis_options:{mutual_demag:Boolean(options.mutual_demag), tolerance:options.tolerance !== false}};
  }

  function sensitiveRings(analysis, design) {
    const rows = analysis?.tolerance?.sensitivity_rings;
    if (!Array.isArray(rows) || !Array.isArray(design?.rings)) return [];
    const byId = new Map(rows.filter(row => finite(row.sensitivity_ppm) && row.sensitivity_ppm >= 0).map(row => [String(row.ring_id),row.sensitivity_ppm]));
    const maximum = Math.max(0,...byId.values());
    return design.rings.filter(ring => byId.has(String(ring.id))).map(ring => {
      const value = byId.get(String(ring.id)), amount = maximum > 0 ? value / maximum : 0;
      return {ring_id:String(ring.id),radius_mm:ring.radius_mm,z_mm:ring.z_mm,sensitivity_ppm:value,
        color:[221,229,236].map((channel,index) => Math.round(channel + ([194,76,83][index]-channel)*amount))};
    });
  }

  function zonalTerms(harmonics) {
    const spherical = harmonics?.spherical || harmonics || {};
    const source = spherical.zonal || spherical.zonal_terms || [];
    return Array.isArray(source) ? source.filter(row => finite(row.coefficient_ppm)).map(row => ({degree:Number(row.degree),label:String(row.label || 'Z'+row.degree),coefficient_ppm:row.coefficient_ppm})) : [];
  }

  function harmonicBars(harmonics) {
    const terms = zonalTerms(harmonics), maximum = Math.max(1e-12,...terms.map(row=>Math.abs(row.coefficient_ppm)));
    if (!terms.length) return '<div class="analysis-muted">当前评价域尚无可用球谐谱；平面 FOV 不以球谐拟合冒充三维诊断。</div>';
    return '<div class="harmonic-bars" role="img" aria-label="轴对称球谐系数，ppm，带符号">'+terms.map(term=>'<div class="harmonic-bar"><span>'+escapeHTML(term.label)+'</span><i><b style="width:'+Math.max(.6,Math.abs(term.coefficient_ppm)/maximum*100)+'%;background:'+(term.coefficient_ppm<0?'#5b86b4':'#b97574')+'"></b></i><strong>'+escapeHTML(term.coefficient_ppm.toPrecision(5))+' ppm</strong></div>').join('')+'</div>';
  }

  function toleranceSummaryHTML(tolerance) {
    const tol=tolerance||{},summary=tol.summary||{},assumptions=tol.assumptions||{};
    const number=(value,digits=3)=>finite(value)?value.toFixed(digits).replace(/\.?0+$/,''):'—';
    if (tol.available===false||tol.enabled===false) return '<div class="analysis-muted">'+escapeHTML(tol.reason||'本次未启用公差分析。')+'</div>';
    const scope=summary.yield_scope==='field_only_no_ppm_limit'?'仅场强通过率（未设ppm门槛）':summary.yield_scope==='field_and_sampled_ppm'?'场强及采样PPM通过率':'样本通过率（口径未声明）';
    let html='<div class="analysis-summary"><span>P50 '+number(summary.p50_ppm)+' ppm</span><span>P90 '+number(summary.p90_ppm)+' ppm</span><span>'+scope+' '+(finite(summary.yield_fraction)?number(summary.yield_fraction*100,1)+'%':'未提供')+'</span></div>';
    const interval=summary.yield_confidence95;
    if(Array.isArray(interval)&&interval.length===2&&interval.every(finite)) html+='<div class="analysis-muted">该通过率的 95% Wilson 区间：'+number(interval[0]*100,2)+'%～'+number(interval[1]*100,2)+'%。</div>';
    html+='<div class="analysis-muted">公差求场基准：'+escapeHTML(tol.polarization_model||'固定源快速模型')+'；标称 '+number(tol.nominal?.pp_norm_ppm)+' ppm。'+escapeHTML(tol.method||'固定雅可比的一阶扰动代理')+'。</div>';
    if(tol.scope_note)html+='<div class="analysis-muted">'+escapeHTML(tol.scope_note)+'</div>';
    if(finite(assumptions.remanence_half_range_fraction)&&finite(assumptions.sector_translation_half_range_mm))html+='<div class="analysis-muted">有界均匀分布半幅：各磁块剩磁独立 ±'+number(100*assumptions.remanence_half_range_fraction)+'%；同层同扇区共用 XYZ 平移 ±'+number(assumptions.sector_translation_half_range_mm)+' mm。以上为半幅，不是标准差 σ。</div>';
    if(finite(tol.timings?.precompute_seconds)&&finite(tol.timings?.sampling_seconds))html+='<div class="analysis-muted">预计算 '+number(tol.timings.precompute_seconds)+' s；抽样 '+number(tol.timings.sampling_seconds)+' s。</div>';
    return html+'<div class="analysis-muted">统计仅针对声明的公差分布、一阶位移代理及本次有限采样；不含碰撞或制造良率、密集极值和 FEM 认证。</div>';
  }

  const exported = {physicalSignature, matchingAnalysis, ppmColor, sliceCells, requestPayload, sensitiveRings, zonalTerms, harmonicBars, toleranceSummaryHTML};
  root.MRIAnalysis = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window === 'undefined' ? globalThis : window);
