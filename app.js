const VENUE_ORDER = { '札幌': 1, '函館': 2, '福島': 3, '新潟': 4, '東京': 5, '中山': 6, '中京': 7, '京都': 8, '阪神': 9, '小倉': 10 };
const STAKE_PER_RACE = 3000;

const yen = n => `${Number(n || 0).toLocaleString('ja-JP')}円`;
const percent = n => `${Number(n || 0).toFixed(1)}%`;

function frameNumber(horseNo, horseCount) {
  const n = Math.max(Number(horseCount) || Number(horseNo), Number(horseNo));
  const h = Number(horseNo);
  if (n <= 8) return Math.min(h, 8);
  const base = Math.floor(n / 8);
  const extra = n % 8;
  let cursor = 1;
  for (let frame = 1; frame <= 8; frame++) {
    const count = base + (frame > 8 - extra ? 1 : 0);
    if (h >= cursor && h < cursor + count) return frame;
    cursor += count;
  }
  return 8;
}

function horseBox(no, race) {
  const saved = race?.horseFrames?.[String(no)] ?? race?.horseFrames?.[no];
  const frame = Number(saved) || frameNumber(no, race?.horseCount);
  return `<span class="horse-box frame-${frame}" title="馬番 ${no} / ${frame}枠">${no}</span>`;
}

function predictionBoxes(numbers, race) {
  return `<div class="horses">${numbers.map(n => horseBox(n, race)).join('')}</div>`;
}

function resultBoxes(result, race) {
  if (!result?.places?.length) return '<span class="place-sep">—</span>';
  const groups = result.places.map(group => {
    if (group.length === 1) return horseBox(group[0], race);
    return `<span class="horses">${group.map(n => horseBox(n, race)).join('<span class="place-sep">=</span>')}</span>`;
  });
  return `<div class="horses">${groups.join('<span class="place-sep">›</span>')}</div>`;
}

function judgement(status) {
  if (status === 'hit') return '<span class="judgement hit">的中</span>';
  if (status === 'miss') return '<span class="judgement miss">不的中</span>';
  return '<span class="judgement pending">未確定</span>';
}

function officialTrifectaPayouts(race) {
  const saved = Array.isArray(race?.trifectaPayouts) ? race.trifectaPayouts : [];
  if (saved.length) return saved.map(Number).filter(Number.isFinite);
  return (race?.result?.trifectas || [])
    .map(t => Number(t?.payout))
    .filter(Number.isFinite);
}

function officialPayoutLabel(race) {
  const payouts = officialTrifectaPayouts(race);
  if (!payouts.length) return '—';
  return payouts.map(yen).join(' / ');
}

function daySummary(day) {
  const finished = day.races.filter(r => r.status === 'hit' || r.status === 'miss');
  const hits = finished.filter(r => r.status === 'hit').length;
  const payout = finished.reduce((sum, r) => sum + Number(r.payout || 0), 0);
  const stake = finished.reduce((sum, r) => sum + Number(r.stake || STAKE_PER_RACE), 0);
  const recovery = stake ? payout / stake * 100 : 0;
  return { hits, payout, recovery };
}

function dateLabel(iso) {
  const d = new Date(`${iso}T00:00:00+09:00`);
  const weekdays = ['日','月','火','水','木','金','土'];
  return `${iso}（${weekdays[d.getDay()]}）`;
}

function renderDay(day) {
  const summary = daySummary(day);
  const dl = dateLabel(day.date);
  const races = [...day.races].sort((a,b) => (VENUE_ORDER[a.venue] ?? 99) - (VENUE_ORDER[b.venue] ?? 99) || a.raceNo - b.raceNo);
  const rows = races.map(r => {
    const rate = (r.status === 'hit' || r.status === 'miss') ? (Number(r.payout || 0) / Number(r.stake || STAKE_PER_RACE) * 100) : null;
    return `<tr class="${r.status === 'hit' ? 'hit-row' : ''}">
      <td class="race-name"><span class="venue">${r.venue}</span> ${r.raceNo}R</td>
      <td>${predictionBoxes(r.prediction?.axes?.slice(0,1) || [], r)}</td>
      <td>${predictionBoxes(r.prediction?.axes?.slice(1,2) || [], r)}</td>
      <td>${predictionBoxes(r.prediction?.opponents || [], r)}</td>
      <td>${resultBoxes(r.result, r)}</td>
      <td>${judgement(r.status)}</td>
      <td class="money">${officialPayoutLabel(r)}</td>
      <td class="rate">${rate == null ? '—' : percent(rate)}</td>
    </tr>`;
  }).join('');

  return `<section class="day-card">
    <div class="day-top">
      <div class="date-wrap"><span class="date-label">${dl}</span></div>
      <div class="summary">
        <div class="summary-item"><span class="summary-label">的中数</span><span class="summary-value">${summary.hits} / ${races.length}</span></div>
        <div class="summary-item"><span class="summary-label">払戻総額</span><span class="summary-value">${yen(summary.payout)}</span></div>
        <div class="summary-item"><span class="summary-label">総回収率</span><span class="summary-value">${percent(summary.recovery)}</span></div>
      </div>
    </div>
    <div class="table-scroll">
      <table class="race-table">
        <thead><tr><th>レース</th><th>軸1</th><th>軸2</th><th>相手</th><th>結果</th><th>判定</th><th>払戻金</th><th>回収率</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  </section>`;
}

async function boot() {
  const app = document.getElementById('app');
  try {
    const res = await fetch(`./data/races.json?v=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const days = [...(data.days || [])].sort((a,b) => b.date.localeCompare(a.date));
    app.innerHTML = days.length ? days.map(renderDay).join('') : '<div class="empty">表示できるレースがまだありません。</div>';
  } catch (e) {
    console.error(e);
    app.innerHTML = '<div class="error">レースデータを読み込めませんでした。data/races.json を確認してください。</div>';
  }
}

document.addEventListener('DOMContentLoaded', boot);
