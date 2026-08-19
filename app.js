const VENUE_ORDER = { '札幌': 1, '函館': 2, '福島': 3, '新潟': 4, '東京': 5, '中山': 6, '中京': 7, '京都': 8, '阪神': 9, '小倉': 10 };
const STAKE_PER_RACE = 3000;

const RACE_INDEX_DETAILS = {
  '202601010811': {
    title: '札幌11R 札幌記念',
    horseCount: 16,
    prediction: {
      axes: [8, 4],
      opponents: [10, 7, 14, 15, 12],
      excluded: [13]
    },
    horses: [
      { no: 1, name: 'オニャンコポン', recent: ['88/79/83','61/72/55','63/69/50','64/61/48','66/74/60'], recentIndex: 68, expectedPopularity: 12, pace: 58, course: 64, today: 61, total: 65, rank: 14 },
      { no: 2, name: 'イガッチ', recent: ['52/45/44','74/79/73','88/84/80','73/73/66','83/73/68'], recentIndex: 66, expectedPopularity: 14, pace: 66, course: 62, today: 64, total: 65, rank: 15 },
      { no: 3, name: 'ピンクジン', recent: ['70/73/65','62/67/50','55/57/47','75/70/65','72/68/62'], recentIndex: 63, expectedPopularity: 16, pace: 61, course: 68, today: 65, total: 64, rank: 16 },
      { no: 4, name: 'マジックサンズ', recent: ['73/81/76','86/87/79','77/83/75','74/78/59','82/85/74'], recentIndex: 78, expectedPopularity: 7, pace: 82, course: 82, today: 82, total: 80, rank: 4 },
      { no: 5, name: 'エコロヴァルツ', recent: ['45/47/55','86/88/86','91/89/85','78/88/74','85/86/79'], recentIndex: 73, expectedPopularity: 10, pace: 81, course: 74, today: 78, total: 75, rank: 11 },
      { no: 6, name: 'ローシャムパーク', recent: ['89/88/86','77/82/65','49/51/52','84/90/78','77/82/68'], recentIndex: 75, expectedPopularity: 9, pace: 76, course: 70, today: 73, total: 74, rank: 13 },
      { no: 7, name: 'ショウヘイ', recent: ['63/73/69','93/91/94','59/67/61','91/94/91','92/95/93'], recentIndex: 79, expectedPopularity: 5, pace: 88, course: 68, today: 78, total: 79, rank: 5 },
      { no: 8, name: 'サクラファレル', recent: ['88/84/76','92/87/80','78/78/76','94/81/72','96/83/69'], recentIndex: 82, expectedPopularity: 3, pace: 91, course: 96, today: 94, total: 87, rank: 1 },
      { no: 9, name: 'マイネルモーント', recent: ['86/81/84','69/76/60','83/79/76','82/84/80','55/57/57'], recentIndex: 76, expectedPopularity: 8, pace: 78, course: 70, today: 74, total: 75, rank: 9 },
      { no: 10, name: 'アドマイヤテラ', recent: ['91/97/96','95/96/96','67/78/69','評価外','89/90/87'], recentIndex: 89, expectedPopularity: 1, pace: 77, course: 73, today: 75, total: 83, rank: 2 },
      { no: 11, name: 'アラタ', recent: ['67/73/61','75/79/66','67/73/57','82/78/76','91/86/86'], recentIndex: 71, expectedPopularity: 11, pace: 69, course: 93, today: 81, total: 75, rank: 10 },
      { no: 12, name: 'ゼンダンハヤブサ', recent: ['91/87/90','78/78/68','82/82/69','74/75/65','87/80/68'], recentIndex: 79, expectedPopularity: 4, pace: 73, course: 67, today: 70, total: 75, rank: 8 },
      { no: 13, name: 'グランディア', recent: ['91/90/92','84/84/82','89/88/88','89/86/86','54/61/49'], recentIndex: 85, expectedPopularity: 2, pace: 83, course: 69, today: 76, total: 81, rank: 3, excluded: true },
      { no: 14, name: 'レディネス', recent: ['63/64/52','86/92/84','93/91/86','51/52/44','48/52/58'], recentIndex: 75, expectedPopularity: 15, pace: 84, course: 84, today: 84, total: 79, rank: 6 },
      { no: 15, name: 'シェイクユアハート', recent: ['50/49/57','95/94/95','87/90/85','94/93/94','88/88/88'], recentIndex: 79, expectedPopularity: 6, pace: 85, course: 70, today: 78, total: 78, rank: 7 },
      { no: 16, name: 'ホウオウビスケッツ', recent: ['58/74/62','42/47/51','73/86/68','94/91/91','78/81/73'], recentIndex: 67, expectedPopularity: 13, pace: 87, course: 84, today: 86, total: 74, rank: 12 }
    ]
  }
};

const yen = n => `${Number(n || 0).toLocaleString('ja-JP')}円`;
const percent = n => `${Number(n || 0).toFixed(1)}%`;

function raceDetail(race) {
  return RACE_INDEX_DETAILS[race?.raceId] || null;
}

function effectiveHorseCount(race) {
  return raceDetail(race)?.horseCount || race?.horseCount;
}

function effectivePrediction(race) {
  return raceDetail(race)?.prediction || race?.prediction || { axes: [], opponents: [] };
}

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
  const frame = Number(saved) || frameNumber(no, effectiveHorseCount(race));
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
  if (status === 'miss') return '';
  return '<span class="judgement pending">未確定</span>';
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

function actualTrifectaPayout(race) {
  const trifectas = race?.result?.trifectas || [];
  if (!trifectas.length) return null;
  return trifectas.reduce((sum, item) => sum + Number(item.payout || 0), 0);
}

function raceNameCell(race) {
  const label = `<span class="venue">${race.venue}</span> ${race.raceNo}R`;
  if (!raceDetail(race)) return label;
  return `<button class="race-detail-trigger" type="button" data-race-id="${race.raceId}" aria-label="${race.venue}${race.raceNo}Rの指数表を表示">${label}</button>`;
}

function renderDay(day, initiallyExpanded = true) {
  const summary = daySummary(day);
  const dl = dateLabel(day.date);
  const races = [...day.races].sort((a,b) => (VENUE_ORDER[a.venue] ?? 99) - (VENUE_ORDER[b.venue] ?? 99) || a.raceNo - b.raceNo);
  const rows = races.map(r => {
    const wonPayout = Number(r.payout || 0);
    const actualPayout = actualTrifectaPayout(r);
    const rate = (r.status === 'hit' || r.status === 'miss') ? (wonPayout / Number(r.stake || STAKE_PER_RACE) * 100) : null;
    const prediction = effectivePrediction(r);
    return `<tr class="${r.status === 'hit' ? 'hit-row' : r.status === 'miss' ? 'miss-row' : ''}">
      <td class="race-name">${raceNameCell(r)}</td>
      <td>${predictionBoxes(prediction.axes?.slice(0,1) || [], r)}</td>
      <td>${predictionBoxes(prediction.axes?.slice(1,2) || [], r)}</td>
      <td>${predictionBoxes(prediction.opponents || [], r)}</td>
      <td>${resultBoxes(r.result, r)}</td>
      <td>${judgement(r.status)}</td>
      <td class="money">${actualPayout == null ? '—' : yen(actualPayout)}</td>
      <td class="rate">${rate == null ? '—' : percent(rate)}</td>
    </tr>`;
  }).join('');

  const collapsedClass = initiallyExpanded ? '' : ' is-collapsed';
  return `<section class="day-card${collapsedClass}" data-day-date="${day.date}">
    <div class="day-top">
      <div class="date-wrap"><span class="date-label">${dl}</span></div>
      <div class="day-summary">
        <div class="summary-item"><span class="summary-label">的中数</span><span class="summary-value">${summary.hits} / ${races.length}</span></div>
        <div class="summary-item"><span class="summary-label">払戻総額</span><span class="summary-value">${yen(summary.payout)}</span></div>
        <div class="summary-item"><span class="summary-label">総回収率</span><span class="summary-value">${percent(summary.recovery)}</span></div>
      </div>
      <button class="day-toggle" type="button" aria-label="${initiallyExpanded ? 'この日付を折りたたむ' : 'この日付を開く'}" aria-expanded="${initiallyExpanded ? 'true' : 'false'}">
        <span class="day-toggle-icon" aria-hidden="true"></span>
      </button>
    </div>
    <div class="day-content">
      <div class="table-scroll">
        <table class="race-table">
          <thead>
            <tr class="column-row"><th>レース</th><th>本命</th><th>対抗</th><th>相手</th><th>結果</th><th>判定</th><th>三連単</th><th>回収率</th></tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>
  </section>`;
}

function indexHorseNumber(no, horseCount) {
  const frame = frameNumber(no, horseCount);
  return `<span class="horse-box index-horse-box frame-${frame}">${no}</span>`;
}

function selectionLabel(horse, detail) {
  if (horse.no === detail.prediction.axes[0]) return '<span class="index-pick pick-main">本命</span>';
  if (horse.no === detail.prediction.axes[1]) return '<span class="index-pick pick-second">対抗</span>';
  if (detail.prediction.opponents.includes(horse.no)) return '<span class="index-pick pick-opponent">相手</span>';
  if (horse.excluded) return '<span class="index-danger" role="img" aria-label="危険" title="危険">&#9888;&#xfe0e;</span>';
  return '<span class="index-eval-empty">—</span>';
}

function recentIndexMarkup(value) {
  if (value === '評価外') return '<span class="index-recent-na">評価外</span>';
  const parts = String(value).split('/');
  if (parts.length !== 3) return value;
  return `<span class="index-recent-score"><span class="index-recent-label">展</span>${parts[0]}<span class="index-recent-label">時</span>${parts[1]}<span class="index-recent-label">成</span>${parts[2]}</span>`;
}

function renderIndexDetail(detail) {
  const rows = detail.horses.map(horse => `
    <tr>
      <td class="index-evaluation">${selectionLabel(horse, detail)}</td>
      <td>${indexHorseNumber(horse.no, detail.horseCount)}</td>
      <td class="index-horse-name">${horse.name}</td>
      <td class="index-popularity">${horse.expectedPopularity}</td>
      <td class="index-total">${horse.total}</td>
      <td class="index-rank">${horse.rank}</td>
      ${horse.recent.map(value => `<td>${recentIndexMarkup(value)}</td>`).join('')}
      <td class="index-strong index-recent-total">${horse.recentIndex}</td>
      <td>${horse.pace}</td>
      <td>${horse.course}</td>
      <td class="index-strong index-today">${horse.today}</td>
    </tr>`).join('');

  return `
    <div class="index-modal-backdrop" data-index-close="true">
      <section class="index-modal" role="dialog" aria-modal="true" aria-labelledby="index-modal-title">
        <div class="index-modal-header">
          <h2 id="index-modal-title">${detail.title}</h2>
          <button class="index-modal-close" type="button" data-index-close="true" aria-label="指数表を閉じる">×</button>
        </div>
        <div class="index-table-scroll">
          <table class="index-table">
            <thead>
              <tr>
                <th>評価</th><th>馬番</th><th>馬名</th><th>想人</th><th>総合</th><th>順位</th><th>前走</th><th>2走前</th><th>3走前</th><th>4走前</th><th>5走前</th><th>近走</th><th>展開</th><th>コース</th><th>今回</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <div class="index-logic">
          <p>近5走は各レースを「展開・タイム・成績」の3指数で個別評価し、すべて整数で1の位まで精査します。展開は通過順・位置取り・上がり・着差・ペース、タイムは走破時計・レースレベル・馬場・着差・上がり、成績は着順に加えてレース格と相手レベルを評価します。</p>
          <p>「近走」は単純な着順平均ではなく、直近を重視した基礎評価に、近5走の上位パフォーマンスから算出する能力上限と再現性を加味します。長期休養明け、大幅な馬体重変動など結果の信頼度を下げる客観的要因が重なった一走は、その凡走を能力低下と断定せず影響を抑えます。これにより、一度の大敗だけで過度に評価を落とさない設計とします。「想人」は近走成績から推定する想定人気で、実際のオッズは使用しません。</p>
          <p>「展開」は今回の想定ペースと脚質・位置取りの適合度を評価します。「コース」は当該コース実績を最重視し、当該場の他距離実績を補助評価します。当該コース未経験の場合は同距離の他場実績と類似条件への適応力で補完しますが、実績馬より上限を抑えます。「今回」は展開50％＋コース50％、総合指数は近走60％＋今回40％を基本とします。</p>
          <p><strong>軸馬選定：</strong>本命は想定3番人気以内のうち総合指数最上位、対抗は想定4番人気以下のうち総合指数最上位とします。ただし、想定3番人気以内で総合指数最下位の馬は危険馬として買い目から除外します。相手はそれ以外の総合指数上位から選び、頭数は既定の出走頭数ルールに従います。</p>
        </div>
      </section>
    </div>`;
}
function openIndexDetail(raceId, trigger) {
  const detail = RACE_INDEX_DETAILS[raceId];
  if (!detail) return;
  closeIndexDetail(false);
  lastIndexTrigger = trigger || null;
  document.body.insertAdjacentHTML('beforeend', renderIndexDetail(detail));
  document.body.classList.add('index-modal-open');
  document.querySelector('.index-modal-close')?.focus();
}

function closeIndexDetail(restoreFocus = true) {
  document.querySelector('.index-modal-backdrop')?.remove();
  document.body.classList.remove('index-modal-open');
  if (restoreFocus) lastIndexTrigger?.focus();
  lastIndexTrigger = null;
}

function bindDayToggles() {
  document.addEventListener('click', event => {
    const toggle = event.target instanceof Element ? event.target.closest('.day-toggle') : null;
    if (!toggle) return;

    const card = toggle.closest('.day-card');
    if (!card) return;

    const collapsed = card.classList.toggle('is-collapsed');
    toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    toggle.setAttribute('aria-label', collapsed ? 'この日付を開く' : 'この日付を折りたたむ');
  });
}

function bindIndexDetails() {
  document.addEventListener('click', event => {
    const trigger = event.target instanceof Element ? event.target.closest('.race-detail-trigger') : null;
    if (trigger) {
      openIndexDetail(trigger.dataset.raceId, trigger);
      return;
    }

    const close = event.target instanceof Element ? event.target.closest('[data-index-close="true"]') : null;
    if (!close) return;
    if (close.classList.contains('index-modal-backdrop') && event.target !== close) return;
    closeIndexDetail();
  });

  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && document.querySelector('.index-modal-backdrop')) {
      closeIndexDetail();
    }
  });
}

/*
 * SPで横方向に「引っ張る」操作が端を越えないようにする。
 * - 表の途中では通常どおり横スクロール可能
 * - 左端/右端では、それ以上外側へのドラッグを抑止
 * - 表以外では横方向のドラッグを抑止
 * - 縦スクロールは維持
 */
function lockHorizontalPull() {
  let startX = 0;
  let startY = 0;
  let scroller = null;

  document.addEventListener('touchstart', event => {
    if (event.touches.length !== 1) return;
    const touch = event.touches[0];
    startX = touch.clientX;
    startY = touch.clientY;
    scroller = event.target instanceof Element ? event.target.closest('.table-scroll, .index-table-scroll') : null;
  }, { passive: true });

  document.addEventListener('touchmove', event => {
    if (event.touches.length !== 1) return;

    const touch = event.touches[0];
    const dx = touch.clientX - startX;
    const dy = touch.clientY - startY;

    if (Math.abs(dx) <= Math.abs(dy)) return;

    if (!scroller) {
      event.preventDefault();
      return;
    }

    const maxScrollLeft = Math.max(0, scroller.scrollWidth - scroller.clientWidth);
    const atLeftEdge = scroller.scrollLeft <= 0.5;
    const atRightEdge = scroller.scrollLeft >= maxScrollLeft - 0.5;

    if ((atLeftEdge && dx > 0) || (atRightEdge && dx < 0)) {
      event.preventDefault();
    }
  }, { passive: false });
}

async function boot() {
  const app = document.getElementById('app');
  try {
    const res = await fetch(`./data/races.json?v=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const days = [...(data.days || [])].sort((a,b) => b.date.localeCompare(a.date));
    app.innerHTML = days.length ? days.map((day, index) => renderDay(day, index < 2)).join('') : '<div class="empty">表示できるレースがまだありません。</div>';
  } catch (e) {
    console.error(e);
    app.innerHTML = '<div class="error">レースデータを読み込めませんでした。data/races.json を確認してください。</div>';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const backToTop = document.getElementById('back-to-top');
  backToTop?.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
  bindDayToggles();
  bindIndexDetails();
  lockHorizontalPull();
  boot();
});
