const VENUE_ORDER = { '札幌': 1, '函館': 2, '福島': 3, '新潟': 4, '東京': 5, '中山': 6, '中京': 7, '京都': 8, '阪神': 9, '小倉': 10 };
const STAKE_PER_RACE = 3000;

/*
 * 想定人気（想人）モデル
 * ------------------------------------------------------------
 * 実運用時の入力に「当日オッズ・実人気・馬体重/増減」は使用しない。
 * 実人気は過去レースで係数を検証する教師ラベルとしてのみ利用する。
 *
 * popularityFactors は 0〜100 の事前情報スコア。
 * 各プロフィールで利用可能な要素だけを再正規化して加重平均するため、
 * レース区分や取得できるデータに応じて共通ロジックを流用できる。
 */
const POPULARITY_MODEL_VERSION = '2.0';
// 2026-08-15 / 2026-08-16 の実人気はモデル検証用ラベルとして扱い、
// 将来レースの想人算出時には参照しない。

const POPULARITY_PROFILES = {
  // OP・L・重賞：格、騎手、レーティングも市場評価に強く反映
  open: {
    recent: .32,
    class: .22,
    consistency: .06,
    jockey: .12,
    trainer: .06,
    rating: .08,
    upward: .08,
    age: .06
  },
  // 1勝・2勝・3勝：現級実績と上昇度を重視
  class: {
    recent: .42,
    class: .13,
    consistency: .11,
    jockey: .10,
    trainer: .05,
    rating: .05,
    upward: .10,
    age: .04
  },
  // 未勝利：前走〜2走前の見栄えと安定度を最重視
  maiden: {
    recent: .52,
    class: .04,
    consistency: .13,
    jockey: .12,
    trainer: .06,
    rating: .02,
    upward: .09,
    age: .02
  },
  // 障害：障害の直近内容と障害クラス実績を中心に評価
  jump: {
    recent: .50,
    class: .17,
    consistency: .10,
    jockey: .11,
    trainer: .05,
    rating: .04,
    upward: .03
  },
  // 新馬：近走がないため馬情報専用
  debut: {
    bloodline: .25,
    siblings: .15,
    jockey: .20,
    trainer: .20,
    breeder: .08,
    owner: .05,
    coursePedigree: .07
  }
};

function popularityScore(horse, detail) {
  const profileName = detail.popularityProfile || 'class';
  const weights = POPULARITY_PROFILES[profileName] || POPULARITY_PROFILES.class;
  const factors = horse.popularityFactors || {};
  let total = 0;
  let weightTotal = 0;

  Object.entries(weights).forEach(([key, weight]) => {
    const value = Number(factors[key]);
    if (!Number.isFinite(value)) return;
    total += value * weight;
    weightTotal += weight;
  });

  return weightTotal ? total / weightTotal : Number(horse.recentIndex || 0);
}

function assignExpectedPopularities(detail) {
  const ranked = detail.horses
    .map(horse => ({ horse, score: popularityScore(horse, detail) }))
    .sort((a, b) =>
      b.score - a.score ||
      Number(b.horse.recentIndex || 0) - Number(a.horse.recentIndex || 0) ||
      a.horse.no - b.horse.no
    );

  ranked.forEach(({ horse, score }, index) => {
    horse.popularityScore = Math.round(score);
    horse.expectedPopularity = index + 1;
  });
}

function predictionTargetCountForIndex(horseCount) {
  return Math.min(Math.ceil(Number(horseCount || 0) / 2), 7);
}

function buildPredictionFromIndex(detail) {
  const targetCount = predictionTargetCountForIndex(detail.horseCount);

  // 危険馬：想定1〜3番人気のうち総合評価が最も低い1頭。
  // 選定対象馬を決める前に除外する。
  const danger = detail.horses
    .filter(h => h.expectedPopularity <= 3)
    .sort((a, b) =>
      a.total - b.total ||
      a.recentIndex - b.recentIndex ||
      b.expectedPopularity - a.expectedPopularity ||
      b.no - a.no
    )[0];

  const selected = [...detail.horses]
    .filter(h => h.no !== danger?.no)
    .sort((a, b) =>
      b.total - a.total ||
      b.recentIndex - a.recentIndex ||
      a.no - b.no
    )
    .slice(0, targetCount);

  const main = selected[0];

  const second = [...selected]
    .filter(h => h.no !== main?.no)
    .sort((a, b) =>
      b.expectedPopularity - a.expectedPopularity ||
      b.total - a.total ||
      a.no - b.no
    )[0];

  const opponents = selected
    .filter(h => h.no !== main?.no && h.no !== second?.no)
    .sort((a, b) =>
      b.total - a.total ||
      b.recentIndex - a.recentIndex ||
      a.no - b.no
    )
    .map(h => h.no);

  detail.horses.forEach(h => { h.excluded = h.no === danger?.no; });

  return {
    axes: [main?.no, second?.no].filter(Boolean),
    opponents,
    excluded: danger ? [danger.no] : []
  };
}

function finalizeIndexDetail(detail) {
  assignExpectedPopularities(detail);
  detail.prediction = buildPredictionFromIndex(detail);
  return detail;
}

const RACE_INDEX_DETAILS = {
  '202601010811': {
    title: '札幌11R 札幌記念',
    horseCount: 16,
    popularityProfile: 'open',
    horses: [
      { no: 1, name: 'オニャンコポン', recent: ['88/79/83','61/72/55','63/69/50','64/61/48','66/74/60'], recentIndex: 68, popularityFactors: { recent: 66, class: 80, consistency: 60, jockey: 74, trainer: 72, rating: 76, upward: 60, age: 65 }, pace: 58, course: 64, today: 61, total: 65, rank: 14 },
      { no: 2, name: 'イガッチ', recent: ['52/45/44','74/79/73','88/84/80','73/73/66','83/73/68'], recentIndex: 66, popularityFactors: { recent: 68, class: 66, consistency: 74, jockey: 72, trainer: 70, rating: 70, upward: 82, age: 95 }, pace: 66, course: 62, today: 64, total: 65, rank: 15 },
      { no: 3, name: 'ピンクジン', recent: ['70/73/65','62/67/50','55/57/47','75/70/65','72/68/62'], recentIndex: 63, popularityFactors: { recent: 62, class: 62, consistency: 70, jockey: 65, trainer: 64, rating: 64, upward: 65, age: 80 }, pace: 61, course: 68, today: 65, total: 64, rank: 16 },
      { no: 4, name: 'マジックサンズ', recent: ['73/81/76','86/87/79','77/83/75','74/78/59','82/85/74'], recentIndex: 78, popularityFactors: { recent: 84, class: 86, consistency: 74, jockey: 88, trainer: 90, rating: 84, upward: 82, age: 95 }, pace: 82, course: 82, today: 82, total: 80, rank: 4 },
      { no: 5, name: 'エコロヴァルツ', recent: ['45/47/55','86/88/86','91/89/85','78/88/74','85/86/79'], recentIndex: 73, popularityFactors: { recent: 76, class: 88, consistency: 78, jockey: 83, trainer: 74, rating: 86, upward: 72, age: 90 }, pace: 81, course: 74, today: 78, total: 75, rank: 11 },
      { no: 6, name: 'ローシャムパーク', recent: ['89/88/86','77/82/65','49/51/52','84/90/78','77/82/68'], recentIndex: 75, popularityFactors: { recent: 78, class: 93, consistency: 70, jockey: 80, trainer: 88, rating: 93, upward: 70, age: 65 }, pace: 76, course: 70, today: 73, total: 74, rank: 13 },
      { no: 7, name: 'ショウヘイ', recent: ['63/73/69','93/91/94','59/67/61','91/94/91','92/95/93'], recentIndex: 79, popularityFactors: { recent: 82, class: 94, consistency: 83, jockey: 98, trainer: 95, rating: 91, upward: 88, age: 95 }, pace: 88, course: 68, today: 78, total: 79, rank: 5 },
      { no: 8, name: 'サクラファレル', recent: ['88/84/76','92/87/80','78/78/76','94/81/72','96/83/69'], recentIndex: 82, popularityFactors: { recent: 88, class: 78, consistency: 92, jockey: 94, trainer: 96, rating: 84, upward: 96, age: 95 }, pace: 91, course: 96, today: 94, total: 87, rank: 1 },
      { no: 9, name: 'マイネルモーント', recent: ['86/81/84','69/76/60','83/79/76','82/84/80','55/57/57'], recentIndex: 76, popularityFactors: { recent: 80, class: 78, consistency: 82, jockey: 80, trainer: 76, rating: 80, upward: 84, age: 80 }, pace: 78, course: 70, today: 74, total: 75, rank: 9 },
      { no: 10, name: 'アドマイヤテラ', recent: ['91/97/96','95/96/96','67/78/69','評価外','89/90/87'], recentIndex: 89, popularityFactors: { recent: 96, class: 98, consistency: 90, jockey: 91, trainer: 95, rating: 98, upward: 92, age: 90 }, pace: 77, course: 73, today: 75, total: 83, rank: 2 },
      { no: 11, name: 'アラタ', recent: ['67/73/61','75/79/66','67/73/57','82/78/76','91/86/86'], recentIndex: 71, popularityFactors: { recent: 70, class: 85, consistency: 67, jockey: 78, trainer: 72, rating: 83, upward: 60, age: 45 }, pace: 69, course: 93, today: 81, total: 75, rank: 10 },
      { no: 12, name: 'ゼンダンハヤブサ', recent: ['91/87/90','78/78/68','82/82/69','74/75/65','87/80/68'], recentIndex: 79, popularityFactors: { recent: 86, class: 76, consistency: 82, jockey: 76, trainer: 70, rating: 78, upward: 95, age: 95 }, pace: 73, course: 67, today: 70, total: 75, rank: 8 },
      { no: 13, name: 'グランディア', recent: ['91/90/92','84/84/82','89/88/88','89/86/86','54/61/49'], recentIndex: 85, popularityFactors: { recent: 90, class: 82, consistency: 92, jockey: 85, trainer: 98, rating: 83, upward: 92, age: 65 }, pace: 83, course: 69, today: 76, total: 81, rank: 3 },
      { no: 14, name: 'レディネス', recent: ['63/64/52','86/92/84','93/91/86','51/52/44','48/52/58'], recentIndex: 75, popularityFactors: { recent: 74, class: 80, consistency: 65, jockey: 78, trainer: 75, rating: 78, upward: 76, age: 95 }, pace: 84, course: 84, today: 84, total: 79, rank: 6 },
      { no: 15, name: 'シェイクユアハート', recent: ['50/49/57','95/94/95','87/90/85','94/93/94','88/88/88'], recentIndex: 79, popularityFactors: { recent: 80, class: 90, consistency: 80, jockey: 70, trainer: 72, rating: 92, upward: 78, age: 80 }, pace: 85, course: 70, today: 78, total: 78, rank: 7 },
      { no: 16, name: 'ホウオウビスケッツ', recent: ['58/74/62','42/47/51','73/86/68','94/91/91','78/81/73'], recentIndex: 67, popularityFactors: { recent: 70, class: 95, consistency: 72, jockey: 85, trainer: 82, rating: 93, upward: 65, age: 80 }, pace: 87, course: 84, today: 86, total: 74, rank: 12 }
    ]
  }
};

Object.values(RACE_INDEX_DETAILS).forEach(finalizeIndexDetail);

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
  if (horse.excluded) return '<span class="index-pick pick-danger">危険</span>';
  return '<span class="index-eval-empty">—</span>';
}

function recentIndexMarkup(value) {
  if (value === '評価外') return '<span class="index-recent-na">評価外</span>';
  const parts = String(value).split('/');
  if (parts.length !== 3) return value;
  return `<span class="index-recent-score"><span class="index-recent-part"><span class="index-recent-label">展</span>${parts[0]}</span><span class="index-recent-part"><span class="index-recent-label">時</span>${parts[1]}</span><span class="index-recent-part"><span class="index-recent-label">成</span>${parts[2]}</span></span>`;
}


function recentSortValue(value) {
  if (value === '評価外') return -1;
  const parts = String(value).split('/').map(Number);
  if (parts.length !== 3 || parts.some(Number.isNaN)) return -1;
  return parts[0] * 0.25 + parts[1] * 0.35 + parts[2] * 0.40;
}

function evaluationSortValue(horse, detail) {
  const firstAxis = detail.prediction?.axes?.[0];
  const secondAxis = detail.prediction?.axes?.[1];
  const opponents = detail.prediction?.opponents || [];
  const excluded = detail.prediction?.excluded || [];
  if (horse.no === firstAxis) return 1;
  if (horse.no === secondAxis) return 2;
  if (opponents.includes(horse.no)) return 3;
  if (excluded.includes(horse.no)) return 4;
  return 5;
}

function indexHorseRow(horse, detail) {
  return `
    <tr>
      <td class="index-evaluation" data-sort-value="${evaluationSortValue(horse, detail)}">${selectionLabel(horse, detail)}</td>
      <td data-sort-value="${horse.no}">${indexHorseNumber(horse.no, detail.horseCount)}</td>
      <td class="index-horse-name" data-sort-value="${horse.name}">${horse.name}</td>
      <td class="index-popularity" data-sort-value="${horse.expectedPopularity}">${horse.expectedPopularity}</td>
      <td class="index-total" data-sort-value="${horse.total}">${horse.total}</td>
      <td class="index-rank" data-sort-value="${horse.rank}">${horse.rank}</td>
      ${horse.recent.map(value => `<td data-sort-value="${recentSortValue(value)}">${recentIndexMarkup(value)}</td>`).join('')}
      <td class="index-strong index-recent-total" data-sort-value="${horse.recentIndex}">${horse.recentIndex}</td>
      <td data-sort-value="${horse.pace}">${horse.pace}</td>
      <td data-sort-value="${horse.course}">${horse.course}</td>
      <td class="index-strong index-today" data-sort-value="${horse.today}">${horse.today}</td>
    </tr>`;
}

function sortIndexTable(header) {
  const table = header.closest('.index-table');
  const tbody = table?.querySelector('tbody');
  if (!table || !tbody) return;

  const headers = [...table.querySelectorAll('thead th')];
  const columnIndex = headers.indexOf(header);
  if (columnIndex < 0) return;

  const previousDirection = header.dataset.sortDirection;
  const firstDirection = header.dataset.initialSort || 'asc';
  const direction = previousDirection
    ? (previousDirection === 'asc' ? 'desc' : 'asc')
    : firstDirection;

  headers.forEach(th => {
    delete th.dataset.sortDirection;
    th.setAttribute('aria-sort', 'none');
  });
  header.dataset.sortDirection = direction;
  header.setAttribute('aria-sort', direction === 'asc' ? 'ascending' : 'descending');

  const rows = [...tbody.querySelectorAll('tr')];
  rows.sort((a, b) => {
    const aCell = a.children[columnIndex];
    const bCell = b.children[columnIndex];
    const aRaw = aCell?.dataset.sortValue ?? '';
    const bRaw = bCell?.dataset.sortValue ?? '';

    const aNum = Number(aRaw);
    const bNum = Number(bRaw);
    let cmp;
    if (aRaw !== '' && bRaw !== '' && Number.isFinite(aNum) && Number.isFinite(bNum)) {
      cmp = aNum - bNum;
    } else {
      cmp = String(aRaw).localeCompare(String(bRaw), 'ja');
    }

    if (cmp === 0) {
      const aNo = Number(a.children[1]?.dataset.sortValue ?? 0);
      const bNo = Number(b.children[1]?.dataset.sortValue ?? 0);
      cmp = aNo - bNo;
    }
    return direction === 'asc' ? cmp : -cmp;
  });

  rows.forEach(row => tbody.appendChild(row));
}

function renderIndexDetail(detail) {
  const rows = [...detail.horses]
    .sort((a, b) => a.no - b.no)
    .map(horse => indexHorseRow(horse, detail))
    .join('');

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
                <th class="index-sortable" tabindex="0" role="button" aria-sort="none">評価</th><th class="index-sortable" tabindex="0" role="button" aria-sort="ascending" data-sort-direction="asc">馬番</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none">馬名</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none">想人</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">総合</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none">順位</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">前走</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">2走前</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">3走前</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">4走前</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">5走前</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">近走</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">展開</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">コース</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">今回</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <div class="index-logic">
          <section class="index-logic-item">
            <h3>評価</h3>
            <p>予想での役割を示します。まず想定1〜3番人気のうち総合評価が最も低い1頭を危険馬として選定対象から除外します。そのうえで、出走頭数の半数切り上げ・最大7頭を総合評価上位から選定対象馬とします。本命は選定対象馬の総合評価1位、対抗は選定対象馬のうち想定人気が最も低い馬、残りを相手とします。</p>
          </section>
          <section class="index-logic-item">
            <h3>想人</h3>
            <p>近走成績と馬情報から市場人気を推定した想定人気です。近走の着順・着差・レース格・安定度を中心に、騎手・調教師・レーティング・上昇度など事前に取得できる情報をレース区分に応じて評価します。当日オッズ、実際の人気、馬体重・馬体重増減は予測入力に使用しません。</p>
          </section>
          <section class="index-logic-item">
            <h3>総合</h3>
            <p>馬の近走能力と今回条件への適合度を統合した最終評価です。「近走」60％＋「今回」40％を基本として算出し、表示は小数点以下を丸めた整数とします。同点時の順位判定では、必要に応じて丸め前の内部値を使用します。</p>
          </section>
          <section class="index-logic-item">
            <h3>近走</h3>
            <p>近5走の能力評価です。各レースを「展開・タイム・成績」の3指数で個別評価し、直近を重視した基礎評価に、上位パフォーマンスから見た能力上限と再現性を加味します。長期休養明けや大幅な馬体重変動など、結果の信頼度を下げる客観的要因が重なった凡走は影響を抑え、一度の大敗だけで過度に評価を落とさないようにします。</p>
          </section>
          <section class="index-logic-item">
            <h3>今回</h3>
            <p>今回のレース条件に対する適合度です。「展開」50％＋「コース」50％を基本として算出します。「展開」は想定ペースと脚質・位置取りの適合度を評価し、「コース」は当該コース実績を最重視します。当該コース未経験の場合は、同距離の他場実績や類似条件への適応力で補完します。</p>
          </section>
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
    const sortHeader = event.target instanceof Element ? event.target.closest('.index-table th.index-sortable') : null;
    if (sortHeader) {
      sortIndexTable(sortHeader);
      return;
    }

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
    const sortHeader = event.target instanceof Element ? event.target.closest('.index-table th.index-sortable') : null;
    if (sortHeader && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      sortIndexTable(sortHeader);
      return;
    }

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
