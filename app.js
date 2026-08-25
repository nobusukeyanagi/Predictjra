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
// 過去レースの実人気は想定人気モデルの検証・学習用ラベルとしてのみ扱い、
// 対象となる将来レース自身の実人気は想人算出時に参照しない。

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
// 100-point evaluations are shown with exactly two digits. Internal values stay unchanged.
const score2 = value => {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  const shown = Math.max(0, Math.min(99, Math.round(n)));
  return String(shown).padStart(2, '0');
};

const RACE_DETAIL_CACHE = {};
let INDEX_MODAL_RACE_ORDER = [];
let activeIndexRaceId = null;
let lastIndexTrigger = null;


const INDEX_LOGIC_V2_HTML = `
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
  </section>`;

const INDEX_LOGIC_V3_HTML = `
  <section class="index-logic-item">
    <h3>基本</h3>
    <p>能力評価はすべて0〜100点です。近5走は各レースを「走・展・力」で評価し、1走評価＝走40％＋展25％＋力35％。近走総合＝前走35％＋2走前25％＋3走前18％＋4走前13％＋5走前9％（不足時は取得できた走だけで再正規化）。今走の走・展・力から今回＝走40％＋展25％＋力35％。最終の総合指数＝近走55％＋今回45％です。表示は2桁に統一し、100点は便宜上99、1桁は0を付けて表示します。順位判定には丸め前の内部値を使います。</p>
  </section>
  <section class="index-logic-item">
    <h3>近走「走」</h3>
    <p>競走タイムそのものの強さです。4着以下で実走タイムが3着タイム＋1.0秒より遅い場合、評価用タイム＝3着タイム＋1.0秒として、それ以上の大敗差は付けません。過去データに3着タイムがない場合は、勝ち馬とのタイム差のうち1.0秒を超える部分を評価用タイムから除く代替処理を使います。評価用1000m秒＝評価用タイム×1000÷距離。標準時計は、予想対象日より前に終了したレースだけから「競馬場×芝/ダート/障害×距離×馬場状態」ごとの3着1000m秒を集め、その中央値Mを使います。MAD＝median(|各3着1000m秒−M|)、σ＝max(0.20, 1.4826×MAD)。標準化値Z＝(M−評価用1000m秒)÷σ、走＝clamp(50＋15×Z, 0, 100)。基準群は①同競馬場・同路面・同距離・同馬場、②同競馬場・同路面・同距離、③全競馬場・同路面・同距離・同馬場、④全競馬場・同路面・同距離の順で、まず5レース以上、なければ3レース以上を採用します。履歴不足時だけ固定公開式へフォールバックし、基準1000m秒＝芝58.4＋0.0015×max(距離−1000,0)、ダート60.6＋0.0018×max(距離−1000,0)、障害64.0＋0.0007×max(距離−2500,0)、馬場補正は芝:稍重+0.6/重+1.4/不良+2.4、ダート:稍重−0.2/重−0.4/不良−0.2、障害:稍重+0.3/重+0.7/不良+1.0、σ=1.25とします。</p>
  </section>
  <section class="index-logic-item">
    <h3>近走「展」</h3>
    <p>展開に対してどれだけ頑張ったかです。頭数補正した前方度＝最初の2つまでの通過順位について平均{1−(通過順位−1)÷(頭数−1)}、追上げ度＝(最終通過順位−着順)÷(頭数−1)。過去レース全体の通過順位が取れる場合、前方半分の平均着順強度−後方半分の平均着順強度をレース前方バイアス（−1〜+1）とします。恩恵＝前方バイアス×(2×前方度−1)、不利=max(0,−恩恵)、有利=max(0,恩恵)。着順強度＝1−(着順−1)÷(頭数−1)。展＝50＋30×追上げ度＋30×不利×着順強度＋15×(着順強度−0.5)−12×有利。通過順位がない場合は50＋10×(着順強度−0.5)で補完し、0〜100に収めます。前残りで差して健闘、差し決着で前に残る、といった内容を高く評価します。</p>
  </section>
  <section class="index-logic-item">
    <h3>近走「力」</h3>
    <p>着順とレースレベルだけで能力を評価します。レース格点は、新馬・未勝利45、1勝55、2勝64、3勝73、OP・L82、GIII 88、GII 94、GI 100。着順点＝100×{1−(着順−1)÷(頭数−1)}。力＝レース格点50％＋着順点50％で、0〜100に収めます。</p>
  </section>
  <section class="index-logic-item">
    <h3>今回「走」</h3>
    <p>過去5走の「走」から今回の想定競走タイムの強さを作ります。各走の基礎重みは35・25・18・13・9％。同じ芝/ダート/障害なら表面係数1.00、異なる場合0.35、不明0.75。距離係数＝max(0.40, 1−|過去距離−今回距離|÷1200)（距離不明は0.70）。条件加重平均を算出し、今回走＝条件加重平均80％＋過去5走の走最高点20％。データがない場合は50です。</p>
  </section>
  <section class="index-logic-item">
    <h3>今回「展」</h3>
    <p>各馬の過去通過順位から前方度を求め、前方度0.62以上の馬が3頭以上なら速い流れ、1頭以下なら遅い流れ、それ以外を平均と想定します。速い流れは今回展＝35＋65×(1−前方度)、遅い流れは35＋65×前方度、平均は50＋20×{1−|前方度−0.5|×2}。脚質データがない場合は50です。ここは能力ではなく「今回どれだけ展開の恩恵を受けそうか」を表します。</p>
  </section>
  <section class="index-logic-item">
    <h3>今回「力」</h3>
    <p>基礎力は過去5走の「力」を35・25・18・13・9％で集約します。コース実績は、①同競馬場＋同芝/ダート/障害＋今回距離±100m、②同競馬場＋同芝/ダート/障害、③同芝/ダート/障害＋今回距離±200mの順に探し、その条件での1走総合評価を直近重視で集約します。該当実績がなければ基礎力をそのまま使います。今回力＝基礎力75％＋コース実績25％です。</p>
  </section>
  <section class="index-logic-item">
    <h3>想人・予想選定</h3>
    <p>想人は当日オッズ・当日実人気・馬体重/増減を使わず、過去人気・近走評価・騎手/調教師など事前情報から市場人気を推定します。想定1〜3番人気のうち総合指数が最も低い1頭を危険馬として除外し、残りから出走頭数の半数切り上げ・最大7頭を総合指数順に選定。本命は総合指数1位、対抗は選定馬のうち想定人気が最も低い馬、残りを相手とします。</p>
  </section>`;

function raceDetail(race) {
  if (!race?.raceId) return null;
  return RACE_DETAIL_CACHE[race.raceId]
    || race?.modelMeta?.indexDetail
    || RACE_INDEX_DETAILS[race.raceId]
    || null;
}

function predictionDisabledDetail(race) {
  const horseCount = Number(race?.horseCount || Object.keys(race?.horseNames || {}).length || Object.keys(race?.horseFrames || {}).length || 0);
  const names = race?.horseNames || {};
  const frameKeys = Object.keys(race?.horseFrames || {}).map(Number).filter(Number.isFinite);
  const nameKeys = Object.keys(names).map(Number).filter(Number.isFinite);
  const numbers = [...new Set([...nameKeys, ...frameKeys, ...Array.from({ length: horseCount }, (_, i) => i + 1)])]
    .filter(no => no > 0 && (!horseCount || no <= horseCount))
    .sort((a, b) => a - b);
  return {
    title: race?.modelMeta?.indexDetail?.title || `${race?.venue || ''}${race?.raceNo || ''}R ${race?.raceName || ''}`.trim(),
    horseCount: horseCount || numbers.length,
    logicVersion: race?.modelMeta?.version || '',
    predictionDisabled: true,
    prediction: { axes: [], opponents: [], excluded: [] },
    horses: numbers.map(no => ({
      no,
      name: String(names[String(no)] ?? names[no] ?? `馬番${no}`),
      expectedPopularity: 0, total: 0, rank: 0, recentIndex: 0, today: 0,
      recent: ['00/00/00','00/00/00','00/00/00','00/00/00','00/00/00'],
      todayParts: '00/00/00', pace: 0, course: 0, excluded: false
    }))
  };
}

function syncRaceDetailFromData(race, dayDate) {
  if (!race?.raceId) return;
  if (isDebutRace(race)) {
    const disabledDetail = predictionDisabledDetail(race);
    disabledDetail.date = dayDate || disabledDetail.date || '';
    RACE_DETAIL_CACHE[race.raceId] = disabledDetail;
    return;
  }
  const detail = race?.modelMeta?.indexDetail || RACE_INDEX_DETAILS[race.raceId];
  if (!detail) return;

  const popularity = race?.modelMeta?.estimatedPopularity || {};
  const danger = new Set(
    (race?.danger || race?.prediction?.excluded || detail?.prediction?.excluded || [])
      .map(Number)
  );

  detail.horseCount = Number(detail.horseCount || race.horseCount || detail.horses?.length || 0);
  detail.title = detail.title || `${race.venue}${race.raceNo}R`;
  detail.date = dayDate || detail.date || '';
  detail.logicVersion = race?.modelMeta?.version || detail.logicVersion || '';

  detail.horses.forEach(horse => {
    const rank = Number(popularity[String(horse.no)] ?? popularity[horse.no] ?? horse.expectedPopularity);
    if (Number.isFinite(rank) && rank > 0) horse.expectedPopularity = rank;
    horse.excluded = danger.has(Number(horse.no));
  });

  if (race?.prediction) {
    detail.prediction = {
      axes: [...(race.prediction.axes || [])],
      opponents: [...(race.prediction.opponents || [])],
      excluded: [...danger]
    };
  } else {
    detail.prediction = buildPredictionFromIndex(detail);
  }

  RACE_DETAIL_CACHE[race.raceId] = detail;
}

function effectiveHorseCount(race) {
  return raceDetail(race)?.horseCount || race?.horseCount;
}

function effectivePrediction(race) {
  if (isDebutRace(race)) return { axes: [], opponents: [], excluded: [] };
  return raceDetail(race)?.prediction || race?.prediction || { axes: [], opponents: [], excluded: [] };
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

function horseBox(no, race, extraClass = '') {
  const count = Number(effectiveHorseCount(race));
  const derived = frameNumber(no, count);
  const saved = race?.horseFrames?.[String(no)] ?? race?.horseFrames?.[no];
  const savedFrame = Number(saved);
  // For a normal JRA card the frame is deterministic from horse number + field size.
  // Prefer that value over a contradictory scraped value.  Stored data is used only
  // for legacy rows that do not have a usable horseCount.
  const hasReliableCount = Number.isInteger(count) && count >= Number(no) && count <= 18;
  const frame = hasReliableCount
    ? derived
    : (Number.isInteger(savedFrame) && savedFrame >= 1 && savedFrame <= 8 ? savedFrame : derived);
  const classes = [`horse-box`, `frame-${frame}`, extraClass].filter(Boolean).join(' ');
  return `<span class="${classes}" title="馬番 ${no} / ${frame}枠">${no}</span>`;
}

function predictionBoxes(numbers, race) {
  return `<div class="horses">${numbers.map(n => horseBox(n, race)).join('')}</div>`;
}

function resultRoleClass(no, race, prediction) {
  if (isDebutRace(race)) return '';
  const horseNo = Number(no);
  const main = Number(prediction?.axes?.[0]);
  const second = Number(prediction?.axes?.[1]);
  const danger = new Set([
    ...(Array.isArray(race?.danger) ? race.danger : []),
    ...(Array.isArray(race?.prediction?.excluded) ? race.prediction.excluded : []),
    ...(Array.isArray(prediction?.excluded) ? prediction.excluded : [])
  ].map(Number));
  if (horseNo === main) return 'result-role-main';
  if (horseNo === second) return 'result-role-second';
  if (danger.has(horseNo)) return 'result-role-danger';
  return '';
}

function resultBoxes(result, race, prediction) {
  if (!result?.places?.length) return '<span class="place-sep">—</span>';
  const box = no => horseBox(no, race, resultRoleClass(no, race, prediction));
  const groups = result.places.map(group => {
    if (group.length === 1) return box(group[0]);
    return `<span class="horses">${group.map(n => box(n)).join('<span class="place-sep">=</span>')}</span>`;
  });
  return `<div class="horses">${groups.join('<span class="place-sep">›</span>')}</div>`;
}

function mainHorseWon(race, prediction) {
  const main = Number(prediction?.axes?.[0]);
  if (!Number.isFinite(main)) return false;
  const firstPlace = race?.result?.places?.[0] || [];
  return firstPlace.map(Number).includes(main);
}

function isDebutRace(race) {
  if (race?.predictionDisabledReason === '新馬戦' || race?.predictionDisabled === true) return true;
  const title = race?.modelMeta?.indexDetail?.title || '';
  const raceName = race?.raceName || '';
  return String(title).includes('新馬') || String(raceName).includes('新馬');
}

function hasPublishedPrediction(race) {
  if (isDebutRace(race)) return false;
  return Array.isArray(race?.prediction?.axes) && race.prediction.axes.length === 2;
}

function predictionSettled(race) {
  return hasPublishedPrediction(race)
    && (race?.status === 'hit' || race?.status === 'miss')
    && Array.isArray(race?.result?.places)
    && race.result.places.length > 0;
}

function trifectaHit(race) {
  return predictionSettled(race) && Number(race?.payout || 0) > 0;
}

function winReturn(race, prediction = effectivePrediction(race)) {
  if (!predictionSettled(race)) return 0;
  const stored = Number(race?.winReturn);
  if (Number.isFinite(stored) && stored >= 0 && race?.result?.winPayouts) return stored;
  const main = Number(prediction?.axes?.[0]);
  if (!Number.isFinite(main)) return 0;
  return (race?.result?.winPayouts || []).reduce((sum, item) => {
    const horses = (item?.horses || []).map(Number);
    return sum + (horses.includes(main) ? Number(item?.payout || 0) : 0);
  }, 0);
}

function anyPredictionHit(race, prediction = effectivePrediction(race)) {
  if (!predictionSettled(race)) return false;
  return mainHorseWon(race, prediction) || trifectaHit(race);
}

function judgement(race, prediction) {
  if (isDebutRace(race)) return '';
  if (!predictionSettled(race)) return '<span class="judgement pending">未確定</span>';
  return anyPredictionHit(race, prediction)
    ? '<span class="judgement hit">的中</span>'
    : '';
}

function performanceSummary(races) {
  const predicted = races.filter(hasPublishedPrediction);
  const finished = predicted.filter(predictionSettled);
  const hits = finished.filter(r => anyPredictionHit(r, effectivePrediction(r))).length;
  const winPayout = finished.reduce((sum, r) => sum + winReturn(r, effectivePrediction(r)), 0);
  const winStake = finished.length * 100;
  const triPayout = finished.reduce((sum, r) => sum + Number(r.payout || 0), 0);
  const triStake = finished.reduce((sum, r) => sum + Number(r.stake || STAKE_PER_RACE), 0);
  return {
    hits,
    predictedCount: predicted.length,
    finishedCount: finished.length,
    winRecovery: winStake ? winPayout / winStake * 100 : 0,
    triRecovery: triStake ? triPayout / triStake * 100 : 0,
  };
}

function daySummary(day) {
  return performanceSummary(day.races || []);
}

function overallSummary(days) {
  const races = days.flatMap(day => day.races || []);
  return performanceSummary(races);
}

function dateLabel(iso) {
  const d = new Date(`${iso}T00:00:00+09:00`);
  const weekdays = ['日','月','火','水','木','金','土'];
  return `${iso}（${weekdays[d.getDay()]}）`;
}

function modalDateLabel(iso) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(String(iso || ''))) return '';
  const d = new Date(`${iso}T00:00:00+09:00`);
  const weekdays = ['日','月','火','水','木','金','土'];
  return `${String(iso).slice(5)}(${weekdays[d.getDay()]})`;
}

function actualPayoutAmounts(items) {
  if (!Array.isArray(items) || !items.length) return ['—'];
  return items.map(item => yen(Number(item?.payout || 0)));
}

function payoutAmountMarkup(items, rateText = '') {
  const amounts = actualPayoutAmounts(items);
  if (amounts.length <= 1) {
    return `<span class="payout-amount payout-line">${amounts[0]}</span>${rateText ? `<span class="payout-rate">${rateText}</span>` : ''}`;
  }
  const second = amounts.slice(1).join(' / ');
  return `<span class="payout-amount payout-line">${amounts[0]}</span><span class="payout-amount payout-line">${second}${rateText ? `<span class="payout-rate payout-rate-inline">${rateText}</span>` : ''}</span>`;
}

function raceNameCell(race) {
  const label = `<span class="venue">${race.venue}</span> ${race.raceNo}R`;
  if (!raceDetail(race)) return label;
  return `<button class="race-detail-trigger" type="button" data-race-id="${race.raceId}" aria-label="${race.venue}${race.raceNo}Rの指数表を表示">${label}</button>`;
}

function payoutCellClass({ debut, settled, hit }) {
  if (debut) return 'payout-miss';
  if (!settled) return 'payout-neutral';
  return hit ? 'payout-hit' : 'payout-miss';
}

function renderOverallSummary(days) {
  const summary = overallSummary(days);
  return `<section class="day-card overall-card" aria-label="総合成績">
    <div class="day-top overall-top">
      <div class="date-wrap"><span class="date-label">総合成績</span></div>
      <div class="day-summary">
        <div class="summary-item"><span class="summary-label">的中数</span><span class="summary-value">${summary.hits} / ${summary.finishedCount}</span></div>
        <div class="summary-item"><span class="summary-label">単回収率</span><span class="summary-value${summary.winRecovery > 100 ? ' summary-profit' : ''}">${percent(summary.winRecovery)}</span></div>
        <div class="summary-item"><span class="summary-label">三回収率</span><span class="summary-value${summary.triRecovery > 100 ? ' summary-profit' : ''}">${percent(summary.triRecovery)}</span></div>
      </div>
      <div class="day-toggle-spacer" aria-hidden="true"></div>
    </div>
  </section>`;
}

function renderDay(day, initiallyExpanded = true) {
  const summary = daySummary(day);
  const dl = dateLabel(day.date);
  const races = [...day.races].sort((a,b) => (VENUE_ORDER[a.venue] ?? 99) - (VENUE_ORDER[b.venue] ?? 99) || a.raceNo - b.raceNo);
  const rows = races.map(r => {
    const debut = isDebutRace(r);
    const prediction = effectivePrediction(r);
    const settled = debut
      ? Array.isArray(r?.result?.places) && r.result.places.length > 0
      : predictionSettled(r);
    const winHit = !debut && settled && mainHorseWon(r, prediction);
    const triHit = !debut && settled && trifectaHit(r);
    const anyHit = winHit || triHit;
    const triRate = !debut && settled
      ? (Number(r.payout || 0) / Number(r.stake || STAKE_PER_RACE) * 100)
      : null;
    const rowClass = anyHit ? 'hit-row' : (debut ? 'debut-row' : (settled ? 'miss-row' : ''));
    const singleMarkup = payoutAmountMarkup(r?.result?.winPayouts || []);
    const triMarkup = payoutAmountMarkup(
      r?.result?.trifectas || [],
      triHit && triRate != null ? percent(triRate) : ''
    );
    const mainCell = debut ? '<span class="no-prediction-dash">—</span>' : predictionBoxes(prediction.axes?.slice(0,1) || [], r);
    const secondCell = debut ? '<span class="no-prediction-dash">—</span>' : predictionBoxes(prediction.axes?.slice(1,2) || [], r);
    const opponentCell = debut ? '<span class="no-prediction-label">予想対象外</span>' : predictionBoxes(prediction.opponents || [], r);
    return `<tr class="${rowClass}">
      <td class="race-name">${raceNameCell(r)}</td>
      <td>${mainCell}</td>
      <td>${secondCell}</td>
      <td>${opponentCell}</td>
      <td>${resultBoxes(r.result, r, prediction)}</td>
      <td>${judgement(r, prediction)}</td>
      <td class="money payout-cell ${payoutCellClass({ debut, settled, hit: winHit })}">${singleMarkup}</td>
      <td class="money payout-cell trifecta-cell ${payoutCellClass({ debut, settled, hit: triHit })}">${triMarkup}</td>
    </tr>`;
  }).join('');

  const collapsedClass = initiallyExpanded ? '' : ' is-collapsed';
  return `<section class="day-card${collapsedClass}" data-day-date="${day.date}">
    <div class="day-top">
      <div class="date-wrap"><span class="date-label">${dl}</span></div>
      <div class="day-summary">
        <div class="summary-item"><span class="summary-label">的中数</span><span class="summary-value">${summary.hits} / ${summary.predictedCount}</span></div>
        <div class="summary-item"><span class="summary-label">単回収率</span><span class="summary-value${summary.winRecovery > 100 ? ' summary-profit' : ''}">${percent(summary.winRecovery)}</span></div>
        <div class="summary-item"><span class="summary-label">三回収率</span><span class="summary-value${summary.triRecovery > 100 ? ' summary-profit' : ''}">${percent(summary.triRecovery)}</span></div>
      </div>
      <button class="day-toggle" type="button" aria-label="${initiallyExpanded ? 'この日付を折りたたむ' : 'この日付を開く'}" aria-expanded="${initiallyExpanded ? 'true' : 'false'}">
        <span class="day-toggle-icon" aria-hidden="true"></span>
      </button>
    </div>
    <div class="day-content">
      <div class="table-scroll">
        <table class="race-table">
          <thead>
            <tr class="column-row"><th>レース</th><th>本命</th><th>対抗</th><th>相手</th><th>結果</th><th>判定</th><th>単勝</th><th>三連単</th></tr>
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
  if (detail?.predictionDisabled) return '<span class="index-pick pick-disabled">対象外</span>';
  if (horse.no === detail.prediction.axes[0]) return '<span class="index-pick pick-main">本命</span>';
  if (horse.no === detail.prediction.axes[1]) return '<span class="index-pick pick-second">対抗</span>';
  if (detail.prediction.opponents.includes(horse.no)) return '<span class="index-pick pick-opponent">相手</span>';
  if (horse.excluded) return '<span class="index-pick pick-danger">危険</span>';
  return '<span class="index-eval-empty">—</span>';
}

function tripleIndexMarkup(value, labels = ['走', '展', '力']) {
  if (value == null || value === '' || value === '評価外') return '<span class="index-recent-na">評価外</span>';
  const parts = String(value).split('/');
  if (parts.length !== 3) return value;
  return `<span class="index-recent-score">${parts.map((part, i) => `<span class="index-recent-part"><span class="index-recent-label">${labels[i]}</span>${score2(part)}</span>`).join('')}</span>`;
}

function recentIndexMarkup(value, detail) {
  const v3 = String(detail?.logicVersion || '').includes('v3-run-flow-power');
  return tripleIndexMarkup(value, v3 ? ['走', '展', '力'] : ['展', '時', '成']);
}

function todayIndexMarkup(horse) {
  if (horse?.todayParts) return tripleIndexMarkup(horse.todayParts, ['走', '展', '力']);
  // Before v3 apply, keep legacy production data truthful instead of relabeling course as power.
  if (Number.isFinite(Number(horse?.pace)) || Number.isFinite(Number(horse?.course))) {
    return `<span class="index-recent-score"><span class="index-recent-part"><span class="index-recent-label">展</span>${score2(horse?.pace)}</span><span class="index-recent-part"><span class="index-recent-label">コ</span>${score2(horse?.course)}</span></span>`;
  }
  return '<span class="index-recent-na">評価外</span>';
}

function recentSortValue(value, detail) {
  if (value === '評価外') return -1;
  const parts = String(value).split('/').map(Number);
  if (parts.length !== 3 || parts.some(Number.isNaN)) return -1;
  const v3 = String(detail?.logicVersion || '').includes('v3-run-flow-power');
  return v3
    ? parts[0] * 0.40 + parts[1] * 0.25 + parts[2] * 0.35
    : parts[0] * 0.25 + parts[1] * 0.35 + parts[2] * 0.40;
}

function evaluationSortValue(horse, detail) {
  if (detail?.predictionDisabled) return 5;
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
  const popularityText = detail?.predictionDisabled ? '00' : horse.expectedPopularity;
  const rankText = detail?.predictionDisabled ? '00' : horse.rank;
  return `
    <tr>
      <td class="index-evaluation" data-sort-value="${evaluationSortValue(horse, detail)}">${selectionLabel(horse, detail)}</td>
      <td data-sort-value="${horse.no}">${indexHorseNumber(horse.no, detail.horseCount)}</td>
      <td class="index-horse-name" data-sort-value="${horse.name}">${horse.name}</td>
      <td class="index-popularity" data-sort-value="${horse.expectedPopularity}">${popularityText}</td>
      <td class="index-total" data-sort-value="${horse.total}">${score2(horse.total)}</td>
      <td class="index-rank" data-sort-value="${horse.rank}">${rankText}</td>
      ${horse.recent.map(value => `<td data-sort-value="${recentSortValue(value, detail)}">${recentIndexMarkup(value, detail)}</td>`).join('')}
      <td class="index-strong index-recent-total" data-sort-value="${horse.recentIndex}">${score2(horse.recentIndex)}</td>
      <td data-sort-value="${horse.today}">${todayIndexMarkup(horse)}</td>
      <td class="index-strong index-today" data-sort-value="${horse.today}">${score2(horse.today)}</td>
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

function indexModalNeighbor(raceId, offset) {
  const index = INDEX_MODAL_RACE_ORDER.indexOf(raceId);
  if (index < 0) return null;
  return INDEX_MODAL_RACE_ORDER[index + offset] || null;
}

function renderIndexDetail(detail, raceId) {
  const rows = [...detail.horses]
    .sort((a, b) =>
      Number(b.total || 0) - Number(a.total || 0) ||
      Number(b.recentIndex || 0) - Number(a.recentIndex || 0) ||
      Number(a.no || 0) - Number(b.no || 0)
    )
    .map(horse => indexHorseRow(horse, detail))
    .join('');

  return `
    <div class="index-modal-backdrop" data-index-close="true">
      <section class="index-modal" role="dialog" aria-modal="true" aria-labelledby="index-modal-title">
        <div class="index-modal-header">
          <div class="index-modal-heading">
            <div class="index-modal-nav" aria-label="前後のレースへ移動">
              <button class="index-modal-nav-button" type="button" data-index-nav="-1" aria-label="前のレースへ"${indexModalNeighbor(raceId, -1) ? '' : ' disabled'}>＜</button>
              <button class="index-modal-nav-button" type="button" data-index-nav="1" aria-label="次のレースへ"${indexModalNeighbor(raceId, 1) ? '' : ' disabled'}>＞</button>
            </div>
            <h2 id="index-modal-title">${modalDateLabel(detail.date) ? `${modalDateLabel(detail.date)} ` : ''}${detail.title}</h2>
          </div>
          <button class="index-modal-close" type="button" data-index-close="true" aria-label="指数表を閉じる">×</button>
        </div>
        <div class="index-table-scroll">
          <table class="index-table">
            <thead>
              <tr>
                <th class="index-sortable" tabindex="0" role="button" aria-sort="none">評価</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none">馬番</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none">馬名</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none">想人</th><th class="index-sortable" tabindex="0" role="button" aria-sort="descending" data-sort-direction="desc" data-initial-sort="desc">総合</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none">順位</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">前走</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">2走前</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">3走前</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">4走前</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">5走前</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">近走</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">今走</th><th class="index-sortable" tabindex="0" role="button" aria-sort="none" data-initial-sort="desc">今回</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </section>
    </div>`;
}
function activeLogicInfoHtml() {
  const hasV3 = Object.values(RACE_DETAIL_CACHE).some(detail =>
    String(detail?.logicVersion || '').includes('v3-run-flow-power')
  );
  return hasV3 ? INDEX_LOGIC_V3_HTML : INDEX_LOGIC_V2_HTML;
}

function renderLogicInfo() {
  return `
    <div class="logic-modal-backdrop" data-logic-close="true">
      <section class="logic-modal" role="dialog" aria-modal="true" aria-labelledby="logic-modal-title">
        <div class="logic-modal-header">
          <h2 id="logic-modal-title">指数・予想ロジック</h2>
          <button class="logic-modal-close" type="button" data-logic-close="true" aria-label="ロジック解説を閉じる">×</button>
        </div>
        <div class="logic-modal-content">${activeLogicInfoHtml()}</div>
      </section>
    </div>`;
}

function openLogicInfo() {
  if (document.querySelector('.logic-modal-backdrop')) return;
  document.body.insertAdjacentHTML('beforeend', renderLogicInfo());
  document.body.classList.add('logic-modal-open');
  document.querySelector('.logic-modal-close')?.focus();
}

function closeLogicInfo() {
  document.querySelector('.logic-modal-backdrop')?.remove();
  document.body.classList.remove('logic-modal-open');
  document.getElementById('logic-info-button')?.focus();
}

function bindLogicInfo() {
  document.addEventListener('click', event => {
    const infoButton = event.target instanceof Element ? event.target.closest('#logic-info-button') : null;
    if (infoButton) {
      openLogicInfo();
      return;
    }

    const close = event.target instanceof Element ? event.target.closest('[data-logic-close="true"]') : null;
    if (!close) return;
    if (close.classList.contains('logic-modal-backdrop') && event.target !== close) return;
    closeLogicInfo();
  });

  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && document.querySelector('.logic-modal-backdrop')) {
      event.preventDefault();
      closeLogicInfo();
    }
  });
}

function openIndexDetail(raceId, trigger = null, focusTarget = 'close') {
  const detail = RACE_DETAIL_CACHE[raceId] || RACE_INDEX_DETAILS[raceId];
  if (!detail) return;

  // 前後遷移では元の一覧側フォーカス位置を保持したままモーダルだけ差し替える。
  document.querySelector('.index-modal-backdrop')?.remove();
  if (trigger) lastIndexTrigger = trigger;
  activeIndexRaceId = raceId;

  document.body.insertAdjacentHTML('beforeend', renderIndexDetail(detail, raceId));
  document.body.classList.add('index-modal-open');

  if (focusTarget === 'prev') {
    document.querySelector('.index-modal-nav-button[data-index-nav="-1"]:not(:disabled)')?.focus();
  } else if (focusTarget === 'next') {
    document.querySelector('.index-modal-nav-button[data-index-nav="1"]:not(:disabled)')?.focus();
  } else {
    document.querySelector('.index-modal-close')?.focus();
  }
}

function navigateIndexDetail(offset) {
  if (!activeIndexRaceId) return;
  const targetRaceId = indexModalNeighbor(activeIndexRaceId, offset);
  if (!targetRaceId) return;
  openIndexDetail(targetRaceId, null, offset < 0 ? 'prev' : 'next');
}

function closeIndexDetail(restoreFocus = true) {
  document.querySelector('.index-modal-backdrop')?.remove();
  document.body.classList.remove('index-modal-open');
  activeIndexRaceId = null;
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

    const navButton = event.target instanceof Element ? event.target.closest('.index-modal-nav-button') : null;
    if (navButton) {
      if (!navButton.disabled) navigateIndexDetail(Number(navButton.dataset.indexNav || 0));
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

    if (document.querySelector('.index-modal-backdrop') && event.key === 'ArrowLeft') {
      event.preventDefault();
      navigateIndexDetail(-1);
      return;
    }

    if (document.querySelector('.index-modal-backdrop') && event.key === 'ArrowRight') {
      event.preventDefault();
      navigateIndexDetail(1);
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
  let startScrollLeft = 0;
  let scroller = null;

  const clampScrollLeft = element => {
    if (!(element instanceof Element)) return;
    if (!element.matches('.table-scroll, .index-table-scroll')) return;
    const maxScrollLeft = Math.max(0, element.scrollWidth - element.clientWidth);
    if (element.scrollLeft < 0) element.scrollLeft = 0;
    if (element.scrollLeft > maxScrollLeft) element.scrollLeft = maxScrollLeft;
  };

  document.addEventListener('touchstart', event => {
    if (event.touches.length !== 1) return;
    const touch = event.touches[0];
    startX = touch.clientX;
    startY = touch.clientY;
    scroller = event.target instanceof Element
      ? event.target.closest('.table-scroll, .index-table-scroll')
      : null;
    startScrollLeft = scroller?.scrollLeft || 0;
    if (scroller) clampScrollLeft(scroller);
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
    const requestedScrollLeft = startScrollLeft - dx;

    if (requestedScrollLeft <= 0) {
      scroller.scrollLeft = 0;
      event.preventDefault();
      return;
    }

    if (requestedScrollLeft >= maxScrollLeft) {
      scroller.scrollLeft = maxScrollLeft;
      event.preventDefault();
    }
  }, { passive: false });

  document.addEventListener('touchend', () => {
    if (scroller) clampScrollLeft(scroller);
    scroller = null;
  }, { passive: true });

  document.addEventListener('touchcancel', () => {
    if (scroller) clampScrollLeft(scroller);
    scroller = null;
  }, { passive: true });

  // Safari can temporarily report negative / over-max scrollLeft while rubber-banding.
  // Clamp it immediately so the blank gutter is never exposed.
  document.addEventListener('scroll', event => {
    clampScrollLeft(event.target);
  }, true);
}

function syncDesktopRaceCardWidths() {
  const cards = [...document.querySelectorAll('.day-card')];

  if (window.innerWidth <= 760) {
    cards.forEach(card => { card.style.width = ''; });
    return;
  }

  const tables = [...document.querySelectorAll('.race-table')];
  if (!tables.length) return;

  // Measure the actual rendered table width. This changes only the outer card;
  // no table column width is recalculated or overridden.
  const widestTable = Math.ceil(Math.max(
    ...tables.map(table => table.getBoundingClientRect().width)
  ));
  const cardWidth = widestTable + 2; // include the day-card's left/right border.

  cards.forEach(card => {
    card.style.width = `${cardWidth}px`;
  });
}

async function boot() {
  const app = document.getElementById('app');
  try {
    const res = await fetch(`./data/races.json?v=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const days = [...(data.days || [])]
      .map(day => ({ ...day, races: [...(day.races || [])] }))
      .filter(day => day.races.length > 0)
      .sort((a,b) => b.date.localeCompare(a.date));
    days.forEach(day => (day.races || []).forEach(race => syncRaceDetailFromData(race, day.date)));

    INDEX_MODAL_RACE_ORDER = days.flatMap(day =>
      [...(day.races || [])]
        .sort((a,b) =>
          (VENUE_ORDER[a.venue] ?? 99) - (VENUE_ORDER[b.venue] ?? 99)
          || a.raceNo - b.raceNo
        )
        .filter(race => raceDetail(race))
        .map(race => race.raceId)
    );

    app.innerHTML = days.length
      ? `${renderOverallSummary(days)}${days.map((day, index) => renderDay(day, index < 2)).join('')}`
      : '<div class="empty">表示できるレースがまだありません。</div>';
    requestAnimationFrame(syncDesktopRaceCardWidths);
  } catch (e) {
    console.error(e);
    app.innerHTML = '<div class="error">レースデータを読み込めませんでした。data/races.json を確認してください。</div>';
  }
}

function initializeApp() {
  const backToTop = document.getElementById('back-to-top');
  backToTop?.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
  bindDayToggles();
  bindIndexDetails();
  bindLogicInfo();
  lockHorizontalPull();
  window.addEventListener('resize', syncDesktopRaceCardWidths, { passive: true });
  boot();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeApp, { once: true });
} else {
  initializeApp();
}
