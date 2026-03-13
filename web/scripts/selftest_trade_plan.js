const http = require('http');
const assert = require('assert');

function getJson(baseUrl, path) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, baseUrl);
    const req = http.request(url, { method: 'GET' }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        try {
          const data = JSON.parse(body);
          if (res.statusCode && res.statusCode >= 400) {
            const msg = data && data.error ? String(data.error) : `HTTP ${res.statusCode}`;
            reject(new Error(msg));
            return;
          }
          resolve(data);
        } catch (e) {
          reject(e);
        }
      });
    });
    req.on('error', reject);
    req.end();
  });
}

async function main() {
  const app = require('../server');
  const { createTradePlanAll } = require('../tradePlan');
  const server = await new Promise((resolve, reject) => {
    const s = app.listen(0, '127.0.0.1', () => resolve(s));
    s.on('error', reject);
  });

  const { port } = server.address();
  const baseUrl = `http://127.0.0.1:${port}`;

  try {
    const categories = await getJson(baseUrl, '/api/categories');
    assert(Array.isArray(categories.categories), 'categories.categories 应为数组');
    const supported = new Set([
      '5.56x45mm',
      '.300BLK',
      '9x19mm',
      '9x39mm',
      '7.62x39mm',
      '7.62x51mm',
      '7.62x54R',
      '5.45x39mm',
      '5.7x28mm',
      '5.8x42mm',
      '6.8x51mm',
      '4.6x30mm',
      '12.7x55mm',
      '12 Gauge',
      '.357 Magnum',
      '45-70 Govt',
      '.45 ACP',
      '.50 AE',
      '箭矢'
    ]);
    const candidateCategory = categories.categories.find((c) => supported.has(c));
    assert(candidateCategory, '未找到任一支持模型的分类');

    const bullets = await getJson(baseUrl, `/api/bullets?category=${encodeURIComponent(candidateCategory)}`);
    assert(Array.isArray(bullets.bullets) && bullets.bullets.length > 0, 'bullets.bullets 应为非空数组');
    const bullet = bullets.bullets[0];

    const plan = await getJson(baseUrl, `/api/trade-plan/bullet?bullet=${encodeURIComponent(bullet)}&strategy=aggressive`);
    assert(plan && typeof plan === 'object', 'plan 应为对象');
    assert(plan.ok === true, 'plan.ok 应为 true');
    assert(plan.bullet === bullet, 'plan.bullet 应与请求一致');
    assert(typeof plan.modelGroup === 'string' && plan.modelGroup.length > 0, 'plan.modelGroup 应为字符串');
    assert(typeof plan.modelId === 'string' && plan.modelId.length > 0, 'plan.modelId 应为字符串');
    assert(Number.isFinite(plan.stackSize), 'plan.stackSize 应为数字');
    assert(Number.isFinite(plan.confidence), 'plan.confidence 应为数字');
    assert(Number.isFinite(plan.riskAdjustedProfitPerSlot), 'plan.riskAdjustedProfitPerSlot 应为数字');
    assert(Array.isArray(plan.trades), 'plan.trades 应为数组');
    assert(plan.strategy === 'aggressive', 'plan.strategy 应为 aggressive');

    const bullet2 = bullets.bullets.length > 1 ? bullets.bullets[1] : null;
    const selectedBullets = bullet2 ? [bullet, bullet2] : [bullet];
    const forecasts = new Map();
    for (const b of selectedBullets) {
      const f = await getJson(baseUrl, `/api/forecast?bullet=${encodeURIComponent(b)}`);
      assert(f && f.available === true, 'forecast.available 应为 true');
      forecasts.set(b, {
        available: true,
        bullet: f.bullet,
        modelGroup: f.modelGroup,
        modelId: f.modelId,
        predLen: f.predLen,
        points: f.points
      });
    }

    const dataParserStub = {
      getCategories() {
        return [candidateCategory];
      },
      getBulletsByCategory(category) {
        if (category !== candidateCategory) return [];
        return selectedBullets.slice();
      }
    };

    const planAll = await createTradePlanAll({
      dataParser: dataParserStub,
      forecastProvider: async (b, modelGroupHint) => {
        const f = forecasts.get(b);
        if (!f) return { available: false, bullet: b, reason: 'missing forecast' };
        if (modelGroupHint && f.modelGroup !== modelGroupHint) {
          return { ...f, modelGroup: modelGroupHint };
        }
        return f;
      },
      paramsOverride: { slotsTotal: 100, maxBullets: 5, minConfidence: 0, strategy: 'aggressive' }
    });
    assert(planAll && Array.isArray(planAll.plans), 'planAll.plans 应为数组');
    assert(planAll.params && planAll.params.strategy === 'aggressive', 'planAll.params.strategy 应为 aggressive');
    assert(planAll.slotsTotal === 100, 'planAll.slotsTotal 应为 100');
    assert(planAll.slotsUsed === 100 || planAll.plans.length === 0, 'planAll.slotsUsed 应尽量分满');
    for (const p of planAll.plans) {
      assert(Number.isFinite(p.positionRatio) && p.positionRatio >= 0 && p.positionRatio <= 1, 'positionRatio 应在 0..1');
    }

    await new Promise((resolve) => {
      process.stdout.write(`selftest ok: category=${candidateCategory}, bullet=${bullet}, trades=${plan.trades.length}, planAll=${planAll.plans.length}\n`, resolve);
    });
  } finally {
    await new Promise((resolve) => server.close(resolve));
    if (app && typeof app.shutdown === 'function') {
      await app.shutdown();
    }
  }
}

main()
  .catch(async (e) => {
    await new Promise((resolve) => {
      process.stderr.write(`selftest failed: ${e && e.stack ? e.stack : String(e)}\n`, resolve);
    });
    process.exitCode = 1;
  })
  .finally(() => {
    process.exit(process.exitCode || 0);
  });
