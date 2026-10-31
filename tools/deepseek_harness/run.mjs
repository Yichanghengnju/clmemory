// Official SDK owns the native runtime and its model/tool loop.
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import fs from 'node:fs/promises';
const require = createRequire(path.join(process.env.AM_DSH_RUNTIME, 'package.json'));
const {DeepSeekHarness} = await import(pathToFileURL(require.resolve('@deepseek-ai/dsh-sdk-client')));
const input = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const harness = new DeepSeekHarness({profile:'sdk-minimal', patches:[input.patch],
  dshHome:input.home,processCwd:input.home,cwd:input.home,provider:'agentmemory-mini',
  model:input.model ?? 'gpt-4o-mini',maxTokens:input.max_tokens,initializeTimeoutMs:30000,
  requestTimeoutMs:input.timeout_ms,shutdownTimeoutMs:1000,
  env:{...process.env,DSH_TELEMETRY:'0'}});
let timer;
try {
  const result = await Promise.race([
    harness.run(input.prompt,{sessionId:input.session_id}),
    new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Native query deadline exceeded')),input.timeout_ms);}),
  ]);
  await fs.writeFile(input.result,JSON.stringify(result));
} catch (e) {
  await fs.writeFile(input.result,JSON.stringify({error:String(e)}));
  process.exitCode=1;
} finally { clearTimeout(timer); await harness.close(); }
