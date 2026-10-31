// One official SDK runtime, multiple fresh sessions. No model/tool loop here.
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import fs from 'node:fs/promises';
import readline from 'node:readline';
const require=createRequire(path.join(process.env.AM_DSH_RUNTIME,'package.json'));
const {DeepSeekHarness}=await import(pathToFileURL(require.resolve('@deepseek-ai/dsh-sdk-client')));
const config=JSON.parse(await fs.readFile(process.argv[2],'utf8'));
const harness=new DeepSeekHarness({profile:'sdk-minimal',patches:[config.patch],
  dshHome:config.home,processCwd:config.home,cwd:config.home,provider:'agentmemory-mini',
  model:'gpt-4o-mini',maxTokens:config.max_tokens,initializeTimeoutMs:30000,
  requestTimeoutMs:config.timeout_ms,shutdownTimeoutMs:1000,
  env:{...process.env,DSH_TELEMETRY:'0'}});
const emit=value=>process.stdout.write(JSON.stringify(value)+'\n');
try {
  const started=performance.now();await harness.start();
  emit({ready:true,pid:process.pid,startup_s:(performance.now()-started)/1000});
  let completed=0;
  for await(const line of readline.createInterface({input:process.stdin,crlfDelay:Infinity})) {
    const input=JSON.parse(line);
    if(input.command==='close')break;
    if(input.command!=='run')throw new Error('Unknown worker command');
    let timer;
    try {
      const started=performance.now();
      const result=await Promise.race([harness.run(input.prompt,{sessionId:input.session_id}),
        new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Native query deadline exceeded')),input.timeout_ms);})]);
      await fs.writeFile(input.result,JSON.stringify(result));
      emit({complete:true,session_id:input.session_id,worker_session_number:++completed,
        sdk_run_s:(performance.now()-started)/1000,pid:process.pid});
    }catch(error){
      await fs.writeFile(input.result,JSON.stringify({error:String(error)}));
      emit({error:String(error),session_id:input.session_id,pid:process.pid});
      process.exitCode=1;break; // Never reuse a timed-out or failed runtime.
    }finally{clearTimeout(timer);}
  }
}catch(error){emit({error:String(error),pid:process.pid});process.exitCode=1;}
finally{await harness.close();}
