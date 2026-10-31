// Thin official dsh tool plugin. All scope and source checks live on the server.
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import {AsyncLocalStorage} from 'node:async_hooks';
const require = createRequire(path.join(process.env.AM_DSH_RUNTIME, 'package.json'));
const { defineTool } = await import(pathToFileURL(require.resolve('@deepseek-ai/dsh-tools')));
export const name = 'agentmemory-source-tools';
export const inject = ['tools','llm'];
export function apply(ctx) {
  if(process.env.AM_POOLED==='1'){
    // Out-of-band session capability. Keep frozen agent-loop messages unchanged.
    const sessions=new AsyncLocalStorage();
    const originalFetch=globalThis.fetch;
    const prefix=process.env.AM_GATEWAY_PREFIX;
    globalThis.fetch=(input,init)=>{
      const url=typeof input==='string'?input:input instanceof URL?input.href:input.url;
      if(url.startsWith(prefix+'/v1/')){
        const session=sessions.getStore();
        if(!session)throw new Error('Model transport has no bound session');
        const headers=new Headers(init?.headers ?? (input instanceof Request?input.headers:undefined));
        headers.set('X-AM-Session',session);
        return originalFetch(input,{...init,headers});
      }
      return originalFetch(input,init);
    };
    ctx.on('llm/stream',async function*(options,next){
      const session=String(options.sessionId ?? '');
      if(!session)throw new Error('Missing native session identity');
      const iterator=sessions.run(session,()=>next()[Symbol.asyncIterator]());
      try{
        while(true){const value=await sessions.run(session,()=>iterator.next());if(value.done)return;yield value.value;}
      }finally{if(iterator.return)await sessions.run(session,()=>iterator.return());}
    });
  }
  const tools = [
    ['hybrid_search', 'Search the shared hybrid index. Returns exact visible source text and IDs.', {query: {type:'string',required:true}}],
    ['grep', 'Literal case-insensitive search in visible source text. Empty matches are not proof of absence.', {pattern:{type:'string',required:true}}],
    ['read_source', 'Read exact source text. Only server-visible IDs in this query scope are accepted. Use offset to read beyond a clipped earlier observation.', {source_id:{type:'string',required:true},offset:{type:'integer'}}],
    ['finish_evidence', 'Explicitly select observed source IDs in relevance order. No answer prose. Then end your turn.', {source_ids:{type:'array',items:{type:'string'},required:true}}],
  ];
  if(process.env.AM_CHAIN_TOOL==='1')tools.push([
    'follow_evidence_chain',
    'From an already observed source, read bounded neighboring turns in the same session to resolve speaker, subject or correction context. Every internal read plus macro dispatch consumes the ordinary tool budget. No automatic final selection.',
    {source_id:{type:'string',required:true},radius:{type:'integer'},max_sources:{type:'integer'}},
  ]);
  for (const [name, description, parameters] of tools) {
    ctx.tools.register(defineTool({name,description,parameters,
      output:{schema:{type:'string'},render:(_args,value)=>[{type:'text',text:value}]},
      async execute(args, exec) {
        const r = await fetch(process.env.AM_TOOL_URL, {method:'POST',
          headers:{'Content-Type':'application/json',Authorization:'Bearer '+process.env.AM_CAPABILITY,
            ...(process.env.AM_POOLED==='1'?{'X-AM-Session':String(exec.agent?.session?.id ?? '')}:{})},
          body:JSON.stringify({name,args}),signal:exec.signal});
        if (!r.ok) throw new Error('Source tool rejected: HTTP '+r.status);
        return await r.text();
      },
    }));
  }
}
