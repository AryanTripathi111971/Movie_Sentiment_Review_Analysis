(() => {
  "use strict";
  const SAMPLES = {
    pos: "I walked in expecting very little and walked out completely won over. The performances are warm and believable, the story is beautifully paced, and the final scene had the whole theatre quietly in tears. One of the best films I have seen this year, and I want to watch it again.",
    neg: "A dull, bloated mess. The plot makes no sense, the dialogue is painfully wooden, and the acting is awful from start to finish. I checked my watch every ten minutes and left feeling that I had wasted two hours of my life. Terrible film, avoid it.",
    mix: "The cinematography is gorgeous and the lead actor does his best, but the script is thin and the ending feels rushed. Some scenes work really well, others fall flat. It is not a bad film, just a forgettable one."
  };
  const $ = id => document.getElementById(id);
  const els = {
    review:$("review"), wordCount:$("wordCount"), analyze:$("analyze"), clear:$("clear"),
    result:$("result"), verdict:$("verdict"), verdictSub:$("verdictSub"), needle:$("needle"),
    scoreText:$("scoreText"), words:$("words"), legend:$("legend"), note:$("note"), error:$("error")
  };
  let inflight = null;
  const pct = (x, digits=0) => `${(x*100).toFixed(digits)}%`;
  const countWords = t => (t.trim().match(/\S+/g) || []).length;
  function showError(text) { els.error.textContent=text; els.error.hidden=!text; }
  function readError(body) {
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return "Write a review first (up to 20,000 characters).";
    return "";
  }
  async function analyze() {
    const text=els.review.value.trim();
    if (!text) { showError("Write a review first."); els.review.focus(); return; }
    if (inflight) inflight.abort();
    const controller=new AbortController(); inflight=controller;
    els.analyze.disabled=true; els.analyze.textContent="Analyzing…";
    els.result.classList.add("busy"); showError("");
    try {
      const res=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text}),signal:controller.signal});
      const body=await res.json().catch(()=>({}));
      if (!res.ok) throw new Error(readError(body) || `Request failed (${res.status}).`);
      render(body);
    } catch(err) {
      if (err.name==="AbortError") return;
      showError(err instanceof TypeError ? "Can't reach the server. Check that it is running." : err.message);
    } finally {
      if (inflight===controller) {
        inflight=null; els.analyze.disabled=false; els.analyze.textContent="Analyze review";
        els.result.classList.remove("busy");
      }
    }
  }
  function render(r) {
    const names={positive:"Positive",negative:"Negative",unclear:"Unclear"};
    els.verdict.textContent=names[r.label]; els.verdict.className=`verdict-word ${r.label}`;
    els.verdictSub.textContent=r.label==="unclear"
      ? `The model can't decide. It gives this review a ${pct(r.score)} chance of being positive.`
      : `The model is ${pct(r.confidence)} sure this review is ${r.label}.`;
    els.needle.style.left=`${(r.score*100).toFixed(1)}%`;
    els.scoreText.textContent=`${pct(r.score,1)} positive`;
    els.words.replaceChildren();
    r.tokens.forEach((t,i)=>{
      const span=document.createElement("span"); span.textContent=t.word;
      if (!t.known) { span.className="unk"; span.title="Not in the model's vocabulary"; }
      els.words.appendChild(span);
      if (i<r.tokens.length-1) els.words.appendChild(document.createTextNode(" "));
    });
    els.legend.hidden=r.unknown_count===0;
    const notes=[];
    if (r.truncated) notes.push(`Your review is long, so the model only read the last ${r.max_len} words.`);
    if (r.unknown_count>0 && r.unknown_count/Math.max(r.tokens.length,1)>0.3)
      notes.push("Many words are unknown to the model, so treat this result with care.");
    els.note.textContent=notes.join(" "); els.note.hidden=notes.length===0;
  }
  function reset() {
    if (inflight) inflight.abort();
    els.review.value=""; els.wordCount.textContent="0 words";
    els.verdict.textContent="No review yet"; els.verdict.className="verdict-word empty";
    els.verdictSub.textContent="Write a review or pick an example to see the result.";
    els.needle.style.left="50%"; els.scoreText.textContent="–";
    els.words.textContent="The words will appear here."; els.legend.hidden=true; els.note.hidden=true;
    els.analyze.disabled=false; els.analyze.textContent="Analyze review";
    els.result.classList.remove("busy"); showError(""); els.review.focus();
  }
  els.analyze.addEventListener("click",analyze);
  els.clear.addEventListener("click",reset);
  els.review.addEventListener("input",()=>{
    const n=countWords(els.review.value); els.wordCount.textContent=`${n} ${n===1?"word":"words"}`;
  });
  els.review.addEventListener("keydown",e=>{
    if ((e.ctrlKey||e.metaKey)&&e.key==="Enter") { e.preventDefault(); analyze(); }
  });
  document.querySelectorAll("[data-sample]").forEach(btn=>{
    btn.addEventListener("click",()=>{
      els.review.value=SAMPLES[btn.dataset.sample]; els.review.dispatchEvent(new Event("input")); analyze();
    });
  });
})();
