import json, re, statistics as st
H="/Users/abundancemachine/streaming-machine/gpu_fr2/"
refs=json.load(open(H+"../refs_many.json"))
orig=[float(re.search(r"([\d.]+) words/s, prompt included", open(H+f"s_orig_{r}.log").read()).group(1)) for r in range(1,11)]
print(f"base2_stream.swift (one command buffer, so only prompt included): {st.median(orig):.1f} ({min(orig):.1f}-{max(orig):.1f})")
base=None
for m in ["fr3","fr2nofuse","fr2"]:
    J=[json.load(open(H+f"s_{m}_{r}.json")) for r in range(1,11)]
    inc=[j["words_per_s_prompt_included"] for j in J]; aft=[j["words_per_s_after_prompt"] for j in J]
    words=[[p["words"] for p in j["per_prompt"]] for j in J]
    if base is None: base=(st.median(inc), st.median(aft), words[0])
    same=[sum(a==b for a,b in zip(w,base[2])) for w in words]; fp=[sum(a==r["ref"][:16] for a,r in zip(w,refs)) for w in words]
    print(f"{m:10s} incl {st.median(inc):6.1f} ({min(inc):.1f}-{max(inc):.1f})  after {st.median(aft):6.1f} ({min(aft):.1f}-{max(aft):.1f})  "
          f"vs fr3: incl {100*(st.median(inc)/base[0]-1):+.1f}% after {100*(st.median(aft)/base[1]-1):+.1f}%  vs base2_stream.swift incl {100*(st.median(inc)/st.median(orig)-1):+.1f}%  "
          f"words = fr3 {min(same)}/30, = fp32 {min(fp)}/30  ms/token after {1e3/st.median(aft):.3f}")
