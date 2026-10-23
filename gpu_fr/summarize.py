import json, re, sys, statistics as st
H="/Users/abundancemachine/streaming-machine/gpu_fr/"; pre=sys.argv[1] if len(sys.argv)>1 else "u"
refs=json.load(open(H+"../refs_many.json"))
orig=[float(re.search(r"([\d.]+) words/s, prompt included", open(H+f"{pre}_orig_{r}.log").read()).group(1)) for r in range(1,6)]
print(f"base2_stream.swift as it is (one command buffer): prompt included {orig}  median {st.median(orig):.1f}")
for m in ["plain","plaine","fr2","fr2e","fr3","fr3e"]:
    try: J=[json.load(open(H+f"{pre}_{m}_{r}.json")) for r in range(1,6)]
    except FileNotFoundError: continue
    inc=[j["words_per_s_prompt_included"] for j in J]; aft=[j["words_per_s_after_prompt"] for j in J]
    words=[[p["words"] for p in j["per_prompt"]] for j in J]
    if m=="plain": base=words[0]; bi, ba = st.median(inc), st.median(aft)
    same=[sum(a==b for a,b in zip(w,base)) for w in words]; fp=[sum(a==r["ref"][:16] for a,r in zip(w,refs)) for w in words]
    print(f"{m:7s} incl {st.median(inc):6.1f} ({min(inc):.1f}-{max(inc):.1f})  after {st.median(aft):6.1f} ({min(aft):.1f}-{max(aft):.1f})  vs plain: incl {100*(st.median(inc)/bi-1):+.1f}% after {100*(st.median(aft)/ba-1):+.1f}%  words=plain {min(same)}/30  =fp32 {min(fp)}/30  weights/token {J[0]['weight_bytes_per_token']/1e6:.2f} MB")
