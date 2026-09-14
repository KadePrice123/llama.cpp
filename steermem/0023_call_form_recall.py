"""fork edit 0023 (2026-09-14): a recall written in call form reaches the tool's memory.

With auto-inject on, the step-600 checkpoint read the injected block of calculate_averages -- a pointer ending "it
worked as calculate_averages(filtered_data=iot_sensor_data.json, interval=1min)" -- and then wrote
<recall>calculate_averages(filtered_data=iot_sensor_data.json, interval=1min)<recall>. The exact lookup missed, the
null block's nearest keys missed too (the arguments put the bare name beyond the edit-distance bar), and the model
answered that it did not have the tool it had just read. A recall key that is a stored key followed by a parenthesised
argument list now resolves to that key -- only when the key as written is not itself stored and the name before the
parenthesis is, exactly.

  python 0023_call_form_recall.py path/to/steermem_protocol.hpp
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()
if "0023" in s:
    raise SystemExit("0023 is already applied to %s" % p)
old = r'''    std::string k = proto_trim(asked);
    auto al = PROTO.alias.find(k);
    if (al != PROTO.alias.end()) k = al->second;
'''
new = r'''    std::string k = proto_trim(asked);
    auto al = PROTO.alias.find(k);
    if (al != PROTO.alias.end()) k = al->second;
    if (!k.empty() && k.back() == ')' && k.find('(') != std::string::npos) {   // 0023: a key written in call form, KEY(args)
        const std::string base = proto_trim(k.substr(0, k.find('(')));
        bool as_written = false, named = false;
        for (auto & c : E.cells) { if (c.key == k) as_written = true; if (c.key == base) named = true; }
        if (named && !as_written) k = base;
    }
'''
assert s.count(old) == 1, "anchor: %d matches" % s.count(old)
io.open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("0023 applied to %s" % p)
