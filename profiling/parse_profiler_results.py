import json
import os
import glob

def parse_traces(output_dir: str = "profiling_output"):
    traces = glob.glob(os.path.join(output_dir, "*_trace.json"))
    if not traces:
        print(f"No traces found in {output_dir}")
        return
    
    for trace_file in traces:
        name = os.path.basename(trace_file).replace("_trace.json", "")
        print(f"\n=== Bottleneck Breakdown for {name} ===")
        with open(trace_file, "r") as f:
            data = json.load(f)
        
        events = data.get("traceEvents", [])
        nvtx_events = [e for e in events if e.get("cat") == "user_annotation" or "0" in e.get("name", "")]
        print(f"Found {len(events)} total trace events ({len(nvtx_events)} NVTX annotations).")

if __name__ == "__main__":
    parse_traces()
