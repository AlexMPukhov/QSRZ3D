"""Section 15: an openPMD diagnostic group (JSON backend) contains only the selected records.
usage: group_openpmd_check.py file.json step expected_meshes expected_species nxi_expected npart_expected"""
import json
import sys

fn, step = sys.argv[1], sys.argv[2]
want_m = sorted(sys.argv[3].split(","))
want_p = sorted(sys.argv[4].split(","))
nxi, npart = int(sys.argv[5]), int(sys.argv[6])
it = json.load(open(fn))["data"][step]
meshes = sorted(k + ("." + "".join(sorted(c for c in v if c in ("r", "t", "z"))) if any(c in v for c in ("r", "t", "z")) else "")
                for k, v in it["fields"].items() if k != "attributes")
species = sorted(k for k in it["particles"] if k != "attributes")
ok = meshes == want_m and species == want_p
shape_ok, nparts = True, 0
for k, v in it["fields"].items():
    if k == "attributes":
        continue
    d = v["z"]["data"] if "z" in v else v["data"]
    shape_ok = shape_ok and len(d[0][0]) == nxi
for s in species:
    nparts = len(it["particles"][s]["position"]["x"]["data"])
ok = ok and shape_ok and nparts == npart
print(f"  openPMD group: meshes {meshes}, species {species}, {nxi if shape_ok else 'wrong'} slices, {nparts} particles"
      f"  {'OK' if ok else 'FAIL (expected ' + str(want_m) + ' ' + str(want_p) + ')'}")
sys.exit(0 if ok else 1)
