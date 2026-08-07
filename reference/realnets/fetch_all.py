#!/usr/bin/env python3
"""
fetch_all.py -- end-to-end acquisition of the real-world benchmark networks used in
the minimum-weight-cycle (weighted girth) experiments.

Run with:
    uv run --python 3.12 --with networkx,numpy,requests,osmnx python fetch_all.py

(osmnx is only needed for the OpenStreetMap network; pass --skip-osm to omit it.)

Every network is emitted in one uniform format:
  <name>.edges  ->  "u v w" per line, u,v in 0..n-1, w > 0, undirected, simple,
                    restricted to the largest connected component.
  <name>.json   ->  provenance metadata.
Plus manifest.json summarising all of them.

Design rules applied uniformly:
  * self-loops removed;
  * parallel edges collapsed by an explicitly recorded rule (min for cost-like
    weights such as distance/travel time, sum for count-like weights such as
    synapses or passengers);
  * directed sources symmetrised by the same rule;
  * largest connected component only;
  * node ids relabelled 0..n-1 in a deterministic sorted order of the original
    labels, so re-running reproduces byte-identical .edges files;
  * NO unit conversion -- weights are the source values, and the unit (or the
    fact that the source does not document one) is recorded in the metadata.
"""

import argparse
import bz2
import csv
import datetime as _dt
import gzip
import io
import json
import math
import os
import shutil
import sys
import tarfile
import zipfile
from collections import defaultdict

import networkx as nx
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
TODAY = _dt.date.today().isoformat()
os.makedirs(RAW, exist_ok=True)

# Fixed seed for the one synthetic weighting in the collection.
SYNTH_SEED = 20260806
SYNTH_DESC = ("i.i.d. uniform integer weights drawn from {1,...,1000}; numpy "
              "Generator(PCG64) seeded with 20260806; drawn in the sorted "
              "(u,v) order of the final relabelled LCC edge list, so the "
              "assignment is fully reproducible")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def fetch(url, fname, timeout=300):
    """Download url into raw/fname unless already present. Returns the path."""
    path = os.path.join(RAW, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    print(f"  downloading {url}", flush=True)
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    with open(path, "wb") as fh:
        for chunk in r.iter_content(1 << 16):
            fh.write(chunk)
    return path


def _sortkey(x):
    """Deterministic ordering over mixed int/str node labels."""
    try:
        return (0, int(x), "")
    except (TypeError, ValueError):
        return (1, 0, str(x))


def read_gml_text(text):
    """
    Parse GML text. Newman's celegansneural.gml contains genuinely duplicated
    (parallel) edges but does not declare `multigraph 1`, which NetworkX rejects.
    On that specific failure we re-parse with the multigraph flag injected, so the
    parallel edges are preserved rather than silently dropped.
    """
    lines = text.splitlines()
    try:
        return nx.parse_gml(lines, label="id"), False
    except nx.NetworkXError as exc:
        if "duplicated" not in str(exc):
            raise
        out, done = [], False
        for ln in lines:
            out.append(ln)
            if not done and ln.strip() == "[":
                out.append("  multigraph 1")
                done = True
        return nx.parse_gml(out, label="id"), True


def finalize(name, pairs, meta, out_dir=HERE):
    """
    pairs : dict {(a,b): w} with a,b original labels (unordered pair, already
            deduplicated and symmetrised by the caller), w > 0.
    meta  : provenance dict; n/m/mu are filled in here.
    """
    G = nx.Graph()
    for (a, b), w in pairs.items():
        if a == b:
            continue
        G.add_edge(a, b, weight=float(w))
    G.remove_edges_from(nx.selfloop_edges(G))

    n_before, m_before = G.number_of_nodes(), G.number_of_edges()
    ncomp = nx.number_connected_components(G)
    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()

    order = sorted(G.nodes(), key=_sortkey)
    idx = {u: i for i, u in enumerate(order)}
    edges = sorted(((min(idx[u], idx[v]), max(idx[u], idx[v]), d["weight"])
                    for u, v, d in G.edges(data=True)))

    if meta.get("synthetic_weighting"):
        rng = np.random.default_rng(SYNTH_SEED)
        w = rng.integers(1, 1001, size=len(edges))
        edges = [(u, v, float(w[i])) for i, (u, v, _) in enumerate(edges)]

    assert all(w > 0 for _, _, w in edges), f"{name}: non-positive weight"

    n, m = len(order), len(edges)
    meta.update(name=name, n=n, m=m, mu=m - n + 1, date_downloaded=TODAY)
    meta["preprocessing"] = (meta["preprocessing"]
                             + f" Graph before LCC extraction: {n_before} nodes, "
                               f"{m_before} edges, {ncomp} connected component(s); "
                               f"kept the largest ({n} nodes). "
                               "Node ids relabelled 0..n-1 in sorted order of the "
                               "original labels.")

    with open(os.path.join(out_dir, name + ".edges"), "w") as fh:
        for u, v, w in edges:
            fh.write(f"{u} {v} {w!r}\n" if isinstance(w, float) else f"{u} {v} {w}\n")
    with open(os.path.join(out_dir, name + ".json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"  -> {name}: n={n} m={m} mu={m-n+1} mu/n={(m-n+1)/n:.3f}", flush=True)
    return meta


# --------------------------------------------------------------------------- #
# 1. Les Miserables co-appearance network  (networkx built-in / Knuth SGB)
# --------------------------------------------------------------------------- #
def build_lesmis():
    G = nx.les_miserables_graph()
    pairs = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    return finalize("lesmis", pairs, dict(
        source_url="https://networkx.org/documentation/stable/reference/generated/"
                   "networkx.generators.social.les_miserables_graph.html "
                   "(original data: D. E. Knuth, The Stanford GraphBase, ACM Press 1993)",
        license_or_terms="Shipped with NetworkX (3-clause BSD). Original data from "
                         "Knuth's Stanford GraphBase, distributed freely for research use.",
        weight_semantics="number of chapters of Les Miserables in which the two "
                         "characters co-appear (integer count, >= 1)",
        weights_native=True,
        synthetic_weighting=None,
        preprocessing="Source is already undirected, simple and weighted; no "
                      "symmetrisation or deduplication was needed. No unit conversion."))


# --------------------------------------------------------------------------- #
# 2. C. elegans neural network  (Watts-Strogatz / White et al., via M. Newman)
# --------------------------------------------------------------------------- #
def build_celegans():
    url = "http://www-personal.umich.edu/~mejn/netdata/celegansneural.zip"
    p = fetch(url, "celegansneural.zip")
    with zipfile.ZipFile(p) as z:
        gml = z.read("celegansneural.gml").decode("latin-1")
    G, was_multi = read_gml_text(gml)
    assert G.is_directed()
    pairs, nself, npar = defaultdict(float), 0, 0
    it = (G.edges(data=True, keys=True) if G.is_multigraph()
          else ((u, v, None, d) for u, v, d in G.edges(data=True)))
    for u, v, _k, d in it:
        if u == v:
            nself += 1
            continue
        key = (min(u, v), max(u, v))
        if key in pairs:
            npar += 1
        pairs[key] += float(d.get("value", d.get("weight", 1)))
    return finalize("celegans-neural", dict(pairs), dict(
        source_url=url + "  (index page: http://www-personal.umich.edu/~mejn/netdata/)",
        license_or_terms="M. Newman's network data page states the files are made "
                         "freely available for research; cite White et al. (1986) "
                         "and Watts & Strogatz (1998).",
        weight_semantics="connection strength between two neurons, summed over both "
                         "directions (positive integer). Newman's accompanying "
                         "celegansneural.txt states only 'Edge weights are the weights "
                         "given by Watts'; in the underlying White et al. (1986) "
                         "wiring data this quantity is the number of synaptic "
                         "connections / gap junctions between the neuron pair. We "
                         "record the distributed value verbatim and did not "
                         "re-derive it from White et al.",
        weights_native=True,
        synthetic_weighting=None,
        preprocessing=(f"Source is a weighted DIRECTED graph in GML. It contains "
                       f"genuinely duplicated (parallel) arcs but omits the "
                       f"`multigraph 1` declaration, so it was re-parsed with that flag "
                       f"injected in order to preserve them (multigraph_reparse="
                       f"{was_multi}). Symmetrised by SUMMING the arc weights over both "
                       f"directions and over parallel arcs ({npar} arcs merged into an "
                       f"already-seen node pair). {nself} self-loops removed. "
                       f"No unit conversion.")))


# --------------------------------------------------------------------------- #
# 3/8/10. TNTP road networks (bstabler/TransportationNetworks)
# --------------------------------------------------------------------------- #
TNTP_BASE = ("https://raw.githubusercontent.com/bstabler/TransportationNetworks/"
             "master/")


def _parse_tntp(path):
    rows, hdr = [], {}
    with open(path, errors="replace") as fh:
        body = False
        for line in fh:
            line = line.strip().rstrip(";").strip()
            if not body:
                for tag, key in (("<NUMBER OF NODES>", "nodes"),
                                 ("<NUMBER OF LINKS>", "links"),
                                 ("<NUMBER OF ZONES>", "zones")):
                    if line.startswith(tag):
                        hdr[key] = int(line.split()[-1])
                if line.startswith("<END OF METADATA>"):
                    body = True
                continue
            if not line or line.startswith("~"):
                continue
            p = line.split()
            if len(p) < 5:
                continue
            try:
                rows.append((int(p[0]), int(p[1]), float(p[3])))   # init, term, length
            except ValueError:
                continue
    return hdr, rows


def build_tntp(name, repo_path, unit_desc, extra_note=""):
    url = TNTP_BASE + repo_path
    p = fetch(url, os.path.basename(repo_path))
    hdr, rows = _parse_tntp(p)
    pairs, npar, nself = {}, 0, 0
    for u, v, w in rows:
        if u == v:
            nself += 1
            continue
        assert w > 0, f"{name}: non-positive length in source"
        key = (min(u, v), max(u, v))
        if key in pairs:
            npar += 1
            pairs[key] = min(pairs[key], w)
        else:
            pairs[key] = w
    return finalize(name, pairs, dict(
        source_url=url,
        license_or_terms="No LICENSE file in the bstabler/TransportationNetworks "
                         "repository (GitHub reports license: null); the repository "
                         "and per-city READMEs ask users to cite the original data "
                         "providers. Treat as freely redistributable research data "
                         "with attribution.",
        weight_semantics=f"link length as distributed in the TNTP `length` column ({unit_desc})",
        weights_native=True,
        synthetic_weighting=None,
        preprocessing=(f"TNTP header declares {hdr.get('nodes')} nodes, "
                       f"{hdr.get('links')} directed links, {hdr.get('zones')} traffic "
                       f"analysis zones. The published network was kept intact -- the "
                       f"zone-centroid nodes (ids 1..{hdr.get('zones')}) and their "
                       f"artificial centroid-connector links were NOT removed. The "
                       f"`length` column is strictly positive on every link in this "
                       f"file (the `free_flow_time` column is not, which is why length "
                       f"was used). Source is DIRECTED: symmetrised by keeping the "
                       f"MINIMUM length among the directed arcs joining a node pair "
                       f"({npar} arcs collapsed into an existing pair); {nself} "
                       f"self-loops removed. No unit conversion. {extra_note}")))


# --------------------------------------------------------------------------- #
# 4. US airports 2010 (Opsahl, via KONECT)
# --------------------------------------------------------------------------- #
def build_usairport():
    url = "http://konect.cc/files/download.tsv.opsahl-usairport.tar.bz2"
    p = fetch(url, "usairport.tar.bz2")
    with tarfile.open(p, "r:bz2") as t:
        member = [m for m in t.getmembers() if m.name.endswith("out.opsahl-usairport")][0]
        raw = t.extractfile(member).read().decode("utf-8", "replace")
    pairs, npar, nself, nzero = defaultdict(float), 0, 0, 0
    for line in raw.splitlines():
        if not line.strip() or line.startswith("%"):
            continue
        p_ = line.split()
        u, v, w = int(p_[0]), int(p_[1]), float(p_[2])
        if w <= 0:
            nzero += 1
            continue
        if u == v:
            nself += 1
            continue
        key = (min(u, v), max(u, v))
        if key in pairs:
            npar += 1
        pairs[key] += w
    return finalize("usairport-2010", dict(pairs), dict(
        source_url="http://konect.cc/networks/opsahl-usairport  "
                   "(file: http://konect.cc/files/download.tsv.opsahl-usairport.tar.bz2; "
                   "original: https://toreopsahl.com/datasets/#usairports network 14b)",
        license_or_terms="KONECT does not attach an explicit licence to this network; "
                         "it asks for citation of Opsahl (2011) and the KONECT entry. "
                         "The underlying figures are US DOT Bureau of Transportation "
                         "Statistics T-100 data (US federal government, public domain).",
        weight_semantics="number of passengers carried between the two airports during "
                         "2010, summed over both directions (BTS Transtats T-100, "
                         "table id 292, Passengers column)",
        weights_native=True,
        synthetic_weighting=None,
        preprocessing=("Source is DIRECTED and positively weighted (KONECT header "
                       "'% asym posweighted', 28236 arcs on 1574 airports). Symmetrised "
                       f"by SUMMING the two directions ({npar} reciprocal/parallel arcs "
                       f"merged); {nself} self-loops and {nzero} non-positive-weight arcs "
                       "removed. NOTE ON SEMANTICS: the KONECT metadata describes the "
                       "weight as 'the number of flights on that connection'; Opsahl's "
                       "own dataset page, which is the authoritative description of the "
                       "2010 US airport network, states the weights are the Passengers "
                       "column of BTS T-100 with duplicate ties summed and zero-weight "
                       "(cargo-only) ties and self-loops already removed. We report "
                       "Opsahl's definition and did not independently re-derive the "
                       "figures from BTS. No unit conversion.")))


# --------------------------------------------------------------------------- #
# 5. OpenFlights airline route network, great-circle weights
# --------------------------------------------------------------------------- #
def build_openflights():
    a_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    r_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"
    ap = fetch(a_url, "airports.dat")
    rp = fetch(r_url, "routes.dat")

    coords = {}
    with open(ap, encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 8:
                continue
            try:
                coords[row[0].strip()] = (float(row[6]), float(row[7]))
            except ValueError:
                continue

    R = 6371.0088  # mean Earth radius, km (IUGG)

    def haversine(p, q):
        la1, lo1 = math.radians(p[0]), math.radians(p[1])
        la2, lo2 = math.radians(q[0]), math.radians(q[1])
        h = (math.sin((la2 - la1) / 2) ** 2
             + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
        return 2 * R * math.asin(min(1.0, math.sqrt(h)))

    pairs, nrows, nskip_id, nskip_stop, nself, nzero, ndup = {}, 0, 0, 0, 0, 0, 0
    with open(rp, encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 9:
                continue
            nrows += 1
            if row[7].strip() != "0":          # keep only non-stop routes
                nskip_stop += 1
                continue
            s, d = row[3].strip(), row[5].strip()
            if s not in coords or d not in coords:
                nskip_id += 1
                continue
            if s == d:
                nself += 1
                continue
            key = (min(s, d), max(s, d))
            if key in pairs:
                ndup += 1
                continue
            w = haversine(coords[s], coords[d])
            if w <= 0:
                nzero += 1
                continue
            pairs[key] = w
    return finalize("openflights-air", pairs, dict(
        source_url=f"{r_url} and {a_url}  (project page: https://openflights.org/data.php)",
        license_or_terms="OpenFlights Airport and Route databases are made available "
                         "under the Open Database License (ODbL); individual contents "
                         "under the Database Contents License. Attribution to "
                         "OpenFlights required; derived works must be openly licensed.",
        weight_semantics="great-circle (haversine) distance in kilometres between the "
                         "two airports' published latitude/longitude, mean Earth radius "
                         "R = 6371.0088 km",
        weights_native=False,
        synthetic_weighting=None,
        preprocessing=(f"routes.dat is a DIRECTED list of airline route legs. Read "
                       f"{nrows} route rows; kept only non-stop legs (stops == 0), "
                       f"discarding {nskip_stop}; discarded {nskip_id} legs whose "
                       f"OpenFlights airport id was missing from airports.dat; removed "
                       f"{nself} self-routes and {nzero} zero-distance pairs "
                       f"(co-located airports). Because the weight is a function of the "
                       f"two endpoints only, symmetrisation and multi-edge collapse are "
                       f"exact: {ndup} further route legs mapped onto an already-created "
                       f"undirected pair and were dropped without changing any weight. "
                       f"The edge SET is native (real airline routes); the WEIGHTS are "
                       f"computed by us from the real published airport coordinates -- "
                       f"they are a genuine physical quantity, not a synthetic/random "
                       f"weighting, but they are not shipped as a weight column by the "
                       f"source, hence weights_native=false. No unit conversion beyond "
                       f"degrees -> km via the haversine formula.")))


# --------------------------------------------------------------------------- #
# 6. Rome 1999 road network (9th DIMACS Implementation Challenge)
# --------------------------------------------------------------------------- #
def build_rome99():
    url = "https://www.diag.uniroma1.it/challenge9/data/rome/rome99.gr"
    p = fetch(url, "rome99.gr")
    pairs, npar, nself, nzero = {}, 0, 0, 0
    with open(p, errors="replace") as fh:
        for line in fh:
            if not line.startswith("a "):
                continue
            _, u, v, w = line.split()
            u, v, w = int(u), int(v), float(w)
            if w <= 0:
                nzero += 1
                continue
            if u == v:
                nself += 1
                continue
            key = (min(u, v), max(u, v))
            if key in pairs:
                npar += 1
                pairs[key] = min(pairs[key], w)
            else:
                pairs[key] = w
    return finalize("rome99-road", pairs, dict(
        source_url=url + "  (challenge page: https://www.diag.uniroma1.it/~challenge9/download.shtml)",
        license_or_terms="Distributed by the 9th DIMACS Implementation Challenge "
                         "(Shortest Paths) for free research use; contributed by "
                         "G. Storchi, P. Dell'Olmo and M. Gentili.",
        weight_semantics="physical road distance in METRES (stated verbatim in the "
                         "file header: 'Edge costs are physical distances in meters')",
        weights_native=True,
        synthetic_weighting=None,
        preprocessing=("DIMACS .gr arc list, DIRECTED (header 'p sp 3353 8870'). "
                       "Symmetrised by keeping the MINIMUM arc cost joining a node pair "
                       f"({npar} arcs collapsed into an existing pair -- almost all of "
                       f"these are the reverse arcs of two-way streets); {nself} "
                       f"self-loops and {nzero} non-positive-cost arcs removed. "
                       "No unit conversion.")))


# --------------------------------------------------------------------------- #
# 7. US Western States power grid  -- SYNTHETIC weights
# --------------------------------------------------------------------------- #
def build_powergrid():
    url = "http://www-personal.umich.edu/~mejn/netdata/power.zip"
    p = fetch(url, "power.zip")
    with zipfile.ZipFile(p) as z:
        gml = z.read("power.gml").decode("latin-1")
    G, _ = read_gml_text(gml)
    if G.is_directed():
        G = G.to_undirected()
    pairs = {(min(u, v), max(u, v)): 1.0 for u, v in G.edges() if u != v}
    return finalize("uspowergrid-synth", pairs, dict(
        source_url=url + "  (index page: http://www-personal.umich.edu/~mejn/netdata/)",
        license_or_terms="M. Newman's network data page states the files are made "
                         "freely available for research; cite Watts & Strogatz (1998).",
        weight_semantics="SYNTHETIC. The source graph is topological only (a "
                         "transmission line either exists or does not); no line length, "
                         "impedance or capacity is distributed with it.",
        weights_native=False,
        synthetic_weighting=SYNTH_DESC,
        preprocessing=("Topology of the Western States (USA) high-voltage power grid, "
                       "4941 nodes / 6594 undirected unweighted edges as distributed "
                       "in power.gml. Self-loops and duplicate edges removed (there are "
                       "none in the source). Weights are NOT from the source -- see "
                       "synthetic_weighting. Any weighted-girth number for this network "
                       "measures the synthetic weighting, not the physical grid.")))


# --------------------------------------------------------------------------- #
# 9. OpenStreetMap drivable road network of Portland, OR (osmnx)
# --------------------------------------------------------------------------- #
OSM_PLACE = "Portland, Oregon, USA"


def build_osm():
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.cache_folder = os.path.join(RAW, "osm_cache")
    G = ox.graph_from_place(OSM_PLACE, network_type="drive")
    pairs, npar, nself, nzero = {}, 0, 0, 0
    for u, v, d in G.edges(data=True):
        if u == v:
            nself += 1
            continue
        w = float(d.get("length", 0.0))
        if w <= 0:
            nzero += 1
            continue
        key = (min(u, v), max(u, v))
        if key in pairs:
            npar += 1
            pairs[key] = min(pairs[key], w)
        else:
            pairs[key] = w
    return finalize("osm-portland-drive", pairs, dict(
        source_url="OpenStreetMap via the Overpass API, queried with osmnx "
                   f"{ox.__version__}: ox.graph_from_place({OSM_PLACE!r}, "
                   "network_type='drive'). https://www.openstreetmap.org/",
        license_or_terms="(c) OpenStreetMap contributors, Open Database License (ODbL) "
                         "v1.0. Attribution required; derived databases must be shared "
                         "alike. https://www.openstreetmap.org/copyright",
        weight_semantics="geodesic length of the road segment in METRES, as computed by "
                         "osmnx from the OSM way geometry (the `length` edge attribute)",
        weights_native=True,
        synthetic_weighting=None,
        preprocessing=(f"osmnx returns a DIRECTED MULTIgraph of the drivable street "
                       f"network within the Portland, OR administrative boundary "
                       f"(one-way streets appear once, two-way streets twice). "
                       f"Symmetrised/simplified by keeping the MINIMUM `length` among "
                       f"all arcs joining a node pair ({npar} arcs collapsed into an "
                       f"existing pair); {nself} self-loops and {nzero} zero-length "
                       f"arcs removed. No unit conversion. REPRODUCIBILITY CAVEAT: "
                       f"OpenStreetMap is a live database -- re-running this query on a "
                       f"later date returns a slightly different graph. The .edges file "
                       f"in this directory is the snapshot actually used; the osmnx "
                       f"HTTP cache is kept under raw/osm_cache/.")))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
BUILDERS = [
    ("lesmis", build_lesmis),
    ("celegans-neural", build_celegans),
    ("chicago-sketch-road", lambda: build_tntp(
        "chicago-sketch-road", "Chicago-Sketch/ChicagoSketch_net.tntp",
        "MILES -- the in-file ORIGINAL HEADER reads 'length (miles)'",
        "Aggregated 'sketch planning' network of the Chicago region (Eash et al. 1983).")),
    ("usairport-2010", build_usairport),
    ("openflights-air", build_openflights),
    ("rome99-road", build_rome99),
    ("uspowergrid-synth", build_powergrid),
    ("chicago-regional-road", lambda: build_tntp(
        "chicago-regional-road", "chicago-regional/ChicagoRegional_net.tntp",
        "UNIT NOT DOCUMENTED by the source -- the in-file header is just 'length' and "
        "the repository README documents only Time: Minutes; the value range "
        "[0.02, 9.99] is consistent with miles but we do not assert this",
        "Detailed Chicago region network from the Chicago Area Transportation Study.")),
    ("osm-portland-drive", build_osm),
    ("sydney-road", lambda: build_tntp(
        "sydney-road", "Sydney/Sydney_net.tntp",
        "KILOMETRES -- the in-file ORIGINAL HEADER reads 'length (km)'",
        "Source: Veitch Lister Consultancy, Brisbane; provided by M. Bliemer, "
        "converted to TNTP by D. Rey.")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-osm", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated network names")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    metas = []
    for name, fn in BUILDERS:
        if only and name not in only:
            continue
        if args.skip_osm and name == "osm-portland-drive":
            continue
        print(f"[{name}]", flush=True)
        metas.append(fn())

    # merge with any pre-existing metadata so partial runs still yield a full manifest
    existing = {}
    mpath = os.path.join(HERE, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as fh:
            for e in json.load(fh)["networks"]:
                existing[e["name"]] = e
    for m in metas:
        existing[m["name"]] = m
    order = [n for n, _ in BUILDERS if n in existing]
    nets = [existing[n] for n in order]
    nets.sort(key=lambda m: m["n"])

    with open(mpath, "w") as fh:
        json.dump({
            "collection": "real-world weighted undirected simple graphs for "
                          "minimum-weight-cycle (weighted girth) experiments",
            "generated": TODAY,
            "format": "<name>.edges: one line per undirected edge, 'u v w', "
                      "u,v integer ids in 0..n-1, w > 0 float. Simple, connected "
                      "(largest component only).",
            "n_networks": len(nets),
            "networks": [{k: m[k] for k in
                          ("name", "n", "m", "mu", "weights_native",
                           "synthetic_weighting", "weight_semantics",
                           "source_url", "license_or_terms", "preprocessing",
                           "date_downloaded")} for m in nets],
        }, fh, indent=2)
        fh.write("\n")
    print(f"\nmanifest.json written with {len(nets)} networks")


if __name__ == "__main__":
    main()
