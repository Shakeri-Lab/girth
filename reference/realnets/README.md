# Real-world benchmark networks for minimum-weight-cycle (weighted girth) experiments

Ten real-world networks, all reduced to **weighted, undirected, simple, connected**
graphs with **strictly positive** edge weights — the exact input class the paper's
algorithm assumes. Acquired 2026-08-06.

Everything here is reproduced end to end by

```
uv run --python 3.12 --with networkx,numpy,requests,osmnx python fetch_all.py
uv run --python 3.12 --with networkx,numpy python validate.py
```

`fetch_all.py` caches raw downloads under `raw/` and is idempotent; `--skip-osm`
omits the one network that needs `osmnx`, and `--only NAME[,NAME]` rebuilds a subset.

## File format

For each network `<name>`:

* **`<name>.edges`** — one undirected edge per line, `u v w`, whitespace-separated.
  `u, v` are integers in `0..n-1`; `w` is a float `> 0`. Each edge appears exactly
  once, there are no self-loops and no parallel edges, and the graph is the largest
  connected component of the source.
* **`<name>.json`** — provenance: `name, n, m, mu, source_url, license_or_terms,
  weight_semantics, weights_native, synthetic_weighting, preprocessing, date_downloaded`.

`manifest.json` collects all ten. `mu = m - n + 1` is the cyclomatic number
(circuit rank), which for a connected graph equals the dimension of the cycle space.

## The collection

| network | n | m | mu | mu/n | weight semantics | weights |
|---|---:|---:|---:|---:|---|---|
| `lesmis` | 77 | 254 | 178 | 2.31 | chapters of *Les Misérables* in which two characters co-appear | native |
| `celegans-neural` | 297 | 2 148 | 1 852 | 6.24 | neuron-pair connection strength (synaptic count), summed over both directions | native |
| `chicago-sketch-road` | 933 | 1 475 | 543 | 0.58 | link length in **miles** | native |
| `usairport-2010` | 1 572 | 17 214 | 15 643 | 9.95 | passengers carried in 2010, summed over both directions | native |
| `openflights-air` | 3 188 | 18 833 | 15 646 | 4.91 | great-circle distance in **km** between the two airports' published coordinates | derived |
| `rome99-road` | 3 353 | 4 831 | 1 479 | 0.44 | physical road distance in **metres** | native |
| `uspowergrid-synth` | 4 941 | 6 594 | 1 654 | 0.34 | **none in the source** | **synthetic** |
| `chicago-regional-road` | 12 979 | 20 627 | 7 649 | 0.59 | link length, unit undocumented by the source | native |
| `osm-portland-drive` | 20 154 | 30 477 | 10 324 | 0.51 | road-segment geodesic length in **metres** | native |
| `sydney-road` | 32 956 | 38 787 | 5 832 | 0.18 | link length in **km** | native |

Nine of the ten carry weights that come from the data itself; the tenth
(`uspowergrid-synth`) is included only with an explicitly labelled synthetic
weighting and is named accordingly, so that a weighted-girth number reported for it
can never be mistaken for a physical measurement.

Density spans two orders of magnitude in `mu/n`: the road networks are near-planar
and cycle-poor (`mu/n` from 0.18 to 0.59), the biological and transport-flow
networks are cycle-rich (`mu/n` up to 9.95).

### Cost budgeting

The algorithm is `O(min{n, mu} · (m + n log n))`. `min{n, mu}` is the number of
Dijkstra-like passes actually executed:

| network | min(n, mu) | `min(n,mu)·(m + n log₂ n)` |
|---|---:|---:|
| `lesmis` | 77 | 5.7 × 10⁴ |
| `celegans-neural` | 297 | 1.4 × 10⁶ |
| `chicago-sketch-road` | 543 | 5.8 × 10⁶ |
| `usairport-2010` | 1 572 | 5.3 × 10⁷ |
| `rome99-road` | 1 479 | 6.5 × 10⁷ |
| `uspowergrid-synth` | 1 654 | 1.1 × 10⁸ |
| `openflights-air` | 3 188 | 1.8 × 10⁸ |
| `chicago-regional-road` | 7 649 | 1.5 × 10⁹ |
| `sydney-road` | 5 832 | 3.1 × 10⁹ |
| `osm-portland-drive` | 10 324 | 3.3 × 10⁹ |

The first seven are comfortable single-run targets in pure Python. The last three
are ~10⁹ operations and should be expected to take many minutes to hours in Python;
note that `sydney-road` is the largest network by `n` but *not* the most expensive,
because its cycle space is small — which is itself a point worth making in the
experiments section.

## Provenance, network by network

**`lesmis` — Les Misérables character co-appearance.** Shipped with NetworkX as
`networkx.les_miserables_graph()`; the data originate in D. E. Knuth, *The Stanford
GraphBase* (ACM Press, 1993), pp. 74–87. Already undirected, simple and weighted, so
no symmetrisation or deduplication was required. The weight is the number of chapters
in which the two characters both appear. Included as a small sanity case.
NetworkX is BSD-licensed; the GraphBase data are distributed freely for research.

**`celegans-neural` — *C. elegans* neural network.** `celegansneural.gml` from Mark
Newman's network data page, compiled by Watts and Strogatz from the White et al.
(1986) wiring reconstruction. The source is a *weighted directed* graph, and it
contains genuinely duplicated arcs while omitting the GML `multigraph 1`
declaration — NetworkX refuses to parse it as distributed, so `fetch_all.py`
re-parses with that flag injected in order to preserve the parallel arcs rather than
silently dropping them. Symmetrised by **summing** arc weights over both directions
and over parallel arcs; self-loops removed. Newman's accompanying `.txt` documents
the weights only as "the weights given by Watts"; in the underlying White et al.
data this is the number of synaptic connections and gap junctions between the pair.
We record the distributed value verbatim and did not re-derive it. Cite White et al.
(1986) and Watts & Strogatz (1998).

**`chicago-sketch-road`, `chicago-regional-road`, `sydney-road` — traffic-assignment
road networks.** From the `bstabler/TransportationNetworks` repository (the
maintained successor to Hillel Bar-Gera's TNTP archive). All three are TNTP
`*_net.tntp` files: directed link lists with `capacity`, `length`,
`free_flow_time`, and BPR parameters. We use the **`length`** column, which is
strictly positive on every link in all three files; the `free_flow_time` column is
*not* (it is zero on the artificial zone-centroid connector links in the two Chicago
files), which is why length was chosen. The published network is kept intact: the
traffic-analysis-zone centroid nodes and their connector links were **not** removed,
so `n` matches the TNTP header. Symmetrised by keeping the **minimum** length among
the directed arcs joining a node pair — the natural rule for a distance-like cost.
Units are taken from each file's own `ORIGINAL HEADER` line: miles for
Chicago-Sketch, km for Sydney. **Chicago Regional's distance unit is not documented**
by either the file header or the repository README (which documents only "Time:
Minutes"); the value range [0.02, 9.99] is consistent with miles but we do not assert
it, and no unit conversion was applied to any network. The repository has no LICENSE
file (GitHub reports `license: null`); it and its per-city READMEs ask users to cite
the original data providers — CATS for Chicago Regional, Eash et al. (1983) for
Chicago-Sketch, and Veitch Lister Consultancy / M. Bliemer / D. Rey for Sydney.

**`usairport-2010` — US airport passenger network.** KONECT's `opsahl-usairport`,
originally Tore Opsahl's network 14b. Directed and positively weighted (`% asym
posweighted`, 28 236 arcs on 1 574 airports); symmetrised by **summing** the two
directions, which is the natural rule for a flow count. Two airports fall outside
the largest component and are dropped. **A semantic discrepancy worth flagging:**
KONECT's metadata describes the weight as "the number of flights on that connection",
but Opsahl's own dataset page — the authoritative description of this 2010 network —
states that the weights are the *Passengers* column of the US DOT Bureau of
Transportation Statistics Transtats T-100 table (id 292), with duplicate ties summed
and zero-weight (cargo-only) ties and self-loops already removed. We report Opsahl's
definition and did not independently re-derive the figures from BTS. KONECT attaches
no explicit licence and asks for citation; the underlying BTS figures are US federal
government output.

**`openflights-air` — airline route network with great-circle weights.**
`routes.dat` and `airports.dat` from the OpenFlights GitHub snapshot. The **edge set
is native** (real scheduled airline route legs, restricted to non-stop legs with
`stops == 0`); the **weights are computed by us** as the haversine great-circle
distance in km between the two airports' published latitude/longitude, using mean
Earth radius R = 6 371.0088 km. This is a genuine physical quantity, not a synthetic
or random weighting, but the source ships no weight column, so `weights_native` is
recorded as `false` with `synthetic_weighting: null`. Route legs referencing an
airport id absent from `airports.dat`, self-routes, and zero-distance pairs
(co-located airports) are dropped; counts are in the `.json`. Because the weight is a
function of the endpoints alone, collapsing the directed multigraph is exact — no
weight is affected. OpenFlights data are under the Open Database License (ODbL) with
contents under the Database Contents License; attribution required and derived works
must be openly licensed.

**`rome99-road` — Rome 1999 road network.** `rome99.gr` from the 9th DIMACS
Implementation Challenge (Shortest Paths), contributed by G. Storchi, P. Dell'Olmo
and M. Gentili. The file header states verbatim that "Edge costs are physical
distances in meters". Directed (`p sp 3353 8870`); symmetrised by keeping the
**minimum** arc cost per node pair, which collapses the reverse arcs of two-way
streets. Free for research use.

**`uspowergrid-synth` — Western US power grid, synthetic weights.** `power.gml` from
Mark Newman's page: the topology of the Western States high-voltage transmission
grid used by Watts & Strogatz (1998), 4 941 buses and 6 594 lines, already undirected,
simple and connected. **The source is purely topological** — no line length,
impedance or capacity is distributed with it. We therefore attach an explicitly
synthetic weighting: **i.i.d. uniform integer weights drawn from {1, …, 1000}**, using
`numpy.random.default_rng(20260806)` (PCG64), drawn in the sorted `(u, v)` order of
the final relabelled edge list so the assignment is bit-reproducible. Any weighted
girth reported for this network is a property of that weighting, not of the physical
grid; it is included to exercise a large, very sparse, non-spatial topology and is
named `-synth` so this cannot be forgotten.

**`osm-portland-drive` — OpenStreetMap drivable street network, Portland, Oregon.**
Queried through `osmnx` 2.1.1 as
`ox.graph_from_place("Portland, Oregon, USA", network_type="drive")`. Weights are the
`length` edge attribute: the geodesic length of the road segment in metres, computed
by osmnx from the OSM way geometry. osmnx returns a directed multigraph (one-way
streets once, two-way streets twice, plus parallel ways); we keep the **minimum**
length among all arcs joining a node pair and drop self-loops and zero-length arcs.
**Reproducibility caveat:** OpenStreetMap is a live database, so re-running this query
at a later date returns a slightly different graph. The `.edges` file in this
directory is the snapshot actually used, and the osmnx HTTP cache is retained under
`raw/osm_cache/`. © OpenStreetMap contributors, ODbL v1.0.

## Uniform preprocessing rules

Applied identically to every network and recorded in each `.json`:

1. Self-loops removed.
2. Parallel edges and, for directed sources, reciprocal arcs collapsed into a single
   undirected edge by an explicitly recorded rule — **minimum** for cost-like weights
   (distance, travel time: the road networks, Rome, OSM) and **sum** for count-like
   weights (synapses, passengers).
3. Largest connected component only. The `.json` records the node/edge/component
   counts *before* the restriction.
4. Node ids relabelled `0..n-1` in a deterministic sorted order of the original
   labels, so re-running yields byte-identical `.edges` files.
5. **No unit conversion.** Weights are the source values; the unit — or the fact that
   the source does not document one — is recorded in `weight_semantics`.

## Validation

`validate.py` reloads every `.edges` file with NetworkX and asserts: undirected,
non-multigraph, connected, no self-loops, no repeated edges, node ids exactly
`0..n-1`, all weights `> 0`, and `n`/`m`/`mu` consistent with both `<name>.json` and
`manifest.json`. All ten networks pass.

## Candidates considered and dropped

* **`ca-GrQc` / `ca-HepTh` (SNAP collaboration networks)** — download cleanly but are
  natively unweighted; dropped to avoid a second synthetic weighting.
* **Bitcoin OTC / Bitcoin Alpha trust networks (SNAP)** — natively weighted, but the
  weights are trust ratings in [−10, 10] including negatives and zero, so they cannot
  be used without an arbitrary monotone transformation. Dropped.
* **Berlin-Center (TransportationNetworks)** — dropped: both `length` and
  `free_flow_time` are exactly zero on its centroid-connector links, so no column
  gives strictly positive weights on the published network.
* **Barcelona, Anaheim, Austin, Philadelphia (TransportationNetworks)** — usable, but
  redundant in size/density with the three road networks kept. Barcelona additionally
  documents no unit for `length`.
* **`astro-ph` / `cond-mat` weighted collaboration networks (Newman)** — genuinely
  weighted and available, but dense enough (`min{n, mu} = n ≈ 15 000–40 000` with
  `m ≈ 120 000–176 000`) that a single run is ~10¹⁰–10¹¹ operations. Dropped on cost,
  not availability.
* **9th DIMACS full US road networks (`USA-road-d.*`)** — real distance weights but
  the smallest (NY) has 264 000 nodes, well beyond the stated budget. `rome99` is the
  small member of the same family and was used instead.

## Things we could not verify

* The distance unit of `chicago-regional-road` is undocumented upstream (see above).
* The `usairport-2010` weight semantics are documented inconsistently by KONECT and
  by Opsahl; we follow Opsahl but did not re-derive the numbers from BTS T-100.
* `celegans-neural` weights are documented upstream only as "the weights given by
  Watts"; the synaptic-count interpretation comes from the White et al. source data,
  not from the distributed file.
* The `bstabler/TransportationNetworks` repository carries no licence file. We treat
  it as freely redistributable research data with attribution, which matches how it
  is used in the transportation literature, but this is an inference.
