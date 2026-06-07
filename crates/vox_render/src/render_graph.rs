//! CPU-first render-graph (RDG / FrameGraph analogue).
//!
//! Inspired by Unreal's RDG and Frostbite's FrameGraph: passes declare the
//! typed resources they READ and WRITE up front, and the graph — not the
//! caller — decides the schedule. From the declared reads/writes we build a
//! DAG (writer-before-reader edges), topologically sort it, detect cycles,
//! and CULL DEAD PASSES: any pass whose writes don't transitively reach a
//! requested output is dropped from the schedule (its name is still recorded
//! in `culled` for observability).
//!
//! The load-bearing contract is DECLARED-ACCESS ENFORCEMENT: a pass closure
//! may only `read`/`write` resources it declared. Touching an undeclared
//! resource PANICS with the offending pass + resource names. That panic is
//! what makes the DAG trustworthy — the declarations can't silently lie.
//!
//! Resources are CPU pixel buffers (`Vec<[f32;4]>`) for now; `ResourceFormat`
//! carries a `Spectral16` variant for forward-compat, but only `Rgba32F`
//! executes today. Buffers are allocated lazily on first use during execute.
//!
//! Canonical consumer: [`crate::postprocess::PostProcessPipeline::apply_via_graph`],
//! which expresses bloom → tonemap → vignette as a graph whose wiring adapts to
//! which effects are enabled, and is bit-identical to the legacy hardcoded chain.

use std::collections::{HashMap, HashSet};

/// Typed handle to a graph resource. Opaque newtype over an index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(pub u32);

/// Pixel format of a graph resource. Only `Rgba32F` executes today;
/// `Spectral16` exists for forward-compat with the spectral pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceFormat {
    Rgba32F,
    Spectral16,
}

/// Description of a CPU pixel buffer resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceDesc {
    pub width: usize,
    pub height: usize,
    pub format: ResourceFormat,
}

impl ResourceDesc {
    fn len(&self) -> usize {
        self.width * self.height
    }
}

/// Errors produced by [`RenderGraphBuilder::compile`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// The declared reads/writes form a cycle. Names the passes involved.
    Cycle(Vec<String>),
    /// A pass reads a resource that is neither imported nor written by any pass.
    ReadBeforeWrite { pass: String, resource: String },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::Cycle(passes) => {
                write!(f, "render graph cycle among passes: {}", passes.join(", "))
            }
            GraphError::ReadBeforeWrite { pass, resource } => write!(
                f,
                "pass '{pass}' reads resource '{resource}' before it is written or imported"
            ),
        }
    }
}

impl std::error::Error for GraphError {}

struct ResourceInfo {
    name: String,
    desc: ResourceDesc,
    /// Initial contents for imported resources; `None` for graph-internal ones.
    imported: Option<Vec<[f32; 4]>>,
}

struct PassDecl {
    name: String,
    reads: Vec<ResourceId>,
    writes: Vec<ResourceId>,
    exec: Box<dyn FnMut(&mut PassResources)>,
}

/// Builds a render graph from resource and pass declarations.
#[derive(Default)]
pub struct RenderGraphBuilder {
    resources: Vec<ResourceInfo>,
    passes: Vec<PassDecl>,
}

impl RenderGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a graph-internal resource. Returns its typed handle.
    pub fn create_resource(&mut self, name: &str, desc: ResourceDesc) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceInfo {
            name: name.to_string(),
            desc,
            imported: None,
        });
        id
    }

    /// Declare an external input resource seeded with `initial` contents.
    pub fn import_resource(
        &mut self,
        name: &str,
        desc: ResourceDesc,
        initial: Vec<[f32; 4]>,
    ) -> ResourceId {
        assert_eq!(
            initial.len(),
            desc.len(),
            "imported resource '{name}' initial buffer length {} != desc {}x{} = {}",
            initial.len(),
            desc.width,
            desc.height,
            desc.len()
        );
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceInfo {
            name: name.to_string(),
            desc,
            imported: Some(initial),
        });
        id
    }

    /// Declare a pass with its read set, write set, and execution closure.
    pub fn add_pass(
        &mut self,
        name: &str,
        reads: &[ResourceId],
        writes: &[ResourceId],
        exec: Box<dyn FnMut(&mut PassResources)>,
    ) {
        self.passes.push(PassDecl {
            name: name.to_string(),
            reads: reads.to_vec(),
            writes: writes.to_vec(),
            exec,
        });
    }

    /// Compile the declarations into an executable schedule.
    ///
    /// Topological order is derived from writer-before-reader edges. When two
    /// passes write the same resource, they are ordered by INSERTION ORDER
    /// (the order of `add_pass` calls), since neither reads the other's output.
    /// Passes that don't transitively feed `outputs` are culled.
    pub fn compile(self, outputs: &[ResourceId]) -> Result<RenderGraph, GraphError> {
        let RenderGraphBuilder { resources, passes } = self;
        let n = passes.len();

        // Which passes write each resource (in insertion order).
        let mut writers: HashMap<u32, Vec<usize>> = HashMap::new();
        for (pi, p) in passes.iter().enumerate() {
            for w in &p.writes {
                writers.entry(w.0).or_default().push(pi);
            }
        }

        // ReadBeforeWrite: a read of a resource that is neither imported nor
        // written by any pass is an error.
        for p in &passes {
            for r in &p.reads {
                let imported = resources[r.0 as usize].imported.is_some();
                let written = writers.contains_key(&r.0);
                if !imported && !written {
                    return Err(GraphError::ReadBeforeWrite {
                        pass: p.name.clone(),
                        resource: resources[r.0 as usize].name.clone(),
                    });
                }
            }
        }

        // Build edges: for each read, every writer of that resource must run
        // before the reader. Also serialize multiple writers of one resource
        // by insertion order (writer[i] before writer[i+1]).
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        let add_edge = |from: usize, to: usize, adj: &mut Vec<HashSet<usize>>| {
            if from != to {
                adj[from].insert(to);
            }
        };
        for (pi, p) in passes.iter().enumerate() {
            for r in &p.reads {
                if let Some(ws) = writers.get(&r.0) {
                    for &w in ws {
                        add_edge(w, pi, &mut adj);
                    }
                }
            }
        }
        for ws in writers.values() {
            for pair in ws.windows(2) {
                add_edge(pair[0], pair[1], &mut adj);
            }
        }

        // Cycle detection via DFS (iterative coloring). On detection, recover
        // the cycle path for the error message.
        if let Some(cycle) = detect_cycle(&adj, &passes) {
            return Err(GraphError::Cycle(cycle));
        }

        // Topological sort (Kahn), tie-broken by insertion order for stability.
        let mut indeg = vec![0usize; n];
        for set in &adj {
            for &to in set {
                indeg[to] += 1;
            }
        }
        let mut ready: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
        ready.sort_unstable();
        let mut topo: Vec<usize> = Vec::with_capacity(n);
        while let Some(&next) = ready.iter().min_by_key(|&&i| i) {
            ready.retain(|&i| i != next);
            topo.push(next);
            let mut succs: Vec<usize> = adj[next].iter().copied().collect();
            succs.sort_unstable();
            for s in succs {
                indeg[s] -= 1;
                if indeg[s] == 0 {
                    ready.push(s);
                }
            }
        }
        // Kahn completeness is guaranteed because cycles were already rejected.
        debug_assert_eq!(topo.len(), n);

        // Dead-pass culling: a pass is LIVE if any of its writes is an output,
        // or if it is an ancestor of a live pass. Walk backwards over edges.
        let output_ids: HashSet<u32> = outputs.iter().map(|o| o.0).collect();
        let mut live = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        for (pi, p) in passes.iter().enumerate() {
            if p.writes.iter().any(|w| output_ids.contains(&w.0)) && !live[pi] {
                live[pi] = true;
                stack.push(pi);
            }
        }
        // Reverse adjacency for ancestor walk.
        let mut radj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (from, set) in adj.iter().enumerate() {
            for &to in set {
                radj[to].push(from);
            }
        }
        while let Some(pi) = stack.pop() {
            for &pred in &radj[pi] {
                if !live[pred] {
                    live[pred] = true;
                    stack.push(pred);
                }
            }
        }

        let mut schedule: Vec<usize> = Vec::new();
        let mut culled: Vec<String> = Vec::new();
        for &pi in &topo {
            if live[pi] {
                schedule.push(pi);
            } else {
                culled.push(passes[pi].name.clone());
            }
        }

        Ok(RenderGraph {
            resources,
            passes,
            schedule,
            culled,
            trace: Vec::new(),
            buffers: None,
        })
    }
}

/// Detect a cycle in `adj`; return the cycle's pass names if one exists.
fn detect_cycle(adj: &[HashSet<usize>], passes: &[PassDecl]) -> Option<Vec<String>> {
    let n = adj.len();
    // 0 = white, 1 = gray (on stack), 2 = black (done).
    let mut color = vec![0u8; n];
    let mut parent = vec![usize::MAX; n];
    for start in 0..n {
        if color[start] != 0 {
            continue;
        }
        // Iterative DFS.
        let mut stack: Vec<(usize, Vec<usize>)> =
            vec![(start, adj[start].iter().copied().collect())];
        color[start] = 1;
        while let Some((node, succs)) = stack.last_mut() {
            let node = *node;
            if let Some(next) = succs.pop() {
                match color[next] {
                    0 => {
                        color[next] = 1;
                        parent[next] = node;
                        stack.push((next, adj[next].iter().copied().collect()));
                    }
                    1 => {
                        // Back edge node->next: cycle from next..node.
                        let mut cyc = vec![node];
                        let mut cur = node;
                        while cur != next && parent[cur] != usize::MAX {
                            cur = parent[cur];
                            cyc.push(cur);
                        }
                        cyc.reverse();
                        return Some(cyc.into_iter().map(|i| passes[i].name.clone()).collect());
                    }
                    _ => {}
                }
            } else {
                color[node] = 2;
                stack.pop();
            }
        }
    }
    None
}

/// Resource access surface handed to a pass closure during execution.
///
/// Enforces declared access: `read`/`write` panic (naming the pass and
/// resource) if the requested resource wasn't declared in the pass's
/// read/write set.
pub struct PassResources<'a> {
    pass_name: &'a str,
    reads: &'a [ResourceId],
    writes: &'a [ResourceId],
    names: &'a [String],
    buffers: &'a mut HashMap<u32, Vec<[f32; 4]>>,
}

impl PassResources<'_> {
    /// Borrow a declared read resource. Panics on undeclared access.
    pub fn read(&self, id: ResourceId) -> &[[f32; 4]] {
        if !self.reads.contains(&id) {
            panic!(
                "pass '{}' accessed undeclared READ resource '{}'",
                self.pass_name,
                self.resource_name(id)
            );
        }
        self.buffers
            .get(&id.0)
            .unwrap_or_else(|| {
                panic!(
                    "pass '{}' read resource '{}' that has no allocated buffer",
                    self.pass_name,
                    self.resource_name(id)
                )
            })
            .as_slice()
    }

    /// Mutably borrow a declared write resource. Panics on undeclared access.
    pub fn write(&mut self, id: ResourceId) -> &mut [[f32; 4]] {
        if !self.writes.contains(&id) {
            panic!(
                "pass '{}' accessed undeclared WRITE resource '{}'",
                self.pass_name,
                self.resource_name(id)
            );
        }
        self.buffers
            .get_mut(&id.0)
            .unwrap_or_else(|| {
                panic!(
                    "pass '{}' wrote resource '{}' that has no allocated buffer",
                    self.pass_name,
                    "<unknown>"
                )
            })
            .as_mut_slice()
    }

    fn resource_name(&self, id: ResourceId) -> &str {
        self.names
            .get(id.0 as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unknown>")
    }
}

/// A compiled, executable render graph.
pub struct RenderGraph {
    resources: Vec<ResourceInfo>,
    passes: Vec<PassDecl>,
    schedule: Vec<usize>,
    culled: Vec<String>,
    trace: Vec<String>,
    /// Holds executed buffers between `execute()` and `take_output()`.
    buffers: Option<HashMap<u32, Vec<[f32; 4]>>>,
}

impl std::fmt::Debug for RenderGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderGraph")
            .field("scheduled", &self.schedule.len())
            .field("culled", &self.culled)
            .field("trace", &self.trace)
            .finish()
    }
}

impl RenderGraph {
    /// Run the scheduled passes in topological order, allocating buffers
    /// lazily on first use. Imported resources start from their seeded
    /// contents; internal resources start zeroed on first write.
    pub fn execute(&mut self) {
        self.trace.clear();
        let mut buffers: HashMap<u32, Vec<[f32; 4]>> = HashMap::new();

        // Seed imported buffers.
        for (i, info) in self.resources.iter().enumerate() {
            if let Some(init) = &info.imported {
                buffers.insert(i as u32, init.clone());
            }
        }

        let names: Vec<String> = self.resources.iter().map(|r| r.name.clone()).collect();

        for &pi in &self.schedule {
            // Lazily allocate any read/write buffers not yet present.
            let pass = &self.passes[pi];
            for id in pass.reads.iter().chain(pass.writes.iter()) {
                buffers.entry(id.0).or_insert_with(|| {
                    let desc = self.resources[id.0 as usize].desc;
                    vec![[0.0f32; 4]; desc.len()]
                });
            }

            let reads = pass.reads.clone();
            let writes = pass.writes.clone();
            let pass_name = pass.name.clone();
            self.trace.push(pass_name.clone());

            let mut res = PassResources {
                pass_name: &pass_name,
                reads: &reads,
                writes: &writes,
                names: &names,
                buffers: &mut buffers,
            };
            // `res` borrows locals only (buffers/names/reads/...); the closure
            // borrows `self.passes[pi]`. Disjoint, so both borrows coexist.
            (self.passes[pi].exec)(&mut res);
        }

        self.buffers = Some(buffers);
    }

    /// Take an output buffer after [`execute`]. Returns the resource contents.
    pub fn take_output(&mut self, id: ResourceId) -> Vec<[f32; 4]> {
        self.buffers
            .as_mut()
            .expect("take_output called before execute")
            .remove(&id.0)
            .unwrap_or_else(|| panic!("output resource {} was not produced", id.0))
    }

    /// Pass names that ran, in execution order. Frame-debugger hook + tests.
    pub fn execution_trace(&self) -> &[String] {
        &self.trace
    }

    /// Pass names dropped by dead-pass culling, for observability.
    pub fn culled(&self) -> &[String] {
        &self.culled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(w: usize, h: usize) -> ResourceDesc {
        ResourceDesc {
            width: w,
            height: h,
            format: ResourceFormat::Rgba32F,
        }
    }

    #[test]
    fn dead_pass_culling_drops_unconsumed_pass_and_preserves_output() {
        // Graph WITH a dead pass: import in -> copy to out (live);
        // separate dead pass writes `garbage` that nobody reads.
        let build = |with_dead: bool| {
            let mut b = RenderGraphBuilder::new();
            let input = b.import_resource("in", desc(2, 1), vec![[1.0, 2.0, 3.0, 4.0]; 2]);
            let out = b.create_resource("out", desc(2, 1));
            b.add_pass(
                "copy",
                &[input],
                &[out],
                Box::new(move |r| {
                    let src: Vec<[f32; 4]> = r.read(input).to_vec();
                    r.write(out).copy_from_slice(&src);
                }),
            );
            if with_dead {
                let garbage = b.create_resource("garbage", desc(2, 1));
                b.add_pass(
                    "dead",
                    &[input],
                    &[garbage],
                    Box::new(move |r| {
                        r.write(garbage)[0] = [9.0; 4];
                    }),
                );
            }
            (b.compile(&[out]).unwrap(), out)
        };

        let (mut g_dead, out) = build(true);
        g_dead.execute();
        assert_eq!(g_dead.culled(), &["dead".to_string()]);
        assert!(!g_dead
            .execution_trace()
            .iter()
            .any(|n| n == "dead"));
        assert_eq!(g_dead.execution_trace(), &["copy".to_string()]);
        let out_dead = g_dead.take_output(out);

        let (mut g_clean, out2) = build(false);
        g_clean.execute();
        let out_clean = g_clean.take_output(out2);

        assert_eq!(out_dead, out_clean);
        assert_eq!(out_dead, vec![[1.0, 2.0, 3.0, 4.0]; 2]);
    }

    #[test]
    fn cycle_is_detected_and_names_both_passes() {
        let mut b = RenderGraphBuilder::new();
        let x = b.import_resource("X", desc(1, 1), vec![[0.0; 4]]);
        let y = b.create_resource("Y", desc(1, 1));
        // A reads X writes Y; B reads Y writes X -> cycle on A,B.
        b.add_pass("A", &[x], &[y], Box::new(|_| {}));
        b.add_pass("B", &[y], &[x], Box::new(|_| {}));
        let err = b.compile(&[y]).unwrap_err();
        match err {
            GraphError::Cycle(names) => {
                assert!(names.contains(&"A".to_string()), "got {names:?}");
                assert!(names.contains(&"B".to_string()), "got {names:?}");
            }
            other => panic!("expected Cycle, got {other:?}"),
        }
    }

    #[test]
    fn undeclared_access_panics_with_pass_and_resource_names() {
        // Pass 'naughty' declares reads=[res_a], writes=[res_b] but its
        // closure reads res_c, which it never declared. res_c is produced by a
        // separate live pass so the graph compiles; the panic fires at runtime.
        let mut b = RenderGraphBuilder::new();
        let a = b.import_resource("res_a", desc(1, 1), vec![[0.0; 4]]);
        let res_b = b.create_resource("res_b", desc(1, 1));
        let c = b.create_resource("res_c", desc(1, 1));
        b.add_pass(
            "producer_c",
            &[],
            &[c],
            Box::new(move |r| {
                r.write(c)[0] = [1.0; 4];
            }),
        );
        b.add_pass(
            "naughty",
            &[a],
            &[res_b, c],
            Box::new(move |r| {
                let _ = r.read(a);
                // Undeclared READ of res_c (declared in writes, not reads).
                let _ = r.read(c);
            }),
        );
        let mut graph = b.compile(&[res_b]).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            graph.execute();
        }));
        let err = result.unwrap_err();
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(msg.contains("naughty"), "panic msg missing pass: {msg}");
        assert!(msg.contains("res_c"), "panic msg missing resource: {msg}");
    }

    #[test]
    fn read_before_write_is_rejected_at_compile() {
        let mut b = RenderGraphBuilder::new();
        let phantom = b.create_resource("phantom", desc(1, 1));
        let out = b.create_resource("out", desc(1, 1));
        // Pass reads `phantom` which is neither imported nor written anywhere.
        b.add_pass(
            "reader",
            &[phantom],
            &[out],
            Box::new(|_| {}),
        );
        let err = b.compile(&[out]).unwrap_err();
        assert_eq!(
            err,
            GraphError::ReadBeforeWrite {
                pass: "reader".to_string(),
                resource: "phantom".to_string(),
            }
        );
    }

    #[test]
    fn diamond_topo_order_a_first_d_last_bc_between() {
        // A writes t1; B reads t1 writes t2; C reads t1 writes t3;
        // D reads t2,t3 writes out.
        let mut b = RenderGraphBuilder::new();
        let t1 = b.create_resource("t1", desc(1, 1));
        let t2 = b.create_resource("t2", desc(1, 1));
        let t3 = b.create_resource("t3", desc(1, 1));
        let out = b.create_resource("out", desc(1, 1));
        b.add_pass("A", &[], &[t1], Box::new(move |r| {
            r.write(t1)[0] = [1.0; 4];
        }));
        b.add_pass("B", &[t1], &[t2], Box::new(move |r| {
            let v = r.read(t1)[0];
            r.write(t2)[0] = v;
        }));
        b.add_pass("C", &[t1], &[t3], Box::new(move |r| {
            let v = r.read(t1)[0];
            r.write(t3)[0] = v;
        }));
        b.add_pass("D", &[t2, t3], &[out], Box::new(move |r| {
            let a = r.read(t2)[0];
            let _ = r.read(t3)[0];
            r.write(out)[0] = a;
        }));
        let mut g = b.compile(&[out]).unwrap();
        g.execute();
        let trace = g.execution_trace();
        let pos = |name: &str| trace.iter().position(|n| n == name).unwrap();
        let (pa, pb, pc, pd) = (pos("A"), pos("B"), pos("C"), pos("D"));
        assert_eq!(pa, 0, "A must be first: {trace:?}");
        assert_eq!(pd, trace.len() - 1, "D must be last: {trace:?}");
        assert!(pa < pb && pa < pc, "A before B,C: {trace:?}");
        assert!(pb < pd && pc < pd, "B,C before D: {trace:?}");
    }
}
