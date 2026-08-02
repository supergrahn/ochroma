//! Runtime-kernel contract: per-instance material variation must be WIRED —
//! keyed, bound, counted, and applied where it survives.
//!
//! WHY THIS EXISTS (witnessed defect, 2026-07-25)
//! ---------------------------------------------
//! Every copy of a prototype rendered BYTE-IDENTICAL. A street of the same
//! walk-up, a park of the same tree — clone-stamped. The feature that was meant
//! to prevent that had existed in `megakernel.slang` since it was written and
//! had NEVER ONCE EXECUTED, in four independent ways:
//!
//!   1. `u_num_instances` was declared and read in the kernel and had NO Rust
//!      binder anywhere in either repo. Unbound uniforms are zero, so the guard
//!      `instance_id < u_num_instances` was `x < 0` — false, unconditionally.
//!   2. `g_var_color_r/g/b`, `g_var_roughness`, `g_var_uv_offset_x/y`,
//!      `g_var_uv_rotation` had no Rust binder either, so they resolved to the
//!      backend's dummy descriptor.
//!   3. The key it read, `g_hit_instance`, is -1 across the ENTIRE shipping
//!      hardware-RT path: `rt_query.slang::run_closest` never wrote it. And it
//!      could not simply be made to carry the real index, because
//!      `is_instance_hit` is derived from it — a real value there diverts every
//!      hardware-RT hit into the positions-only software-BLAS branch (no UVs, no
//!      per-triangle materials). The fix therefore needed a SEPARATE
//!      `g_hit_tlas_instance` buffer.
//!   4. Subtlest and most dangerous: even with 1-3 fixed, the colour and
//!      roughness jitter would STILL have been a silent no-op, because the block
//!      sat BEFORE the texture fetch and
//!
//!          apply_base_color_texture(f, texel, flags) = (flags & 2) ? f*texel : texel
//!          apply_roughness_texture(f, texel, flags, svt) = (svt || flags & 4) ? f*texel : texel
//!
//!      REPLACE rather than modulate for any material that does not set the
//!      flag — i.e. for essentially every textured facade and plant in the game.
//!      A build with 1-3 fixed measures as "wired" and looks plausible while
//!      changing nothing at all.
//!
//! Every assertion below is aimed at one of those four, plus the invariant that
//! the fix must NOT disturb: `is_instance_hit` must keep deriving from
//! `g_hit_instance` alone.
//!
//! This is a SOURCE-LEVEL tripwire on that defect class: it grades the kernel
//! tree and host binder this build would actually dispatch, on every platform,
//! with no GPU required. The behavioural counterpart (a real GPU render proving
//! 36 copies of one prototype go from 1 distinct colour to 36) is
//! `spectra/rust/spectra-renderer/tests/instance_variation_gpu.rs`.
//!
//! PRE-FIX FAILURE PROOF: point `OCHROMA_SPECTRA_ROOT` at a checkout from before
//! the fix and every kernel-side assertion here fires.

use std::path::PathBuf;

/// The Spectra checkout THIS BUILD WOULD ACTUALLY USE.
///
/// Resolution mirrors the runtime's own order: the explicit
/// `OCHROMA_SPECTRA_ROOT` override first (which is also how the pre-fix failure
/// is demonstrated), then the `spectra-renderer` path dependency declared in
/// `Cargo.toml` (`../../../spectra/rust/spectra-renderer`). Grading the tree the
/// renderer loads is the whole point — a contract checked against a tree we
/// never dispatch would prove nothing. If it is missing, the crate could not
/// have been built against it, so a hard failure is correct — never skip.
fn spectra_root() -> PathBuf {
    if let Ok(dir) = std::env::var("OCHROMA_SPECTRA_ROOT") {
        let candidate = PathBuf::from(dir);
        if candidate.is_dir() {
            return candidate;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spectra")
}

fn read(rel: &str) -> String {
    let path = spectra_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {} — {e}", path.display()))
}

/// The seven per-instance variation primvar buffers the kernel declares.
const VAR_BUFFERS: [&str; 7] = [
    "g_var_color_r",
    "g_var_color_g",
    "g_var_color_b",
    "g_var_roughness",
    "g_var_uv_offset_x",
    "g_var_uv_offset_y",
    "g_var_uv_rotation",
];

/// Strip `//` line comments so a contract is never satisfied by prose. Without
/// this, the pre-fix tree would PASS several assertions below purely on the
/// strength of a code comment that named `g_hit_tlas_instance` as future work.
fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The body of `fn <name>` / `void <name>` … balanced from its opening brace.
fn function_body(src: &str, signature_fragment: &str) -> String {
    let start = src
        .find(signature_fragment)
        .unwrap_or_else(|| panic!("`{signature_fragment}` not found in source"));
    let open = start
        + src[start..]
            .find('{')
            .unwrap_or_else(|| panic!("no body for `{signature_fragment}`"));
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    let mut cursor = open;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return src[open..=cursor].to_string();
                }
            }
            _ => {}
        }
        cursor += 1;
    }
    panic!("unbalanced body for `{signature_fragment}`");
}

// ---------------------------------------------------------------------------
// Defect 3 — a REAL per-instance key must reach the shader, on a NEW buffer.
// ---------------------------------------------------------------------------

#[test]
fn hardware_closest_hit_writes_a_real_per_instance_key() {
    let rt_query = strip_line_comments(&read("slang/rt_query.slang"));

    assert!(
        rt_query.contains("g_hit_tlas_instance"),
        "rt_query.slang must declare g_hit_tlas_instance — without a per-instance \
         key reaching the shader, every copy of a prototype renders identically"
    );

    let closest = function_body(&rt_query, "void run_closest<T : IRayTracer>");
    assert!(
        closest.contains("g_hit_tlas_instance[ray_idx] ="),
        "run_closest must WRITE g_hit_tlas_instance. This is the defect that made \
         per-instance variation dead on the shipping hardware-RT path: the kernel \
         keyed off g_hit_instance, which run_closest never wrote and which \
         RayQueueBuffers fills with -1 once at allocation."
    );
    assert!(
        closest.contains("hit.tlas_instance"),
        "the written value must come from TraceHit::tlas_instance, not be \
         synthesised locally"
    );

    let tracer = strip_line_comments(&read("slang/rt_tracer.slang"));
    assert!(
        tracer.contains("CommittedInstanceIndex()"),
        "TraceHit::tlas_instance must be sourced from the hardware's committed \
         TLAS instance INDEX"
    );
    assert!(
        tracer.contains("h.tlas_instance = -1;"),
        "trace_miss() must sentinel tlas_instance to -1, or a miss inherits the \
         previous hit's identity"
    );
}

/// The reason this could not be a one-liner, pinned so nobody "simplifies" it
/// back into the bug.
#[test]
fn is_instance_hit_still_derives_from_g_hit_instance_alone() {
    let mega = strip_line_comments(&read("slang/megakernel.slang"));

    let derivation = mega
        .lines()
        .find(|l| l.contains("bool is_instance_hit"))
        .expect("megakernel must derive is_instance_hit");
    assert!(
        !derivation.contains("g_hit_tlas_instance") && !derivation.contains("tlas_instance"),
        "is_instance_hit MUST NOT be derived from the TLAS instance index. It \
         selects the positions-only software-BLAS branch (no UVs, no \
         per-triangle materials); feeding it a real hardware index diverts every \
         hardware-RT hit into that branch. That is exactly why the variation key \
         is a separate buffer. Offending line: {derivation}"
    );

    // And the variation key must be read-only in the shade kernel: a
    // StructuredBuffer cannot be written, so this pins that nothing starts
    // branching scene classification off it.
    assert!(
        mega.contains("StructuredBuffer<int>   g_hit_tlas_instance;")
            || mega.contains("StructuredBuffer<int> g_hit_tlas_instance;"),
        "megakernel must take g_hit_tlas_instance as a READ-ONLY StructuredBuffer"
    );
}

// ---------------------------------------------------------------------------
// Defects 1 + 2 — everything the kernel reads must actually be bound.
// ---------------------------------------------------------------------------

#[test]
fn every_variation_buffer_the_kernel_reads_has_a_host_binder() {
    let mega = strip_line_comments(&read("slang/megakernel.slang"));
    let gpu_scene = read("rust/spectra-scene-upload/src/gpu_scene.rs");
    let ray_queue = read("rust/spectra-integrator/src/ray_queue.rs");

    // ANTI-VACUITY: derive the buffer list from what the KERNEL actually reads,
    // not from the constant above. If someone adds an eighth `g_var_*` buffer,
    // this test starts requiring a binder for it too, automatically.
    let read_by_kernel: Vec<&str> = VAR_BUFFERS
        .iter()
        .copied()
        .filter(|b| mega.contains(&format!("{b}[")))
        .collect();
    assert_eq!(
        read_by_kernel.len(),
        VAR_BUFFERS.len(),
        "expected the megakernel to index all {} variation buffers; it indexes {:?}. \
         If the set changed, update VAR_BUFFERS *and* the host binder together — \
         that coupling is the entire point of this test.",
        VAR_BUFFERS.len(),
        read_by_kernel
    );

    for buffer in read_by_kernel {
        assert!(
            gpu_scene.contains(&format!("\"{buffer}\"")),
            "kernel indexes `{buffer}` but no host binder names it. An unbound \
             buffer resolves to the backend's dummy descriptor and the feature \
             silently does nothing — the exact failure mode this tripwire exists \
             for."
        );
    }

    assert!(
        gpu_scene.contains("\"u_num_instances\""),
        "`u_num_instances` must be SET by the host. It is the bound on the \
         kernel's variation lookup; unbound it is 0 and the whole block is dead \
         code that reads as wired."
    );
    assert!(
        ray_queue.contains("\"g_hit_tlas_instance\""),
        "RayQueueBuffers must bind g_hit_tlas_instance, or run_closest writes to \
         an unbound descriptor"
    );
}

#[test]
fn variation_count_and_buffers_cannot_disagree_silently() {
    let gpu_scene = read("rust/spectra-scene-upload/src/gpu_scene.rs");

    assert!(
        gpu_scene.contains("panic!") && gpu_scene.contains("instance_variation_count"),
        "GpuScene must FAIL LOUDLY when the variation count and the variation \
         buffers disagree. A silent no-op here is indistinguishable from a \
         working render, which is how this defect survived so long."
    );
    assert!(
        gpu_scene.contains("struct InstanceVariationBuffers"),
        "the variation channels must travel as ONE struct so a PARTIAL binding \
         (some channels bound, some resolving to a dummy descriptor) is not \
         representable"
    );
}

// ---------------------------------------------------------------------------
// Defect 4 — the tint must be applied where the texture fetch cannot erase it.
// ---------------------------------------------------------------------------

#[test]
fn colour_and_roughness_variation_is_applied_after_the_texture_fetch() {
    // Scoped to the SHADE KERNEL BODY. Ordering only means anything inside the
    // one function that actually shades a hit; the file also contains helper
    // definitions and self-test functions that call the same helpers, and
    // grading against those would be measuring nothing.
    let mega = strip_line_comments(&read("slang/megakernel.slang"));
    let mega = function_body(&mega, "void shade_and_bounce(uint3 tid");

    let apply_at = mega
        .find("apply_variation_color_roughness(")
        .expect("shade_and_bounce must apply the colour/roughness half of the variation");

    // The LAST place a texture can overwrite mat.albedo / mat.roughness.
    let last_albedo_texture = mega
        .rfind("apply_base_color_texture(")
        .expect("shade_and_bounce must sample a base-colour texture");
    let last_roughness_texture = mega
        .rfind("apply_metallic_roughness_texture(")
        .expect("shade_and_bounce must apply a packed metallic/roughness texture");

    assert!(
        apply_at > last_albedo_texture && apply_at > last_roughness_texture,
        "per-instance colour/roughness variation is applied at byte {apply_at}, \
         BEFORE the last texture fetch (albedo {last_albedo_texture}, roughness \
         {last_roughness_texture}).\n\n\
         apply_base_color_texture / apply_metallic_roughness_texture can REPLACE \
         channels for materials without the corresponding flag bits. Applied before them, the \
         jitter is silently overwritten and per-instance variation is a NO-OP \
         that still measures as 'wired'. It must be applied after."
    );

    // The UV half is the mirror image: it MUST land before the UV transform that
    // consumes it, so the two halves cannot be collapsed back together.
    let uv_apply = mega
        .find("apply_variation_uv(")
        .expect("megakernel must apply the UV half of the variation");
    let uv_transform = mega
        .find("apply_uv_transform(hit_uv")
        .expect("megakernel must build sampling UVs via apply_uv_transform");
    assert!(
        uv_apply < uv_transform,
        "UV variation is applied at byte {uv_apply}, AFTER apply_uv_transform at \
         {uv_transform} — it would have no effect on the sampled coordinates"
    );
}

#[test]
fn variation_is_keyed_on_the_tlas_instance_and_bounded_by_the_bound_count() {
    let mega = strip_line_comments(&read("slang/megakernel.slang"));

    let key_line = mega
        .lines()
        .find(|l| l.contains("var_instance_id") && l.contains("g_hit_tlas_instance"))
        .expect(
            "the variation key must be read from g_hit_tlas_instance — keying off \
             g_hit_instance is the original defect",
        );
    assert!(
        key_line.contains('='),
        "expected an assignment binding the variation key, got: {key_line}"
    );

    let guard = mega
        .lines()
        .find(|l| l.contains("var_instance_id") && l.contains("u_num_instances"))
        .expect(
            "the variation lookup must be bounded by u_num_instances — an \
             unbounded index into the primvar buffers reads out of bounds for any \
             instance beyond the uploaded count",
        );
    assert!(
        guard.contains("var_instance_id >= 0"),
        "the guard must also reject the -1 miss/non-instanced sentinel, got: {guard}"
    );
}

// ---------------------------------------------------------------------------
// Config-first + determinism, as source contracts.
// ---------------------------------------------------------------------------

#[test]
fn variation_ranges_are_config_first_not_hardcoded() {
    let mega = strip_line_comments(&read("slang/megakernel.slang"));
    let sample = read("rust/spectra-renderer/src/renderer/sample.rs");
    let ron = read("config/spectra.ron");

    for uniform in [
        "u_var_color_amount",
        "u_var_roughness_amount",
        "u_var_uv_offset_amount",
        "u_var_uv_rotation_amount",
    ] {
        assert!(
            mega.contains(uniform),
            "megakernel must scale the normalized primvars by `{uniform}` rather \
             than baking magnitudes into the buffers"
        );
        assert!(
            sample.contains(&format!("\"{uniform}\"")),
            "`{uniform}` must be bound from config in sample.rs, or the amount is \
             zero and variation is off no matter what the config says"
        );
    }

    assert!(
        ron.contains("variation: ("),
        "spectra.ron must carry a `variation:` section — tuning the ranges must \
         never require a rebuild"
    );
}

#[test]
fn variation_primvars_are_derived_from_a_stable_id_not_load_order() {
    let uploader = read("rust/spectra-scene-upload/src/uploader.rs");
    let body = function_body(&uploader, "fn instance_variation_primvars");

    assert!(
        body.contains("inst.transform[0][3]")
            && body.contains("inst.transform[1][3]")
            && body.contains("inst.transform[2][3]"),
        "per-instance variation must be derived from the instance's WORLD \
         POSITION. Determinism is a LAW here: replay-exactness is a product moat, \
         so an instance's look may not depend on where it happens to sit in an \
         array."
    );
    assert!(
        body.contains("POSITION_QUANTUM_M"),
        "the position must be QUANTIZED before hashing, or f32 round-trip noise \
         in a transform can flip an instance's appearance between runs"
    );

    // The INSTANCE loop must not yield an index at all — `for inst in instances`
    // is the structural guarantee that array position cannot reach the hash.
    // (An `.enumerate()` over the seven CHANNELS is fine and expected: that
    // index selects which primvar is being produced, not which instance.)
    assert!(
        !body.contains("instances.iter().enumerate()")
            && !body.contains("instances.iter().copied().enumerate()"),
        "the instance's ARRAY POSITION must not participate in the hash — that is \
         load order, and it would make an instance's look depend on upload \
         sequencing rather than on where it stands in the world"
    );
    assert!(
        !body.contains("rand") && !body.contains("Instant") && !body.contains("SystemTime"),
        "variation must be a pure function of the scene: no RNG, no clock"
    );
}
