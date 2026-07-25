//! The `.vxp` texture register: residency decided from `index.json` ALONE.
//!
//! These tests exist to hold one property that a unit test on a struct cannot:
//! that a loader can list every texture a pack needs, and its exact byte cost,
//! **without opening a single `.Surface`**. Several of them prove that by
//! DESTROYING every `.Surface` payload in the pack first — if the residency
//! path touched one, it would fail loudly on the CRC, and the test would go red.

use std::path::{Path, PathBuf};

use vox_data::vxp::{
    VXP_INDEX_ENTRY, VXP_TAIL_MAX_DIM, VxpError, VxpIndex, VxpReader, VxpTexture, VxpTextureDesc,
    VxpTextureFormat, VxpTextureSlots, VxpWriter, content_id, full_mip_count, mip_dimensions,
    tail_mip_for,
};

// ─────────────────────────────────────────────────────────────────────────────
// A small corpus, shaped like the real one: several packs sharing textures.
// ─────────────────────────────────────────────────────────────────────────────

fn texture(seed: &[u8], width: u32, height: u32, format: VxpTextureFormat) -> VxpTexture {
    VxpTexture::new(
        content_id(seed),
        width,
        height,
        format,
        full_mip_count(width, height),
    )
    .expect("well-formed texture")
}

/// `brick` and `slate` are shared by every pack; `glass` only by one. That is
/// the real corpus shape — a handful of textures reached from hundreds of
/// assets across dozens of packs.
fn brick() -> VxpTexture {
    texture(b"brick_wall_001/2k/diffuse", 2048, 2048, VxpTextureFormat::Bc7UnormSrgb)
}
fn brick_normal() -> VxpTexture {
    texture(b"brick_wall_001/2k/normal", 2048, 2048, VxpTextureFormat::Bc5Unorm)
}
fn slate() -> VxpTexture {
    texture(b"roof_slates_02/1k/diffuse", 1024, 1024, VxpTextureFormat::Bc7UnormSrgb)
}
fn glass() -> VxpTexture {
    texture(b"glass_clear/roughness", 512, 512, VxpTextureFormat::Bc4Unorm)
}

fn write_pack(dir: &Path, pack_id: &str, textures: &[VxpTexture], asset_ids: &[&str]) -> PathBuf {
    let mut w = VxpWriter::new(pack_id);
    for t in textures {
        w.register_texture(t.clone()).expect("register");
    }
    for id in asset_ids {
        w.add_geometry(id, b"VXPG-not-really".to_vec());
        w.add_surface(id, format!(r#"{{"materials":["{id}"]}}"#).into_bytes());
        w.add_metadata(id, format!(r#"{{"id":"{id}"}}"#).into_bytes());
        for t in textures {
            w.reference_texture(id, &t.id);
        }
    }
    let path = dir.join(format!("{pack_id}.vxp"));
    w.finish(&path, false).expect("write pack");
    path
}

/// Rewrite a copy of the pack with every `.Surface` payload corrupted.
///
/// Reading one now fails its CRC-32, so any code path that needs a `.Surface`
/// dies loudly. Anything that still works provably never read one.
fn with_every_surface_destroyed(pack: &Path, out: &Path) -> usize {
    let mut bytes = std::fs::read(pack).expect("read pack");
    let mut at = 0usize;
    let mut wrecked = 0usize;
    while at + 30 <= bytes.len() && bytes[at..at + 4] == 0x0403_4b50u32.to_le_bytes() {
        let size = u32::from_le_bytes(bytes[at + 18..at + 22].try_into().unwrap()) as usize;
        let name_len = u16::from_le_bytes([bytes[at + 26], bytes[at + 27]]) as usize;
        let extra_len = u16::from_le_bytes([bytes[at + 28], bytes[at + 29]]) as usize;
        let name = String::from_utf8_lossy(&bytes[at + 30..at + 30 + name_len]).into_owned();
        let data_at = at + 30 + name_len + extra_len;
        if name.ends_with(".Surface") && size > 0 {
            bytes[data_at] ^= 0xFF;
            wrecked += 1;
        }
        at = data_at + size;
    }
    assert!(wrecked > 0, "the fixture must contain .Surface entries to destroy");
    std::fs::write(out, &bytes).expect("write doctored pack");
    wrecked
}

// ─────────────────────────────────────────────────────────────────────────────
// Done When #1 / #2 — the register, and residency from `index.json` alone.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn residency_is_decidable_from_index_json_alone_with_every_surface_destroyed() {
    let dir = tempfile::tempdir().unwrap();
    let all = [brick(), brick_normal(), slate(), glass()];
    let pack = write_pack(dir.path(), "city.care", &all, &["a_one", "b_two", "c_three"]);

    // Baseline: the intact pack's answer.
    let intact = VxpReader::open(&pack).expect("open intact");
    let wanted: Vec<String> = intact.index().textures.iter().map(|t| t.id.clone()).collect();
    let want_bytes = intact.index().texture_bytes();
    let want_tail = intact.index().resident_tail_bytes();
    drop(intact);

    // Now ruin every `.Surface` and ask the same questions.
    let doctored = dir.path().join("no-surfaces.vxp");
    let wrecked = with_every_surface_destroyed(&pack, &doctored);
    assert_eq!(wrecked, 3, "one .Surface per asset");

    let mut reader = VxpReader::open(&doctored).expect("index alone must still open");
    let index = reader.index();
    assert_eq!(
        index.textures.iter().map(|t| t.id.clone()).collect::<Vec<_>>(),
        wanted,
        "the full texture set must come from index.json"
    );
    assert_eq!(index.texture_bytes(), want_bytes);
    assert_eq!(index.resident_tail_bytes(), want_tail);
    // Every asset's texture set, too — still no Surface involved.
    for asset in &index.assets {
        assert_eq!(
            asset.texture_ids, wanted,
            "{}: per-asset texture ids must come from the index",
            asset.asset_id
        );
    }

    // And the control: a `.Surface` really is unreadable in this file, so the
    // assertions above cannot have been served by one.
    let surface_name = index.assets[0].surface.clone().expect("surface entry");
    let err = reader
        .read_entry(&surface_name)
        .expect_err("the doctored .Surface must be unreadable");
    assert!(
        format!("{err}").contains("crc32") || format!("{err}").contains("content hash"),
        "expected a corruption error, got: {err}"
    );
}

#[test]
fn the_register_carries_every_field_a_residency_decision_needs() {
    let dir = tempfile::tempdir().unwrap();
    let pack = write_pack(dir.path(), "city.office", &[brick(), glass()], &["only"]);
    // Read `index.json` as raw JSON, exactly as an external tool would.
    let mut reader = VxpReader::open(&pack).unwrap();
    let raw = reader.read_entry(VXP_INDEX_ENTRY).unwrap();
    let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let register = json["textures"].as_array().expect("index.json has `textures`");
    assert_eq!(register.len(), 2);
    for entry in register {
        for field in [
            "id",
            "width",
            "height",
            "format",
            "mip_count",
            "bytes",
            "tail_mip",
            "tail_bytes",
        ] {
            assert!(
                !entry[field].is_null(),
                "register entry is missing `{field}`: {entry}"
            );
        }
        assert_eq!(entry["id"].as_str().unwrap().len(), 32);
    }
    // Format is spelled the way the cook's residency manifest spells it.
    let formats: Vec<&str> = register
        .iter()
        .map(|e| e["format"].as_str().unwrap())
        .collect();
    assert!(formats.contains(&"BC7_UNORM_SRGB"), "{formats:?}");
    assert!(formats.contains(&"BC4_UNORM"), "{formats:?}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Done When #3 — the always-resident mip tail.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn every_texture_pins_a_legible_tail_and_the_tail_is_a_small_fraction_of_the_chain() {
    for t in [brick(), brick_normal(), slate(), glass()] {
        assert!(t.tail_mip < t.mip_count, "{}: tail must exist", t.id);
        assert!(t.tail_bytes > 0, "{}: tail must cost something", t.id);
        let (w, h) = mip_dimensions(t.width, t.height, t.tail_mip);
        assert!(
            w.max(h) <= VXP_TAIL_MAX_DIM,
            "{}: tail starts at {w}x{h}, above the {VXP_TAIL_MAX_DIM} threshold",
            t.id
        );
        // And the level ABOVE the tail must be too big — i.e. the tail is the
        // coarsest-first level that qualifies, not an arbitrary one.
        if t.tail_mip > 0 {
            let (pw, ph) = mip_dimensions(t.width, t.height, t.tail_mip - 1);
            assert!(pw.max(ph) > VXP_TAIL_MAX_DIM, "{}: tail starts too late", t.id);
        }
        assert_eq!(t.tail_bytes + t.streamable_bytes(), t.bytes);
        // The property that makes pinning affordable at corpus scale: the tail
        // is bounded by a CONSTANT — the cost of a threshold-sized chain — no
        // matter how large the texture is. A 2K and an 8K texture pin the same
        // ~21 KB; only the streamable part grows.
        let ceiling = t.format.level_bytes(VXP_TAIL_MAX_DIM, VXP_TAIL_MAX_DIM) * 2;
        assert!(
            t.tail_bytes <= ceiling,
            "{}: tail {} exceeds the size-independent ceiling {ceiling}",
            t.id,
            t.tail_bytes
        );
    }
    // Concretely: the 2048x2048 BC7 pins 0.4% of its chain, and the 1024x1024
    // pins the SAME number of bytes.
    assert_eq!(
        brick().tail_bytes,
        slate().tail_bytes,
        "a 2048 and a 1024 BC7 texture pin exactly the same tail"
    );
    assert!(
        brick().tail_bytes * 200 < brick().bytes,
        "the 2K chain's tail must be well under 0.5% of it"
    );
}

#[test]
fn byte_costs_are_bcn_block_math_not_an_estimate() {
    // 2048x2048 BC7: 512x512 blocks of 16 B, then the chain, with every level
    // below 4x4 still costing one whole block.
    let t = brick();
    assert_eq!(t.mip_count, 12);
    assert_eq!(t.level_bytes(0), 512 * 512 * 16);
    assert_eq!(t.level_bytes(11), 16, "a 1x1 BC7 level is still one block");
    let expected: u64 = 4_194_304
        + 1_048_576
        + 262_144
        + 65_536
        + 16_384
        + 4_096
        + 1_024
        + 256
        + 64
        + 16
        + 16
        + 16;
    assert_eq!(t.bytes, expected);
    assert_eq!(t.tail_mip, 4, "128x128 is level 4 of a 2048 chain");
    assert_eq!(t.tail_bytes, 16_384 + 4_096 + 1_024 + 256 + 64 + 16 + 16 + 16);

    // BC4 is half of BC7 at the same size.
    let bc4 = texture(b"scalar", 512, 512, VxpTextureFormat::Bc4Unorm);
    let bc7 = texture(b"colour", 512, 512, VxpTextureFormat::Bc7UnormSrgb);
    assert_eq!(bc4.bytes * 2, bc7.bytes);
}

#[test]
fn a_texture_already_smaller_than_the_threshold_is_resident_in_full() {
    let small = texture(b"decal", 64, 64, VxpTextureFormat::Bc7UnormSrgb);
    assert_eq!(small.tail_mip, 0);
    assert_eq!(small.tail_bytes, small.bytes);
    assert_eq!(small.streamable_bytes(), 0);
    assert_eq!(tail_mip_for(64, 64, full_mip_count(64, 64)), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Done When #4 — slots keyed by content hash, not load order.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn slots_are_identical_however_the_packs_are_ordered() {
    let dir = tempfile::tempdir().unwrap();
    let care = write_pack(dir.path(), "city.care", &[brick(), slate()], &["care_a"]);
    let office = write_pack(
        dir.path(),
        "city.office",
        &[brick(), brick_normal(), glass()],
        &["office_a"],
    );
    let res = write_pack(dir.path(), "city.res_low", &[slate(), glass()], &["res_a"]);

    let load = |order: [&PathBuf; 3]| -> VxpTextureSlots {
        let indices: Vec<VxpIndex> = order
            .iter()
            .map(|p| VxpReader::open(p).expect("open").index().clone())
            .collect();
        VxpTextureSlots::from_indices(indices.iter()).expect("slots")
    };

    let forward = load([&care, &office, &res]);
    let reversed = load([&res, &office, &care]);
    let shuffled = load([&office, &care, &res]);

    assert_eq!(forward.len(), 4, "four distinct textures across three packs");
    assert_eq!(forward, reversed, "pack order must not move a slot");
    assert_eq!(forward, shuffled, "pack order must not move a slot");

    // Named, not just structural: the shared `brick` sits in one slot whichever
    // pack introduced it.
    for slots in [&forward, &reversed, &shuffled] {
        assert_eq!(slots.slot(&brick().id), forward.slot(&brick().id));
        assert_eq!(slots.slot(&glass().id), forward.slot(&glass().id));
    }
    // Slots are dense 0..n.
    let mut seen: Vec<u32> = forward.iter().map(|(slot, _)| slot).collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..4).collect::<Vec<_>>());

    // The corpus-wide residency number, counted once per distinct texture even
    // though `brick`, `slate` and `glass` each appear in two packs.
    assert_eq!(
        forward.resident_tail_bytes(),
        brick().tail_bytes + brick_normal().tail_bytes + slate().tail_bytes + glass().tail_bytes
    );
}

#[test]
fn packs_that_disagree_about_one_content_hash_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let honest = write_pack(dir.path(), "honest", &[brick()], &["a"]);
    // Same content id, different declared size — one of the two is a lie.
    let mut liar = brick();
    liar.width = 1024;
    liar.height = 1024;
    let liar = VxpTexture::new(liar.id, 1024, 1024, liar.format, 11).unwrap();
    let lying = write_pack(dir.path(), "lying", &[liar], &["b"]);

    let a = VxpReader::open(&honest).unwrap().index().clone();
    let b = VxpReader::open(&lying).unwrap().index().clone();
    let err = VxpTextureSlots::from_indices([&a, &b]).expect_err("must refuse");
    assert!(
        format!("{err}").contains("disagree about the same content hash"),
        "got: {err}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Fail loudly — never a silent default.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn an_asset_referencing_an_unregistered_texture_fails_the_write_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = VxpWriter::new("t");
    w.register_texture(brick()).unwrap();
    w.add_metadata("house", b"{}".to_vec());
    w.reference_texture("house", &brick().id);
    w.reference_texture("house", &slate().id); // never registered
    let err = w
        .finish(&dir.path().join("t.vxp"), false)
        .expect_err("an unclosed register must fail the write");
    let text = format!("{err}");
    assert!(text.contains("house"), "must name the asset: {text}");
    assert!(text.contains(&slate().id), "must name the texture: {text}");
    assert!(text.contains("absent from the pack's texture register"), "{text}");
}

#[test]
fn a_register_entry_that_disagrees_with_its_payload_is_a_named_error() {
    let payload = b"pretend these are BC7 blocks".to_vec();
    let good = VxpTexture::new(
        content_id(&payload),
        256,
        256,
        VxpTextureFormat::Bc7UnormSrgb,
        9,
    )
    .unwrap();
    // Right texture, right description.
    good.verify_against_payload(
        &payload,
        VxpTextureDesc {
            width: 256,
            height: 256,
            format: VxpTextureFormat::Bc7UnormSrgb,
            mip_count: 9,
            texel_bytes: good.bytes,
        },
    )
    .expect("a matching payload must verify");

    // Wrong bytes entirely.
    let err = good
        .verify_against_payload(
            b"different bytes",
            VxpTextureDesc {
                width: 256,
                height: 256,
                format: VxpTextureFormat::Bc7UnormSrgb,
                mip_count: 9,
                texel_bytes: good.bytes,
            },
        )
        .expect_err("a different payload must not verify");
    assert!(format!("{err}").contains("payload content id is"), "{err}");

    // Right bytes, description that disagrees on every axis.
    let err = good
        .verify_against_payload(
            &payload,
            VxpTextureDesc {
                width: 512,
                height: 128,
                format: VxpTextureFormat::Bc5Unorm,
                mip_count: 7,
                texel_bytes: 1,
            },
        )
        .expect_err("a disagreeing description must not verify");
    let text = format!("{err}");
    for expected in ["dimensions 512x128", "format BC5_UNORM", "mip_count 7", "texel bytes 1"] {
        assert!(text.contains(expected), "missing {expected:?} in: {text}");
    }
    assert!(text.contains(&good.id), "must name the texture: {text}");
}

#[test]
fn impossible_register_entries_are_refused_at_construction() {
    let id = content_id(b"x");
    let cases: [(&str, VxpError); 4] = [
        (
            "id",
            VxpTexture::new("not-a-hash", 64, 64, VxpTextureFormat::Bc7Unorm, 7).unwrap_err(),
        ),
        (
            "degenerate dimensions",
            VxpTexture::new(id.clone(), 0, 64, VxpTextureFormat::Bc7Unorm, 7).unwrap_err(),
        ),
        (
            "mip_count 0",
            VxpTexture::new(id.clone(), 64, 64, VxpTextureFormat::Bc7Unorm, 0).unwrap_err(),
        ),
        (
            "exceeds",
            VxpTexture::new(id.clone(), 64, 64, VxpTextureFormat::Bc7Unorm, 99).unwrap_err(),
        ),
    ];
    for (needle, err) in cases {
        assert!(format!("{err}").contains(needle), "expected {needle:?}, got: {err}");
    }
}

#[test]
fn a_hand_edited_register_is_rejected_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = VxpIndex {
        version: vox_data::vxp::VXP_INDEX_VERSION,
        pack_id: "tampered".to_string(),
        assets: Vec::new(),
        textures: vec![brick()],
    };
    index.validate_texture_register().expect("sound to start with");
    // Halve the declared cost — the exact silent default that would under-size
    // a residency budget.
    index.textures[0].bytes /= 2;
    let err = index
        .validate_texture_register()
        .expect_err("a doctored byte count must be refused");
    let text = format!("{err}");
    assert!(text.contains(&brick().id), "{text}");
    assert!(text.contains("declares bytes="), "{text}");
    let _ = dir;
}

#[test]
fn a_pack_written_before_the_register_existed_is_refused_rather_than_read_blind() {
    // Version 1 packs carry no `textures` field at all. Reading one as if it
    // did would silently report "this pack needs no textures".
    let dir = tempfile::tempdir().unwrap();
    let pack = write_pack(dir.path(), "v2", &[brick()], &["a"]);
    let mut bytes = std::fs::read(&pack).unwrap();
    // `index.json` is the first entry; rewrite `"version":2` to `"version":1`
    // in place (same length), leaving the CRC stale so BOTH gates would fire.
    let needle = br#""version":2"#;
    let at = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("index.json carries a version");
    bytes[at + needle.len() - 1] = b'1';
    let downgraded = dir.path().join("v1.vxp");
    std::fs::write(&downgraded, &bytes).unwrap();
    let err = VxpReader::open(&downgraded).expect_err("a v1 pack must be refused");
    let text = format!("{err}");
    assert!(
        text.contains("crc32") || text.contains("index version 1"),
        "expected a loud version/corruption refusal, got: {text}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism (project LAW).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_register_does_not_change_the_bytes_when_only_the_call_order_changes() {
    let dir = tempfile::tempdir().unwrap();
    let ordered = write_pack(
        dir.path(),
        "ordered",
        &[brick(), brick_normal(), slate(), glass()],
        &["a", "b"],
    );
    // Same content, registered and referenced in reverse.
    let mut w = VxpWriter::new("ordered");
    for t in [glass(), slate(), brick_normal(), brick()] {
        w.register_texture(t).unwrap();
    }
    for id in ["b", "a"] {
        w.add_geometry(id, b"VXPG-not-really".to_vec());
        w.add_surface(id, format!(r#"{{"materials":["{id}"]}}"#).into_bytes());
        w.add_metadata(id, format!(r#"{{"id":"{id}"}}"#).into_bytes());
        for t in [glass(), slate(), brick_normal(), brick()] {
            w.reference_texture(id, &t.id);
        }
    }
    let reversed = dir.path().join("reversed.vxp");
    w.finish(&reversed, false).unwrap();
    assert_eq!(
        std::fs::read(&ordered).unwrap(),
        std::fs::read(&reversed).unwrap(),
        "register order must be a function of content, not of call order"
    );
}
