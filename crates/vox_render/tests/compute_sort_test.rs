use vox_render::gpu::compute_sort::SortEntry;

#[test]
fn sort_entry_size() {
    assert_eq!(std::mem::size_of::<SortEntry>(), 8);
}

// GPU sort can only be tested with a real GPU; test the CPU fallback path
#[test]
fn cpu_sort_entries_by_depth() {
    let mut entries = vec![
        SortEntry { depth: 3.0, index: 0 },
        SortEntry { depth: 1.0, index: 1 },
        SortEntry { depth: 2.0, index: 2 },
    ];
    entries.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
    assert_eq!(entries[0].index, 1);
    assert_eq!(entries[1].index, 2);
    assert_eq!(entries[2].index, 0);
}
