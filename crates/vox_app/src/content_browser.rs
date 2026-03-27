use std::path::{Path, PathBuf};

/// Type classification for content entries based on file extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentType {
    GaussianSplat, // .ply
    OchromaAsset,  // .vxm
    AudioClip,     // .wav, .ogg
    Script,        // .rhai
    Mesh,          // .glb, .gltf
    Map,           // .ochroma_map
    Unknown,
}

impl ContentType {
    /// Short label for display in the browser UI.
    pub fn label(&self) -> &'static str {
        match self {
            ContentType::GaussianSplat => "[PLY]",
            ContentType::OchromaAsset => "[VXM]",
            ContentType::AudioClip => "[WAV]",
            ContentType::Script => "[SCRIPT]",
            ContentType::Mesh => "[MESH]",
            ContentType::Map => "[MAP]",
            ContentType::Unknown => "[???]",
        }
    }
}

/// Classify a file path by its extension.
pub fn classify(path: &Path) -> ContentType {
    match path.extension().and_then(|e| e.to_str()) {
        Some("ply") => ContentType::GaussianSplat,
        Some("vxm") => ContentType::OchromaAsset,
        Some("wav") | Some("ogg") => ContentType::AudioClip,
        Some("rhai") => ContentType::Script,
        Some("glb") | Some("gltf") => ContentType::Mesh,
        Some("ochroma_map") => ContentType::Map,
        _ => ContentType::Unknown,
    }
}

/// A single entry in the content browser.
#[derive(Debug, Clone)]
pub struct ContentEntry {
    pub name: String,
    pub path: PathBuf,
    pub entry_type: ContentType,
    pub size_bytes: u64,
}

/// Actions triggered by user interaction in the content browser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentAction {
    LoadAsset(PathBuf),
    OpenMap(PathBuf),
    PlayAudio(PathBuf),
}

/// Editor panel that shows files in a directory, with type labels and search.
pub struct ContentBrowser {
    pub root_path: PathBuf,
    pub entries: Vec<ContentEntry>,
    pub selected: Option<usize>,
    pub search_query: String,
    pub current_dir: PathBuf,
}

impl ContentBrowser {
    /// Create a new content browser rooted at the given path.
    pub fn new(root: &Path) -> Self {
        let root_path = root.to_path_buf();
        let current_dir = root_path.clone();
        Self {
            root_path,
            entries: Vec::new(),
            selected: None,
            search_query: String::new(),
            current_dir,
        }
    }

    /// Scan the current directory and populate entries.
    pub fn scan(&mut self) {
        self.entries.clear();
        self.selected = None;

        let read_dir = match std::fs::read_dir(&self.current_dir) {
            Ok(rd) => rd,
            Err(_) => return,
        };

        for entry in read_dir.flatten() {
            let path = entry.path();
            // Skip directories — we only list files.
            if path.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let entry_type = classify(&path);

            self.entries.push(ContentEntry {
                name,
                path,
                entry_type,
                size_bytes,
            });
        }

        // Sort by name for stable ordering.
        self.entries.sort_by(|a, b| a.name.cmp(&b.name));
    }

    /// Return entries filtered by the current search query (case-insensitive substring match).
    pub fn filtered_entries(&self) -> Vec<&ContentEntry> {
        if self.search_query.is_empty() {
            return self.entries.iter().collect();
        }
        let query = self.search_query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| e.name.to_lowercase().contains(&query))
            .collect()
    }

    /// Navigate to a subdirectory and rescan.
    pub fn navigate_to(&mut self, dir: &Path) {
        self.current_dir = dir.to_path_buf();
        self.scan();
    }

    /// Navigate to the parent directory (does not go above root).
    pub fn parent_dir(&mut self) {
        if self.current_dir == self.root_path {
            return;
        }
        if let Some(parent) = self.current_dir.parent() {
            // Clamp to root — never navigate above it.
            if parent.starts_with(&self.root_path) || parent == self.root_path {
                self.current_dir = parent.to_path_buf();
            } else {
                self.current_dir = self.root_path.clone();
            }
        }
        self.scan();
    }

    /// Total number of entries (unfiltered).
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Render the content browser as an egui side panel.
    /// Returns a `ContentAction` when the user double-clicks an entry.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<ContentAction> {
        let mut action = None;

        egui::SidePanel::left("content_browser")
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Content Browser");
                ui.separator();

                // --- Path breadcrumb ---
                ui.horizontal(|ui| {
                    if ui.small_button("^").clicked() {
                        self.parent_dir();
                    }
                    let display_path = self
                        .current_dir
                        .strip_prefix(&self.root_path)
                        .unwrap_or(&self.current_dir);
                    ui.label(format!("/{}", display_path.display()));
                });

                ui.separator();

                // --- Search bar ---
                ui.horizontal(|ui| {
                    ui.label("Search:");
                    ui.text_edit_singleline(&mut self.search_query);
                });

                ui.separator();

                // --- File list ---
                // Collect filtered indices to avoid borrow conflict with self.selected.
                let query_lower = self.search_query.to_lowercase();
                let filtered_indices: Vec<usize> = self
                    .entries
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| {
                        query_lower.is_empty()
                            || e.name.to_lowercase().contains(&query_lower)
                    })
                    .map(|(i, _)| i)
                    .collect();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (display_idx, &entry_idx) in filtered_indices.iter().enumerate() {
                        let entry = &self.entries[entry_idx];
                        let is_selected = self.selected == Some(display_idx);
                        let label = format!(
                            "{} {}  ({})",
                            entry.entry_type.label(),
                            entry.name,
                            format_size(entry.size_bytes),
                        );

                        let response = ui.selectable_label(is_selected, &label);

                        if response.clicked() {
                            self.selected = Some(display_idx);
                        }

                        if response.double_clicked() {
                            action = Some(action_for_entry(entry));
                        }
                    }
                });
            });

        action
    }
}

/// Determine the content action for an entry based on its type.
fn action_for_entry(entry: &ContentEntry) -> ContentAction {
    match entry.entry_type {
        ContentType::GaussianSplat
        | ContentType::OchromaAsset
        | ContentType::Mesh
        | ContentType::Script
        | ContentType::Unknown => ContentAction::LoadAsset(entry.path.clone()),
        ContentType::Map => ContentAction::OpenMap(entry.path.clone()),
        ContentType::AudioClip => ContentAction::PlayAudio(entry.path.clone()),
    }
}

/// Format byte sizes for display.
fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn classify_ply() {
        assert_eq!(classify(Path::new("scene.ply")), ContentType::GaussianSplat);
    }

    #[test]
    fn classify_wav() {
        assert_eq!(classify(Path::new("sound.wav")), ContentType::AudioClip);
    }

    #[test]
    fn classify_ogg() {
        assert_eq!(classify(Path::new("music.ogg")), ContentType::AudioClip);
    }

    #[test]
    fn classify_rhai() {
        assert_eq!(classify(Path::new("player.rhai")), ContentType::Script);
    }

    #[test]
    fn classify_glb() {
        assert_eq!(classify(Path::new("model.glb")), ContentType::Mesh);
    }

    #[test]
    fn classify_gltf() {
        assert_eq!(classify(Path::new("model.gltf")), ContentType::Mesh);
    }

    #[test]
    fn classify_ochroma_map() {
        assert_eq!(classify(Path::new("level.ochroma_map")), ContentType::Map);
    }

    #[test]
    fn classify_vxm() {
        assert_eq!(classify(Path::new("asset.vxm")), ContentType::OchromaAsset);
    }

    #[test]
    fn classify_unknown_extension() {
        assert_eq!(classify(Path::new("readme.txt")), ContentType::Unknown);
    }

    #[test]
    fn scan_finds_files_in_temp_dir() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("scene.ply"), b"data").unwrap();
        fs::write(tmp.path().join("clip.wav"), b"audio").unwrap();
        fs::write(tmp.path().join("script.rhai"), b"fn main() {}").unwrap();

        let mut browser = ContentBrowser::new(tmp.path());
        browser.scan();

        assert_eq!(browser.entry_count(), 3);
        let names: Vec<&str> = browser.entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"scene.ply"));
        assert!(names.contains(&"clip.wav"));
        assert!(names.contains(&"script.rhai"));
    }

    #[test]
    fn search_filters_by_name_substring() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("hero.ply"), b"data").unwrap();
        fs::write(tmp.path().join("villain.ply"), b"data").unwrap();
        fs::write(tmp.path().join("background.wav"), b"audio").unwrap();

        let mut browser = ContentBrowser::new(tmp.path());
        browser.scan();

        browser.search_query = "hero".to_string();
        let filtered = browser.filtered_entries();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "hero.ply");
    }

    #[test]
    fn navigate_to_changes_current_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("subdir");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("model.glb"), b"mesh").unwrap();

        let mut browser = ContentBrowser::new(tmp.path());
        browser.navigate_to(&sub);

        assert_eq!(browser.current_dir, sub);
        assert_eq!(browser.entry_count(), 1);
        assert_eq!(browser.entries[0].name, "model.glb");
    }

    #[test]
    fn parent_dir_goes_up_one_level() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("subdir");
        fs::create_dir(&sub).unwrap();

        let mut browser = ContentBrowser::new(tmp.path());
        browser.navigate_to(&sub);
        assert_eq!(browser.current_dir, sub);

        browser.parent_dir();
        assert_eq!(browser.current_dir, tmp.path());
    }

    #[test]
    fn parent_dir_does_not_go_above_root() {
        let tmp = tempfile::tempdir().unwrap();
        let mut browser = ContentBrowser::new(tmp.path());
        browser.parent_dir();
        assert_eq!(browser.current_dir, tmp.path().to_path_buf());
    }

    #[test]
    fn empty_directory_returns_empty_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let mut browser = ContentBrowser::new(tmp.path());
        browser.scan();
        assert_eq!(browser.entry_count(), 0);
        assert!(browser.filtered_entries().is_empty());
    }
}
