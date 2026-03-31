use crate::editor_app::AssetId;

/// An asset with a pre-computed LVM embedding for semantic search.
pub struct AssetEntry {
    id: AssetId,
    name: String,
    embedding: Vec<f32>,
}

impl AssetEntry {
    pub fn new(id: AssetId, name: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self { id, name: name.into(), embedding }
    }

    pub fn id(&self) -> AssetId { self.id }
    pub fn name(&self) -> &str { &self.name }
    pub fn embedding(&self) -> &[f32] { &self.embedding }
}

/// The semantic asset library.
pub struct AssetLibrary {
    entries: Vec<AssetEntry>,
    threshold: f32,
}

impl AssetLibrary {
    pub fn new() -> Self {
        Self { entries: Vec::new(), threshold: 0.6 }
    }

    pub fn with_entries(entries: Vec<AssetEntry>) -> Self {
        Self { entries, threshold: 0.6 }
    }

    pub fn add(&mut self, entry: AssetEntry) {
        self.entries.push(entry);
    }

    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "embedding length mismatch");
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-8 || norm_b < 1e-8 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    pub fn search_with_embedding(&self, query: &[f32]) -> Vec<(&AssetEntry, f32)> {
        let mut results: Vec<(&AssetEntry, f32)> = self.entries.iter()
            .map(|e| {
                let sim = Self::cosine_similarity(query, e.embedding());
                (e, sim)
            })
            .filter(|(_, sim)| *sim >= self.threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn entries(&self) -> &[AssetEntry] { &self.entries }
}

/// Transient state for the asset library UI tab.
pub struct AssetLibraryUi {
    query: String,
    cached_results: Vec<(usize, f32)>,
}

impl AssetLibraryUi {
    pub fn new() -> Self {
        Self { query: String::new(), cached_results: Vec::new() }
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        library: &AssetLibrary,
        get_embedding: &dyn Fn(&str) -> Vec<f32>,
        on_select: &mut dyn FnMut(AssetId),
    ) {
        let changed = ui.add(
            egui::TextEdit::singleline(&mut self.query)
                .hint_text("Search assets by feeling…")
                .desired_width(ui.available_width())
        ).changed();

        if changed && !self.query.is_empty() {
            let embedding = get_embedding(&self.query);
            let results = library.search_with_embedding(&embedding);
            self.cached_results = results.iter()
                .map(|(e, sim)| {
                    let idx = library.entries().iter().position(|x| x.id() == e.id()).unwrap_or(0);
                    (idx, *sim)
                })
                .collect();
        }

        ui.separator();

        if self.query.is_empty() {
            egui::ScrollArea::vertical().show(ui, |ui| {
                for entry in library.entries() {
                    if ui.selectable_label(false, entry.name()).clicked() {
                        on_select(entry.id());
                    }
                }
            });
        } else if self.cached_results.is_empty() {
            ui.label(egui::RichText::new("No matching assets").color(egui::Color32::from_gray(160)));
            if ui.button("✦ Generate with AI").clicked() {
                // Phase 1 placeholder
            }
        } else {
            egui::ScrollArea::vertical().show(ui, |ui| {
                for (idx, sim) in &self.cached_results {
                    if let Some(entry) = library.entries().get(*idx) {
                        let label = format!("{} ({:.0}%)", entry.name(), sim * 100.0);
                        if ui.selectable_label(false, label).clicked() {
                            on_select(entry.id());
                        }
                    }
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rock_library() -> AssetLibrary {
        AssetLibrary::with_entries(vec![
            AssetEntry::new(AssetId(1), "Rock A", vec![0.99f32, 0.14, 0.0]),
            AssetEntry::new(AssetId(2), "Rock B", vec![0.0f32, 1.0, 0.0]),
            AssetEntry::new(AssetId(3), "Rock C", vec![0.98f32, 0.20, 0.0]),
        ])
    }

    fn query_embedding() -> Vec<f32> {
        vec![1.0f32, 0.0, 0.0]
    }

    #[test]
    fn search_returns_only_results_above_threshold() {
        let library = rock_library();
        let results = library.search_with_embedding(&query_embedding());
        assert!(results.iter().all(|(e, _)| e.id() != AssetId(2)));
    }

    #[test]
    fn search_results_are_sorted_descending_by_score() {
        let library = rock_library();
        let results = library.search_with_embedding(&query_embedding());
        for i in 0..results.len().saturating_sub(1) {
            assert!(results[i].1 >= results[i + 1].1);
        }
    }

    #[test]
    fn search_returns_two_high_similarity_rocks() {
        let library = rock_library();
        let results = library.search_with_embedding(&query_embedding());
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_empty_library_returns_empty() {
        let library = AssetLibrary::with_entries(vec![]);
        let results = library.search_with_embedding(&[1.0, 0.0, 0.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn cosine_similarity_identical_vectors_is_one() {
        let v = vec![0.6f32, 0.8, 0.0];
        let sim = AssetLibrary::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "sim={}", sim);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors_is_zero() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let sim = AssetLibrary::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "sim={}", sim);
    }
}
