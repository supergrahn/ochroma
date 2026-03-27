pub fn begin_frame() {
    puffin::GlobalProfiler::lock().new_frame();
}

pub use puffin::profile_scope;
pub use puffin::profile_function;
