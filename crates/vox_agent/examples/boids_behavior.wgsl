// Boids rules using spatial hash neighbour queries.
// Included after bind_group_layout_source() which provides all bindings.

@compute @workgroup_size(64)
fn agent_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= uniforms.agent_count { return; }
    if (agent_flags[i] & 1u) == 0u { return; }

    let px = positions_in[i*3u];
    let py = positions_in[i*3u+1u];
    let pz = positions_in[i*3u+2u];
    let vx = velocities_in[i*3u];
    let vy = velocities_in[i*3u+1u];
    let vz = velocities_in[i*3u+2u];

    var sep_x = 0.0; var sep_z = 0.0;
    var aln_x = 0.0; var aln_z = 0.0;
    var coh_x = 0.0; var coh_z = 0.0;
    var count = 0u;

    let cell = spatial_cells[i];
    let gw = uniforms.grid_width;
    let cx_i = cell % gw;
    let cz_i = cell / gw;

    // Check 3x3 neighbourhood of cells
    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let nx = i32(cx_i) + dx;
            let nz = i32(cz_i) + dz;
            if nx < 0 || nz < 0 || u32(nx) >= gw || u32(nz) >= gw { continue; }
            let ncell = u32(nz) * gw + u32(nx);
            let start = cell_offsets[ncell];
            let end   = cell_offsets[ncell + 1u];
            for (var k = start; k < end; k = k + 1u) {
                let j = cell_data[k];
                if j == i { continue; }
                let jx = positions_in[j*3u];
                let jy = positions_in[j*3u+1u];
                let jz = positions_in[j*3u+2u];
                let dx2 = jx - px; let dz2 = jz - pz;
                let dist2 = dx2*dx2 + dz2*dz2;
                if dist2 > 400.0 { continue; } // 20m radius
                // Separation: push away from very close neighbours
                if dist2 < 4.0 { sep_x = sep_x - dx2; sep_z = sep_z - dz2; }
                // Alignment: match velocity
                aln_x = aln_x + velocities_in[j*3u];
                aln_z = aln_z + velocities_in[j*3u+2u];
                // Cohesion: steer toward centre of mass
                coh_x = coh_x + jx; coh_z = coh_z + jz;
                count = count + 1u;
            }
        }
    }

    var new_vx = vx; var new_vz = vz;
    if count > 0u {
        let inv = 1.0 / f32(count);
        new_vx = new_vx + sep_x * 0.05 + (aln_x * inv - vx) * 0.1 + (coh_x * inv - px) * 0.01;
        new_vz = new_vz + sep_z * 0.05 + (aln_z * inv - vz) * 0.1 + (coh_z * inv - pz) * 0.01;
    }

    // Clamp speed to [0.05, 2.0]
    let speed = sqrt(new_vx * new_vx + new_vz * new_vz);
    if speed > 0.001 {
        let clamped = clamp(speed, 0.05, 2.0);
        new_vx = new_vx / speed * clamped;
        new_vz = new_vz / speed * clamped;
    }

    // Integrate position with world-space wrapping
    var new_px = px + new_vx * uniforms.dt;
    var new_pz = pz + new_vz * uniforms.dt;
    if new_px >  500.0 { new_px = new_px - 1000.0; }
    if new_px < -500.0 { new_px = new_px + 1000.0; }
    if new_pz >  500.0 { new_pz = new_pz - 1000.0; }
    if new_pz < -500.0 { new_pz = new_pz + 1000.0; }

    positions_out[i*3u]    = new_px;
    positions_out[i*3u+1u] = py;
    positions_out[i*3u+2u] = new_pz;
    velocities_out[i*3u]   = new_vx;
    velocities_out[i*3u+1u] = vy;
    velocities_out[i*3u+2u] = new_vz;
}
