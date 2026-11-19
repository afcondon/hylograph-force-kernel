//! Force simulation kernel for WebAssembly: d3-force 3.x's forces, in f32.
//!
//! The forces are ports of d3-force (see `forces.rs`) over a port of
//! d3-quadtree (see `quadtree.rs`). What a tick does, in order:
//!
//! 1. apply the six built-in force slots that are enabled, in the fixed
//!    order centre, many-body, link, x, y, collide (the original API:
//!    `configure_*` + `enable_force` / `enable_forces_all`);
//! 2. apply the force list in the order the forces were added (`add_*`,
//!    any number of instances of any type, each with optional per-node or
//!    per-link parameter arrays and an optional node mask);
//! 3. integrate: `v *= velocity_decay; p += v`, or snap to a fixed coordinate;
//! 4. apply hard containment, if any force in the list has it, then snap
//!    fixed coordinates again;
//! 5. cool: `alpha += (alpha_target - alpha) * alpha_decay`.
//!
//! Seven of the force types are d3-force's. Six things are not in d3:
//! towards, rectangle collision, containment, group cohesion, scoped
//! many-body and the directional link constraint. Each of those is defined
//! by the doc comment on its `apply_*` function in `forces.rs`, precisely
//! enough to reimplement from, and checked against a plain reference
//! implementation in `reference.rs`.
//!
//! Determinism. The kernel has no source of nondeterminism: no clock, no
//! hashing, no threads, no host callbacks. d3 separates coincident nodes
//! with a random nudge; the kernel uses the same generator d3's own
//! `forceSimulation` defaults to (an LCG seeded with 1, one stream per
//! simulation, drawn in the same order), so the same calls give the same
//! positions bit for bit on every run. `reset_random` rewinds the stream.
//!
//! Differences from d3-force that are deliberate:
//! - arithmetic is f32 (d3 is f64);
//! - a node whose position is not finite is skipped by many-body and
//!   collide rather than poisoning its neighbours, and a velocity that is
//!   not finite is zeroed at integration;
//! - a link naming a node index that does not exist is skipped (d3 throws);
//! - a node whose x, y or radial target is NaN is left alone by that force
//!   (d3 means to, by giving it strength 0, but 0 * NaN is NaN);
//! - a per-node or per-link array shorter than the node or link count
//!   leaves the elements it does not cover untouched by that force.

use wasm_bindgen::prelude::*;

mod forces;
mod quadtree;
#[cfg(test)]
mod reference;
#[cfg(all(test, not(feature = "f64")))]
mod tests;
#[cfg(test)]
mod tests_new_forces;

use forces::{Force, Kind, Lcg, Per, Region, Scope, State};

/// The float type of the whole kernel: positions, velocities, parameters
/// and every intermediate. f32 as shipped.
///
/// The `f64` cargo feature builds the same source at f64 instead. That build
/// is not for shipping (its JS API takes and returns Float64Array): it exists
/// so the port can be checked against d3-force bit for bit, which separates
/// "the algorithm is d3's" from "f32 rounds differently from f64".
#[cfg(not(feature = "f64"))]
pub type Real = f32;
#[cfg(feature = "f64")]
pub type Real = f64;

// For console logging during debug
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

#[cfg(feature = "console_error_panic_hook")]
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Initialize the WASM module (call once at startup)
#[wasm_bindgen]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    set_panic_hook();
}

// The built-in force slots, in the order a tick applies them.
const SLOT_CENTER: usize = 0;
const SLOT_MANY_BODY: usize = 1;
const SLOT_LINK: usize = 2;
const SLOT_X: usize = 3;
const SLOT_Y: usize = 4;
const SLOT_COLLIDE: usize = 5;

const SNAPSHOT_VERSION: f64 = 1.0;
const SNAPSHOT_HEADER: usize = 8;

/// Force simulation state
///
/// Memory layout (all f32):
/// - positions: [x0, y0, x1, y1, ...] - length = n * 2
/// - velocities: [vx0, vy0, vx1, vy1, ...] - length = n * 2
/// - links: [source0, target0, source1, target1, ...] - u32 indices
#[wasm_bindgen]
pub struct Simulation {
    state: State,

    alpha: Real,
    alpha_min: Real,
    alpha_decay: Real,
    alpha_target: Real,
    velocity_decay: Real,

    /// The built-in slots (one force of each of six types, uniform parameters).
    slots: [Force; 6],
    slot_enabled: [bool; 6],

    /// The force list: any number of forces, applied in order after the slots.
    forces: Vec<Force>,

    /// Positions as they were when the last tick began, for `max_displacement`.
    previous: Vec<Real>,

    /// Each node's step in the tick before the last one (its change of
    /// position, [dx0, dy0, ...]), and how the last tick's steps differed
    /// from them: for `max_acceleration`. `history` counts the ticks made
    /// since the measures were last reset, up to 2.
    previous_step: Vec<Real>,
    acceleration: f64,
    history: u8,
}

#[wasm_bindgen]
impl Simulation {
    /// Create a new simulation with the given number of nodes
    #[wasm_bindgen(constructor)]
    pub fn new(n_nodes: usize) -> Simulation {
        let mut sim = Simulation {
            state: State::new(n_nodes),

            // Default simulation parameters (matching D3)
            alpha: 1.0,
            alpha_min: 0.001,
            alpha_decay: (1.0 - f64::powf(0.001, 1.0 / 300.0)) as Real, // ~0.0228
            alpha_target: 0.0,
            velocity_decay: 0.6,

            slots: [
                Force::center(0.0, 0.0, 1.0),
                Force::many_body(-30.0, 0.9, 1.0, f64::INFINITY),
                Force::link(30.0, 1.0, 1),
                Force::x(0.0, 0.1),
                Force::y(0.0, 0.1),
                Force::collide(10.0, 1.0, 1),
            ],
            slot_enabled: [true, true, true, false, false, false],

            forces: Vec::new(),
            previous: Vec::new(),
            previous_step: Vec::new(),
            acceleration: 0.0,
            history: 0,
        };

        // Initialize positions in a phyllotaxis pattern (like D3)
        sim.initialize_positions();

        sim
    }

    /// d3's default position for node `i`: a point on a phyllotaxis
    /// (sunflower) spiral.
    fn phyllotaxis(i: usize) -> (Real, Real) {
        let golden_angle = std::f64::consts::PI * (3.0 - f64::sqrt(5.0));
        let r = 10.0 * f64::sqrt(0.5 + i as f64);
        let theta = i as f64 * golden_angle;
        ((r * theta.cos()) as Real, (r * theta.sin()) as Real)
    }

    /// Initialize node positions in a phyllotaxis (sunflower) pattern
    fn initialize_positions(&mut self) {
        for i in 0..self.state.n {
            let (x, y) = Self::phyllotaxis(i);
            self.state.positions[i * 2] = x;
            self.state.positions[i * 2 + 1] = y;
        }
    }

    /// Set the number of nodes and resize arrays. Every position is reset
    /// to the phyllotaxis pattern; velocities and fixed coordinates keep
    /// their values by index. Per-node arrays given to forces are not
    /// resized: set them again.
    #[wasm_bindgen]
    pub fn set_node_count(&mut self, n: usize) {
        self.previous_step.clear();
        self.history = 0;
        self.state.n = n;
        self.state.positions.resize(n * 2, 0.0);
        self.state.velocities.resize(n * 2, 0.0);
        self.state.fixed.resize(n * 2, Real::NAN);
        self.initialize_positions();
        self.state.index_links();
    }

    /// Set links from a flat array of [source, target, source, target, ...]
    /// of node indices. Shared by every link force.
    #[wasm_bindgen]
    pub fn set_links(&mut self, links: &[u32]) {
        self.state.links = links[..links.len() / 2 * 2].to_vec();
        self.state.index_links();
    }

    /// Get pointer to positions array (for reading from JS)
    #[wasm_bindgen]
    pub fn positions_ptr(&self) -> *const Real {
        self.state.positions.as_ptr()
    }

    /// Get pointer to velocities array
    #[wasm_bindgen]
    pub fn velocities_ptr(&self) -> *const Real {
        self.state.velocities.as_ptr()
    }

    /// Set a single node's position
    #[wasm_bindgen]
    pub fn set_position(&mut self, index: usize, x: Real, y: Real) {
        if index < self.state.n {
            self.state.positions[index * 2] = x;
            self.state.positions[index * 2 + 1] = y;
        }
    }

    /// Set every position from [x0, y0, x1, y1, ...]. A node given a NaN
    /// coordinate gets its phyllotaxis position instead, as in d3.
    #[wasm_bindgen]
    pub fn set_positions(&mut self, positions: &[Real]) {
        let count = self.state.n.min(positions.len() / 2);
        for i in 0..count {
            let (mut x, mut y) = (positions[i * 2], positions[i * 2 + 1]);
            if x.is_nan() || y.is_nan() {
                (x, y) = Self::phyllotaxis(i);
            }
            self.state.positions[i * 2] = x;
            self.state.positions[i * 2 + 1] = y;
        }
    }

    /// Set a single node's velocity
    #[wasm_bindgen]
    pub fn set_velocity(&mut self, index: usize, vx: Real, vy: Real) {
        if index < self.state.n {
            self.state.velocities[index * 2] = vx;
            self.state.velocities[index * 2 + 1] = vy;
        }
    }

    /// Set every velocity from [vx0, vy0, vx1, vy1, ...]. NaN becomes 0.
    #[wasm_bindgen]
    pub fn set_velocities(&mut self, velocities: &[Real]) {
        let count = (self.state.n * 2).min(velocities.len());
        for k in 0..count {
            let v = velocities[k];
            self.state.velocities[k] = if v.is_nan() { 0.0 } else { v };
        }
    }

    /// Fix a node at a position (like D3's fx/fy). Pass NaN for a
    /// coordinate to leave that axis free.
    #[wasm_bindgen]
    pub fn fix_node(&mut self, index: usize, x: Real, y: Real) {
        if index < self.state.n {
            self.state.fixed[index * 2] = x;
            self.state.fixed[index * 2 + 1] = y;
        }
    }

    /// Unfix a node
    #[wasm_bindgen]
    pub fn unfix_node(&mut self, index: usize) {
        if index < self.state.n {
            self.state.fixed[index * 2] = Real::NAN;
            self.state.fixed[index * 2 + 1] = Real::NAN;
        }
    }

    /// Set every node's fixed coordinates from [fx0, fy0, fx1, fy1, ...],
    /// NaN meaning "free". Nodes past the end of the array are freed.
    #[wasm_bindgen]
    pub fn set_fixed(&mut self, fixed: &[Real]) {
        for k in 0..self.state.n * 2 {
            self.state.fixed[k] = fixed.get(k).copied().unwrap_or(Real::NAN);
        }
    }

    // ------------------------------------------------------------------
    // Built-in force slots (one of each type, uniform parameters)
    // ------------------------------------------------------------------

    /// Configure many-body force
    #[wasm_bindgen]
    pub fn configure_many_body(&mut self, strength: Real, theta: f64, dist_min: f64, dist_max: f64) {
        self.slots[SLOT_MANY_BODY] = Force::many_body(strength, theta, dist_min, dist_max);
    }

    /// Configure link force
    #[wasm_bindgen]
    pub fn configure_links(&mut self, distance: Real, strength: Real, iterations: u32) {
        self.slots[SLOT_LINK] = Force::link(distance, strength, iterations);
    }

    /// Configure center force
    #[wasm_bindgen]
    pub fn configure_center(&mut self, x: Real, y: Real, strength: Real) {
        self.slots[SLOT_CENTER] = Force::center(x, y, strength);
    }

    /// Configure forceX - pulls nodes toward an x position
    #[wasm_bindgen]
    pub fn configure_force_x(&mut self, target: Real, strength: Real) {
        self.slots[SLOT_X] = Force::x(target, strength);
    }

    /// Configure forceY - pulls nodes toward a y position
    #[wasm_bindgen]
    pub fn configure_force_y(&mut self, target: Real, strength: Real) {
        self.slots[SLOT_Y] = Force::y(target, strength);
    }

    /// Configure forceCollide - collision detection
    #[wasm_bindgen]
    pub fn configure_collide(&mut self, radius: Real, strength: Real, iterations: u32) {
        self.slots[SLOT_COLLIDE] = Force::collide(radius, strength, iterations);
    }

    /// Enable/disable forces (original 3)
    #[wasm_bindgen]
    pub fn enable_force(&mut self, many_body: bool, links: bool, center: bool) {
        self.slot_enabled[SLOT_MANY_BODY] = many_body;
        self.slot_enabled[SLOT_LINK] = links;
        self.slot_enabled[SLOT_CENTER] = center;
    }

    /// Enable/disable all forces
    #[wasm_bindgen]
    pub fn enable_forces_all(&mut self, many_body: bool, links: bool, center: bool, force_x: bool, force_y: bool, collide: bool) {
        self.slot_enabled[SLOT_MANY_BODY] = many_body;
        self.slot_enabled[SLOT_LINK] = links;
        self.slot_enabled[SLOT_CENTER] = center;
        self.slot_enabled[SLOT_X] = force_x;
        self.slot_enabled[SLOT_Y] = force_y;
        self.slot_enabled[SLOT_COLLIDE] = collide;
    }

    // ------------------------------------------------------------------
    // The force list (any number of forces of any type)
    // ------------------------------------------------------------------

    /// Remove every force from the force list. The built-in slots are not
    /// affected; turn those off with `enable_forces_all`.
    #[wasm_bindgen]
    pub fn clear_forces(&mut self) {
        self.forces.clear();
    }

    /// Number of forces in the force list
    #[wasm_bindgen]
    pub fn force_count(&self) -> usize {
        self.forces.len()
    }

    fn push_force(&mut self, force: Force) -> u32 {
        self.forces.push(force);
        (self.forces.len() - 1) as u32
    }

    /// Add a many-body (charge) force. Returns its id in the force list.
    #[wasm_bindgen]
    pub fn add_many_body(&mut self, strength: Real, theta: f64, dist_min: f64, dist_max: f64) -> u32 {
        self.push_force(Force::many_body(strength, theta, dist_min, dist_max))
    }

    /// Add a link (spring) force over the simulation's links. Returns its id.
    #[wasm_bindgen]
    pub fn add_links(&mut self, distance: Real, strength: Real, iterations: u32) -> u32 {
        self.push_force(Force::link(distance, strength, iterations))
    }

    /// Add a centering force. Returns its id.
    #[wasm_bindgen]
    pub fn add_center(&mut self, x: Real, y: Real, strength: Real) -> u32 {
        self.push_force(Force::center(x, y, strength))
    }

    /// Add a force pulling nodes toward an x coordinate. Returns its id.
    #[wasm_bindgen]
    pub fn add_force_x(&mut self, target: Real, strength: Real) -> u32 {
        self.push_force(Force::x(target, strength))
    }

    /// Add a force pulling nodes toward a y coordinate. Returns its id.
    #[wasm_bindgen]
    pub fn add_force_y(&mut self, target: Real, strength: Real) -> u32 {
        self.push_force(Force::y(target, strength))
    }

    /// Add a collision force. Returns its id.
    #[wasm_bindgen]
    pub fn add_collide(&mut self, radius: Real, strength: Real, iterations: u32) -> u32 {
        self.push_force(Force::collide(radius, strength, iterations))
    }

    /// Add a radial force: pull nodes toward a circle of the given radius
    /// around (x, y). Returns its id.
    #[wasm_bindgen]
    pub fn add_radial(&mut self, radius: Real, x: Real, y: Real, strength: Real) -> u32 {
        self.push_force(Force::radial(radius, x, y, strength))
    }

    /// Give a force one strength per node (many-body, x, y, radial, towards,
    /// cohesion) or one per link (link, directional). Returns false, changing
    /// nothing, for a force whose strength is a single number (center,
    /// collide, rect-collide, containment) or an unknown id.
    #[wasm_bindgen]
    pub fn set_force_strengths(&mut self, id: u32, strengths: &[Real]) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(force) if !force.strength_is_uniform() => {
                force.strength = Per::Each(strengths.to_vec());
                true
            }
            _ => false,
        }
    }

    /// Give a force one radius per node (collide, radial, containment).
    /// Returns false for any other force.
    #[wasm_bindgen]
    pub fn set_force_radii(&mut self, id: u32, radii: &[Real]) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(force) => match &mut force.kind {
                Kind::Collide { radius, .. } | Kind::Radial { radius, .. } | Kind::Contain { radius, .. } => {
                    // The same number of radii as before: overwrite in place,
                    // so radii can be replaced every tick without allocating.
                    if let Per::Each(current) = radius {
                        if current.len() == radii.len() {
                            current.copy_from_slice(radii);
                            return true;
                        }
                    }
                    *radius = Per::Each(radii.to_vec());
                    true
                }
                _ => false,
            },
            None => false,
        }
    }

    /// Give a force one target coordinate per node (x, y). A NaN target
    /// leaves that node alone. Returns false for any other force.
    #[wasm_bindgen]
    pub fn set_force_targets(&mut self, id: u32, targets: &[Real]) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(force) => match &mut force.kind {
                Kind::X { target } | Kind::Y { target } => {
                    *target = Per::Each(targets.to_vec());
                    true
                }
                _ => false,
            },
            None => false,
        }
    }

    /// Give a link force one rest length per link, in the order of
    /// `set_links`. Returns false for any other force.
    #[wasm_bindgen]
    pub fn set_force_distances(&mut self, id: u32, distances: &[Real]) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(force) => match &mut force.kind {
                Kind::Link { distance, .. } => {
                    *distance = Per::Each(distances.to_vec());
                    true
                }
                _ => false,
            },
            None => false,
        }
    }

    /// Filter a force by node: one byte per node, 0 = the node has strength
    /// 0 under this force. For x, y and radial that means the force leaves
    /// the node alone; for many-body it means the node exerts no charge but
    /// is still pushed by the others (the D3 kernel's filtered many-body
    /// does the same). Towards and cohesion take a mask too. Returns false
    /// for forces without a per-node strength (link, center, collide,
    /// rect-collide, containment, directional).
    #[wasm_bindgen]
    pub fn set_force_mask(&mut self, id: u32, mask: &[u8]) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(force) if force.strength_is_per_node() => {
                force.mask = Some(mask.to_vec());
                true
            }
            _ => false,
        }
    }

    /// Remove a force's node filter.
    #[wasm_bindgen]
    pub fn clear_force_mask(&mut self, id: u32) {
        if let Some(force) = self.forces.get_mut(id as usize) {
            force.mask = None;
        }
    }

    // ------------------------------------------------------------------
    // Collision that yields (not in d3). The rule is on `apply_collide`.
    // ------------------------------------------------------------------

    /// Make a collide force's strength follow alpha (strength * alpha) or,
    /// with false, be d3's again (the default). Returns false for any other
    /// force.
    #[wasm_bindgen]
    pub fn set_collide_alpha_scaled(&mut self, id: u32, scaled: bool) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(Force { kind: Kind::Collide { alpha_scaled, .. }, .. }) => {
                *alpha_scaled = scaled;
                true
            }
            _ => false,
        }
    }

    /// Filter a collide force by node: one byte per node, 0 (or past the end
    /// of the array) = the node collides with nothing under this force.
    /// `clear_force_mask` removes the filter. Returns false for any other
    /// force. (`set_force_mask` still refuses a collide force, as it always
    /// has: its masks mean "strength 0", and collide has no per-node
    /// strength.)
    #[wasm_bindgen]
    pub fn set_collide_filter(&mut self, id: u32, mask: &[u8]) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(force) if matches!(force.kind, Kind::Collide { .. } | Kind::RectCollide { .. }) => {
                force.mask = Some(mask.to_vec());
                true
            }
            _ => false,
        }
    }

    // ------------------------------------------------------------------
    // Forces d3-force does not have. Each rule is stated in full on the
    // matching `apply_*` function in forces.rs.
    // ------------------------------------------------------------------

    /// Add a force pulling each node toward its own target point (set the
    /// targets with `set_force_targets_xy`; until then it does nothing).
    /// Per node: v += (target - position) * strength * alpha. Returns its id.
    #[wasm_bindgen]
    pub fn add_towards(&mut self, strength: Real) -> u32 {
        self.push_force(Force::towards(strength))
    }

    /// Give a towards force its targets, [tx0, ty0, tx1, ty1, ...]. A node
    /// with a NaN in its target, or past the end of the array, is left
    /// alone. Calling this again with an array of the same length overwrites
    /// the targets in place (no allocation), so targets can be replaced
    /// every tick. Returns false for any other force.
    #[wasm_bindgen]
    pub fn set_force_targets_xy(&mut self, id: u32, targets: &[Real]) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(Force { kind: Kind::Towards { targets: current }, .. }) => {
                if current.len() == targets.len() {
                    current.copy_from_slice(targets);
                } else {
                    *current = targets.to_vec();
                }
                true
            }
            _ => false,
        }
    }

    /// Add a rectangle collision force: nodes are axis-aligned rectangles
    /// with the given half-width and half-height, pushed apart along the
    /// axis of least overlap. Returns its id.
    #[wasm_bindgen]
    pub fn add_rect_collide(&mut self, half_width: Real, half_height: Real, strength: Real, iterations: u32) -> u32 {
        self.push_force(Force::rect_collide(half_width, half_height, strength, iterations))
    }

    /// Give a rectangle collision force one half-width per node, or a
    /// containment force the half-width of each node's body (default 0: a
    /// circle of its radius). Returns false for any other force.
    #[wasm_bindgen]
    pub fn set_force_half_widths(&mut self, id: u32, half_widths: &[Real]) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(Force { kind: Kind::RectCollide { half_width, .. } | Kind::Contain { half_width, .. }, .. }) => {
                *half_width = Per::Each(half_widths.to_vec());
                true
            }
            _ => false,
        }
    }

    /// Give a rectangle collision force one half-height per node, or a
    /// containment force the half-height of each node's body. Returns false
    /// for any other force.
    #[wasm_bindgen]
    pub fn set_force_half_heights(&mut self, id: u32, half_heights: &[Real]) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(Force { kind: Kind::RectCollide { half_height, .. } | Kind::Contain { half_height, .. }, .. }) => {
                *half_height = Per::Each(half_heights.to_vec());
                true
            }
            _ => false,
        }
    }

    /// Restrict a rectangle collision by group (set the groups with
    /// `set_force_groups`): 0 = every pair, 1 = only two nodes of the same
    /// group, 2 = only two nodes not of the same group. A node filter is set
    /// with `set_collide_filter`. Returns false for any other force or any
    /// other number.
    #[wasm_bindgen]
    pub fn set_rect_collide_pairs(&mut self, id: u32, pairs: u8) -> bool {
        let scope = match pairs {
            0 => Scope::All,
            1 => Scope::Within,
            2 => Scope::Between,
            _ => return false,
        };
        match self.forces.get_mut(id as usize) {
            Some(Force { kind: Kind::RectCollide { pairs: current, .. }, .. }) => {
                *current = scope;
                true
            }
            _ => false,
        }
    }

    /// What a rectangle pushed from both sides does: false (the default), it
    /// stays where the pushes cancel; true, it goes through one side (THE
    /// SQUEEZE, on `apply_rect_collide`). Returns false for any other force.
    #[wasm_bindgen]
    pub fn set_rect_collide_through(&mut self, id: u32, through: bool) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(Force { kind: Kind::RectCollide { through: current, .. }, .. }) => {
                *current = through;
                true
            }
            _ => false,
        }
    }

    /// Add a containment force. It has no regions yet: add them with
    /// `containment_add_rect` / `_circle` / `_polygon`. Every node is kept in
    /// region 0 unless `set_force_regions` assigns regions per node; node
    /// radii come from `set_force_radii` (default 0). `strength` steers
    /// nodes back in before integration (0 turns that off); `hard` also
    /// clamps positions into the region after integration. Returns its id.
    #[wasm_bindgen]
    pub fn add_containment(&mut self, strength: Real, hard: bool) -> u32 {
        self.push_force(Force::contain(strength, hard))
    }

    fn push_region(&mut self, id: u32, region: Region) -> i32 {
        match self.forces.get_mut(id as usize) {
            Some(Force { kind: Kind::Contain { regions, .. }, .. }) => {
                regions.push(region);
                (regions.len() - 1) as i32
            }
            _ => -1,
        }
    }

    /// Add a rectangular region to a containment force. Returns the
    /// region's index, or -1 if `id` is not a containment force.
    #[wasm_bindgen]
    pub fn containment_add_rect(&mut self, id: u32, x0: Real, y0: Real, x1: Real, y1: Real) -> i32 {
        self.push_region(id, Region::rect(x0, y0, x1, y1))
    }

    /// Add a circular region to a containment force. Returns the region's
    /// index, or -1 if `id` is not a containment force.
    #[wasm_bindgen]
    pub fn containment_add_circle(&mut self, id: u32, x: Real, y: Real, radius: Real) -> i32 {
        self.push_region(id, Region::Circle { x, y, r: radius })
    }

    /// Add a convex polygon region, [x0, y0, x1, y1, ...] in either winding,
    /// to a containment force. Returns the region's index, or -1 if `id` is
    /// not a containment force.
    #[wasm_bindgen]
    pub fn containment_add_polygon(&mut self, id: u32, vertices: &[Real]) -> i32 {
        self.push_region(id, Region::polygon(vertices))
    }

    /// Assign each node a region of a containment force by index
    /// (0xFFFFFFFF, or an index with no region, or past the end of the
    /// array: the node is free). Returns false for any other force.
    #[wasm_bindgen]
    pub fn set_force_regions(&mut self, id: u32, regions: &[u32]) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(Force { kind: Kind::Contain { assignment, .. }, .. }) => {
                *assignment = Some(regions.to_vec());
                true
            }
            _ => false,
        }
    }

    /// Add a group cohesion force: each node is pulled toward the centroid
    /// of its group (set the groups with `set_force_groups`). Returns its id.
    #[wasm_bindgen]
    pub fn add_cohesion(&mut self, strength: Real) -> u32 {
        self.push_force(Force::cohesion(strength))
    }

    /// Give a cohesion, many-body or rectangle collision force one group id per node
    /// (0xFFFFFFFF, or past the end of the array: in no group). Returns
    /// false for any other force.
    #[wasm_bindgen]
    pub fn set_force_groups(&mut self, id: u32, groups: &[u32]) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(force) if matches!(force.kind, Kind::Cohesion | Kind::ManyBody { .. } | Kind::RectCollide { .. }) => {
                force.groups = Some(groups.to_vec());
                true
            }
            _ => false,
        }
    }

    /// Restrict a many-body force by group: 0 = every pair (d3's), 1 = only
    /// pairs in the same group, 2 = only pairs in different groups. Returns
    /// false for any other force or any other number.
    #[wasm_bindgen]
    pub fn set_many_body_scope(&mut self, id: u32, scope: u8) -> bool {
        let scope = match scope {
            0 => Scope::All,
            1 => Scope::Within,
            2 => Scope::Between,
            _ => return false,
        };
        match self.forces.get_mut(id as usize) {
            Some(Force { kind: Kind::ManyBody { scope: current, .. }, .. }) => {
                *current = scope;
                true
            }
            _ => false,
        }
    }

    /// Add a directional link constraint over the simulation's links: each
    /// link's target is pushed to be at least `gap` beyond its source along
    /// the axis (normalised here; (0, 1) is "below" in screen coordinates).
    /// Per-link gaps with `set_force_gaps`, per-link strengths with
    /// `set_force_strengths`. Returns its id.
    #[wasm_bindgen]
    pub fn add_directional(&mut self, gap: Real, axis_x: Real, axis_y: Real, strength: Real, iterations: u32) -> u32 {
        self.push_force(Force::directional(gap, axis_x, axis_y, strength, iterations))
    }

    /// Make a directional constraint hard: after integration its links are
    /// projected so that the gap asked for is the gap kept (HARD DIRECTIONAL
    /// CONSTRAINT, on `Force::constrain`). The soft force still applies
    /// unless its strength is 0. Returns false for any other force.
    #[wasm_bindgen]
    pub fn set_directional_hard(&mut self, id: u32, hard: bool) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(Force { kind: Kind::Directional { hard: current, .. }, .. }) => {
                *current = hard;
                true
            }
            _ => false,
        }
    }

    /// Give a directional constraint one gap per link, in the order of
    /// `set_links`. Returns false for any other force.
    #[wasm_bindgen]
    pub fn set_force_gaps(&mut self, id: u32, gaps: &[Real]) -> bool {
        match self.forces.get_mut(id as usize) {
            Some(Force { kind: Kind::Directional { gap, .. }, .. }) => {
                *gap = Per::Each(gaps.to_vec());
                true
            }
            _ => false,
        }
    }

    // ------------------------------------------------------------------
    // Snapshot, restore, and measures of convergence
    // ------------------------------------------------------------------

    /// Everything that changes as the simulation runs, as one flat array:
    /// [version, node count, alpha, alpha_min, alpha_decay, alpha_target,
    /// velocity_decay, random stream state, positions (2n), velocities (2n),
    /// fixed coordinates (2n, NaN = free)].
    ///
    /// `restore` on a simulation with the same forces and links, followed by
    /// N ticks, gives exactly the positions that N more ticks would have
    /// given here. Forces and links are configuration, not state, and are
    /// not in the snapshot. (f64 holds every f32 and every u32 exactly.)
    #[wasm_bindgen]
    pub fn snapshot(&self) -> Vec<f64> {
        let st = &self.state;
        let mut out = Vec::with_capacity(SNAPSHOT_HEADER + st.n * 6);
        out.extend([
            SNAPSHOT_VERSION,
            st.n as f64,
            self.alpha as f64,
            self.alpha_min as f64,
            self.alpha_decay as f64,
            self.alpha_target as f64,
            self.velocity_decay as f64,
            st.rng.state() as f64,
        ]);
        out.extend(st.positions.iter().map(|&v| v as f64));
        out.extend(st.velocities.iter().map(|&v| v as f64));
        out.extend(st.fixed.iter().map(|&v| v as f64));
        out
    }

    /// Put the simulation back in a state `snapshot` recorded. Returns
    /// false, changing nothing, if the array is not a snapshot of this
    /// version or its length does not match its node count. If the node
    /// count differs from the current one the simulation is resized.
    #[wasm_bindgen]
    pub fn restore(&mut self, snapshot: &[f64]) -> bool {
        if snapshot.len() < SNAPSHOT_HEADER || snapshot[0] != SNAPSHOT_VERSION {
            return false;
        }
        let n = snapshot[1];
        if !(n >= 0.0 && n.fract() == 0.0 && n <= u32::MAX as f64) {
            return false;
        }
        let n = n as usize;
        if snapshot.len() != SNAPSHOT_HEADER + n * 6 {
            return false;
        }
        let seed = snapshot[7];
        if !(seed >= 0.0 && seed.fract() == 0.0 && seed <= u32::MAX as f64) {
            return false;
        }
        self.alpha = snapshot[2] as Real;
        self.alpha_min = snapshot[3] as Real;
        self.alpha_decay = snapshot[4] as Real;
        self.alpha_target = snapshot[5] as Real;
        self.velocity_decay = snapshot[6] as Real;
        let st = &mut self.state;
        st.rng = Lcg::from_state(seed as u32);
        let body = &snapshot[SNAPSHOT_HEADER..];
        let width = n * 2;
        st.positions = body[..width].iter().map(|&v| v as Real).collect();
        st.velocities = body[width..width * 2].iter().map(|&v| v as Real).collect();
        st.fixed = body[width * 2..].iter().map(|&v| v as Real).collect();
        if st.n != n {
            st.n = n;
            st.index_links();
        }
        self.previous.clear();
        self.previous_step.clear();
        self.acceleration = 0.0;
        self.history = 0;
        true
    }

    /// Kinetic energy after the last tick, taking every node to have unit
    /// mass: the sum over nodes of (vx^2 + vy^2) / 2, where the velocity is
    /// the displacement the node made in that tick's integration. Summed in
    /// f64. Falls toward 0 as the layout settles.
    #[wasm_bindgen]
    pub fn kinetic_energy(&self) -> f64 {
        self.state.velocities.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / 2.0
    }

    /// The furthest any node moved in the last tick: the largest distance
    /// between a node's position when the tick began and when it ended,
    /// everything included (forces, centering, clamps, pins). 0 before the
    /// first tick and straight after `restore` or a change of node count.
    #[wasm_bindgen]
    pub fn max_displacement(&self) -> f64 {
        let now = &self.state.positions;
        if self.previous.len() != now.len() {
            return 0.0;
        }
        let mut worst = 0.0f64;
        for i in 0..now.len() / 2 {
            let dx = now[i * 2] as f64 - self.previous[i * 2] as f64;
            let dy = now[i * 2 + 1] as f64 - self.previous[i * 2 + 1] as f64;
            let d2 = dx * dx + dy * dy;
            if d2 > worst {
                worst = d2;
            }
        }
        worst.sqrt()
    }

    /// The largest change in any node's step between the last two ticks:
    /// with s(t) a node's change of position in tick t, everything included,
    /// the largest |s(last) - s(the one before)| over the nodes. It is the
    /// acceleration the node actually underwent, in distance per tick per
    /// tick. 0 until two ticks have been made since the simulation was
    /// created, restored or resized.
    ///
    /// This is what `max_displacement` and `kinetic_energy` cannot see. A
    /// node on a spring is momentarily still at the far end of every swing,
    /// and if the nodes swing in step all of them are still at once: both of
    /// those measures vanish there, far from rest. The acceleration does
    /// not: at the end of a swing it is at its largest. Conversely a node
    /// crossing its resting place at speed has no acceleration but a large
    /// step. A simulation is at rest when both are small: see `settled`.
    ///
    /// It is measured from positions, not from the forces, so it is right
    /// whatever moved the node: a node held against a hard region by a force
    /// that keeps pushing has a net force but no acceleration, and counts as
    /// at rest, which it is.
    #[wasm_bindgen]
    pub fn max_acceleration(&self) -> f64 {
        if self.history < 2 {
            0.0
        } else {
            self.acceleration
        }
    }

    /// Is the simulation at rest? True when, over the last two ticks, no
    /// node moved further than `step` in a tick (`max_displacement`) and no
    /// node's step changed by more than `acceleration` (`max_acceleration`).
    /// False until two ticks have been made since the simulation was
    /// created, restored or resized.
    ///
    /// It says what the last two ticks did. Changing a force, a target or a
    /// position does not make it false until a tick has been made.
    ///
    /// What it promises. For a node held by a spring of stiffness k
    /// (velocity += k * distance each tick) with velocity retention r, a
    /// step below s and an acceleration below a put the node within
    /// (a + (1 - r) * (s + a)) / (k * r) of where it will stop, and it does
    /// not get further away afterwards. So stop on the first tick `settled`
    /// is true. (While a swing is still dying away, a later tick can be
    /// false again, by a hair: the two measures peak at different moments
    /// of a swing.)
    #[wasm_bindgen]
    pub fn settled(&self, step: f64, acceleration: f64) -> bool {
        self.history >= 2 && self.max_displacement() < step && self.acceleration < acceleration
    }

    // ------------------------------------------------------------------
    // Simulation control
    // ------------------------------------------------------------------

    /// Set alpha (temperature)
    #[wasm_bindgen]
    pub fn set_alpha(&mut self, alpha: Real) {
        self.alpha = alpha;
    }

    /// Get current alpha
    #[wasm_bindgen]
    pub fn get_alpha(&self) -> Real {
        self.alpha
    }

    /// Set alpha decay rate
    #[wasm_bindgen]
    pub fn set_alpha_decay(&mut self, decay: Real) {
        self.alpha_decay = decay;
    }

    /// Set the alpha below which the simulation counts as finished
    #[wasm_bindgen]
    pub fn set_alpha_min(&mut self, alpha_min: Real) {
        self.alpha_min = alpha_min;
    }

    /// Set the alpha the simulation cools toward
    #[wasm_bindgen]
    pub fn set_alpha_target(&mut self, alpha_target: Real) {
        self.alpha_target = alpha_target;
    }

    /// Set velocity decay (friction): the factor velocities are multiplied
    /// by each tick.
    #[wasm_bindgen]
    pub fn set_velocity_decay(&mut self, decay: Real) {
        self.velocity_decay = decay;
    }

    /// Set the share of its velocity a node keeps from one tick to the next
    /// (0.6 by default: d3's `velocityDecay` of 0.4). The same setting as
    /// `set_velocity_decay`, under the name that says which way it runs.
    #[wasm_bindgen]
    pub fn set_velocity_retention(&mut self, retention: Real) {
        self.velocity_decay = retention;
    }

    /// Reheat the simulation
    #[wasm_bindgen]
    pub fn reheat(&mut self) {
        self.alpha = 1.0;
    }

    /// Rewind the random stream used to separate coincident nodes to its
    /// initial state.
    #[wasm_bindgen]
    pub fn reset_random(&mut self) {
        self.state.rng = Lcg::new();
    }

    /// Check if any position or velocity is NaN
    fn has_nan(&self) -> bool {
        self.state.positions.iter().any(|x| x.is_nan()) ||
        self.state.velocities.iter().any(|x| x.is_nan())
    }

    /// Debug helper: log first few positions and velocities
    fn log_state(&self, label: &str) {
        let p0 = self.state.positions.get(0).unwrap_or(&0.0);
        let p1 = self.state.positions.get(1).unwrap_or(&0.0);
        let v0 = self.state.velocities.get(0).unwrap_or(&0.0);
        let v1 = self.state.velocities.get(1).unwrap_or(&0.0);
        console_log!("{}: pos[0]=({:.2},{:.2}) vel[0]=({:.2},{:.2}) nan={}",
            label, p0, p1, v0, v1, self.has_nan());
    }

    /// Run a single simulation tick with debug logging
    #[wasm_bindgen]
    pub fn debug_tick(&mut self) -> Real {
        self.log_state("START");
        let alpha = self.tick();
        if self.has_nan() {
            self.log_state("AFTER TICK - NaN!");
        }
        alpha
    }

    /// Run a single simulation tick
    /// Returns the new alpha value
    #[wasm_bindgen]
    pub fn tick(&mut self) -> Real {
        let alpha = self.alpha;

        self.previous.clear();
        self.previous.extend_from_slice(&self.state.positions);

        for (slot, force) in self.slots.iter().enumerate() {
            if self.slot_enabled[slot] {
                force.apply(&mut self.state, alpha);
            }
        }
        for force in &self.forces {
            force.apply(&mut self.state, alpha);
        }

        // Integrate positions
        self.integrate();

        // Hard constraints, in the order of the forces, then pins again: a
        // pin beats a region and a gap.
        let mut constrained = false;
        for force in &self.forces {
            constrained |= force.constrain(&mut self.state);
        }
        if constrained {
            let State { positions, velocities, fixed, .. } = &mut self.state;
            for k in 0..positions.len() {
                if !fixed[k].is_nan() {
                    positions[k] = fixed[k];
                    velocities[k] = 0.0;
                }
            }
        }

        self.measure_acceleration();

        // Decay alpha
        self.alpha += (self.alpha_target - self.alpha) * self.alpha_decay;

        self.alpha
    }

    /// Record this tick's steps and how they differ from the last tick's.
    /// Reads positions only; nothing the simulation computes depends on it.
    fn measure_acceleration(&mut self) {
        let now = &self.state.positions;
        if self.previous.len() != now.len() {
            self.previous_step.clear();
            self.history = 0;
            return;
        }
        let comparable = self.history >= 1 && self.previous_step.len() == now.len();
        if !comparable {
            self.previous_step.clear();
            self.previous_step.resize(now.len(), 0.0);
            self.history = 0;
        }
        let mut worst = 0.0f64;
        for i in 0..now.len() / 2 {
            let sx = now[i * 2] - self.previous[i * 2];
            let sy = now[i * 2 + 1] - self.previous[i * 2 + 1];
            if comparable {
                let ax = sx as f64 - self.previous_step[i * 2] as f64;
                let ay = sy as f64 - self.previous_step[i * 2 + 1] as f64;
                let a2 = ax * ax + ay * ay;
                if a2 > worst {
                    worst = a2;
                }
            }
            self.previous_step[i * 2] = sx;
            self.previous_step[i * 2 + 1] = sy;
        }
        self.acceleration = worst.sqrt();
        self.history = if comparable { 2 } else { 1 };
    }

    /// Run multiple ticks
    #[wasm_bindgen]
    pub fn tick_n(&mut self, n: u32) -> Real {
        for _ in 0..n {
            if self.alpha < self.alpha_min {
                break;
            }
            self.tick();
        }
        self.alpha
    }

    /// Check if simulation is still running
    #[wasm_bindgen]
    pub fn is_running(&self) -> bool {
        self.alpha >= self.alpha_min
    }

    /// Integrate velocities into positions, one axis at a time: a fixed
    /// coordinate snaps to its value with zero velocity, a free one decays
    /// its velocity and moves.
    fn integrate(&mut self) {
        let decay = self.velocity_decay;
        let State { positions, velocities, fixed, .. } = &mut self.state;
        for k in 0..positions.len() {
            let pin = fixed[k];
            if pin.is_nan() {
                let mut v = velocities[k];
                if !v.is_finite() {
                    v = 0.0;
                }
                v *= decay;
                velocities[k] = v;
                positions[k] += v;
            } else {
                positions[k] = pin;
                velocities[k] = 0.0;
            }
        }
    }

    /// Get all positions as a flat array (for copying to JS)
    #[wasm_bindgen]
    pub fn get_positions(&self) -> Vec<Real> {
        self.state.positions.clone()
    }

    /// Get all velocities as a flat array (for copying to JS)
    #[wasm_bindgen]
    pub fn get_velocities(&self) -> Vec<Real> {
        self.state.velocities.clone()
    }

    /// Get node count
    #[wasm_bindgen]
    pub fn node_count(&self) -> usize {
        self.state.n
    }

    /// Get link count
    #[wasm_bindgen]
    pub fn link_count(&self) -> usize {
        self.state.links.len() / 2
    }
}
