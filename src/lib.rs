//! High-performance force simulation kernel for WebAssembly
//!
//! This module implements D3-compatible force simulation algorithms
//! optimized for execution in WebAssembly.

use wasm_bindgen::prelude::*;

mod quadtree;
use quadtree::{Quadtree, QuadNode, accumulate};

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

/// Force simulation state
///
/// Memory layout (all f32):
/// - positions: [x0, y0, x1, y1, ...] - length = n * 2
/// - velocities: [vx0, vy0, vx1, vy1, ...] - length = n * 2
/// - links: [source0, target0, source1, target1, ...] - u32 indices
#[wasm_bindgen]
pub struct Simulation {
    n_nodes: usize,
    n_links: usize,

    // Node state (interleaved x, y)
    positions: Vec<f32>,
    velocities: Vec<f32>,

    // Fixed positions (NaN = not fixed)
    fixed: Vec<f32>,

    // Links as index pairs
    links: Vec<u32>,

    // Simulation parameters
    alpha: f32,
    alpha_min: f32,
    alpha_decay: f32,
    alpha_target: f32,
    velocity_decay: f32,

    // Force parameters
    many_body_strength: f32,
    many_body_theta: f32,
    many_body_distance_min: f32,
    many_body_distance_max: f32,

    link_distance: f32,
    link_strength: f32,
    link_iterations: u32,

    center_x: f32,
    center_y: f32,
    center_strength: f32,

    // ForceX parameters - pull toward x value
    force_x_target: f32,
    force_x_strength: f32,

    // ForceY parameters - pull toward y value
    force_y_target: f32,
    force_y_strength: f32,

    // ForceCollide parameters - collision detection
    collide_radius: f32,
    collide_strength: f32,
    collide_iterations: u32,

    // Which forces are enabled
    enable_many_body: bool,
    enable_links: bool,
    enable_center: bool,
    enable_force_x: bool,
    enable_force_y: bool,
    enable_collide: bool,
}

#[wasm_bindgen]
impl Simulation {
    /// Create a new simulation with the given number of nodes
    #[wasm_bindgen(constructor)]
    pub fn new(n_nodes: usize) -> Simulation {
        let mut sim = Simulation {
            n_nodes,
            n_links: 0,
            positions: vec![0.0; n_nodes * 2],
            velocities: vec![0.0; n_nodes * 2],
            fixed: vec![f32::NAN; n_nodes * 2],
            links: Vec::new(),

            // Default simulation parameters (matching D3)
            alpha: 1.0,
            alpha_min: 0.001,
            alpha_decay: 1.0 - f32::powf(0.001, 1.0 / 300.0), // ~0.0228
            alpha_target: 0.0,
            velocity_decay: 0.6,

            // Default force parameters
            many_body_strength: -30.0,
            many_body_theta: 0.9,
            many_body_distance_min: 1.0,
            many_body_distance_max: f32::INFINITY,

            link_distance: 30.0,
            link_strength: 1.0,
            link_iterations: 1,

            center_x: 0.0,
            center_y: 0.0,
            center_strength: 1.0,

            force_x_target: 0.0,
            force_x_strength: 0.1,

            force_y_target: 0.0,
            force_y_strength: 0.1,

            collide_radius: 10.0,
            collide_strength: 1.0,
            collide_iterations: 1,

            enable_many_body: true,
            enable_links: true,
            enable_center: true,
            enable_force_x: false,
            enable_force_y: false,
            enable_collide: false,
        };

        // Initialize positions in a phyllotaxis pattern (like D3)
        sim.initialize_positions();

        sim
    }

    /// Initialize node positions in a phyllotaxis (sunflower) pattern
    fn initialize_positions(&mut self) {
        let golden_angle = std::f32::consts::PI * (3.0 - f32::sqrt(5.0));
        let initial_radius = 10.0;

        for i in 0..self.n_nodes {
            let r = initial_radius * f32::sqrt(0.5 + i as f32);
            let theta = (i as f32) * golden_angle;
            self.positions[i * 2] = r * f32::cos(theta);
            self.positions[i * 2 + 1] = r * f32::sin(theta);
        }
    }

    /// Set the number of nodes and resize arrays
    #[wasm_bindgen]
    pub fn set_node_count(&mut self, n: usize) {
        self.n_nodes = n;
        self.positions.resize(n * 2, 0.0);
        self.velocities.resize(n * 2, 0.0);
        self.fixed.resize(n * 2, f32::NAN);
        self.initialize_positions();
    }

    /// Set links from a flat array of [source, target, source, target, ...]
    #[wasm_bindgen]
    pub fn set_links(&mut self, links: &[u32]) {
        self.links = links.to_vec();
        self.n_links = links.len() / 2;
    }

    /// Get pointer to positions array (for reading from JS)
    #[wasm_bindgen]
    pub fn positions_ptr(&self) -> *const f32 {
        self.positions.as_ptr()
    }

    /// Get pointer to velocities array
    #[wasm_bindgen]
    pub fn velocities_ptr(&self) -> *const f32 {
        self.velocities.as_ptr()
    }

    /// Set a single node's position
    #[wasm_bindgen]
    pub fn set_position(&mut self, index: usize, x: f32, y: f32) {
        if index < self.n_nodes {
            self.positions[index * 2] = x;
            self.positions[index * 2 + 1] = y;
        }
    }

    /// Fix a node at a position (like D3's fx/fy)
    #[wasm_bindgen]
    pub fn fix_node(&mut self, index: usize, x: f32, y: f32) {
        if index < self.n_nodes {
            self.fixed[index * 2] = x;
            self.fixed[index * 2 + 1] = y;
        }
    }

    /// Unfix a node
    #[wasm_bindgen]
    pub fn unfix_node(&mut self, index: usize) {
        if index < self.n_nodes {
            self.fixed[index * 2] = f32::NAN;
            self.fixed[index * 2 + 1] = f32::NAN;
        }
    }

    /// Configure many-body force
    #[wasm_bindgen]
    pub fn configure_many_body(&mut self, strength: f32, theta: f32, dist_min: f32, dist_max: f32) {
        self.many_body_strength = strength;
        self.many_body_theta = theta;
        self.many_body_distance_min = dist_min;
        self.many_body_distance_max = dist_max;
    }

    /// Configure link force
    #[wasm_bindgen]
    pub fn configure_links(&mut self, distance: f32, strength: f32, iterations: u32) {
        self.link_distance = distance;
        self.link_strength = strength;
        self.link_iterations = iterations;
    }

    /// Configure center force
    #[wasm_bindgen]
    pub fn configure_center(&mut self, x: f32, y: f32, strength: f32) {
        self.center_x = x;
        self.center_y = y;
        self.center_strength = strength;
    }

    /// Configure forceX - pulls nodes toward an x position
    #[wasm_bindgen]
    pub fn configure_force_x(&mut self, target: f32, strength: f32) {
        self.force_x_target = target;
        self.force_x_strength = strength;
    }

    /// Configure forceY - pulls nodes toward a y position
    #[wasm_bindgen]
    pub fn configure_force_y(&mut self, target: f32, strength: f32) {
        self.force_y_target = target;
        self.force_y_strength = strength;
    }

    /// Configure forceCollide - collision detection
    #[wasm_bindgen]
    pub fn configure_collide(&mut self, radius: f32, strength: f32, iterations: u32) {
        self.collide_radius = radius;
        self.collide_strength = strength;
        self.collide_iterations = iterations;
    }

    /// Enable/disable forces (original 3)
    #[wasm_bindgen]
    pub fn enable_force(&mut self, many_body: bool, links: bool, center: bool) {
        self.enable_many_body = many_body;
        self.enable_links = links;
        self.enable_center = center;
    }

    /// Enable/disable all forces
    #[wasm_bindgen]
    pub fn enable_forces_all(&mut self, many_body: bool, links: bool, center: bool, force_x: bool, force_y: bool, collide: bool) {
        self.enable_many_body = many_body;
        self.enable_links = links;
        self.enable_center = center;
        self.enable_force_x = force_x;
        self.enable_force_y = force_y;
        self.enable_collide = collide;
    }

    /// Set alpha (temperature)
    #[wasm_bindgen]
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha;
    }

    /// Get current alpha
    #[wasm_bindgen]
    pub fn get_alpha(&self) -> f32 {
        self.alpha
    }

    /// Set alpha decay rate
    #[wasm_bindgen]
    pub fn set_alpha_decay(&mut self, decay: f32) {
        self.alpha_decay = decay;
    }

    /// Set velocity decay (friction)
    #[wasm_bindgen]
    pub fn set_velocity_decay(&mut self, decay: f32) {
        self.velocity_decay = decay;
    }

    /// Reheat the simulation
    #[wasm_bindgen]
    pub fn reheat(&mut self) {
        self.alpha = 1.0;
    }

    /// Check if any position or velocity is NaN
    fn has_nan(&self) -> bool {
        self.positions.iter().any(|x| x.is_nan()) ||
        self.velocities.iter().any(|x| x.is_nan())
    }

    /// Debug helper: log first few positions and velocities
    fn log_state(&self, label: &str) {
        let p0 = self.positions.get(0).unwrap_or(&0.0);
        let p1 = self.positions.get(1).unwrap_or(&0.0);
        let v0 = self.velocities.get(0).unwrap_or(&0.0);
        let v1 = self.velocities.get(1).unwrap_or(&0.0);
        console_log!("{}: pos[0]=({:.2},{:.2}) vel[0]=({:.2},{:.2}) nan={}",
            label, p0, p1, v0, v1, self.has_nan());
    }

    /// Run a single simulation tick with debug logging
    #[wasm_bindgen]
    pub fn debug_tick(&mut self) -> f32 {
        self.log_state("START");

        if self.enable_center {
            self.apply_center_force();
            if self.has_nan() { self.log_state("AFTER CENTER - NaN!"); }
        }

        if self.enable_many_body {
            self.apply_many_body_force();
            if self.has_nan() { self.log_state("AFTER MANY_BODY - NaN!"); }
        }

        if self.enable_links {
            for iter in 0..self.link_iterations {
                self.apply_link_force();
                if self.has_nan() {
                    console_log!("AFTER LINK iter {} - NaN!", iter);
                    break;
                }
            }
        }

        self.integrate();
        if self.has_nan() { self.log_state("AFTER INTEGRATE - NaN!"); }

        self.alpha += (self.alpha_target - self.alpha) * self.alpha_decay;
        self.alpha
    }

    /// Run a single simulation tick
    /// Returns the new alpha value
    #[wasm_bindgen]
    pub fn tick(&mut self) -> f32 {
        // Apply forces (order matters for consistency with D3)
        if self.enable_center {
            self.apply_center_force();
        }

        if self.enable_many_body {
            self.apply_many_body_force();
        }

        if self.enable_links {
            for _ in 0..self.link_iterations {
                self.apply_link_force();
            }
        }

        if self.enable_force_x {
            self.apply_force_x();
        }

        if self.enable_force_y {
            self.apply_force_y();
        }

        if self.enable_collide {
            for _ in 0..self.collide_iterations {
                self.apply_collide_force();
            }
        }

        // Integrate positions
        self.integrate();

        // Decay alpha
        self.alpha += (self.alpha_target - self.alpha) * self.alpha_decay;

        self.alpha
    }

    /// Run multiple ticks
    #[wasm_bindgen]
    pub fn tick_n(&mut self, n: u32) -> f32 {
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

    /// Apply center force - pulls all nodes toward center
    fn apply_center_force(&mut self) {
        if self.center_strength == 0.0 {
            return;
        }

        // Calculate centroid
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;

        for i in 0..self.n_nodes {
            cx += self.positions[i * 2];
            cy += self.positions[i * 2 + 1];
        }

        cx /= self.n_nodes as f32;
        cy /= self.n_nodes as f32;

        // Calculate offset from target center
        let dx = (self.center_x - cx) * self.center_strength;
        let dy = (self.center_y - cy) * self.center_strength;

        // Apply offset to all nodes
        for i in 0..self.n_nodes {
            self.positions[i * 2] += dx;
            self.positions[i * 2 + 1] += dy;
        }
    }

    /// Apply many-body force using Barnes-Hut algorithm (O(n log n))
    /// Direct port of D3's forceManyBody
    fn apply_many_body_force(&mut self) {
        let strength = self.many_body_strength;
        let theta2 = self.many_body_theta * self.many_body_theta;
        let dist_min2 = self.many_body_distance_min * self.many_body_distance_min;
        let dist_max2 = self.many_body_distance_max * self.many_body_distance_max;
        let alpha = self.alpha;

        // Build per-node strengths array (all same strength for now)
        let strengths: Vec<f32> = vec![strength; self.n_nodes];

        // Build quadtree from current positions
        let mut tree = Quadtree::from_nodes(&self.positions, &strengths, self.n_nodes);

        // Accumulate values up the tree (visitAfter)
        tree.visit_after(|node, _x0, _y0, _x1, _y1| {
            accumulate(node, &strengths);
        });

        // For each node, traverse tree and apply forces
        for i in 0..self.n_nodes {
            let node_x = self.positions[i * 2];
            let node_y = self.positions[i * 2 + 1];
            let mut vx = 0.0f32;
            let mut vy = 0.0f32;

            // Barnes-Hut traversal
            tree.visit(|quad, x0, _y0, x1, _y1| {
                let quad_value = quad.get_value();
                if quad_value == 0.0 {
                    return true; // Skip empty nodes
                }

                let mut dx = quad.get_x() - node_x;
                let mut dy = quad.get_y() - node_y;
                let w = x1 - x0;  // Cell width
                let mut l = dx * dx + dy * dy;

                // Barnes-Hut approximation check: w²/θ² < l
                // If true, cell is far enough to use aggregate
                if w * w / theta2 < l {
                    if l < dist_max2 {
                        // Jiggle coincident points
                        if dx == 0.0 {
                            dx = Self::jiggle(i);
                            l += dx * dx;
                        }
                        if dy == 0.0 {
                            dy = Self::jiggle(i);
                            l += dy * dy;
                        }
                        // Clamp minimum distance
                        if l < dist_min2 {
                            l = (dist_min2 * l).sqrt();
                        }
                        // Apply aggregate force
                        vx += dx * quad_value * alpha / l;
                        vy += dy * quad_value * alpha / l;
                    }
                    return true; // Don't recurse - used aggregate
                }

                // Cell is too close - need more detail
                // For internal nodes: recurse into children (return false)
                // For leaf nodes: process directly then stop (return true)

                if !quad.is_leaf() {
                    // Internal node too close - must recurse
                    return false;
                }

                // It's a leaf node - process directly
                if let QuadNode::Leaf { index, x: qx, y: qy, next, .. } = quad {
                    // Process this leaf and any coincident points
                    if *index != i {
                        let mut dx = qx - node_x;
                        let mut dy = qy - node_y;
                        let mut l = dx * dx + dy * dy;

                        if l < dist_max2 {
                            if dx == 0.0 {
                                dx = Self::jiggle(i);
                                l += dx * dx;
                            }
                            if dy == 0.0 {
                                dy = Self::jiggle(i);
                                l += dy * dy;
                            }
                            if l < dist_min2 {
                                l = (dist_min2 * l).sqrt();
                            }
                            let force = strengths[*index] * alpha / l;
                            vx += dx * force;
                            vy += dy * force;
                        }
                    }

                    // Process linked list of coincident points
                    let mut current = next.as_ref();
                    while let Some(n) = current {
                        if let QuadNode::Leaf { index: ni, x: nx, y: ny, next: nn, .. } = n.as_ref() {
                            if *ni != i {
                                let mut dx = nx - node_x;
                                let mut dy = ny - node_y;
                                let mut l = dx * dx + dy * dy;

                                if l < dist_max2 {
                                    if dx == 0.0 {
                                        dx = Self::jiggle(i);
                                        l += dx * dx;
                                    }
                                    if dy == 0.0 {
                                        dy = Self::jiggle(i);
                                        l += dy * dy;
                                    }
                                    if l < dist_min2 {
                                        l = (dist_min2 * l).sqrt();
                                    }
                                    let force = strengths[*ni] * alpha / l;
                                    vx += dx * force;
                                    vy += dy * force;
                                }
                            }
                            current = nn.as_ref();
                        } else {
                            break;
                        }
                    }
                }

                true // Leaf processed - no children to recurse into
            });

            self.velocities[i * 2] += vx;
            self.velocities[i * 2 + 1] += vy;
        }
    }

    /// Small deterministic jitter to avoid division by zero
    fn jiggle(i: usize) -> f32 {
        ((i as f32 * 1.2345).sin() * 0.5 + 0.5) * 1e-6 - 0.5e-6
    }

    /// Apply link force (springs)
    /// D3-compatible: uses position + velocity for distance calculation
    fn apply_link_force(&mut self) {
        if self.n_links == 0 {
            return;
        }

        let distance = self.link_distance;
        let strength = self.link_strength;

        for i in 0..self.n_links {
            let source = self.links[i * 2] as usize;
            let target = self.links[i * 2 + 1] as usize;

            if source >= self.n_nodes || target >= self.n_nodes {
                continue;
            }

            // D3 uses position + velocity for the effective position
            // This allows multiple iterations to converge properly
            let sx = self.positions[source * 2] + self.velocities[source * 2];
            let sy = self.positions[source * 2 + 1] + self.velocities[source * 2 + 1];
            let tx = self.positions[target * 2] + self.velocities[target * 2];
            let ty = self.positions[target * 2 + 1] + self.velocities[target * 2 + 1];

            let mut dx = tx - sx;
            let mut dy = ty - sy;
            let mut d2 = dx * dx + dy * dy;

            // Add jitter if at same position (and recalculate d2)
            if d2 < 0.0001 {
                dx = (source as f32 - target as f32) * 0.001;
                dy = 0.001;
                d2 = dx * dx + dy * dy;
            }

            let d = f32::sqrt(d2).max(0.01);

            // Spring force: (distance - d) / d * strength * alpha
            let f = (distance - d) / d * strength * self.alpha * 0.5;

            // Apply to both endpoints
            let fx = dx * f;
            let fy = dy * f;

            self.velocities[source * 2] -= fx;
            self.velocities[source * 2 + 1] -= fy;
            self.velocities[target * 2] += fx;
            self.velocities[target * 2 + 1] += fy;
        }
    }

    /// Apply forceX - pulls nodes toward target x position
    /// D3-compatible: applies velocity based on distance from target
    fn apply_force_x(&mut self) {
        let strength = self.force_x_strength;
        let target = self.force_x_target;
        let alpha = self.alpha;

        for i in 0..self.n_nodes {
            let x = self.positions[i * 2];
            let dx = target - x;
            self.velocities[i * 2] += dx * strength * alpha;
        }
    }

    /// Apply forceY - pulls nodes toward target y position
    /// D3-compatible: applies velocity based on distance from target
    fn apply_force_y(&mut self) {
        let strength = self.force_y_strength;
        let target = self.force_y_target;
        let alpha = self.alpha;

        for i in 0..self.n_nodes {
            let y = self.positions[i * 2 + 1];
            let dy = target - y;
            self.velocities[i * 2 + 1] += dy * strength * alpha;
        }
    }

    /// Apply forceCollide - collision detection using quadtree
    /// Prevents nodes from overlapping by pushing them apart
    fn apply_collide_force(&mut self) {
        let radius = self.collide_radius;
        let strength = self.collide_strength;

        // Build quadtree from current positions (with zero strengths - we don't need mass for collide)
        let zero_strengths: Vec<f32> = vec![0.0; self.n_nodes];
        let tree = Quadtree::from_nodes(&self.positions, &zero_strengths, self.n_nodes);

        // For each node, find nearby nodes and apply separation
        for i in 0..self.n_nodes {
            let xi = self.positions[i * 2];
            let yi = self.positions[i * 2 + 1];
            let ri = radius;  // Could be per-node radius in future

            // Query quadtree for potentially colliding nodes
            tree.visit(|quad, x0, y0, x1, y1| {
                // Check if this quad could contain colliding nodes
                // The bounding box expanded by radius
                if x0 > xi + ri + ri || x1 < xi - ri - ri ||
                   y0 > yi + ri + ri || y1 < yi - ri - ri {
                    return true; // Skip this branch - too far away
                }

                // For leaf nodes, check actual collision
                if let QuadNode::Leaf { index: j, x: xj, y: yj, .. } = quad {
                    if *j != i {
                        let mut dx = xi - xj;
                        let mut dy = yi - yj;
                        let mut l = dx * dx + dy * dy;
                        let r = ri + radius;  // ri + rj if per-node radius

                        if l < r * r {
                            // Collision detected - push apart
                            if l == 0.0 {
                                // Coincident - jiggle
                                dx = Self::jiggle(i);
                                dy = Self::jiggle(i + 1);
                                l = dx * dx + dy * dy;
                            }
                            let l_sqrt = f32::sqrt(l);
                            let overlap = (r - l_sqrt) / l_sqrt * strength * 0.5;

                            // Push node i away from node j
                            // Note: We only update node i here; node j will be updated in its own iteration
                            self.velocities[i * 2] += dx * overlap;
                            self.velocities[i * 2 + 1] += dy * overlap;
                        }
                    }
                }

                false // Continue traversing
            });
        }
    }

    /// Integrate velocities into positions
    fn integrate(&mut self) {
        for i in 0..self.n_nodes {
            let idx = i * 2;
            let idy = i * 2 + 1;

            // Clamp velocities to prevent explosion (NaN protection)
            let max_velocity = 1000.0;
            if self.velocities[idx].is_nan() || self.velocities[idx].abs() > max_velocity {
                self.velocities[idx] = 0.0;
            }
            if self.velocities[idy].is_nan() || self.velocities[idy].abs() > max_velocity {
                self.velocities[idy] = 0.0;
            }

            // Apply velocity decay
            self.velocities[idx] *= self.velocity_decay;
            self.velocities[idy] *= self.velocity_decay;

            // Check for fixed position
            if self.fixed[idx].is_nan() {
                // Not fixed - update position
                self.positions[idx] += self.velocities[idx];
                self.positions[idy] += self.velocities[idy];
            } else {
                // Fixed - snap to fixed position, zero velocity
                self.positions[idx] = self.fixed[idx];
                self.positions[idy] = self.fixed[idy];
                self.velocities[idx] = 0.0;
                self.velocities[idy] = 0.0;
            }
        }
    }

    /// Get all positions as a flat array (for copying to JS)
    #[wasm_bindgen]
    pub fn get_positions(&self) -> Vec<f32> {
        self.positions.clone()
    }

    /// Get node count
    #[wasm_bindgen]
    pub fn node_count(&self) -> usize {
        self.n_nodes
    }

    /// Get link count
    #[wasm_bindgen]
    pub fn link_count(&self) -> usize {
        self.n_links
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_creation() {
        let sim = Simulation::new(100);
        assert_eq!(sim.node_count(), 100);
        assert!(sim.is_running());
    }

    #[test]
    fn test_tick_reduces_alpha() {
        let mut sim = Simulation::new(10);
        let alpha_before = sim.get_alpha();
        sim.tick();
        let alpha_after = sim.get_alpha();
        assert!(alpha_after < alpha_before);
    }

    #[test]
    fn test_fixed_nodes() {
        let mut sim = Simulation::new(10);
        sim.fix_node(0, 100.0, 200.0);

        // Run several ticks
        for _ in 0..10 {
            sim.tick();
        }

        // Node 0 should still be at fixed position
        let positions = sim.get_positions();
        assert_eq!(positions[0], 100.0);
        assert_eq!(positions[1], 200.0);
    }
}
