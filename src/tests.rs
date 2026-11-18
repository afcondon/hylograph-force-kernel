//! Unit tests: the quadtree, each force against a hand calculation or a
//! brute-force reference, and the simulation as a whole (determinism,
//! equivalence of the two APIs, fixed nodes, cooling).
//!
//! Agreement with d3-force itself is checked from Node, not here: see
//! `differential.mjs` beside the timing script in the knowledge base.

use super::*;
use crate::forces::Force;
use crate::quadtree::{is_leaf, leaf_point, Quadtree, NONE};

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// A small deterministic generator for test layouts (not the kernel's).
struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// A simulation with no forces at all and `points` as its node positions.
fn bare(points: &[(f32, f32)]) -> Simulation {
    let mut sim = Simulation::new(points.len());
    sim.enable_forces_all(false, false, false, false, false, false);
    for (i, &(x, y)) in points.iter().enumerate() {
        sim.set_position(i, x, y);
    }
    sim
}

/// `n` points scattered over a square of the given side.
fn cloud(n: usize, side: f32, seed: u64) -> Vec<(f32, f32)> {
    let mut rng = TestRng(seed);
    (0..n).map(|_| (rng.next() * side - side / 2.0, rng.next() * side - side / 2.0)).collect()
}

/// A ring with chords: every node linked to the next, every seventh to a far one.
fn ring_links(n: usize) -> Vec<u32> {
    let mut links = Vec::new();
    for i in 0..n {
        links.extend([i as u32, ((i + 1) % n) as u32]);
        if i % 7 == 0 && (i * 13 + 5) % n != i {
            links.extend([i as u32, ((i * 13 + 5) % n) as u32]);
        }
    }
    links
}

fn vel(sim: &Simulation, i: usize) -> (f32, f32) {
    (sim.state.velocities[i * 2], sim.state.velocities[i * 2 + 1])
}

fn pos(sim: &Simulation, i: usize) -> (f32, f32) {
    (sim.state.positions[i * 2], sim.state.positions[i * 2 + 1])
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Many-body without the quadtree: every pair, d3's distance rules.
fn many_body_brute(
    points: &[(f32, f32)],
    strengths: &[f32],
    distance_min: f32,
    distance_max: f32,
    alpha: f32,
) -> Vec<(f32, f32)> {
    let min2 = distance_min * distance_min;
    let max2 = distance_max * distance_max;
    points
        .iter()
        .enumerate()
        .map(|(i, &(xi, yi))| {
            let (mut vx, mut vy) = (0.0f64, 0.0f64);
            for (j, &(xj, yj)) in points.iter().enumerate() {
                if i == j {
                    continue;
                }
                let x = xj - xi;
                let y = yj - yi;
                let mut l = x * x + y * y;
                if l >= max2 {
                    continue;
                }
                if l < min2 {
                    l = (min2 * l).sqrt();
                }
                vx += (x * strengths[j] * alpha / l) as f64;
                vy += (y * strengths[j] * alpha / l) as f64;
            }
            (vx as f32, vy as f32)
        })
        .collect()
}

/// The largest velocity error against a reference, as a fraction of the
/// largest reference velocity.
fn worst_relative_error(sim: &Simulation, reference: &[(f32, f32)]) -> f32 {
    let scale = reference.iter().map(|&(x, y)| x.abs().max(y.abs())).fold(0.0f32, f32::max);
    let worst = reference
        .iter()
        .enumerate()
        .map(|(i, &(rx, ry))| {
            let (vx, vy) = vel(sim, i);
            (vx - rx).abs().max((vy - ry).abs())
        })
        .fold(0.0f32, f32::max);
    worst / scale
}

/// Every point index stored in the tree, by walking it.
fn points_in(tree: &Quadtree) -> Vec<usize> {
    let mut found = Vec::new();
    let mut stack = Vec::new();
    tree.visit(&mut stack, |q, _, _, _, _| {
        if is_leaf(q) {
            let mut p = leaf_point(q) as u32;
            while p != NONE {
                found.push(p as usize);
                p = tree.next[p as usize];
            }
        }
        false
    });
    found
}

// ----------------------------------------------------------------------
// Random stream
// ----------------------------------------------------------------------

#[test]
fn lcg_matches_d3s_default_random_source() {
    // The first three values of d3-force's lcg(), and the jiggles made from them.
    let mut rng = Lcg::new();
    assert_eq!(rng.next(), 0.23645552527159452);
    assert_eq!(rng.next(), 0.3692706737201661);
    assert_eq!(rng.next(), 0.5042420323006809);
    let mut rng = Lcg::new();
    assert_eq!(rng.jiggle(), -2.6354447472840547e-7f64 as f32);
    assert_eq!(rng.jiggle(), -1.307293262798339e-7f64 as f32);
    assert_eq!(rng.jiggle(), 4.242032300680876e-9f64 as f32);
}

// ----------------------------------------------------------------------
// Quadtree
// ----------------------------------------------------------------------

#[test]
fn quadtree_root_square_has_integer_origin_and_power_of_two_side() {
    let points = cloud(200, 730.0, 1);
    let mut tree = Quadtree::new();
    tree.build(points.len(), |i| points[i]);
    let side = tree.x1 - tree.x0;
    assert_eq!(side, tree.y1 - tree.y0);
    assert_eq!(side.log2().fract(), 0.0, "side {side} is not a power of two");
    assert_eq!(tree.x0.fract(), 0.0);
    assert_eq!(tree.y0.fract(), 0.0);
    for &(x, y) in &points {
        assert!(tree.x0 <= x && x < tree.x1 && tree.y0 <= y && y < tree.y1);
    }
}

#[test]
fn quadtree_follows_d3_cover_on_a_known_extent() {
    // d3: quadtree([[0.5, 0.5], [9.5, 3.5]]).extent() is [[0, 0], [16, 16]].
    let points = [(0.5f32, 0.5f32), (9.5, 3.5)];
    let mut tree = Quadtree::new();
    tree.build(2, |i| points[i]);
    assert_eq!((tree.x0, tree.y0, tree.x1, tree.y1), (0.0, 0.0, 16.0, 16.0));

    // d3: quadtree([[-3.2, 7.9], [-0.1, -5.5]]).extent() is [[-4, -6], [12, 10]].
    let points = [(-3.2f32, 7.9f32), (-0.1, -5.5)];
    let mut tree = Quadtree::new();
    tree.build(2, |i| points[i]);
    assert_eq!((tree.x0, tree.y0, tree.x1, tree.y1), (-4.0, -6.0, 12.0, 10.0));

    // d3: with [100.25, -77] added the extent is [[-4, -77], [124, 51]].
    let points = [(-3.2f32, 7.9f32), (-0.1, -5.5), (100.25, -77.0)];
    let mut tree = Quadtree::new();
    tree.build(3, |i| points[i]);
    assert_eq!((tree.x0, tree.y0, tree.x1, tree.y1), (-4.0, -77.0, 124.0, 51.0));
}

#[test]
fn quadtree_holds_every_point_exactly_once() {
    let mut points = cloud(500, 400.0, 2);
    points.push(points[3]); // a coincident pair
    points.push(points[3]); // and a triple
    let mut tree = Quadtree::new();
    tree.build(points.len(), |i| points[i]);
    let mut found = points_in(&tree);
    found.sort();
    assert_eq!(found, (0..points.len()).collect::<Vec<_>>());
}

#[test]
fn quadtree_chains_coincident_points_newest_first() {
    let points = [(1.0f32, 1.0f32), (5.0, 5.0), (1.0, 1.0), (1.0, 1.0)];
    let mut tree = Quadtree::new();
    tree.build(points.len(), |i| points[i]);
    assert_eq!(tree.next[3], 2);
    assert_eq!(tree.next[2], 0);
    assert_eq!(tree.next[0], NONE);
    assert_eq!(tree.next[1], NONE);
}

#[test]
fn quadtree_leaves_out_points_that_are_not_finite() {
    let points = [(1.0f32, 1.0f32), (f32::NAN, 2.0), (3.0, f32::INFINITY), (4.0, 4.0)];
    let mut tree = Quadtree::new();
    tree.build(points.len(), |i| points[i]);
    let mut found = points_in(&tree);
    found.sort();
    assert_eq!(found, vec![0, 3]);

    let mut empty = Quadtree::new();
    empty.build(1, |_| (f32::NAN, f32::NAN));
    assert_eq!(empty.root, NONE);
}

#[test]
fn quadtree_visits_quadrants_in_d3_order() {
    // One point per quadrant of [0, 2) x [0, 2), inserted out of order:
    // the visit order is top-left, top-right, bottom-left, bottom-right.
    let points = [(1.5f32, 1.5f32), (0.5, 0.5), (1.5, 0.5), (0.5, 1.5)];
    let mut tree = Quadtree::new();
    tree.build(points.len(), |i| points[i]);
    assert_eq!(points_in(&tree), vec![1, 2, 3, 0]);
}

#[test]
fn quadtree_separates_points_one_ulp_apart() {
    let a = 100.0f32;
    let b = f32::from_bits(a.to_bits() + 1);
    let points = [(a, a), (b, a), (a, b)];
    let mut tree = Quadtree::new();
    tree.build(points.len(), |i| points[i]);
    let mut found = points_in(&tree);
    found.sort();
    assert_eq!(found, vec![0, 1, 2]);
    assert!(tree.next.iter().all(|&next| next == NONE), "distinct points were chained");
}

// ----------------------------------------------------------------------
// Center
// ----------------------------------------------------------------------

#[test]
fn center_moves_the_mean_position_to_the_target() {
    let mut sim = bare(&[(0.0, 0.0), (10.0, 0.0), (20.0, 30.0)]);
    Force::center(100.0, -50.0, 1.0).apply(&mut sim.state, 1.0);
    // Mean was (10, 10): everything shifts by (+90, -60), shape unchanged.
    assert_eq!(pos(&sim, 0), (90.0, -60.0));
    assert_eq!(pos(&sim, 1), (100.0, -60.0));
    assert_eq!(pos(&sim, 2), (110.0, -30.0));
    assert_eq!(vel(&sim, 0), (0.0, 0.0), "center moves positions, not velocities");
}

#[test]
fn center_strength_scales_the_shift() {
    let mut sim = bare(&[(0.0, 0.0), (20.0, 20.0)]);
    Force::center(0.0, 0.0, 0.25).apply(&mut sim.state, 1.0);
    // Mean (10, 10), a quarter of the way to the origin: shift (-2.5, -2.5).
    assert_eq!(pos(&sim, 0), (-2.5, -2.5));
    assert_eq!(pos(&sim, 1), (17.5, 17.5));
}

// ----------------------------------------------------------------------
// X and Y
// ----------------------------------------------------------------------

#[test]
fn force_x_pulls_toward_the_target_in_proportion() {
    let mut sim = bare(&[(10.0, 7.0), (-30.0, 7.0)]);
    Force::x(50.0, 0.1).apply(&mut sim.state, 0.5);
    // (target - x) * strength * alpha
    assert_close(vel(&sim, 0).0, 40.0 * 0.1 * 0.5, 1e-6);
    assert_close(vel(&sim, 1).0, 80.0 * 0.1 * 0.5, 1e-6);
    assert_eq!(vel(&sim, 0).1, 0.0);
    assert_eq!(vel(&sim, 1).1, 0.0);
}

#[test]
fn force_y_pulls_toward_the_target_in_proportion() {
    let mut sim = bare(&[(10.0, 7.0), (10.0, -13.0)]);
    Force::y(-3.0, 0.2).apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 0).1, -10.0 * 0.2, 1e-6);
    assert_close(vel(&sim, 1).1, 10.0 * 0.2, 1e-6);
    assert_eq!(vel(&sim, 0).0, 0.0);
}

#[test]
fn force_x_takes_a_target_per_node_and_skips_nan_targets() {
    let mut sim = bare(&[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    let id = sim.add_force_x(0.0, 0.5);
    assert!(sim.set_force_targets(id, &[10.0, -20.0, f32::NAN]));
    sim.forces[0].apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 0).0, 5.0, 1e-6);
    assert_close(vel(&sim, 1).0, -10.0, 1e-6);
    assert_eq!(vel(&sim, 2).0, 0.0);
}

#[test]
fn force_y_takes_a_strength_per_node() {
    let mut sim = bare(&[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]);
    let id = sim.add_force_y(0.0, 0.1);
    assert!(sim.set_force_strengths(id, &[1.0, 0.5, 0.0]));
    sim.forces[0].apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 0).1, -10.0, 1e-6);
    assert_close(vel(&sim, 1).1, -5.0, 1e-6);
    assert_eq!(vel(&sim, 2).1, 0.0);
}

#[test]
fn force_x_mask_leaves_filtered_nodes_alone() {
    let mut sim = bare(&[(10.0, 0.0), (10.0, 0.0)]);
    let id = sim.add_force_x(0.0, 1.0);
    assert!(sim.set_force_mask(id, &[1, 0]));
    sim.forces[0].apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 0).0, -10.0, 1e-6);
    assert_eq!(vel(&sim, 1).0, 0.0);

    sim.clear_force_mask(id);
    sim.forces[0].apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 1).0, -10.0, 1e-6);
}

// ----------------------------------------------------------------------
// Radial
// ----------------------------------------------------------------------

#[test]
fn radial_pulls_toward_the_circle() {
    // Outside the circle on +x, inside it on +y.
    let mut sim = bare(&[(200.0, 0.0), (0.0, 50.0)]);
    Force::radial(100.0, 0.0, 0.0, 0.1).apply(&mut sim.state, 1.0);
    // k = (radius - r) * strength * alpha / r; v += d * k
    assert_close(vel(&sim, 0).0, 200.0 * ((100.0 - 200.0) * 0.1 / 200.0), 1e-5); // -10
    assert_close(vel(&sim, 1).1, 50.0 * ((100.0 - 50.0) * 0.1 / 50.0), 1e-5); // +5
    // The other axis sits on the centre line: d3 substitutes 1e-6 there.
    assert!(vel(&sim, 0).1.abs() < 1e-6);
    assert!(vel(&sim, 1).0.abs() < 1e-5);
}

#[test]
fn radial_respects_its_centre() {
    let mut sim = bare(&[(130.0, 40.0)]);
    Force::radial(10.0, 100.0, 0.0, 1.0).apply(&mut sim.state, 1.0);
    // Relative to the centre the node is at (30, 40), r = 50, k = -0.8.
    assert_close(vel(&sim, 0).0, 30.0 * -0.8, 1e-5);
    assert_close(vel(&sim, 0).1, 40.0 * -0.8, 1e-5);
}

#[test]
fn radial_at_the_centre_is_finite() {
    let mut sim = bare(&[(0.0, 0.0)]);
    Force::radial(100.0, 0.0, 0.0, 0.1).apply(&mut sim.state, 1.0);
    let (vx, vy) = vel(&sim, 0);
    // d = (1e-6, 1e-6), r = sqrt(2)e-6, k = 100 * 0.1 / r: v = 10 / sqrt(2) each.
    assert_close(vx, 7.0710678, 1e-4);
    assert_close(vy, 7.0710678, 1e-4);
}

#[test]
fn radial_takes_a_radius_per_node_and_a_mask() {
    let mut sim = bare(&[(50.0, 0.0), (50.0, 0.0), (50.0, 0.0), (50.0, 0.0)]);
    let id = sim.add_radial(0.0, 0.0, 0.0, 1.0);
    assert!(sim.set_force_radii(id, &[50.0, 100.0, f32::NAN, 100.0]));
    assert!(sim.set_force_mask(id, &[1, 1, 1, 0]));
    sim.forces[0].apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 0).0, 0.0, 1e-6); // already on its circle
    assert_close(vel(&sim, 1).0, 50.0, 1e-4); // (100 - 50) / 50 * 50
    assert_eq!(vel(&sim, 2), (0.0, 0.0)); // no radius: left alone
    assert_eq!(vel(&sim, 3), (0.0, 0.0)); // filtered out
}

// ----------------------------------------------------------------------
// Link
// ----------------------------------------------------------------------

#[test]
fn link_pushes_a_short_link_apart_evenly() {
    let mut sim = bare(&[(0.0, 0.0), (10.0, 0.0)]);
    sim.set_links(&[0, 1]);
    Force::link(30.0, 1.0, 1).apply(&mut sim.state, 1.0);
    // x = 10, l = 10: (l - 30) / l = -2, so x becomes -20, split 50/50.
    assert_close(vel(&sim, 0).0, -10.0, 1e-5);
    assert_close(vel(&sim, 1).0, 10.0, 1e-5);
    assert_eq!(vel(&sim, 0).1.abs() < 1e-5, true);
}

#[test]
fn link_pulls_a_long_link_together_scaled_by_strength_and_alpha() {
    let mut sim = bare(&[(0.0, 0.0), (0.0, 100.0)]);
    sim.set_links(&[0, 1]);
    Force::link(20.0, 0.5, 1).apply(&mut sim.state, 0.5);
    // y = 100: (100 - 20) / 100 * 0.5 * 0.5 = 0.2, so y becomes 20, split 50/50.
    assert_close(vel(&sim, 0).1, 10.0, 1e-5);
    assert_close(vel(&sim, 1).1, -10.0, 1e-5);
}

#[test]
fn link_bias_moves_the_less_connected_end_more() {
    // A hub (node 0) with three spokes. Each link: hub has 3 links, leaf 1.
    let mut sim = bare(&[(0.0, 0.0), (10.0, 0.0), (-10.0, 0.0), (0.0, 10.0)]);
    sim.set_links(&[0, 1, 0, 2, 0, 3]);
    assert_eq!(sim.state.link_bias, vec![0.75, 0.75, 0.75]);

    let mut one = bare(&[(0.0, 0.0), (10.0, 0.0), (-1000.0, 0.0), (0.0, 1000.0)]);
    one.set_links(&[0, 1, 0, 2, 0, 3]);
    // Only look at the first link's effect on node 1: x = 10 -> -20 after
    // the spring; the target takes bias 0.75 of it, the source 0.25.
    let before = vel(&one, 1).0;
    Force::link(30.0, 1.0, 1).apply(&mut one.state, 1.0);
    assert_close(vel(&one, 1).0 - before, 15.0, 1e-4);
}

#[test]
fn link_takes_a_distance_and_a_strength_per_link() {
    let mut sim = bare(&[(0.0, 0.0), (10.0, 0.0), (0.0, 500.0), (10.0, 500.0), (0.0, 900.0), (10.0, 900.0)]);
    sim.set_links(&[0, 1, 2, 3, 4, 5]);
    let id = sim.add_links(30.0, 1.0, 1);
    assert!(sim.set_force_distances(id, &[30.0, 10.0, 30.0]));
    assert!(sim.set_force_strengths(id, &[1.0, 1.0, 0.5]));
    sim.forces[0].apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 1).0, 10.0, 1e-5); // as in the uniform case
    assert_close(vel(&sim, 3).0, 0.0, 1e-6); // already at its rest length
    assert_close(vel(&sim, 5).0, 5.0, 1e-5); // half strength
}

#[test]
fn link_iterations_repeat_the_solve_against_updated_velocities() {
    let mut once = bare(&[(0.0, 0.0), (10.0, 0.0)]);
    once.set_links(&[0, 1]);
    Force::link(30.0, 0.5, 1).apply(&mut once.state, 1.0);
    // After one pass the ends are 10 + 2 * 5 = 20 apart (position + velocity).
    assert_close(vel(&once, 1).0, 5.0, 1e-5);

    let mut twice = bare(&[(0.0, 0.0), (10.0, 0.0)]);
    twice.set_links(&[0, 1]);
    Force::link(30.0, 0.5, 2).apply(&mut twice.state, 1.0);
    // The second pass sees 20, closes half of the remaining 10: 25 apart.
    assert_close(vel(&twice, 1).0, 7.5, 1e-5);
}

#[test]
fn link_between_coincident_nodes_is_finite_and_separates_them() {
    let mut sim = bare(&[(5.0, 5.0), (5.0, 5.0)]);
    sim.set_links(&[0, 1]);
    Force::link(30.0, 1.0, 1).apply(&mut sim.state, 1.0);
    let (ax, ay) = vel(&sim, 0);
    let (bx, by) = vel(&sim, 1);
    assert!(ax.is_finite() && ay.is_finite() && bx.is_finite() && by.is_finite());
    // The jiggled direction is arbitrary but the length is the rest length.
    let apart = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
    assert_close(apart, 30.0, 1e-3);
}

#[test]
fn link_naming_a_missing_node_is_skipped() {
    let mut sim = bare(&[(0.0, 0.0), (10.0, 0.0)]);
    sim.set_links(&[0, 7, 0, 1]);
    assert!(sim.state.link_bias[0].is_nan());
    assert_eq!(sim.state.link_bias[1], 0.5, "the bad link must not count toward bias");
    Force::link(30.0, 1.0, 1).apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 1).0, 10.0, 1e-5);
}

// ----------------------------------------------------------------------
// Many-body
// ----------------------------------------------------------------------

#[test]
fn many_body_repels_two_nodes_by_strength_over_distance() {
    let mut sim = bare(&[(0.0, 0.0), (10.0, 0.0)]);
    Force::many_body(-30.0, 0.9, 1.0, f64::INFINITY).apply(&mut sim.state, 1.0);
    // v += d * strength * alpha / d^2 = 10 * -30 / 100
    assert_close(vel(&sim, 0).0, -3.0, 1e-5);
    assert_close(vel(&sim, 1).0, 3.0, 1e-5);
    // The y offset is exactly zero, so d3 jiggles it: tiny, not zero.
    assert!(vel(&sim, 0).1.abs() < 1e-6);
}

#[test]
fn many_body_positive_strength_attracts() {
    let mut sim = bare(&[(0.0, 0.0), (0.0, 20.0)]);
    Force::many_body(40.0, 0.9, 1.0, f64::INFINITY).apply(&mut sim.state, 0.5);
    assert_close(vel(&sim, 0).1, 20.0 * 40.0 * 0.5 / 400.0, 1e-5);
    assert_close(vel(&sim, 1).1, -20.0 * 40.0 * 0.5 / 400.0, 1e-5);
}

#[test]
fn many_body_with_theta_zero_is_the_exact_pairwise_sum() {
    let points = cloud(300, 600.0, 3);
    let mut sim = bare(&points);
    Force::many_body(-30.0, 0.0, 1.0, f64::INFINITY).apply(&mut sim.state, 1.0);
    let reference = many_body_brute(&points, &vec![-30.0; points.len()], 1.0, f32::INFINITY, 1.0);
    let error = worst_relative_error(&sim, &reference);
    assert!(error < 1e-5, "worst relative error {error}");
}

#[test]
fn many_body_barnes_hut_stays_close_to_the_pairwise_sum() {
    let points = cloud(1000, 1000.0, 4);
    let mut sim = bare(&points);
    Force::many_body(-30.0, 0.9, 1.0, f64::INFINITY).apply(&mut sim.state, 1.0);
    let reference = many_body_brute(&points, &vec![-30.0; points.len()], 1.0, f32::INFINITY, 1.0);
    let error = worst_relative_error(&sim, &reference);
    assert!(error < 0.05, "worst relative error {error}");
}

#[test]
fn many_body_takes_a_strength_per_node() {
    let points = cloud(200, 500.0, 5);
    let strengths: Vec<f32> = (0..points.len()).map(|i| -10.0 - (i % 9) as f32 * 7.0).collect();
    let mut sim = bare(&points);
    let id = sim.add_many_body(-30.0, 0.0, 1.0, f64::INFINITY);
    assert!(sim.set_force_strengths(id, &strengths));
    sim.forces[0].apply(&mut sim.state, 1.0);
    let reference = many_body_brute(&points, &strengths, 1.0, f32::INFINITY, 1.0);
    let error = worst_relative_error(&sim, &reference);
    assert!(error < 1e-5, "worst relative error {error}");
}

#[test]
fn many_body_mask_silences_a_node_but_does_not_shield_it() {
    // Three in a row; the middle one is filtered out.
    let mut sim = bare(&[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]);
    let id = sim.add_many_body(-30.0, 0.0, 1.0, f64::INFINITY);
    assert!(sim.set_force_mask(id, &[1, 0, 1]));
    sim.forces[0].apply(&mut sim.state, 1.0);
    // The ends feel only each other (20 apart), not the middle.
    assert_close(vel(&sim, 0).0, 20.0 * -30.0 / 400.0, 1e-5);
    assert_close(vel(&sim, 2).0, -20.0 * -30.0 / 400.0, 1e-5);
    // The middle is still pushed by both ends, which cancel on this line.
    assert_close(vel(&sim, 1).0, 0.0, 1e-5);

    // Off centre, the filtered node is visibly pushed.
    let mut sim = bare(&[(0.0, 0.0), (5.0, 0.0), (20.0, 0.0)]);
    let id = sim.add_many_body(-30.0, 0.0, 1.0, f64::INFINITY);
    assert!(sim.set_force_mask(id, &[1, 0, 1]));
    sim.forces[0].apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 1).0, -5.0 * -30.0 / 25.0 + 15.0 * -30.0 / 225.0, 1e-5);
}

#[test]
fn many_body_ignores_nodes_beyond_distance_max() {
    let mut sim = bare(&[(0.0, 0.0), (10.0, 0.0), (500.0, 0.0)]);
    Force::many_body(-30.0, 0.0, 1.0, 100.0).apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 0).0, -3.0, 1e-5);
    assert_close(vel(&sim, 1).0, 3.0, 1e-5);
    assert_eq!(vel(&sim, 2), (0.0, 0.0));
}

#[test]
fn many_body_softens_the_force_inside_distance_min() {
    // 2 apart with distanceMin 10: l = 4 becomes sqrt(100 * 4) = 20.
    let mut sim = bare(&[(0.0, 0.0), (2.0, 0.0)]);
    Force::many_body(-30.0, 0.0, 10.0, f64::INFINITY).apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 0).0, 2.0 * -30.0 / 20.0, 1e-5);
    assert_close(vel(&sim, 1).0, -2.0 * -30.0 / 20.0, 1e-5);
}

#[test]
fn many_body_separates_coincident_nodes() {
    let mut sim = bare(&[(3.0, 3.0), (3.0, 3.0), (3.0, 3.0), (50.0, 3.0)]);
    Force::many_body(-30.0, 0.9, 1.0, f64::INFINITY).apply(&mut sim.state, 1.0);
    for i in 0..4 {
        let (vx, vy) = vel(&sim, i);
        assert!(vx.is_finite() && vy.is_finite(), "node {i} has velocity ({vx}, {vy})");
    }
    for i in 0..3 {
        let (vx, vy) = vel(&sim, i);
        assert!(vx.abs() + vy.abs() > 1.0, "coincident node {i} was not pushed");
    }
    assert_ne!(vel(&sim, 0), vel(&sim, 1));
    assert_ne!(vel(&sim, 1), vel(&sim, 2));
}

// ----------------------------------------------------------------------
// Collide
// ----------------------------------------------------------------------

#[test]
fn collide_pushes_equal_circles_apart_by_half_the_overlap_each() {
    let mut sim = bare(&[(0.0, 0.0), (10.0, 0.0)]);
    Force::collide(10.0, 1.0, 1).apply(&mut sim.state, 1.0);
    // Radii sum 20, distance 10: overlap 10, shared equally.
    assert_close(vel(&sim, 0).0, -5.0, 1e-5);
    assert_close(vel(&sim, 1).0, 5.0, 1e-5);
    // The y offset is exactly zero, so d3 jiggles it: tiny, not zero.
    assert!(vel(&sim, 0).1.abs() < 1e-5);
}

#[test]
fn collide_strength_scales_the_correction() {
    let mut sim = bare(&[(0.0, 0.0), (0.0, 10.0)]);
    Force::collide(10.0, 0.5, 1).apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 0).1, -2.5, 1e-5);
    assert_close(vel(&sim, 1).1, 2.5, 1e-5);
}

#[test]
fn collide_leaves_circles_that_do_not_overlap() {
    let mut sim = bare(&[(0.0, 0.0), (20.5, 0.0), (0.0, 30.0)]);
    Force::collide(10.0, 1.0, 1).apply(&mut sim.state, 1.0);
    for i in 0..3 {
        assert_eq!(vel(&sim, i), (0.0, 0.0));
    }
}

#[test]
fn collide_with_radii_per_node_moves_the_smaller_circle_more() {
    let mut sim = bare(&[(0.0, 0.0), (10.0, 0.0)]);
    let id = sim.add_collide(1.0, 1.0, 1);
    assert!(sim.set_force_radii(id, &[10.0, 30.0]));
    sim.forces[0].apply(&mut sim.state, 1.0);
    // Radii sum 40, distance 10: overlap 30. Shares by squared radius:
    // the small one takes 900 / 1000, the large one 100 / 1000.
    assert_close(vel(&sim, 0).0, -27.0, 1e-4);
    assert_close(vel(&sim, 1).0, 3.0, 1e-4);
}

#[test]
fn collide_uses_position_plus_velocity() {
    // Apart now, but the first is moving into the second.
    let mut sim = bare(&[(0.0, 0.0), (30.0, 0.0)]);
    sim.set_velocity(0, 20.0, 0.0);
    Force::collide(10.0, 1.0, 1).apply(&mut sim.state, 1.0);
    // Anticipated distance 10: overlap 10, shared equally.
    assert_close(vel(&sim, 0).0, 15.0, 1e-5);
    assert_close(vel(&sim, 1).0, 5.0, 1e-5);
}

#[test]
fn collide_quadtree_search_finds_every_overlapping_pair() {
    // Overlapping pairs scattered over a wide area, far enough from each
    // other that each node meets at most one other. (In a dense layout d3's
    // sweep is order-dependent: it prunes with the tree as built, while the
    // nodes it has already pushed have moved. Agreement there is checked
    // against d3 itself, not against a pairwise reference.)
    let mut rng = TestRng(6);
    let mut points = Vec::new();
    let mut radii = Vec::new();
    for cell in 0..200 {
        let cx = (cell % 20) as f32 * 120.0 - 1200.0 + rng.next() * 30.0;
        let cy = (cell / 20) as f32 * 120.0 - 600.0 + rng.next() * 30.0;
        let ra = 4.0 + rng.next() * 10.0;
        let rb = 4.0 + rng.next() * 10.0;
        // Odd cells overlap, even cells sit just clear of each other.
        let gap = if cell % 2 == 1 { 0.2 + rng.next() * 0.7 } else { 1.05 + rng.next() * 0.5 };
        let angle = rng.next() * std::f32::consts::TAU;
        points.push((cx, cy));
        radii.push(ra);
        points.push((cx + angle.cos() * (ra + rb) * gap, cy + angle.sin() * (ra + rb) * gap));
        radii.push(rb);
    }
    // Shuffle so partners are not neighbours in index order.
    for i in (1..points.len()).rev() {
        let j = (rng.next() * (i + 1) as f32) as usize % (i + 1);
        points.swap(i, j);
        radii.swap(i, j);
    }

    let mut sim = bare(&points);
    let id = sim.add_collide(1.0, 0.7, 1);
    assert!(sim.set_force_radii(id, &radii));
    sim.forces[0].apply(&mut sim.state, 1.0);

    let n = points.len();
    let mut v = vec![(0.0f32, 0.0f32); n];
    let mut overlapping = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = radii[i] + radii[j];
            let mut x = points[i].0 - points[j].0;
            let mut y = points[i].1 - points[j].1;
            let l = x * x + y * y;
            if l < r * r {
                overlapping += 1;
                let l = l.sqrt();
                let l = (r - l) / l * 0.7;
                x *= l;
                y *= l;
                let share = radii[j] * radii[j] / (radii[i] * radii[i] + radii[j] * radii[j]);
                v[i].0 += x * share;
                v[i].1 += y * share;
                v[j].0 -= x * (1.0 - share);
                v[j].1 -= y * (1.0 - share);
            }
        }
    }
    assert_eq!(overlapping, 100);
    let error = worst_relative_error(&sim, &v);
    assert!(error < 1e-5, "worst relative error {error}");
    let untouched = (0..n).filter(|&i| vel(&sim, i) == (0.0, 0.0)).count();
    assert_eq!(untouched, 200, "nodes that overlap nothing must not be moved");
}

#[test]
fn collide_iterations_leave_less_overlap() {
    let overlap_after = |iterations: u32| {
        let points = cloud(150, 60.0, 7);
        let mut sim = bare(&points);
        sim.set_velocity_decay(1.0);
        sim.add_collide(4.0, 1.0, iterations);
        sim.tick();
        let mut total = 0.0f32;
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let (ax, ay) = pos(&sim, i);
                let (bx, by) = pos(&sim, j);
                let d = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
                total += (8.0 - d).max(0.0);
            }
        }
        total
    };
    let one = overlap_after(1);
    let four = overlap_after(4);
    assert!(four < one * 0.8, "overlap {one} after 1 iteration, {four} after 4");
}

#[test]
fn collide_separates_coincident_nodes() {
    let mut sim = bare(&[(1.0, 1.0), (1.0, 1.0)]);
    Force::collide(5.0, 1.0, 1).apply(&mut sim.state, 1.0);
    let (ax, ay) = vel(&sim, 0);
    let (bx, by) = vel(&sim, 1);
    assert!(ax.is_finite() && ay.is_finite() && bx.is_finite() && by.is_finite());
    // The jiggled direction is arbitrary; the separation is the radius sum.
    let apart = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
    assert_close(apart, 10.0, 1e-3);
}

#[test]
fn collide_moves_every_one_of_several_coincident_nodes() {
    // d3's collide looks at the head of a chain of coincident points and
    // not at the rest of it. With three nodes on one spot, 0 and 1 are each
    // tested against 2 (the head) but never against each other.
    let mut sim = bare(&[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (500.0, 500.0)]);
    sim.set_velocity_decay(1.0);
    sim.add_collide(5.0, 1.0, 1);
    sim.tick();
    for i in 0..3 {
        assert!(pos(&sim, i).0.is_finite() && pos(&sim, i).1.is_finite());
        assert_ne!(pos(&sim, i), (1.0, 1.0), "node {i} was not moved");
    }
    assert_eq!(pos(&sim, 3), (500.0, 500.0));
}

// ----------------------------------------------------------------------
// Simulation
// ----------------------------------------------------------------------

#[test]
fn test_simulation_creation() {
    let sim = Simulation::new(100);
    assert_eq!(sim.node_count(), 100);
    assert!(sim.is_running());
}

#[test]
fn new_nodes_sit_on_d3s_phyllotaxis_spiral() {
    let sim = Simulation::new(50);
    for i in 0..50 {
        let r = 10.0 * (0.5 + i as f64).sqrt();
        let a = i as f64 * std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
        assert_eq!(pos(&sim, i), ((r * a.cos()) as f32, (r * a.sin()) as f32));
    }
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
fn alpha_cools_toward_its_target_and_tick_n_stops_at_alpha_min() {
    let mut sim = Simulation::new(3);
    sim.set_alpha_decay(0.5);
    sim.tick();
    assert_close(sim.get_alpha(), 0.5, 1e-6);
    sim.tick();
    assert_close(sim.get_alpha(), 0.25, 1e-6);

    sim.set_alpha_min(0.1);
    sim.tick_n(100);
    assert!(!sim.is_running());
    assert_close(sim.get_alpha(), 0.0625, 1e-6); // two more ticks, then it stopped

    // tick_n does nothing once alpha is below alpha_min; tick always ticks.
    sim.set_alpha_target(0.8);
    sim.tick_n(100);
    assert_close(sim.get_alpha(), 0.0625, 1e-6);
    for _ in 0..100 {
        sim.tick();
    }
    assert_close(sim.get_alpha(), 0.8, 1e-5);
    assert!(sim.is_running());
}

#[test]
fn integration_decays_velocity_then_moves() {
    let mut sim = bare(&[(1.0, 2.0)]);
    sim.set_velocity(0, 10.0, -20.0);
    sim.set_velocity_decay(0.6);
    sim.tick();
    assert_close(vel(&sim, 0).0, 6.0, 1e-5);
    assert_close(vel(&sim, 0).1, -12.0, 1e-5);
    assert_close(pos(&sim, 0).0, 7.0, 1e-5);
    assert_close(pos(&sim, 0).1, -10.0, 1e-5);
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

    sim.unfix_node(0);
    sim.reheat();
    sim.tick();
    assert_ne!(sim.get_positions()[0], 100.0);
}

#[test]
fn a_node_can_be_fixed_on_one_axis_only() {
    let mut sim = bare(&[(0.0, 0.0)]);
    sim.add_force_x(100.0, 0.5);
    sim.add_force_y(100.0, 0.5);
    sim.fix_node(0, 7.0, f32::NAN);
    for _ in 0..5 {
        sim.tick();
    }
    assert_eq!(pos(&sim, 0).0, 7.0);
    assert_eq!(vel(&sim, 0).0, 0.0);
    assert!(pos(&sim, 0).1 > 10.0, "the free axis should have moved");

    sim.set_fixed(&[f32::NAN, -3.0]);
    sim.tick();
    assert_eq!(pos(&sim, 0).1, -3.0);
    assert_ne!(pos(&sim, 0).0, 7.0);
}

#[test]
fn set_positions_puts_nan_nodes_on_the_spiral() {
    let mut sim = bare(&[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    let spiral = pos(&Simulation::new(3), 1);
    sim.set_positions(&[5.0, 6.0, f32::NAN, 1.0, -7.0, 8.0]);
    assert_eq!(pos(&sim, 0), (5.0, 6.0));
    assert_eq!(pos(&sim, 1), spiral);
    assert_eq!(pos(&sim, 2), (-7.0, 8.0));
}

#[test]
fn velocities_round_trip() {
    let mut sim = bare(&[(0.0, 0.0), (1.0, 1.0)]);
    sim.set_velocities(&[1.0, 2.0, f32::NAN, 4.0]);
    assert_eq!(sim.get_velocities(), vec![1.0, 2.0, 0.0, 4.0]);
}

/// Every force type at once, per-node arrays, a mask, fixed and coincident
/// nodes: the busiest configuration the kernel supports.
fn everything(n: usize) -> Simulation {
    let mut points = cloud(n, 400.0, 8);
    points[5] = points[4]; // coincident pair
    points[6] = points[4]; // coincident triple
    let mut sim = bare(&points);
    sim.set_links(&ring_links(n));
    sim.fix_node(0, 10.0, 10.0);
    sim.fix_node(1, f32::NAN, -25.0);

    let charge = sim.add_many_body(-30.0, 0.9, 1.0, f64::INFINITY);
    sim.set_force_mask(charge, &(0..n).map(|i| (i % 4 != 0) as u8).collect::<Vec<_>>());
    let charge2 = sim.add_many_body(-5.0, 0.5, 1.0, 200.0);
    sim.set_force_strengths(charge2, &(0..n).map(|i| -((i % 6) as f32)).collect::<Vec<_>>());
    let springs = sim.add_links(30.0, 0.7, 2);
    sim.set_force_distances(springs, &(0..sim.link_count()).map(|i| 20.0 + (i % 3) as f32 * 10.0).collect::<Vec<_>>());
    sim.add_center(0.0, 0.0, 1.0);
    let fx = sim.add_force_x(0.0, 0.05);
    sim.set_force_targets(fx, &(0..n).map(|i| (i % 5) as f32 * 40.0 - 80.0).collect::<Vec<_>>());
    sim.add_force_y(0.0, 0.05);
    let bump = sim.add_collide(5.0, 0.8, 2);
    sim.set_force_radii(bump, &(0..n).map(|i| 3.0 + (i % 4) as f32).collect::<Vec<_>>());
    let ring = sim.add_radial(150.0, 0.0, 0.0, 0.02);
    sim.set_force_mask(ring, &(0..n).map(|i| (i % 2) as u8).collect::<Vec<_>>());
    sim
}

#[test]
fn the_same_inputs_give_the_same_positions_bit_for_bit() {
    let mut a = everything(300);
    let mut b = everything(300);
    for tick in 1..=300 {
        a.tick();
        b.tick();
        if tick == 1 || tick % 50 == 0 {
            assert_eq!(bits(&a.get_positions()), bits(&b.get_positions()), "diverged by tick {tick}");
            assert_eq!(bits(&a.get_velocities()), bits(&b.get_velocities()), "diverged by tick {tick}");
        }
    }
    assert!(a.get_positions().iter().all(|v| v.is_finite()));
    // And the run did something: nodes moved and the coincident ones parted.
    let start = everything(300).get_positions();
    let moved = a.get_positions().iter().zip(&start).filter(|(p, q)| (*p - *q).abs() > 1.0).count();
    assert!(moved > 300, "only {moved} coordinates moved");
    assert_ne!(pos(&a, 4), pos(&a, 5));
    assert_ne!(pos(&a, 5), pos(&a, 6));
}

#[test]
fn ticking_in_batches_gives_the_same_positions_as_one_at_a_time() {
    let mut a = everything(120);
    let mut b = everything(120);
    for _ in 0..60 {
        a.tick();
    }
    b.tick_n(25);
    b.tick_n(35);
    assert_eq!(bits(&a.get_positions()), bits(&b.get_positions()));
}

#[test]
fn reset_random_rewinds_the_jiggle_stream() {
    let run = |sim: &mut Simulation| {
        sim.set_positions(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 9.0]);
        sim.set_velocities(&[0.0; 8]);
        sim.set_alpha(1.0);
        sim.tick();
        bits(&sim.get_positions())
    };
    let mut sim = Simulation::new(4);
    sim.enable_forces_all(true, false, false, false, false, false);
    let first = run(&mut sim);
    let second = run(&mut sim);
    assert_ne!(first, second, "the stream should have advanced");
    sim.reset_random();
    assert_eq!(run(&mut sim), first);
}

#[test]
fn the_slots_and_the_force_list_are_the_same_forces() {
    let points = cloud(200, 300.0, 9);
    let links = ring_links(200);

    let mut slots = bare(&points);
    slots.set_links(&links);
    slots.configure_center(5.0, -5.0, 1.0);
    slots.configure_many_body(-25.0, 0.8, 2.0, 500.0);
    slots.configure_links(40.0, 0.6, 2);
    slots.configure_force_x(10.0, 0.05);
    slots.configure_force_y(-10.0, 0.07);
    slots.configure_collide(6.0, 0.9, 2);
    slots.enable_forces_all(true, true, true, true, true, true);

    let mut list = bare(&points);
    list.set_links(&links);
    list.add_center(5.0, -5.0, 1.0);
    list.add_many_body(-25.0, 0.8, 2.0, 500.0);
    list.add_links(40.0, 0.6, 2);
    list.add_force_x(10.0, 0.05);
    list.add_force_y(-10.0, 0.07);
    list.add_collide(6.0, 0.9, 2);
    assert_eq!(list.force_count(), 6);

    for _ in 0..100 {
        slots.tick();
        list.tick();
    }
    assert_eq!(bits(&slots.get_positions()), bits(&list.get_positions()));
}

#[test]
fn enable_force_leaves_the_other_three_slots_as_they_were() {
    let mut sim = Simulation::new(4);
    sim.enable_forces_all(false, false, false, true, true, true);
    sim.enable_force(true, false, true);
    assert_eq!(sim.slot_enabled, [true, true, false, true, true, true]);
}

#[test]
fn two_instances_of_one_force_type_both_act() {
    // Two many-body forces over complementary halves of the nodes add up to
    // one many-body force over all of them.
    let points = cloud(150, 400.0, 10);
    let n = points.len();
    let mut split = bare(&points);
    let a = split.add_many_body(-30.0, 0.0, 1.0, f64::INFINITY);
    let b = split.add_many_body(-30.0, 0.0, 1.0, f64::INFINITY);
    assert!(split.set_force_mask(a, &(0..n).map(|i| (i % 2) as u8).collect::<Vec<_>>()));
    assert!(split.set_force_mask(b, &(0..n).map(|i| ((i + 1) % 2) as u8).collect::<Vec<_>>()));
    split.tick();

    let mut whole = bare(&points);
    whole.add_many_body(-30.0, 0.0, 1.0, f64::INFINITY);
    whole.tick();

    let reference: Vec<(f32, f32)> = (0..n).map(|i| vel(&whole, i)).collect();
    let error = worst_relative_error(&split, &reference);
    assert!(error < 1e-5, "worst relative error {error}");

    // And two different x forces on different nodes pull to different places.
    let mut sim = bare(&[(0.0, 0.0), (0.0, 0.0)]);
    let left = sim.add_force_x(-100.0, 1.0);
    let right = sim.add_force_x(100.0, 1.0);
    sim.set_force_mask(left, &[1, 0]);
    sim.set_force_mask(right, &[0, 1]);
    sim.set_velocity_decay(1.0);
    sim.tick();
    assert_close(pos(&sim, 0).0, -100.0, 1e-4);
    assert_close(pos(&sim, 1).0, 100.0, 1e-4);
}

#[test]
fn forces_apply_in_the_order_they_were_added() {
    // Center moves positions, x reads them: the order is observable.
    let run = |center_first: bool| {
        let mut sim = bare(&[(10.0, 0.0), (30.0, 0.0)]);
        if center_first {
            sim.add_center(0.0, 0.0, 1.0);
            sim.add_force_x(0.0, 1.0);
        } else {
            sim.add_force_x(0.0, 1.0);
            sim.add_center(0.0, 0.0, 1.0);
        }
        sim.set_velocity_decay(1.0);
        sim.tick();
        pos(&sim, 0).0
    };
    // Center first: positions become -10, +10, then x pulls each to 0.
    assert_close(run(true), 0.0, 1e-5);
    // X first: velocity -10 from x = 10, then center shifts to -10: ends at -20.
    assert_close(run(false), -20.0, 1e-5);
}

#[test]
fn clear_forces_empties_the_list() {
    let mut sim = bare(&[(0.0, 0.0), (10.0, 0.0)]);
    sim.add_many_body(-30.0, 0.9, 1.0, f64::INFINITY);
    sim.add_force_x(50.0, 1.0);
    sim.clear_forces();
    assert_eq!(sim.force_count(), 0);
    sim.tick();
    assert_eq!(sim.get_positions(), vec![0.0, 0.0, 10.0, 0.0]);
    // Ids start again from 0.
    assert_eq!(sim.add_center(0.0, 0.0, 1.0), 0);
}

#[test]
fn per_element_setters_refuse_forces_they_do_not_apply_to() {
    let mut sim = bare(&[(0.0, 0.0), (10.0, 0.0)]);
    let charge = sim.add_many_body(-30.0, 0.9, 1.0, f64::INFINITY);
    let springs = sim.add_links(30.0, 1.0, 1);
    let center = sim.add_center(0.0, 0.0, 1.0);
    let fx = sim.add_force_x(0.0, 0.1);
    let bump = sim.add_collide(5.0, 1.0, 1);
    let ring = sim.add_radial(50.0, 0.0, 0.0, 0.1);
    let two = [1.0f32, 1.0];

    assert!(!sim.set_force_strengths(center, &two));
    assert!(!sim.set_force_strengths(bump, &two));
    assert!(sim.set_force_strengths(charge, &two));
    assert!(sim.set_force_strengths(springs, &two));

    assert!(!sim.set_force_radii(charge, &two));
    assert!(sim.set_force_radii(bump, &two));
    assert!(sim.set_force_radii(ring, &two));

    assert!(!sim.set_force_targets(ring, &two));
    assert!(sim.set_force_targets(fx, &two));

    assert!(!sim.set_force_distances(fx, &two));
    assert!(sim.set_force_distances(springs, &two));

    assert!(!sim.set_force_mask(springs, &[1, 1]));
    assert!(!sim.set_force_mask(center, &[1, 1]));
    assert!(!sim.set_force_mask(bump, &[1, 1]));
    assert!(sim.set_force_mask(charge, &[1, 1]));

    assert!(!sim.set_force_strengths(99, &two));
    assert!(!sim.set_force_mask(99, &[1, 1]));
}

#[test]
fn arrays_shorter_than_the_node_count_leave_the_rest_alone() {
    let mut sim = everything(60);
    sim.set_node_count(90); // every per-node array is now 30 short
    sim.set_links(&ring_links(90)); // and the per-link distances are short too
    for _ in 0..20 {
        sim.tick();
    }
    assert!(sim.get_positions().iter().all(|v| v.is_finite()));

    let mut sim = bare(&[(10.0, 0.0), (10.0, 0.0)]);
    let fx = sim.add_force_x(0.0, 1.0);
    sim.set_force_strengths(fx, &[1.0]);
    sim.forces[0].apply(&mut sim.state, 1.0);
    assert_close(vel(&sim, 0).0, -10.0, 1e-6);
    assert_eq!(vel(&sim, 1).0, 0.0);
}

#[test]
fn an_empty_simulation_ticks() {
    let mut sim = Simulation::new(0);
    sim.enable_forces_all(true, true, true, true, true, true);
    sim.add_radial(10.0, 0.0, 0.0, 0.1);
    sim.tick();
    assert_eq!(sim.get_positions().len(), 0);
}
