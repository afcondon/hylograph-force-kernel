//! Tests for what d3-force does not have: towards, rectangle collision,
//! containment, group cohesion, scoped many-body, the directional link
//! constraint, snapshot/restore and the convergence measures.
//!
//! There is no d3 to compare with, so each force is held to the plain
//! reference implementation in `reference.rs`: exactly (bit for bit) where
//! the rule fixes the order of every floating-point operation, and within a
//! stated bound where it does not (scoped many-body, whose quadtree visits
//! pairs in a different order from a loop over pairs). Hand-worked cases
//! pin the rules themselves.
//!
//! Written in `Real`, so the same tests run at f32 (`cargo test`) and at
//! f64 (`cargo test --features f64`).

use super::*;
use crate::forces::NONE_ID;
use crate::reference::{self, Pairs, Point, Shape};

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> Real {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as Real) / (1u64 << 24) as Real
    }
}

fn bare(points: &[Point]) -> Simulation {
    let mut sim = Simulation::new(points.len());
    sim.enable_forces_all(false, false, false, false, false, false);
    for (i, &(x, y)) in points.iter().enumerate() {
        sim.set_position(i, x, y);
    }
    sim
}

fn cloud(n: usize, side: Real, seed: u64) -> Vec<Point> {
    let mut rng = TestRng(seed);
    (0..n).map(|_| (rng.next() * side - side / 2.0, rng.next() * side - side / 2.0)).collect()
}

fn give_velocities(sim: &mut Simulation, scale: Real, seed: u64) -> Vec<Point> {
    let v = cloud(sim.node_count(), scale, seed);
    for (i, &(vx, vy)) in v.iter().enumerate() {
        sim.set_velocity(i, vx, vy);
    }
    v
}

fn positions(sim: &Simulation) -> Vec<Point> {
    (0..sim.node_count()).map(|i| (sim.state.positions[i * 2], sim.state.positions[i * 2 + 1])).collect()
}

fn velocities(sim: &Simulation) -> Vec<Point> {
    (0..sim.node_count()).map(|i| (sim.state.velocities[i * 2], sim.state.velocities[i * 2 + 1])).collect()
}

/// Apply force `id` of the list once, without integrating.
fn apply(sim: &mut Simulation, id: u32, alpha: Real) {
    sim.forces[id as usize].apply(&mut sim.state, alpha);
}

fn bits(points: &[Point]) -> Vec<(u64, u64)> {
    points.iter().map(|&(x, y)| ((x as f64).to_bits(), (y as f64).to_bits())).collect()
}

/// Identical, bit for bit.
fn assert_exact(actual: &[Point], expected: &[Point]) {
    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        assert!(
            bits(&actual[i..=i]) == bits(&expected[i..=i]),
            "node {i}: got {:?}, reference {:?}",
            actual[i],
            expected[i]
        );
    }
}

fn assert_close(actual: Real, expected: Real, tolerance: Real) {
    assert!((actual - expected).abs() <= tolerance, "expected {expected}, got {actual} (tolerance {tolerance})");
}

/// The largest difference between two velocity fields, as a fraction of
/// the largest reference component.
fn worst_relative(actual: &[Point], expected: &[Point]) -> Real {
    let scale = expected.iter().map(|p| p.0.abs().max(p.1.abs())).fold(0.0, Real::max);
    let worst = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a.0 - e.0).abs().max((a.1 - e.1).abs()))
        .fold(0.0, Real::max);
    worst / scale
}

fn group_ids(groups: &[Option<u32>]) -> Vec<u32> {
    groups.iter().map(|g| g.unwrap_or(NONE_ID)).collect()
}

// ----------------------------------------------------------------------
// 1. Towards
// ----------------------------------------------------------------------

#[test]
fn towards_pulls_each_node_to_its_own_point() {
    let mut sim = bare(&[(0.0, 0.0), (10.0, 10.0), (5.0, 5.0)]);
    let id = sim.add_towards(0.5);
    assert!(sim.set_force_targets_xy(id, &[100.0, -40.0, 10.0, 30.0, Real::NAN, 0.0]));
    apply(&mut sim, id, 0.5);
    // (target - position) * strength * alpha
    assert_exact(&velocities(&sim), &[(25.0, -10.0), (0.0, 5.0), (0.0, 0.0)]);
}

#[test]
fn towards_matches_its_reference_exactly() {
    let points = cloud(300, 500.0, 11);
    let mut sim = bare(&points);
    let v = give_velocities(&mut sim, 20.0, 12);
    let targets: Vec<Option<Point>> = cloud(300, 800.0, 13).into_iter().enumerate().map(|(i, t)| (i % 9 != 4).then_some(t)).collect();
    let strengths: Vec<Real> = (0..300).map(|i| 0.01 + (i % 5) as Real * 0.07).collect();
    let id = sim.add_towards(0.1);
    let flat: Vec<Real> = targets.iter().flat_map(|t| t.map_or([Real::NAN, Real::NAN], |(x, y)| [x, y])).collect();
    assert!(sim.set_force_targets_xy(id, &flat));
    assert!(sim.set_force_strengths(id, &strengths));
    apply(&mut sim, id, 0.37);
    assert_exact(&velocities(&sim), &reference::towards(&points, &v, &targets, &strengths, 0.37));
}

#[test]
fn towards_is_force_x_and_force_y_with_targets_per_node() {
    // The two forces it replaces are checked against d3 itself.
    let points = cloud(200, 400.0, 14);
    let targets = cloud(200, 400.0, 15);
    let mut one = bare(&points);
    let id = one.add_towards(0.2);
    one.set_force_targets_xy(id, &targets.iter().flat_map(|&(x, y)| [x, y]).collect::<Vec<_>>());
    let mut two = bare(&points);
    let fx = two.add_force_x(0.0, 0.2);
    let fy = two.add_force_y(0.0, 0.2);
    two.set_force_targets(fx, &targets.iter().map(|t| t.0).collect::<Vec<_>>());
    two.set_force_targets(fy, &targets.iter().map(|t| t.1).collect::<Vec<_>>());
    for _ in 0..50 {
        one.tick();
        two.tick();
    }
    assert_exact(&positions(&one), &positions(&two));
}

#[test]
fn towards_targets_are_replaced_in_place_and_morph_the_layout() {
    let n = 100;
    let layout_a = cloud(n, 300.0, 16);
    let layout_b = cloud(n, 300.0, 17);
    let flat = |layout: &[Point]| layout.iter().flat_map(|&(x, y)| [x, y]).collect::<Vec<Real>>();
    let mut sim = bare(&cloud(n, 10.0, 18));
    sim.set_alpha_decay(0.0);
    let id = sim.add_towards(0.3);
    sim.set_force_targets_xy(id, &flat(&layout_a));
    let storage = |sim: &Simulation| match &sim.forces[id as usize].kind {
        Kind::Towards { targets } => targets.as_ptr(),
        _ => unreachable!(),
    };
    let before = storage(&sim);
    for _ in 0..200 {
        sim.tick();
    }
    for (p, t) in positions(&sim).iter().zip(&layout_a) {
        assert_close(p.0, t.0, 1e-2);
        assert_close(p.1, t.1, 1e-2);
    }
    // Replace the targets every tick, blending toward the second layout.
    for step in 1..=100 {
        let k = step as Real / 100.0;
        let blend: Vec<Real> = layout_a.iter().zip(&layout_b).flat_map(|(a, b)| [a.0 + (b.0 - a.0) * k, a.1 + (b.1 - a.1) * k]).collect();
        assert!(sim.set_force_targets_xy(id, &blend));
        sim.tick();
    }
    assert_eq!(storage(&sim), before, "same-length targets must be overwritten, not reallocated");
    for _ in 0..200 {
        sim.tick();
    }
    for (p, t) in positions(&sim).iter().zip(&layout_b) {
        assert_close(p.0, t.0, 1e-2);
        assert_close(p.1, t.1, 1e-2);
    }
}

#[test]
fn towards_takes_a_mask_and_leaves_uncovered_nodes_alone() {
    let mut sim = bare(&[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    let id = sim.add_towards(1.0);
    sim.set_force_targets_xy(id, &[10.0, 10.0, 10.0, 10.0]); // no target for node 2
    assert!(sim.set_force_mask(id, &[1, 0, 1]));
    apply(&mut sim, id, 1.0);
    assert_exact(&velocities(&sim), &[(10.0, 10.0), (0.0, 0.0), (0.0, 0.0)]);
    let fx = sim.add_force_x(0.0, 0.1);
    assert!(!sim.set_force_targets_xy(fx, &[1.0, 1.0]));
}

// ----------------------------------------------------------------------
// 2. Rectangle collision
// ----------------------------------------------------------------------

#[test]
fn rectangles_separate_along_the_axis_of_least_overlap() {
    // Two 20x10 rectangles (half 10 x 5), centres 14 apart in x and 2 in y:
    // overlap 6 in x, 8 in y. They part in x, 3 each.
    let mut sim = bare(&[(0.0, 0.0), (14.0, 2.0)]);
    let id = sim.add_rect_collide(10.0, 5.0, 1.0, 1);
    apply(&mut sim, id, 1.0);
    assert_exact(&velocities(&sim), &[(-3.0, 0.0), (3.0, 0.0)]);

    // Centres 4 apart in x and 7 in y: overlap 16 in x, 3 in y. They part in y.
    let mut sim = bare(&[(0.0, 0.0), (4.0, -7.0)]);
    let id = sim.add_rect_collide(10.0, 5.0, 1.0, 1);
    apply(&mut sim, id, 1.0);
    assert_exact(&velocities(&sim), &[(0.0, 1.5), (0.0, -1.5)]);
}

#[test]
fn rectangles_that_only_touch_or_miss_are_left_alone() {
    let mut sim = bare(&[(0.0, 0.0), (20.0, 0.0), (0.0, 10.5), (-30.0, -30.0)]);
    let id = sim.add_rect_collide(10.0, 5.0, 1.0, 1);
    apply(&mut sim, id, 1.0);
    assert_exact(&velocities(&sim), &[(0.0, 0.0); 4]);
}

#[test]
fn the_larger_rectangle_moves_less_and_strength_scales_the_push() {
    // Areas 10x10 = 100 and 30x10 = 300 (half extents): overlap in x is
    // (10 + 30) - 35 = 5, in y 20. The small one takes 300/400 of it.
    let mut sim = bare(&[(0.0, 0.0), (35.0, 0.0)]);
    let id = sim.add_rect_collide(1.0, 1.0, 0.5, 1);
    assert!(sim.set_force_half_widths(id, &[10.0, 30.0]));
    assert!(sim.set_force_half_heights(id, &[10.0, 10.0]));
    apply(&mut sim, id, 1.0);
    assert_exact(&velocities(&sim), &[(-5.0 * 0.5 * 0.75, 0.0), (5.0 * 0.5 * 0.25, 0.0)]);
}

#[test]
fn coincident_rectangles_part_by_index_and_equal_overlaps_part_in_x() {
    let mut sim = bare(&[(3.0, 3.0), (3.0, 3.0)]);
    let id = sim.add_rect_collide(4.0, 4.0, 1.0, 1);
    apply(&mut sim, id, 1.0);
    // Overlap 8 on both axes: x wins the tie; the lower index goes negative.
    assert_exact(&velocities(&sim), &[(-4.0, 0.0), (4.0, 0.0)]);
}

#[test]
fn rectangle_collision_uses_position_plus_velocity() {
    let mut sim = bare(&[(0.0, 0.0), (30.0, 0.0)]);
    sim.set_velocity(0, 16.0, 0.0);
    let id = sim.add_rect_collide(10.0, 5.0, 1.0, 1);
    apply(&mut sim, id, 1.0);
    // Anticipated centres 14 apart: overlap 6, 3 each.
    assert_exact(&velocities(&sim), &[(13.0, 0.0), (3.0, 0.0)]);
}

#[test]
fn rectangle_collision_matches_the_all_pairs_reference_exactly() {
    for (n, side, iterations, seed) in [(400usize, 300.0 as Real, 1u32, 21u64), (300, 120.0, 3, 22), (60, 15.0, 4, 23)] {
        let mut points = cloud(n, side, seed);
        points[7] = points[3]; // coincident centres
        points[9] = points[3];
        points[11].0 = points[5].0; // same x, different y
        let half_widths: Vec<Real> = (0..n).map(|i| 2.0 + (i % 7) as Real * 3.0).collect();
        let half_heights: Vec<Real> = (0..n).map(|i| 1.0 + (i % 4) as Real * 2.5).collect();
        let mut sim = bare(&points);
        let v = give_velocities(&mut sim, 8.0, seed + 100);
        let id = sim.add_rect_collide(1.0, 1.0, 0.7, iterations);
        sim.set_force_half_widths(id, &half_widths);
        sim.set_force_half_heights(id, &half_heights);
        apply(&mut sim, id, 1.0);
        let expected = reference::rect_collide(&points, &v, &half_widths, &half_heights, 0.7, iterations);
        let touched = expected.iter().zip(&v).filter(|(a, b)| a != b).count();
        assert!(touched > n / 4, "only {touched} of {n} rectangles collide in this layout");
        assert_exact(&velocities(&sim), &expected);
    }
}

#[test]
fn rectangle_collision_with_one_huge_rectangle_matches_the_reference() {
    // One rectangle spanning the whole layout: the search must not prune it.
    let n = 200;
    let points = cloud(n, 2000.0, 24);
    let mut half_widths = vec![3.0 as Real; n];
    let mut half_heights = vec![3.0 as Real; n];
    half_widths[150] = 1500.0;
    half_heights[150] = 40.0;
    let mut sim = bare(&points);
    let id = sim.add_rect_collide(1.0, 1.0, 1.0, 2);
    sim.set_force_half_widths(id, &half_widths);
    sim.set_force_half_heights(id, &half_heights);
    apply(&mut sim, id, 1.0);
    let expected = reference::rect_collide(&points, &vec![(0.0, 0.0); n], &half_widths, &half_heights, 1.0, 2);
    assert_exact(&velocities(&sim), &expected);
}

#[test]
fn rectangle_collision_clears_overlapping_labels() {
    let n = 80;
    let points = cloud(n, 220.0, 25);
    let half_widths: Vec<Real> = (0..n).map(|i| 8.0 + (i % 5) as Real * 4.0).collect();
    let mut sim = bare(&points);
    sim.set_alpha_decay(0.0);
    let id = sim.add_rect_collide(1.0, 4.0, 1.0, 4);
    sim.set_force_half_widths(id, &half_widths);
    let overlap = |sim: &Simulation| {
        let p = positions(sim);
        let mut total = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let ox = half_widths[i] + half_widths[j] - (p[i].0 - p[j].0).abs();
                let oy = 8.0 - (p[i].1 - p[j].1).abs();
                if ox > 0.0 && oy > 0.0 {
                    total += ox.min(oy);
                }
            }
        }
        total
    };
    let before = overlap(&sim);
    for _ in 0..300 {
        sim.tick();
    }
    let after = overlap(&sim);
    assert!(before > 100.0, "the layout starts with overlap {before}");
    assert!(after < before * 0.001, "overlap {before} before, {after} after");
}

// ----------------------------------------------------------------------
// 3. Containment
// ----------------------------------------------------------------------

fn pentagon() -> Vec<Point> {
    vec![(0.0, -100.0), (95.0, -31.0), (59.0, 81.0), (-59.0, 81.0), (-95.0, -31.0)]
}

fn kernel_region(shape: &Shape) -> Region {
    match shape {
        Shape::Rect(x0, y0, x1, y1) => Region::rect(*x0, *y0, *x1, *y1),
        Shape::Circle(x, y, r) => Region::Circle { x: *x, y: *y, r: *r },
        Shape::Polygon(vertices) => Region::polygon(&vertices.iter().flat_map(|&(x, y)| [x, y]).collect::<Vec<_>>()),
    }
}

fn add_shape(sim: &mut Simulation, id: u32, shape: &Shape) -> i32 {
    match shape {
        Shape::Rect(x0, y0, x1, y1) => sim.containment_add_rect(id, *x0, *y0, *x1, *y1),
        Shape::Circle(x, y, r) => sim.containment_add_circle(id, *x, *y, *r),
        Shape::Polygon(vertices) => sim.containment_add_polygon(id, &vertices.iter().flat_map(|&(x, y)| [x, y]).collect::<Vec<_>>()),
    }
}

fn shapes() -> Vec<Shape> {
    vec![
        Shape::Rect(-120.0, -60.0, 80.0, 90.0),
        Shape::Circle(30.0, -20.0, 110.0),
        Shape::Polygon(pentagon()),
        Shape::Polygon(pentagon().into_iter().rev().collect()),
        Shape::Rect(0.0, 0.0, 6.0, 300.0), // narrower than some nodes
    ]
}

#[test]
fn projection_worked_examples() {
    let rect = Region::rect(0.0, 0.0, 100.0, 50.0);
    assert_eq!(rect.project(0.0, 30.0, 20.0), (30.0, 20.0)); // inside: untouched
    assert_eq!(rect.project(0.0, -10.0, 70.0), (0.0, 50.0)); // to the corner
    assert_eq!(rect.project(5.0, 200.0, 25.0), (95.0, 25.0)); // radius keeps it 5 in
    assert_eq!(rect.project(30.0, 10.0, 10.0), (30.0, 25.0)); // too tall to fit: mid-height

    let circle = Region::Circle { x: 10.0, y: 10.0, r: 50.0 };
    assert_eq!(circle.project(0.0, 20.0, 20.0), (20.0, 20.0));
    assert_eq!(circle.project(0.0, 110.0, 10.0), (60.0, 10.0));
    assert_eq!(circle.project(10.0, 10.0, -90.0), (10.0, -30.0));
    assert_eq!(circle.project(80.0, 40.0, 10.0), (10.0, 10.0)); // cannot fit: centre

    // A square as a polygon behaves as the rectangle does.
    let square = Region::polygon(&[0.0, 0.0, 100.0, 0.0, 100.0, 50.0, 0.0, 50.0]);
    assert_eq!(square.project(0.0, 30.0, 20.0), (30.0, 20.0));
    assert_eq!(square.project(0.0, -10.0, 70.0), (0.0, 50.0));
    assert_eq!(square.project(5.0, 200.0, 25.0), (95.0, 25.0));
    // Fewer than three vertices confines nothing.
    assert_eq!(Region::polygon(&[0.0, 0.0, 1.0, 1.0]).project(0.0, 9.0, 9.0), (9.0, 9.0));
}

#[test]
fn projection_matches_its_reference_exactly_and_lands_inside() {
    let probes = cloud(2000, 500.0, 31);
    for shape in shapes() {
        let region = kernel_region(&shape);
        for (k, &p) in probes.iter().enumerate() {
            let r = (k % 4) as Real * 3.0;
            let c = region.project(r, p.0, p.1);
            assert_exact(&[c], &[reference::project(&shape, r, p)]);
            // Projecting again changes (almost) nothing.
            let again = region.project(r, c.0, c.1);
            assert_close(again.0, c.0, 1e-3);
            assert_close(again.1, c.1, 1e-3);
            // The result is inside, radius and all.
            match &shape {
                Shape::Rect(x0, y0, x1, y1) => {
                    if x0 + r <= x1 - r {
                        assert!(c.0 >= x0 + r && c.0 <= x1 - r);
                    }
                    assert!(c.1 >= y0 + r && c.1 <= y1 - r);
                }
                Shape::Circle(x, y, radius) => {
                    let d = ((c.0 - x).powi(2) + (c.1 - y).powi(2)).sqrt();
                    assert!(d <= radius - r + 1e-3, "distance {d} from the centre");
                }
                Shape::Polygon(vertices) => {
                    let ccw = vertices[0].1 < 0.0 && vertices[1].0 > 0.0;
                    for i in 0..vertices.len() {
                        let (a, b) = (vertices[i], vertices[(i + 1) % vertices.len()]);
                        let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
                        let cross = ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)) / len;
                        let inside = if ccw { cross } else { -cross };
                        assert!(inside >= r - 1e-2, "edge {i}: {inside} inside, radius {r}");
                    }
                }
            }
        }
    }
}

#[test]
fn projection_onto_a_rectangle_and_a_circle_is_the_nearest_allowed_point() {
    // No point of the shrunken region, on a fine grid, is nearer than the
    // projection (beyond what the grid's spacing allows).
    let probes = cloud(200, 500.0, 32);
    for (shape, r) in [(Shape::Rect(-120.0, -60.0, 80.0, 90.0), 7.0 as Real), (Shape::Circle(30.0, -20.0, 110.0), 12.0)] {
        let region = kernel_region(&shape);
        let mut allowed = Vec::new();
        for gx in -150..=150 {
            for gy in -150..=150 {
                let q = (gx as Real, gy as Real);
                let c = region.project(r, q.0, q.1);
                if c == q {
                    allowed.push(q);
                }
            }
        }
        for &p in &probes {
            let c = region.project(r, p.0, p.1);
            let mine = ((c.0 - p.0).powi(2) + (c.1 - p.1).powi(2)).sqrt();
            let best = allowed.iter().map(|q| ((q.0 - p.0).powi(2) + (q.1 - p.1).powi(2)).sqrt()).fold(Real::INFINITY, Real::min);
            assert!(mine <= best + 1e-3, "projection is {mine} away, a grid point is {best} away");
        }
    }
}

fn contained(hard: bool, strength: Real) -> (Simulation, u32, Vec<Shape>, Vec<Option<usize>>, Vec<Real>) {
    let n = 400;
    let mut sim = bare(&cloud(n, 600.0, 33));
    let id = sim.add_containment(strength, hard);
    let shapes = shapes();
    for (k, shape) in shapes.iter().enumerate() {
        assert_eq!(add_shape(&mut sim, id, shape), k as i32);
    }
    // Every sixth node is free; the rest are spread over the five regions.
    let region_of: Vec<Option<usize>> = (0..n).map(|i| (i % 6 != 5).then_some(i % 6)).collect();
    let radii: Vec<Real> = (0..n).map(|i| (i % 4) as Real * 2.0).collect();
    assert!(sim.set_force_regions(id, &region_of.iter().map(|r| r.map_or(NONE_ID, |k| k as u32)).collect::<Vec<_>>()));
    assert!(sim.set_force_radii(id, &radii));
    (sim, id, shapes, region_of, radii)
}

#[test]
fn soft_containment_matches_its_reference_exactly() {
    let (mut sim, id, shapes, region_of, radii) = contained(false, 0.6);
    let p = positions(&sim);
    let v = give_velocities(&mut sim, 30.0, 34);
    apply(&mut sim, id, 0.2); // alpha is not used
    let expected = reference::contain_soft(&p, &v, &shapes, &region_of, &radii, 0.6);
    assert!(expected.iter().zip(&v).filter(|(a, b)| a != b).count() > 100);
    assert_exact(&velocities(&sim), &expected);
    // Free nodes are untouched.
    for i in (5..400).step_by(6) {
        assert_eq!(velocities(&sim)[i], v[i]);
    }
}

#[test]
fn soft_containment_aims_a_node_at_the_boundary() {
    // Anticipated position (130, 20), region x <= 100: strength 1 aims it at 100.
    let mut sim = bare(&[(120.0, 20.0)]);
    sim.set_velocity(0, 10.0, 0.0);
    let id = sim.add_containment(1.0, false);
    sim.containment_add_rect(id, 0.0, 0.0, 100.0, 50.0);
    apply(&mut sim, id, 1.0);
    assert_exact(&velocities(&sim), &[(-20.0, 0.0)]);
    // Half strength closes half the excursion.
    let mut sim = bare(&[(120.0, 20.0)]);
    sim.set_velocity(0, 10.0, 0.0);
    let id = sim.add_containment(0.5, false);
    sim.containment_add_rect(id, 0.0, 0.0, 100.0, 50.0);
    apply(&mut sim, id, 1.0);
    assert_exact(&velocities(&sim), &[(-5.0, 0.0)]);
}

#[test]
fn hard_containment_matches_its_reference_exactly() {
    let (mut sim, _, shapes, region_of, radii) = contained(true, 0.0);
    sim.set_velocity_decay(1.0);
    let p0 = positions(&sim);
    let v0 = give_velocities(&mut sim, 30.0, 35);
    sim.tick();
    // Integration with decay 1, then the clamp.
    let moved: Vec<Point> = p0.iter().zip(&v0).map(|(p, v)| (p.0 + v.0, p.1 + v.1)).collect();
    let (p, v) = reference::contain_hard(&moved, &v0, &shapes, &region_of, &radii);
    assert_exact(&positions(&sim), &p);
    assert_exact(&velocities(&sim), &v);
}

#[test]
fn hard_containment_keeps_every_node_inside_whatever_pushes_it() {
    let n = 300;
    let mut sim = bare(&cloud(n, 100.0, 36));
    sim.add_many_body(-200.0, 0.9, 1.0, f64::INFINITY); // far too strong for the box
    let id = sim.add_containment(0.5, true);
    sim.containment_add_rect(id, -80.0, -50.0, 80.0, 50.0);
    sim.containment_add_circle(id, 300.0, 0.0, 60.0);
    sim.set_force_regions(id, &(0..n as u32).map(|i| i % 2).collect::<Vec<_>>());
    sim.set_force_radii(id, &vec![4.0; n]);
    for tick in 0..100 {
        sim.tick();
        for (i, p) in positions(&sim).iter().enumerate() {
            if i % 2 == 0 {
                assert!(p.0 >= -76.0 && p.0 <= 76.0 && p.1 >= -46.0 && p.1 <= 46.0, "tick {tick}: node {i} at {p:?}");
            } else {
                let d = ((p.0 - 300.0).powi(2) + p.1.powi(2)).sqrt();
                assert!(d <= 56.0 + 1e-3, "tick {tick}: node {i} is {d} from the centre");
            }
        }
    }
}

#[test]
fn soft_containment_alone_brings_nodes_in_and_holds_them_near() {
    let n = 200;
    let mut sim = bare(&cloud(n, 900.0, 37));
    let id = sim.add_containment(1.0, false);
    sim.containment_add_polygon(id, &pentagon().iter().flat_map(|&(x, y)| [x, y]).collect::<Vec<_>>());
    for _ in 0..60 {
        sim.tick();
    }
    let region = kernel_region(&Shape::Polygon(pentagon()));
    for p in positions(&sim) {
        let c = region.project(0.0, p.0, p.1);
        assert_close(p.0, c.0, 1e-2);
        assert_close(p.1, c.1, 1e-2);
    }
}

#[test]
fn a_pin_beats_a_region() {
    let mut sim = bare(&[(500.0, 500.0), (500.0, 500.0)]);
    sim.fix_node(0, 500.0, Real::NAN);
    let id = sim.add_containment(1.0, true);
    sim.containment_add_rect(id, 0.0, 0.0, 100.0, 100.0);
    sim.tick();
    assert_eq!(positions(&sim), vec![(500.0, 100.0), (100.0, 100.0)]);
    assert_eq!(sim.state.velocities[0], 0.0);
}

// ----------------------------------------------------------------------
// 4. Group cohesion and scoped many-body
// ----------------------------------------------------------------------

fn grouped(n: usize, seed: u64) -> (Vec<Point>, Vec<Option<u32>>) {
    // Group ids are arbitrary numbers, not 0..k; every eleventh node has none.
    let ids = [7u32, 3, 1000, 42, 8];
    (cloud(n, 500.0, seed), (0..n).map(|i| (i % 11 != 10).then_some(ids[(i * 7 + i / 3) % 5])).collect())
}

#[test]
fn cohesion_pulls_toward_the_group_centroid() {
    // Group 5 is nodes 0, 1, 2 (centroid (10, 20)); node 3 is alone in group 9.
    let mut sim = bare(&[(0.0, 0.0), (30.0, 0.0), (0.0, 60.0), (100.0, 100.0), (-50.0, -50.0)]);
    let id = sim.add_cohesion(0.5);
    assert!(sim.set_force_groups(id, &[5, 5, 5, 9, NONE_ID]));
    apply(&mut sim, id, 0.5);
    assert_exact(&velocities(&sim), &[(2.5, 5.0), (-5.0, 5.0), (2.5, -10.0), (0.0, 0.0), (0.0, 0.0)]);
}

#[test]
fn cohesion_matches_its_reference_exactly() {
    let (points, groups) = grouped(500, 41);
    let strengths: Vec<Real> = (0..500).map(|i| 0.02 + (i % 6) as Real * 0.05).collect();
    let mut sim = bare(&points);
    let v = give_velocities(&mut sim, 10.0, 42);
    let id = sim.add_cohesion(0.1);
    sim.set_force_groups(id, &group_ids(&groups));
    sim.set_force_strengths(id, &strengths);
    apply(&mut sim, id, 0.6);
    assert_exact(&velocities(&sim), &reference::cohesion(&points, &v, &groups, &strengths, 0.6));
}

#[test]
fn cohesion_gathers_each_group() {
    let (points, groups) = grouped(200, 43);
    let mut sim = bare(&points);
    sim.set_alpha_decay(0.0);
    let id = sim.add_cohesion(0.3);
    sim.set_force_groups(id, &group_ids(&groups));
    for _ in 0..200 {
        sim.tick();
    }
    let p = positions(&sim);
    for i in 0..200 {
        for j in 0..200 {
            if groups[i].is_some() && groups[i] == groups[j] {
                assert_close(p[i].0, p[j].0, 1e-2);
                assert_close(p[i].1, p[j].1, 1e-2);
            }
        }
        if groups[i].is_none() {
            assert_eq!(p[i], points[i]);
        }
    }
}

#[test]
fn many_body_within_groups_is_d3s_many_body_on_each_group_alone() {
    // Exactly: each group, simulated by itself with the (d3-checked)
    // unscoped force, gets bit-identical velocities.
    let (points, groups) = grouped(600, 44);
    let strengths: Vec<Real> = (0..600).map(|i| -10.0 - (i % 5) as Real * 9.0).collect();
    let mut sim = bare(&points);
    let id = sim.add_many_body(-30.0, 0.9, 2.0, 400.0);
    sim.set_force_strengths(id, &strengths);
    sim.set_force_groups(id, &group_ids(&groups));
    assert!(sim.set_many_body_scope(id, 1));
    apply(&mut sim, id, 0.8);
    let got = velocities(&sim);

    for group in [1u32, 3, 7, 8, 42, 1000] {
        let members: Vec<usize> = (0..600).filter(|&i| groups[i] == Some(group)).collect();
        if group == 1 {
            assert!(members.is_empty());
            continue;
        }
        assert!(members.len() > 50);
        let mut alone = bare(&members.iter().map(|&i| points[i]).collect::<Vec<_>>());
        let f = alone.add_many_body(-30.0, 0.9, 2.0, 400.0);
        alone.set_force_strengths(f, &members.iter().map(|&i| strengths[i]).collect::<Vec<_>>());
        apply(&mut alone, f, 0.8);
        let expected = velocities(&alone);
        assert_exact(&members.iter().map(|&i| got[i]).collect::<Vec<_>>(), &expected);
    }
    for i in 0..600 {
        if groups[i].is_none() {
            assert_eq!(got[i], (0.0, 0.0), "a node in no group feels nothing within groups");
        }
    }
}

/// Scoped many-body with theta = 0 visits every pair, as the reference
/// does, but in the quadtree's order rather than index order, so the sums
/// round differently. Bound: 1000 ulps of the largest velocity for
/// same-group pairs; 100,000 for different-group pairs, which are computed
/// as (all pairs) - (same-group pairs) and so also lose bits to the
/// subtraction.
#[test]
fn scoped_many_body_matches_the_all_pairs_reference_within_rounding() {
    let (points, groups) = grouped(400, 45);
    let strengths: Vec<Real> = (0..400).map(|i| -10.0 - (i % 5) as Real * 9.0).collect();
    let zero = vec![(0.0, 0.0); 400];
    for (scope, pairs, ulps) in [(0u8, Pairs::All, 1e3), (1, Pairs::SameGroup, 1e3), (2, Pairs::DifferentGroups, 1e5)] {
        let mut sim = bare(&points);
        let id = sim.add_many_body(-30.0, 0.0, 3.0, 350.0);
        sim.set_force_strengths(id, &strengths);
        sim.set_force_groups(id, &group_ids(&groups));
        assert!(sim.set_many_body_scope(id, scope));
        apply(&mut sim, id, 0.9);
        let expected = reference::many_body(&points, &zero, &strengths, &groups, pairs, 3.0, 350.0, 0.9);
        let error = worst_relative(&velocities(&sim), &expected);
        assert!(error <= ulps as Real * Real::EPSILON, "scope {scope}: relative error {error}");
    }
}

#[test]
fn between_group_many_body_cancels_exactly_inside_one_group() {
    // Everyone in one group, nothing between groups: all - within = 0, and
    // exactly 0, since both passes build the same tree.
    let points = cloud(300, 400.0, 46);
    let mut sim = bare(&points);
    let id = sim.add_many_body(-30.0, 0.9, 1.0, f64::INFINITY);
    sim.set_force_groups(id, &vec![4; 300]);
    sim.set_many_body_scope(id, 2);
    apply(&mut sim, id, 1.0);
    for v in velocities(&sim) {
        assert_close(v.0, 0.0, 1e-3);
        assert_close(v.1, 0.0, 1e-3);
    }
}

#[test]
fn groups_do_not_change_an_unscoped_many_body_and_scope_needs_a_many_body() {
    let (points, groups) = grouped(200, 47);
    let mut plain = bare(&points);
    let a = plain.add_many_body(-30.0, 0.9, 1.0, f64::INFINITY);
    apply(&mut plain, a, 1.0);
    let mut with_groups = bare(&points);
    let b = with_groups.add_many_body(-30.0, 0.9, 1.0, f64::INFINITY);
    with_groups.set_force_groups(b, &group_ids(&groups));
    apply(&mut with_groups, b, 1.0);
    assert_exact(&velocities(&plain), &velocities(&with_groups));

    let c = plain.add_center(0.0, 0.0, 1.0);
    assert!(!plain.set_many_body_scope(c, 1));
    assert!(!plain.set_many_body_scope(a, 3));
    assert!(!plain.set_force_groups(c, &[1]));
}

// ----------------------------------------------------------------------
// 6. Directional link constraint
// ----------------------------------------------------------------------

#[test]
fn directional_pushes_a_child_below_its_parent() {
    // Parent at y = 0, child at y = 10, wanted gap 40 along +y: 30 short.
    let mut sim = bare(&[(0.0, 0.0), (25.0, 10.0)]);
    sim.set_links(&[0, 1]);
    let id = sim.add_directional(40.0, 0.0, 1.0, 1.0, 1);
    apply(&mut sim, id, 1.0);
    // One link each: bias 1/2. The child goes down 15, the parent up 15.
    assert_exact(&velocities(&sim), &[(0.0, -15.0), (0.0, 15.0)]);

    // Strength and alpha scale it; x is never touched.
    let mut sim = bare(&[(0.0, 0.0), (25.0, 10.0)]);
    sim.set_links(&[0, 1]);
    let id = sim.add_directional(40.0, 0.0, 1.0, 0.5, 1);
    apply(&mut sim, id, 0.5);
    assert_exact(&velocities(&sim), &[(0.0, -3.75), (0.0, 3.75)]);
}

#[test]
fn directional_leaves_a_satisfied_link_alone() {
    let mut sim = bare(&[(0.0, 0.0), (-300.0, 40.0), (0.0, 200.0)]);
    sim.set_links(&[0, 1, 0, 2]);
    let id = sim.add_directional(40.0, 0.0, 1.0, 1.0, 1);
    apply(&mut sim, id, 1.0);
    assert_exact(&velocities(&sim), &[(0.0, 0.0); 3]);
}

#[test]
fn directional_normalises_its_axis_and_shares_by_bias() {
    // Axis (3, 4) is (0.6, 0.8). A hub with three children: bias 3/4, so a
    // child takes three quarters of each push.
    let mut sim = bare(&[(0.0, 0.0), (0.0, 0.0), (1000.0, 1000.0), (2000.0, 2000.0)]);
    sim.set_links(&[0, 1, 0, 2, 0, 3]);
    let id = sim.add_directional(10.0, 3.0, 4.0, 1.0, 1);
    apply(&mut sim, id, 1.0);
    let v = velocities(&sim);
    assert_close(v[1].0, 0.6 * 10.0 * 0.75, 1e-5);
    assert_close(v[1].1, 0.8 * 10.0 * 0.75, 1e-5);
    assert_close(v[0].0, -0.6 * 10.0 * 0.25, 1e-5);
    assert_eq!(v[2], (0.0, 0.0));
}

#[test]
fn directional_matches_its_reference_exactly() {
    let n = 300;
    let points = cloud(n, 400.0, 51);
    let mut links = Vec::new();
    for i in 1..n {
        links.push(((i - 1) / 3, i)); // a ternary tree
        if i % 10 == 0 {
            links.push((i, (i * 7) % n)); // and some cross links, a few backwards
        }
    }
    let links: Vec<(usize, usize)> = links.into_iter().filter(|(s, t)| s != t).collect();
    let gaps: Vec<Real> = (0..links.len()).map(|i| 20.0 + (i % 4) as Real * 15.0).collect();
    let strengths: Vec<Real> = (0..links.len()).map(|i| 0.2 + (i % 3) as Real * 0.3).collect();
    let mut sim = bare(&points);
    let v = give_velocities(&mut sim, 15.0, 52);
    sim.set_links(&links.iter().flat_map(|&(s, t)| [s as u32, t as u32]).collect::<Vec<_>>());
    let id = sim.add_directional(1.0, 0.0, 1.0, 1.0, 3);
    assert!(sim.set_force_gaps(id, &gaps));
    assert!(sim.set_force_strengths(id, &strengths));
    apply(&mut sim, id, 0.7);
    assert_exact(&velocities(&sim), &reference::directional(&points, &v, &links, &gaps, (0.0, 1.0), &strengths, 3, 0.7));

    // A slanted axis too. The kernel normalises (1, 2); so does this.
    let len = (5.0 as Real).sqrt();
    let axis = (1.0 / len, 2.0 / len);
    let mut sim = bare(&points);
    sim.set_links(&links.iter().flat_map(|&(s, t)| [s as u32, t as u32]).collect::<Vec<_>>());
    let id = sim.add_directional(30.0, 1.0, 2.0, 0.5, 2);
    apply(&mut sim, id, 1.0);
    let uniform = |value: Real| vec![value; links.len()];
    assert_exact(&velocities(&sim), &reference::directional(&points, &vec![(0.0, 0.0); n], &links, &uniform(30.0), axis, &uniform(0.5), 2, 1.0));
}

#[test]
fn directional_layers_a_tree() {
    let n = 121;
    let mut sim = bare(&cloud(n, 200.0, 53));
    let links: Vec<(usize, usize)> = (1..n).map(|i| ((i - 1) / 3, i)).collect();
    sim.set_links(&links.iter().flat_map(|&(s, t)| [s as u32, t as u32]).collect::<Vec<_>>());
    sim.set_alpha_decay(0.0);
    sim.add_directional(30.0, 0.0, 1.0, 1.0, 2);
    for _ in 0..400 {
        sim.tick();
    }
    let p = positions(&sim);
    for &(parent, child) in &links {
        assert!(p[child].1 - p[parent].1 >= 30.0 - 1e-2, "child {child} is {} below parent {parent}", p[child].1 - p[parent].1);
    }
}

// ----------------------------------------------------------------------
// 5. Snapshot, restore, convergence
// ----------------------------------------------------------------------

/// Every kind of force the kernel has, with coincident nodes (so the
/// random stream is in play) and pins.
fn everything(n: usize) -> Simulation {
    let (mut points, groups) = grouped(n, 61);
    points[5] = points[4];
    points[6] = points[4];
    let mut sim = bare(&points);
    let mut links = Vec::new();
    for i in 1..n {
        links.extend([((i - 1) / 2) as u32, i as u32]);
    }
    sim.set_links(&links);
    sim.fix_node(0, 0.0, -150.0);
    sim.fix_node(1, Real::NAN, -100.0);

    sim.add_many_body(-30.0, 0.9, 1.0, f64::INFINITY);
    let between = sim.add_many_body(-20.0, 0.8, 1.0, 300.0);
    sim.set_force_groups(between, &group_ids(&groups));
    sim.set_many_body_scope(between, 2);
    let within = sim.add_many_body(15.0, 0.9, 5.0, 200.0);
    sim.set_force_groups(within, &group_ids(&groups));
    sim.set_many_body_scope(within, 1);
    sim.add_links(30.0, 0.3, 1);
    sim.add_collide(4.0, 0.8, 1);
    sim.add_force_x(0.0, 0.02);
    sim.add_force_y(0.0, 0.02);
    sim.add_radial(120.0, 0.0, 0.0, 0.01);
    let towards = sim.add_towards(0.05);
    sim.set_force_targets_xy(towards, &cloud(n, 300.0, 62).iter().flat_map(|&(x, y)| [x, y]).collect::<Vec<_>>());
    let rects = sim.add_rect_collide(6.0, 3.0, 0.7, 2);
    sim.set_force_half_widths(rects, &(0..n).map(|i| 4.0 + (i % 3) as Real * 2.0).collect::<Vec<_>>());
    let cohesion = sim.add_cohesion(0.05);
    sim.set_force_groups(cohesion, &group_ids(&groups));
    sim.add_directional(25.0, 0.0, 1.0, 0.4, 1);
    let contain = sim.add_containment(0.5, true);
    sim.containment_add_circle(contain, 0.0, 0.0, 260.0);
    sim.set_force_radii(contain, &vec![3.0; n]);
    sim.add_center(0.0, 0.0, 0.5);
    sim
}

#[test]
fn restore_then_n_ticks_is_bit_identical_to_having_continued() {
    let mut original = everything(250);
    for _ in 0..40 {
        original.tick();
    }
    let snapshot = original.snapshot();
    assert_eq!(snapshot.len(), 8 + 250 * 6);

    // Restore into a simulation that was built the same way but has since
    // been somewhere else entirely.
    let mut other = everything(250);
    for _ in 0..7 {
        other.tick();
    }
    other.set_alpha_decay(0.5);
    other.set_velocity_decay(0.1);
    other.fix_node(9, 1.0, 1.0);
    assert!(other.restore(&snapshot));
    assert_eq!(other.max_displacement(), 0.0);

    for tick in 1..=120 {
        original.tick();
        other.tick();
        assert!(bits(&positions(&original)) == bits(&positions(&other)), "positions diverged {tick} ticks after restore");
        assert!(bits(&velocities(&original)) == bits(&velocities(&other)), "velocities diverged {tick} ticks after restore");
        assert_eq!(original.get_alpha().to_bits(), other.get_alpha().to_bits());
    }
    assert!(positions(&original).iter().all(|p| p.0.is_finite() && p.1.is_finite()));
    assert_eq!(original.snapshot().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), other.snapshot().iter().map(|v| v.to_bits()).collect::<Vec<_>>());

    // And the same simulation can go back to its own past.
    let later: Vec<Point> = positions(&original);
    assert!(original.restore(&snapshot));
    for _ in 0..120 {
        original.tick();
    }
    assert_exact(&positions(&original), &later);
}

#[test]
fn the_random_stream_is_part_of_the_snapshot() {
    // Coincident nodes under many-body: every tick draws from the stream.
    let stacked = || {
        let mut sim = Simulation::new(6);
        sim.enable_forces_all(true, false, false, false, false, false);
        sim.set_alpha_decay(0.0);
        sim
    };
    let restack = |sim: &mut Simulation| {
        sim.set_positions(&[1.0; 12]);
        sim.set_velocities(&[0.0; 12]);
    };
    let mut a = stacked();
    restack(&mut a);
    a.tick();
    restack(&mut a);
    let snapshot = a.snapshot();
    a.tick();

    let mut b = stacked();
    assert!(b.restore(&snapshot));
    b.tick();
    assert_exact(&positions(&a), &positions(&b));

    // Without the stream's state the same positions give a different tick.
    let mut c = stacked();
    restack(&mut c);
    c.tick();
    assert!(bits(&positions(&a)) != bits(&positions(&c)));
}

#[test]
fn restore_resizes_and_rejects_what_is_not_a_snapshot() {
    let mut small = everything(40);
    small.tick();
    let snapshot = small.snapshot();

    let mut sim = Simulation::new(3);
    assert!(sim.restore(&snapshot));
    assert_eq!(sim.node_count(), 40);
    assert_exact(&positions(&sim), &positions(&small));

    let before = sim.snapshot();
    assert!(!sim.restore(&[]));
    assert!(!sim.restore(&snapshot[..snapshot.len() - 1]));
    let mut wrong_version = snapshot.clone();
    wrong_version[0] = 99.0;
    assert!(!sim.restore(&wrong_version));
    let mut wrong_count = snapshot.clone();
    wrong_count[1] = 41.0;
    assert!(!sim.restore(&wrong_count));
    assert_eq!(sim.snapshot().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), before.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
}

#[test]
fn kinetic_energy_and_max_displacement_worked_example() {
    let mut sim = bare(&[(0.0, 0.0), (10.0, 10.0), (5.0, 5.0)]);
    sim.set_velocity_decay(0.5);
    sim.set_velocity(0, 6.0, 8.0); // moves (3, 4): distance 5
    sim.set_velocity(1, 0.0, -2.0); // moves (0, -1): distance 1
    assert_eq!(sim.max_displacement(), 0.0);
    sim.tick();
    assert_eq!(sim.max_displacement(), 5.0);
    assert_eq!(sim.kinetic_energy(), (9.0 + 16.0 + 1.0) / 2.0);
    sim.tick();
    assert_eq!(sim.max_displacement(), 2.5);
    assert_eq!(sim.kinetic_energy(), (25.0 + 1.0) / 4.0 / 2.0);
}

#[test]
fn max_displacement_counts_moves_that_are_not_velocity() {
    // Centering moves positions directly.
    let mut sim = bare(&[(10.0, 0.0), (30.0, 0.0)]);
    sim.add_center(0.0, 0.0, 1.0);
    sim.tick();
    assert_eq!(sim.max_displacement(), 20.0);
    assert_eq!(sim.kinetic_energy(), 0.0);
}

#[test]
fn the_measures_fall_as_the_layout_settles() {
    let mut sim = everything(200);
    let mut energy = Vec::new();
    let mut moved = Vec::new();
    for _ in 0..300 {
        sim.tick();
        energy.push(sim.kinetic_energy());
        moved.push(sim.max_displacement());
    }
    assert!(energy[0] > 100.0 * energy[299], "energy {} at the start, {} at the end", energy[0], energy[299]);
    assert!(moved[0] > 10.0 * moved[299], "moved {} at the start, {} at the end", moved[0], moved[299]);
    assert!(energy.iter().chain(&moved).all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn per_element_setters_refuse_the_wrong_kind_of_force() {
    let mut sim = bare(&[(0.0, 0.0), (1.0, 1.0)]);
    let towards = sim.add_towards(0.1);
    let rects = sim.add_rect_collide(1.0, 1.0, 1.0, 1);
    let contain = sim.add_containment(1.0, false);
    let cohesion = sim.add_cohesion(0.1);
    let directional = sim.add_directional(10.0, 0.0, 1.0, 1.0, 1);
    let two = [1.0 as Real, 1.0];

    assert!(sim.set_force_strengths(towards, &two));
    assert!(sim.set_force_strengths(cohesion, &two));
    assert!(sim.set_force_strengths(directional, &two));
    assert!(!sim.set_force_strengths(rects, &two));
    assert!(!sim.set_force_strengths(contain, &two));

    assert!(sim.set_force_mask(towards, &[1, 1]));
    assert!(sim.set_force_mask(cohesion, &[1, 1]));
    assert!(!sim.set_force_mask(rects, &[1, 1]));
    assert!(!sim.set_force_mask(contain, &[1, 1]));
    assert!(!sim.set_force_mask(directional, &[1, 1]));

    assert!(!sim.set_force_half_widths(towards, &two));
    assert!(!sim.set_force_half_heights(contain, &two));
    assert!(!sim.set_force_gaps(rects, &two));
    assert!(!sim.set_force_regions(cohesion, &[0, 0]));
    assert!(!sim.set_force_groups(towards, &[0, 0]));
    assert!(sim.set_force_radii(contain, &two));
    assert!(!sim.set_force_radii(rects, &two));
    assert_eq!(sim.containment_add_rect(towards, 0.0, 0.0, 1.0, 1.0), -1);
    assert_eq!(sim.containment_add_circle(99, 0.0, 0.0, 1.0), -1);
    assert_eq!(sim.containment_add_polygon(rects, &two), -1);
}

#[test]
fn everything_together_repeats_exactly_and_survives_a_resize() {
    let mut a = everything(150);
    let mut b = everything(150);
    for _ in 0..150 {
        a.tick();
        b.tick();
    }
    assert!(bits(&positions(&a)) == bits(&positions(&b)));
    a.set_node_count(220); // every per-node array is now short
    for _ in 0..20 {
        a.tick();
    }
    assert!(positions(&a).iter().all(|p| p.0.is_finite() && p.1.is_finite()));
}

// ----------------------------------------------------------------------
// At rest: max_acceleration and settled
// ----------------------------------------------------------------------

/// Nodes on springs, all released at once: they swing in step.
fn swinging(n: usize) -> (Simulation, Vec<Point>) {
    let points = cloud(n, 400.0, 71);
    let targets = cloud(n, 400.0, 72);
    let mut sim = bare(&points);
    sim.set_alpha_decay(0.0);
    sim.set_velocity_retention(0.9);
    let towards = sim.add_towards(0.02);
    sim.set_force_targets_xy(towards, &targets.iter().flat_map(|&(x, y)| [x, y]).collect::<Vec<_>>());
    (sim, targets)
}

fn furthest(sim: &Simulation, targets: &[Point]) -> f64 {
    positions(sim)
        .iter()
        .zip(targets)
        .map(|(p, t)| (((p.0 - t.0) as f64).powi(2) + ((p.1 - t.1) as f64).powi(2)).sqrt())
        .fold(0.0, f64::max)
}

#[test]
fn max_acceleration_worked_example() {
    // One node, constant pull of 1 per tick in x, nothing lost: steps 1, 2, 3.
    let mut sim = bare(&[(0.0, 0.0)]);
    sim.set_velocity_retention(1.0);
    sim.set_alpha_decay(0.0);
    let pull = sim.add_force_x(1.0e6, 1.0e-6); // (1e6 - x) * 1e-6: 1 to within 1e-5
    let _ = pull;
    assert_eq!(sim.max_acceleration(), 0.0);
    sim.tick();
    assert_eq!(sim.max_acceleration(), 0.0); // one tick: nothing to compare with
    assert!(!sim.settled(1.0e9, 1.0e9));
    sim.tick();
    assert_close(sim.max_acceleration() as Real, 1.0, 1.0e-4);
    assert!(sim.settled(1.0e9, 1.0e9));
    sim.tick();
    assert_close(sim.max_acceleration() as Real, 1.0, 1.0e-4);
    assert_close(sim.max_displacement() as Real, 3.0, 1.0e-4);
}

#[test]
fn a_swing_has_turning_points_where_nothing_moves_and_settled_is_not_fooled() {
    let (mut sim, targets) = swinging(40);
    // A step of 0.01, and the acceleration a spring of stiffness 0.02 gives a
    // node 0.01 from its target.
    let (step, acceleration) = (0.01, 0.01 * 0.02);
    let mut still_but_far = 0;
    let mut settled_at = None;
    for tick in 1..=2000 {
        sim.tick();
        let far = furthest(&sim, &targets);
        if sim.max_displacement() < 0.5 && far > 5.0 {
            // The far end of a swing: hardly moving, nowhere near rest.
            still_but_far += 1;
            assert!(sim.max_acceleration() > 0.05, "tick {tick}: acceleration {} at a turning point", sim.max_acceleration());
            assert!(!sim.settled(0.5, acceleration));
        }
        if sim.settled(step, acceleration) {
            settled_at = Some((tick, far));
            break;
        }
    }
    assert!(still_but_far >= 2, "the test needs turning points; saw {still_but_far}");
    let (tick, far) = settled_at.expect("never settled");
    // The bound `settled` documents, with k = 0.02 and retention r = 0.9:
    // (a + (1 - r) * (s + a)) / (k * r).
    let bound = (acceleration + 0.1 * (step + acceleration)) / (0.02 * 0.9);
    assert!(far < bound, "settled at tick {tick}, {far} from the targets (bound {bound})");
    assert!(bound < 0.1);
    // And no node is ever that far from its target again.
    for _ in 0..300 {
        sim.tick();
        assert!(furthest(&sim, &targets) < bound);
    }
}

#[test]
fn a_node_held_against_a_hard_wall_is_at_rest() {
    // The spring keeps pulling toward a point outside the region.
    let mut sim = bare(&[(0.0, 0.0)]);
    sim.set_alpha_decay(0.0);
    let towards = sim.add_towards(0.1);
    sim.set_force_targets_xy(towards, &[100.0, 0.0]);
    let wall = sim.add_containment(0.0, true);
    sim.containment_add_rect(wall, -10.0, -10.0, 10.0, 10.0);
    for _ in 0..200 {
        sim.tick();
    }
    assert_eq!(positions(&sim)[0], (10.0, 0.0));
    assert!(sim.settled(1.0e-6, 1.0e-6));
}

#[test]
fn the_measures_start_again_after_a_restore_or_a_resize() {
    let (mut sim, _) = swinging(10);
    for _ in 0..5 {
        sim.tick();
    }
    assert!(sim.max_acceleration() > 0.0);
    let snapshot = sim.snapshot();
    assert!(sim.restore(&snapshot));
    assert_eq!(sim.max_acceleration(), 0.0);
    assert!(!sim.settled(1.0e9, 1.0e9));
    sim.tick();
    assert!(!sim.settled(1.0e9, 1.0e9));
    sim.tick();
    assert!(sim.settled(1.0e9, 1.0e9));
    sim.set_node_count(12);
    assert!(!sim.settled(1.0e9, 1.0e9));
    sim.tick();
    sim.tick();
    assert!(sim.settled(1.0e9, 1.0e9));
}

#[test]
fn measuring_changes_nothing() {
    // Two simulations, one asked for its measures every tick.
    let mut a = everything(120);
    let mut b = everything(120);
    for _ in 0..60 {
        a.tick();
        b.tick();
        let _ = (b.max_acceleration(), b.settled(0.1, 0.1), b.max_displacement());
    }
    assert!(bits(&positions(&a)) == bits(&positions(&b)));
    assert!(bits(&velocities(&a)) == bits(&velocities(&b)));
}

#[test]
fn velocity_retention_is_velocity_decay_by_its_right_name() {
    let mut a = everything(80);
    let mut b = everything(80);
    a.set_velocity_decay(0.7);
    b.set_velocity_retention(0.7);
    for _ in 0..30 {
        a.tick();
        b.tick();
    }
    assert!(bits(&positions(&a)) == bits(&positions(&b)));
    // The share kept: a velocity of 10 moves the node 7.
    let mut sim = bare(&[(0.0, 0.0)]);
    sim.set_velocity_retention(0.7);
    sim.set_velocity(0, 10.0, 0.0);
    sim.tick();
    assert_close(positions(&sim)[0].0, 7.0, 1.0e-5);
}

// ----------------------------------------------------------------------
// Collision that yields
// ----------------------------------------------------------------------

fn crowd(n: usize) -> (Simulation, u32) {
    let mut sim = bare(&cloud(n, 60.0, 81));
    give_velocities(&mut sim, 2.0, 82);
    let id = sim.add_collide(4.0, 0.7, 2);
    sim.set_force_radii(id, &(0..n).map(|i| 2.0 + (i % 4) as Real).collect::<Vec<_>>());
    (sim, id)
}

#[test]
fn collide_scaled_by_alpha_worked_example() {
    // Two circles of radius 5, 6 apart: overlap 4, each takes half. (On a
    // diagonal, so that neither coordinate difference is zero and jiggled.)
    let two = [(0.0, 0.0), (3.6, 4.8)];
    let mut sim = bare(&two);
    let id = sim.add_collide(5.0, 1.0, 1);
    apply(&mut sim, id, 0.25);
    assert_points(&velocities(&sim), &[(-1.2, -1.6), (1.2, 1.6)]); // d3's: alpha ignored

    let mut sim = bare(&two);
    let id = sim.add_collide(5.0, 1.0, 1);
    assert!(sim.set_collide_alpha_scaled(id, true));
    apply(&mut sim, id, 0.25);
    assert_points(&velocities(&sim), &[(-0.3, -0.4), (0.3, 0.4)]);
    let mut sim = bare(&two);
    let id = sim.add_collide(5.0, 1.0, 1);
    sim.set_collide_alpha_scaled(id, true);
    apply(&mut sim, id, 0.0);
    assert_eq!(velocities(&sim), vec![(0.0, 0.0), (0.0, 0.0)]);
}

#[test]
fn collide_scaled_by_alpha_is_collide_with_the_strength_multiplied() {
    let (mut scaled, id) = crowd(200);
    scaled.set_collide_alpha_scaled(id, true);
    apply(&mut scaled, id, 0.5);

    let mut plain = bare(&cloud(200, 60.0, 81));
    give_velocities(&mut plain, 2.0, 82);
    let id = plain.add_collide(4.0, 0.7 * 0.5, 2);
    plain.set_force_radii(id, &(0..200).map(|i| 2.0 + (i % 4) as Real).collect::<Vec<_>>());
    apply(&mut plain, id, 1.0);
    assert_exact(&velocities(&scaled), &velocities(&plain));

    // Turned off again, and at alpha 1, it is the collide it was.
    let (mut a, id) = crowd(200);
    a.set_collide_alpha_scaled(id, true);
    a.set_collide_alpha_scaled(id, false);
    apply(&mut a, id, 0.3);
    let (mut b, id) = crowd(200);
    apply(&mut b, id, 0.3);
    assert_exact(&velocities(&a), &velocities(&b));
    let (mut c, id) = crowd(200);
    c.set_collide_alpha_scaled(id, true);
    apply(&mut c, id, 1.0);
    assert_exact(&velocities(&c), &velocities(&b));
}

#[test]
fn a_collide_filter_leaves_nodes_out_of_the_collision() {
    // Three circles in a row, all overlapping; the middle one is left out.
    let mut sim = bare(&[(0.0, 0.0), (2.4, 3.2), (4.8, 6.4)]);
    let id = sim.add_collide(5.0, 1.0, 1);
    assert!(sim.set_collide_filter(id, &[1, 0, 1]));
    apply(&mut sim, id, 1.0);
    // 0 and 2 are 8 apart, overlap 2, half each; 1 is untouched.
    assert_points(&velocities(&sim), &[(-0.6, -0.8), (0.0, 0.0), (0.6, 0.8)]);
    assert_eq!(velocities(&sim)[1], (0.0, 0.0));

    // A filter that lets everything through changes nothing, and neither
    // does one that was cleared.
    let (mut plain, id) = crowd(200);
    apply(&mut plain, id, 1.0);
    let (mut all, id) = crowd(200);
    sim_filter(&mut all, id, |_| true);
    apply(&mut all, id, 1.0);
    assert_exact(&velocities(&all), &velocities(&plain));
    let (mut cleared, id) = crowd(200);
    sim_filter(&mut cleared, id, |i| i % 3 == 0);
    cleared.clear_force_mask(id);
    apply(&mut cleared, id, 1.0);
    assert_exact(&velocities(&cleared), &velocities(&plain));
}

fn sim_filter(sim: &mut Simulation, id: u32, keep: impl Fn(usize) -> bool) {
    let mask: Vec<u8> = (0..sim.node_count()).map(|i| keep(i) as u8).collect();
    assert!(sim.set_collide_filter(id, &mask));
}

fn assert_points(actual: &[Point], expected: &[Point]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert_close(a.0, e.0, 1.0e-5);
        assert_close(a.1, e.1, 1.0e-5);
    }
}

#[test]
fn a_filtered_collide_is_collide_among_the_nodes_let_through() {
    // Fifty overlapping pairs, far from each other, and on top of each pair
    // a third circle that is filtered out. (d3's collide sweeps the pairs in
    // an order its quadtree decides and each push is felt by the next, so a
    // crowd would not compare exactly with the same crowd minus some nodes.
    // Isolated pairs do: each kept node is in exactly one collision.)
    let mut kept_points = Vec::new();
    let mut all_points = Vec::new();
    let mut all_radii = Vec::new();
    let mut kept_radii = Vec::new();
    let mut mask = Vec::new();
    for k in 0..50 {
        let x = 100.0 * k as Real;
        let pair = [(x, 0.0), (x + 3.0 + 0.05 * k as Real, 1.0 + 0.01 * k as Real)];
        for (which, &point) in pair.iter().enumerate() {
            let radius = 3.0 + ((k + which) % 4) as Real;
            kept_points.push(point);
            kept_radii.push(radius);
            all_points.push(point);
            all_radii.push(radius);
            mask.push(1u8);
        }
        all_points.push((x + 1.0, 0.5));
        all_radii.push(6.0);
        mask.push(0u8);
    }
    let mut filtered = bare(&all_points);
    let given = give_velocities(&mut filtered, 0.5, 91);
    let id = filtered.add_collide(1.0, 0.7, 1);
    filtered.set_force_radii(id, &all_radii);
    assert!(filtered.set_collide_filter(id, &mask));
    apply(&mut filtered, id, 1.0);
    let after = velocities(&filtered);

    let mut only = bare(&kept_points);
    let kept_index: Vec<usize> = (0..all_points.len()).filter(|&i| mask[i] == 1).collect();
    for (k, &i) in kept_index.iter().enumerate() {
        only.set_velocity(k, given[i].0, given[i].1);
    }
    let id = only.add_collide(1.0, 0.7, 1);
    only.set_force_radii(id, &kept_radii);
    apply(&mut only, id, 1.0);
    let expected = velocities(&only);

    assert_exact(&kept_index.iter().map(|&i| after[i]).collect::<Vec<_>>(), &expected);
    let moved = kept_index.iter().enumerate().filter(|&(k, &i)| expected[k] != given[i]).count();
    assert_eq!(moved, kept_index.len(), "every kept node should be in a collision");
    for i in 0..all_points.len() {
        if mask[i] == 0 {
            assert_eq!(bits(&[after[i]]), bits(&[given[i]]), "node {i} is filtered out and was pushed");
        }
    }
}

#[test]
fn collide_radii_can_be_replaced_between_ticks() {
    let (mut a, id) = crowd(100);
    let grown: Vec<Real> = (0..100).map(|i| 3.0 + (i % 5) as Real).collect();
    assert!(a.set_force_radii(id, &grown)); // same length: overwritten in place
    apply(&mut a, id, 1.0);
    let mut b = bare(&cloud(100, 60.0, 81));
    give_velocities(&mut b, 2.0, 82);
    let id = b.add_collide(4.0, 0.7, 2);
    b.set_force_radii(id, &grown);
    apply(&mut b, id, 1.0);
    assert_exact(&velocities(&a), &velocities(&b));
}

#[test]
fn the_collide_options_are_refused_by_other_forces() {
    let mut sim = bare(&[(0.0, 0.0), (1.0, 1.0)]);
    let towards = sim.add_towards(0.1);
    let rects = sim.add_rect_collide(1.0, 1.0, 1.0, 1);
    let bump = sim.add_collide(1.0, 1.0, 1);
    assert!(!sim.set_collide_alpha_scaled(towards, true));
    assert!(!sim.set_collide_alpha_scaled(rects, true));
    assert!(!sim.set_collide_alpha_scaled(99, true));
    assert!(!sim.set_collide_filter(towards, &[1, 1]));
    assert!(!sim.set_collide_filter(rects, &[1, 1]));
    assert!(!sim.set_force_mask(bump, &[1, 1]));
    assert!(sim.set_collide_filter(bump, &[1, 1]));
    assert!(sim.set_collide_alpha_scaled(bump, true));
}
