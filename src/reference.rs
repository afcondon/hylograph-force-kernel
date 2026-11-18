//! Reference implementations of the forces d3-force does not have.
//!
//! These are the definitions, written to be read and ported, not to be
//! fast: plain loops over all nodes or all pairs, no quadtree, no shared
//! scratch state, nothing borrowed from `forces.rs`. Each takes the state it
//! reads as slices and returns new velocities (and positions, for the hard
//! clamp). The optimised forces are tested against them in
//! `tests_new_forces.rs`, at f32 and (with `--features f64`) at f64.
//!
//! Conventions: a point is `(x, y)`; `None` means "this node has no target /
//! group / region"; all arithmetic is in `Real`, in the order written.

use crate::Real;

pub type Point = (Real, Real);

fn finite(p: Point) -> bool {
    p.0.is_finite() && p.1.is_finite()
}

/// TOWARDS. v_i += (target_i - p_i) * strength_i * alpha, for nodes with a target.
pub fn towards(
    positions: &[Point],
    velocities: &[Point],
    targets: &[Option<Point>],
    strengths: &[Real],
    alpha: Real,
) -> Vec<Point> {
    let mut out = velocities.to_vec();
    for i in 0..positions.len() {
        if let Some((tx, ty)) = targets[i] {
            out[i].0 += (tx - positions[i].0) * strengths[i] * alpha;
            out[i].1 += (ty - positions[i].1) * strengths[i] * alpha;
        }
    }
    out
}

/// RECTANGLE COLLISION, over every ordered pair.
pub fn rect_collide(
    positions: &[Point],
    velocities: &[Point],
    half_widths: &[Real],
    half_heights: &[Real],
    strength: Real,
    iterations: u32,
) -> Vec<Point> {
    let n = positions.len();
    let mut v = velocities.to_vec();
    for _ in 0..iterations {
        // Anticipated positions, all taken before anything moves.
        let a: Vec<Point> = (0..n).map(|i| (positions[i].0 + v[i].0, positions[i].1 + v[i].1)).collect();
        for i in 0..n {
            if !finite(a[i]) {
                continue;
            }
            for j in 0..n {
                if j == i || !finite(a[j]) {
                    continue;
                }
                let dx = a[i].0 - a[j].0;
                let dy = a[i].1 - a[j].1;
                let ox = (half_widths[i] + half_widths[j]) - dx.abs();
                let oy = (half_heights[i] + half_heights[j]) - dy.abs();
                if ox > 0.0 && oy > 0.0 {
                    let area_i = half_widths[i] * half_heights[i];
                    let area_j = half_widths[j] * half_heights[j];
                    let share = if area_i + area_j > 0.0 { area_j / (area_i + area_j) } else { 0.5 };
                    let sign = |d: Real| -> Real {
                        if d > 0.0 {
                            1.0
                        } else if d < 0.0 {
                            -1.0
                        } else if i < j {
                            -1.0
                        } else {
                            1.0
                        }
                    };
                    if ox <= oy {
                        v[i].0 += sign(dx) * ox * strength * share;
                    } else {
                        v[i].1 += sign(dy) * oy * strength * share;
                    }
                }
            }
        }
    }
    v
}

/// A region, as its defining numbers.
pub enum Shape {
    /// x0 <= x1, y0 <= y1
    Rect(Real, Real, Real, Real),
    /// centre x, centre y, radius
    Circle(Real, Real, Real),
    /// Convex, vertices in either winding.
    Polygon(Vec<Point>),
}

/// Where the centre of a circle of radius `r` at `p` must move to, for the
/// circle to lie inside the shape.
pub fn project(shape: &Shape, r: Real, p: Point) -> Point {
    match shape {
        Shape::Rect(x0, y0, x1, y1) => {
            let axis = |v: Real, lo: Real, hi: Real| -> Real {
                if lo + r > hi - r {
                    (lo + hi) / 2.0
                } else if v < lo + r {
                    lo + r
                } else if v > hi - r {
                    hi - r
                } else {
                    v
                }
            };
            (axis(p.0, *x0, *x1), axis(p.1, *y0, *y1))
        }
        Shape::Circle(cx, cy, radius) => {
            let inner = if radius - r > 0.0 { radius - r } else { 0.0 };
            let dx = p.0 - cx;
            let dy = p.1 - cy;
            let d2 = dx * dx + dy * dy;
            if d2 > inner * inner {
                let k = inner / d2.sqrt();
                (cx + dx * k, cy + dy * k)
            } else {
                p
            }
        }
        Shape::Polygon(vertices) => {
            let n = vertices.len();
            // Twice the signed area: positive for counter-clockwise.
            let mut area2 = 0.0 as Real;
            for i in 0..n {
                let (ax, ay) = vertices[i];
                let (bx, by) = vertices[(i + 1) % n];
                area2 += ax * by - bx * ay;
            }
            if n < 3 || area2 == 0.0 || !area2.is_finite() {
                return p;
            }
            let winding: Real = if area2 > 0.0 { 1.0 } else { -1.0 };
            let (mut x, mut y) = p;
            for _pass in 0..8 {
                let mut moved = false;
                for i in 0..n {
                    let (ax, ay) = vertices[i];
                    let (bx, by) = vertices[(i + 1) % n];
                    let (ex, ey) = (bx - ax, by - ay);
                    let len = (ex * ex + ey * ey).sqrt();
                    if !(len > 0.0) {
                        continue;
                    }
                    // Inward unit normal of this edge.
                    let nx = -ey / len * winding;
                    let ny = ex / len * winding;
                    let d = (x - ax) * nx + (y - ay) * ny;
                    if d < r {
                        x += nx * (r - d);
                        y += ny * (r - d);
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
            (x, y)
        }
    }
}

/// CONTAINMENT, soft: v_i += (project(p_i + v_i) - (p_i + v_i)) * strength.
pub fn contain_soft(
    positions: &[Point],
    velocities: &[Point],
    shapes: &[Shape],
    region_of: &[Option<usize>],
    radii: &[Real],
    strength: Real,
) -> Vec<Point> {
    let mut v = velocities.to_vec();
    for i in 0..positions.len() {
        let Some(shape) = region_of[i].and_then(|k| shapes.get(k)) else { continue };
        let a = (positions[i].0 + v[i].0, positions[i].1 + v[i].1);
        if !finite(a) {
            continue;
        }
        let c = project(shape, radii[i], a);
        v[i].0 += (c.0 - a.0) * strength;
        v[i].1 += (c.1 - a.1) * strength;
    }
    v
}

/// CONTAINMENT, hard: after integration, p_i := project(p_i), and v_i
/// changes by the same amount. Returns (positions, velocities).
pub fn contain_hard(
    positions: &[Point],
    velocities: &[Point],
    shapes: &[Shape],
    region_of: &[Option<usize>],
    radii: &[Real],
) -> (Vec<Point>, Vec<Point>) {
    let mut p = positions.to_vec();
    let mut v = velocities.to_vec();
    for i in 0..p.len() {
        let Some(shape) = region_of[i].and_then(|k| shapes.get(k)) else { continue };
        if !finite(p[i]) {
            continue;
        }
        let c = project(shape, radii[i], p[i]);
        v[i].0 += c.0 - p[i].0;
        v[i].1 += c.1 - p[i].1;
        p[i] = c;
    }
    (p, v)
}

/// GROUP COHESION: for each node, find its group's centroid by scanning
/// every node, then v_i += (centroid - p_i) * strength_i * alpha.
pub fn cohesion(
    positions: &[Point],
    velocities: &[Point],
    groups: &[Option<u32>],
    strengths: &[Real],
    alpha: Real,
) -> Vec<Point> {
    let n = positions.len();
    let mut v = velocities.to_vec();
    for i in 0..n {
        let Some(group) = groups[i] else { continue };
        if !finite(positions[i]) {
            continue;
        }
        let (mut sx, mut sy, mut count) = (0.0f64, 0.0f64, 0u32);
        for j in 0..n {
            if groups[j] == Some(group) && finite(positions[j]) {
                sx += positions[j].0 as f64;
                sy += positions[j].1 as f64;
                count += 1;
            }
        }
        let cx = (sx / count as f64) as Real;
        let cy = (sy / count as f64) as Real;
        v[i].0 += (cx - positions[i].0) * strengths[i] * alpha;
        v[i].1 += (cy - positions[i].1) * strengths[i] * alpha;
    }
    v
}

/// Which pairs a many-body force acts between.
#[derive(Clone, Copy, PartialEq)]
pub enum Pairs {
    All,
    SameGroup,
    DifferentGroups,
}

/// MANY-BODY over the chosen pairs, with no approximation: every pair,
/// d3's distance rules (ignore beyond distance_max, soften inside
/// distance_min). Assumes no two nodes coincide (d3 nudges those apart at
/// random). A node in no group has no same-group pairs and is in a
/// different group from everyone.
pub fn many_body(
    positions: &[Point],
    velocities: &[Point],
    strengths: &[Real],
    groups: &[Option<u32>],
    pairs: Pairs,
    distance_min: Real,
    distance_max: Real,
    alpha: Real,
) -> Vec<Point> {
    let n = positions.len();
    let mut v = velocities.to_vec();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let same = groups[i].is_some() && groups[i] == groups[j];
            let acts = match pairs {
                Pairs::All => true,
                Pairs::SameGroup => same,
                Pairs::DifferentGroups => !same,
            };
            if !acts {
                continue;
            }
            let x = positions[j].0 - positions[i].0;
            let y = positions[j].1 - positions[i].1;
            let mut l = x * x + y * y;
            if l >= distance_max * distance_max {
                continue;
            }
            if l < distance_min * distance_min {
                l = (distance_min * distance_min * l).sqrt();
            }
            v[i].0 += x * strengths[j] * alpha / l;
            v[i].1 += y * strengths[j] * alpha / l;
        }
    }
    v
}

/// DIRECTIONAL LINK CONSTRAINT, link by link in order, each reading the
/// velocities the links before it left. `axis` must be a unit vector.
pub fn directional(
    positions: &[Point],
    velocities: &[Point],
    links: &[(usize, usize)],
    gaps: &[Real],
    axis: Point,
    strengths: &[Real],
    iterations: u32,
    alpha: Real,
) -> Vec<Point> {
    let n = positions.len();
    // d3's link bias: the source's share of the two ends' link counts.
    let mut count = vec![0u32; n];
    for &(s, t) in links {
        count[s] += 1;
        count[t] += 1;
    }
    let mut v = velocities.to_vec();
    for _ in 0..iterations {
        for (i, &(s, t)) in links.iter().enumerate() {
            let bias = (count[s] as f64 / (count[s] as f64 + count[t] as f64)) as Real;
            let x = positions[t].0 + v[t].0 - positions[s].0 - v[s].0;
            let y = positions[t].1 + v[t].1 - positions[s].1 - v[s].1;
            let d = x * axis.0 + y * axis.1;
            if d < gaps[i] {
                let push = (gaps[i] - d) * strengths[i] * alpha;
                v[t].0 += axis.0 * push * bias;
                v[t].1 += axis.1 * push * bias;
                let rest = 1.0 - bias;
                v[s].0 -= axis.0 * push * rest;
                v[s].1 -= axis.1 * push * rest;
            }
        }
    }
    v
}
