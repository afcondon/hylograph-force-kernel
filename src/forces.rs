//! The forces: ports of d3-force 3.0.0's manyBody, link, center, x, y,
//! collide and radial, statement for statement, in `Real` (f32 as shipped).
//!
//! Each force reads and writes the simulation `State` directly. Nothing here
//! calls out of the kernel: every per-node or per-link parameter arrives as
//! an array, evaluated by the caller beforehand.

use crate::quadtree::{is_leaf, leaf_point, Quad, Quadtree, NONE};
use crate::Real;

/// d3-force's default random source (`lcg.js`): a linear congruential
/// generator seeded with 1, producing the same sequence as
/// `d3.forceSimulation()` uses to nudge coincident nodes apart. One stream
/// per simulation, shared by all forces, as in d3.
pub struct Lcg {
    s: u32,
}

impl Lcg {
    pub fn new() -> Lcg {
        Lcg { s: 1 }
    }

    /// The generator's whole state, for snapshots.
    pub fn state(&self) -> u32 {
        self.s
    }

    pub fn from_state(s: u32) -> Lcg {
        Lcg { s }
    }

    /// Next value in [0, 1).
    #[inline]
    pub fn next(&mut self) -> f64 {
        self.s = self.s.wrapping_mul(1664525).wrapping_add(1013904223);
        self.s as f64 / 4294967296.0
    }

    /// d3's `jiggle`: a tiny non-zero offset in (-0.5e-6, 0.5e-6].
    #[inline]
    pub fn jiggle(&mut self) -> Real {
        ((self.next() - 0.5) * 1e-6) as Real
    }
}

/// A parameter that is either one value for everything or one value per
/// element (per node, or per link for the link force).
#[derive(Clone)]
pub enum Per {
    Uniform(Real),
    Each(Vec<Real>),
}

impl Per {
    /// The value for element `i`. An array too short to have an entry for
    /// `i` gives NaN, which every force reads as "leave this element alone".
    #[inline]
    pub fn at(&self, i: usize) -> Real {
        match self {
            Per::Uniform(v) => *v,
            Per::Each(a) => match a.get(i) {
                Some(v) => *v,
                None => Real::NAN,
            },
        }
    }
}

/// Everything a force can touch.
pub struct State {
    pub n: usize,
    /// [x0, y0, x1, y1, ...]
    pub positions: Vec<Real>,
    /// [vx0, vy0, vx1, vy1, ...]
    pub velocities: Vec<Real>,
    /// [fx0, fy0, ...]; NaN means that axis is free.
    pub fixed: Vec<Real>,
    /// [source0, target0, source1, target1, ...] as node indices.
    pub links: Vec<u32>,
    /// Per link: d3's bias (the source's share of the two endpoints' link
    /// counts). NaN for a link with an endpoint out of range, which is skipped.
    pub link_bias: Vec<Real>,
    pub rng: Lcg,
    pub tree: Quadtree,
    pub stack: Vec<Quad>,
    pub scratch: Vec<Real>,
    /// More scratch: anticipated positions, node index lists.
    pub scratch_xy: Vec<Real>,
    pub scratch_index: Vec<u32>,
    pub scratch_members: Vec<u32>,
}

impl State {
    pub fn new(n: usize) -> State {
        State {
            n,
            positions: vec![0.0; n * 2],
            velocities: vec![0.0; n * 2],
            fixed: vec![Real::NAN; n * 2],
            links: Vec::new(),
            link_bias: Vec::new(),
            rng: Lcg::new(),
            tree: Quadtree::new(),
            stack: Vec::new(),
            scratch: Vec::new(),
            scratch_xy: Vec::new(),
            scratch_index: Vec::new(),
            scratch_members: Vec::new(),
        }
    }

    /// Recompute the per-link bias (d3 `link.js`, `initialize`). Must follow
    /// any change to the links or to the node count.
    pub fn index_links(&mut self) {
        let m = self.links.len() / 2;
        let n = self.n;
        let mut count = vec![0u32; n];
        for i in 0..m {
            let s = self.links[i * 2] as usize;
            let t = self.links[i * 2 + 1] as usize;
            if s < n && t < n {
                count[s] += 1;
                count[t] += 1;
            }
        }
        self.link_bias.clear();
        for i in 0..m {
            let s = self.links[i * 2] as usize;
            let t = self.links[i * 2 + 1] as usize;
            self.link_bias.push(if s < n && t < n {
                (count[s] as f64 / (count[s] as f64 + count[t] as f64)) as Real
            } else {
                Real::NAN
            });
        }
    }
}

/// "This node belongs to no group / no region."
pub const NONE_ID: u32 = u32::MAX;

/// Which pairs a many-body force acts between.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scope {
    /// Every pair (d3's many-body).
    All,
    /// Only pairs in the same group.
    Within,
    /// Only pairs in different groups.
    Between,
}

/// A region nodes can be confined to. See `Region::project`.
pub enum Region {
    /// x0 <= x1 and y0 <= y1.
    Rect { x0: Real, y0: Real, x1: Real, y1: Real },
    Circle { x: Real, y: Real, r: Real },
    /// A convex polygon as its edges: a point on each edge and the edge's
    /// inward unit normal, in the order the vertices were given.
    Polygon { edges: Vec<(Real, Real, Real, Real)> },
}

/// A polygon projection repeats its pass over the edges until a pass moves
/// nothing, and gives up after this many.
pub const POLYGON_PASSES: usize = 8;

impl Region {
    pub fn rect(x0: Real, y0: Real, x1: Real, y1: Real) -> Region {
        Region::Rect { x0: x0.min(x1), y0: y0.min(y1), x1: x0.max(x1), y1: y0.max(y1) }
    }

    /// A convex polygon from [x0, y0, x1, y1, ...], in either winding.
    /// Convexity is the caller's promise, not checked. Fewer than three
    /// vertices, or zero area, gives a polygon with no edges, which confines
    /// nothing.
    pub fn polygon(xy: &[Real]) -> Region {
        let n = xy.len() / 2;
        let mut area2 = 0.0 as Real;
        for i in 0..n {
            let j = (i + 1) % n;
            area2 += xy[i * 2] * xy[j * 2 + 1] - xy[j * 2] * xy[i * 2 + 1];
        }
        let mut edges = Vec::new();
        if n >= 3 && area2 != 0.0 && area2.is_finite() {
            // Counter-clockwise (area2 > 0): inward is to the left of the edge.
            let sign: Real = if area2 > 0.0 { 1.0 } else { -1.0 };
            for i in 0..n {
                let j = (i + 1) % n;
                let ex = xy[j * 2] - xy[i * 2];
                let ey = xy[j * 2 + 1] - xy[i * 2 + 1];
                let len = (ex * ex + ey * ey).sqrt();
                if len > 0.0 {
                    edges.push((xy[i * 2], xy[i * 2 + 1], -ey / len * sign, ex / len * sign));
                }
            }
        }
        Region::Polygon { edges }
    }

    /// Where a circle of radius `r` centred on (x, y) has to move its centre
    /// to, to lie inside the region: `project_body` with no half-extents.
    pub fn project(&self, r: Real, x: Real, y: Real) -> (Real, Real) {
        self.project_body(r, 0.0, 0.0, x, y)
    }

    /// THE BODY. A node is kept inside as a rectangle of half-width `w` and
    /// half-height `h`, grown all round by a radius `r`: a circle when w and
    /// h are zero, a plain rectangle when r is, a rounded rectangle
    /// otherwise.
    ///
    /// Where such a body centred on (x, y) has to move its centre to, to lie
    /// inside the region. A centre already far enough inside is returned
    /// unchanged (bit for bit).
    ///
    /// - Rect: clamp x to [x0 + (r + w), x1 - (r + w)] and y to
    ///   [y0 + (r + h), y1 - (r + h)]. If the rectangle is narrower than the
    ///   body on an axis, that coordinate goes to the middle of the
    ///   rectangle on that axis.
    /// - Circle: the body counts as the circle round it, of radius
    ///   r + sqrt(w*w + h*h) (exact for a round body, a little cautious for
    ///   a rectangle). With R' = max(R - r - sqrt(w*w + h*h), 0), a centre
    ///   further than R' from the region's centre c moves to
    ///   c + (p - c) * (R' / |p - c|).
    /// - Polygon: one pass visits the edges in order; for each, with inward
    ///   unit normal n and d the signed distance from the (current) centre
    ///   to the edge line, positive inside, the body reaches
    ///   s = (r + |n.x| * w) + |n.y| * h towards the edge, and a centre with
    ///   d < s moves along the normal by s - d. Passes repeat until one
    ///   moves nothing, at most `POLYGON_PASSES` times. (Projection onto
    ///   each half-plane in turn: exact when one edge is violated, and it
    ///   converges into the polygon at corners.)
    pub fn project_body(&self, r: Real, w: Real, h: Real, x: Real, y: Real) -> (Real, Real) {
        match self {
            Region::Rect { x0, y0, x1, y1 } => {
                let clamp = |v: Real, lo: Real, hi: Real, inset: Real| {
                    let (a, b) = (lo + inset, hi - inset);
                    if a > b {
                        (lo + hi) / 2.0
                    } else if v < a {
                        a
                    } else if v > b {
                        b
                    } else {
                        v
                    }
                };
                (clamp(x, *x0, *x1, r + w), clamp(y, *y0, *y1, r + h))
            }
            Region::Circle { x: cx, y: cy, r: radius } => {
                let reach = *radius - r - (w * w + h * h).sqrt();
                let inner = if reach > 0.0 { reach } else { 0.0 };
                let dx = x - cx;
                let dy = y - cy;
                let d2 = dx * dx + dy * dy;
                if d2 > inner * inner {
                    let k = inner / d2.sqrt();
                    (cx + dx * k, cy + dy * k)
                } else {
                    (x, y)
                }
            }
            Region::Polygon { edges } => {
                let (mut x, mut y) = (x, y);
                for _ in 0..POLYGON_PASSES {
                    let mut moved = false;
                    for &(px, py, nx, ny) in edges {
                        let d = (x - px) * nx + (y - py) * ny;
                        let reach = r + nx.abs() * w + ny.abs() * h;
                        if d < reach {
                            x += nx * (reach - d);
                            y += ny * (reach - d);
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
}

/// The most passes a hard directional constraint makes.
pub const DIRECTIONAL_PASSES: usize = 32;

pub enum Kind {
    /// `theta2`, `distance_min2`, `distance_max2` are stored squared, as d3 does.
    ManyBody { theta2: Real, distance_min2: Real, distance_max2: Real, scope: Scope },
    Link { distance: Per, iterations: u32 },
    Center { x: Real, y: Real },
    X { target: Per },
    Y { target: Per },
    /// `alpha_scaled`: multiply the strength by alpha (not d3's; off by default).
    Collide { radius: Per, iterations: u32, alpha_scaled: bool },
    Radial { radius: Per, x: Real, y: Real },
    /// `targets` is [tx0, ty0, tx1, ty1, ...].
    Towards { targets: Vec<Real> },
    /// `pairs`: which pairs collide, by `Force::groups`. `through`: a
    /// squeezed rectangle goes through one side (see `apply_rect_collide`).
    RectCollide { half_width: Per, half_height: Per, iterations: u32, pairs: Scope, through: bool },
    /// `assignment`: per node, an index into `regions` (`NONE_ID` = free).
    /// Without one, every node is in region 0.
    Contain { regions: Vec<Region>, assignment: Option<Vec<u32>>, radius: Per, half_width: Per, half_height: Per, hard: bool },
    Cohesion,
    /// (`axis_x`, `axis_y`) is a unit vector.
    /// `hard`: also project the links after integration (see `constrain`).
    Directional { gap: Per, axis_x: Real, axis_y: Real, iterations: u32, hard: bool },
}

pub struct Force {
    pub kind: Kind,
    /// Per node for many-body, x, y, radial, towards and cohesion; per link
    /// for link and directional; always uniform for center, collide,
    /// rect-collide and contain.
    pub strength: Per,
    /// Optional node filter, for the forces whose strength is per node:
    /// a node whose entry is 0 gets strength 0, exactly as the D3 kernel's
    /// filtered forces do (`filter(d) ? strength : 0`).
    pub mask: Option<Vec<u8>>,
    /// Per node group id (`NONE_ID` = in no group), for cohesion and for a
    /// many-body force with a scope.
    pub groups: Option<Vec<u32>>,
}

impl Force {
    pub fn many_body(strength: Real, theta: f64, distance_min: f64, distance_max: f64) -> Force {
        Force {
            kind: Kind::ManyBody {
                theta2: (theta * theta) as Real,
                distance_min2: (distance_min * distance_min) as Real,
                distance_max2: (distance_max * distance_max) as Real,
                scope: Scope::All,
            },
            strength: Per::Uniform(strength),
            mask: None,
            groups: None,
        }
    }

    pub fn link(distance: Real, strength: Real, iterations: u32) -> Force {
        Force {
            kind: Kind::Link { distance: Per::Uniform(distance), iterations },
            strength: Per::Uniform(strength),
            mask: None,
            groups: None,
        }
    }

    pub fn center(x: Real, y: Real, strength: Real) -> Force {
        Force { kind: Kind::Center { x, y }, strength: Per::Uniform(strength), mask: None, groups: None }
    }

    pub fn x(target: Real, strength: Real) -> Force {
        Force { kind: Kind::X { target: Per::Uniform(target) }, strength: Per::Uniform(strength), mask: None, groups: None }
    }

    pub fn y(target: Real, strength: Real) -> Force {
        Force { kind: Kind::Y { target: Per::Uniform(target) }, strength: Per::Uniform(strength), mask: None, groups: None }
    }

    pub fn collide(radius: Real, strength: Real, iterations: u32) -> Force {
        Force {
            kind: Kind::Collide { radius: Per::Uniform(radius), iterations, alpha_scaled: false },
            strength: Per::Uniform(strength),
            mask: None,
            groups: None,
        }
    }

    pub fn radial(radius: Real, x: Real, y: Real, strength: Real) -> Force {
        Force {
            kind: Kind::Radial { radius: Per::Uniform(radius), x, y },
            strength: Per::Uniform(strength),
            mask: None,
            groups: None,
        }
    }

    fn new(kind: Kind, strength: Real) -> Force {
        Force { kind, strength: Per::Uniform(strength), mask: None, groups: None }
    }

    pub fn towards(strength: Real) -> Force {
        Force::new(Kind::Towards { targets: Vec::new() }, strength)
    }

    pub fn rect_collide(half_width: Real, half_height: Real, strength: Real, iterations: u32) -> Force {
        Force::new(
            Kind::RectCollide {
                half_width: Per::Uniform(half_width),
                half_height: Per::Uniform(half_height),
                iterations,
                pairs: Scope::All,
                through: false,
            },
            strength,
        )
    }

    pub fn contain(strength: Real, hard: bool) -> Force {
        Force::new(
            Kind::Contain {
                regions: Vec::new(),
                assignment: None,
                radius: Per::Uniform(0.0),
                half_width: Per::Uniform(0.0),
                half_height: Per::Uniform(0.0),
                hard,
            },
            strength,
        )
    }

    pub fn cohesion(strength: Real) -> Force {
        Force::new(Kind::Cohesion, strength)
    }

    /// The axis is normalised; a zero or non-finite axis becomes (0, 1).
    pub fn directional(gap: Real, axis_x: Real, axis_y: Real, strength: Real, iterations: u32) -> Force {
        let len = (axis_x * axis_x + axis_y * axis_y).sqrt();
        let (axis_x, axis_y) = if len > 0.0 && len.is_finite() { (axis_x / len, axis_y / len) } else { (0.0, 1.0) };
        Force::new(Kind::Directional { gap: Per::Uniform(gap), axis_x, axis_y, iterations, hard: false }, strength)
    }

    /// Does this force take a strength per node (and therefore a mask)?
    pub fn strength_is_per_node(&self) -> bool {
        matches!(
            self.kind,
            Kind::ManyBody { .. }
                | Kind::X { .. }
                | Kind::Y { .. }
                | Kind::Radial { .. }
                | Kind::Towards { .. }
                | Kind::Cohesion
        )
    }

    /// Does this force take its strength as a single number only?
    pub fn strength_is_uniform(&self) -> bool {
        matches!(
            self.kind,
            Kind::Center { .. } | Kind::Collide { .. } | Kind::RectCollide { .. } | Kind::Contain { .. }
        )
    }

    /// The group node `i` is in, or `NONE_ID`.
    #[inline]
    fn group_of(&self, i: usize) -> u32 {
        match &self.groups {
            Some(groups) => groups.get(i).copied().unwrap_or(NONE_ID),
            None => NONE_ID,
        }
    }

    /// The strength node `i` has under this force: zero if masked out or unset.
    #[inline]
    fn node_strength(&self, i: usize) -> Real {
        if let Some(mask) = &self.mask {
            match mask.get(i) {
                Some(0) | None => return 0.0,
                Some(_) => {}
            }
        }
        let s = self.strength.at(i);
        if s.is_nan() {
            0.0
        } else {
            s
        }
    }

    /// The single strength of a force that has only one (center, collide).
    fn uniform_strength(&self) -> Real {
        match &self.strength {
            Per::Uniform(v) => *v,
            Per::Each(_) => 0.0,
        }
    }

    pub fn apply(&self, st: &mut State, alpha: Real) {
        match &self.kind {
            Kind::ManyBody { theta2, distance_min2, distance_max2, scope } => {
                self.apply_many_body(st, *theta2, *distance_min2, *distance_max2, *scope, alpha)
            }
            Kind::Link { distance, iterations } => self.apply_link(st, distance, *iterations, alpha),
            Kind::Center { x, y } => self.apply_center(st, *x, *y),
            Kind::X { target } => self.apply_position(st, target, 0, alpha),
            Kind::Y { target } => self.apply_position(st, target, 1, alpha),
            Kind::Collide { radius, iterations, alpha_scaled } => {
                self.apply_collide(st, radius, *iterations, if *alpha_scaled { Some(alpha) } else { None })
            }
            Kind::Radial { radius, x, y } => self.apply_radial(st, radius, *x, *y, alpha),
            Kind::Towards { targets } => self.apply_towards(st, targets, alpha),
            Kind::RectCollide { half_width, half_height, iterations, pairs, through } => {
                self.apply_rect_collide(st, half_width, half_height, *iterations, *pairs, *through)
            }
            Kind::Contain { regions, assignment, radius, half_width, half_height, .. } => {
                self.apply_contain(st, regions, assignment, radius, half_width, half_height)
            }
            Kind::Cohesion => self.apply_cohesion(st, alpha),
            Kind::Directional { gap, axis_x, axis_y, iterations, .. } => {
                self.apply_directional(st, gap, *axis_x, *axis_y, *iterations, alpha)
            }
        }
    }

    /// The part of a force that runs after integration: a hard containment
    /// and a hard directional constraint have one. Returns true if it may
    /// have moved a node.
    ///
    /// HARD CONTAINMENT. After positions have been integrated, every node in
    /// a region is moved to `region.project_body(radius, half_width,
    /// half_height, position)`, and its velocity is changed by the same
    /// amount, so that the velocity is still the displacement the node
    /// actually made this tick. (The caller then puts pinned coordinates
    /// back: a pin beats a region.)
    ///
    /// HARD DIRECTIONAL CONSTRAINT. The gap asked for is the gap kept. A node
    /// is HELD along the axis u if it is pinned on a coordinate the axis has
    /// a part in (x when u.x is not 0, y when u.y is not 0). One pass takes
    /// each link in order, on the positions as the links before it left
    /// them, both ends finite:
    ///   d = (p_target - p_source) . u
    ///   if d < gap_link:
    ///     short = gap_link - d
    ///     if the target is not held:      p_target += u * short, v_target += u * short
    ///     else if the source is not held: p_source -= u * short, v_source -= u * short
    ///     else:                           nothing
    /// So the target gives way, and the source only where the target cannot.
    /// Passes repeat until one moves nothing, `DIRECTIONAL_PASSES` at most (a
    /// chain of links listed child-first needs one pass per link; a cycle
    /// can never be satisfied and stops at the limit). A link whose gap or
    /// strength is NaN, or with an end out of range, is left alone, as the
    /// soft force leaves it.
    pub fn constrain(&self, st: &mut State) -> bool {
        match &self.kind {
            Kind::Contain { regions, assignment, radius, half_width, half_height, hard: true } => {
                for i in 0..st.n {
                    let Some(region) = region_of(regions, assignment, i) else { continue };
                    let x = st.positions[i * 2];
                    let y = st.positions[i * 2 + 1];
                    if !(x.is_finite() && y.is_finite()) {
                        continue;
                    }
                    let (cx, cy) = region.project_body(
                        node_radius(radius, i),
                        half_extent(half_width, i),
                        half_extent(half_height, i),
                        x,
                        y,
                    );
                    st.velocities[i * 2] += cx - x;
                    st.velocities[i * 2 + 1] += cy - y;
                    st.positions[i * 2] = cx;
                    st.positions[i * 2 + 1] = cy;
                }
                true
            }
            Kind::Directional { gap, axis_x, axis_y, hard: true, .. } => {
                let (axis_x, axis_y) = (*axis_x, *axis_y);
                let m = st.links.len() / 2;
                let State { positions, velocities, fixed, links, link_bias, .. } = st;
                let held = |fixed: &[Real], i: usize| {
                    (axis_x != 0.0 && !fixed[i * 2].is_nan()) || (axis_y != 0.0 && !fixed[i * 2 + 1].is_nan())
                };
                for _ in 0..DIRECTIONAL_PASSES {
                    let mut moved = false;
                    for i in 0..m {
                        let g = gap.at(i);
                        if link_bias[i].is_nan() || g.is_nan() || self.strength.at(i).is_nan() {
                            continue;
                        }
                        let s = links[i * 2] as usize;
                        let t = links[i * 2 + 1] as usize;
                        let (tx, ty, sx, sy) = (positions[t * 2], positions[t * 2 + 1], positions[s * 2], positions[s * 2 + 1]);
                        if !(tx.is_finite() && ty.is_finite() && sx.is_finite() && sy.is_finite()) {
                            continue;
                        }
                        let x = tx - sx;
                        let y = ty - sy;
                        let d = x * axis_x + y * axis_y;
                        if !(d < g) {
                            continue;
                        }
                        let short = g - d;
                        let step_x = axis_x * short;
                        let step_y = axis_y * short;
                        if !held(fixed, t) {
                            velocities[t * 2] += step_x;
                            velocities[t * 2 + 1] += step_y;
                            positions[t * 2] += step_x;
                            positions[t * 2 + 1] += step_y;
                            moved = true;
                        } else if !held(fixed, s) {
                            velocities[s * 2] -= step_x;
                            velocities[s * 2 + 1] -= step_y;
                            positions[s * 2] -= step_x;
                            positions[s * 2 + 1] -= step_y;
                            moved = true;
                        }
                    }
                    if !moved {
                        break;
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// d3 `manyBody.js`: Barnes-Hut over a quadtree of the current positions.
    ///
    /// SCOPED MANY-BODY (not in d3). With per-node group ids:
    /// - `Within`: for each group in ascending id order, d3's many-body is
    ///   applied to that group's nodes alone (its own quadtree, its nodes in
    ///   index order). Nodes in no group neither exert nor feel anything.
    /// - `Between`: d3's many-body over all nodes, followed by `Within` with
    ///   every strength negated. With theta = 0 that is the sum over pairs in
    ///   different groups; with theta > 0 it is the difference of two
    ///   Barnes-Hut approximations. A node in no group interacts with all.
    fn apply_many_body(
        &self,
        st: &mut State,
        theta2: Real,
        distance_min2: Real,
        distance_max2: Real,
        scope: Scope,
        alpha: Real,
    ) {
        let n = st.n;
        let State { positions, velocities, tree, stack, scratch, scratch_members, rng, .. } = st;
        let positions: &[Real] = positions;
        let strengths = scratch;

        if scope != Scope::Within {
            strengths.clear();
            strengths.extend((0..n).map(|i| self.node_strength(i)));
            many_body_pass(
                positions, velocities, tree, stack, rng, n, |k| k, strengths, theta2, distance_min2, distance_max2,
                alpha,
            );
        }
        if scope == Scope::All {
            return;
        }

        // The grouped nodes, by group and then by index.
        let sign: Real = if scope == Scope::Between { -1.0 } else { 1.0 };
        let members = scratch_members;
        members.clear();
        members.extend((0..n as u32).filter(|&i| self.group_of(i as usize) != NONE_ID));
        members.sort_by_key(|&i| self.group_of(i as usize));
        let mut start = 0;
        while start < members.len() {
            let group = self.group_of(members[start] as usize);
            let mut end = start + 1;
            while end < members.len() && self.group_of(members[end] as usize) == group {
                end += 1;
            }
            let run = &members[start..end];
            strengths.clear();
            strengths.extend(run.iter().map(|&i| sign * self.node_strength(i as usize)));
            many_body_pass(
                positions,
                velocities,
                tree,
                stack,
                rng,
                run.len(),
                |k| run[k] as usize,
                strengths,
                theta2,
                distance_min2,
                distance_max2,
                alpha,
            );
            start = end;
        }
    }

    /// d3 `link.js`: springs along the links, split between the endpoints by
    /// bias and solved against position + velocity.
    fn apply_link(&self, st: &mut State, distance: &Per, iterations: u32, alpha: Real) {
        let m = st.links.len() / 2;
        let State { positions, velocities, links, link_bias, rng, .. } = st;
        let positions: &[Real] = positions;

        for _ in 0..iterations {
            for i in 0..m {
                let b = link_bias[i];
                let d = distance.at(i);
                let k = self.strength.at(i);
                if b.is_nan() || d.is_nan() || k.is_nan() {
                    continue;
                }
                let s = links[i * 2] as usize;
                let t = links[i * 2 + 1] as usize;

                let mut x = positions[t * 2] + velocities[t * 2] - positions[s * 2] - velocities[s * 2];
                if !(x != 0.0) || x.is_nan() {
                    x = rng.jiggle();
                }
                let mut y = positions[t * 2 + 1] + velocities[t * 2 + 1]
                    - positions[s * 2 + 1]
                    - velocities[s * 2 + 1];
                if !(y != 0.0) || y.is_nan() {
                    y = rng.jiggle();
                }
                let mut l = (x * x + y * y).sqrt();
                l = (l - d) / l * alpha * k;
                x *= l;
                y *= l;
                velocities[t * 2] -= x * b;
                velocities[t * 2 + 1] -= y * b;
                let b = 1.0 - b;
                velocities[s * 2] += x * b;
                velocities[s * 2 + 1] += y * b;
            }
        }
    }

    /// d3 `center.js`: translate every node so the mean position moves
    /// toward (x, y). Moves positions, not velocities.
    fn apply_center(&self, st: &mut State, x: Real, y: Real) {
        let n = st.n;
        if n == 0 {
            return;
        }
        let strength = self.uniform_strength() as f64;
        let mut sx = 0.0f64;
        let mut sy = 0.0f64;
        for i in 0..n {
            sx += st.positions[i * 2] as f64;
            sy += st.positions[i * 2 + 1] as f64;
        }
        let sx = ((sx / n as f64 - x as f64) * strength) as Real;
        let sy = ((sy / n as f64 - y as f64) * strength) as Real;
        for i in 0..n {
            st.positions[i * 2] -= sx;
            st.positions[i * 2 + 1] -= sy;
        }
    }

    /// d3 `x.js` / `y.js` (`axis` 0 or 1): pull each node toward its target
    /// coordinate. A node whose target is NaN is left alone.
    fn apply_position(&self, st: &mut State, target: &Per, axis: usize, alpha: Real) {
        for i in 0..st.n {
            let z = target.at(i);
            if z.is_nan() {
                continue;
            }
            let k = self.node_strength(i);
            let at = i * 2 + axis;
            st.velocities[at] += (z - st.positions[at]) * k * alpha;
        }
    }

    /// d3 `collide.js`: push overlapping circles apart, heavier (larger)
    /// circles moving less. Works on position + velocity and rebuilds the
    /// quadtree for every iteration.
    ///
    /// YIELDING COLLISION (not in d3). Two options, both off unless asked
    /// for, and with both off this is d3's collide statement for statement:
    /// - `scale` = Some(alpha): the strength used is strength * alpha, so the
    ///   collision weakens as the simulation cools and gives way to forces
    ///   that hold a node at a place;
    /// - a node filter (`self.mask`): a pair is tested only if both of its
    ///   nodes are in the filter. A node left out neither pushes nor is
    ///   pushed. (It is still in the quadtree, where it can only widen the
    ///   search, never change a result.)
    fn apply_collide(&self, st: &mut State, radius: &Per, iterations: u32, scale: Option<Real>) {
        let n = st.n;
        let strength = match scale {
            Some(alpha) => self.uniform_strength() * alpha,
            None => self.uniform_strength(),
        };
        let mask = self.mask.as_deref();
        let included = |i: usize| match mask {
            Some(mask) => !matches!(mask.get(i), Some(0) | None),
            None => true,
        };
        let State { positions, velocities, tree, stack, scratch, rng, .. } = st;
        let positions: &[Real] = positions;

        let radii = scratch;
        radii.clear();
        radii.extend((0..n).map(|i| {
            let r = radius.at(i);
            if r.is_nan() {
                0.0
            } else {
                r
            }
        }));
        let radii: &[Real] = radii;

        for _ in 0..iterations {
            {
                let velocities: &[Real] = velocities;
                tree.build(n, |i| {
                    (positions[i * 2] + velocities[i * 2], positions[i * 2 + 1] + velocities[i * 2 + 1])
                });
            }
            prepare(tree, radii);
            let tree: &Quadtree = tree;

            for i in 0..n {
                let ri = radii[i];
                let ri2 = ri * ri;
                let xi = positions[i * 2] + velocities[i * 2];
                let yi = positions[i * 2 + 1] + velocities[i * 2 + 1];
                if !(xi.is_finite() && yi.is_finite()) || !included(i) {
                    continue;
                }

                tree.visit(stack, |q, x0, y0, x1, y1| {
                    if is_leaf(q) {
                        // Only the head of a chain of coincident points is
                        // tested, as in d3.
                        let j = leaf_point(q);
                        if j > i && included(j) {
                            let mut rj = tree.lv[j];
                            let r = ri + rj;
                            let mut x = xi - positions[j * 2] - velocities[j * 2];
                            let mut y = yi - positions[j * 2 + 1] - velocities[j * 2 + 1];
                            let mut l = x * x + y * y;
                            if l < r * r {
                                if x == 0.0 {
                                    x = rng.jiggle();
                                    l += x * x;
                                }
                                if y == 0.0 {
                                    y = rng.jiggle();
                                    l += y * y;
                                }
                                l = l.sqrt();
                                l = (r - l) / l * strength;
                                x *= l;
                                y *= l;
                                rj *= rj;
                                let share = rj / (ri2 + rj);
                                velocities[i * 2] += x * share;
                                velocities[i * 2 + 1] += y * share;
                                let share = 1.0 - share;
                                velocities[j * 2] -= x * share;
                                velocities[j * 2 + 1] -= y * share;
                            }
                        }
                        return true;
                    }
                    let r = ri + tree.iv[q as usize];
                    x0 > xi + r || x1 < xi - r || y0 > yi + r || y1 < yi - r
                });
            }
        }
    }

    /// d3 `radial.js`: pull each node toward a circle of its target radius
    /// around (x, y). A node whose radius is NaN is left alone.
    fn apply_radial(&self, st: &mut State, radius: &Per, x: Real, y: Real, alpha: Real) {
        for i in 0..st.n {
            let target = radius.at(i);
            if target.is_nan() {
                continue;
            }
            let k = self.node_strength(i);
            let mut dx = st.positions[i * 2] - x;
            if !(dx != 0.0) || dx.is_nan() {
                dx = 1e-6;
            }
            let mut dy = st.positions[i * 2 + 1] - y;
            if !(dy != 0.0) || dy.is_nan() {
                dy = 1e-6;
            }
            let r = (dx * dx + dy * dy).sqrt();
            let k = (target - r) * k * alpha / r;
            st.velocities[i * 2] += dx * k;
            st.velocities[i * 2 + 1] += dy * k;
        }
    }

    /// TOWARDS (not in d3): a spring from each node to its own target point.
    ///
    /// For every node i with a target (tx, ty), both finite:
    ///   vx += (tx - x) * strength_i * alpha
    ///   vy += (ty - y) * strength_i * alpha
    /// A node with no target, or a NaN in it, is left alone. This is d3's
    /// forceX and forceY with per-node targets, as one force whose targets
    /// are one array that can be overwritten between ticks.
    fn apply_towards(&self, st: &mut State, targets: &[Real], alpha: Real) {
        let count = st.n.min(targets.len() / 2);
        for i in 0..count {
            let tx = targets[i * 2];
            let ty = targets[i * 2 + 1];
            if tx.is_nan() || ty.is_nan() {
                continue;
            }
            let k = self.node_strength(i);
            st.velocities[i * 2] += (tx - st.positions[i * 2]) * k * alpha;
            st.velocities[i * 2 + 1] += (ty - st.positions[i * 2 + 1]) * k * alpha;
        }
    }

    /// RECTANGLE COLLISION (not in d3): push overlapping axis-aligned
    /// rectangles apart along the axis on which they overlap least.
    ///
    /// Node i is a rectangle of half-width w_i and half-height h_i centred
    /// on its anticipated position a_i = position + velocity. A node is HELD
    /// on an axis if it cannot be moved along it: it is pinned there (its
    /// `fixed` coordinate on that axis is not NaN). One iteration:
    ///
    /// 1. Take a_i for every node, all at once, before anything moves.
    /// 2. For every node i that takes part, and every other node j that
    ///    takes part and may collide with it (see "Who collides"), both a
    ///    finite, taking j in ascending order:
    ///      dx = a_i.x - a_j.x          dy = a_i.y - a_j.y
    ///      ox = (w_i + w_j) - |dx|     oy = (h_i + h_j) - |dy|
    ///    The pair overlaps when ox > 0 and oy > 0. If it does, the push is
    ///    along x if ox <= oy and along y otherwise. If i is held on that
    ///    axis, nothing more happens for this pair. Otherwise
    ///      share = 1                      if j is held on that axis
    ///            = A_j / (A_i + A_j)      otherwise, with A = w * h
    ///                                     (1/2 if A_i + A_j is not above 0)
    ///      push  = s * overlap * strength * share
    ///      v_i[axis] += push
    ///    where s is the sign of dx (or dy): +1 if positive, -1 if negative,
    ///    and if exactly zero, -1 when i < j and +1 when i > j.
    ///
    /// So a held rectangle is never moved, and what overlaps it takes the
    /// whole correction; two free rectangles share it, the smaller moving
    /// more, the two parts sum to the overlap, and with strength 1 one
    /// isolated pair is exactly separated.
    ///
    /// THE SQUEEZE. A wide rectangle nearly always overlaps another least
    /// vertically, so it is only ever pushed up or down; caught between two
    /// things that do not give way (two pinned dots, a dot and a packed row
    /// of other labels) the pushes cancel and it sits on both for ever. With
    /// `through`, after the last iteration of one application of the force,
    /// for each node i and each axis it is not held on: let `up` be the sum
    /// of the positive pushes step 2 gave it on that axis over all the
    /// iterations, and `down` the sum of the negative ones, as a magnitude.
    /// If both are above 0 the node is squeezed, and it goes with one side
    /// and ignores the other:
    ///   v_i[axis] += down     if it goes up   (the downward pushes are undone)
    ///   v_i[axis] -= up       if it goes down
    /// It goes the way it was already moving, if it was moving: with w its
    /// velocity on that axis as it was when the force was applied, if
    /// |w| > 0.001 * (up + down) it goes up when w > 0 and down when w < 0.
    /// Otherwise (a velocity that small is what is left of a movement long
    /// over, or the last of a pull dying away) it goes with the stronger
    /// side: up if up > down, down if up < down, and if they are equal, up
    /// when i is odd and down when it is even.
    ///
    /// Going the way it is moving is what makes this resolve: the rectangle
    /// keeps on through whatever was squeezing it, a dot or another
    /// rectangle, until it is clear, where a rule that only compared the two
    /// sides would turn round as soon as it went deeper into the weaker one.
    /// It suits rectangles that nothing else is pressing together. Under a
    /// force that keeps pressing, a squeeze is permanent and the rectangle
    /// will be walked through its neighbour, so let such a pull die away
    /// before expecting rest. Without `through`, step 2 is all there is.
    ///
    /// Who collides. With a node filter (`self.mask`), a node left out
    /// collides with nothing and nothing collides with it. With groups
    /// (`self.groups`), `pairs` restricts the pairs: only two nodes of the
    /// same group (`Within`), or only two nodes not of the same group
    /// (`Between`; a node in no group is in a different group from
    /// everyone).
    ///
    /// Every correction in an iteration is computed from the same snapshot
    /// (Jacobi, not d3-collide's ordered sweep), so the result does not
    /// depend on the order nodes are visited in and no node is favoured by
    /// its index; the price is that crowded rectangles need iterations. Not
    /// scaled by alpha, as d3's collide is not.
    ///
    /// The quadtree only finds the candidates: for each i the rectangles that
    /// can overlap it are collected, sorted by index and applied in that
    /// order, which makes the result identical, bit for bit, to the loop over
    /// all pairs above.
    fn apply_rect_collide(
        &self,
        st: &mut State,
        half_width: &Per,
        half_height: &Per,
        iterations: u32,
        pairs: Scope,
        through: bool,
    ) {
        let n = st.n;
        let strength = self.uniform_strength();
        let mask = self.mask.as_deref();
        let included = |i: usize| match mask {
            Some(mask) => !matches!(mask.get(i), Some(0) | None),
            None => true,
        };
        let may = |i: usize, j: usize| match pairs {
            Scope::All => true,
            Scope::Within => self.group_of(i) != NONE_ID && self.group_of(i) == self.group_of(j),
            Scope::Between => !(self.group_of(i) != NONE_ID && self.group_of(i) == self.group_of(j)),
        };
        let State { positions, velocities, fixed, tree, stack, scratch, scratch_xy, scratch_index, .. } = st;
        let positions: &[Real] = positions;
        let fixed: &[Real] = fixed;
        let anticipated = scratch_xy;
        let candidates = scratch_index;

        // Per node, for the squeeze: the velocity when the force was applied
        // (x, y), then up and down on x, then up and down on y.
        let pushed = scratch;
        pushed.clear();
        if through {
            for i in 0..n {
                pushed.extend([velocities[i * 2], velocities[i * 2 + 1], 0.0, 0.0, 0.0, 0.0]);
            }
        }

        for _ in 0..iterations {
            anticipated.clear();
            anticipated.extend((0..n * 2).map(|k| positions[k] + velocities[k]));
            let a: &[Real] = anticipated;

            tree.build(n, |i| (a[i * 2], a[i * 2 + 1]));
            largest_extents(tree, half_width, half_height);
            let tree: &Quadtree = tree;

            for i in 0..n {
                if !included(i) {
                    continue;
                }
                let xi = a[i * 2];
                let yi = a[i * 2 + 1];
                if !(xi.is_finite() && yi.is_finite()) {
                    continue;
                }
                let wi = half_extent(half_width, i);
                let hi = half_extent(half_height, i);
                let held_x = !fixed[i * 2].is_nan();
                let held_y = !fixed[i * 2 + 1].is_nan();

                candidates.clear();
                tree.visit(stack, |q, x0, y0, x1, y1| {
                    if is_leaf(q) {
                        let mut p = leaf_point(q) as u32;
                        while p != NONE {
                            if p as usize != i && included(p as usize) && may(i, p as usize) {
                                candidates.push(p);
                            }
                            p = tree.next[p as usize];
                        }
                        return true;
                    }
                    // Skip a cell only when it is provably out of reach: these
                    // comparisons are strict and each subtraction is rounded
                    // once, so no overlapping pair is ever skipped.
                    let reach_x = wi + tree.ix[q as usize];
                    let reach_y = hi + tree.iy[q as usize];
                    x0 - xi > reach_x || xi - x1 > reach_x || y0 - yi > reach_y || yi - y1 > reach_y
                });
                candidates.sort_unstable();

                let area_i = wi * hi;
                for &j in candidates.iter() {
                    let j = j as usize;
                    let wj = half_extent(half_width, j);
                    let hj = half_extent(half_height, j);
                    let dx = xi - a[j * 2];
                    let dy = yi - a[j * 2 + 1];
                    let ox = (wi + wj) - dx.abs();
                    let oy = (hi + hj) - dy.abs();
                    if !(ox > 0.0 && oy > 0.0) {
                        continue;
                    }
                    let area_j = wj * hj;
                    let by_area = if area_i + area_j > 0.0 { area_j / (area_i + area_j) } else { 0.5 };
                    let tie: Real = if i < j { -1.0 } else { 1.0 };
                    // 0 for x, 1 for y.
                    let (axis, offset, overlap, held) = if ox <= oy { (0, dx, ox, held_x) } else { (1, dy, oy, held_y) };
                    if held {
                        continue;
                    }
                    let share = if !fixed[j * 2 + axis].is_nan() { 1.0 } else { by_area };
                    let s = if offset > 0.0 { 1.0 } else if offset < 0.0 { -1.0 } else { tie };
                    let push = s * overlap * strength * share;
                    velocities[i * 2 + axis] += push;
                    if through {
                        if push > 0.0 {
                            pushed[i * 6 + 2 + axis * 2] += push;
                        } else if push < 0.0 {
                            pushed[i * 6 + 3 + axis * 2] -= push;
                        }
                    }
                }
            }
        }

        if through {
            for i in 0..n {
                if !included(i) {
                    continue;
                }
                let odd = i % 2 == 1;
                let tx = through_squeeze(odd, pushed[i * 6], pushed[i * 6 + 2], pushed[i * 6 + 3]);
                let ty = through_squeeze(odd, pushed[i * 6 + 1], pushed[i * 6 + 4], pushed[i * 6 + 5]);
                if tx != 0.0 || ty != 0.0 {
                    velocities[i * 2] += tx;
                    velocities[i * 2 + 1] += ty;
                }
            }
        }
    }

    /// CONTAINMENT (not in d3), the soft part: steer each node back into its
    /// region.
    ///
    /// For every node i that has a region, with a = position + velocity (its
    /// anticipated position) finite and c = region.project_body(radius_i,
    /// half_width_i, half_height_i, a):
    ///   v += (c - a) * strength
    /// With strength 1 the node is aimed exactly at the nearest allowed
    /// point; it arrives short of it by the velocity decay, and the next
    /// tick corrects again. Not scaled by alpha: it is a constraint, like
    /// collide. A node already inside is not touched. See `constrain` for
    /// the hard part and `Region::project` for the geometry.
    fn apply_contain(
        &self,
        st: &mut State,
        regions: &[Region],
        assignment: &Option<Vec<u32>>,
        radius: &Per,
        half_width: &Per,
        half_height: &Per,
    ) {
        let strength = self.uniform_strength();
        for i in 0..st.n {
            let Some(region) = region_of(regions, assignment, i) else { continue };
            let ax = st.positions[i * 2] + st.velocities[i * 2];
            let ay = st.positions[i * 2 + 1] + st.velocities[i * 2 + 1];
            if !(ax.is_finite() && ay.is_finite()) {
                continue;
            }
            let (cx, cy) = region.project_body(
                node_radius(radius, i),
                half_extent(half_width, i),
                half_extent(half_height, i),
                ax,
                ay,
            );
            st.velocities[i * 2] += (cx - ax) * strength;
            st.velocities[i * 2 + 1] += (cy - ay) * strength;
        }
    }

    /// GROUP COHESION (not in d3): pull each node toward the centroid of its
    /// group.
    ///
    /// The centroid of a group is the mean position of its members whose
    /// position is finite, summed in node index order in f64. For every node
    /// i in a group, with a finite position:
    ///   v += (centroid - position) * strength_i * alpha
    /// A node in no group is left alone and counts toward no centroid. A
    /// masked-out node still counts toward its group's centroid.
    fn apply_cohesion(&self, st: &mut State, alpha: Real) {
        let mut sums: std::collections::BTreeMap<u32, (f64, f64, u32)> = std::collections::BTreeMap::new();
        for i in 0..st.n {
            let group = self.group_of(i);
            let x = st.positions[i * 2];
            let y = st.positions[i * 2 + 1];
            if group == NONE_ID || !(x.is_finite() && y.is_finite()) {
                continue;
            }
            let sum = sums.entry(group).or_insert((0.0, 0.0, 0));
            sum.0 += x as f64;
            sum.1 += y as f64;
            sum.2 += 1;
        }
        for i in 0..st.n {
            let x = st.positions[i * 2];
            let y = st.positions[i * 2 + 1];
            if !(x.is_finite() && y.is_finite()) {
                continue;
            }
            let Some(&(sx, sy, count)) = sums.get(&self.group_of(i)) else { continue };
            let cx = (sx / count as f64) as Real;
            let cy = (sy / count as f64) as Real;
            let k = self.node_strength(i);
            st.velocities[i * 2] += (cx - x) * k * alpha;
            st.velocities[i * 2 + 1] += (cy - y) * k * alpha;
        }
    }

    /// DIRECTIONAL LINK CONSTRAINT (not in d3): keep each link's target at
    /// least a gap beyond its source along an axis (children below parents).
    ///
    /// With u the unit axis, and working on anticipated positions
    /// a = position + velocity, for each iteration, for each link in order
    /// (reading the velocities as the links before it left them):
    ///   d = (a_target - a_source) . u
    ///   if d < gap_link:
    ///     push = (gap_link - d) * strength_link * alpha
    ///     v_target += u * push * bias
    ///     v_source -= u * push * (1 - bias)
    /// where bias is d3's link bias: the source's share of the two ends'
    /// link counts, so the better connected end moves less. A link already
    /// satisfied is not touched (the force only pushes apart, never pulls).
    /// With strength 1 and alpha 1 one isolated link is exactly satisfied.
    fn apply_directional(&self, st: &mut State, gap: &Per, axis_x: Real, axis_y: Real, iterations: u32, alpha: Real) {
        let m = st.links.len() / 2;
        let State { positions, velocities, links, link_bias, .. } = st;
        let positions: &[Real] = positions;
        for _ in 0..iterations {
            for i in 0..m {
                let b = link_bias[i];
                let g = gap.at(i);
                let k = self.strength.at(i);
                if b.is_nan() || g.is_nan() || k.is_nan() {
                    continue;
                }
                let s = links[i * 2] as usize;
                let t = links[i * 2 + 1] as usize;
                let x = positions[t * 2] + velocities[t * 2] - positions[s * 2] - velocities[s * 2];
                let y = positions[t * 2 + 1] + velocities[t * 2 + 1] - positions[s * 2 + 1] - velocities[s * 2 + 1];
                let d = x * axis_x + y * axis_y;
                if !(d < g) {
                    continue;
                }
                let push = (g - d) * k * alpha;
                velocities[t * 2] += axis_x * push * b;
                velocities[t * 2 + 1] += axis_y * push * b;
                let b = 1.0 - b;
                velocities[s * 2] -= axis_x * push * b;
                velocities[s * 2 + 1] -= axis_y * push * b;
            }
        }
    }
}

/// d3's many-body over `m` nodes: local node `k` is simulation node
/// `global(k)`, and `strengths` is indexed locally.
#[allow(clippy::too_many_arguments)]
#[inline]
fn many_body_pass<G>(
    positions: &[Real],
    velocities: &mut [Real],
    tree: &mut Quadtree,
    stack: &mut Vec<Quad>,
    rng: &mut Lcg,
    m: usize,
    global: G,
    strengths: &[Real],
    theta2: Real,
    distance_min2: Real,
    distance_max2: Real,
    alpha: Real,
) where
    G: Fn(usize) -> usize + Copy,
{
    tree.build(m, |k| {
        let g = global(k);
        (positions[g * 2], positions[g * 2 + 1])
    });
    accumulate(tree, strengths);
    let tree: &Quadtree = tree;

    for i in 0..m {
        let g = global(i);
        let node_x = positions[g * 2];
        let node_y = positions[g * 2 + 1];
        if !(node_x.is_finite() && node_y.is_finite()) {
            continue;
        }
        let mut vx = velocities[g * 2];
        let mut vy = velocities[g * 2 + 1];

        tree.visit(stack, |q, x1, _, x2, _| {
            let leaf = is_leaf(q);
            let (value, qx, qy) = if leaf {
                let p = leaf_point(q);
                (tree.lv[p], tree.xs[p], tree.ys[p])
            } else {
                let k = q as usize;
                (tree.iv[k], tree.ix[k], tree.iy[k])
            };
            if !(value != 0.0) || value.is_nan() {
                return true;
            }

            let mut x = qx - node_x;
            let mut y = qy - node_y;
            let w = x2 - x1;
            let mut l = x * x + y * y;

            // Apply the Barnes-Hut approximation if possible.
            if w * w / theta2 < l {
                if l < distance_max2 {
                    if x == 0.0 {
                        x = rng.jiggle();
                        l += x * x;
                    }
                    if y == 0.0 {
                        y = rng.jiggle();
                        l += y * y;
                    }
                    if l < distance_min2 {
                        l = (distance_min2 * l).sqrt();
                    }
                    vx += x * value * alpha / l;
                    vy += y * value * alpha / l;
                }
                return true;
            }

            // Otherwise, process points directly.
            if !leaf || l >= distance_max2 {
                return false;
            }

            let head = leaf_point(q);
            if head != i || tree.next[head] != NONE {
                if x == 0.0 {
                    x = rng.jiggle();
                    l += x * x;
                }
                if y == 0.0 {
                    y = rng.jiggle();
                    l += y * y;
                }
                if l < distance_min2 {
                    l = (distance_min2 * l).sqrt();
                }
            }

            let mut p = head as u32;
            while p != NONE {
                if p as usize != i {
                    let w = strengths[p as usize] * alpha / l;
                    vx += x * w;
                    vy += y * w;
                }
                p = tree.next[p as usize];
            }
            true
        });

        velocities[g * 2] = vx;
        velocities[g * 2 + 1] = vy;
    }
}

/// The region node `i` is confined to, if any.
#[inline]
fn region_of<'a>(regions: &'a [Region], assignment: &Option<Vec<u32>>, i: usize) -> Option<&'a Region> {
    let index = match assignment {
        Some(a) => a.get(i).copied().unwrap_or(NONE_ID),
        None => 0,
    };
    regions.get(index as usize)
}

/// A node's radius under a containment force: NaN or unset is 0.
#[inline]
fn node_radius(radius: &Per, i: usize) -> Real {
    let r = radius.at(i);
    if r.is_nan() {
        0.0
    } else {
        r
    }
}

/// What a squeezed rectangle adds to its velocity on one axis: the pushes
/// of the side it ignores, undone. Nothing if it was not squeezed.
#[inline]
fn through_squeeze(odd: bool, start: Real, up: Real, down: Real) -> Real {
    if up > 0.0 && down > 0.0 {
        if start.abs() > 0.001 * (up + down) {
            if start > 0.0 {
                down
            } else {
                -up
            }
        } else if up > down {
            down
        } else if up < down {
            -up
        } else if odd {
            down
        } else {
            -up
        }
    } else {
        0.0
    }
}

/// A half-extent (rectangle collision, containment): NaN or unset is 0.
#[inline]
fn half_extent(extent: &Per, i: usize) -> Real {
    let h = extent.at(i);
    if h.is_nan() {
        0.0
    } else {
        h
    }
}

/// The charge of a leaf: the sum over its chain of coincident points.
#[inline]
fn chain_sum(tree: &Quadtree, head: usize, values: &[Real]) -> Real {
    let mut total = 0.0 as Real;
    let mut p = head as u32;
    while p != NONE {
        total += values[p as usize];
        p = tree.next[p as usize];
    }
    total
}

/// Many-body's `accumulate`, run as d3 runs it under `visitAfter`: every
/// leaf gets the summed strength of its chain, every internal node the total
/// strength below it and the |strength|-weighted centre of its children.
fn accumulate(tree: &mut Quadtree, strengths: &[Real]) {
    tree.order();
    let n_internal = tree.children.len();
    tree.ix.clear();
    tree.ix.resize(n_internal, 0.0);
    tree.iy.clear();
    tree.iy.resize(n_internal, 0.0);
    tree.iv.clear();
    tree.iv.resize(n_internal, 0.0);

    if tree.root != NONE && is_leaf(tree.root) {
        let p = leaf_point(tree.root);
        tree.lv[p] = chain_sum(tree, p, strengths);
        return;
    }

    for at in (0..tree.pre_order.len()).rev() {
        let k = tree.pre_order[at] as usize;
        let mut strength = 0.0 as Real;
        let mut weight = 0.0 as Real;
        let mut x = 0.0 as Real;
        let mut y = 0.0 as Real;
        for c in tree.children[k] {
            if c == NONE {
                continue;
            }
            let (value, cx, cy) = if is_leaf(c) {
                let p = leaf_point(c);
                let v = chain_sum(tree, p, strengths);
                tree.lv[p] = v;
                (v, tree.xs[p], tree.ys[p])
            } else {
                let j = c as usize;
                (tree.iv[j], tree.ix[j], tree.iy[j])
            };
            let w = value.abs();
            if w > 0.0 {
                strength += value;
                weight += w;
                x += w * cx;
                y += w * cy;
            }
        }
        tree.ix[k] = x / weight;
        tree.iy[k] = y / weight;
        tree.iv[k] = strength;
    }
}

/// Collide's `prepare`: a leaf carries the radius of the point at its head,
/// an internal node the largest radius below it.
fn prepare(tree: &mut Quadtree, radii: &[Real]) {
    tree.order();
    let n_internal = tree.children.len();
    tree.iv.clear();
    tree.iv.resize(n_internal, 0.0);

    if tree.root != NONE && is_leaf(tree.root) {
        let p = leaf_point(tree.root);
        tree.lv[p] = radii[p];
        return;
    }

    for at in (0..tree.pre_order.len()).rev() {
        let k = tree.pre_order[at] as usize;
        let mut r = 0.0 as Real;
        for c in tree.children[k] {
            if c == NONE {
                continue;
            }
            let cr = if is_leaf(c) {
                let p = leaf_point(c);
                tree.lv[p] = radii[p];
                radii[p]
            } else {
                tree.iv[c as usize]
            };
            if cr > r {
                r = cr;
            }
        }
        tree.iv[k] = r;
    }
}

/// Rectangle collision's tree payload: for every internal node, the largest
/// half-width (`ix`) and half-height (`iy`) of any rectangle below it.
fn largest_extents(tree: &mut Quadtree, half_width: &Per, half_height: &Per) {
    tree.order();
    let n_internal = tree.children.len();
    tree.ix.clear();
    tree.ix.resize(n_internal, 0.0);
    tree.iy.clear();
    tree.iy.resize(n_internal, 0.0);

    for at in (0..tree.pre_order.len()).rev() {
        let k = tree.pre_order[at] as usize;
        let mut w = 0.0 as Real;
        let mut h = 0.0 as Real;
        for c in tree.children[k] {
            if c == NONE {
                continue;
            }
            if is_leaf(c) {
                let mut p = leaf_point(c) as u32;
                while p != NONE {
                    w = w.max(half_extent(half_width, p as usize));
                    h = h.max(half_extent(half_height, p as usize));
                    p = tree.next[p as usize];
                }
            } else {
                w = w.max(tree.ix[c as usize]);
                h = h.max(tree.iy[c as usize]);
            }
        }
        tree.ix[k] = w;
        tree.iy[k] = h;
    }
}
